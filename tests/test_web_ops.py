"""Offline single-page fetch/security tests, including the real HTTP adapter."""

from contextlib import closing
from io import BytesIO
import socket
from unittest.mock import Mock

import pytest
import requests
from urllib3.response import HTTPResponse
from urllib3.exceptions import ConnectTimeoutError, ReadTimeoutError

from core import error_codes
from core.models import ToolOutputEnvelope
from tools import web_ops as web


def dns_records(*addresses):
    return [
        (socket.AF_INET6 if ":" in address else socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443))
        for address in addresses
    ]


def html_response(body=b"<html><title>Example</title><body><p>Public evidence.</p></body></html>",
                  *, status=200, content_type="text/html; charset=utf-8", headers=None):
    response = requests.Response()
    response.status_code = status
    response.headers.update({"Content-Type": content_type, **(headers or {})})
    response.raw = BytesIO(body)
    return response


@pytest.fixture(autouse=True)
def public_dns(monkeypatch):
    resolver = Mock(return_value=dns_records("93.184.216.34"))
    monkeypatch.setattr(web.socket, "getaddrinfo", resolver)
    # Any unexpected transport path must fail instead of accessing live internet.
    monkeypatch.setattr(web, "create_connection", Mock(side_effect=AssertionError("Unexpected socket")))
    return resolver


@pytest.fixture
def transport(monkeypatch):
    send = Mock(return_value=html_response())
    monkeypatch.setattr(web._PublicHTTPAdapter, "send", send)
    return send


@pytest.mark.parametrize("url", ["https://example.com/", "http://example.com/", "https://[2606:4700:4700::1111]/"])
def test_public_html_fetch_returns_evidence_and_safe_transport(url, transport):
    result = web.fetch_web(url)
    assert result.success and isinstance(result, ToolOutputEnvelope)
    assert result.data["url"] == result.data["final_url"] == url
    assert result.data["status_code"] == 200
    assert result.data["content_type"] == "text/html"
    assert result.data["title"] == "Example"
    assert result.data["text"] == "Public evidence."
    assert result.data["total_chars"] == len("Public evidence.")
    assert result.data["truncated"] is False
    from datetime import datetime, timedelta
    assert datetime.fromisoformat(result.data["retrieved_at"]).utcoffset() == timedelta(0)
    prepared = transport.call_args.args[0]
    assert prepared.headers["User-Agent"].startswith("CortexNode/")
    assert prepared.headers["Accept-Encoding"] == "identity"
    assert transport.call_args.kwargs == {
        "stream": True, "timeout": (5, 15), "verify": True, "proxies": {},
    }
    assert not {"Authorization", "Cookie", "Proxy-Authorization"} & prepared.headers.keys()


@pytest.mark.parametrize("url", [
    "file:///etc/passwd", "ftp://example.com/a", "data:text/html,hello", "custom://example.com",
    "example.com", "https:///page", "http://", "https://user:secret@example.com/",
    "https://user@example.com/", "https://@example.com/", "https://[::1", "https://example.com:bad/",
    "https://example.com:0/", " https://example.com/", "https://example.com/\nhello",
    "https://example.com\\@127.0.0.1/", "https://[fe80::1%25eth0]/",
])
def test_invalid_urls_are_rejected_before_transport(url, transport):
    result = web.fetch_web(url)
    assert not result.success and result.error_code == error_codes.WEB_FETCH_REJECTED
    assert result.data["url"] == url
    transport.assert_not_called()
    assert "secret" not in result.message and "Traceback" not in result.display


@pytest.mark.parametrize("host", [
    "localhost", "LOCALHOST.", "a.localhost", "127.0.0.1", "127.1", "2130706433", "0x7f000001",
    "192.168.1.1", "10.0.0.1", "172.16.0.1", "172.31.255.255", "169.254.169.254", "0.0.0.0",
    "224.0.0.1", "240.0.0.1", "100.64.0.1", "192.0.2.1",
    "[::1]", "[::]", "[fe80::1]", "[fc00::1]", "[ff02::1]", "[2001:db8::1]",
    "[::ffff:127.0.0.1]", "[2002:7f00:1::]", "[64:ff9b::7f00:1]",
])
def test_non_public_destinations_are_rejected(host, public_dns, transport):
    # OS resolvers accept these alternative IPv4 spellings.
    if host in {"127.1", "2130706433", "0x7f000001"}:
        public_dns.return_value = dns_records("127.0.0.1")
    result = web.fetch_web(f"http://{host}/")
    assert not result.success and result.error_code == error_codes.WEB_FETCH_REJECTED
    transport.assert_not_called()


@pytest.mark.parametrize("addresses", [("10.0.0.1",), ("93.184.216.34", "::1"), ()])
def test_dns_must_return_only_public_addresses(addresses, public_dns, transport):
    public_dns.return_value = dns_records(*addresses)
    result = web.fetch_web("https://public-looking.example/")
    assert result.error_code == error_codes.WEB_FETCH_REJECTED
    transport.assert_not_called()


def test_dns_failure_is_structured_without_transport_internals(public_dns, transport):
    public_dns.side_effect = socket.gaierror("sensitive DNS transport details")
    result = web.fetch_web("https://example.com/")
    assert result.error_code == error_codes.WEB_FETCH_FAILED
    assert "sensitive" not in result.to_tool_output()
    transport.assert_not_called()


@pytest.mark.parametrize("destination", [
    "http://127.0.0.1/", "http://localhost/", "http://[::1]/", "http://192.168.1.1/",
    "file:///etc/passwd", "https://user:secret@example.com/", "//10.1.1.1/",
])
def test_redirect_to_unsafe_destination_is_rejected_without_reading_body(destination, transport):
    redirect = html_response(status=302, headers={"Location": destination})
    redirect.raw.read = Mock(side_effect=AssertionError("Redirect body must not be read"))
    transport.return_value = redirect
    result = web.fetch_web("https://example.com/")
    assert result.error_code == error_codes.WEB_FETCH_REJECTED
    assert transport.call_count == 1 and redirect.raw.closed


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_public_relative_and_cross_host_redirects_are_validated(status, transport, public_dns):
    transport.side_effect = [
        html_response(status=status, headers={"Location": "/article"}),
        html_response(status=302, headers={"Location": "http://other.example/final"}),
        html_response(),
    ]
    result = web.fetch_web("https://example.com/start")
    assert result.success and result.data["final_url"] == "http://other.example/final"
    assert result.data["url"] == "https://example.com/start"
    assert [call.args[0].url for call in transport.call_args_list] == [
        "https://example.com/start", "https://example.com/article", "http://other.example/final",
    ]
    assert any(call.args[0] == "other.example" for call in public_dns.call_args_list)


def test_redirect_hostname_resolving_private_is_rejected(public_dns, transport):
    public_dns.side_effect = [dns_records("93.184.216.34"), dns_records("10.0.0.1")]
    transport.return_value = html_response(status=302, headers={"Location": "https://internal.example/"})
    assert web.fetch_web("https://example.com/").error_code == error_codes.WEB_FETCH_REJECTED
    assert transport.call_count == 1


@pytest.mark.parametrize("location", [None, "/loop"])
def test_missing_location_or_redirect_loop_is_bounded(location, transport):
    transport.side_effect = lambda *_args, **_kwargs: html_response(
        status=302, headers={"Location": location} if location else {},
    )
    result = web.fetch_web("https://example.com/")
    assert result.error_code == error_codes.WEB_FETCH_FAILED
    assert transport.call_count == (web.MAX_REDIRECTS + 1 if location else 1)


@pytest.mark.parametrize("content_type", ["application/pdf", "image/png", "text/plain", "", "application/octet-stream"])
def test_unsupported_content_type_is_structured_without_reading_body(content_type, transport):
    response = html_response(content_type=content_type)
    response.raw.read = Mock(side_effect=AssertionError("Binary body must not be read"))
    transport.side_effect = [html_response(status=302, headers={"Location": "/document"}), response]
    result = web.fetch_web("https://example.com/")
    assert result.error_code == error_codes.WEB_FETCH_UNSUPPORTED_CONTENT_TYPE
    assert result.data == {
        "url": "https://example.com/", "final_url": "https://example.com/document",
        "status_code": 200, "content_type": content_type,
    }
    assert response.raw.closed


@pytest.mark.parametrize("error", [requests.ConnectTimeout("secret"), requests.ReadTimeout("secret")])
def test_transport_timeout_is_structured(error, transport):
    transport.side_effect = error
    result = web.fetch_web("https://example.com/")
    assert result.error_code == error_codes.WEB_FETCH_TIMEOUT
    assert "secret" not in result.to_tool_output()


def test_stream_timeout_wrapped_by_requests_is_classified(transport):
    response = html_response()
    response.iter_content = Mock(side_effect=requests.ConnectionError(ReadTimeoutError(None, "/", "secret")))
    transport.return_value = response
    result = web.fetch_web("https://example.com/")
    assert result.error_code == error_codes.WEB_FETCH_TIMEOUT and response.raw.closed
    assert "secret" not in result.to_tool_output()


@pytest.mark.parametrize("error", [requests.ConnectionError("secret"), requests.exceptions.SSLError("secret")])
def test_connection_errors_are_structured(error, transport):
    transport.side_effect = error
    assert web.fetch_web("https://example.com/").error_code == error_codes.WEB_FETCH_FAILED


@pytest.mark.parametrize("status", [304, 400, 404, 429, 500])
def test_non_success_status_is_failure_without_reading_body(status, transport):
    response = html_response(status=status)
    response.raw.read = Mock(side_effect=AssertionError("Error body must not be read"))
    transport.return_value = response
    result = web.fetch_web("https://example.com/")
    assert result.error_code == error_codes.WEB_FETCH_FAILED and result.data["status_code"] == status
    assert response.raw.closed


@pytest.mark.parametrize("markup,expected", [
    ("<body>outside<main>main<article><p>article</p></article></main></body>", "article"),
    ("<body>outside<main><p>main</p></main></body>", "main"),
    ("<body><p>body</p></body>", "body"),
    ("<p>fragment</p>", "fragment"),
])
def test_article_main_body_and_document_fallback(markup, expected, transport):
    transport.return_value = html_response(markup.encode())
    assert web.fetch_web("https://example.com/").data["text"] == expected


def test_html_title_whitespace_inline_text_and_non_content_removal(transport):
    markup = """<head><title>  A &amp; B\n page </title><style>bad style</style></head>
    <body><h1>  Heading </h1><p> Hello  <strong>world</strong>! </p>
    <script>bad script</script><noscript>bad noscript</noscript><svg>bad svg</svg>
    <canvas>bad canvas</canvas><iframe>bad iframe</iframe><form>bad form</form>
    <p> Next&nbsp; paragraph. <br>New line.</p><footer>Semantic footer</footer></body>"""
    transport.return_value = html_response(markup.encode())
    result = web.fetch_web("https://example.com/")
    assert result.data["title"] == "A & B page"
    text = result.data["text"]
    assert "bad" not in text and "page" not in text
    assert "Hello world!" in text and "Next paragraph." in text and "Semantic footer" in text
    assert "\n" in text and "\n\n\n" not in text and "  " not in text
    assert text == text.strip()


@pytest.mark.parametrize("content_type,markup,expected", [
    ("application/xhtml+xml; charset=utf-8", '<html><title>XHTML</title><body>café</body></html>'.encode(), "café"),
    ("text/html", '<meta charset="utf-8"><p>İstanbul ☕</p>'.encode(), "İstanbul ☕"),
    ("text/html; charset=iso-8859-9", '<p>İstanbul</p>'.encode("iso-8859-9"), "İstanbul"),
])
def test_xhtml_and_charset_detection(content_type, markup, expected, transport):
    transport.return_value = html_response(markup, content_type=content_type)
    result = web.fetch_web("https://example.com/")
    assert result.success and result.data["text"] == expected


def test_truncation_counts_full_text_and_keeps_display_concise(transport):
    full_text = "Evidence Ω " * 300
    full_text = full_text.strip()
    transport.return_value = html_response(f"<title>Bounded</title><body>{full_text}</body>".encode())
    result = web.fetch_web("https://example.com/", max_chars=20)
    assert result.data["text"] == full_text[:20]
    assert result.data["total_chars"] == len(full_text) and result.data["truncated"] is True
    assert result.data["total_text_bytes"] == len(full_text.encode())
    assert result.message == result.display and len(result.display) < 300
    assert "Evidence" not in result.display and full_text not in result.message
    assert "truncated" in result.display and "continuation is unavailable" in result.display
    assert ToolOutputEnvelope.try_parse(result.to_tool_output()).data == result.data


@pytest.mark.parametrize("max_chars", [0, -1, 200001, True, "10", 1.5])
def test_invalid_text_bounds_rejected_before_network(max_chars, transport):
    result = web.fetch_web("https://example.com/", max_chars=max_chars)
    assert result.error_code == error_codes.WEB_FETCH_REJECTED
    transport.assert_not_called()


def test_declared_response_size_rejected_before_read(transport):
    response = html_response(headers={"Content-Length": str(web.MAX_RESPONSE_BYTES + 1)})
    response.raw.read = Mock(side_effect=AssertionError("Oversized body must not be read"))
    transport.return_value = response
    result = web.fetch_web("https://example.com/")
    assert result.error_code == error_codes.WEB_FETCH_RESPONSE_TOO_LARGE and response.raw.closed


@pytest.mark.parametrize("length", [None, "1", "invalid"])
def test_stream_size_is_enforced_independently_of_content_length(length, transport):
    response = html_response(headers={"Content-Length": length} if length else {})
    chunks = Mock()

    def stream(*, chunk_size):
        assert chunk_size == 16 * 1024
        yield b"x" * web.MAX_RESPONSE_BYTES
        yield b"x"
        chunks()  # Must never request the remainder of an arbitrarily large body.
        raise AssertionError("Read past the body bound")

    response.iter_content = stream
    transport.return_value = response
    result = web.fetch_web("https://example.com/", max_chars=1)
    assert result.error_code == error_codes.WEB_FETCH_RESPONSE_TOO_LARGE and response.raw.closed
    chunks.assert_not_called()


def test_exact_body_limit_is_allowed(transport):
    transport.return_value = html_response(b"x" * web.MAX_RESPONSE_BYTES)
    result = web.fetch_web("https://example.com/", max_chars=1)
    assert result.success and result.data["truncated"]
    assert result.data["total_chars"] == web.MAX_RESPONSE_BYTES


def test_real_adapter_streams_and_never_reads_redirect_body(monkeypatch):
    oversized_redirect_body = BytesIO(b"unused")
    oversized_redirect_body.read = Mock(side_effect=AssertionError("Redirect body read"))
    redirect = HTTPResponse(
        body=oversized_redirect_body, status=302,
        headers={"Location": "https://other.example/page", "Set-Cookie": "secret=value"},
        preload_content=False,
    )
    final = HTTPResponse(body=BytesIO(b"<head><title>Real adapter</title></head><body><p>Actual streamed evidence.</p></body>"),
                         status=200, headers={"Content-Type": "text/html"}, preload_content=False)
    calls = []
    responses = iter([redirect, final])

    def urlopen(pool, **kwargs):
        calls.append((pool.host, kwargs))
        return next(responses)

    monkeypatch.setattr(web._PublicHTTPSPool, "urlopen", urlopen)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:3128")
    result = web.fetch_web("https://example.com/")
    assert result.success and result.data["text"] == "Actual streamed evidence."
    assert [host for host, _kwargs in calls] == ["example.com", "other.example"]
    for _host, kwargs in calls:
        assert kwargs["redirect"] is False and kwargs["preload_content"] is False
        assert kwargs["decode_content"] is False
        assert "Cookie" not in kwargs["headers"]
    oversized_redirect_body.read.assert_not_called()
    assert oversized_redirect_body.closed and final.closed


def test_compressed_decoded_body_is_size_bounded(monkeypatch):
    import gzip
    final = HTTPResponse(
        body=BytesIO(gzip.compress(b"x" * (web.MAX_RESPONSE_BYTES + 1))), status=200,
        headers={"Content-Type": "text/html", "Content-Encoding": "gzip"}, preload_content=False,
    )
    monkeypatch.setattr(web._PublicHTTPSPool, "urlopen", lambda *_args, **_kwargs: final)
    result = web.fetch_web("https://example.com/")
    assert result.error_code == error_codes.WEB_FETCH_RESPONSE_TOO_LARGE and final.closed


def test_html_parser_errors_are_safe_structured_failures(monkeypatch, transport):
    monkeypatch.setattr(web, "_extract_html", Mock(side_effect=RuntimeError("secret parser internals")))
    result = web.fetch_web("https://example.com/")
    assert result.error_code == error_codes.WEB_FETCH_FAILED
    assert "secret" not in result.to_tool_output()


def test_dns_rebinding_is_blocked_at_actual_connection(public_dns):
    public_dns.side_effect = [dns_records("93.184.216.34"), dns_records("127.0.0.1")]
    result = web.fetch_web("https://example.com/")
    assert result.error_code == error_codes.WEB_FETCH_REJECTED
    web.create_connection.assert_not_called()


@pytest.mark.parametrize("connection_class", [web._PublicHTTPConnection, web._PublicHTTPSConnection])
def test_socket_uses_validated_numeric_ip_while_preserving_hostname(connection_class, monkeypatch):
    create = Mock(return_value=Mock())
    monkeypatch.setattr(web, "create_connection", create)
    conn = connection_class("example.com", port=443, timeout=5)
    assert conn._new_conn() is create.return_value
    assert create.call_args.args[:2] == (("93.184.216.34", 443), 5)
    assert conn.host == "example.com"


def test_adapter_preserves_https_tls_identity_and_uses_instance_local_pools():
    with closing(web._PublicHTTPAdapter()) as adapter:
        # HTTPAdapter is a documented extension point; this tests its actual
        # HTTPS pool/connection construction rather than a fake requests.get.
        prepared = requests.Request("GET", "https://example.com/").prepare()
        pool = adapter.get_connection_with_tls_context(prepared, verify=True, proxies={})
        conn = pool._new_conn()
        assert isinstance(conn, web._PublicHTTPSConnection)
        assert conn.host == "example.com" and conn.cert_reqs == "CERT_REQUIRED"
        assert conn.assert_hostname is None  # urllib3 verifies conn.host by default
        assert pool.host == "example.com"
    with closing(requests.adapters.HTTPAdapter()) as ordinary:
        assert ordinary.poolmanager.pool_classes_by_scheme["https"] is not web._PublicHTTPSPool


@pytest.mark.parametrize("error,expected", [
    (socket.timeout(), ConnectTimeoutError), (OSError(), web.NewConnectionError),
])
def test_socket_errors_keep_requests_timeout_contract(error, expected, monkeypatch):
    monkeypatch.setattr(web, "create_connection", Mock(side_effect=error))
    with pytest.raises(expected):
        web._PublicHTTPConnection("example.com", port=80, timeout=5)._new_conn()
