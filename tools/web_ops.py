from contextlib import closing
from datetime import datetime, timezone
import ipaddress
import re
import socket
from urllib.parse import urldefrag, urljoin, urlsplit

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from pydantic import ValidationError
from requests.adapters import HTTPAdapter
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool
from urllib3.exceptions import ConnectTimeoutError, NewConnectionError, ReadTimeoutError
from urllib3.util.connection import create_connection

from core.models import ToolOutputEnvelope, ToolSerializableModel, WebFetchRequest
from core.error_codes import (
    WEB_FETCH_FAILED,
    WEB_FETCH_REJECTED,
    WEB_FETCH_RESPONSE_TOO_LARGE,
    WEB_FETCH_TIMEOUT,
    WEB_FETCH_UNSUPPORTED_CONTENT_TYPE,
)


# Body bytes after HTTP content decoding, independent of the extracted text cap.
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_REDIRECTS = 5
FETCH_TIMEOUT = (5, 15)  # connect/read seconds; no transport retries
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_HTML_TYPES = {"text/html", "application/xhtml+xml"}


class _RejectedURL(ValueError):
    """A URL or resolved address violates the public-page policy."""


def _is_public_address(value: str) -> bool:
    address = ipaddress.ip_address(value)
    if not address.is_global or any((
        address.is_private, address.is_loopback, address.is_link_local,
        address.is_multicast, address.is_reserved, address.is_unspecified,
    )):
        return False
    if isinstance(address, ipaddress.IPv6Address):
        if address.ipv4_mapped is not None:
            return _is_public_address(str(address.ipv4_mapped))
        # Transition addresses can embed a non-public IPv4 destination.
        if address.sixtofour is not None or address.teredo is not None:
            return False
        if address in ipaddress.ip_network("64:ff9b::/96"):
            return _is_public_address(str(ipaddress.IPv4Address(int(address) & 0xffffffff)))
    return True


def _public_addresses(host: str, port: int) -> list[str]:
    if host.rstrip(".").lower() == "localhost" or host.rstrip(".").lower().endswith(".localhost"):
        raise _RejectedURL("Localhost destinations are not allowed.")
    if "%" in host:
        raise _RejectedURL("Scoped addresses are not allowed.")
    try:
        addresses = [str(ipaddress.ip_address(host))]
    except ValueError:
        addresses = list(dict.fromkeys(
            record[4][0] for record in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        ))
    if not addresses or not all(_is_public_address(address) for address in addresses):
        raise _RejectedURL("The destination must resolve only to public addresses.")
    return addresses


def _parse_web_url(url: str):
    try:
        if url != url.strip() or "\\" in url or any(ord(char) < 32 or ord(char) == 127 for char in url):
            raise ValueError
        parsed = urlsplit(url)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            raise ValueError
        if parsed.username is not None or parsed.password is not None:
            raise ValueError
        port = parsed.port
        if port == 0:
            raise ValueError
    except ValueError:
        raise _RejectedURL("Use an HTTP/HTTPS URL with a hostname and no credentials.") from None
    return parsed


def normalize_web_url(url: str) -> str:
    """Requests-normalized requested identity, without fragments or DNS I/O."""
    _parse_web_url(url)
    prepared = requests.Request("GET", url).prepare()
    _parse_web_url(prepared.url)
    return urldefrag(prepared.url)[0]


def _validate_public_url(url: str) -> None:
    parsed = _parse_web_url(url)
    _public_addresses(parsed.hostname, parsed.port or (443 if parsed.scheme.lower() == "https" else 80))


class _PublicConnectionMixin:
    def _new_conn(self):
        # Resolve again at connection time, then connect to a validated numeric IP.
        # The original host remains on the connection for Host, TLS SNI and cert
        # verification. A DNS change cannot send the socket to an unchecked host.
        addresses = _public_addresses(self.host, self.port)
        try:
            return create_connection(
                (addresses[0], self.port), self.timeout,
                source_address=self.source_address, socket_options=self.socket_options,
            )
        except socket.timeout as exc:
            raise ConnectTimeoutError(self, "Web connection timed out.") from exc
        except OSError as exc:
            raise NewConnectionError(self, "Web connection failed.") from exc


class _PublicHTTPConnection(_PublicConnectionMixin, HTTPConnection):
    pass


class _PublicHTTPSConnection(_PublicConnectionMixin, HTTPSConnection):
    pass


class _PublicHTTPPool(HTTPConnectionPool):
    ConnectionCls = _PublicHTTPConnection


class _PublicHTTPSPool(HTTPSConnectionPool):
    ConnectionCls = _PublicHTTPSConnection


class _PublicHTTPAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        super().init_poolmanager(*args, **kwargs)
        # Instance-local mapping; never change urllib3's global pool classes.
        self.poolmanager.pool_classes_by_scheme = {
            "http": _PublicHTTPPool, "https": _PublicHTTPSPool,
        }


def _normalize_text(text: str) -> str:
    lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _extract_html(body: bytes, encoding: str | None) -> tuple[str, str]:
    soup = BeautifulSoup(body, "html.parser", from_encoding=encoding)
    title = " ".join(soup.title.get_text(" ", strip=True).split()) if soup.title else ""
    for element in soup.find_all(["script", "style", "noscript", "svg", "canvas", "iframe", "form"]):
        element.decompose()
    if soup.head:
        soup.head.decompose()
    content = soup.find("article") or soup.find("main") or soup.body or soup
    # Add boundaries at block elements, retaining inline phrasing and semantics.
    for element in content.find_all([
        "p", "div", "section", "article", "main", "header", "footer", "aside", "nav",
        "h1", "h2", "h3", "h4", "h5", "h6", "li", "ul", "ol", "dl", "dt", "dd",
        "table", "tr", "blockquote", "pre", "br", "hr",
    ]):
        element.insert_before("\n")
        element.insert_after("\n")
    for cell in content.find_all(["td", "th"]):
        cell.append(" ")
    return title[:1000], _normalize_text(content.get_text())


def fetch_web(url: str, max_chars: int = 50_000) -> ToolOutputEnvelope:
    """Read one public HTML page; bounded/truncating, without continuation.

    Stream at most 2 MiB of decoded body (16 KiB chunks), with five manually
    validated redirects. No cookies, netrc credentials, proxies or HTTP retries.
    """
    final_url = None
    status_code = None
    content_type = None

    def failure(code: str, message: str) -> ToolOutputEnvelope:
        return ToolOutputEnvelope(
            success=False, message=message, display=message, error_code=code,
            data={"url": url, "final_url": final_url, "status_code": status_code, "content_type": content_type},
        )

    try:
        request = WebFetchRequest(url=url, max_chars=max_chars)
        current_url = request.url
        with closing(_PublicHTTPAdapter(max_retries=0)) as adapter:
            for hop in range(MAX_REDIRECTS + 1):
                _validate_public_url(current_url)
                prepared = requests.Request("GET", current_url, headers={
                    "User-Agent": "CortexNode/1.0 (public HTML reader)",
                    "Accept": "text/html, application/xhtml+xml",
                    "Accept-Encoding": "identity",
                }).prepare()
                # Requests URL preparation (including IDNA/escaping) must not
                # change which destination was authorized.
                if prepared.url != current_url:
                    _validate_public_url(prepared.url)
                final_url = prepared.url
                status_code = content_type = None
                # Adapter.send avoids Session's automatic redirect-body reads,
                # cookie persistence, netrc and environment proxy configuration.
                with adapter.send(prepared, stream=True, timeout=FETCH_TIMEOUT, verify=True, proxies={}) as response:
                    status_code = response.status_code
                    content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
                    if status_code in _REDIRECT_STATUSES:
                        location = response.headers.get("Location")
                        if not location or hop == MAX_REDIRECTS:
                            return failure(WEB_FETCH_FAILED, "Web fetch failed: missing redirect destination or redirect limit exceeded.")
                        current_url = urljoin(final_url, location)
                        continue
                    response.raise_for_status()
                    if not 200 <= status_code < 300:
                        return failure(WEB_FETCH_FAILED, f"Web fetch failed: HTTP {status_code}.")
                    if content_type not in _HTML_TYPES:
                        return failure(WEB_FETCH_UNSUPPORTED_CONTENT_TYPE, "Web fetch supports HTML content only.")
                    length = response.headers.get("Content-Length", "")
                    if length.isdigit() and int(length) > MAX_RESPONSE_BYTES:
                        return failure(WEB_FETCH_RESPONSE_TOO_LARGE, "Web response exceeds the 2 MiB body limit.")
                    body = bytearray()
                    for chunk in response.iter_content(chunk_size=16 * 1024):
                        if len(body) + len(chunk) > MAX_RESPONSE_BYTES:
                            return failure(WEB_FETCH_RESPONSE_TOO_LARGE, "Web response exceeds the 2 MiB body limit.")
                        body.extend(chunk)
                    # Respect explicit HTTP charset; otherwise let BS detect
                    # HTML meta/BOM instead of Requests' implicit Latin-1 default.
                    encoding = None
                    if "charset=" in response.headers.get("Content-Type", "").lower():
                        encoding = requests.utils.get_encoding_from_headers(response.headers)
                    title, full_text = _extract_html(bytes(body), encoding)
                    text = full_text[:request.max_chars]
                    truncated = len(text) < len(full_text)
                    message = (
                        f"Fetched web page: {title[:160] or '(untitled)'}\n"
                        f"URL: {final_url[:500]}\n"
                        f"Extracted {len(text)} of {len(full_text)} characters."
                        + (" Text is truncated; continuation is unavailable." if truncated else "")
                    )
                    return ToolOutputEnvelope(success=True, message=message, display=message, data={
                        "url": request.url, "final_url": final_url, "status_code": status_code,
                        "content_type": content_type, "title": title, "text": text,
                        "retrieved_at": datetime.now(timezone.utc).isoformat(),
                        "total_chars": len(full_text), "truncated": truncated,
                        "total_text_bytes": len(full_text.encode("utf-8")),
                    })
    except (ValidationError, _RejectedURL):
        return failure(WEB_FETCH_REJECTED, "Web fetch rejected: use a public HTTP/HTTPS URL without credentials and valid max_chars (1..200000).")
    except requests.Timeout:
        return failure(WEB_FETCH_TIMEOUT, "Web fetch timed out.")
    except requests.ConnectionError as exc:
        # Requests wraps streaming read timeouts in ConnectionError.
        if any(isinstance(arg, (ReadTimeoutError, ConnectTimeoutError)) for arg in exc.args):
            return failure(WEB_FETCH_TIMEOUT, "Web fetch timed out.")
        return failure(WEB_FETCH_FAILED, "Web fetch failed: connection unavailable.")
    except requests.HTTPError:
        return failure(WEB_FETCH_FAILED, f"Web fetch failed: HTTP {status_code}.")
    except (requests.RequestException, OSError, ValueError):
        return failure(WEB_FETCH_FAILED, "Web fetch failed: unable to read the page.")
    except Exception:
        # Keep parser/transport failures inside the ordinary tool envelope too.
        return failure(WEB_FETCH_FAILED, "Web fetch failed: unable to read the page.")


def get_web_tools():
    @tool("web_fetch", args_schema=WebFetchRequest)
    def web_fetch(url: str, max_chars: int = 50_000) -> str:
        """Read current text of one known public HTML URL; may truncate, with no continuation."""
        return fetch_web(url, max_chars=max_chars).to_tool_output()

    return [web_fetch]

class WebSearchResult(ToolSerializableModel):
    query: str
    results: list[dict]  # [{"title": "...", "url": "...", "snippet": "..."}]

def search_web(query: str, max_results: int = 5) -> ToolOutputEnvelope:
    """Search DuckDuckGo Lite (no API key, local-only scraping)."""
    try:
        url = "https://lite.duckduckgo.com/lite/"
        params = {"q": query}
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        results = []
        
        for row in soup.find_all("tr")[1:max_results+1]:
            cols = row.find_all("td")
            if len(cols) >= 2:
                title_link = cols[0].find("a")
                snippet = cols[1].get_text(strip=True)
                if title_link:
                    results.append({
                        "title": title_link.get_text(),
                        "url": title_link.get("href", ""),
                        "snippet": snippet[:200]
                    })
        
        display_text = f"Found {len(results)} results for '{query}':\n"
        for r in results:
            display_text += f"- {r['title']}: {r['snippet']}\n"
        
        return ToolOutputEnvelope(
            success=True,
            message=display_text,
            data={"query": query, "results": results},
            display=display_text
        )
    except Exception as e:
        return ToolOutputEnvelope(
            success=False,
            message=f"Web search failed: {str(e)}",
            data={},
            error_code="WEB_SEARCH_FAILED",
            error_details={"error": str(e)}
        )
