import requests
from bs4 import BeautifulSoup
from core.models import ToolResult, ToolSerializableModel
from core.error_codes import ERROR_CODES

class WebSearchResult(ToolSerializableModel):
    query: str
    results: list[dict]  # [{"title": "...", "url": "...", "snippet": "..."}]

def search_web(query: str, max_results: int = 5) -> ToolResult:
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
        
        return ToolResult(
            success=True,
            message=display_text,
            data={"query": query, "results": results},
            display=display_text
        )
    except Exception as e:
        return ToolResult(
            success=False,
            message=f"Web search failed: {str(e)}",
            data={},
            error_code="WEB_SEARCH_FAILED",
            error_details={"error": str(e)}
        )