"""
EpiRAG — search.py
------------------
Multi-provider web search, free fallback chain:

  1. DuckDuckGo (ddg)
  2. Tavily

Tries DDG first. Falls back to Tavily only if DDG returns nothing.
Domain whitelist applied to both.
"""

import urllib.parse

ALLOWED_DOMAINS = [
    "arxiv.org", "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov",
    "semanticscholar.org", "nature.com", "science.org", "cell.com",
    "plos.org", "biorxiv.org", "medrxiv.org", "academic.oup.com",
    "wiley.com", "springer.com", "elsevier.com", "sciencedirect.com",
    "tandfonline.com", "sagepub.com", "jstor.org", "researchgate.net",
    "openalex.org", "europepmc.org", "who.int", "cdc.gov", "nih.gov",
    "pmc.ncbi.nlm.nih.gov", "royalsocietypublishing.org", "pnas.org",
    "bmj.com", "thelancet.com", "jamanetwork.com", "nejm.org",
    "frontiersin.org", "mdpi.com", "acm.org", "ieee.org",
    "dl.acm.org", "ieeexplore.ieee.org", "mathoverflow.net",
    "math.stackexchange.com", "stats.stackexchange.com"
]

MAX_RESULTS = 5


def _is_allowed(url: str) -> bool:
    if not url:
        return False
    try:
        host = urllib.parse.urlparse(url).netloc.lower().lstrip("www.")
        return any(host == d or host.endswith("." + d) for d in ALLOWED_DOMAINS)
    except Exception:
        return False


def _fmt(text: str, title: str, url: str, score: float = 0.5) -> dict:
    return {
        "text":       text,
        "source":     title or url,
        "similarity": round(score, 4),
        "url":        url,
        "type":       "web"
    }


# -- Provider 1: DuckDuckGo------------------------------------------------------------
def _search_ddg(query: str) -> list[dict]:
    try:
        from ddgs import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=MAX_RESULTS * 3):
                if _is_allowed(r.get("href", "")):
                    results.append(_fmt(
                        text  = r.get("body", ""),
                        title = r.get("title", ""),
                        url   = r.get("href", ""),
                        score = 0.6
                    ))
                    if len(results) >= MAX_RESULTS:
                        break
        return results
    except Exception as e:
        print(f"  [DDG] failed: {e}", flush=True)
        return []


# -- Provider 2: Tavily (free 1000/month) ------------------------------------------------------------
def _search_tavily(query: str, api_key: str) -> list[dict]:
    try:
        from tavily import TavilyClient
        client   = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=MAX_RESULTS,
            include_answer=False,
            topic="general",
            include_domains=ALLOWED_DOMAINS,
        )
        return [
            _fmt(
                text  = r.get("content", ""),
                title = r.get("title", r.get("url", "Web")),
                url   = r.get("url", ""),
                score = r.get("score", 0.5)
            )
            for r in response.get("results", [])
            if _is_allowed(r.get("url", ""))
        ]
    except Exception as e:
        print(f"  [Tavily] failed: {e}", flush=True)
        return []


# -- Main entry point ------------------------------------------------------------
def web_search(query: str, tavily_key: str = None, **kwargs) -> list[dict]:
    """
    Try DuckDuckGo first (always free, no key needed).
    Fall back to Tavily if DDG returns nothing.
    """
    print("  [Search] Trying DuckDuckGo...", flush=True)
    results = _search_ddg(query)
    if results:
        print(f"  [Search] DDG: {len(results)} results", flush=True)
        return results

    if tavily_key:
        print("  [Search] DDG empty, falling back to Tavily...", flush=True)
        results = _search_tavily(query, tavily_key)
        if results:
            print(f"  [Search] Tavily: {len(results)} results", flush=True)
            return results

    print("  [Search] All providers returned empty", flush=True)
    return []