import io
import time
import urllib.parse
from typing import Dict, Any, List, Tuple, Optional, Set
import concurrent.futures as cf

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Keyword Rank Checker (ValueSERP) — Fast", layout="wide")

st.title("🔎 Keyword Rank Checker — Fast & URL/Domain Aware (ValueSERP)")
st.caption("Uploads CSV (header can be row 2+). Columns needed: **Target URL** and **Keyword**. "
           "Forward-fills Target URL for subsequent keyword rows. Parallel requests + early stopping.")

# --------------------------- Sidebar controls ---------------------------
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("ValueSERP API Key", type="password", help="Paste your ValueSERP API key")
    override_domain = st.text_input(
        "Override domain (optional)",
        placeholder="example.com",
        help="If set, domain rank checks use this instead of extracting from the Target URL."
    )

    st.markdown("**Location & Google Settings**")
    location_value = st.text_input("Location", value="Australia",
                                   help="e.g. 'Australia' or 'Collingwood, Victoria, Australia'")
    gl = st.text_input("gl (country code)", value="au")
    hl = st.text_input("hl (language)", value="en")
    google_domain = st.text_input("google_domain", value="google.com.au")

    max_pages = st.number_input("Pages to check (1 page ≈ 10 results)", min_value=1, max_value=10, value=10, step=1)
    rps_per_worker = st.number_input("Requests per second per worker", min_value=0.2, max_value=10.0, value=1.0, step=0.2)

    st.divider()
    st.markdown("**Speed vs. Completeness**")
    stop_on_first_domain_hit = st.checkbox(
        "Stop a keyword once the domain ranks (fastest)",
        value=True,
        help="If on, we stop paging a keyword as soon as any URL from the domain is found."
    )
    require_exact_url = st.checkbox(
        "Also find exact Target URL rank if present",
        value=True,
        help="If on, we'll keep paging until either the exact Target URL is found OR pages are exhausted."
    )
    debug_preview = st.checkbox("Show first few organic URLs per page (debug)", value=False)

    st.divider()
    max_workers = st.slider("Parallel keywords (workers)", min_value=1, max_value=16, value=6, step=1)

# --------------------------- CSV upload ---------------------------
uploaded = st.file_uploader("Upload a CSV with **Target URL** and **Keyword**", type=["csv"])
st.write(
    "• Header row can be **row 2 or later**. The app scans for the row containing both **Target URL** and **Keyword**.\n"
    "• Blanks under **Target URL** are **forward-filled** so all keywords inherit the URL above.\n"
    "• The app returns domain-best rank and exact Target URL rank (if requested)."
)

# --------------------------- Helpers ---------------------------
API_BASE = "https://api.valueserp.com/search"

def _norm(s: str) -> str:
    return " ".join((s or "").strip().split()).lower()

def _looks_like(col_name: str, targets: List[str]) -> bool:
    n = _norm(col_name)
    return any(n == _norm(t) for t in targets)

def detect_header_row_and_columns(df_no_header: pd.DataFrame) -> Tuple[int, int, int]:
    """
    Return (header_row_idx, target_url_col_idx, keyword_col_idx), all 0-based.
    Scans up to first 50 rows to find a row that contains both 'Target URL' and 'Keyword' headers.
    """
    TARGET_CANDIDATES = ["target url", "target_url", "url", "page url"]
    KEYWORD_CANDIDATES = ["keyword", "keywords", "search term", "query"]

    max_scan = min(len(df_no_header), 50)
    for r in range(max_scan):
        row_vals = [str(x) for x in df_no_header.iloc[r].tolist()]
        normed = [_norm(v) for v in row_vals]
        t_idx = None
        k_idx = None
        for c_idx, val in enumerate(normed):
            if t_idx is None and _looks_like(val, TARGET_CANDIDATES):
                t_idx = c_idx
            if k_idx is None and _looks_like(val, KEYWORD_CANDIDATES):
                k_idx = c_idx
        if t_idx is not None and k_idx is not None:
            return r, t_idx, k_idx

    raise ValueError("Could not find a header row containing both 'Target URL' and 'Keyword'.")

def hostname_of(url: str) -> str:
    try:
        return urllib.parse.urlparse(url).hostname or ""
    except Exception:
        return ""

def extract_domain_from_url(url: str) -> str:
    host = hostname_of(url).lower()
    if host.startswith("www."):
        host = host[4:]
    return host

def domain_matches(target: str, url: str) -> bool:
    host = hostname_of(url).lower()
    t = (target or "").lower().lstrip(".")
    if host.startswith("www."): host = host[4:]
    if t.startswith("www."): t = t[4:]
    return bool(t) and (host == t or host.endswith("." + t))

def urls_equal(a: str, b: str) -> bool:
    try:
        pa = urllib.parse.urlparse(a)
        pb = urllib.parse.urlparse(b)
        host_a = pa.netloc.lower().lstrip("www.")
        host_b = pb.netloc.lower().lstrip("www.")
        path_a = pa.path.rstrip("/")
        path_b = pb.path.rstrip("/")
        return host_a == host_b and path_a == path_b
    except Exception:
        return a.rstrip("/") == b.rstrip("/")

def extract_organic_positions(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    organic = payload.get("organic_results", []) or payload.get("organic", []) or []
    for item in organic:
        pos = item.get("position") or item.get("rank") or item.get("position_on_page")
        link = item.get("link") or item.get("url")
        title = item.get("title") or item.get("headline")
        disp = item.get("displayed_link") or item.get("displayed_url") or hostname_of(link or "")
        if pos is None or not link:
            continue
        try:
            pos_int = int(pos)
        except Exception:
            continue
        out.append({
            "position": pos_int,
            "title": title or "",
            "link": link,
            "displayed_link": disp or ""
        })
    return out

def fetch_page(api_key: str, query: str, page: int, location_value: str, gl: str, hl: str, google_domain: str) -> Dict[str, Any]:
    params = {
        "api_key": api_key,
        "q": query,
        "location": location_value,
        "gl": gl,
        "hl": hl,
        "google_domain": google_domain,
        "page": page
    }
    r = requests.get(API_BASE, params=params, timeout=60)
    r.raise_for_status()
    return r.json()

# --------------------------- Per-keyword worker (sequential paging + early stop) ---------------------------
def process_keyword(
    kw: str,
    domains_for_kw: Set[str],
    target_urls_for_kw: Set[str],
    *,
    max_pages: int,
    api_key: str,
    location_value: str,
    gl: str,
    hl: str,
    google_domain: str,
    stop_on_first_domain_hit: bool,
    require_exact_url: bool,
    rps: float,
    debug_preview: bool = False,
) -> Dict[str, Any]:
    delay = 1.0 / max(0.01, float(rps))
    results_all_pages: List[Dict[str, Any]] = []
    # Track findings
    found_domains: Dict[str, Optional[Dict[str, Any]]] = {d: None for d in domains_for_kw}
    found_targets: Dict[str, Optional[Dict[str, Any]]] = {u: None for u in target_urls_for_kw}

    for page in range(1, int(max_pages) + 1):
        try:
            payload = fetch_page(api_key, kw, page, location_value, gl, hl, google_domain)
        except Exception as e:
            # skip on error; caller can log if needed
            break

        organic = extract_organic_positions(payload)
        if debug_preview:
            # Return top few for UI; omit noisy print here
            pass

        # Evaluate hits for early stop
        for item in organic:
            pos_abs = item["position"]
            if 1 <= item["position"] <= 10 and page > 1:
                pos_abs = (page - 1) * 10 + item["position"]
            rec = {
                "position": pos_abs,
                "page": page,
                "title": item["title"],
                "url": item["link"],
                "displayed_link": item["displayed_link"],
            }
            results_all_pages.append(rec)

            # Domain hits
            for d in list(found_domains.keys()):
                if found_domains[d] is None and domain_matches(d, rec["url"]):
                    found_domains[d] = rec

            # Exact URL hits
            if require_exact_url and target_urls_for_kw:
                for t in list(found_targets.keys()):
                    if found_targets[t] is None and urls_equal(t, rec["url"]):
                        found_targets[t] = rec

        # Early stop logic:
        # 1) If stop_on_first_domain_hit: as soon as ANY domain has a hit, break
        if stop_on_first_domain_hit and any(v is not None for v in found_domains.values()):
            break

        # 2) Otherwise, stop when all domains resolved AND (if requested) all targets resolved, or pages done
        all_domains_resolved = all(v is not None for v in found_domains.values()) if found_domains else True
        all_targets_resolved = all(v is not None for v in found_targets.values()) if (require_exact_url and found_targets) else True
        if all_domains_resolved and all_targets_resolved:
            break

        time.sleep(delay)

    return {
        "keyword": kw,
        "results": results_all_pages,
        "found_domains": found_domains,
        "found_targets": found_targets if require_exact_url else {},
    }

# --------------------------- Run button ---------------------------
run = st.button("Run ranking check", type="primary", disabled=uploaded is None)

if run:
    # Validate inputs
    if not api_key:
        st.error("Please paste your ValueSERP API key in the sidebar.")
        st.stop()
    if uploaded is None:
        st.error("Please upload a CSV file.")
        st.stop()

    # Read CSV without header first so we can self-detect header row
    try:
        df_raw = pd.read_csv(uploaded, header=None, dtype=str, keep_default_na=False)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    if df_raw.empty:
        st.error("The uploaded CSV is empty.")
        st.stop()

    # Detect header row + column indices
    try:
        header_row, t_col, k_col = detect_header_row_and_columns(df_raw)
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Build a proper DataFrame with detected headers
    header_vals = df_raw.iloc[header_row].tolist()
    df = df_raw.iloc[header_row + 1:].copy()
    df.columns = header_vals

    # Extract relevant columns by the detected indices
    target_col_name = df.columns[t_col]
    keyword_col_name = df.columns[k_col]

    slim = df[[target_col_name, keyword_col_name]].copy()
    slim[target_col_name] = slim[target_col_name].astype(str).str.strip()
    slim[keyword_col_name]  = slim[keyword_col_name].astype(str).str.strip()

    # Forward-fill blank Target URL, drop blank keywords
    slim[target_col_name].replace({"": pd.NA}, inplace=True)
    slim[target_col_name] = slim[target_col_name].ffill()
    slim[keyword_col_name].replace({"": pd.NA}, inplace=True)
    slim = slim.dropna(subset=[keyword_col_name])

    slim = slim.rename(columns={target_col_name: "Target URL", keyword_col_name: "Keyword"})
    slim = slim.drop_duplicates(subset=["Target URL", "Keyword"]).reset_index(drop=True)

    # Domain per row: override or extracted from URL
    if override_domain:
        base_domain = override_domain.strip().lower().lstrip(".")
        base_domain = base_domain[4:] if base_domain.startswith("www.") else base_domain
        slim["Domain"] = base_domain
    else:
        slim["Domain"] = slim["Target URL"].map(lambda u: extract_domain_from_url(str(u)))

    # Report detection
    st.success(f"Detected header row at **{header_row+1}**. Using columns: **Target URL**='{target_col_name}', **Keyword**='{keyword_col_name}'.")
    st.info(f"Tracking {len(slim)} URL–keyword pairs across {slim['Keyword'].nunique()} unique keywords.")

    # Build lookup sets per keyword
    kw_to_targets: Dict[str, Set[str]] = {}
    kw_to_domains: Dict[str, Set[str]] = {}
    for _, r in slim.iterrows():
        kw = r["Keyword"]
        kw_to_targets.setdefault(kw, set()).add(r["Target URL"])
        kw_to_domains.setdefault(kw, set()).add(r["Domain"])

    # Progress UI
    progress = st.progress(0.0)
    status = st.empty()

    # Run keywords in parallel
    keywords = list(kw_to_targets.keys())
    total = len(keywords)
    completed = 0

    results_by_keyword: Dict[str, Dict[str, Any]] = {}

    def submit_kw(executor, kw):
        return executor.submit(
            process_keyword,
            kw,
            kw_to_domains.get(kw, set()),
            kw_to_targets.get(kw, set()),
            max_pages=int(max_pages),
            api_key=api_key,
            location_value=location_value,
            gl=gl,
            hl=hl,
            google_domain=google_domain,
            stop_on_first_domain_hit=bool(stop_on_first_domain_hit),
            require_exact_url=bool(require_exact_url),
            rps=float(rps_per_worker),
            debug_preview=bool(debug_preview),
        )

    with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {submit_kw(executor, kw): kw for kw in keywords}
        for fut in cf.as_completed(future_map):
            kw = future_map[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {"keyword": kw, "results": [], "found_domains": {}, "found_targets": {}}
                st.warning(f"Error processing '{kw}': {e}")
            results_by_keyword[kw] = res
            completed += 1
            status.write(f"Processed **{completed}/{total}** keywords …")
            progress.progress(completed / total)

    status.write("Computing per-row results…")

    # Build detailed per-row output
    output_rows = []
    for _, row in slim.iterrows():
        target_url = row["Target URL"]
        keyword = row["Keyword"]
        domain = row["Domain"]

        res = results_by_keyword.get(keyword, {"results": [], "found_domains": {}, "found_targets": {}})
        # Select best domain hit for this row's domain
        dom_hit = res["found_domains"].get(domain)
        # Select best exact for this row's URL if tracked
        url_hit = res.get("found_targets", {}).get(target_url)

        output_rows.append({
            "Keyword": keyword,
            "Target URL": target_url,
            "Target Domain": domain,
            "Domain Found": dom_hit is not None,
            "Domain Best Position": int(dom_hit["position"]) if dom_hit else None,
            "Domain Best Page": int(dom_hit["page"]) if dom_hit else None,
            "Domain Best URL": dom_hit["url"] if dom_hit else None,
            "Exact URL Found": url_hit is not None,
            "Exact URL Position": int(url_hit["position"]) if url_hit else None,
            "Exact URL Page": int(url_hit["page"]) if url_hit else None,
            "Domain Found But Target URL Not Found": bool(dom_hit is not None and url_hit is None),
        })

    results_df = pd.DataFrame(output_rows).sort_values(
        by=["Domain Found", "Domain Best Position"],
        ascending=[False, True],
        na_position="last"
    ).reset_index(drop=True)

    # Summary per keyword (domain best only; choose first domain tied to that keyword)
    summary_rows = []
    for kw in keywords:
        row_domains = list(kw_to_domains.get(kw, []))
        dom = override_domain.strip().lower().lstrip(".") if override_domain else (row_domains[0] if row_domains else "")
        dom = dom[4:] if dom.startswith("www.") else dom
        res = results_by_keyword.get(kw, {"found_domains": {}})
        dom_hit = res["found_domains"].get(dom)
        summary_rows.append({
            "Keyword": kw,
            "Domain": dom,
            "Found": dom_hit is not None,
            "Best Position": int(dom_hit["position"]) if dom_hit else None,
            "Best Page": int(dom_hit["page"]) if dom_hit else None,
            "Best URL": dom_hit["url"] if dom_hit else None
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["Found", "Best Position"],
        ascending=[False, True],
        na_position="last"
    ).reset_index(drop=True)

    # --------------------------- Show tables ---------------------------
    st.subheader("📊 Summary (per keyword — domain best)")
    st.dataframe(summary_df, use_container_width=True)

    st.subheader("🔎 Detailed Results (per Target URL & Keyword)")
    st.dataframe(results_df, use_container_width=True)

    # --------------------------- Downloads ---------------------------
    ts = pd.Timestamp.utcnow().strftime("%Y%m%d-%H%M%S")
    base = f"valueserp_fast_ranks_{ts}"

    st.download_button(
        "⬇️ Download Summary CSV",
        data=summary_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{base}_summary.csv",
        mime="text/csv",
    )
    st.download_button(
        "⬇️ Download Detailed CSV",
        data=results_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{base}_detailed.csv",
        mime="text/csv",
    )

    st.success("Done ✅")
