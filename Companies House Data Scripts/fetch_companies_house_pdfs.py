#!/usr/bin/env python3
"""
Fetch Companies House founding/closure dates and download all CS01/SH01/AR01/IN01 PDFs.

NEEDS AN CH API KEY IN LINE 705 TO RUN!!!

Changes vs. original:
- Adds full pagination for filing history.
- Downloads all available CS01 + SH01 + AR01 + IN01 PDFs for matched companies.
- Concurrent, retrying downloads (document API) with large connection pools.
- Skips already-downloaded files.
- Writes success/failure logs indicating which companies worked vs. not.
"""

from __future__ import annotations

import csv
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, urlunparse
import re

import requests
from dotenv import load_dotenv
from rapidfuzz import fuzz
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


logger = logging.getLogger("companies_house_dates")

CORPORATE_SUFFIXES_SINGLE = {
    "LIMITED",
    "LTD",
    "PLC",
    "LLP",
    "LP",
    "L.P",
    "L.L.P",
    "LIMITED.",
    "LTD.",
}

CORPORATE_SUFFIXES_MULTI = {
    ("PUBLIC", "LIMITED", "COMPANY"),
    ("LIMITED", "COMPANY"),
    ("PUBLIC", "LIMITED"),
}

AMPERSAND_EQUIVALENTS = {"&": "AND"}


@dataclass
class MatchOutcome:
    """Represents the outcome of trying to match an organisation to Companies House."""
    matched: bool
    reason: str
    company_number: Optional[str] = None
    matched_name: Optional[str] = None
    match_score: Optional[float] = None
    match_source: Optional[str] = None
    company_status: Optional[str] = None
    date_of_creation: Optional[str] = None
    date_of_cessation: Optional[str] = None
    search_url: Optional[str] = None
    search_query: Optional[str] = None
    # Links from company profile
    link_filing_history: Optional[str] = None
    link_officers: Optional[str] = None
    link_psc: Optional[str] = None
    link_charges: Optional[str] = None
    link_insolvency: Optional[str] = None
    link_exemptions: Optional[str] = None
    # Filing history stamps
    last_sh01_date: Optional[str] = None
    last_cs01_date: Optional[str] = None
    last_nm01_date: Optional[str] = None
    last_accounts_made_up_to: Optional[str] = None
    filings_total: Optional[int] = None
    # Exemptions (IPO/listing proxy)
    psc_exempt_uk_from: Optional[str] = None
    psc_exempt_uk_to: Optional[str] = None
    psc_exempt_eu_from: Optional[str] = None
    psc_exempt_eu_to: Optional[str] = None


class CompaniesHouseClient:
    """Minimal client for the Companies House REST + Document API with retry and caching."""

    BASE_URL = "https://api.company-information.service.gov.uk"
    DOC_BASE_URL = "https://document-api.company-information.service.gov.uk"
    FRONTEND_DOC_HOST = "frontend-doc-api.company-information.service.gov.uk"

    def __init__(self, api_key: str, request_pause: float = 0.0, timeout: float = 10.0,
                 pool_connections: int = 64, pool_maxsize: int = 64) -> None:
        if not api_key:
            raise ValueError("Companies House API key must be provided.")
        self._api_key = api_key
        self._pause = max(0.0, request_pause)
        self._timeout = timeout
        self._session = self._build_session(pool_connections, pool_maxsize)
        self._search_cache: Dict[Tuple[str, int], Dict[str, Any]] = {}
        self._profile_cache: Dict[str, Dict[str, Any]] = {}

    def _build_session(self, pool_connections: int, pool_maxsize: int) -> requests.Session:
        session = requests.Session()
        retry_strategy = Retry(
            total=5,
            status=5,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=False,  # retry on all, we gate with status_forcelist
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy,
                              pool_connections=pool_connections,
                              pool_maxsize=pool_maxsize)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.BASE_URL}{path}"
        response = self._session.get(
            url,
            params=params,
            auth=(self._api_key, ""),
            timeout=self._timeout,
        )
        if self._pause:
            time.sleep(self._pause)
        response.raise_for_status()
        return response.json()

    # ---------- Core API ----------
    def search_companies(self, query: str, items_per_page: int = 20) -> Dict[str, Any]:
        cache_key = (query.strip().lower(), items_per_page)
        if cache_key in self._search_cache:
            return self._search_cache[cache_key]
        params = {"q": query, "items_per_page": items_per_page}
        data = self._get("/search/companies", params=params)
        self._search_cache[cache_key] = data
        return data

    def get_company_profile(self, company_number: str) -> Dict[str, Any]:
        if company_number in self._profile_cache:
            return self._profile_cache[company_number]
        data = self._get(f"/company/{company_number}")
        self._profile_cache[company_number] = data
        return data

    def get_filing_history_page(self, company_number: str, items_per_page: int = 100, start_index: int = 0) -> Dict[str, Any]:
        params = {"items_per_page": items_per_page, "start_index": start_index}
        return self._get(f"/company/{company_number}/filing-history", params=params)

    def iter_all_filing_history(self, company_number: str, items_per_page: int = 100) -> Iterable[Dict[str, Any]]:
        """Yield ALL filing history items, paginating as needed."""
        start_index = 0
        total_count: Optional[int] = None
        while True:
            page = self.get_filing_history_page(company_number, items_per_page, start_index)
            items = page.get("items") or []
            for it in items:
                yield it
            if total_count is None:
                total_count = page.get("total_count") or 0
            start_index += len(items)
            if not items or start_index >= total_count:
                break

    def get_exemptions(self, company_number: str) -> Dict[str, Any]:
        return self._get(f"/company/{company_number}/exemptions")

    # ---------- Document API ----------
    @staticmethod
    def _normalize_document_metadata_url(url_or_path: str) -> str:
        """Ensure we have a proper document metadata URL on the DOC_BASE_URL host."""
        if not url_or_path:
            return ""
        # If it's a relative /document/<id> path, attach DOC_BASE_URL
        if url_or_path.startswith("/document/"):
            return f"{CompaniesHouseClient.DOC_BASE_URL}{url_or_path}"
        # If full URL, possibly on frontend-doc host; swap host to document-api
        parsed = urlparse(url_or_path)
        if parsed.netloc == CompaniesHouseClient.FRONTEND_DOC_HOST:
            parsed = parsed._replace(netloc=urlparse(CompaniesHouseClient.DOC_BASE_URL).netloc)
        # Some responses may already be on DOC_BASE_URL; pass through
        return urlunparse(parsed)

    def _metadata_to_content_url(self, metadata_url: str) -> str:
        """Convert a metadata URL to the binary PDF content URL."""
        norm = self._normalize_document_metadata_url(metadata_url).rstrip("/")
        if not norm:
            return ""
        if not norm.endswith("/content"):
            norm = f"{norm}/content"
        return norm

    def download_document_to_path(self, metadata_url: str, out_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Download a single Companies House document (PDF) via Document API.
        Returns (success, error_message).
        """
        content_url = self._metadata_to_content_url(metadata_url)
        if not content_url:
            return False, "missing_document_metadata_link"

        try:
            with self._session.get(
                content_url,
                auth=(self._api_key, ""),
                timeout=self._timeout,
                headers={"Accept": "application/pdf"},
                stream=True,
            ) as r:
                if self._pause:
                    time.sleep(self._pause)
                if r.status_code == 404:
                    return False, "document_not_found_404"
                if r.status_code == 403:
                    return False, "forbidden_403"
                r.raise_for_status()
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with out_path.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 15):
                        if chunk:
                            f.write(chunk)
            return True, None
        except requests.HTTPError as exc:
            return False, f"http_error_{exc.response.status_code if exc.response else 'unknown'}"
        except requests.RequestException as exc:
            return False, f"request_error_{exc.__class__.__name__}"
        except OSError as exc:
            return False, f"io_error_{exc.__class__.__name__}"


def _max_iso(a: Optional[str], b: Optional[str]) -> Optional[str]:
    """Return the later of two YYYY-MM-DD strings (lexicographic safe) or the non-empty one."""
    if not a:
        return b
    if not b:
        return a
    return max(a, b)


def normalise_company_name(name: str) -> str:
    uppercase = name.upper()
    for symbol, replacement in AMPERSAND_EQUIVALENTS.items():
        uppercase = uppercase.replace(symbol, f" {replacement} ")
    cleaned = re.sub(r"[^\w\s]", " ", uppercase)
    tokens = [token for token in cleaned.split() if token]
    stripped = strip_corporate_suffix(" ".join(tokens))
    return stripped


def generate_query_variants(org_name: str, legal_name: Optional[str]) -> List[str]:
    variants: List[str] = []
    seen: set[str] = set()

    def add_variant(name: str) -> None:
        candidate = " ".join(name.strip().split())
        if not candidate:
            return
        if candidate.lower() not in seen:
            seen.add(candidate.lower())
            variants.append(candidate)

    for base in (legal_name, org_name):
        if not base:
            continue
        add_variant(base)
        stripped = strip_corporate_suffix(base)
        if stripped and stripped.lower() != base.lower():
            add_variant(stripped)
        if stripped:
            lower_base = base.strip().lower()
            if not lower_base.endswith((" limited", " ltd")):
                add_variant(f"{stripped} limited")
                add_variant(f"{stripped} ltd")
    return variants or []


def strip_corporate_suffix(name: str) -> str:
    tokens = name.strip().split()
    while tokens:
        upper_tokens = [token.upper().rstrip(".") for token in tokens]
        removed = False
        for suffix in CORPORATE_SUFFIXES_MULTI:
            length = len(suffix)
            if length <= len(upper_tokens) and tuple(upper_tokens[-length:]) == suffix:
                tokens = tokens[:-length]
                removed = True
                break
        if removed:
            continue
        if upper_tokens and upper_tokens[-1] in CORPORATE_SUFFIXES_SINGLE:
            tokens.pop()
            continue
        break
    return " ".join(tokens)


def score_candidates(
    query_variants: Sequence[str], items: Sequence[Dict[str, Any]]
) -> List[Tuple[float, str, str, str, Dict[str, Any]]]:
    scored: List[Tuple[float, str, str, str, Dict[str, Any]]] = []
    for item in items:
        candidate_name = item.get("title", "") or ""
        candidate_variants: List[Tuple[str, str]] = [(candidate_name, "title")]
        for previous in item.get("previous_company_names", []) or []:
            previous_name = previous.get("name") or ""
            if previous_name:
                candidate_variants.append((previous_name, "previous_company_name"))

        best_score = -1.0
        best_match_name = candidate_name
        best_match_source = "title"
        best_query_used = query_variants[0] if query_variants else ""

        for query in query_variants:
            normalised_query = normalise_company_name(query)
            for candidate_variant, source in candidate_variants:
                normalised_candidate = normalise_company_name(candidate_variant)
                raw_score = fuzz.token_set_ratio(query, candidate_variant)
                normalised_score = fuzz.token_set_ratio(normalised_query, normalised_candidate)
                combined = max(raw_score, normalised_score)
                if combined > best_score:
                    best_score = combined
                    best_match_name = candidate_variant
                    best_match_source = source
                    best_query_used = query

        scored.append((best_score, best_match_name, best_match_source, best_query_used, item))

    scored.sort(key=lambda s: s[0], reverse=True)
    return scored


def make_search_url(query: str) -> str:
    return f"https://find-and-update.company-information.service.gov.uk/search?q={query.replace(' ', '+')}"


def match_company(
    client: CompaniesHouseClient,
    org_name: str,
    legal_name: Optional[str],
    *,
    items_per_page: int,
    min_score: float,
) -> MatchOutcome:
    """Search and pick the best Companies House match for the provided organisation name."""
    query_variants = generate_query_variants(org_name, legal_name)
    if not query_variants:
        return MatchOutcome(matched=False, reason="missing_org_name")

    best_score = -1.0
    best_result: Optional[Tuple[float, str, str, str, Dict[str, Any]]] = None
    best_search_query = None
    best_search_url = None

    for query in query_variants:
        try:
            search_results = client.search_companies(query, items_per_page=items_per_page)
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response else "unknown"
            logger.warning("HTTP error while searching for '%s': %s", query, exc)
            return MatchOutcome(
                matched=False,
                reason=f"search_http_{status_code}",
                search_url=make_search_url(query),
                search_query=query,
            )
        except requests.RequestException as exc:
            logger.warning("Request error while searching for '%s': %s", query, exc)
            return MatchOutcome(
                matched=False,
                reason="search_request_error",
                search_url=make_search_url(query),
                search_query=query,
            )

        items: Sequence[Dict[str, Any]] = search_results.get("items") or []
        if not items:
            continue

        scored_items = score_candidates(query_variants, items)
        if not scored_items:
            continue
        score, matched_name, match_source, query_used, best_item_candidate = scored_items[0]

        if score > best_score:
            best_score = score
            best_result = (score, matched_name, match_source, query_used, best_item_candidate)
            best_search_query = query
            best_search_url = make_search_url(query)

        if best_score >= 99:
            break

    if not best_result:
        return MatchOutcome(
            matched=False,
            reason="no_results",
            search_url=best_search_url or (make_search_url(query_variants[0]) if query_variants else None),
            search_query=best_search_query,
        )

    score, matched_name, match_source, query_used, best_item = best_result

    if score < min_score:
        return MatchOutcome(
            matched=False,
            reason="below_score_threshold",
            match_score=score,
            matched_name=matched_name,
            match_source=match_source,
            search_url=best_search_url,
            search_query=query_used,
        )

    company_number = best_item.get("company_number")
    if not company_number:
        return MatchOutcome(
            matched=False,
            reason="missing_company_number",
            match_score=score,
            matched_name=matched_name,
            match_source=match_source,
            search_url=best_search_url,
            search_query=query_used,
        )

    try:
        profile = client.get_company_profile(company_number)
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response else "unknown"
        logger.warning("HTTP error while retrieving profile %s: %s", company_number, exc)
        return MatchOutcome(
            matched=False,
            reason=f"profile_http_{status_code}",
            company_number=company_number,
            match_score=score,
            matched_name=matched_name,
            match_source=match_source,
            search_url=best_search_url,
            search_query=query_used,
        )
    except requests.RequestException as exc:
        logger.warning("Request error while retrieving profile %s: %s", company_number, exc)
        return MatchOutcome(
            matched=False,
            reason="profile_request_error",
            company_number=company_number,
            match_score=score,
            matched_name=matched_name,
            match_source=match_source,
            search_url=best_search_url,
            search_query=query_used,
        )

    # ---- Links from profile.links (handy for debugging / later calls)
    links = profile.get("links", {}) or {}
    link_filing_history = links.get("filing_history")
    link_officers = links.get("officers")
    link_psc = links.get("persons_with_significant_control")
    link_charges = links.get("charges")
    link_insolvency = links.get("insolvency")
    link_exemptions = links.get("exemptions") or f"/company/{company_number}/exemptions"

    # ---- Exemptions (IPO/listing proxy)
    psc_exempt_uk_from = psc_exempt_uk_to = None
    psc_exempt_eu_from = psc_exempt_eu_to = None
    try:
        ex = client.get_exemptions(company_number) or {}
        ex_map = ex.get("exemptions", {}) or {}
        for key, obj in ex_map.items():
            items = []
            if isinstance(obj, dict):
                items = obj.get("items") or []
            elif isinstance(obj, list):
                items = obj
            is_eu = "eu" in (key or "").lower()
            for item in items:
                ef = item.get("exempt_from")
                et = item.get("exempt_to")
                if is_eu:
                    psc_exempt_eu_from = _max_iso(psc_exempt_eu_from, ef)
                    psc_exempt_eu_to = _max_iso(psc_exempt_eu_to, et)
                else:
                    psc_exempt_uk_from = _max_iso(psc_exempt_uk_from, ef)
                    psc_exempt_uk_to = _max_iso(psc_exempt_uk_to, et)
    except requests.RequestException as exc:
        logger.warning("Exemptions fetch failed for %s: %s", company_number, exc)

    # ---- Filing history stamps (event dates and totals)
    last_sh01_date = last_cs01_date = last_nm01_date = None
    last_accounts_made_up_to = None
    filings_total: Optional[int] = None
    try:
        # Only fetch one page here for speed; iter_all used later for downloads anyway
        fh = client.get_filing_history_page(company_number, items_per_page=100) or {}
        filings_total = fh.get("total_count")
        for item in (fh.get("items") or []):
            code = (item.get("type") or "").upper()
            d = item.get("date") or item.get("action_date")
            if code == "SH01":
                last_sh01_date = _max_iso(last_sh01_date, d)
            elif code == "CS01":
                last_cs01_date = _max_iso(last_cs01_date, d)
            elif code == "NM01":
                last_nm01_date = _max_iso(last_nm01_date, d)
            desc_vals = item.get("description_values") or {}
            made_up_to = desc_vals.get("made_up_to")
            if made_up_to:
                last_accounts_made_up_to = _max_iso(last_accounts_made_up_to, made_up_to)
    except requests.RequestException as exc:
        logger.warning("Filing history fetch failed for %s: %s", company_number, exc)

    return MatchOutcome(
        matched=True,
        reason="matched",
        company_number=company_number,
        matched_name=matched_name,
        match_score=score,
        match_source=match_source,
        company_status=profile.get("company_status"),
        date_of_creation=profile.get("date_of_creation"),
        date_of_cessation=profile.get("date_of_cessation"),
        search_url=best_search_url,
        search_query=query_used,
        # links
        link_filing_history=link_filing_history,
        link_officers=link_officers,
        link_psc=link_psc,
        link_charges=link_charges,
        link_insolvency=link_insolvency,
        link_exemptions=link_exemptions,
        # filing stamps
        last_sh01_date=last_sh01_date,
        last_cs01_date=last_cs01_date,
        last_nm01_date=last_nm01_date,
        last_accounts_made_up_to=last_accounts_made_up_to,
        filings_total=filings_total,
        # exemptions
        psc_exempt_uk_from=psc_exempt_uk_from,
        psc_exempt_uk_to=psc_exempt_uk_to,
        psc_exempt_eu_from=psc_exempt_eu_from,
        psc_exempt_eu_to=psc_exempt_eu_to,
    )


def iter_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Input file {path} has no header row.")
        for row in reader:
            yield {key: (value or "").strip() for key, value in row.items()}


def write_rows(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_environment(env_path: Optional[Path]) -> str:
    if env_path and env_path.exists():
        load_dotenv(dotenv_path=env_path)
        return f"Loaded environment variables from explicit path: {env_path}"
    if env_path and not env_path.exists():
        logger.warning("Specified env file %s does not exist. Falling back to defaults.", env_path)

    repo_root = Path(__file__).resolve().parent

    local_env = repo_root / ".env"
    if local_env.exists():
        load_dotenv(dotenv_path=local_env)
        return f"Loaded environment variables from local file: {local_env}"

    parent_env = repo_root.parent / ".env"
    if parent_env.exists():
        load_dotenv(dotenv_path=parent_env)
        return f"Loaded environment variables from parent file: {parent_env}"

    load_dotenv(override=False)
    return "Loaded environment variables from existing process environment."


def safe_filename(text: str) -> str:
    text = re.sub(r"[^\w.\-]+", "_", text.strip())
    return text.strip("_")[:120] or "untitled"


def collect_target_links(client: CompaniesHouseClient, company_number: str,
                         allowed_types: Sequence[str]) -> List[Tuple[str, Dict[str, Any]]]:
    """Collect (document_metadata_url, filing_item) for all target filing codes."""
    allowed = {t.upper() for t in allowed_types}
    links: List[Tuple[str, Dict[str, Any]]] = []
    for item in client.iter_all_filing_history(company_number, items_per_page=100):
        code = (item.get("type") or "").upper()
        if code not in allowed:
            continue
        item_links = item.get("links") or {}
        meta = item_links.get("document_metadata") or item_links.get("document")
        if meta:
            links.append((meta, item))
    return links


def build_output_path(base_dir: Path, company_number: str, company_name: str, item: Dict[str, Any]) -> Path:
    code = (item.get("type") or "DOC").upper()
    date_str = item.get("date") or item.get("action_date") or "unknown_date"
    txid = item.get("transaction_id") or item.get("barcode") or ""
    name_part = safe_filename(f"{company_number}_{company_name}")[:100]
    fname = safe_filename(f"{date_str}_{code}_{txid}.pdf")
    return base_dir / name_part / code / fname


def download_company_documents(
    client: CompaniesHouseClient,
    company_number: str,
    company_name: str,
    base_dir: Path,
    allowed_types: Sequence[str],
    executor: ThreadPoolExecutor,
    skip_existing: bool = True,
) -> Dict[str, Any]:
    """
    Download all target PDFs (CS01/SH01/AR01/IN01) for a company concurrently.
    Returns summary dict with per-code counts and an error list.
    """
    allowed_upper = [t.upper() for t in allowed_types]
    all_links = collect_target_links(client, company_number, allowed_upper)

    # dynamic counts
    found_counts = {code: 0 for code in allowed_upper}
    for _, item in all_links:
        code = (item.get("type") or "").upper()
        if code in found_counts:
            found_counts[code] += 1

    futures = {}
    skipped_existing = 0
    for meta_url, item in all_links:
        out_path = build_output_path(base_dir, company_number, company_name, item)
        if skip_existing and out_path.exists():
            skipped_existing += 1
            continue
        fut = executor.submit(client.download_document_to_path, meta_url, out_path)
        futures[fut] = (out_path, (item.get("type") or "").upper())

    downloaded_counts = {code: 0 for code in allowed_upper}
    errors: List[str] = []

    for fut in as_completed(futures):
        out_path, code = futures[fut]
        ok, err = fut.result()
        if ok:
            if code in downloaded_counts:
                downloaded_counts[code] += 1
        else:
            errors.append(f"{code}:{out_path.name}:{err}")

    summary: Dict[str, Any] = {
        "skipped_existing": skipped_existing,
        "errors": errors,
        "found_total": sum(found_counts.values()),
        "downloaded_total": sum(downloaded_counts.values()),
    }
    # Flatten per-code counts (e.g., found_cs01, downloaded_ar01)
    for code in allowed_upper:
        lc = code.lower()
        summary[f"found_{lc}"] = found_counts.get(code, 0)
        summary[f"downloaded_{lc}"] = downloaded_counts.get(code, 0)
    return summary


@dataclass
class ScriptConfig:
    """Holds script-level configuration that can be customised without CLI arguments."""

    input_path: Path = Path("/Users/stefan/Desktop/Thesis/v3/Study/cb data pre-processing/uk_companies.csv")
    output_path: Path = Path("/Users/stefan/Desktop/Thesis/v3/Companies House Data/uk_companies_with_ch_data_test.csv")
    env_file: Optional[Path] = None
    api_key: Optional[str] = "" # <- get ur own key lol
    items_per_page: int = 20
    min_score: float = 80.0
    request_pause: float = 0.0        # set to 0 for speed; increase if you hit 429s
    timeout: float = 15.0
    verbosity: int = 1
    progress_every: int = 25

    # --- Download config ---
    download_enabled: bool = True
    download_dir: Path = Path("/Users/stefan/Desktop/Thesis/v3/Companies House Data/filings")
    download_types: Tuple[str, ...] = ("CS01", "SH01", "AR01", "IN01")
    max_workers: int = 8
    skip_existing: bool = True
    http_pool_connections: int = 64
    http_pool_maxsize: int = 64

    # Logs
    download_success_log: Path = Path("/Users/stefan/Desktop/Thesis/v3/Companies House Data/download_log_success.csv")
    download_failure_log: Path = Path("/Users/stefan/Desktop/Thesis/v3/Companies House Data/download_log_failure.csv")


# Update the values below to run the script without command-line arguments.
CONFIG = ScriptConfig()


def main(config: ScriptConfig = CONFIG) -> int:
    configure_logging(config.verbosity)
    print(f"Starting Companies House fetch with configuration: {asdict(config)}")

    env_path = Path(config.env_file).expanduser() if config.env_file else None
    env_status = load_environment(env_path)
    print(env_status)

    api_key = config.api_key or os.getenv("COMPANIES_HOUSE_API_KEY")
    if not api_key:
        logger.error("Companies House API key not provided. Set COMPANIES_HOUSE_API_KEY or populate CONFIG.api_key.")
        print("❌ Companies House API key missing. Aborting run.")
        return 1

    input_path = Path(config.input_path).expanduser()
    output_path = Path(config.output_path).expanduser()

    if not input_path.exists():
        logger.error("Input file %s does not exist.", input_path)
        print(f"❌ Input file not found: {input_path}")
        return 1

    client = CompaniesHouseClient(
        api_key,
        request_pause=config.request_pause,
        timeout=config.timeout,
        pool_connections=config.http_pool_connections,
        pool_maxsize=config.http_pool_maxsize,
    )

    input_rows = list(iter_rows(input_path))
    total_rows = len(input_rows)
    print(f"Loaded {total_rows} organisations from {input_path}")
    if not input_rows:
        logger.warning("Input file %s is empty.", input_path)
        print("Warning: input file is empty.")

    fieldnames: List[str] = []
    if input_rows:
        fieldnames = list(input_rows[0].keys())
    else:
        with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []

    additional_fields = [
        "ch_company_number",
        "ch_company_name",
        "ch_match_score",
        "ch_match_source",
        "ch_company_status",
        "ch_date_of_creation",
        "ch_date_of_cessation",
        "ch_match_reason",
        "ch_search_url",
        "ch_search_query",
        # Links
        "ch_link_filing_history",
        "ch_link_officers",
        # Filing history stamps
        "ch_last_sh01_date",
        "ch_last_cs01_date",
        "ch_last_nm01_date",
        # Download outcomes
        "ch_cs01_downloaded",
        "ch_sh01_downloaded",
        "ch_ar01_downloaded",
        "ch_in01_downloaded",
        "ch_download_errors_count",
    ]

    for extra in additional_fields:
        if extra not in fieldnames:
            fieldnames.append(extra)

    output_rows: List[Dict[str, Any]] = []
    matched_count = 0

    # Logs
    success_log_rows: List[Dict[str, Any]] = []
    failure_log_rows: List[Dict[str, Any]] = []

    # One global executor for download concurrency
    executor = ThreadPoolExecutor(max_workers=max(1, config.max_workers))

    try:
        for idx, row in enumerate(input_rows, start=1):
            org_name = row.get("org_name", "")
            legal_name = row.get("legal_name", "") if isinstance(row, dict) else ""
            should_report = (
                config.progress_every <= 1
                or idx == 1
                or idx == total_rows
                or (config.progress_every and idx % config.progress_every == 0)
            )
            if should_report:
                print(f"[{idx}/{total_rows}] Processing organisation: '{org_name}'")

            outcome = match_company(
                client,
                org_name,
                legal_name,
                items_per_page=config.items_per_page,
                min_score=config.min_score,
            )

            enriched_row = dict(row)
            enriched_row["ch_match_reason"] = outcome.reason
            enriched_row["ch_search_url"] = outcome.search_url
            enriched_row["ch_search_query"] = outcome.search_query or ""

            if outcome.matched:
                matched_count += 1
                if should_report:
                    score_display = f"{outcome.match_score:.1f}" if outcome.match_score is not None else "n/a"
                    print(
                        f"  ✔ Match: {outcome.company_number} | {outcome.matched_name} "
                        f"(score={score_display}, query='{outcome.search_query or 'n/a'}')"
                    )
                enriched_row["ch_company_number"] = outcome.company_number
                enriched_row["ch_company_name"] = outcome.matched_name
                enriched_row["ch_match_score"] = f"{outcome.match_score:.1f}" if outcome.match_score is not None else ""
                enriched_row["ch_match_source"] = outcome.match_source
                enriched_row["ch_company_status"] = outcome.company_status
                enriched_row["ch_date_of_creation"] = outcome.date_of_creation
                enriched_row["ch_date_of_cessation"] = outcome.date_of_cessation

                # Links
                enriched_row["ch_link_filing_history"] = outcome.link_filing_history or ""
                enriched_row["ch_link_officers"] = outcome.link_officers or ""

                # Filing history stamps
                enriched_row["ch_last_sh01_date"] = outcome.last_sh01_date or ""
                enriched_row["ch_last_cs01_date"] = outcome.last_cs01_date or ""
                enriched_row["ch_last_nm01_date"] = outcome.last_nm01_date or ""

                # --- Download all target docs
                dl_summary: Dict[str, Any] = {
                    "found_total": 0,
                    "downloaded_total": 0,
                    "errors": [],
                }
                if config.download_enabled and outcome.company_number:
                    dl_summary = download_company_documents(
                        client=client,
                        company_number=outcome.company_number,
                        company_name=outcome.matched_name or outcome.company_number,
                        base_dir=Path(config.download_dir).expanduser(),
                        allowed_types=config.download_types,
                        executor=executor,
                        skip_existing=config.skip_existing,
                    )

                # populate per-code counts safely
                enriched_row["ch_cs01_downloaded"] = dl_summary.get("downloaded_cs01", 0)
                enriched_row["ch_sh01_downloaded"] = dl_summary.get("downloaded_sh01", 0)
                enriched_row["ch_ar01_downloaded"] = dl_summary.get("downloaded_ar01", 0)
                enriched_row["ch_in01_downloaded"] = dl_summary.get("downloaded_in01", 0)
                enriched_row["ch_download_errors_count"] = len(dl_summary.get("errors", []))

                # ---- success/failure logging per company
                if dl_summary.get("found_total", 0) == 0:
                    failure_log_rows.append({
                        "company_number": outcome.company_number,
                        "company_name": outcome.matched_name or "",
                        "status": "no_target_filings_found",
                        "errors": ";".join(dl_summary.get("errors", [])),
                    })
                elif len(dl_summary.get("errors", [])) == 0:
                    success_log_rows.append({
                        "company_number": outcome.company_number,
                        "company_name": outcome.matched_name or "",
                        "downloaded_cs01": dl_summary.get("downloaded_cs01", 0),
                        "downloaded_sh01": dl_summary.get("downloaded_sh01", 0),
                        "downloaded_ar01": dl_summary.get("downloaded_ar01", 0),
                        "downloaded_in01": dl_summary.get("downloaded_in01", 0),
                        "skipped_existing": dl_summary.get("skipped_existing", 0),
                    })
                else:
                    failure_log_rows.append({
                        "company_number": outcome.company_number,
                        "company_name": outcome.matched_name or "",
                        "status": "partial_or_failed_downloads",
                        "found_cs01": dl_summary.get("found_cs01", 0),
                        "found_sh01": dl_summary.get("found_sh01", 0),
                        "found_ar01": dl_summary.get("found_ar01", 0),
                        "found_in01": dl_summary.get("found_in01", 0),
                        "downloaded_cs01": dl_summary.get("downloaded_cs01", 0),
                        "downloaded_sh01": dl_summary.get("downloaded_sh01", 0),
                        "downloaded_ar01": dl_summary.get("downloaded_ar01", 0),
                        "downloaded_in01": dl_summary.get("downloaded_in01", 0),
                        "errors": ";".join(dl_summary.get("errors", [])),
                    })

            else:
                if should_report:
                    score_display = f"{outcome.match_score:.1f}" if outcome.match_score is not None else "n/a"
                    print(
                        f"  ✖ No acceptable match (reason: {outcome.reason}, score={score_display}, "
                        f"query='{outcome.search_query or 'n/a'}')"
                    )
                enriched_row["ch_company_number"] = ""
                enriched_row["ch_company_name"] = outcome.matched_name or ""
                enriched_row["ch_match_score"] = f"{outcome.match_score:.1f}" if outcome.match_score else ""
                enriched_row["ch_match_source"] = outcome.match_source or ""
                enriched_row["ch_company_status"] = ""
                enriched_row["ch_date_of_creation"] = ""
                enriched_row["ch_date_of_cessation"] = ""
                enriched_row["ch_link_filing_history"] = ""
                enriched_row["ch_link_officers"] = ""
                enriched_row["ch_last_sh01_date"] = ""
                enriched_row["ch_last_cs01_date"] = ""
                enriched_row["ch_last_nm01_date"] = ""
                # log unmatched as a failure
                failure_log_rows.append({
                    "company_number": "",
                    "company_name": outcome.matched_name or (row.get("org_name") or ""),
                    "status": f"unmatched:{outcome.reason}",
                    "errors": "",
                })

            output_rows.append(enriched_row)
    finally:
        executor.shutdown(wait=True)

    write_rows(output_path, fieldnames, output_rows)

    # Write logs
    if config.download_enabled:
        # ensure consistent headers
        if success_log_rows:
            success_fields = [
                "company_number", "company_name",
                "downloaded_cs01", "downloaded_sh01", "downloaded_ar01", "downloaded_in01",
                "skipped_existing"
            ]
            write_rows(Path(config.download_success_log).expanduser(), success_fields, success_log_rows)
        if failure_log_rows:
            # fields are superset; writer ignores extras
            failure_fields = [
                "company_number", "company_name", "status",
                "found_cs01", "found_sh01", "found_ar01", "found_in01",
                "downloaded_cs01", "downloaded_sh01", "downloaded_ar01", "downloaded_in01",
                "errors"
            ]
            write_rows(Path(config.download_failure_log).expanduser(), failure_fields, failure_log_rows)

    match_rate = (matched_count / len(output_rows) * 100) if output_rows else 0.0
    logger.info(
        "Processed %d organisations. Matches found for %d entries (%.2f%%).",
        len(output_rows),
        matched_count,
        match_rate,
    )
    logger.info("Output written to %s", output_path)
    print(
        f"Finished run: processed {len(output_rows)} organisations, "
        f"{matched_count} matched ({match_rate:.2f}%)."
    )
    print(f"Results saved to {output_path}")

    if config.download_enabled:
        print(f"Download success log: {Path(config.download_success_log).expanduser()}")
        print(f"Download failure log: {Path(config.download_failure_log).expanduser()}")
        print(f"Filings saved under: {Path(config.download_dir).expanduser()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
