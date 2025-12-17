#!/usr/bin/env python3
"""
Fetch Companies House founding and closure dates for organisations listed in a CSV file.

NEEDS AN CH API KEY IN LINE 949 TO RUN!!!

All runtime configuration lives inside this module via the `CONFIG` object so the script
can be executed directly without providing command-line arguments. Adjust `CONFIG` to
point to the desired CSV paths, API key, thresholds, etc., and then simply run:

    python fetch_companies_house_dates.py

The script reads organisations from the configured CSV (expects an `org_name` column and
optionally `legal_name`), searches the Companies House API using:

- name matching strongly prioritising legal_name exact matches (with LTD/LIMITED variants),
- founding date proximity as a scoring bonus (not a hard constraint),
- founder/officer name matching (heavy bonus when Crunchbase founders match Companies House officers),

and writes the best matches together with their `date_of_creation` and `date_of_cessation`
fields to an output CSV.

In addition, all candidate Companies House matches for each organisation are written to a
separate CSV so that matching decisions can be inspected later.

Rows where the Companies House *search* failed due to HTTP/request errors are written to a
separate CSV (stem + "_search_errors.csv") for convenient re-processing.

The Companies House API key is read from `CONFIG.api_key` when provided, otherwise from
the `COMPANIES_HOUSE_API_KEY` environment variable. A `.env` file one directory above the
repository root is loaded automatically when present (or you can point to a custom file
through `CONFIG.env_file`).
"""

from __future__ import annotations

import csv
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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

# Heuristic keys for Crunchbase-style data in the input CSV
CB_FOUNDED_DATE_KEYS = ("founded_on", "cb_founded_on", "founded_date")
CB_CITY_KEYS = ("city", "city_name", "headquarters_city")  # kept for metadata only (not used in scoring)
CB_HOMEPAGE_URL_KEYS = ("homepage_url", "cb_homepage_url", "website")
CB_FOUNDER_NAME_KEYS = ("founder_names", "founder_name", "founders", "cb_founders")


@dataclass
class CBContext:
    """Subset of Crunchbase-style fields we use for matching."""
    founded_on: Optional[str] = None
    city: Optional[str] = None  # not used for scoring anymore
    homepage_url: Optional[str] = None
    founder_names: List[str] = field(default_factory=list)


@dataclass
class CandidateMatch:
    """A single Companies House candidate for an organisation."""
    company_number: Optional[str]
    title: str
    matched_name: str
    match_source: str
    name_score: float
    date_score: float
    city_score: float  # always 0.0 now (city not used in scoring)
    officer_match_score: float
    total_score: float
    date_within_one_year: bool  # kept for diagnostics only
    search_query: str
    date_of_creation: Optional[str]
    date_of_cessation: Optional[str]
    company_status: Optional[str]
    address_locality: Optional[str]
    address_postal_code: Optional[str]
    address_region: Optional[str]
    address_country: Optional[str]
    officer_best_officer_name: Optional[str] = None
    officer_best_founder_name: Optional[str] = None
    officer_match: bool = False


@dataclass
class MatchOutcome:
    """Represents the outcome of trying to match an organisation to Companies House."""

    matched: bool
    reason: str
    company_number: Optional[str] = None
    matched_name: Optional[str] = None
    match_score: Optional[float] = None  # this is the name_score of the chosen candidate
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
    # City from Companies House (registered office locality)
    company_city: Optional[str] = None
    # Officer info for the chosen candidate
    officer_match_score: Optional[float] = None
    officer_best_officer_name: Optional[str] = None
    officer_best_founder_name: Optional[str] = None
    officer_match: Optional[bool] = None


class CompaniesHouseClient:
    """Minimal client for the Companies House REST API with retry and caching."""

    BASE_URL = "https://api.company-information.service.gov.uk"

    def __init__(self, api_key: str, request_pause: float = 0.0, timeout: float = 10.0) -> None:
        if not api_key:
            raise ValueError("Companies House API key must be provided.")

        self._api_key = api_key
        self._pause = max(0.0, request_pause)
        self._timeout = timeout
        self._session = self._build_session()
        self._search_cache: Dict[Tuple[str, int], Dict[str, Any]] = {}
        self._profile_cache: Dict[str, Dict[str, Any]] = {}
        self._officers_cache: Dict[str, Dict[str, Any]] = {}

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        retry_strategy = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=False,  # retry on all methods
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
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

    def get_officers(self, company_number: str, items_per_page: int = 50) -> Dict[str, Any]:
        if company_number in self._officers_cache:
            return self._officers_cache[company_number]
        params = {"items_per_page": items_per_page, "start_index": 0}
        data = self._get(f"/company/{company_number}/officers", params=params)
        self._officers_cache[company_number] = data
        return data

    def get_filing_history(self, company_number: str, items_per_page: int = 100) -> Dict[str, Any]:
        params = {"items_per_page": items_per_page, "start_index": 0}
        return self._get(f"/company/{company_number}/filing-history", params=params)

    def get_exemptions(self, company_number: str) -> Dict[str, Any]:
        return self._get(f"/company/{company_number}/exemptions")


def normalise_company_name(name: str) -> str:
    """
    Normalise a company name for comparison:
    - uppercase
    - replace & with AND
    - remove punctuation
    - strip common corporate suffixes (LTD, LIMITED, PLC, etc.)
    """
    uppercase = name.upper()
    for symbol, replacement in AMPERSAND_EQUIVALENTS.items():
        uppercase = uppercase.replace(symbol, f" {replacement} ")
    cleaned = re.sub(r"[^\w\s]", " ", uppercase)
    tokens = [token for token in cleaned.split() if token]
    stripped = strip_corporate_suffix(" ".join(tokens))
    return stripped


def generate_query_variants(org_name: str, legal_name: Optional[str]) -> List[str]:
    """
    Generate query strings for Companies House search.
    (This is only for pulling candidates; actual scoring uses legal_name / org_name
    as described in compute_name_score.)
    """
    variants: List[str] = []
    seen: set[str] = set()

    def add_variant(name: str) -> None:
        candidate = " ".join(name.strip().split())
        if not candidate:
            return
        key = candidate.lower()
        if key not in seen:
            seen.add(key)
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


def extract_cb_context(row: Dict[str, Any]) -> CBContext:
    """Extract Crunchbase-style context (founded_on, city, homepage, founders) from a CSV row."""

    def first(keys: Sequence[str]) -> Optional[str]:
        for k in keys:
            v = (row.get(k) or "").strip()
            if v:
                return v
        return None

    founded_on = first(CB_FOUNDED_DATE_KEYS)
    city = first(CB_CITY_KEYS)
    homepage_url = first(CB_HOMEPAGE_URL_KEYS)
    founder_raw = first(CB_FOUNDER_NAME_KEYS)

    founder_names: List[str] = []
    if founder_raw:
        # Founders are in the form "founder1 | founder2 | ..."
        for token in re.split(r"[|;,/]", founder_raw):
            token = token.strip()
            if token:
                founder_names.append(token)

    return CBContext(
        founded_on=founded_on,
        city=city,
        homepage_url=homepage_url,
        founder_names=founder_names,
    )


def _parse_year(date_str: Optional[str]) -> Optional[int]:
    """Parse a year from a 'YYYY' or 'YYYY-MM-DD' style string."""
    if not date_str:
        return None
    m = re.match(r"(\d{4})", date_str)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def compute_date_score(cb_founded_on: Optional[str], ch_date_of_creation: Optional[str]) -> float:
    """
    Return a date score (0–20) based on how close the founding/incorporation years are.
    This is now a **soft scoring component**, not a hard constraint.
    """
    cb_year = _parse_year(cb_founded_on)
    ch_year = _parse_year(ch_date_of_creation)
    if cb_year is None or ch_year is None:
        return 0.0
    diff = abs(cb_year - ch_year)
    if diff == 0:
        return 20.0
    if diff == 1:
        return 15.0
    if diff == 2:
        return 10.0
    if diff <= 5:
        return 5.0
    return 0.0


def within_one_year(cb_founded_on: Optional[str], ch_date_of_creation: Optional[str]) -> bool:
    """
    Diagnostic helper: True iff both dates are present and the year difference is <= 1 (absolute).
    This is NOT used as a hard filter anymore, just written to the candidates CSV.
    """
    cb_year = _parse_year(cb_founded_on)
    ch_year = _parse_year(ch_date_of_creation)
    if cb_year is None or ch_year is None:
        return False
    return abs(cb_year - ch_year) <= 1


def _normalise_person_name(s: str) -> str:
    """Normalise a person name for fuzzy comparison."""
    s = s.upper()
    s = re.sub(r"[^A-Z\s]", " ", s)
    return " ".join(s.split())


def compute_officer_match_score(
    founder_names: Sequence[str],
    officer_names: Sequence[str],
) -> Tuple[float, Optional[str], Optional[str]]:
    """
    Compute a heavy bonus score based on overlap between founders (CB) and officers (CH).

    Returns (score, best_officer_name, best_founder_name).
    """
    if not founder_names or not officer_names:
        return 0.0, None, None

    best_sim = 0.0
    best_officer = None
    best_founder = None

    for founder in founder_names:
        nf = _normalise_person_name(founder)
        if not nf:
            continue
        for officer in officer_names:
            no = _normalise_person_name(officer)
            if not no:
                continue
            sim = fuzz.token_set_ratio(nf, no)
            if sim > best_sim:
                best_sim = sim
                best_officer = officer
                best_founder = founder

    if best_sim >= 95:
        score = 60.0
    elif best_sim >= 90:
        score = 40.0
    elif best_sim >= 80:
        score = 20.0
    elif best_sim >= 70:
        score = 10.0
    else:
        score = 0.0

    return score, best_officer, best_founder


def _legal_name_variants(name: str) -> List[str]:
    """
    Generate simple variants of a legal/org name for exact matching:
    - as-is
    - LTD <-> LIMITED swaps
    """
    variants: set[str] = set()
    base = name.strip()
    if not base:
        return []

    variants.add(base)

    # Swaps of LTD and LIMITED (case-insensitive)
    patterns = [
        (r"\bLTD\b\.?", "LIMITED"),
        (r"\bLIMITED\b\.?", "LTD"),
    ]
    for pattern, repl in patterns:
        new = re.sub(pattern, repl, base, flags=re.IGNORECASE)
        variants.add(new)

    return list(variants)


def compute_name_score(
    legal_name: Optional[str],
    org_name: Optional[str],
    candidate_variants: Sequence[Tuple[str, str]],
) -> Tuple[float, str, str]:
    """
    Compute the name_score and decide which variant matched best.

    Rules:
    1) If legal_name exists, first try **exact (normalised) matches** against legal_name variants.
    2) If no legal exact match, try **exact matches using org_name** variants.
    3) If still nothing, use fuzzy matching:
        - fuzzy(legal_name, candidate) if legal_name available
        - then fuzzy(org_name, candidate) as backup
    Returns (score, best_match_name, match_source).
    """
    if not candidate_variants:
        return 0.0, "", "no_candidate_variants"

    # 1) Exact match using legal_name variants
    if legal_name:
        legal_vars = _legal_name_variants(legal_name)
        legal_norms = {normalise_company_name(v): v for v in legal_vars if v.strip()}
        for cand_str, source in candidate_variants:
            cand_norm = normalise_company_name(cand_str)
            if cand_norm in legal_norms:
                return 100.0, cand_str, "legal_exact"

    # 2) Exact match using org_name variants (only if no legal exact match)
    if org_name:
        org_vars = _legal_name_variants(org_name)
        org_norms = {normalise_company_name(v): v for v in org_vars if v.strip()}
        for cand_str, source in candidate_variants:
            cand_norm = normalise_company_name(cand_str)
            if cand_norm in org_norms:
                return 98.0, cand_str, "org_exact"

    # 3) Fuzzy matching as fallback
    best_score = -1.0
    best_match_name = candidate_variants[0][0]
    best_source = "fuzzy_unknown"

    # Prefer fuzzy legal_name if available
    if legal_name:
        for cand_str, source in candidate_variants:
            score = fuzz.token_set_ratio(legal_name, cand_str)
            if score > best_score:
                best_score = score
                best_match_name = cand_str
                best_source = "legal_fuzzy"

    # Then allow fuzzy org_name if it improves the score
    if org_name:
        for cand_str, source in candidate_variants:
            score = fuzz.token_set_ratio(org_name, cand_str)
            if score > best_score:
                best_score = score
                best_match_name = cand_str
                best_source = "org_fuzzy"

    if best_score < 0:
        return 0.0, best_match_name, best_source
    return float(best_score), best_match_name, best_source


def score_candidates_for_single_query(
    query: str,
    items: Sequence[Dict[str, Any]],
    cb_context: Optional[CBContext],
    org_name: Optional[str],
    legal_name: Optional[str],
) -> List[CandidateMatch]:
    """
    Score Companies House search results for a single query.

    Scoring components:
    - name_score: from compute_name_score (legal_name first, then org_name, exact > fuzzy).
    - date_score: soft bonus based on year diff between CB founded_on and CH date_of_creation.
    - officer_match_score: heavy bonus if CB founder names match CH officers.
      (Officer scores are added later, once we fetch officers per company.)

    total_score (initially) = name_score + date_score.
    City is **not used** as a scoring feature (city_score is always 0.0).
    """
    scored: List[CandidateMatch] = []

    for item in items:
        candidate_name = item.get("title", "") or ""
        candidate_variants: List[Tuple[str, str]] = [(candidate_name, "title")]

        # If previous names ever appear in search results, they would be handled here.
        for previous in item.get("previous_company_names", []) or []:
            previous_name = previous.get("name") or ""
            if previous_name:
                candidate_variants.append((previous_name, "previous_company_name"))

        name_score, best_match_name, match_source = compute_name_score(
            legal_name=legal_name,
            org_name=org_name,
            candidate_variants=candidate_variants,
        )

        ch_date_creation = item.get("date_of_creation")
        address = item.get("address") or {}
        locality = address.get("locality")
        postal_code = address.get("postal_code")
        region = address.get("region")
        country = address.get("country")

        if cb_context:
            date_score = compute_date_score(cb_context.founded_on, ch_date_creation)
            date_ok = within_one_year(cb_context.founded_on, ch_date_creation)
        else:
            date_score = 0.0
            date_ok = False

        city_score = 0.0  # city is not used for matching anymore
        officer_match_score = 0.0  # filled in later after officer fetch

        total_score = float(name_score) + date_score

        scored.append(
            CandidateMatch(
                company_number=item.get("company_number"),
                title=candidate_name,
                matched_name=best_match_name,
                match_source=match_source,
                name_score=float(name_score),
                date_score=date_score,
                city_score=city_score,
                officer_match_score=officer_match_score,
                total_score=total_score,
                date_within_one_year=date_ok,
                search_query=query,
                date_of_creation=ch_date_creation,
                date_of_cessation=item.get("date_of_cessation"),
                company_status=item.get("company_status"),
                address_locality=locality,
                address_postal_code=postal_code,
                address_region=region,
                address_country=country,
            )
        )

    # Officer scores are added later; here we just sort by current total (name + date).
    scored.sort(key=lambda c: c.total_score, reverse=True)
    return scored


def make_search_url(query: str) -> str:
    return f"https://find-and-update.company-information.service.gov.uk/search?q={query.replace(' ', '+')}"


def enrich_candidates_with_officer_scores(
    client: CompaniesHouseClient,
    candidates: List[CandidateMatch],
    cb_context: Optional[CBContext],
) -> None:
    """
    For each candidate (with a company_number), fetch officers from CH and compute
    an officer_match_score against the Crunchbase founders.

    officer_match_score is then added to each candidate.total_score.

    NOTE: This will issue one API call per unique company_number (cached), and can
    be slow / rate-limited on large datasets.
    """
    if not cb_context or not cb_context.founder_names:
        return

    founders = cb_context.founder_names

    for cand in candidates:
        if not cand.company_number:
            continue
        try:
            data = client.get_officers(cand.company_number)
        except requests.RequestException as exc:
            logger.warning("Officers fetch failed for %s: %s", cand.company_number, exc)
            continue

        officer_items = data.get("items") or []
        officer_names = [item.get("name") for item in officer_items if item.get("name")]

        score, best_officer, best_founder = compute_officer_match_score(founders, officer_names)
        cand.officer_match_score = score
        cand.officer_best_officer_name = best_officer
        cand.officer_best_founder_name = best_founder
        cand.officer_match = score > 0.0
        cand.total_score += score  # heavy bonus if founders/officers line up


def match_company(
    client: CompaniesHouseClient,
    org_name: str,
    legal_name: Optional[str],
    cb_context: Optional[CBContext],
    *,
    items_per_page: int,
    min_score: float,
) -> Tuple[MatchOutcome, List[CandidateMatch]]:
    """
    Search and pick the best Companies House match for the provided organisation name.

    Matching rules:
    - Aggregate candidates from all query variants.
    - For each candidate, compute:
        * name_score (legal_name exact > org_name exact > fuzzy),
        * date_score (soft bonus),
        * officer_match_score (heavy bonus if founders match officers).
    - total_score = name_score + date_score + officer_match_score.
    - A candidate is only eligible as a final match if:
        * name_score >= min_score (90+), AND
        * officer_match_score > 0  (at least one founder–officer match).
    - Among eligible candidates, select the one with the highest total_score.

    If no candidate passes both thresholds, no match is returned.
    """
    query_variants = generate_query_variants(org_name, legal_name)
    if not query_variants:
        return MatchOutcome(matched=False, reason="missing_org_name"), []

    aggregated: Dict[str, CandidateMatch] = {}
    any_results = False

    for query in query_variants:
        try:
            search_results = client.search_companies(query, items_per_page=items_per_page)
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response else "unknown"
            logger.warning("HTTP error while searching for '%s': %s", query, exc)
            outcome = MatchOutcome(
                matched=False,
                reason=f"search_http_{status_code}",
                search_url=make_search_url(query),
                search_query=query,
            )
            return outcome, []
        except requests.RequestException as exc:
            logger.warning("Request error while searching for '%s': %s", query, exc)
            outcome = MatchOutcome(
                matched=False,
                reason="search_request_error",
                search_url=make_search_url(query),
                search_query=query,
            )
            return outcome, []

        items: Sequence[Dict[str, Any]] = search_results.get("items") or []
        if not items:
            continue
        any_results = True

        scored_candidates = score_candidates_for_single_query(
            query=query,
            items=items,
            cb_context=cb_context,
            org_name=org_name,
            legal_name=legal_name,
        )

        for cand in scored_candidates:
            key = cand.company_number or f"{cand.title}|{cand.date_of_creation or ''}"
            existing = aggregated.get(key)
            if existing is None:
                aggregated[key] = cand
            else:
                # Keep the candidate with higher name_score, then higher total_score.
                if (
                    cand.name_score > existing.name_score
                    or (cand.name_score == existing.name_score and cand.total_score > existing.total_score)
                ):
                    aggregated[key] = cand

    candidates = list(aggregated.values())

    if not any_results or not candidates:
        first_query = query_variants[0]
        outcome = MatchOutcome(
            matched=False,
            reason="no_results",
            search_url=make_search_url(first_query),
            search_query=first_query,
        )
        return outcome, candidates

    # Enrich with officer-based scores (heavy bonus when founders match officers)
    enrich_candidates_with_officer_scores(client, candidates, cb_context)

    # Final ranking based on total_score (name + date + officers)
    candidates.sort(key=lambda c: c.total_score, reverse=True)

    # Hard constraints for potential matches:
    #   - name_score >= min_score (90+)
    #   - officer_match_score > 0
    any_name_ok = any(c.name_score >= min_score for c in candidates)
    eligible = [
        c for c in candidates
        if c.name_score >= min_score and c.officer_match_score > 0.0
    ]

    if eligible:
        best_candidate = max(eligible, key=lambda c: c.total_score)
    else:
        best_candidate = None

    top_candidate = candidates[0]
    search_url = make_search_url(top_candidate.search_query)

    if best_candidate is None:
        if any_name_ok:
            reason = "no_officer_match_above_zero"
        else:
            reason = "below_score_threshold"

        outcome = MatchOutcome(
            matched=False,
            reason=reason,
            company_number=top_candidate.company_number,
            matched_name=top_candidate.matched_name,
            match_score=top_candidate.name_score,
            match_source=top_candidate.match_source,
            company_status=top_candidate.company_status,
            date_of_creation=top_candidate.date_of_creation,
            date_of_cessation=top_candidate.date_of_cessation,
            search_url=search_url,
            search_query=top_candidate.search_query,
            company_city=top_candidate.address_locality,
        )
        return outcome, candidates

    company_number = best_candidate.company_number
    if not company_number:
        outcome = MatchOutcome(
            matched=False,
            reason="missing_company_number",
            matched_name=best_candidate.matched_name,
            match_score=best_candidate.name_score,
            match_source=best_candidate.match_source,
            search_url=search_url,
            search_query=best_candidate.search_query,
            company_city=best_candidate.address_locality,
            officer_match_score=best_candidate.officer_match_score,
            officer_best_officer_name=best_candidate.officer_best_officer_name,
            officer_best_founder_name=best_candidate.officer_best_founder_name,
            officer_match=best_candidate.officer_match,
        )
        return outcome, candidates

    try:
        profile = client.get_company_profile(company_number)
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response else "unknown"
        logger.warning("HTTP error while retrieving profile %s: %s", company_number, exc)
        outcome = MatchOutcome(
            matched=False,
            reason=f"profile_http_{status_code}",
            company_number=company_number,
            matched_name=best_candidate.matched_name,
            match_score=best_candidate.name_score,
            match_source=best_candidate.match_source,
            search_url=search_url,
            search_query=best_candidate.search_query,
            company_city=best_candidate.address_locality,
            officer_match_score=best_candidate.officer_match_score,
            officer_best_officer_name=best_candidate.officer_best_officer_name,
            officer_best_founder_name=best_candidate.officer_best_founder_name,
            officer_match=best_candidate.officer_match,
        )
        return outcome, candidates
    except requests.RequestException as exc:
        logger.warning("Request error while retrieving profile %s: %s", company_number, exc)
        outcome = MatchOutcome(
            matched=False,
            reason="profile_request_error",
            company_number=company_number,
            matched_name=best_candidate.matched_name,
            match_score=best_candidate.name_score,
            match_source=best_candidate.match_source,
            search_url=search_url,
            search_query=best_candidate.search_query,
            company_city=best_candidate.address_locality,
            officer_match_score=best_candidate.officer_match_score,
            officer_best_officer_name=best_candidate.officer_best_officer_name,
            officer_best_founder_name=best_candidate.officer_best_founder_name,
            officer_match=best_candidate.officer_match,
        )
        return outcome, candidates

    # City from registered office address; fall back to search-locality if missing
    registered_address = profile.get("registered_office_address") or {}
    profile_city = registered_address.get("locality")
    company_city = profile_city or best_candidate.address_locality

    # ---- Links from profile.links
    links = profile.get("links", {}) or {}
    link_filing_history = links.get("filing_history")
    link_officers = links.get("officers")
    link_psc = links.get("persons_with_significant_control")
    link_charges = links.get("charges")
    link_insolvency = links.get("insolvency")

    return MatchOutcome(
        matched=True,
        reason="matched",
        company_number=company_number,
        matched_name=best_candidate.matched_name,
        match_score=best_candidate.name_score,
        match_source=best_candidate.match_source,
        company_status=profile.get("company_status"),
        date_of_creation=profile.get("date_of_creation"),
        date_of_cessation=profile.get("date_of_cessation"),
        search_url=search_url,
        search_query=best_candidate.search_query,
        # links
        link_filing_history=link_filing_history,
        link_officers=link_officers,
        link_psc=link_psc,
        link_charges=link_charges,
        link_insolvency=link_insolvency,
        # city
        company_city=company_city,
        # officer info for chosen candidate
        officer_match_score=best_candidate.officer_match_score,
        officer_best_officer_name=best_candidate.officer_best_officer_name,
        officer_best_founder_name=best_candidate.officer_best_founder_name,
        officer_match=best_candidate.officer_match,
    ), candidates


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


@dataclass
class ScriptConfig:
    """Holds script-level configuration that can be customised without CLI arguments."""

    input_path: Path = Path(
        "/Users/stefan/Desktop/Thesis/v4/Study/constructing features/uk_cb_only_features_apples.csv"
    )
    output_path: Path = Path(
        "/Users/stefan/Desktop/Thesis/v4/Companies House Data/uk_comps_with_ch_data_final.csv"
    )
    # Where to dump all candidate matches; if None, derived from output_path
    candidates_output_path: Optional[Path] = None
    env_file: Optional[Path] = None
    api_key: Optional[str] = "" # I removed my API key here! get ur own haha
    items_per_page: int = 20
    min_score: float = 90.0   # tightened fuzzy name threshold
    request_pause: float = 0.5  # increased to reduce 429s
    timeout: float = 10.0
    verbosity: int = 1
    progress_every: int = 25


# Fields for the candidates CSV
CANDIDATE_FIELDNAMES = [
    "input_index",
    "org_name",
    "legal_name",
    "cb_founded_on",
    "cb_city",
    "cb_homepage_url",
    "candidate_rank",
    "candidate_company_number",
    "candidate_company_name",
    "candidate_company_status",
    "candidate_date_of_creation",
    "candidate_date_of_cessation",
    "candidate_address_locality",
    "candidate_address_postal_code",
    "candidate_address_region",
    "candidate_address_country",
    "candidate_match_score_name",
    "candidate_match_score_date",
    "candidate_match_score_city",
    "candidate_match_score_officer",
    "candidate_match_score_total",
    "candidate_date_within_one_year",
    "candidate_officer_match",
    "candidate_officer_best_officer_name",
    "candidate_officer_best_founder_name",
    "candidate_search_query",
]

# Columns to drop from final output CSV (but still read from input if present)
# NOTE: 'founded_on' is intentionally *not* dropped so it is kept in the final CSV.
COLUMNS_TO_DROP_FROM_OUTPUT = {
    "ttf_percentile_binned",
    "_ttf_percentile_raw",
    "first_funding_raised_usd",
    "total_funding_usd",
    "num_funding_rounds",
    "first_funding_archetype",
    "first_funding_investor_type",
    "first_round_investor_uuid",
    "first_funding_leads",
    "employee_count",
    "founding_team_size",
    "founding_team_diversity",
    "founder_education",
    "founder_uni_reputation",
    "prior_founding_experience",
    "founders_descriptions",
    "first_funding_date",
    "ttf_days",
    "ttf_months",
    "_ttf_months_raw",
    "ttf_percentile",
    "success",
    "founding_cohort",
}


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

    if config.candidates_output_path is not None:
        candidates_output_path = Path(config.candidates_output_path).expanduser()
    else:
        candidates_output_path = output_path.with_name(output_path.stem + "_candidates.csv")

    # path for rows where search failed due to HTTP/request errors
    errors_output_path = output_path.with_name(output_path.stem + "_search_errors.csv")

    if not input_path.exists():
        logger.error("Input file %s does not exist.", input_path)
        print(f"❌ Input file not found: {input_path}")
        return 1

    client = CompaniesHouseClient(api_key, request_pause=config.request_pause, timeout=config.timeout)

    input_rows = list(iter_rows(input_path))
    total_rows = len(input_rows)
    print(f"Loaded {total_rows} organisations from {input_path}")
    if not input_rows:
        logger.warning("Input file %s is empty.", input_path)
        print("Warning: input file is empty.")

    # Determine base fieldnames from input
    if input_rows:
        fieldnames: List[str] = list(input_rows[0].keys())
    else:
        with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []

    # Remove unwanted CB/outcome columns from final output CSV
    fieldnames = [f for f in fieldnames if f not in COLUMNS_TO_DROP_FROM_OUTPUT]

    additional_fields = [
        "ch_company_number",
        "ch_company_name",
        "ch_match_score",
        "ch_match_source",
        "ch_company_status",
        "ch_date_of_creation",
        "ch_date_of_cessation",
        "ch_city",
        "ch_match_reason",
        "ch_search_url",
        "ch_search_query",
        # Links
        "ch_link_filing_history",
        "ch_link_officers",
        # Officer info
        "ch_officer_match",
        "ch_officer_match_score",
        "ch_officer_best_officer_name",
        "ch_officer_best_founder_name",
    ]

    for extra in additional_fields:
        if extra not in fieldnames:
            fieldnames.append(extra)

    output_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []  # rows where search failed due to HTTP/request errors
    matched_count = 0

    for idx, row in enumerate(input_rows, start=1):
        org_name = row.get("org_name", "")
        legal_name = row.get("legal_name", "") if isinstance(row, dict) else ""
        cb_context = extract_cb_context(row)

        should_report = (
            config.progress_every <= 1
            or idx == 1
            or idx == total_rows
            or (config.progress_every and idx % config.progress_every == 0)
        )
        if should_report:
            print(f"[{idx}/{total_rows}] Processing organisation: '{org_name}'")

        outcome, candidates = match_company(
            client,
            org_name,
            legal_name,
            cb_context,
            items_per_page=config.items_per_page,
            min_score=config.min_score,
        )

        # Record all candidates for inspection
        cb_founded_on = cb_context.founded_on if cb_context else None
        cb_city = cb_context.city if cb_context else None
        cb_homepage_url = cb_context.homepage_url if cb_context else None

        for rank, cand in enumerate(candidates, start=1):
            candidate_rows.append(
                {
                    "input_index": idx,
                    "org_name": org_name,
                    "legal_name": legal_name,
                    "cb_founded_on": cb_founded_on or "",
                    "cb_city": cb_city or "",
                    "cb_homepage_url": cb_homepage_url or "",
                    "candidate_rank": rank,
                    "candidate_company_number": cand.company_number or "",
                    "candidate_company_name": cand.title,
                    "candidate_company_status": cand.company_status or "",
                    "candidate_date_of_creation": cand.date_of_creation or "",
                    "candidate_date_of_cessation": cand.date_of_cessation or "",
                    "candidate_address_locality": cand.address_locality or "",
                    "candidate_address_postal_code": cand.address_postal_code or "",
                    "candidate_address_region": cand.address_region or "",
                    "candidate_address_country": cand.address_country or "",
                    "candidate_match_score_name": f"{cand.name_score:.1f}",
                    "candidate_match_score_date": f"{cand.date_score:.1f}",
                    "candidate_match_score_city": f"{cand.city_score:.1f}",
                    "candidate_match_score_officer": f"{cand.officer_match_score:.1f}",
                    "candidate_match_score_total": f"{cand.total_score:.1f}",
                    "candidate_date_within_one_year": "1" if cand.date_within_one_year else "0",
                    "candidate_officer_match": "1" if cand.officer_match else "0",
                    "candidate_officer_best_officer_name": cand.officer_best_officer_name or "",
                    "candidate_officer_best_founder_name": cand.officer_best_founder_name or "",
                    "candidate_search_query": cand.search_query,
                }
            )

        enriched_row = dict(row)
        enriched_row["ch_match_reason"] = outcome.reason
        enriched_row["ch_search_url"] = outcome.search_url
        enriched_row["ch_search_query"] = outcome.search_query or ""

        # Officer fields default
        officer_match_flag = "1" if outcome.officer_match else "0"
        officer_match_score_str = (
            f"{outcome.officer_match_score:.1f}" if outcome.officer_match_score is not None else ""
        )
        officer_best_officer_name = outcome.officer_best_officer_name or ""
        officer_best_founder_name = outcome.officer_best_founder_name or ""

        if outcome.matched:
            matched_count += 1
            if should_report:
                score_display = f"{outcome.match_score:.1f}" if outcome.match_score is not None else "n/a"
                print(
                    f"  ✔ Match: {outcome.company_number} | {outcome.matched_name} "
                    f"(name_score={score_display}, query='{outcome.search_query or 'n/a'}')"
                )
            enriched_row["ch_company_number"] = outcome.company_number
            enriched_row["ch_company_name"] = outcome.matched_name
            enriched_row["ch_match_score"] = (
                f"{outcome.match_score:.1f}" if outcome.match_score is not None else ""
            )
            enriched_row["ch_match_source"] = outcome.match_source
            enriched_row["ch_company_status"] = outcome.company_status
            enriched_row["ch_date_of_creation"] = outcome.date_of_creation
            enriched_row["ch_date_of_cessation"] = outcome.date_of_cessation
            enriched_row["ch_city"] = outcome.company_city or ""

            # Links
            enriched_row["ch_link_filing_history"] = outcome.link_filing_history or ""
            enriched_row["ch_link_officers"] = outcome.link_officers or ""
        else:
            if should_report:
                score_display = f"{outcome.match_score:.1f}" if outcome.match_score is not None else "n/a"
                print(
                    f"  ✖ No acceptable match (reason: {outcome.reason}, name_score={score_display}, "
                    f"query='{outcome.search_query or 'n/a'}')"
                )
            enriched_row["ch_company_number"] = ""
            enriched_row["ch_company_name"] = outcome.matched_name or ""
            enriched_row["ch_match_score"] = (
                f"{outcome.match_score:.1f}" if outcome.match_score is not None else ""
            )
            enriched_row["ch_match_source"] = outcome.match_source or ""
            enriched_row["ch_company_status"] = ""
            enriched_row["ch_date_of_creation"] = ""
            enriched_row["ch_date_of_cessation"] = ""
            enriched_row["ch_city"] = outcome.company_city or ""

            # Ensure link columns are present even when unmatched
            enriched_row["ch_link_filing_history"] = ""
            enriched_row["ch_link_officers"] = ""

        # Officer info columns in main CSV
        enriched_row["ch_officer_match"] = officer_match_flag
        enriched_row["ch_officer_match_score"] = officer_match_score_str
        enriched_row["ch_officer_best_officer_name"] = officer_best_officer_name
        enriched_row["ch_officer_best_founder_name"] = officer_best_founder_name

        # collect rows where the search itself failed due to HTTP/request errors
        if outcome.reason in ("search_request_error",) or (outcome.reason or "").startswith("search_http_"):
            error_rows.append(enriched_row)

        output_rows.append(enriched_row)

    write_rows(output_path, fieldnames, output_rows)

    if candidate_rows:
        write_rows(candidates_output_path, CANDIDATE_FIELDNAMES, candidate_rows)
        print(f"Candidate details saved to {candidates_output_path}")
    else:
        print("No candidate results to save.")

    if error_rows:
        write_rows(errors_output_path, fieldnames, error_rows)
        print(f"Search-error rows saved to {errors_output_path}")
    else:
        print("No search error rows to save.")

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
