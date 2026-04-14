from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Set

import requests
from dateutil import parser as dtparser
import numpy as np
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


LOGGER = logging.getLogger(__name__)
_EMBEDDER: Any = None
_EMBEDDER_NAME = ""

DOMAIN_SYNONYMS: Dict[str, List[str]] = {
    "ice nucleation in water": [
        "ice nucleation",
        "homogeneous ice nucleation",
        "heterogeneous ice nucleation",
        "water nucleation",
        "freezing nucleation",
    ],
    "supercooled water": [
        "deeply supercooled water",
        "metastable water",
        "undercooled water",
        "supercooling",
    ],
    "critical nucleus": [
        "critical cluster",
        "nucleus size",
        "critical embryo",
    ],
    "classical nucleation theory": [
        "cnt",
        "nucleation theory",
        "free energy barrier",
    ],
    "glass transition": [
        "glass transition temperature",
        "tg",
        "vitrification",
        "amorphous transition",
    ],
    "fragile strong thransition": [
        "fragile-to-strong transition",
        "fragile strong transition",
        "fst",
        "dynamic crossover",
    ],
    "fragile-to-strong transition": [
        "fragile strong transition",
        "fragile strong thransition",
        "fst",
        "dynamic crossover",
    ],
    "amorphous ice": [
        "low-density amorphous ice",
        "high-density amorphous ice",
        "very-high-density amorphous ice",
        "lda ice",
        "hda ice",
        "vhda ice",
    ],
    "glassy water": [
        "amorphous water",
        "vitreous water",
        "non-crystalline water",
    ],
    "amorphous ice / glassy water": [
        "amorphous ice",
        "glassy water",
        "amorphous water",
    ],
}

JOURNAL_ALIASES: Dict[str, str] = {
    "prl": "physical review letters",
    "physical review letters": "physical review letters",
    "prx": "physical review x",
    "physical review x": "physical review x",
    "prx life": "physical review x life",
    "physical review x life": "physical review x life",
    "pnas": "proceedings of the national academy of sciences",
    "proceedings of the national academy of sciences": "proceedings of the national academy of sciences",
    "jacs": "journal of the american chemical society",
    "journal of the american chemical society": "journal of the american chemical society",
    "jpcl": "journal of physical chemistry letters",
    "journal of physical chemistry letters": "journal of physical chemistry letters",
    "jpcb": "journal of physical chemistry b",
    "journal of physical chemistry b": "journal of physical chemistry b",
    "jcp": "journal of chemical physics",
    "journal of chemical physics": "journal of chemical physics",
}


@dataclass
class PaperRecord:
    source: str
    source_id: str
    title: str
    abstract_en: str
    doi: str = ""
    url: str = ""
    journal: str = ""
    issn: List[str] = field(default_factory=list)
    published_at: Optional[datetime] = None
    authors: List[str] = field(default_factory=list)
    semantic_score: float = 0.0


def create_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "LiquidPaperFetch/0.1"})
    return session


def parse_datetime(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    try:
        dt = dtparser.parse(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def extract_openalex_abstract(inv_idx: Dict[str, List[int]]) -> str:
    if not inv_idx:
        return ""
    max_pos = 0
    for positions in inv_idx.values():
        if positions:
            max_pos = max(max_pos, max(positions))
    tokens = [""] * (max_pos + 1)
    for token, positions in inv_idx.items():
        for p in positions:
            if 0 <= p < len(tokens):
                tokens[p] = token
    return " ".join(t for t in tokens if t).strip()


class BaseClient:
    def __init__(self, base_url: str, session: requests.Session):
        self.base_url = base_url.rstrip("/")
        self.session = session

    def _request(self, method: str, path: str, **kwargs: Any) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            resp = self.session.request(method=method, url=url, timeout=30, **kwargs)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", "3"))
                LOGGER.warning("Rate limited by %s, sleeping %ss", url, retry_after)
                time.sleep(retry_after)
                resp = self.session.request(method=method, url=url, timeout=30, **kwargs)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            LOGGER.exception("API request failed: %s", exc)
            return {}


class OpenAlexClient(BaseClient):
    def fetch(self, since_date: datetime, max_records: int, query_text: str = "", mailto: str = "") -> List[PaperRecord]:
        params = {
            "filter": f"from_publication_date:{since_date.date().isoformat()}",
            "per-page": max_records,
            "sort": "publication_date:desc",
            "select": "id,doi,display_name,abstract_inverted_index,publication_date,ids,primary_location,authorships,biblio",
        }
        if query_text.strip():
            params["search"] = query_text.strip()
        if mailto:
            params["mailto"] = mailto
        payload = self._request("GET", "/works", params=params)
        works = payload.get("results", [])
        records: List[PaperRecord] = []
        for work in works:
            location = work.get("primary_location") or {}
            source = location.get("source") or {}
            journal = source.get("display_name", "")
            issn = source.get("issn", []) or []
            doi = (work.get("doi") or "").replace("https://doi.org/", "").strip()
            records.append(
                PaperRecord(
                    source="openalex",
                    source_id=work.get("id", ""),
                    title=(work.get("display_name") or "").strip(),
                    abstract_en=extract_openalex_abstract(work.get("abstract_inverted_index", {})),
                    doi=doi,
                    url=(work.get("ids") or {}).get("openalex", ""),
                    journal=journal,
                    issn=issn,
                    published_at=parse_datetime(work.get("publication_date")),
                    authors=[a.get("author", {}).get("display_name", "") for a in work.get("authorships", [])],
                )
            )
        return records


class CrossrefClient(BaseClient):
    def _parse_items(self, items: List[Dict[str, Any]]) -> List[PaperRecord]:
        records: List[PaperRecord] = []
        for item in items:
            title = ""
            if item.get("title"):
                title = item["title"][0]
            abstract = re.sub(r"<[^>]+>", "", item.get("abstract", "") or "")
            published = item.get("published-online") or item.get("published-print") or {}
            date_parts = (published.get("date-parts") or [[None, None, None]])[0]
            y, m, d = (date_parts + [1, 1, 1])[:3]
            published_at = None
            if y:
                published_at = datetime(y, m or 1, d or 1, tzinfo=timezone.utc)
            records.append(
                PaperRecord(
                    source="crossref",
                    source_id=item.get("DOI", ""),
                    title=title.strip(),
                    abstract_en=abstract.strip(),
                    doi=item.get("DOI", "").strip(),
                    url=item.get("URL", ""),
                    journal=((item.get("container-title") or [""])[0]).strip(),
                    issn=item.get("ISSN") or [],
                    published_at=published_at,
                    authors=[" ".join(filter(None, [a.get("given"), a.get("family")])).strip() for a in item.get("author", [])],
                )
            )
        return records

    def fetch(
        self,
        since_date: datetime,
        max_records: int,
        query_text: str = "",
        journal_titles: Optional[List[str]] = None,
        mailto: str = "",
    ) -> List[PaperRecord]:
        filter_part = f"from-pub-date:{since_date.date().isoformat()},type:journal-article"
        base_params = {
            "filter": filter_part,
            "sort": "published",
            "order": "desc",
            "select": "DOI,title,abstract,published-online,published-print,container-title,author,ISSN,URL",
        }
        if query_text.strip():
            base_params["query.bibliographic"] = query_text.strip()
        if mailto:
            base_params["mailto"] = mailto

        records: List[PaperRecord] = []
        journals = [j for j in (journal_titles or []) if j.strip()]
        if journals:
            rows_per_journal = max(3, min(12, max_records // max(len(journals), 1)))
            for title in journals:
                params = dict(base_params)
                params["rows"] = rows_per_journal
                params["query.container-title"] = title
                payload = self._request("GET", "/works", params=params)
                items = (payload.get("message") or {}).get("items", [])
                records.extend(self._parse_items(items))
            return records

        params = dict(base_params)
        params["rows"] = max_records
        payload = self._request("GET", "/works", params=params)
        items = (payload.get("message") or {}).get("items", [])
        return self._parse_items(items)


class SemanticScholarClient(BaseClient):
    def __init__(self, base_url: str, session: requests.Session, api_key: str = ""):
        super().__init__(base_url, session)
        self.api_key = api_key.strip()

    def fetch(self, since_date: datetime, max_records: int) -> List[PaperRecord]:
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        params = {
            "query": "water crystallization ice supercooled liquid",
            "limit": max_records,
            "fields": "paperId,title,abstract,year,publicationDate,externalIds,url,journal,authors",
            "sort": "publicationDate:desc",
        }
        payload = self._request("GET", "/paper/search", params=params, headers=headers)
        data = payload.get("data", [])
        records: List[PaperRecord] = []
        for item in data:
            published_at = parse_datetime(item.get("publicationDate"))
            if published_at and published_at < since_date:
                continue
            ext_ids = item.get("externalIds") or {}
            doi = ext_ids.get("DOI", "")
            journal_obj = item.get("journal") or {}
            records.append(
                PaperRecord(
                    source="semanticscholar",
                    source_id=item.get("paperId", ""),
                    title=(item.get("title") or "").strip(),
                    abstract_en=(item.get("abstract") or "").strip(),
                    doi=doi,
                    url=item.get("url", ""),
                    journal=(journal_obj.get("name") or "").strip(),
                    published_at=published_at,
                    authors=[(a.get("name") or "").strip() for a in item.get("authors", [])],
                )
            )
        return records


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def in_window(pub: Optional[datetime], since_date: datetime, until_date: datetime) -> bool:
    return bool(pub and since_date <= pub <= until_date)


def journal_match(record: PaperRecord, issn_allow: Set[str], title_allow: Set[str]) -> bool:
    raw_title = _norm_text(record.journal)
    title = JOURNAL_ALIASES.get(raw_title, raw_title)
    if title and title in title_allow:
        return True
    # For multi-token titles, permit conservative containment matching.
    title_tokens = [t for t in title.split(" ") if t]
    if len(title_tokens) >= 2:
        for allowed in title_allow:
            allowed_norm = JOURNAL_ALIASES.get(allowed, allowed)
            allowed_tokens = [t for t in allowed_norm.split(" ") if t]
            if len(allowed_tokens) >= 2 and (allowed_norm in title or title in allowed_norm):
                return True
    for code in record.issn:
        if code.strip() in issn_allow:
            return True
    return False


def keyword_match(record: PaperRecord, keywords: Iterable[str]) -> bool:
    haystack = f"{record.title}\n{record.abstract_en}".lower()
    return any(k.lower() in haystack for k in keywords)


def expand_keywords(keywords: Iterable[str]) -> List[str]:
    expanded: Set[str] = set()
    for keyword in keywords:
        kw = (keyword or "").strip()
        if not kw:
            continue
        expanded.add(kw)
        expanded.add(kw.lower())
        expanded.add(kw.replace("-", " "))
        expanded.add(kw.replace("/", " "))

        norm = _norm_text(kw)
        for syn in DOMAIN_SYNONYMS.get(norm, []):
            expanded.add(syn)

        # Basic morphological variants for robustness.
        if "nucleation" in norm:
            expanded.add(norm.replace("nucleation", "nucleate"))
            expanded.add(norm.replace("nucleation", "nucleating"))
        if "crystallization" in norm:
            expanded.add(norm.replace("crystallization", "crystal growth"))
            expanded.add(norm.replace("crystallization", "crystallisation"))
        if "transition" in norm:
            expanded.add(norm.replace("transition", "transitions"))

    return sorted({k.strip() for k in expanded if k.strip()})


def build_source_query_text(keywords: Iterable[str]) -> str:
    expanded = expand_keywords(keywords)
    preferred = [
        "supercooled water",
        "ice nucleation",
        "crystallization",
        "critical nucleus",
        "classical nucleation theory",
        "glass transition",
        "amorphous ice",
        "glassy water",
    ]
    selected: List[str] = []
    pool = {p.lower(): p for p in preferred}
    for term in preferred:
        if any(_norm_text(x) == _norm_text(term) for x in expanded):
            selected.append(term)
    if not selected:
        selected = preferred[:5]
    return " ".join(selected)


def _get_embedder(model_name: str) -> Any:
    global _EMBEDDER, _EMBEDDER_NAME
    if SentenceTransformer is None:
        return None
    if _EMBEDDER is None or _EMBEDDER_NAME != model_name:
        _EMBEDDER = SentenceTransformer(model_name)
        _EMBEDDER_NAME = model_name
    return _EMBEDDER


def semantic_rerank(
    records: List[PaperRecord],
    keywords: Iterable[str],
    model_name: str,
    similarity_threshold: float,
    top_k: int,
    min_hits: int,
    query_expansion: bool,
) -> List[PaperRecord]:
    queries = expand_keywords(keywords) if query_expansion else [q.strip() for q in keywords if q and q.strip()]
    if not records or not queries:
        return []

    embedder = _get_embedder(model_name)
    if embedder is None:
        LOGGER.warning("sentence-transformers not installed, fallback to lexical keyword matching")
        return [r for r in records if keyword_match(r, queries)]

    try:
        docs = [f"{r.title}\n{r.abstract_en}".strip() for r in records]
        doc_vec = embedder.encode(docs, normalize_embeddings=True, show_progress_bar=False)
        q_vec = embedder.encode(queries, normalize_embeddings=True, show_progress_bar=False)

        doc_mat = np.asarray(doc_vec)
        qry_mat = np.asarray(q_vec)
        scores = doc_mat @ qry_mat.T
        max_scores = scores.max(axis=1)

        scored_all: List[PaperRecord] = []
        for idx, rec in enumerate(records):
            rec.semantic_score = float(max_scores[idx])
            scored_all.append(rec)

        scored_all.sort(
            key=lambda x: (x.semantic_score, x.published_at or datetime.min.replace(tzinfo=timezone.utc)),
            reverse=True,
        )

        scored = [rec for rec in scored_all if rec.semantic_score >= similarity_threshold]

        if len(scored) < min_hits:
            keep_n = max(min_hits, top_k if top_k > 0 else min_hits)
            keep_n = min(keep_n, len(scored_all))
            LOGGER.info(
                "Semantic hits (%s) below min_hits (%s), backfilling from top-ranked candidates",
                len(scored),
                min_hits,
            )
            scored = scored_all[:keep_n]

        if not scored and scored_all:
            # Last-resort guard to avoid empty output under extreme thresholds.
            scored = scored_all[: min(max(min_hits, 1), len(scored_all))]

        if top_k > 0:
            scored = scored[: max(top_k, min_hits)]

        return scored
    except Exception as exc:
        LOGGER.exception("Semantic rerank failed, fallback to lexical matching: %s", exc)
        return [r for r in records if keyword_match(r, queries)]


def apply_filters(
    records: Iterable[PaperRecord],
    since_date: datetime,
    journal_issn_allow: Iterable[str],
    journal_title_allow: Iterable[str],
    keywords: Iterable[str],
    semantic_cfg: Optional[Dict[str, Any]] = None,
) -> List[PaperRecord]:
    now = datetime.now(timezone.utc)
    issn_allow = {x.strip() for x in journal_issn_allow if x.strip()}
    title_allow = {JOURNAL_ALIASES.get(_norm_text(x), _norm_text(x)) for x in journal_title_allow if x.strip()}

    base_candidates: List[PaperRecord] = []
    for rec in records:
        if not rec.title:
            continue
        if not in_window(rec.published_at, since_date, now):
            continue
        if issn_allow or title_allow:
            if not journal_match(rec, issn_allow, title_allow):
                continue
        base_candidates.append(rec)

    semantic_cfg = semantic_cfg or {}
    semantic_enabled = bool(semantic_cfg.get("enabled", False))
    if semantic_enabled and keywords:
        return semantic_rerank(
            records=base_candidates,
            keywords=keywords,
            model_name=str(semantic_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")),
            similarity_threshold=float(semantic_cfg.get("similarity_threshold", 0.36)),
            top_k=int(semantic_cfg.get("top_k", 120)),
            min_hits=int(semantic_cfg.get("min_hits", 15)),
            query_expansion=bool(semantic_cfg.get("query_expansion", True)),
        )

    expanded = expand_keywords(keywords) if bool(semantic_cfg.get("query_expansion", True)) else list(keywords)
    return [rec for rec in base_candidates if keyword_match(rec, expanded)]


def deduplicate(records: Iterable[PaperRecord]) -> List[PaperRecord]:
    by_key: Dict[str, PaperRecord] = {}
    for rec in records:
        key = rec.doi.lower().strip() if rec.doi else _norm_text(rec.title)
        if not key:
            continue
        existing = by_key.get(key)
        if not existing:
            by_key[key] = rec
            continue
        if (not existing.abstract_en) and rec.abstract_en:
            by_key[key] = rec
            continue
        if (existing.published_at or datetime.min.replace(tzinfo=timezone.utc)) < (
            rec.published_at or datetime.min.replace(tzinfo=timezone.utc)
        ):
            by_key[key] = rec
    return sorted(by_key.values(), key=lambda x: x.published_at or datetime.min.replace(tzinfo=timezone.utc), reverse=True)


def fetch_all(config: Dict[str, Any]) -> List[PaperRecord]:
    now = datetime.now(timezone.utc)
    lookback_days = int(config["general"]["lookback_days"])
    since_date = now - timedelta(days=lookback_days)
    max_records = int(config["general"].get("max_records_per_source", 80))

    filters = config.get("filters", {})
    keywords = filters.get("keywords", [])
    semantic_cfg = filters.get("semantic_search", {})
    journal_issn = (filters.get("journals") or {}).get("issn", [])
    journal_titles = (filters.get("journals") or {}).get("titles", [])
    query_text = build_source_query_text(keywords)

    apis = config.get("apis", {})
    session = create_session()

    all_records: List[PaperRecord] = []

    if (apis.get("openalex") or {}).get("enabled", True):
        openalex_cfg = apis.get("openalex", {})
        client = OpenAlexClient(openalex_cfg.get("base_url", "https://api.openalex.org"), session)
        recs = client.fetch(since_date, max_records, query_text=query_text, mailto=openalex_cfg.get("mailto", ""))
        LOGGER.info("OpenAlex fetched: %s", len(recs))
        all_records.extend(recs)

    if (apis.get("crossref") or {}).get("enabled", True):
        crossref_cfg = apis.get("crossref", {})
        client = CrossrefClient(crossref_cfg.get("base_url", "https://api.crossref.org"), session)
        recs = client.fetch(
            since_date,
            max_records,
            query_text=query_text,
            journal_titles=journal_titles,
            mailto=crossref_cfg.get("mailto", ""),
        )
        LOGGER.info("Crossref fetched: %s", len(recs))
        all_records.extend(recs)

    if (apis.get("semanticscholar") or {}).get("enabled", True):
        s2_cfg = apis.get("semanticscholar", {})
        client = SemanticScholarClient(
            s2_cfg.get("base_url", "https://api.semanticscholar.org/graph/v1"),
            session,
            api_key=s2_cfg.get("api_key", ""),
        )
        recs = client.fetch(since_date, max_records)
        LOGGER.info("Semantic Scholar fetched: %s", len(recs))
        all_records.extend(recs)

    filtered = apply_filters(
        records=all_records,
        since_date=since_date,
        journal_issn_allow=journal_issn,
        journal_title_allow=journal_titles,
        keywords=keywords,
        semantic_cfg=semantic_cfg,
    )
    deduped = deduplicate(filtered)

    LOGGER.info("Records after filter+dedupe: %s", len(deduped))
    return deduped