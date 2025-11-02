#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wildberries multi-search → CSV (refactored + brand-based price filtering)

Additions in this update:
- Script-embedded filtering parameters (no external file):
  * key_brand (default: "Regner Coin")
  * price_diff_pct (default: 25.0)
- CLI overrides are supported: --key-brand, --price-diff-pct
- If multiple products of key_brand exist, the *minimum* price is used as the reference.
- Products whose price differs by more than price_diff_pct from the reference are removed.
- Original order from WB is strictly preserved during filtering.

Python 3.9+
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote_plus

import requests
from http.cookiejar import MozillaCookieJar
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import httpx  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    httpx = None  # noqa: N816


# ============================
# Constants & configuration
# ============================
BASE_URL = "https://u-search.wb.ru/exactmatch/ru/common/v18/search"

DEFAULT_PARAMS: Dict[str, str] = {
    "ab_testing": "false",
    "appType": "1",
    "autoselectFilters": "false",
    "curr": "rub",
    "dest": "123585480",
    "inheritFilters": "false",
    "lang": "ru",
    "resultset": "catalog",
    "spp": "30",
    "sort": "popular",
    "suppressSpellcheck": "false",
    "uclusters": "2",
}

UA_TPL = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.{minor}.0 Safari/537.36"
)

# Anti-preset strategy knobs
VARIANT_BUDGET = 10  # how many profiles to attempt per page
RETRY_JITTER = 1.0  # base sleep (sec) between suspicious responses


# ============================
# Exceptions
# ============================
class BadResponseVariant(Exception):
    """WB returned a "preset" structure with only flat prices (no sizes.price.product)."""


# ============================
# Utilities & filter defaults
# ============================
logger = logging.getLogger("wb_multi_search")


@dataclass
class FilterConfig:
    key_brand: str = "Regner Coin"
    price_diff_pct: float = 25.0  # percent, e.g. 25.0 → ±25%


def setup_logging(level: str = "INFO") -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def random_user_agent() -> str:
    return UA_TPL.format(minor=random.randint(0, 9999))


def make_headers(
    ua: Optional[str] = None,
    *,
    mimic_browser: bool = False,
    referer_query: Optional[str] = None,
) -> Dict[str, str]:
    """Base headers + optional browser-like extras."""
    h = {
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "ru-RU,ru;q=0.9",
        "User-Agent": ua or random_user_agent(),
        "X-Requested-With": "XMLHttpRequest",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

    if mimic_browser:
        h.update(
            {
                "Origin": "https://www.wildberries.ru",
                "Sec-Fetch-Site": "same-origin",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Dest": "empty",
                "sec-ch-ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="24"',
                "sec-ch-ua-platform": '"Windows"',
                "sec-ch-ua-mobile": "?0",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
            }
        )
        ref = (
            f"https://www.wildberries.ru/catalog/0/search.aspx?search={quote_plus(referer_query)}"
            if referer_query
            else "https://www.wildberries.ru/"
        )
        h["Referer"] = ref
    else:
        h["Referer"] = "https://www.wildberries.ru/"
        h["Connection"] = "close"

    return h


def make_requests_session(cookie_file: Optional[str] = None) -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.7,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.trust_env = False

    if cookie_file:
        jar = MozillaCookieJar()
        try:
            jar.load(cookie_file, ignore_discard=True, ignore_expires=True)
            s.cookies = jar
            logger.info("Используются cookies (requests) из файла: %s", cookie_file)
        except Exception as e:  # pragma: no cover - IO branch
            logger.warning("Не удалось загрузить cookies '%s' для requests: %s", cookie_file, e)

    return s


def make_httpx_client(cookie_file: Optional[str] = None, http2: bool = True):
    if httpx is None:
        return None

    cookies = None
    if cookie_file:
        # Transfer cookies from MozillaCookieJar to httpx.Cookies
        cj = MozillaCookieJar()
        try:
            cj.load(cookie_file, ignore_discard=True, ignore_expires=True)
            cookies = httpx.Cookies()
            for c in cj:
                try:
                    cookies.set(c.name, c.value, domain=c.domain, path=c.path or "/")
                except Exception:
                    cookies.set(c.name, c.value)
            logger.info("Используются cookies (httpx) из файла: %s", cookie_file)
        except Exception as e:  # pragma: no cover - IO branch
            logger.warning("Не удалось загрузить cookies '%s' для httpx: %s", cookie_file, e)

    return httpx.Client(http2=http2, cookies=cookies, timeout=None, trust_env=False)


def parse_queries(s: str) -> List[str]:
    """Split queries by ';' and strip empties."""
    return [q.strip() for q in s.split(";") if q.strip()]


def remove_none(d: Dict[str, Optional[str]]) -> Dict[str, str]:
    return {k: v for k, v in d.items() if v is not None}


# ============================
# Product extraction / heuristics
# ============================
KNOWN_PATHS: List[Tuple[str, ...]] = [
    ("data", "products"),
    ("data", "catalog", "products"),
    ("products",),
]


def dive(payload: Any, path: Tuple[str, ...]) -> Any:
    cur: Any = payload
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return None
    return cur


def product_list_candidate_score(lst: Any, sample_n: int = 12) -> int:
    """Heuristic score to detect a WB product list-looking array."""
    if not isinstance(lst, list) or not lst:
        return 0

    score = 0
    cnt = 0
    for item in lst[:sample_n]:
        if not isinstance(item, dict):
            continue
        cnt += 1
        has_id = "id" in item
        has_name = "name" in item
        has_brand = "brand" in item
        has_sizes = isinstance(item.get("sizes"), list)
        price_prod = False
        if has_sizes:
            for s in item.get("sizes") or []:
                p = s.get("price") if isinstance(s, dict) else None
                if isinstance(p, dict) and "product" in p:
                    price_prod = True
                    break
        score += 3 * has_id + 4 * has_name + 3 * has_brand + 2 * has_sizes + 3 * price_prod

    return 0 if cnt == 0 else score


def scan_for_product_lists(payload: Any) -> List[Tuple[str, List[Dict[str, Any]], int]]:
    out: List[Tuple[str, List[Dict[str, Any]], int]] = []

    def walk(node: Any, path: List[str]) -> None:
        if isinstance(node, list):
            sc = product_list_candidate_score(node)
            if sc > 0:
                out.append(("->".join(path), node, sc))
        elif isinstance(node, dict):
            for k, v in node.items():
                walk(v, path + [k])

    walk(payload, [])
    out.sort(key=lambda t: (t[2], len(t[1])), reverse=True)
    return out


def choose_products_list(payload: Any) -> Tuple[List[Dict[str, Any]], str]:
    chosen: List[Dict[str, Any]] = []
    chosen_path: str = "not_found"

    for path in KNOWN_PATHS:
        found = dive(payload, path)
        if isinstance(found, list):
            chosen = found
            chosen_path = "->".join(path)
            break

    if not chosen or len(chosen) <= 1:
        candidates = scan_for_product_lists(payload)
        if candidates:
            cand_path, cand_list, cand_score = candidates[0]
            if (not chosen) or len(cand_list) > len(chosen):
                logger.debug(
                    "Эвристика: выбран путь '%s' (score=%s, len=%s) вместо '%s' (len=%s)",
                    cand_path,
                    cand_score,
                    len(cand_list),
                    chosen_path,
                    len(chosen) if chosen else 0,
                )
                chosen = cand_list
                chosen_path = f"heuristic:{cand_path}"
            for i, (p, lst, sc) in enumerate(candidates[:5], 1):
                logger.debug("Кандидат %s: path=%s | score=%s | len=%s", i, p, sc, len(lst))

    return chosen, chosen_path


def first_price_product_value(prod: Dict[str, Any]) -> Optional[int]:
    for s in (prod.get("sizes") or []):
        if not isinstance(s, dict):
            continue
        price = s.get("price")
        if isinstance(price, dict) and "product" in price:
            return price.get("product")  # type: ignore[return-value]
    return None


def resolve_ui_price(prod: Dict[str, Any]) -> Optional[int]:
    """WB UI-like price: prefer salePriceU/priceU; fallback to nested price.product."""
    for k in ("salePriceU", "priceU"):
        v = prod.get(k)
        if isinstance(v, (int, float)):
            return int(v)
    return first_price_product_value(prod)


def extract_item(prod: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "brand": prod.get("brand"),
        "name": prod.get("name"),
        "price_product": resolve_ui_price(prod),
    }


# ============================
# Bad variant detector
# ============================

def is_bad_wb_variant(payload: Any, products: List[Dict[str, Any]]) -> Tuple[bool, str]:
    if not isinstance(payload, dict) or not isinstance(products, list):
        return (False, "payload/products type mismatch")

    meta = payload.get("metadata") or {}
    meta_preset = False
    if isinstance(meta, dict):
        if str(meta.get("catalog_type", "")).lower() == "preset":
            meta_preset = True
        cv = str(meta.get("catalog_value", ""))
        if "preset=" in cv:
            meta_preset = True

    sample = [p for p in products if isinstance(p, dict)][:40]
    total = len(sample)
    if total == 0:
        return (False, "no products in sample")

    nested_price_cnt = 0
    flat_price_cnt = 0

    for p in sample:
        has_nested = False
        for s in (p.get("sizes") or []):
            if isinstance(s, dict):
                pr = s.get("price")
                if isinstance(pr, dict) and ("product" in pr) and isinstance(pr.get("product"), (int, float)):
                    has_nested = True
                    break
        if has_nested:
            nested_price_cnt += 1
        if ("priceU" in p) or ("salePriceU" in p):
            flat_price_cnt += 1

    cond_meta = meta_preset
    cond_no_nested = nested_price_cnt == 0
    cond_flat_majority = flat_price_cnt >= max(1, total // 2)

    bad = cond_meta and cond_no_nested and cond_flat_majority
    reason = f"preset={cond_meta}, nested={nested_price_cnt}/{total}, flat={flat_price_cnt}/{total}"
    return (bad, reason)


# ============================
# Networking
# ============================

def debug_log_full_response(resp: Any, base_url: str) -> None:
    """Log full HTTP response body on DEBUG without risking unterminated strings."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    try:
        body = resp.text  # type: ignore[assignment]
    except Exception:
        try:
            body = resp.content.decode("utf-8", "replace")  # type: ignore[assignment]
        except Exception:
            body = "<unreadable body>"
    status = getattr(resp, "status_code", "?")
    msg = "FULL RESPONSE BODY (%s) status=%s:%s"
    logger.debug(msg, base_url, status, body if isinstance(body, str) else str(body))


@dataclass
class Variant:
    engine: str  # "httpx" | "requests"
    http2: bool = True
    mimic: bool = True
    spp: str = "30"
    cache_bust: bool = True


def build_params(
    *, query: str, page: int, limit: int, spp: Optional[str], dest: Optional[str], cache_bust: bool
) -> Dict[str, str]:
    params = dict(DEFAULT_PARAMS)
    params.update({"query": query, "page": str(page), "limit": str(limit)})
    if spp is not None:
        params["spp"] = str(spp)
    if dest is not None:
        params["dest"] = str(dest)
    if cache_bust:
        params["rnd"] = str(time.time_ns() % 10**9)
    return remove_none(params)


def attempt_with_client(client: Any, url: str, params: Dict[str, str], headers: Dict[str, str], timeout: float) -> Optional[Dict[str, Any]]:
    t0 = time.perf_counter()
    resp = client.get(url, params=params, headers=headers, timeout=timeout)
    elapsed = time.perf_counter() - t0
    logger.debug("GET %s", getattr(resp, "url", url))
    logger.info(
        " status=%s | %.3fs | bytes=%s",
        getattr(resp, "status_code", "?"),
        elapsed,
        len(getattr(resp, "content", b"")),
    )
    debug_log_full_response(resp, url)

    if hasattr(resp, "raise_for_status"):
        with suppress(Exception):
            resp.raise_for_status()

    try:
        return resp.json()
    except Exception:
        text_preview = getattr(resp, "text", "")
        logger.warning("Ответ не JSON. Фрагмент: %s", text_preview[:300])
        return None


def fetch_page_best_effort(
    *,
    query: str,
    page: int,
    limit: int,
    timeout: float,
    dest: Optional[str],
    debug_dump_dir: Optional[Path],
    cookie_file: Optional[str],
    allow_flat_fallback: bool,
) -> Tuple[List[Dict[str, Any]], str, Dict[str, Optional[str]]]:
    """
    Try multiple client profiles to avoid the preset structure.
    Returns: (items, mode, override)
      - items: list of extracted items
      - mode: {"ok", "fallback", "bad"}
      - override: server hints like dest/spp if present
    """

    variants: List[Variant] = []
    if httpx is not None:
        variants.append(Variant(engine="httpx", http2=True, mimic=True, spp="0", cache_bust=False))
        variants.append(Variant(engine="httpx", http2=True, mimic=True, spp="30", cache_bust=True))
    variants.append(Variant(engine="requests", http2=False, mimic=True, spp="0", cache_bust=False))
    variants.append(Variant(engine="requests", http2=False, mimic=True, spp="30", cache_bust=True))
    variants.append(Variant(engine="requests", http2=False, mimic=False, spp="0", cache_bust=True))

    # expand up to VARIANT_BUDGET with light mutations
    while len(variants) < VARIANT_BUDGET and variants:
        base = variants[len(variants) % len(variants)]
        mutated = Variant(
            engine=base.engine,
            http2=base.http2,
            mimic=base.mimic,
            spp="0" if base.spp != "0" else "30",
            cache_bust=not base.cache_bust,
        )
        variants.append(mutated)
    variants = variants[:VARIANT_BUDGET]

    last_payload: Optional[Dict[str, Any]] = None
    last_bad_reason: str = ""
    last_override: Dict[str, Optional[str]] = {}

    for idx, v in enumerate(variants, 1):
        logger.info(" → Попытка %s/%s: %s", idx, len(variants), v)

        # Create a fresh client per attempt
        if v.engine == "httpx":
            if httpx is None:
                logger.debug(" httpx недоступен, пропускаем.")
                continue
            client = make_httpx_client(cookie_file=cookie_file, http2=v.http2)
        else:
            client = make_requests_session(cookie_file=cookie_file)

        try:
            headers = make_headers(ua=random_user_agent(), mimic_browser=v.mimic, referer_query=query)
            params = build_params(
                query=query,
                page=page,
                limit=limit,
                spp=v.spp,
                dest=dest,
                cache_bust=v.cache_bust,
            )
            payload = attempt_with_client(client, BASE_URL, params=params, headers=headers, timeout=timeout)
            if payload is None:
                continue

            # Note server-provided dest/spp (for logs and later reuse)
            srv_params = payload.get("params") or {}
            srv_dest = srv_params.get("dest")
            srv_spp = srv_params.get("spp")
            override: Dict[str, Optional[str]] = {}
            if srv_dest is not None:
                override["dest"] = str(srv_dest)
            if srv_spp is not None:
                override["spp"] = str(srv_spp)
            if srv_dest is not None:
                logger.debug(" Server dest=%s (requested %s)", srv_dest, dest)
            if srv_spp is not None:
                logger.debug(" Server spp=%s", srv_spp)

            products_raw, where = choose_products_list(payload)
            logger.info(" Выбран путь: %s | товаров: %s", where, len(products_raw))

            bad, reason = is_bad_wb_variant(payload, products_raw)
            if bad:
                last_payload = payload
                last_bad_reason = reason
                last_override = override
                logger.warning(" Аномалия WB (detected): %s", reason)

                if debug_dump_dir:
                    with suppress(OSError):
                        debug_dump_dir.mkdir(parents=True, exist_ok=True)
                        dump_path = debug_dump_dir / f"wb_bad_{int(time.time())}_p{page}_v{idx}.json"
                        dump_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                        logger.warning(" Дамп «плохого» ответа сохранён: %s", dump_path)

                if RETRY_JITTER > 0:
                    time.sleep(RETRY_JITTER + random.uniform(0, RETRY_JITTER / 2))
                continue

            # Normal structure — parse UI-like prices
            items = [extract_item(p) for p in products_raw]
            return items, "ok", override

        except requests.HTTPError as e:  # pragma: no cover - network branch
            code = e.response.status_code if getattr(e, "response", None) is not None else "?"
            logger.error("HTTP %s для '%s' page=%s", code, query, page)
        except (requests.RequestException, Exception) as e:  # pragma: no cover - network branch
            logger.error("Сетевая/клиентская ошибка: %s", e)
        finally:
            with suppress(Exception):
                # Close both requests and httpx clients
                close = getattr(client, "close", None)
                if callable(close):
                    close()

    # All attempts yielded preset. Try flat fallback if allowed.
    if allow_flat_fallback and isinstance(last_payload, dict):
        flat_list, _ = choose_products_list(last_payload)
        items = [extract_item(p) for p in flat_list]
        logger.warning("Все профили дали preset (%s). Применён fallback по flat-ценам.", last_bad_reason or "no-reason")
        return items, "fallback", last_override

    logger.warning("Все профили дали preset. Fallback отключён — помечаем как Bad response.")
    return [], "bad", last_override


# ============================
# CSV / formatting
# ============================

def format_price_rub(v: Any) -> str:
    """Format WB integer price in kopeks to human string like '1 234,56'."""
    if v is None:
        return ""
    try:
        cents = int(round(float(v)))  # WB gives prices in kopeks
    except (TypeError, ValueError):
        return str(v)

    sign = "-" if cents < 0 else ""
    rub = abs(cents) // 100
    kop = abs(cents) % 100
    return f"{sign}{rub}" if kop == 0 else f"{sign}{rub},{kop:02d}"


def save_csv_report(result: Dict[str, Any]) -> Path:
    ts_human = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    ts_fs = ts_human.replace(":", "-")  # for Windows
    filename = f"wb_search_report_{ts_fs}.csv"
    path = Path(filename)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        for block in result.get("queries", []):
            q = block.get("query", "")
            products = block.get("products", []) or []

            writer.writerow([q])  # query header
            writer.writerow(["№", "brand", "name", "price_product", "price_rating"])  # added column

            for i, p in enumerate(products, start=1):
                price = format_price_rub(p.get("price_product"))
                rating = p.get("price_rating")
                writer.writerow([i, p.get("brand"), p.get("name"), price, rating if rating is not None else ""])  # noqa: E501

            writer.writerow([])  # separator

    logger.info("CSV отчёт сохранён: %s", path)
    return path


# ============================
# Brand-based filtering helpers
# ============================

def compute_key_price(items: List[Dict[str, Any]], brand: str) -> Optional[int]:
    """Pick reference price for key brand.
    If multiple items of the brand exist, choose the minimal price_product.
    Returns price in kopeks (int) or None if no suitable item found.
    """
    brand_lc = (brand or "").strip().lower()
    candidates: List[int] = []
    for it in items:
        it_brand = str(it.get("brand") or "").strip().lower()
        price = it.get("price_product")
        if it_brand == brand_lc and isinstance(price, (int, float)):
            try:
                candidates.append(int(price))
            except Exception:
                pass
    if not candidates:
        return None
    return min(candidates)


def filter_by_price_diff(
    items: List[Dict[str, Any]],
    ref_price: int,
    pct: float,
) -> List[Dict[str, Any]]:
    """Keep items whose absolute diff from ref_price is within ±pct%.
    Items with missing/invalid price are kept (can't compare reliably).
    Order is preserved.
    """
    threshold = abs(ref_price) * (abs(pct) / 100.0)

    def keep(it: Dict[str, Any]) -> bool:
        v = it.get("price_product")
        if not isinstance(v, (int, float)):
            return True
        try:
            diff = abs(int(v) - int(ref_price))
        except Exception:
            return True
        return diff <= threshold

    return [it for it in items if keep(it)]


# ============================
# Price rating (ranking) helpers
# ============================

def _dense_rank_by_price(items: List[Dict[str, Any]]) -> Dict[int, int]:
    """Compute dense ranks by price_product across the given *final* list.
    Returns mapping price(int)->rank (1 = cheapest). Equal prices share the same rank,
    and the next distinct price gets the next consecutive rank (dense ranking).
    """
    prices: List[int] = []
    for it in items:
        v = it.get("price_product")
        if isinstance(v, (int, float)):
            try:
                prices.append(int(v))
            except Exception:
                pass
    uniq_sorted = sorted(set(prices))
    return {price: i + 1 for i, price in enumerate(uniq_sorted)}


def annotate_key_brand_price_ratings(items: List[Dict[str, Any]], key_brand: str) -> None:
    """Annotate items in-place with 'price_rating' for key brand products only.
    Does not change order. Works on the final list (after filtering & trimming).
    """
    rank_map = _dense_rank_by_price(items)
    key = (key_brand or "").strip().lower()
    for it in items:
        brand = str(it.get("brand") or "").strip().lower()
        if brand != key:
            # ensure non-key products have no rating field or reset to None
            it.pop("price_rating", None)
            continue
        v = it.get("price_product")
        if isinstance(v, (int, float)):
            try:
                it["price_rating"] = rank_map.get(int(v))
            except Exception:
                it["price_rating"] = None
        else:
            it["price_rating"] = None



# ============================
# CLI
# ============================

def positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return ivalue


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="WB multi-search → CSV и (DEBUG) JSON")
    ap.add_argument(
        "-q",
        "--queries",
        required=True,
        help="Запросы через ';' (пример: 'монета крокодил гена; монета россия')",
    )
    ap.add_argument("--pages", type=positive_int, default=1, help="Сколько страниц брать (по умолчанию 1)")
    ap.add_argument("--limit", type=positive_int, default=20, help="Сколько товаров на страницу (до ~100)")
    ap.add_argument(
        "--top", type=positive_int, default=20, help="Сколько первых товаров вернуть по каждому запросу (по умолчанию 20)"
    )
    ap.add_argument("--timeout", type=float, default=12.0, help="Таймаут запроса (сек)")
    ap.add_argument("--delay", type=float, default=0.5, help="Пауза между страницами + джиттер (сек)")
    ap.add_argument("--dest", type=str, default=None, help="Переопределить dest (регион)")
    ap.add_argument("--pretty", action="store_true", help="Красивый JSON (для DEBUG-лога)")
    ap.add_argument("--output", type=str, default=None, help="Куда сохранить JSON (опционально)")
    ap.add_argument("--log-level", type=str, default="INFO", help="DEBUG | INFO | WARNING | ERROR")
    ap.add_argument("--debug-dump-dir", type=str, default=None, help="Папка для дампа сырого ответа при подозрениях")
    ap.add_argument("--cookie-file", type=str, default=None, help="Файл cookies (Netscape/Mozilla) для u-search/wildberries")
    ap.add_argument("--no-flat-fallback", action="store_true", help="Отключить запасной парсинг по flat-ценам (salePriceU/priceU)")

    # Filtering overrides (script has defaults)
    ap.add_argument("--key-brand", type=str, default=None, help="Переопределить ключевой бренд (иначе значение по умолчанию)")
    ap.add_argument(
        "--price-diff-pct",
        type=float,
        default=None,
        help="Переопределить порог процента отклонения (иначе значение по умолчанию)",
    )

    return ap


def main(argv: Optional[Iterable[str]] = None) -> None:
    ap = build_arg_parser()
    args = ap.parse_args(argv)

    setup_logging(args.log_level)

    # Script-embedded filter config + CLI overrides
    cfg = FilterConfig()
    if isinstance(args.key_brand, str) and args.key_brand.strip():
        cfg.key_brand = args.key_brand.strip()
    if isinstance(args.price_diff_pct, float):
        cfg.price_diff_pct = float(args.price_diff_pct)

    top_n = max(0, int(args.top))
    queries = parse_queries(args.queries)
    if not queries:
        logger.error("Пустой список запросов.")
        sys.exit(2)

    debug_dump_dir = Path(args.debug_dump_dir) if args.debug_dump_dir else None

    result: Dict[str, Any] = {"queries": []}
    t0 = time.perf_counter()

    # Track server-provided dest (if it changes mid-run)
    current_dest: Optional[str] = args.dest

    for qi, q in enumerate(queries, 1):
        logger.info("===== Query %s/%s: '%s' =====", qi, len(queries), q)
        acc: List[Dict[str, Any]] = []
        query_status: str = "ok"

        for page in range(1, args.pages + 1):
            items, mode, override = fetch_page_best_effort(
                query=q,
                page=page,
                limit=args.limit,
                timeout=args.timeout,
                dest=current_dest,
                debug_dump_dir=debug_dump_dir,
                cookie_file=args.cookie_file,
                allow_flat_fallback=not args.no_flat_fallback,
            )

            # If server hinted its own dest — switch to it for subsequent requests
            if override.get("dest") and override["dest"] != (current_dest or ""):
                logger.info(" Переключаем dest на серверный: %s → %s", current_dest, override["dest"])
                current_dest = override["dest"]

            if mode == "bad":
                query_status = "Bad response"
                break
            elif mode == "fallback" and query_status == "ok":
                query_status = "Fallback(flat)"

            if items:
                # IMPORTANT: preserve page order — extend as-is
                acc.extend(items)
            else:
                logger.debug(" Пустая страница или ошибка.")

            if len(acc) >= top_n:
                logger.info(" Достигнут лимит --top=%s, ранняя остановка по '%s' на page=%s", top_n, q, page)
                break

            if page < args.pages and args.delay > 0:
                time.sleep(args.delay + random.uniform(0, args.delay / 2))

        logger.info("Собрано по '%s': %s товаров (до фильтрации)", q, len(acc))

        # ===== Brand-based price filtering (order-preserving) =====
        ref_price = compute_key_price(acc, cfg.key_brand)
        if ref_price is not None:
            before = len(acc)
            acc = filter_by_price_diff(acc, ref_price, cfg.price_diff_pct)
            after = len(acc)
            logger.info(
                "Фильтрация: бренд='%s', порог=%.2f%%, опорная цена=%s → осталось %s из %s",
                cfg.key_brand,
                cfg.price_diff_pct,
                format_price_rub(ref_price),
                after,
                before,
            )
        else:
            logger.info(
                "Фильтрация пропущена: не найден ключевой бренд '%s' в результатах.",
                cfg.key_brand,
            )

        # ===== Trim after filtering =====
        trimmed = acc[:top_n]
        if len(acc) > len(trimmed):
            logger.info("Обрезаем до первых %s элементов.", len(trimmed))

        # ===== Price rating for key brand (on final list) =====
        annotate_key_brand_price_ratings(trimmed, cfg.key_brand)

        entry: Dict[str, Any] = {"query": q, "products": trimmed}
        if query_status != "ok":
            entry["status"] = query_status  # e.g., Fallback(flat) | Bad response
        result["queries"].append(entry)

    logger.info("Готово. Всего страниц max: %s. Время: %.3fs", len(queries) * args.pages, time.perf_counter() - t0)

    out_json = json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None)
    logger.debug("FINAL JSON RESULT: %s", out_json)

    if args.output:
        try:
            Path(args.output).write_text(out_json, encoding="utf-8")
            logger.info("JSON сохранён в: %s", args.output)
        except OSError as e:  # pragma: no cover - IO branch
            logger.error("Не удалось сохранить '%s': %s", args.output, e)

    # CSV report
    save_csv_report(result)


if __name__ == "__main__":
    main()
