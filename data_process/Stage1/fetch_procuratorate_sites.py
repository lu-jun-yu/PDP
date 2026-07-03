#!/usr/bin/env python3
"""
fetch_procuratorate_sites.py

通用检察院法律文书公开栏目爬虫。按 data/public_sources/Stage1/url_prefixes_list.json
批量抓取 16 个站点，兼容两类分页：
    1) TRS CMS 标准分页：首页含 `var countPage = N`，后续页 index_1.{shtml,jhtml}
    2) VSB CMS 分页：subei / yongjing，形如 {basename}_{N}.htm，逐页到 404 为止

需要 JS 渲染的站点（ah、jxshangyou）写进 SKIP_NETLOCS 跳过。

进度文件与 fetch_12309.py / fetch_jljiutai.py 共用 12309_progress.json，
其中新增 `procuratorate_sites` 字段记录每站独立页码，`fetched_urls` 与
`found_documents` 仍保持共享去重。

用法：
    # 默认 dry-run，列出将被抓取的详情 URL
    python scripts/fetch_procuratorate_sites.py --site www.nmetuoke.jcy.gov.cn --max-pages 1

    # 正式抓（单站）
    python scripts/fetch_procuratorate_sites.py --site www.nmetuoke.jcy.gov.cn --no-dry-run

    # 全量
    python scripts/fetch_procuratorate_sites.py --no-dry-run

    # 断点续跑
    python scripts/fetch_procuratorate_sites.py --no-dry-run --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import defaultdict
from html import unescape
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from data_process.extract_documents import parse_document, UNKNOWN_WARNINGS  # noqa: E402


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# 跳过：需 JS 渲染 / AJAX 分页的站点
SKIP_NETLOCS = {
    "www.ah.jcy.gov.cn",
    "www.jxshangyou.jcy.gov.cn",
}

# VSB CMS 分页：basename_{N}.htm
VSB_NETLOCS = {
    "www.subei.jcy.gov.cn",
    "www.yongjing.jcy.gov.cn",
}

# 详情 URL 识别正则：TRS (YYYYMM/tDDDDDDDD_N.shtml) 或 VSB (info/NNNN/NNNN.htm[l])
# 允许任意多层 `./` / `../` 前缀（VSB 子目录分页会有 `../../info/...`）
DETAIL_HREF_RE = re.compile(
    r'href=["\']('
    r'(?:https?://[^"\']+?|(?:\.{1,2}/)*[\w/\-]*?)'
    r'(?:\d{6}/t\d+_\d+\.s?html|info/\d+/\d+\.html?)'
    r')["\']',
    flags=re.I,
)

CASE_NUM_RE = re.compile(
    r"[\u4e00-\u9fff]+\s*[〔\[【]\s*\d{4}\s*[〕\]】]\s*[A-Za-z]?\s*\d+\s*号"
)
DATE_RE = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")

# 页面正文节点候选（优先级从高到低）；每条形如 (attr, value)，attr ∈ {id, class}
CONTENT_SELECTORS: list[tuple[str, str]] = [
    ("id", "fontzoom"),
    ("id", "zoom"),
    ("class", "TRS_Editor"),
    ("class", "TRS_PreAppend"),
    ("class", "contbox"),
    ("class", "content"),
    ("id", "article-body"),
    ("class", "article-body"),
    ("class", "article"),
]


# ---------------------------------------------------------------------------
# 编码、HTML 与文本工具
# ---------------------------------------------------------------------------

def normalize_space(text: str) -> str:
    text = text.replace("\r", "").replace("\xa0", " ")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_html_tags(html: str) -> str:
    s = html
    s = re.sub(r"(?is)<script[^>]*>.*?</script>", "", s)
    s = re.sub(r"(?is)<style[^>]*>.*?</style>", "", s)
    s = re.sub(r"(?i)<br\s*/?>", "\n", s)
    s = re.sub(r"(?i)</p\s*>", "\n", s)
    s = re.sub(r"(?i)</div\s*>", "\n", s)
    s = re.sub(r"(?s)<[^>]+>", "", s)
    s = unescape(s)
    return normalize_space(s)


def pick_response_encoding(resp: requests.Response) -> str:
    """优先读 HTML meta charset；其次 response header；最后 gb2312。"""
    raw_preview = resp.content[:4000].decode("ascii", errors="ignore")
    m = re.search(r'charset=["\']?([a-zA-Z0-9\-_]+)', raw_preview, flags=re.I)
    if m:
        return m.group(1).lower()
    if resp.encoding and resp.encoding.lower() not in {"iso-8859-1", "ascii"}:
        return resp.encoding
    return "gb2312"


def fetch_html(session: requests.Session, url: str, timeout: int) -> str:
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    resp.encoding = pick_response_encoding(resp)
    return resp.text


def guess_ext_from_url(url: str) -> str:
    m = re.search(r"\.(shtml|html|jhtml|htm)(?:$|[?#])", url, flags=re.I)
    return "." + m.group(1).lower() if m else ".shtml"


def case_num_to_filename(case_num: str) -> str:
    s = case_num.replace("〔", "[").replace("〕", "]").replace("【", "[").replace("】", "]")
    return re.sub(r'[<>:"/\\|?*]', "_", s)


def normalize_case_num(s: str) -> str:
    return s.replace("〔", "[").replace("〕", "]").replace("【", "[").replace("】", "]")


def _source_url_prefix(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return ""
    parsed = urlparse(u if "://" in u else f"http://{u}")
    netloc = (parsed.netloc or "").strip().lower()
    if not netloc and parsed.path:
        parsed = urlparse(f"http://{u}")
        netloc = (parsed.netloc or "").strip().lower()
    if not netloc:
        return ""
    scheme = (parsed.scheme or "http").strip().lower()
    return f"{scheme}://{netloc}"


def merge_dedup_key(rec: dict) -> str:
    rid = (rec.get("id") or "").strip()
    case_part = normalize_case_num(rid) if rid else ""
    url_pref = _source_url_prefix(rec.get("source_url", ""))
    if case_part and url_pref:
        return f"{case_part}\x1e{url_pref}"
    if case_part:
        return f"{case_part}\x1e"
    return url_pref or ""


def extract_date_from_text(text: str) -> str | None:
    tail = text[-500:] if len(text) > 500 else text
    dates = DATE_RE.findall(tail)
    if dates:
        y, m, d = dates[-1]
        try:
            return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
        except ValueError:
            pass
    return None


def clean_case_num(raw: str) -> str:
    s = raw
    for sep in ["不起诉决定书", "起诉书"]:
        if sep in s:
            s = s[s.rindex(sep) + len(sep):]
            break
    if "人民检察院" in s:
        s = s[s.rindex("人民检察院") + len("人民检察院"):]
    m = re.search(r"[罪案](?=[\u4e00-\u9fff])", s)
    if m and m.start() < len(s) // 2:
        s = s[m.end():]
    m = CASE_NUM_RE.search(s)
    if m:
        return m.group().replace(" ", "")
    m = CASE_NUM_RE.search(raw)
    return m.group().replace(" ", "") if m else raw


def identify_document(text: str) -> dict | None:
    if not text or len(text) < 200:
        return None
    if "人民检察院" not in text and "检察院" not in text:
        return None

    case_m = CASE_NUM_RE.search(text)
    if not case_m:
        return None
    case_num = clean_case_num(case_m.group())
    if "检" not in case_num or "诉" not in case_num:
        return None

    head = text[:2000]
    if "不起诉决定书" in head or ("不起诉" in head and "起诉书" not in head):
        doc_type = "不起诉"
    elif "起诉书" in head or "提起公诉" in text[:5000]:
        doc_type = "起诉"
    else:
        return None

    return {
        "case_num": case_num,
        "doc_type": doc_type,
        "date": extract_date_from_text(text),
    }


def extract_matched_block(html: str, attr: str, value: str) -> str | None:
    """抽出 <TAG attr="value" ...>...</TAG> 的内容（含嵌套处理较弱，取第一个闭合）。"""
    # 匹配开标签：<tag ... attr="value" ...>
    pat = re.compile(
        rf'<(\w+)[^>]*\b{attr}\s*=\s*["\'][^"\']*\b{re.escape(value)}\b[^"\']*["\'][^>]*>',
        flags=re.I,
    )
    m = pat.search(html)
    if not m:
        return None
    tag = m.group(1)
    start = m.end()
    # 从 start 起找对应 </tag>，用 depth 粗略匹配嵌套
    open_re = re.compile(rf"<{tag}\b", flags=re.I)
    close_re = re.compile(rf"</{tag}\s*>", flags=re.I)
    depth = 1
    pos = start
    while depth > 0:
        next_open = open_re.search(html, pos)
        next_close = close_re.search(html, pos)
        if not next_close:
            return None
        if next_open and next_open.start() < next_close.start():
            depth += 1
            pos = next_open.end()
        else:
            depth -= 1
            pos = next_close.end()
            if depth == 0:
                return html[start:next_close.start()]
    return None


def extract_content(html: str) -> str | None:
    for attr, value in CONTENT_SELECTORS:
        block = extract_matched_block(html, attr, value)
        if block:
            text = strip_html_tags(block)
            if len(text) >= 100:
                return text
    # fallback：取 body 中最长的 div
    body_m = re.search(r"(?is)<body[^>]*>(.*?)</body>", html)
    body = body_m.group(1) if body_m else html
    best = None
    best_len = 0
    for m in re.finditer(r"(?is)<div\b[^>]*>(.*?)</div>", body):
        t = strip_html_tags(m.group(1))
        if len(t) > best_len and len(t) >= 300:
            best_len = len(t)
            best = t
    return best


def extract_meta_pubdate(html: str) -> str | None:
    m = re.search(
        r'<meta[^>]+name=["\'](?:publishdate|PubDate|pubdate)["\'][^>]+content=["\']([^"\']+)["\']',
        html,
        flags=re.I,
    )
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# 列表页解析
# ---------------------------------------------------------------------------

def parse_count_page(html: str) -> int:
    m = re.search(r"var\s+countPage\s*=\s*(\d+)", html)
    return max(1, int(m.group(1))) if m else 1


def parse_list_entries(list_html: str, list_url: str) -> list[dict]:
    """从列表 HTML 抽详情链接，返回 [{detail_url, title}, ...]。"""
    entries: list[dict] = []
    seen = set()
    for m in DETAIL_HREF_RE.finditer(list_html):
        href = m.group(1).strip()
        detail_url = urljoin(list_url, href)
        # 同主机约束：避免抓到友情链接
        if urlparse(detail_url).netloc != urlparse(list_url).netloc:
            continue
        if detail_url in seen:
            continue
        seen.add(detail_url)
        # 尝试取这一段 <a>...</a> 的文本作为 title
        title = ""
        a_match = re.search(
            r'<a[^>]+href=["\']' + re.escape(href) + r'["\'][^>]*>(.*?)</a>',
            list_html,
            flags=re.I | re.S,
        )
        if a_match:
            title = strip_html_tags(a_match.group(1))[:120]
        entries.append({"detail_url": detail_url, "title": title})
    return entries


# ---------------------------------------------------------------------------
# 分页迭代器
# ---------------------------------------------------------------------------

def iter_pages_trs(
    session: requests.Session,
    first_url: str,
    first_html: str,
    max_pages: int | None,
    timeout: int,
):
    """TRS 标准分页：yield (page_num, page_url, page_html)。"""
    total = parse_count_page(first_html)
    if max_pages is not None:
        total = min(total, max(1, max_pages))
    yield 1, first_url, first_html, total
    ext = guess_ext_from_url(first_url)
    base_dir = first_url.rsplit("/", 1)[0] + "/"
    for n in range(1, total):
        url = urljoin(base_dir, f"index_{n}{ext}")
        try:
            html = fetch_html(session, url, timeout)
        except Exception as e:
            log.warning("  TRS 分页抓取失败: %s | %s", url, e)
            break
        yield n + 1, url, html, total


_VSB_TOTAL_RE = re.compile(r"_simple_list_gotopage(?:_fun)?\s*\(\s*(\d+)")


def iter_pages_vsb(
    session: requests.Session,
    first_url: str,
    first_html: str,
    max_pages: int | None,
    timeout: int,
):
    """VSB CMS (pub_mode=2) 分页：首页 {dir}/{base}.htm；后续页 {dir}/{base}/{k}.htm，
    其中 k = 1..total-1（注意站点上页码编号与文件名反向，抓取顺序无所谓）。

    总页数从 HTML 里的 `_simple_list_gotopage_fun(N, ...)` 调用解析。
    """
    m = _VSB_TOTAL_RE.search(first_html)
    total = int(m.group(1)) if m else 1
    if max_pages is not None:
        total = min(total, max(1, max_pages))
    log.info("  VSB totalPages=%d", total)

    yield 1, first_url, first_html, total
    if total <= 1:
        return

    base_dir, fname = first_url.rsplit("/", 1)
    base_dir += "/"
    stem, _, _ = fname.rpartition(".")
    page_dir = urljoin(base_dir, f"{stem}/")

    # 抓文件名 1.htm 到 (total-1).htm。浏览器视角的 "第 2..total 页" 与这些文件
    # 反向一一对应；我们只关心拿到全部内容，逐个抓即可。
    for k in range(1, total):
        url = urljoin(page_dir, f"{k}.htm")
        try:
            resp = session.get(url, timeout=timeout)
        except Exception as e:
            log.warning("  VSB 分页请求失败: %s | %s", url, e)
            continue
        if resp.status_code == 404:
            log.info("  VSB 文件不存在（404 at %s），跳过", url)
            continue
        if resp.status_code >= 400:
            log.warning("  VSB 分页 HTTP %d: %s", resp.status_code, url)
            continue
        resp.encoding = pick_response_encoding(resp)
        html = resp.text
        entries = parse_list_entries(html, url)
        if not entries:
            log.info("  VSB 空列表 → 跳过 %s", url)
            continue
        # k=1 对应站点的"最后一页"，但我们的 page_num 只用于进度/日志；
        # 为了 resume 顺序稳定，用 k+1 作为 page_num
        yield k + 1, url, html, total


# ---------------------------------------------------------------------------
# 进度文件
# ---------------------------------------------------------------------------

def load_progress(path: Path) -> dict:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data.setdefault("fetched_urls", {})
            data.setdefault("found_documents", {})
            data.setdefault("procuratorate_sites", {})
            return data
    return {
        "fetched_urls": {},
        "found_documents": {},
        "procuratorate_sites": {},
    }


def save_progress(path: Path, progress: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# 解析与落盘
# ---------------------------------------------------------------------------

def _has_quality_issues(parsed: dict) -> list[str]:
    return [w for w in parsed.get("_warnings", []) if w in UNKNOWN_WARNINGS]


def _save_stage1_dataset_json(output_dir: Path, parsed_results: dict[str, list[dict]]) -> None:
    rows = parsed_results.get("test") or []
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = output_dir / "dataset.json.tmp"
    final_path = output_dir / "dataset.json"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    tmp_path.replace(final_path)


def _load_dataset_case_keys(output_dir: Path) -> set[str]:
    keys: set[str] = set()
    for ds in [
        output_dir / "dataset.json",
        output_dir / "test" / "dataset.json",
        output_dir / "train" / "dataset.json",
    ]:
        if not ds.exists():
            continue
        try:
            with open(ds, "r", encoding="utf-8") as f:
                rows = json.load(f)
            if isinstance(rows, list):
                for rec in rows:
                    rid = str(rec.get("id", "")).strip()
                    if rid:
                        keys.add(normalize_case_num(rid))
        except Exception as e:
            log.warning("读取 dataset 失败: %s | %s", ds, e)
    return keys


# ---------------------------------------------------------------------------
# 单站主循环
# ---------------------------------------------------------------------------

def process_site(
    start_url: str,
    args: argparse.Namespace,
    session: requests.Session,
    progress: dict,
    progress_path: Path,
    output_dir: Path,
    parsed_results: dict,
    decision_counts: dict,
    seen_parse_keys: set,
    existing_case_keys: set,
    stats: dict,
) -> None:
    netloc = urlparse(start_url).netloc
    if netloc in SKIP_NETLOCS:
        log.info("跳过 %s（需 JS 渲染，暂不支持）", netloc)
        return

    site_state = progress["procuratorate_sites"].setdefault(
        netloc, {"current_page": 1, "total_pages": None, "done": False}
    )
    if not args.resume:
        site_state["current_page"] = 1
        site_state["done"] = False
    resume_page = max(1, int(site_state.get("current_page", 1)))

    log.info("")
    log.info("=" * 60)
    log.info(" 站点: %s  start_url=%s  resume_page=%d",
             netloc, start_url, resume_page)
    log.info("=" * 60)

    try:
        first_html = fetch_html(session, start_url, args.timeout)
    except Exception as e:
        log.error("首页抓取失败 %s: %s", start_url, e)
        stats["site_fetch_failed"] += 1
        return

    iter_pages = iter_pages_vsb if netloc in VSB_NETLOCS else iter_pages_trs
    pages = iter_pages(session, start_url, first_html, args.max_pages, args.timeout)

    test_dir = output_dir
    test_dir.mkdir(parents=True, exist_ok=True)

    page_num = 0
    for page_num, page_url, page_html, total_pages in pages:
        if page_num < resume_page:
            continue
        entries = parse_list_entries(page_html, page_url)
        if total_pages:
            site_state["total_pages"] = total_pages
        log.info("[%s] 第 %d 页 (%s): %d 条", netloc, page_num, page_url, len(entries))

        for i, item in enumerate(entries, start=1):
            detail_url = item["detail_url"]
            title = item.get("title") or ""
            tag = f"[{netloc} p{page_num} #{i}/{len(entries)}]"

            if detail_url in progress["fetched_urls"]:
                stats["skipped"] += 1
                continue

            if args.dry_run:
                log.info("%s [dry-run] %s | %s", tag, detail_url, title[:40])
                stats["dry_run"] += 1
                continue

            try:
                detail_html = fetch_html(session, detail_url, args.timeout)
            except Exception as e:
                log.warning("%s 详情抓取失败: %s | %s", tag, detail_url, e)
                progress["fetched_urls"][detail_url] = "fetch_failed"
                stats["fetch_failed"] += 1
                continue

            text = extract_content(detail_html) or ""
            if len(text) < 100:
                log.info("%s ✗ 正文过短 (%d)，跳过", tag, len(text))
                progress["fetched_urls"][detail_url] = "empty_content"
                stats["empty_content"] += 1
                continue

            # 严格内容校验：非检察文书 → 丢弃
            doc_info = identify_document(text)
            if not doc_info:
                progress["fetched_urls"][detail_url] = "not_legal_doc"
                stats["not_legal_doc"] += 1
                continue

            case_num = doc_info["case_num"]
            doc_type_cn = doc_info["doc_type"]
            doc_type_key = "prosecution" if doc_type_cn == "起诉" else "non_prosecution"
            case_key = normalize_case_num(case_num)

            if case_key in existing_case_keys:
                progress["fetched_urls"][detail_url] = "duplicate"
                stats["duplicate"] += 1
                continue

            # 落盘
            doc_dir = test_dir / doc_type_cn
            doc_dir.mkdir(parents=True, exist_ok=True)
            file_stem = case_num_to_filename(case_num)
            txt_path = doc_dir / f"{file_stem}.txt"
            if txt_path.exists():
                progress["fetched_urls"][detail_url] = "duplicate"
                stats["duplicate"] += 1
                existing_case_keys.add(case_key)
                continue
            txt_path.write_text(text, encoding="utf-8")

            publish_date = extract_meta_pubdate(detail_html) or doc_info.get("date")
            progress["fetched_urls"][detail_url] = "found"
            progress["found_documents"][case_key] = {
                "case_num": case_num,
                "date": publish_date,
                "source_url": detail_url,
                "save_path": str(txt_path),
                "split": "test",
                "netloc": netloc,
            }

            parsed = parse_document(str(txt_path), doc_type_key)
            if parsed and parsed.get("decision") and parsed.get("fact"):
                hit = _has_quality_issues(parsed)
                if hit:
                    log.info("%s → 解析质量不合格: %s", tag, ", ".join(hit))
                    stats["quality_fail"] += 1
                else:
                    parsed.pop("_warnings", None)
                    parsed["source_url"] = detail_url
                    pk = merge_dedup_key(parsed)
                    if pk not in seen_parse_keys:
                        seen_parse_keys.add(pk)
                        parsed_results["test"].append(parsed)
                        dec = parsed["decision"]
                        decision_counts["test"][dec] += 1
                        stats[f"parsed_test_{dec}"] += 1
            else:
                stats["parse_fail"] += 1

            log.info("%s ✓ %s | %s | %s", tag, case_num, doc_type_cn,
                     publish_date or "?")
            stats["docs_found"] += 1
            existing_case_keys.add(case_key)

            if args.sleep > 0:
                time.sleep(args.sleep)

        # 每页结束：保存进度 + dataset
        site_state["current_page"] = page_num + 1
        save_progress(progress_path, progress)
        if parsed_results["test"]:
            _save_stage1_dataset_json(output_dir, parsed_results)

    site_state["current_page"] = page_num + 1 if page_num else 1
    site_state["done"] = True
    save_progress(progress_path, progress)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "12309_progress.json"
    progress = load_progress(progress_path)

    # 加载站点列表
    with args.sites_file.open("r", encoding="utf-8") as f:
        sites = json.load(f)
    if args.site:
        sites = [u for u in sites if urlparse(u).netloc == args.site]
        if not sites:
            log.error("--site %s 未在 %s 中出现", args.site, args.sites_file)
            return
    log.info("计划抓取 %d 个站点", len(sites))

    session = requests.Session()
    session.headers.update({"User-Agent": UA})

    # 已有 dataset 合并（保证 dataset.json 不丢旧记录）
    parsed_results: dict[str, list[dict]] = {"test": []}
    decision_counts: dict[str, dict[str, int]] = {"test": defaultdict(int)}
    seen_parse_keys: set[str] = set()

    def _merge_dataset_file(ds_path: Path, label: str) -> None:
        if not ds_path.exists():
            return
        with open(ds_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        merged = 0
        for rec in existing:
            dk = merge_dedup_key(rec)
            if dk in seen_parse_keys:
                continue
            seen_parse_keys.add(dk)
            parsed_results["test"].append(rec)
            dec = rec.get("decision", "")
            if dec:
                decision_counts["test"][dec] += 1
            merged += 1
        log.info("  [%s] 载入 %d 条解析记录（文件中共 %d 条）",
                 label, merged, len(existing))

    _merge_dataset_file(out_dir / "dataset.json", "dataset.json")
    _merge_dataset_file(out_dir / "test" / "dataset.json", "test/dataset.json")
    _merge_dataset_file(out_dir / "train" / "dataset.json", "train/dataset.json")

    existing_case_keys: set[str] = set(progress.get("found_documents", {}).keys())
    existing_case_keys.update(_load_dataset_case_keys(out_dir))

    stats: dict = defaultdict(int)
    try:
        for url in sites:
            process_site(
                url,
                args,
                session,
                progress,
                progress_path,
                out_dir,
                parsed_results,
                decision_counts,
                seen_parse_keys,
                existing_case_keys,
                stats,
            )
    except KeyboardInterrupt:
        log.info("\n⚠ 用户中断，保存进度...")
        save_progress(progress_path, progress)
        if parsed_results["test"]:
            _save_stage1_dataset_json(out_dir, parsed_results)
        raise

    save_progress(progress_path, progress)
    if parsed_results["test"]:
        _save_stage1_dataset_json(out_dir, parsed_results)

    log.info("")
    log.info("=" * 60)
    log.info("  完成")
    log.info("=" * 60)
    for k in sorted(stats.keys()):
        log.info("  %-24s %d", k, stats[k])
    log.info("  进度文件: %s", progress_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="通用检察院法律文书公开栏目爬虫",
    )
    parser.add_argument(
        "--sites-file",
        type=Path,
        default=Path("data/public_sources/Stage1/url_prefixes_list.json"),
        help="站点列表 JSON 文件",
    )
    parser.add_argument(
        "--site",
        default=None,
        help="只处理指定主机名（如 www.nmetuoke.jcy.gov.cn）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/public_sources/Stage1"),
        help="Stage1 output directory (dataset.json, prosecution/non-prosecution txt dirs, and 12309_progress.json)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        metavar="N",
        help="每站最多抓 N 页",
    )
    parser.add_argument("--timeout", type=int, default=25)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--no-dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.no_dry_run:
        args.dry_run = False
    run(args)


if __name__ == "__main__":
    main()



