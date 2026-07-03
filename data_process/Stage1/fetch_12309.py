#!/usr/bin/env python3
"""
fetch_12309.py

从 12309 中国检察网 (www.12309.gov.cn) 爬取法律文书，输出到 Stage1 目录。
通过搜索结果列表逐个访问文书详情页：支持 PDF 渲染页截获 PDF，以及 HTML 嵌套正文页直接抽取文本。

前置条件：
    pip install playwright pdfplumber
    start chrome --remote-debugging-port=9222

用法：
    # 测试（仅打印，不下载）
    python scripts/fetch_12309.py --max-pages 2 --dry-run

    # 正式爬取（默认一直翻到最后一页，可用 --max-pages 限制）
    python scripts/fetch_12309.py --no-dry-run

    # 断点续跑
    python scripts/fetch_12309.py --no-dry-run --resume

    # 已在浏览器中打开了搜索结果页，跳过自动搜索
    python scripts/fetch_12309.py --no-dry-run --skip-search

    # 换 --search 再爬时加 --resume 沿用 URL 去重；预设变了会自动把页码置 1。
    # --search：non_prosecution / prosecution
"""

import io
import json
import logging
import re
import sys
import time
import random
import argparse
from pathlib import Path
from collections import defaultdict

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from data_process.extract_documents import parse_document, UNKNOWN_WARNINGS
from data_process.Stage1.fetch_procuratorate_sites import identify_document, merge_dedup_key

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

SEARCH_URL = "https://www.12309.gov.cn/searchui/?siteCode=zgrmjcy#/12309/search"
_CDP_URL = "http://localhost:9222"

# CLI --search 预设（英文）→ 12309 搜索框实际输入（中文）
SEARCH_QUERY_PRESETS: dict[str, str] = {
    "non_prosecution": "不起诉决定书",
    "prosecution": "起诉书",
}

# 案号：支持 〔〕 [] 【】 以及中间可能有空格
CASE_NUM_RE = re.compile(
    r"[\u4e00-\u9fff]+\s*[〔\[【]\s*\d{4}\s*[〕\]】]\s*[A-Za-z]?\s*\d+\s*号"
)
DATE_RE = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")

# ---------------------------------------------------------------------------
# 浏览器管理（CDP 连接用户 Chrome）
# ---------------------------------------------------------------------------

_playwright_inst = None
_cdp_browser = None
_main_page = None


def _launch_browser():
    global _playwright_inst, _cdp_browser
    import atexit
    from playwright.sync_api import sync_playwright

    if _playwright_inst is None:
        _playwright_inst = sync_playwright().start()

        def _cleanup():
            try:
                if _cdp_browser:
                    _cdp_browser.close()
                _playwright_inst.stop()
            except Exception:
                pass
        atexit.register(_cleanup)

    if _cdp_browser is None:
        try:
            _cdp_browser = _playwright_inst.chromium.connect_over_cdp(_CDP_URL)
        except Exception:
            log.error("无法连接到 Chrome 调试端口 (%s)", _CDP_URL)
            log.error("请先关闭所有 Chrome 窗口，然后运行：")
            log.error("  start chrome --remote-debugging-port=9222")
            raise SystemExit(1)

    log.info("已连接到本机 Chrome")


def _get_main_page():
    """获取主页面（搜索结果页）。"""
    global _main_page
    if _cdp_browser is None:
        _launch_browser()
    if _main_page is not None:
        try:
            _main_page.url  # probe
            return _main_page
        except Exception:
            _main_page = None
    _main_page = _cdp_browser.contexts[0].new_page()
    return _main_page


def _close_browser():
    global _cdp_browser, _main_page
    for obj in [_main_page, _cdp_browser]:
        try:
            if obj:
                obj.close()
        except Exception:
            pass
    _cdp_browser = None
    _main_page = None


# ---------------------------------------------------------------------------
# PDF 下载 & 文本提取
# ---------------------------------------------------------------------------

def normalize_pdf_text(text: str) -> str:
    """将 pdfplumber 提取的文本规范化，修复 PDF 换行/分页导致的格式问题。

    处理：
    1. 去除页码行（"1"、"- 2 -" 等）
    2. 合并被换行打断的段落（如 "第一百七十\\n七条" → "第一百七十七条"）
    3. 修复数字与中文之间的多余空格（"2025 年" → "2025年"）
    """
    lines = text.split("\n")

    # ---- 1. 去除页码行 ----
    cleaned = []
    for line in lines:
        s = line.strip()
        if re.match(r"^-?\s*\d{1,4}\s*-?$", s) and len(s) < 10:
            continue
        cleaned.append(s)

    # ---- 2. 合并段落内换行 ----
    # 遇到段落起始标记或空行 → 新段落；否则拼接到当前段落
    _PARA_START = re.compile(
        r"^("
        r"[\u4e00-\u9fff]{2,20}人民检察院\s*$"       # 检察院名
        r"|不起诉决定书|起诉书"
        r"|[\u4e00-\u9fff]+\s*[〔\[【]\s*\d{4}"       # 案号
        r"|(?:\d+[.、]\s*)?(?:被不起诉人|被告人|犯罪嫌疑人|被不起诉单位)"
        r"|辩护人|指定辩护人"
        r"|本案由"
        r"|经本院|经依法|现依法|经我院|经审查|依法审查"
        r"|本院认为|本院审查认为|经本院审查"
        r"|被不起诉人如不服|被害人.*?如不服"
        r"|认定上述事实|证明上述"
        r"|综上|据此|此致"
        r"|\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日"  # 日期行
        r")"
    )

    paragraphs: list[str] = []
    current = ""

    for s in cleaned:
        if not s:  # 空行 → 段落边界
            if current:
                paragraphs.append(current)
                current = ""
            continue

        if _PARA_START.match(s):
            if current:
                paragraphs.append(current)
            current = s
        else:
            current = (current + s) if current else s

    if current:
        paragraphs.append(current)

    # ---- 3. 修复多余空格 ----
    result = "\n".join(paragraphs)
    # "2025 年 9 月" → "2025年9月"
    result = re.sub(r"(\d)\s+(年|月|日|时|分|秒|条|款|项|号|元)", r"\1\2", result)
    # "第 1 款" → "第1款"
    result = re.sub(r"(第)\s+(\d)", r"\1\2", result)
    # "120 、" → "120、"
    result = re.sub(r"(\d)\s+([、，。；])", r"\1\2", result)

    return result

def extract_text_from_pdf(pdf_bytes_list: list[bytes]) -> str | None:
    """从一组 PDF 字节流提取文本并拼接（多页文书可能有多个 PDF）。"""
    try:
        import pdfplumber
    except ImportError:
        log.error("请安装 pdfplumber：pip install pdfplumber")
        raise SystemExit(1)

    all_texts = []
    for pdf_bytes in pdf_bytes_list:
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for p in pdf.pages:
                    t = p.extract_text()
                    if t:
                        all_texts.append(t)
        except Exception as e:
            log.warning("  PDF 文本提取失败: %s", e)

    full = "\n".join(all_texts)
    return full if len(full) > 50 else None


# 格式一：正文直接嵌在 HTML（如 #article-body）；格式二：#pdf-container + pdf.js 拉取 PDF
_JS_EXTRACT_HTML_ARTICLE = r"""
() => {
    function clean(t) {
        if (!t) return "";
        return t.replace(/\r/g, "")
            .replace(/\u00a0/g, " ")
            .replace(/[ \t\f\v]+\n/g, "\n")
            .replace(/\n{3,}/g, "\n\n")
            .trim();
    }
    const selectors = ["#article-body", ".article-body", ".contbox", ".TRS_Editor", "#zoom"];
    let best = "";
    for (const sel of selectors) {
        const el = document.querySelector(sel);
        if (!el) continue;
        const t = clean(el.innerText || "");
        if (t.length > best.length) best = t;
    }
    return best;
}
"""


def fetch_document_from_detail(
    detail_url: str,
) -> tuple[list[bytes] | None, list[str] | None, str | None]:
    """
    在新标签页打开文书详情页，获取正文。

    12309 有两种常见详情形态：
    - PDF 版：页面含 #pdf-container，通过请求加载 PDF（沿用原截获逻辑）。
    - HTML 版：正文在 #article-body / .article-body 等节点内，无 PDF 请求。

    返回 (pdf_bytes 列表, pdf_url 列表, html 正文)。成功时 pdf 与 html 二选一非空。
    """
    if _cdp_browser is None:
        _launch_browser()

    context = _cdp_browser.contexts[0]
    page = context.new_page()
    pdf_list: list[dict] = []
    seen_urls: set = set()

    def _on_response(response):
        url = response.url
        # 匹配 PDF URL（minioweb 路径 或 .pdf 后缀）
        if (".pdf" in url.lower() or "/minioweb/" in url) and response.status == 200:
            if url in seen_urls:
                return
            seen_urls.add(url)
            try:
                body = response.body()
                if body and len(body) > 500:
                    pdf_list.append({"url": url, "body": body})
            except Exception:
                pdf_list.append({"url": url})

    page.on("response", _on_response)

    html_text: str | None = None
    bodies: list[bytes] = []
    urls: list[str] = []

    try:
        page.goto(detail_url, wait_until="domcontentloaded", timeout=30000)

        # 等 PDF canvas 渲染（最多 15 秒）
        try:
            page.wait_for_selector("#pdf-container canvas", timeout=15000)
        except Exception:
            pass

        # PDF 版页面多等一会；HTML 版无 #pdf-container，尽快进入正文抽取
        try:
            is_pdf_layout = page.evaluate(
                "() => !!document.querySelector('#pdf-container')"
            )
        except Exception:
            is_pdf_layout = False
        idle_no_pdf = 15 if is_pdf_layout else 5

        # 等待 PDF：已有 PDF 时连续 3 秒无新增即结束；一直无 PDF 则 idle_no_pdf 秒后放弃
        stable = 0
        last_count = 0
        for _ in range(45):
            time.sleep(1)
            if len(pdf_list) > last_count:
                last_count = len(pdf_list)
                stable = 0
            else:
                stable += 1
                if pdf_list and stable >= 3:
                    break
                if not pdf_list and stable >= idle_no_pdf:
                    break

        urls = [p["url"] for p in pdf_list]
        bodies = [p["body"] for p in pdf_list if "body" in p]

        # 有 URL 但未截获 body 时，用 requests 兜底下载（须在关闭页面前拿到 cookies）
        if urls and not bodies:
            import requests
            cookies = {c["name"]: c["value"] for c in context.cookies()}
            for pdf_url in urls:
                try:
                    resp = requests.get(
                        pdf_url,
                        headers={
                            "User-Agent": "Mozilla/5.0",
                            "Referer": "https://www.12309.gov.cn/",
                        },
                        cookies=cookies,
                        timeout=30,
                    )
                    if resp.status_code == 200 and len(resp.content) > 500:
                        bodies.append(resp.content)
                except Exception as e:
                    log.warning("  requests 下载 PDF 失败: %s", e)

        # 无可用 PDF 字节时尝试 HTML 正文（格式一）
        if not bodies:
            try:
                raw_html = page.evaluate(_JS_EXTRACT_HTML_ARTICLE)
                if isinstance(raw_html, str):
                    t = raw_html.strip()
                    if len(t) >= 200:
                        html_text = t
            except Exception as e:
                log.debug("  HTML 正文抽取异常: %s", e)

    except Exception as e:
        log.warning("  打开详情页失败: %s | %s", detail_url[:60], e)
    finally:
        try:
            page.remove_listener("response", _on_response)
            page.close()
        except Exception:
            pass

    if bodies:
        return bodies, urls, None
    if html_text:
        return None, urls if urls else [], html_text
    if not urls:
        return None, None, None
    return None, urls, None


# ---------------------------------------------------------------------------
# 搜索结果页操作
# ---------------------------------------------------------------------------

_JS_EXTRACT_LINKS = """
() => {
    const results = [];

    // 方式 1：从 .text-left-list-title 附近找 <a>
    const titles = document.querySelectorAll('.text-left-list-title');
    for (const el of titles) {
        let url = '';
        // title 内部的 <a>
        const innerA = el.querySelector('a[href]');
        if (innerA) { url = innerA.href; }
        // 向上找：父级可能是 <a>
        if (!url) {
            let p = el.parentElement;
            for (let i = 0; i < 3 && p; i++) {
                if (p.tagName === 'A' && p.href) { url = p.href; break; }
                const a = p.querySelector(':scope > a[href]');
                if (a) { url = a.href; break; }
                p = p.parentElement;
            }
        }
        const text = el.textContent.trim().slice(0, 120);
        results.push({ title: text, url: url });
    }

    // 方式 2：兜底——找所有指向 /12309/ 的链接
    if (results.length === 0 || results.every(r => !r.url)) {
        const allLinks = document.querySelectorAll('a[href*="/12309/"]');
        for (const a of allLinks) {
            const href = a.href;
            if (href.includes('.html') && !href.includes('searchui')) {
                results.push({ title: a.textContent.trim().slice(0, 120), url: href });
            }
        }
    }

    return results;
}
"""


def get_result_links(page) -> list[dict]:
    """从当前搜索结果页提取文书链接。"""
    try:
        raw = page.evaluate(_JS_EXTRACT_LINKS)
        # 去重、过滤
        seen = set()
        out = []
        for r in raw:
            url = r.get("url", "")
            if not url or url in seen:
                continue
            if "12309.gov.cn" not in url:
                continue
            seen.add(url)
            out.append(r)
        return out
    except Exception as e:
        log.warning("提取链接失败: %s", e)
        return []


def get_result_links_by_click(page) -> list[dict]:
    """
    无法通过 DOM 提取 URL 时，逐个点击标题获取详情页 URL。
    返回 [{title, url}, ...]。
    """
    results = []
    try:
        titles = page.locator(".text-left-list-title")
        count = titles.count()
    except Exception:
        return []

    for i in range(count):
        try:
            title_text = titles.nth(i).text_content().strip()[:120]
        except Exception:
            title_text = f"result_{i}"

        _human_delay(1.5, 3.0)

        try:
            # 尝试检测新标签页
            with page.context.expect_page(timeout=8000) as new_page_info:
                titles.nth(i).click()
            popup = new_page_info.value
            popup.wait_for_load_state("domcontentloaded", timeout=10000)
            url = popup.url
            popup.close()
            results.append({"title": title_text, "url": url})
        except Exception:
            # 没打开新标签——可能是同页跳转
            try:
                _human_delay(2, 4)
                url = page.url
                if url != SEARCH_URL and ".html" in url:
                    results.append({"title": title_text, "url": url})
                page.go_back(wait_until="domcontentloaded", timeout=10000)
                _human_delay(2, 3)
                page.wait_for_selector(".text-left-list-title", timeout=10000)
            except Exception as e:
                log.warning("  点击回退失败: %s", e)
                break

    return results


def click_next_page(page) -> bool:
    """点击下一页。返回 True 表示成功翻页。"""
    try:
        btn = page.locator("button.btn-next")
        if not btn.is_visible(timeout=3000):
            log.info("未找到下一页按钮")
            return False
        if btn.is_disabled():
            log.info("已到最后一页")
            return False
        _human_delay(1.5, 3.0)
        btn.click()
        _human_delay(3, 6)
        try:
            page.wait_for_load_state("networkidle", timeout=10000)
        except Exception:
            pass
        # 等新结果加载
        try:
            page.wait_for_selector(".text-left-list-title", timeout=10000)
        except Exception:
            pass
        return True
    except Exception as e:
        log.warning("翻页失败: %s", e)
        return False


def _human_delay(lo: float = 1.5, hi: float = 3.5):
    """模拟人类操作间隔。"""
    time.sleep(random.uniform(lo, hi))


def navigate_to_search(page, keyword: str):
    """打开搜索页，搜索关键词，并切换到「法律文书 → 标题」筛选。"""
    log.info("打开 12309 搜索页...")
    page.goto(SEARCH_URL, wait_until="networkidle", timeout=30000)
    _human_delay(3, 5)

    # 找搜索框并输入
    for sel in ["input.el-input__inner", ".search-input input", "input[type='text']"]:
        try:
            loc = page.locator(sel).first
            if loc.is_visible(timeout=3000):
                loc.click()
                _human_delay(0.5, 1.0)
                loc.type(keyword, delay=random.randint(80, 180))
                _human_delay(0.8, 1.5)
                page.keyboard.press("Enter")
                _human_delay(3, 5)
                try:
                    page.wait_for_selector(".text-left-list-title", timeout=15000)
                except Exception:
                    pass
                log.info("已搜索「%s」", keyword)
                break
        except Exception:
            continue
    else:
        log.error("找不到搜索框，请手动搜索后用 --skip-search 重试")
        raise SystemExit(1)

    # ---- 点击「法律文书」标签 ----
    try:
        tab = page.locator("p.content-nav-item2", has_text="法律文书")
        if tab.is_visible(timeout=5000):
            tab.click()
            log.info("已点击「法律文书」标签")
            _human_delay(2, 4)
            try:
                page.wait_for_selector(".text-left-list-title", timeout=10000)
            except Exception:
                pass
        else:
            log.warning("未找到「法律文书」标签，跳过")
    except Exception as e:
        log.warning("点击「法律文书」失败: %s", e)

    # ---- 点击「标题」单选按钮 ----
    try:
        # 找包含"标题"文本的 radio label
        radio = page.locator("label.el-radio", has_text="标题")
        if radio.is_visible(timeout=5000):
            radio.click()
            log.info("已选择「标题」筛选")
            _human_delay(2, 4)
            try:
                page.wait_for_selector(".text-left-list-title", timeout=10000)
            except Exception:
                pass
        else:
            log.warning("未找到「标题」单选按钮，跳过")
    except Exception as e:
        log.warning("点击「标题」失败: %s", e)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def normalize_case_num(s: str) -> str:
    return s.replace("〔", "[").replace("〕", "]")


def case_num_to_filename(case_num: str) -> str:
    s = case_num.replace("〔", "[").replace("〕", "]")
    return re.sub(r'[<>:"/\\|?*]', "_", s)


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


def _infer_doc_info(text: str, case_num: str) -> dict | None:
    """识别检察文书类型，失败时用标题关键词兜底。"""
    info = identify_document(text)
    if info:
        return info
    head = text[:2500]
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


def _has_quality_issues(parsed: dict) -> list[str]:
    return [w for w in parsed.get("_warnings", []) if w in UNKNOWN_WARNINGS]


def save_progress(path: Path, progress: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def _save_stage1_dataset_json(output_dir: Path, parsed_results: dict[str, list[dict]]) -> None:
    rows = parsed_results.get("test") or []
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = output_dir / "dataset.json.tmp"
    final_path = output_dir / "dataset.json"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    tmp_path.replace(final_path)


def run(args):
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "12309_cache"
    pdf_dir = cache_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    # ---- Resume ----
    progress_path = output_dir / "12309_progress.json"
    if args.resume and progress_path.exists():
        with open(progress_path, "r", encoding="utf-8") as f:
            progress = json.load(f)
        log.info("已加载进度: %d 个已爬文书", len(progress.get("found_documents", {})))
    else:
        progress = {
            "fetched_urls": {},
            "found_documents": {},
            "current_page": 1,
        }

    prev_search = progress.get("active_search")
    if prev_search is not None and prev_search != args.search:
        log.info("搜索预设与上次不同（%s → %s），页码重置为 1", prev_search, args.search)
        progress["current_page"] = 1
    progress["active_search"] = args.search

    # ---- 已有 dataset：始终加载并去重 ----
    parsed_results: dict[str, list[dict]] = {"test": []}
    decision_counts: dict[str, dict[str, int]] = {"test": defaultdict(int)}
    seen_parse_keys: set[str] = set()

    def _merge_dataset_file(ds_path: Path, label: str) -> None:
        if not ds_path.exists():
            return
        with open(ds_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        merged_n = 0
        for rec in existing:
            dk = merge_dedup_key(rec)
            if dk in seen_parse_keys:
                continue
            seen_parse_keys.add(dk)
            parsed_results["test"].append(rec)
            dec = rec.get("decision", "")
            if dec:
                decision_counts["test"][dec] += 1
            merged_n += 1
        log.info("  [%s] 载入 %d 条解析记录（文件中共 %d 条）",
                 label, merged_n, len(existing))

    _merge_dataset_file(output_dir / "test" / "dataset.json", "test/dataset.json")
    _merge_dataset_file(output_dir / "train" / "dataset.json", "train/dataset.json")
    _merge_dataset_file(output_dir / "dataset.json", "dataset.json")
    log.info("输出目录: %s（当前 dataset 合计 %d 条）", output_dir, len(parsed_results["test"]))

    # ---- 连接浏览器 ----
    page = _get_main_page()

    # ---- 爬取 ----
    stats = defaultdict(int)
    interrupted = False

    try:
        if not args.skip_search:
            log.info(
                "搜索：%s → 「%s」",
                args.search,
                args.search_query_zh,
            )
            navigate_to_search(page, args.search_query_zh)
        else:
            log.info(
                "跳过自动搜索，使用当前页结果（预设 %s / 「%s」）",
                args.search,
                args.search_query_zh,
            )

        resume_page = progress.get("current_page", 1)
        if args.resume and resume_page > 1:
            log.info("翻页到第 %d 页...", resume_page)
            for i in range(resume_page - 1):
                if not click_next_page(page):
                    log.warning(
                        "翻页到第 %d 页失败（在第 %d 页停下）",
                        resume_page,
                        i + 1,
                    )
                    break
                _human_delay(2, 4)

        page_num = resume_page
        while True:
            log.info("")
            log.info("=" * 50)
            log.info("  第 %d 页", page_num)
            log.info("=" * 50)

            # 等结果加载
            try:
                page.wait_for_selector(".text-left-list-title", timeout=10000)
            except Exception:
                log.warning("未找到搜索结果，停止")
                break

            _human_delay(2, 3)

            # 提取链接
            results = get_result_links(page)
            if not results:
                log.info("  DOM 方式未提取到链接，尝试点击方式...")
                results = get_result_links_by_click(page)
            if not results:
                log.warning("  本页无可用结果，停止")
                break

            log.info("  本页 %d 个结果", len(results))

            for ridx, result in enumerate(results):
                url = result["url"]
                title = result.get("title", "")[:60]
                tag = f"[页{page_num} #{ridx + 1}/{len(results)}]"

                if url in progress["fetched_urls"]:
                    stats["skipped"] += 1
                    continue

                log.info("%s %s", tag, title)

                if args.dry_run:
                    log.info("%s  [dry-run] %s", tag, url[:80])
                    stats["dry_run"] += 1
                    continue

                # ---- 获取正文（PDF 或 HTML 嵌套）----
                pdf_bytes_list, pdf_urls, html_text = fetch_document_from_detail(url)

                if html_text:
                    raw_text = html_text
                    log.info("%s  HTML 正文 %d 字", tag, len(raw_text))
                elif pdf_bytes_list:
                    total_kb = sum(len(b) for b in pdf_bytes_list) / 1024
                    log.info("%s  PDF %.1f KB (%d 个文件)", tag, total_kb, len(pdf_bytes_list))
                    raw_text = extract_text_from_pdf(pdf_bytes_list)
                else:
                    log.info("%s  ✗ 文书获取失败（无 PDF 且无 HTML 正文）", tag)
                    stats["pdf_failed"] += 1
                    progress["fetched_urls"][url] = "pdf_failed"
                    _human_delay(3, 6)
                    continue

                if not raw_text:
                    log.info("%s  ✗ 文本为空或 PDF 文本提取失败（可能是扫描件）", tag)
                    stats["text_failed"] += 1
                    progress["fetched_urls"][url] = "text_failed"
                    if pdf_bytes_list:
                        for i, b in enumerate(pdf_bytes_list):
                            (pdf_dir / f"_notext_{int(time.time())}_{i}.pdf").write_bytes(b)
                    _human_delay(3, 6)
                    continue

                # ---- 规范化文本（修复 PDF 换行/页码/空格）----
                text = normalize_pdf_text(raw_text)

                # ---- 识别案号 ----
                case_m = CASE_NUM_RE.search(text)
                if not case_m:
                    log.info("%s  ✗ 未找到案号 (len=%d)", tag, len(text))
                    log.info("%s    前300字: %s", tag, text[:300].replace("\n", "↵"))
                    # 保存文本到调试目录
                    debug_dir = output_dir / "_debug"
                    debug_dir.mkdir(exist_ok=True)
                    (debug_dir / f"nocase_{int(time.time())}_{ridx}.txt").write_text(
                        text, encoding="utf-8")
                    stats["no_case_num"] += 1
                    progress["fetched_urls"][url] = "no_case_num"
                    _human_delay(3, 6)
                    continue

                case_num = case_m.group()
                norm_case = normalize_case_num(case_num)

                # 去重
                if norm_case in progress["found_documents"]:
                    log.info("%s  ✗ 重复: %s", tag, case_num)
                    stats["duplicate"] += 1
                    progress["fetched_urls"][url] = "duplicate"
                    continue

                doc_date = extract_date_from_text(text)
                doc_info = _infer_doc_info(text, case_num)
                if not doc_info:
                    log.info("%s  ✗ 无法识别文书类型（起诉/不起诉），跳过", tag)
                    stats["unknown_doc_type"] += 1
                    progress["fetched_urls"][url] = "unknown_doc_type"
                    _human_delay(3, 6)
                    continue

                doc_type_cn = doc_info["doc_type"]
                doc_type_key = (
                    "prosecution" if doc_type_cn == "起诉" else "non_prosecution"
                )

                # ---- 保存 ----
                safe_name = case_num_to_filename(case_num)
                if pdf_bytes_list:
                    combined_pdf = b"".join(pdf_bytes_list)
                    (pdf_dir / f"{safe_name}.pdf").write_bytes(combined_pdf)
                split = "test"

                # Stage1 平铺输出：{起诉|不起诉}/案号.txt
                doc_dir = output_dir / doc_type_cn
                doc_dir.mkdir(parents=True, exist_ok=True)
                doc_path = doc_dir / f"{safe_name}.txt"
                doc_path.write_text(text, encoding="utf-8")

                # ---- 解析 ----
                parsed = parse_document(str(doc_path), doc_type_key)
                parsed_decision = None

                if parsed and parsed.get("decision") and parsed.get("fact"):
                    hit = _has_quality_issues(parsed)
                    if hit:
                        log.info("%s  → 解析质量不合格: %s", tag, ", ".join(hit))
                        stats["quality_fail"] += 1
                    else:
                        parsed.pop("_warnings", None)
                        parsed["source_url"] = url
                        parsed_decision = parsed["decision"]
                        pk = merge_dedup_key(parsed)
                        if pk not in seen_parse_keys:
                            seen_parse_keys.add(pk)
                            parsed_results[split].append(parsed)
                            decision_counts[split][parsed_decision] += 1
                            stats[f"parsed_{split}_{parsed_decision}"] += 1
                        else:
                            log.info("%s  → dataset 已存在同键记录，跳过追加", tag)
                else:
                    stats["parse_fail"] += 1

                log.info("%s  ✓ %s | %s | %s | %s",
                         tag, case_num, parsed_decision or "解析失败",
                         doc_date or "?", split)

                progress["found_documents"][norm_case] = {
                    "case_num": case_num,
                    "date": doc_date,
                    "source_url": url,
                    "pdf_url": pdf_urls or [],
                    "parsed_decision": parsed_decision,
                    "split": split,
                }
                progress["fetched_urls"][url] = "found"
                stats["docs_found"] += 1

                _human_delay(4, 8)

            # ---- 每页结束：保存进度 ----
            progress["current_page"] = page_num + 1
            save_progress(progress_path, progress)

            if parsed_results["test"]:
                _save_stage1_dataset_json(output_dir, parsed_results)

            # ---- 翻页 ----
            if args.max_pages is not None and (page_num - resume_page + 1) >= args.max_pages:
                log.info("已达 --max-pages=%d，停止", args.max_pages)
                break

            log.info("翻到下一页...")
            if not click_next_page(page):
                log.info("已到最后一页，停止")
                break
            _human_delay(4, 8)
            page_num += 1

    except KeyboardInterrupt:
        interrupted = True
        log.info("\n⚠ 用户中断，保存进度...")
    finally:
        save_progress(progress_path, progress)
        if parsed_results["test"]:
            _save_stage1_dataset_json(output_dir, parsed_results)
        if interrupted:
            log.info("进度已保存，下次用 --resume 继续。")
            _close_browser()
            return

    # ---- 汇总 ----
    log.info("")
    log.info("=" * 50)
    log.info("  爬取完成")
    log.info("=" * 50)
    log.info("  找到文书:      %d", stats["docs_found"])
    log.info("  PDF 下载失败:  %d", stats["pdf_failed"])
    log.info("  文本提取失败:  %d", stats["text_failed"])
    log.info("  无案号:        %d", stats["no_case_num"])
    log.info("  重复:          %d", stats["duplicate"])
    log.info("  解析失败:      %d", stats["parse_fail"])
    log.info("  文书类型不明:  %d", stats["unknown_doc_type"])
    log.info("  质量不合格:    %d", stats["quality_fail"])
    log.info("  跳过(已爬):    %d", stats["skipped"])
    log.info("-" * 50)
    log.info("  累计文书:      %d", len(progress.get("found_documents", {})))
    counts = decision_counts["test"]
    if sum(counts.values()):
        log.info("  [test] %s", dict(counts))
    log.info("=" * 50)
    log.info("  进度文件: %s", progress_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="从 12309 检察网爬取法律文书（起诉/不起诉等）")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/public_sources/Stage1"),
        help="Stage1 output directory (dataset.json and prosecution/non-prosecution txt directories)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        metavar="N",
        help="最多爬取 N 页后停止；省略则直到搜索结果最后一页",
    )
    parser.add_argument(
        "--search",
        default="non_prosecution",
        choices=tuple(SEARCH_QUERY_PRESETS.keys()),
        help="non_prosecution=不起诉决定书，prosecution=起诉书",
    )
    parser.add_argument("--skip-search", action="store_true",
                        help="跳过搜索（假设浏览器已在搜索结果页）")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--no-dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="从上次中断处继续")
    args = parser.parse_args()
    args.search_query_zh = SEARCH_QUERY_PRESETS[args.search]

    if args.no_dry_run:
        args.dry_run = False

    run(args)


if __name__ == "__main__":
    main()


