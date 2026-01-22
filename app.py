import os
import re
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import requests
import streamlit as st
from bs4 import BeautifulSoup
import pdfplumber
from pypdf import PdfReader
from openai import OpenAI

# ======================================================
# App Config
# ======================================================
APP_TITLE = "ALIO 연구보고서 요약 에이전트 (K-water 표준 A / 초보자용 진행 UI)"
BASE = "https://www.alio.go.kr"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": BASE,
    "Accept": "application/json,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

SYSTEM_PROMPT = """
당신은 한국수자원공사(K-water) 및 공공기관 연구보고서를 전문적으로 분석하는 정책·기술 전문가입니다.

아래 보고서를 'K-water 연구보고서 표준 A 요약 형식'에 맞춰 요약하십시오.

[출력 형식 — 반드시 준수]

## 1. 연구 배경 및 필요성
## 2. 연구 목적
## 3. 연구 범위 및 방법
## 4. 주요 연구 결과
## 5. 정책적·실무적 시사점
## 6. 결론 및 향후 과제
"""

# ======================================================
# Models
# ======================================================
@dataclass
class ListProbeResult:
    endpoint: str
    method: str
    payload: Dict[str, Any]
    list_key: str
    total_key: Optional[str]
    apba_type: Optional[str]

@dataclass
class ReportCandidate:
    title: str
    org: str
    date: str
    detail_url: Optional[str]
    raw: Dict[str, Any]

# ======================================================
# Progress UI helpers
# ======================================================
def ui_step(container, title: str, status: str, details: Optional[str] = None):
    """
    status: "todo" | "doing" | "done" | "fail"
    """
    icons = {"todo": "⬜", "doing": "🔄", "done": "✅", "fail": "❌"}
    with container:
        st.markdown(f"### {icons.get(status,'⬜')} {title}")
        if details:
            st.caption(details)

def set_progress(progress_bar, value: int, text: str, status_box=None):
    progress_bar.progress(value, text=text)
    if status_box is not None:
        status_box.info(text)

# ======================================================
# HTTP helpers
# ======================================================
def safe_get(url: str, params: Optional[Dict[str, Any]] = None, timeout: int = 15) -> requests.Response:
    r = requests.get(url, params=params, headers=HEADERS, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return r

def safe_post(url: str, data: Optional[Dict[str, Any]] = None, json_body: Optional[Dict[str, Any]] = None, timeout: int = 15) -> requests.Response:
    r = requests.post(url, data=data, json=json_body, headers=HEADERS, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return r

def is_json_response(resp: requests.Response) -> bool:
    ct = (resp.headers.get("Content-Type") or "").lower()
    if "application/json" in ct:
        return True
    try:
        resp.json()
        return True
    except Exception:
        return False

# ======================================================
# 0) apbaType 자동 추출
# ======================================================
def fetch_apba_type(apba_id: str, report_form_root_no: str) -> Optional[str]:
    url = f"{BASE}/item/itemOrganList.do"
    params = {"apbaId": apba_id, "reportFormRootNo": report_form_root_no}
    r = safe_get(url, params=params)
    html = r.text

    patterns = [
        r"apbaType\s*[:=]\s*['\"]?([A-Za-z0-9]+)['\"]?",
        r"name=['\"]apbaType['\"][^>]*value=['\"]([^'\"]+)['\"]",
        r"['\"]apbaType['\"]\s*,\s*['\"]([^'\"]+)['\"]",
    ]
    for p in patterns:
        m = re.search(p, html)
        if m:
            return m.group(1)
    return None

# ======================================================
# 1) 목록 API 자동 탐색
# ======================================================
LIST_ENDPOINT_CANDIDATES = [
    (f"{BASE}/item/itemReportListSusi.json", "POST"),
    (f"{BASE}/item/itemReportList.json", "POST"),
]

PAYLOAD_SETS = [
    {"apbaId": None, "apbaType": None, "reportFormRootNo": None, "pageNo": 1, "pageCnt": 30},
    {"apbaId": None, "apbaType": None, "reportFormRootNo": None, "pageIndex": 1, "pageSize": 30},
    {"apbaId": None, "apbaType": None, "reportFormRootNo": None, "curPage": 1, "pageSize": 30},
]

POSSIBLE_LIST_KEYS = ["list", "data", "result", "rows", "items"]
POSSIBLE_TOTAL_KEYS = ["totalCount", "total", "records", "count", "totCnt"]

def extract_list_from_json(data: Any) -> Tuple[Optional[List[Any]], Optional[str]]:
    if isinstance(data, list):
        return data, "(root_list)"
    if not isinstance(data, dict):
        return None, None

    for k in POSSIBLE_LIST_KEYS:
        v = data.get(k)
        if isinstance(v, list):
            return v, k

    for k, v in data.items():
        if isinstance(v, dict):
            for kk in POSSIBLE_LIST_KEYS:
                vv = v.get(kk)
                if isinstance(vv, list):
                    return vv, f"{k}.{kk}"

    return None, None

def guess_total_key(data: Any) -> Optional[str]:
    if not isinstance(data, dict):
        return None
    for tk in POSSIBLE_TOTAL_KEYS:
        if tk in data:
            return tk
    for k, v in data.items():
        if isinstance(v, dict):
            for tk in POSSIBLE_TOTAL_KEYS:
                if tk in v:
                    return f"{k}.{tk}"
    return None

def probe_report_list_api(apba_id: str, report_root: str, page_size: int) -> ListProbeResult:
    apba_type = fetch_apba_type(apba_id, report_root)
    apba_type_candidates = [apba_type] if apba_type else []
    apba_type_candidates += ["1", "2", "A", "B"]

    last_err: Optional[Exception] = None

    for endpoint, method in LIST_ENDPOINT_CANDIDATES:
        for base_payload in PAYLOAD_SETS:
            for apba_type_try in apba_type_candidates:
                payload = dict(base_payload)
                payload["apbaId"] = apba_id
                payload["reportFormRootNo"] = report_root
                payload["apbaType"] = apba_type_try

                if "pageCnt" in payload:
                    payload["pageCnt"] = page_size
                if "pageSize" in payload:
                    payload["pageSize"] = page_size

                try:
                    resp = safe_post(endpoint, data=payload) if method == "POST" else safe_get(endpoint, params=payload)
                    if not is_json_response(resp):
                        continue

                    data = resp.json()
                    items, list_key = extract_list_from_json(data)

                    if not items:
                        continue

                    return ListProbeResult(
                        endpoint=endpoint,
                        method=method,
                        payload=payload,
                        list_key=list_key or "",
                        total_key=guess_total_key(data),
                        apba_type=apba_type_try,
                    )
                except Exception as e:
                    last_err = e
                    continue

    raise RuntimeError(f"목록 API 자동 탐색 실패 (마지막 에러: {last_err})")

def fetch_list_with_probe(probe: ListProbeResult, page: int, page_size: int) -> Any:
    payload = dict(probe.payload)

    if "pageNo" in payload:
        payload["pageNo"] = page
    elif "pageIndex" in payload:
        payload["pageIndex"] = page
    elif "curPage" in payload:
        payload["curPage"] = page

    if "pageCnt" in payload:
        payload["pageCnt"] = page_size
    if "pageSize" in payload:
        payload["pageSize"] = page_size

    resp = safe_post(probe.endpoint, data=payload) if probe.method == "POST" else safe_get(probe.endpoint, params=payload)
    if not is_json_response(resp):
        raise RuntimeError("목록 API 응답이 JSON이 아닙니다.")
    return resp.json()

def normalize_candidates(list_json: Any) -> List[ReportCandidate]:
    items, _ = extract_list_from_json(list_json)
    if not items:
        return []

    candidates: List[ReportCandidate] = []
    for it in items:
        if not isinstance(it, dict):
            continue

        title = it.get("reportTitle") or it.get("rtitle") or it.get("title") or it.get("sj") or it.get("reportSj") or "(제목없음)"
        org = it.get("apbaNm") or it.get("orgNm") or it.get("instNm") or it.get("org") or it.get("apbaName") or ""
        date = it.get("regDate") or it.get("regDt") or it.get("pubDate") or it.get("publishDate") or it.get("ymd") or it.get("wrtDt") or ""

        detail_url = it.get("detailUrl") or it.get("detailURL") or it.get("linkUrl") or it.get("url")
        if not detail_url:
            rid = it.get("reportNo") or it.get("reportSn") or it.get("rptNo") or it.get("id") or it.get("seq")
            if rid:
                detail_url = f"{BASE}/item/itemDetail.do?reportNo={rid}"

        if isinstance(detail_url, str) and detail_url.startswith("/"):
            detail_url = urljoin(BASE, detail_url)

        candidates.append(
            ReportCandidate(
                title=str(title).strip(),
                org=str(org).strip(),
                date=str(date).strip(),
                detail_url=detail_url,
                raw=it,
            )
        )
    return candidates

# ======================================================
# 2) PDF 링크 추출(상세 JSON 후보 → 실패 시 상세 HTML 파싱)
# ======================================================
DETAIL_ENDPOINT_CANDIDATES = [
    f"{BASE}/item/itemReportDetail.json",
    f"{BASE}/item/itemReportView.json",
    f"{BASE}/iris/api/report/detail.json",
    f"{BASE}/iris/api/report/detail",
]

DETAIL_ID_KEYS = ["reportNo", "reportSn", "rptNo", "id", "seq", "reportId", "reportFormNo", "reportRootNo"]

def guess_id_fields(item: Dict[str, Any]) -> Dict[str, Any]:
    found = {}
    for k in DETAIL_ID_KEYS:
        if k in item and item[k] not in (None, "", 0):
            found[k] = item[k]
    return found

def extract_pdf_from_detail_json(detail_json: Any) -> Optional[str]:
    if not isinstance(detail_json, dict):
        return None

    for key in ["attachFiles", "files", "fileList", "attachments"]:
        v = detail_json.get(key)
        if isinstance(v, list):
            for f in v:
                if not isinstance(f, dict):
                    continue
                ext = (f.get("fileExt") or f.get("ext") or "").lower()
                name = (f.get("fileNm") or f.get("name") or f.get("fileName") or "").lower()
                url = f.get("downloadUrl") or f.get("downUrl") or f.get("url")
                if isinstance(url, str) and url:
                    if ext == "pdf" or name.endswith(".pdf") or ".pdf" in url.lower():
                        return urljoin(BASE, url) if url.startswith("/") else url

    for key in ["pdfUrl", "pdfURL", "downloadUrl", "downUrl", "url"]:
        v = detail_json.get(key)
        if isinstance(v, str) and v and (".pdf" in v.lower() or "filedown" in v.lower() or "download" in v.lower()):
            return urljoin(BASE, v) if v.startswith("/") else v

    return None

def probe_detail_api_for_pdf(item: Dict[str, Any], apba_id: str, report_root: str, apba_type: Optional[str]) -> Optional[str]:
    id_fields = guess_id_fields(item)
    if not id_fields:
        return None

    param_candidates: List[Dict[str, Any]] = []
    for k in ["reportNo", "reportSn", "rptNo", "reportId", "id", "seq"]:
        if k in id_fields:
            param_candidates.append({k: id_fields[k]})

    extras = {}
    if apba_id:
        extras["apbaId"] = apba_id
    if report_root:
        extras["reportFormRootNo"] = report_root
    if apba_type:
        extras["apbaType"] = apba_type

    if extras:
        for base in list(param_candidates):
            merged = dict(extras)
            merged.update(base)
            param_candidates.append(merged)

    for endpoint in DETAIL_ENDPOINT_CANDIDATES:
        for params in param_candidates:
            try:
                resp = safe_get(endpoint, params=params)
                if is_json_response(resp):
                    pdf = extract_pdf_from_detail_json(resp.json())
                    if pdf:
                        return pdf
                resp2 = safe_post(endpoint, data=params)
                if is_json_response(resp2):
                    pdf2 = extract_pdf_from_detail_json(resp2.json())
                    if pdf2:
                        return pdf2
            except Exception:
                continue

    return None

def extract_pdf_links_from_detail_html(detail_url: str) -> List[str]:
    resp = safe_get(detail_url)
    html = resp.text
    soup = BeautifulSoup(html, "lxml")

    links: List[str] = []
    for a in soup.select("a[href]"):
        href = a.get("href", "")
        low = href.lower()
        if ".pdf" in low or "filedown" in low or "download" in low:
            links.append(urljoin(BASE, href) if href.startswith("/") else href)

    for m in re.findall(r'(https?://[^\s"\']+)', html):
        if ".pdf" in m.lower() or "filedown" in m.lower() or "download" in m.lower():
            links.append(m)

    for m in re.findall(r'(/download/[^"\']+)', html):
        if "fileNo=" in m or "download" in m.lower():
            links.append(urljoin(BASE, m))

    return list(dict.fromkeys(links))

def pick_best_pdf_link(links: List[str]) -> Optional[str]:
    if not links:
        return None
    for l in links:
        if ".pdf" in l.lower():
            return l
    return links[0]

# ======================================================
# PDF extraction
# ======================================================
def download_pdf_bytes(url: str, timeout: int = 25) -> bytes:
    r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return r.content

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        text = "\n".join(pages).strip()
        if text:
            return text
    except Exception:
        pass

    reader = PdfReader(BytesIO(pdf_bytes))
    pages = [p.extract_text() or "" for p in reader.pages]
    return "\n".join(pages).strip()

def chunk_text(text: str, max_chars: int = 6000, overlap: int = 400) -> List[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunks.append(text[start:end])
        start = end - overlap if end < n else end
    return chunks

# ======================================================
# OpenAI (new SDK)
# ======================================================
def get_openai_client() -> OpenAI:
    key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not found in secrets/env")
    return OpenAI(api_key=key)

def summarize_kwater_standard_a(client: OpenAI, model: str, text: str) -> str:
    partial = []
    for chunk in chunk_text(text):
        r = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": chunk},
            ],
        )
        partial.append(r.output_text.strip())

    if len(partial) == 1:
        return partial[0]

    combined = "\n\n".join(partial)
    r = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": combined},
        ],
    )
    return r.output_text.strip()

# ======================================================
# Streamlit UI
# ======================================================
st.set_page_config(page_title=APP_TITLE, page_icon="💧", layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.header("ALIO 검색 설정")
    apba_id = st.text_input("기관 코드 (apbaId)", value="C0221")
    report_root = st.text_input("보고서 유형 코드 (reportFormRootNo)", value="B1040")
    model = st.selectbox("모델", ["gpt-4o-mini", "gpt-4o"], index=0)
    page_size = st.slider("페이지 크기", 10, 50, 30, 5)
    beginner_view = st.checkbox("초보자 모드(중간 확인 패널 표시)", value=True)

# session state
if "probe" not in st.session_state:
    st.session_state.probe = None
if "candidates" not in st.session_state:
    st.session_state.candidates = []
if "debug" not in st.session_state:
    st.session_state.debug = {}
if "preview" not in st.session_state:
    st.session_state.preview = {}

st.divider()

# ======================================================
# Step 1: list
# ======================================================
st.subheader("1) 목록 조회 (진행 상황 표시)")
progress = st.progress(0, text="대기 중")
status_box = st.empty()
steps_panel = st.container()

if st.button("목록 API 자동 탐색 + 목록 조회", type="primary"):
    try:
        # Step A
        set_progress(progress, 10, "Step A: apbaType 추출 중...", status_box)
        apba_type = fetch_apba_type(apba_id, report_root)
        if beginner_view:
            st.session_state.preview["apbaType_extracted"] = apba_type

        ui_step(steps_panel, "Step A: apbaType 추출", "done" if apba_type else "doing",
                f"추출값: {apba_type if apba_type else '(미검출 - fallback 시도)'}")

        # Step B
        set_progress(progress, 35, "Step B: 목록 API 후보 프로빙 중...", status_box)
        probe = probe_report_list_api(apba_id, report_root, page_size=page_size)
        st.session_state.probe = probe

        st.session_state.debug = {
            "chosen_endpoint": probe.endpoint,
            "method": probe.method,
            "payload_used": probe.payload,
            "apbaType_used": probe.apba_type,
            "list_key": probe.list_key,
            "total_key": probe.total_key,
        }
        ui_step(steps_panel, "Step B: 목록 API 자동 선택", "done",
                f"endpoint={probe.endpoint} / method={probe.method} / apbaType={probe.apba_type}")

        # Step C
        set_progress(progress, 60, "Step C: 목록 데이터 가져오는 중...", status_box)
        list_json = fetch_list_with_probe(probe, page=1, page_size=page_size)
        candidates = normalize_candidates(list_json)
        st.session_state.candidates = candidates

        ui_step(steps_panel, "Step C: 목록 파싱", "done",
                f"파싱된 항목 수: {len(candidates)}")

        # Step D (preview)
        set_progress(progress, 85, "Step D: 사용자 미리보기 구성 중...", status_box)
        if beginner_view and candidates:
            st.session_state.preview["top3"] = [
                {"title": c.title, "date": c.date, "org": c.org} for c in candidates[:3]
            ]
        ui_step(steps_panel, "Step D: 미리보기", "done", "상위 3건 미리보기 준비 완료")

        set_progress(progress, 100, "목록 조회 완료", status_box)
        st.success(f"목록 조회 성공: {len(candidates)}건")

    except Exception as e:
        ui_step(steps_panel, "실패", "fail", str(e))
        set_progress(progress, 100, "실패", status_box)
        st.error(f"목록 조회 실패: {e}")

# beginner preview panel
if beginner_view and st.session_state.debug:
    with st.expander("🔎 (초보자 모드) 중간 확인 패널: 목록 단계 결과"):
        st.write("**자동 선택된 동작 값(operating values)**")
        st.json(st.session_state.debug)

        apba_type_extracted = st.session_state.preview.get("apbaType_extracted")
        st.write("**apbaType 추출 결과**:", apba_type_extracted if apba_type_extracted else "미검출(후보값으로 시도)")

        top3 = st.session_state.preview.get("top3", [])
        if top3:
            st.write("**목록 상위 3건 미리보기**")
            st.table(top3)

st.divider()

# ======================================================
# Step 2: PDF + summary
# ======================================================
st.subheader("2) 보고서 선택 → PDF 추출 → 요약 (진행 상황 표시)")

if not st.session_state.candidates:
    st.info("먼저 1) 목록 조회를 실행하세요.")
else:
    options = list(range(len(st.session_state.candidates)))
    idx = st.selectbox(
        "보고서 선택",
        options,
        format_func=lambda i: f"{st.session_state.candidates[i].title} ({st.session_state.candidates[i].date}) {st.session_state.candidates[i].org}",
    )
    cand: ReportCandidate = st.session_state.candidates[idx]

    if beginner_view:
        with st.expander("📦 (초보자 모드) 선택 항목 원본 데이터 보기"):
            st.json(cand.raw)

    progress2 = st.progress(0, text="대기 중")
    status_box2 = st.empty()
    steps_panel2 = st.container()

    if st.button("PDF 추출 + K-water 표준 A 요약"):
        try:
            probe: ListProbeResult = st.session_state.probe
            if not probe:
                raise RuntimeError("먼저 1) 목록 조회를 실행하세요.")

            set_progress(progress2, 15, "Step 1: 상세 JSON 프로빙으로 PDF 링크 찾는 중...", status_box2)
            ui_step(steps_panel2, "Step 1: 상세 JSON 프로빙", "doing")

            pdf_url = probe_detail_api_for_pdf(
                cand.raw,
                apba_id=apba_id,
                report_root=report_root,
                apba_type=probe.apba_type,
            )

            if pdf_url:
                ui_step(steps_panel2, "Step 1: 상세 JSON 프로빙", "done", f"PDF 링크 발견")
            else:
                ui_step(steps_panel2, "Step 1: 상세 JSON 프로빙", "done", "실패 → HTML 파싱으로 fallback")

            set_progress(progress2, 35, "Step 2: (필요 시) 상세 HTML 파싱으로 PDF 링크 찾는 중...", status_box2)
            if not pdf_url:
                if not cand.detail_url:
                    raise RuntimeError("상세 URL이 없어 HTML 파싱도 불가합니다. (detailUrl/reportNo 부재)")
                links = extract_pdf_links_from_detail_html(cand.detail_url)
                if beginner_view:
                    st.session_state.preview["pdf_links"] = links[:10]
                pdf_url = pick_best_pdf_link(links)

            if not pdf_url:
                raise RuntimeError("PDF 링크를 찾지 못했습니다. (상세 JSON/HTML 모두 실패)")

            ui_step(steps_panel2, "Step 2: PDF 링크 확정", "done", pdf_url)
            st.info(f"PDF URL: {pdf_url}")

            set_progress(progress2, 55, "Step 3: PDF 다운로드 중...", status_box2)
            ui_step(steps_panel2, "Step 3: PDF 다운로드", "doing")
            pdf_bytes = download_pdf_bytes(pdf_url)
            ui_step(steps_panel2, "Step 3: PDF 다운로드", "done", f"{len(pdf_bytes):,} bytes")

            set_progress(progress2, 70, "Step 4: PDF 텍스트 추출 중...", status_box2)
            ui_step(steps_panel2, "Step 4: 텍스트 추출", "doing")
            text = extract_text_from_pdf(pdf_bytes).strip()
            if not text:
                raise RuntimeError("PDF에서 텍스트를 추출하지 못했습니다. (스캔본 가능성)")
            ui_step(steps_panel2, "Step 4: 텍스트 추출", "done", f"추출 길이: {len(text):,} chars")

            set_progress(progress2, 85, "Step 5: OpenAI 요약 생성 중...", status_box2)
            ui_step(steps_panel2, "Step 5: 요약 생성", "doing")
            client = get_openai_client()
            summary = summarize_kwater_standard_a(client, model, text)
            ui_step(steps_panel2, "Step 5: 요약 생성", "done")

            set_progress(progress2, 100, "완료", status_box2)
            st.markdown(summary)

            if beginner_view:
                with st.expander("🔎 (초보자 모드) 중간 확인 패널: PDF 링크 후보/원문 미리보기"):
                    links_preview = st.session_state.preview.get("pdf_links", [])
                    if links_preview:
                        st.write("**HTML 파싱에서 찾은 링크 후보(최대 10개)**")
                        st.write(links_preview)
                    st.write("**원문 텍스트 일부(1200자)**")
                    st.write(text[:1200])

        except Exception as e:
            ui_step(steps_panel2, "실패", "fail", str(e))
            set_progress(progress2, 100, "실패", status_box2)
            st.error(f"처리 실패: {e}")
