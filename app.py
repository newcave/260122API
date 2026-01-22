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
APP_TITLE = "ALIO 연구보고서 요약 에이전트 (K-water 표준 A / B안 자동대응)"
BASE = "https://alio.go.kr"

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
    params: Dict[str, Any]
    list_key: str
    total_key: Optional[str]

@dataclass
class ReportCandidate:
    title: str
    org: str
    date: str
    detail_url: Optional[str]
    raw: Dict[str, Any]  # keep original for debugging / id extraction

# ======================================================
# Utils: HTTP
# ======================================================
def safe_get(url: str, params: Optional[Dict[str, Any]] = None, timeout: int = 15) -> requests.Response:
    r = requests.get(url, params=params, headers=HEADERS, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return r

def safe_post(url: str, json_body: Optional[Dict[str, Any]] = None, timeout: int = 15) -> requests.Response:
    r = requests.post(url, json=json_body, headers=HEADERS, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return r

def is_json_response(resp: requests.Response) -> bool:
    ct = (resp.headers.get("Content-Type") or "").lower()
    if "application/json" in ct:
        return True
    # Some servers mislabel; attempt json parsing
    try:
        resp.json()
        return True
    except Exception:
        return False

# ======================================================
# 1) B안 핵심: 목록 API 자동 탐색(프로빙)
# ======================================================
LIST_ENDPOINT_CANDIDATES = [
    # 가장 흔히 쓰이는 패턴(기관/유형/페이징)
    (f"{BASE}/iris/api/report/list.json", "GET"),
    (f"{BASE}/iris/api/report/list", "GET"),
    # 다른 시스템에서 흔한 이름 후보들(기관별/공시별 커스텀 케이스 대비)
    (f"{BASE}/iris/api/report/itemReportList.json", "GET"),
    (f"{BASE}/iris/api/report/itemReportListSusi.json", "GET"),
]

PARAM_SETS = [
    # 케이스 A: 질문에서 쓰신 파라미터명
    {"apbaId": None, "reportFormRootNo": None, "pageIndex": 1, "pageSize": 30},
    # 케이스 B: page/pageSize
    {"apbaId": None, "reportFormRootNo": None, "page": 1, "pageSize": 30},
    # 케이스 C: size/curPage 같은 변형
    {"apbaId": None, "reportFormRootNo": None, "curPage": 1, "pageSize": 30},
    {"apbaId": None, "reportFormRootNo": None, "pageNo": 1, "pageCnt": 30},
]

POSSIBLE_LIST_KEYS = ["list", "data", "result", "rows", "items"]
POSSIBLE_TOTAL_KEYS = ["totalCount", "total", "records", "count", "totCnt"]

def extract_list_from_json(data: Any) -> Tuple[Optional[List[Any]], Optional[str]]:
    """
    반환 JSON에서 리스트 후보 키를 찾아 실제 list를 뽑아냄
    """
    if isinstance(data, list):
        return data, "(root_list)"
    if not isinstance(data, dict):
        return None, None

    for k in POSSIBLE_LIST_KEYS:
        v = data.get(k)
        if isinstance(v, list):
            return v, k

    # 2-depth 탐색 (예: {"result": {"list":[...]}})
    for k in data.keys():
        v = data.get(k)
        if isinstance(v, dict):
            for kk in POSSIBLE_LIST_KEYS:
                vv = v.get(kk)
                if isinstance(vv, list):
                    return vv, f"{k}.{kk}"

    return None, None

def probe_report_list_api(apba_id: str, report_root: str) -> ListProbeResult:
    """
    여러 후보 endpoint/params를 시도해서 '실제로 동작하는' 목록 API 조합을 찾아 반환
    """
    last_err = None
    for endpoint, method in LIST_ENDPOINT_CANDIDATES:
        for base_params in PARAM_SETS:
            params = dict(base_params)
            params["apbaId"] = apba_id
            params["reportFormRootNo"] = report_root

            try:
                if method == "GET":
                    resp = safe_get(endpoint, params=params)
                else:
                    resp = safe_post(endpoint, json_body=params)

                if not is_json_response(resp):
                    continue

                data = resp.json()
                items, list_key = extract_list_from_json(data)
                if items is None or len(items) == 0:
                    # 리스트가 비어도 total이 있으면 성공일 수 있지만,
                    # 여기서는 "실제로 리스트 키를 찾았는가"를 우선
                    continue

                # total 키 추정
                total_key = None
                if isinstance(data, dict):
                    for tk in POSSIBLE_TOTAL_KEYS:
                        if tk in data:
                            total_key = tk
                            break
                    # 2-depth total
                    if total_key is None:
                        for k in data.keys():
                            if isinstance(data.get(k), dict):
                                for tk in POSSIBLE_TOTAL_KEYS:
                                    if tk in data[k]:
                                        total_key = f"{k}.{tk}"
                                        break

                return ListProbeResult(
                    endpoint=endpoint,
                    method=method,
                    params=params,
                    list_key=list_key,
                    total_key=total_key,
                )
            except Exception as e:
                last_err = e
                continue

    raise RuntimeError(f"목록 API 자동 탐색 실패 (마지막 에러: {last_err})")

def get_list_with_probe(probe: ListProbeResult, page: int = 1, page_size: int = 30) -> Dict[str, Any]:
    """
    탐색된 probe 조합으로 실제 목록을 가져온다 (페이지 반영)
    """
    params = dict(probe.params)
    # 페이지 키 자동 반영
    if "pageIndex" in params:
        params["pageIndex"] = page
    elif "page" in params:
        params["page"] = page
    elif "curPage" in params:
        params["curPage"] = page
    elif "pageNo" in params:
        params["pageNo"] = page

    if "pageSize" in params:
        params["pageSize"] = page_size
    elif "pageCnt" in params:
        params["pageCnt"] = page_size

    if probe.method == "GET":
        resp = safe_get(probe.endpoint, params=params)
    else:
        resp = safe_post(probe.endpoint, json_body=params)

    if not is_json_response(resp):
        raise RuntimeError("목록 API 응답이 JSON이 아닙니다.")
    return resp.json()

def normalize_candidates(list_json: Any) -> List[ReportCandidate]:
    """
    목록 JSON에서 '제목/기관/일자/상세링크'를 최대한 복원
    """
    items, _ = extract_list_from_json(list_json)
    if items is None:
        return []

    candidates: List[ReportCandidate] = []
    for it in items:
        if not isinstance(it, dict):
            continue

        title = (
            it.get("reportTitle")
            or it.get("rtitle")
            or it.get("title")
            or it.get("sj")
            or "(제목없음)"
        )
        org = (
            it.get("apbaNm")
            or it.get("orgNm")
            or it.get("instNm")
            or it.get("org")
            or ""
        )
        date = (
            it.get("regDate")
            or it.get("regDt")
            or it.get("pubDate")
            or it.get("publishDate")
            or it.get("ymd")
            or ""
        )

        # 상세 URL 후보
        detail_url = (
            it.get("detailUrl")
            or it.get("detailURL")
            or it.get("linkUrl")
            or it.get("url")
            or None
        )
        # 일부는 상대경로일 수 있어 join 처리
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
# 2) PDF 링크 추출: (A) JSON 상세 API 프로빙 → 실패시 (B) HTML 파싱
# ======================================================
DETAIL_ENDPOINT_CANDIDATES = [
    # 흔한 상세 패턴 후보들
    f"{BASE}/iris/api/report/detail.json",
    f"{BASE}/iris/api/report/detail",
    f"{BASE}/iris/api/report/view.json",
    f"{BASE}/iris/api/report/view",
]

DETAIL_ID_KEYS = ["reportNo", "reportSn", "rptNo", "id", "seq", "reportId", "reportFormNo", "reportRootNo"]

def guess_id_fields(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    목록 item에서 상세 조회에 쓸만한 ID 필드를 최대한 찾아 dict로 반환
    """
    found = {}
    for k in DETAIL_ID_KEYS:
        if k in item and item[k] not in (None, "", 0):
            found[k] = item[k]
    return found

def extract_pdf_from_detail_json(detail_json: Any) -> Optional[str]:
    """
    상세 JSON에서 PDF 다운로드 URL을 찾는다.
    """
    if not isinstance(detail_json, dict):
        return None

    # 흔한 첨부 구조들
    for key in ["attachFiles", "files", "fileList", "attachments"]:
        v = detail_json.get(key)
        if isinstance(v, list):
            for f in v:
                if not isinstance(f, dict):
                    continue
                ext = (f.get("fileExt") or f.get("ext") or "").lower()
                name = (f.get("fileNm") or f.get("name") or f.get("fileName") or "").lower()
                url = f.get("downloadUrl") or f.get("downUrl") or f.get("url")

                if url and isinstance(url, str):
                    if ext == "pdf" or name.endswith(".pdf") or ".pdf" in url.lower():
                        return urljoin(BASE, url) if url.startswith("/") else url

    # 어떤 경우는 단일 pdfUrl 필드가 있음
    for key in ["pdfUrl", "pdfURL", "downloadUrl"]:
        v = detail_json.get(key)
        if isinstance(v, str) and v:
            if ".pdf" in v.lower() or "filedown" in v.lower() or "download" in v.lower():
                return urljoin(BASE, v) if v.startswith("/") else v

    return None

def probe_detail_api_for_pdf(item: Dict[str, Any]) -> Optional[str]:
    """
    상세 API를 여러 후보 엔드포인트/파라미터로 시도해서 PDF URL을 얻는다.
    """
    id_fields = guess_id_fields(item)
    if not id_fields:
        return None

    # 가능한 파라미터 조합을 만든다:
    # 1) reportNo 우선, 2) id/seq 등 대체
    param_candidates: List[Dict[str, Any]] = []

    # 우선순위: reportNo 계열
    for k in ["reportNo", "reportSn", "rptNo", "reportId", "id", "seq"]:
        if k in id_fields:
            param_candidates.append({k: id_fields[k]})

    # 복합 파라미터 케이스 대비: apbaId + rootNo 등을 같이 넣는 경우
    # (목록 item에 들어있다면 함께)
    extra_keys = ["apbaId", "reportFormRootNo", "reportRootNo", "reportFormNo"]
    extras = {k: item.get(k) for k in extra_keys if item.get(k)}
    if extras:
        for base in list(param_candidates):
            merged = dict(extras)
            merged.update(base)
            param_candidates.append(merged)

    for endpoint in DETAIL_ENDPOINT_CANDIDATES:
        for params in param_candidates:
            try:
                resp = safe_get(endpoint, params=params)
                if not is_json_response(resp):
                    continue
                dj = resp.json()
                pdf = extract_pdf_from_detail_json(dj)
                if pdf:
                    return pdf
            except Exception:
                continue
    return None

def extract_pdf_links_from_detail_html(detail_url: str) -> List[str]:
    """
    상세 HTML에서 PDF/fileDown 링크를 파싱(2차 안전장치)
    """
    resp = safe_get(detail_url)
    html = resp.text
    soup = BeautifulSoup(html, "lxml")

    links: List[str] = []
    for a in soup.select("a[href]"):
        href = a.get("href", "")
        low = href.lower()
        if ".pdf" in low or "filedown" in low or "download" in low:
            links.append(urljoin(BASE, href) if href.startswith("/") else href)

    # JS에 숨겨진 fileDown 경로가 있는 케이스 대비: 정규식으로도 1회 스캔
    for m in re.findall(r'(https?://[^\s"\']+)', html):
        if ".pdf" in m.lower() or "filedown" in m.lower():
            links.append(m)

    # 중복 제거
    deduped = list(dict.fromkeys(links))
    return deduped

def pick_best_pdf_link(links: List[str]) -> Optional[str]:
    if not links:
        return None
    # 가장 그럴듯한 것 우선순위
    for l in links:
        if ".pdf" in l.lower():
            return l
    return links[0]

# ======================================================
# PDF text extraction
# ======================================================
def download_pdf_bytes(url: str, timeout: int = 25) -> bytes:
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.content

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    # 1) pdfplumber
    try:
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        text = "\n".join(pages).strip()
        if text:
            return text
    except Exception:
        pass

    # 2) pypdf fallback
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
# OpenAI summarization (new SDK)
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
    st.header("ALIO 검색 설정 (B안 자동대응)")
    apba_id = st.text_input("기관 코드 (apbaId)", value="C0221")
    report_root = st.text_input("보고서 유형 코드 (reportFormRootNo)", value="B1040")
    model = st.selectbox("모델", ["gpt-4o-mini", "gpt-4o"], index=0)
    page_size = st.slider("페이지 크기", 10, 50, 30, 5)
    st.divider()
    st.caption("※ B안은 여러 내부 API 후보를 자동으로 시도해 동작 조합을 찾아냅니다.")

# session state
if "probe" not in st.session_state:
    st.session_state.probe = None
if "candidates" not in st.session_state:
    st.session_state.candidates = []
if "last_debug" not in st.session_state:
    st.session_state.last_debug = {}

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("1) 목록 API 자동 탐색 + 목록 조회", type="primary"):
        try:
            probe = probe_report_list_api(apba_id, report_root)
            st.session_state.probe = probe

            list_json = get_list_with_probe(probe, page=1, page_size=page_size)
            candidates = normalize_candidates(list_json)
            st.session_state.candidates = candidates

            st.session_state.last_debug = {
                "chosen_endpoint": probe.endpoint,
                "method": probe.method,
                "params": probe.params,
                "list_key": probe.list_key,
                "total_key": probe.total_key,
            }

            if not candidates:
                st.warning("목록은 응답했지만 항목을 파싱하지 못했습니다. (스키마가 매우 특이한 케이스)")
            else:
                st.success(f"조회 성공: {len(candidates)}건")
        except Exception as e:
            st.error(f"조회 실패: {e}")

with col2:
    if st.session_state.last_debug:
        with st.expander("디버그: 자동 선택된 실제 동작 값(operating values)"):
            st.json(st.session_state.last_debug)

st.divider()
st.subheader("2) 보고서 선택 → PDF 추출(상세 JSON 프로빙 → 실패 시 HTML 파싱) → 요약")

if st.session_state.candidates:
    options = list(range(len(st.session_state.candidates)))
    idx = st.selectbox(
        "보고서 선택",
        options,
        format_func=lambda i: f"{st.session_state.candidates[i].title} ({st.session_state.candidates[i].date}) {st.session_state.candidates[i].org}",
    )

    cand: ReportCandidate = st.session_state.candidates[idx]

    with st.expander("선택 항목 원본(raw) 보기"):
        st.json(cand.raw)

    if st.button("PDF 추출 + K-water 표준 A 요약"):
        try:
            # 1) 상세 JSON API로 PDF 시도
            pdf_url = probe_detail_api_for_pdf(cand.raw)

            # 2) 실패하면 상세 HTML 파싱
            if not pdf_url:
                if not cand.detail_url:
                    raise RuntimeError("상세 URL이 없어 HTML 파싱도 불가합니다. (목록 JSON에 detailUrl이 없음)")
                links = extract_pdf_links_from_detail_html(cand.detail_url)
                pdf_url = pick_best_pdf_link(links)

            if not pdf_url:
                raise RuntimeError("PDF 링크를 찾지 못했습니다. (상세 JSON/HTML 모두에서 추출 실패)")

            st.info(f"PDF URL: {pdf_url}")

            pdf_bytes = download_pdf_bytes(pdf_url)
            text = extract_text_from_pdf(pdf_bytes).strip()
            if not text:
                st.warning("PDF에서 텍스트를 추출하지 못했습니다. (스캔본 가능성)")
                st.stop()

            client = get_openai_client()
            with st.spinner("요약 생성 중..."):
                summary = summarize_kwater_standard_a(client, model, text)

            st.markdown(summary)

            with st.expander("원문 텍스트 미리보기"):
                st.write(text[:1200])

        except Exception as e:
            st.error(f"처리 실패: {e}")
else:
    st.info("먼저 '목록 API 자동 탐색 + 목록 조회'를 실행하세요.")
