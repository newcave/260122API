# ======================================================
# ALIO 연구보고서 수집 + K-water 표준 A 요약 에이전트
# (A안: 내부 JSON API 기반)
# ======================================================

import os
from dataclasses import dataclass
from io import BytesIO
from typing import List, Dict, Optional

import requests
import streamlit as st
import pdfplumber
from pypdf import PdfReader
from openai import OpenAI

# ======================================================
# 기본 설정
# ======================================================
APP_TITLE = "ALIO 연구보고서 요약 에이전트 (K-water 표준 A)"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://alio.go.kr",
}

# ======================================================
# K-water 표준 A 프롬프트
# ======================================================
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
# 데이터 모델
# ======================================================
@dataclass
class ReportItem:
    title: str
    detail_url: str
    pdf_url: Optional[str]

# ======================================================
# ALIO 내부 API 접근 (A안 핵심)
# ======================================================
def fetch_alio_report_list(
    apba_id: str,
    report_form_root_no: str,
    page: int = 1,
    page_size: int = 50,
) -> List[Dict]:
    """
    ALIO 연구보고서 목록 JSON 호출
    ※ 실제 브라우저 Network 탭에서 확인되는 엔드포인트 패턴
    """

    api_url = "https://alio.go.kr/iris/api/report/list"

    payload = {
        "apbaId": apba_id,
        "reportFormRootNo": report_form_root_no,
        "pageIndex": page,
        "pageSize": page_size,
    }

    response = requests.post(api_url, json=payload, headers=HEADERS, timeout=15)
    response.raise_for_status()
    data = response.json()

    return data.get("list", [])


def extract_pdf_url(detail_api_url: str) -> Optional[str]:
    """
    상세 페이지 JSON에서 PDF 다운로드 URL 추출
    """
    response = requests.get(detail_api_url, headers=HEADERS, timeout=15)
    response.raise_for_status()
    data = response.json()

    for file in data.get("attachFiles", []):
        if file.get("fileExt", "").lower() == "pdf":
            return file.get("downloadUrl")

    return None


# ======================================================
# PDF 처리
# ======================================================
def download_pdf(url: str) -> bytes:
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.content


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        if text.strip():
            return text
    except Exception:
        pass

    reader = PdfReader(BytesIO(pdf_bytes))
    return "\n".join(p.extract_text() or "" for p in reader.pages)


def chunk_text(text: str, size: int = 6000, overlap: int = 400):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start = end - overlap if end < len(text) else end
    return chunks


# ======================================================
# OpenAI 요약
# ======================================================
def get_openai_client() -> OpenAI:
    key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=key)


def summarize_text(client: OpenAI, model: str, text: str) -> str:
    summaries = []

    for chunk in chunk_text(text):
        r = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": chunk},
            ],
        )
        summaries.append(r.output_text)

    combined = "\n".join(summaries)

    r = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": combined},
        ],
    )
    return r.output_text


# ======================================================
# Streamlit UI
# ======================================================
st.set_page_config(page_title=APP_TITLE, page_icon="💧", layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.header("ALIO 검색 설정")
    apba_id = st.text_input("기관 코드 (apbaId)", value="C0221")
    report_root = st.text_input("보고서 유형 코드", value="B1040")
    model = st.selectbox("모델", ["gpt-4o-mini", "gpt-4o"])

st.subheader("연구보고서 목록 조회")

if st.button("ALIO 연구보고서 조회"):
    try:
        items = fetch_alio_report_list(apba_id, report_root)
        st.session_state.items = items
        st.success(f"{len(items)}건 조회됨")
    except Exception as e:
        st.error(f"조회 실패: {e}")

if "items" in st.session_state:
    titles = [item.get("reportTitle") for item in st.session_state.items]
    idx = st.selectbox("보고서 선택", range(len(titles)), format_func=lambda i: titles[i])

    if st.button("PDF 다운로드 및 요약"):
        item = st.session_state.items[idx]

        try:
            pdf_url = extract_pdf_url(item["detailApiUrl"])
            pdf_bytes = download_pdf(pdf_url)
            text = extract_text_from_pdf(pdf_bytes)

            client = get_openai_client()
            with st.spinner("K-water 표준 A 요약 중..."):
                summary = summarize_text(client, model, text)

            st.markdown(summary)

        except Exception as e:
            st.error(f"처리 실패: {e}")
