import importlib.util
import os
from dataclasses import dataclass
from io import BytesIO
from typing import Any, List, Optional
from urllib.parse import urljoin, urlparse

import pdfplumber
import requests
import streamlit as st
from bs4 import BeautifulSoup
from pypdf import PdfReader

APP_TITLE = "K-water 보고서 요약 에이전트"
SYSTEM_PROMPT = (
    "당신은 수자원 및 공공 정책 전문가입니다. 제공된 보고서의 핵심 내용, "
    "연구 목적, 결론을 요약하여 Markdown 형식으로 출력하세요."
)
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


@dataclass
class ReportSource:
    pdf_url: Optional[str]
    text: str


def fetch_html(url: str, timeout: int = 12) -> str:
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    response.raise_for_status()
    return response.text


def scrape_pdf_links(page_url: str) -> List[str]:
    html = fetch_html(page_url)
    soup = BeautifulSoup(html, "lxml")
    base_url = f"{urlparse(page_url).scheme}://{urlparse(page_url).netloc}"
    links = []
    for anchor in soup.select("a[href]"):
        href = anchor.get("href", "")
        lower_href = href.lower()
        if ".pdf" in lower_href or "filedown" in lower_href or "download" in lower_href:
            links.append(urljoin(base_url, href))
    deduped = list(dict.fromkeys(links))
    return deduped


def download_pdf(url: str, timeout: int = 20) -> bytes:
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    response.raise_for_status()
    return response.content


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        pages_text = [page.extract_text() or "" for page in pdf.pages]
    text = "\n".join(pages_text).strip()
    if text:
        return text
    reader = PdfReader(BytesIO(pdf_bytes))
    pages_text = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages_text).strip()


def chunk_text(text: str, max_chars: int = 6000, overlap: int = 400) -> List[str]:
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + max_chars, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if end == text_length:
            break
    return chunks


def summarize_text(client: Any, model: str, text: str) -> str:
    chunks = chunk_text(text)
    summaries = []
    for chunk in chunks:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": chunk},
            ],
        )
        summaries.append(response.choices[0].message.content.strip())
    if len(summaries) == 1:
        return summaries[0]
    combined = "\n\n".join(summaries)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": combined},
        ],
    )
    return response.choices[0].message.content.strip()


def get_openai_client(api_key: str) -> Any:
    if importlib.util.find_spec("openai") is None:
        st.error("OpenAI 라이브러리가 설치되지 않았습니다. requirements.txt를 확인하세요.")
        st.stop()
    from openai import OpenAI

    return OpenAI(api_key=api_key)


st.set_page_config(page_title=APP_TITLE, page_icon="💧", layout="wide")

with st.sidebar:
    st.header("설정")
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "")),
        help="Streamlit Cloud에서는 secrets.toml에 저장한 키를 자동으로 사용합니다.",
    )
    model = st.selectbox("모델", ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"], index=1)
    preview_limit = st.slider("텍스트 미리보기 길이", min_value=300, max_value=2000, value=800)

st.title(APP_TITLE)

st.subheader("보고서 입력")
url_input = st.text_input(
    "ALIO 게시글 URL",
    placeholder="https://alio.go.kr/item/itemDetail.do?...",
)
uploaded_pdf = st.file_uploader("PDF 파일 직접 업로드", type=["pdf"])

if "report_text" not in st.session_state:
    st.session_state.report_text = ""
if "report_source" not in st.session_state:
    st.session_state.report_source = None
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "pdf_links" not in st.session_state:
    st.session_state.pdf_links = []
if "scrape_warning" not in st.session_state:
    st.session_state.scrape_warning = ""

load_button = st.button("보고서 불러오기", type="primary")

if load_button:
    st.session_state.summary = ""
    st.session_state.report_text = ""
    st.session_state.report_source = None
    st.session_state.pdf_links = []
    st.session_state.scrape_warning = ""

    if not url_input and not uploaded_pdf:
        st.warning("URL 또는 PDF 파일을 입력해주세요.")
    else:
        if url_input:
            try:
                st.session_state.pdf_links = scrape_pdf_links(url_input)
                if not st.session_state.pdf_links:
                    st.session_state.scrape_warning = (
                        "스크래핑이 차단되었습니다. PDF를 직접 업로드해주세요."
                    )
            except requests.RequestException:
                st.session_state.scrape_warning = (
                    "스크래핑이 차단되었습니다. PDF를 직접 업로드해주세요."
                )
        if uploaded_pdf is not None:
            pdf_bytes = uploaded_pdf.read()
            try:
                report_text = extract_text_from_pdf(pdf_bytes)
            except Exception:
                st.error("PDF 파싱 오류가 발생했습니다. 다른 파일을 업로드해주세요.")
                report_text = ""
            if report_text:
                st.session_state.report_text = report_text
                st.session_state.report_source = ReportSource(
                    pdf_url="업로드된 파일",
                    text=report_text,
                )
            else:
                st.warning("PDF에서 텍스트를 추출하지 못했습니다. 스캔본 여부를 확인해주세요.")

if st.session_state.scrape_warning:
    st.warning(st.session_state.scrape_warning)

if st.session_state.pdf_links:
    selected_pdf = st.selectbox("발견된 PDF 링크", st.session_state.pdf_links)
    if st.button("선택한 PDF 불러오기"):
        try:
            pdf_bytes = download_pdf(selected_pdf)
            report_text = extract_text_from_pdf(pdf_bytes)
        except requests.RequestException:
            st.error("PDF 다운로드에 실패했습니다. PDF를 직접 업로드해주세요.")
            report_text = ""
        except Exception:
            st.error("PDF 파싱 오류가 발생했습니다. 다른 파일을 업로드해주세요.")
            report_text = ""
        if report_text:
            st.session_state.report_text = report_text
            st.session_state.report_source = ReportSource(
                pdf_url=selected_pdf,
                text=report_text,
            )
        else:
            st.warning("PDF에서 텍스트를 추출하지 못했습니다. 스캔본 여부를 확인해주세요.")

if st.session_state.report_source:
    st.success("보고서 로딩 완료")
    st.caption(f"사용한 소스: {st.session_state.report_source.pdf_url}")

st.divider()

st.subheader("요약")
if st.button("요약 생성"):
    if not api_key:
        st.warning("OpenAI API Key를 입력하세요.")
    elif not st.session_state.report_text:
        st.warning("먼저 보고서를 불러오세요.")
    else:
        try:
            client = get_openai_client(api_key)
            with st.spinner("요약 생성 중..."):
                st.session_state.summary = summarize_text(
                    client, model, st.session_state.report_text
                )
        except Exception:
            st.error("요약 생성 중 오류가 발생했습니다. API Key 또는 모델을 확인하세요.")

if st.session_state.summary:
    st.markdown(st.session_state.summary)

if st.session_state.report_text:
    with st.expander("원본 텍스트 미리보기"):
        st.write(st.session_state.report_text[:preview_limit])
