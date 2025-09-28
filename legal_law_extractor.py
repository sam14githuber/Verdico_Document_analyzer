"""
legal_law_extractor.py
Jupyter-friendly + Streamlit app that: upload a case document -> returns laws/sections likely relevant.

Usage:
  # Streamlit:
  streamlit run legal_law_extractor.py

  # Jupyter:
  from legal_law_extractor import extract_laws_from_text, summarize_case_text
  text = open("sample_case.txt").read()
  summary = summarize_case_text(text)
  laws = extract_laws_from_text(text)
"""

import os
import io
import re
import textwrap
from typing import List, Dict, Tuple, Any

# Document libs
from docx import Document
from PyPDF2 import PdfReader

# GPT client
import google.generativeai as genai

# Streamlit (optional UI)
try:
    import streamlit as st
except Exception:
    st = None


# -------------------------
# Configure Gemini
# -------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDcJXTc_FM2sNqfWrvCrYYsAPKssCPl1AQ")
if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
    # Not failing here to keep Jupyter friendly; functions will error if key not set when calling Gemini.
    pass
genai.configure(api_key=GEMINI_API_KEY)

# Recommended model (change if you have access to other models)
MODEL_NAME = "gemini-2.0-flash"
model = genai.GenerativeModel(MODEL_NAME)


# -------------------------
# Document reading helpers
# -------------------------
def read_pdf_bytes(file_bytes: bytes) -> str:
    """Extract text from PDF bytes."""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
        return "\n".join(text_parts)
    except Exception as e:
        return f""  # return empty if cannot parse


def read_docx_bytes(file_bytes: bytes) -> str:
    """Extract text from docx bytes."""
    try:
        doc = Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)
    except Exception:
        return ""


def read_txt_bytes(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8", errors="replace")
    except Exception:
        return ""


def extract_text_from_upload(uploaded_file) -> str:
    """Streamlit file uploader object or a file-like with .read()"""
    if uploaded_file is None:
        return ""
    raw = uploaded_file.read()
    name = getattr(uploaded_file, "name", "") or ""
    if name.lower().endswith(".pdf") or getattr(uploaded_file, "type", "") == "application/pdf":
        return read_pdf_bytes(raw)
    if name.lower().endswith(".docx") or "word" in getattr(uploaded_file, "type", ""):
        return read_docx_bytes(raw)
    # fallback txt
    return read_txt_bytes(raw)


# -------------------------
# Text chunking (for long docs)
# -------------------------
def chunk_text(text: str, max_chars: int = 4000) -> List[str]:
    """
    Break text into chunks with approximate max_chars.
    Keeps paragraphs intact where possible.
    """
    if not text:
        return []
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    cur = []
    cur_len = 0
    for p in paragraphs:
        if cur_len + len(p) + 1 > max_chars and cur:
            chunks.append("\n\n".join(cur))
            cur = [p]
            cur_len = len(p)
        else:
            cur.append(p)
            cur_len += len(p) + 2
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


# -------------------------
# Gemini interaction helpers
# -------------------------
def call_gemini(prompt: str, temperature: float = 0.0, max_output_tokens: int = 1500) -> str:
    """
    Single-call helper to Gemini model.
    Returns text response or raises exception if api key missing/invalid.
    """
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
        raise RuntimeError("GEMINI_API_KEY is not configured. Set the GEMINI_API_KEY environment variable.")

    # For the google.generativeai library, we call generate_content with a prompt string.
    response = model.generate_content(prompt)
    # model.generate_content may return an object - try to access .text
    # The library may also return other structured data depending on version.
    if hasattr(response, "text"):
        return response.text
    # fallback: str(response)
    return str(response)


# -------------------------
# Main functions: summarize & extract laws
# -------------------------
def summarize_case_text(text: str) -> str:
    """
    Create a concise factual summary of the case document using Gemini.
    We limit input by chunking and ask Gemini to produce a short unified summary.
    """
    if not text or not text.strip():
        return "No text provided."

    # If long, summarize chunks then aggregate
    chunks = chunk_text(text, max_chars=3500)
    if len(chunks) == 1:
        prompt = (
            "You are a legal summarization assistant. Produce a concise summary (3-8 sentences) "
            "of the following case facts, focusing on: who, what happened, where, when, relevant actors, "
            "and any damages/claims. Avoid legal conclusions; stick to facts.\n\n"
            f"CASE TEXT:\n\n{chunks[0]}\n\nSUMMARY:"
        )
        return call_gemini(prompt).strip()
    else:
        # Summarize each chunk then ask to synthesize
        partial_summaries = []
        for i, c in enumerate(chunks):
            p = (
                "You are a legal summarization assistant. Produce a 2-4 sentence factual summary of the excerpt. "
                "Focus on the key facts and actors. Output only the summary.\n\n"
                f"EXCERPT {i+1}:\n{c}\n\nSUMMARY:"
            )
            s = call_gemini(p).strip()
            partial_summaries.append(s)
        # combine partial summaries
        combined = "\n\n".join([f"Excerpt {i+1} summary: {s}" for i, s in enumerate(partial_summaries)])
        synth_prompt = (
            "You are a legal summarization assistant. Given the partial summaries below, synthesize a single concise "
            "factual summary (4-8 sentences) that captures the most important facts, in plain language.\n\n"
            f"{combined}\n\nSYNTHESIZED SUMMARY:"
        )
        return call_gemini(synth_prompt).strip()


def extract_laws_from_text(text: str, top_n: int = 12) -> List[Dict[str, Any]]:
    """
    Given a case text, returns a list of likely relevant laws/sections.
    Each item: { 'law': 'Indian Penal Code 302', 'section': '302', 'reason': 'Because the facts show...'}
    The list is ordered by relevance suggested by the model.
    """
    if not text or not text.strip():
        return []

    # Step 1: get a short case sketch to feed into the law extraction prompt
    case_summary = summarize_case_text(text)

    # Build a careful prompt that asks for laws + sections + short justification + confidence (low/med/high)
    prompt = textwrap.dedent(
        f"""
        You are a legal research assistant. You will be given a short factual summary of a case below.
        Provide a JSON array (only valid JSON) of up to {top_n} objects where each object has these fields:
          - "law": the statutory law or code name (e.g., "Indian Penal Code", "Evidence Act", "Code of Criminal Procedure", "Contract Act")
          - "section": the specific section or provision (e.g., "302", "420", "Section 75") — if multiple sections apply, list them separated by comma.
          - "title": a short title of the provision (e.g., "Murder", "Cheating")
          - "reason": 1-2 sentence justification why this law/section is relevant to the facts.
          - "confidence": one-word value among ["low", "medium", "high"] indicating how confident you are that this law is relevant based on the facts.
        Only include laws you think are plausibly relevant to the factual summary. Do not give legal advice. Output must be valid JSON only (no explanation, no extra text).
        
        FACTUAL SUMMARY:
        {case_summary}
        """
    )

    raw = call_gemini(prompt)
    # Attempt to extract JSON from raw output
    json_text = _extract_json_like(raw)
    import json

    try:
        parsed = json.loads(json_text)
        # ensure structure
        if isinstance(parsed, list):
            return parsed
        # if it's a dict with key 'results' or similar, try smart extraction
        if isinstance(parsed, dict):
            # if keys are numeric or 'laws'
            if "laws" in parsed and isinstance(parsed["laws"], list):
                return parsed["laws"]
            # otherwise convert dict values to list
            return [parsed]
    except Exception:
        # If parsing fails, fallback to asking Gemini to reformat with a more strict instruction
        strict_prompt = (
            "The previous output was not valid JSON. Re-output EXACTLY a JSON array (no explanation) using the same structure."
            f"\n\nPrevious output:\n{raw}\n\nNow produce valid JSON array only:"
        )
        raw2 = call_gemini(strict_prompt)
        try:
            parsed2 = json.loads(_extract_json_like(raw2))
            if isinstance(parsed2, list):
                return parsed2
        except Exception:
            # give fallback: return as single crude item
            return [{"law": "Could not parse model output", "section": "", "title": "", "reason": raw.strip()[:400], "confidence": "low"}]

    return []


def _extract_json_like(text: str) -> str:
    """
    Rudimentary extractor: find the first '[' and last ']' and return slice.
    If not found, return the whole text (to let json.loads fail and trigger fallback).
    """
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


# -------------------------
# Streamlit app UI
# -------------------------
def run_streamlit_app():
    if st is None:
        raise RuntimeError("Streamlit is not installed. Install streamlit to run the app.")

    st.set_page_config(page_title="Case → Relevant Laws", layout="wide")
    st.title("⚖️ Case → Likely Relevant Laws")
    st.markdown(
        "Upload a case file (PDF / DOCX / TXT) or paste the case text. The app will summarize the facts and suggest laws/sections that may be relevant. "
        "**This is informational only, not legal advice.**"
    )
    st.sidebar.markdown("**Settings**")
    top_n = st.sidebar.slider("Max laws to return", min_value=3, max_value=30, value=12)
    use_uploaded = st.radio("Input method", ["Upload file", "Paste text"])
    case_text = ""
    if use_uploaded == "Upload file":
        uploaded = st.file_uploader("Upload PDF / DOCX / TXT", type=["pdf", "docx", "txt"])
        if uploaded is not None:
            case_text = extract_text_from_upload(uploaded)
            if not case_text.strip():
                st.warning("Could not extract text from uploaded file. Try another file or paste text.")
    else:
        case_text = st.text_area("Paste case text here", height=300)

    if st.button("Analyze") and case_text.strip():
        with st.spinner("Summarizing case..."):
            summary = summarize_case_text(case_text)
        st.subheader("Case summary (facts)")
        st.write(summary)

        with st.spinner("Extracting likely laws and sections..."):
            try:
                laws = extract_laws_from_text(case_text, top_n=top_n)
            except Exception as e:
                st.error(f"Error calling Gemini: {e}")
                laws = []

        st.subheader("Likely relevant laws / provisions")
        if not laws:
            st.write("No laws returned.")
        else:
            for idx, item in enumerate(laws, 1):
                law = item.get("law", "")
                section = item.get("section", "")
                title = item.get("title", "")
                reason = item.get("reason", "")
                confidence = item.get("confidence", "")
                st.markdown(f"**{idx}. {law} — {section} — {title}**")
                st.write(f"- **Why:** {reason}")
                st.write(f"- **Confidence:** {confidence}")
                st.markdown("---")

        st.info("⚠️ Verify all statutory references in authoritative sources (Gazette / Bare Acts / official databases). This tool provides information, not legal advice.")

    st.markdown("### Tips")
    st.markdown(
        "- For best results, upload clean text (OCR PDFs sometimes require preprocessing)."
        "- If the document is long, the app summarizes first and then extracts laws; you can paste a concise factsheet for faster responses."
    )


# -------------------------
# Expose simple functions for Jupyter usage
# -------------------------
def analyze_case_file_bytes(file_bytes: bytes, filename: str = "uploaded") -> Dict[str, Any]:
    """Helper for Jupyter: takes raw bytes and returns summary and laws list."""
    if filename.lower().endswith(".pdf"):
        txt = read_pdf_bytes(file_bytes)
    elif filename.lower().endswith(".docx"):
        txt = read_docx_bytes(file_bytes)
    else:
        txt = read_txt_bytes(file_bytes)

    summary = summarize_case_text(txt)
    laws = extract_laws_from_text(txt)
    return {"summary": summary, "laws": laws}


def analyze_case_text(text: str) -> Dict[str, Any]:
    return {"summary": summarize_case_text(text), "laws": extract_laws_from_text(text)}


# -------------------------
# CLI / streamlit run
# -------------------------
if __name__ == "__main__":
    if st:
        run_streamlit_app()
    else:
        print("Run this module inside Streamlit (streamlit run legal_law_extractor.py) or import functions into Jupyter.")


