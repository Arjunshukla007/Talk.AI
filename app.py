"""Talk.AI Streamlit interface. Run: streamlit run app.py"""
from __future__ import annotations

import html
from dataclasses import replace

import streamlit as st

from talkai import analysis
from talkai.config import PROVIDERS, ConfigError, Settings
from talkai.factory import build_service
from talkai.ingestion import DocumentError
from talkai.llm import LLMError

st.set_page_config(page_title="Talk.AI", page_icon="💬", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;800&family=Source+Serif+4:ital@0;1&display=swap');
html, body, [class*="css"], .stMarkdown, .stButton { font-family: 'Manrope', sans-serif; }
.stApp { background: #F5F7FA; }
[data-testid="stSidebar"] { background: #0F1B2D; }
[data-testid="stSidebar"] * { color: #E6ECF4 !important; }
[data-testid="stSidebar"] [data-testid="stFileUploader"] section { background: #17294A; border: 1.5px dashed #4C7DBA; }
[data-testid="stSidebar"] input, [data-testid="stSidebar"] [data-baseweb="select"] * { color: #0F1B2D !important; }
.brand { font-size: 2.1rem; font-weight: 800; letter-spacing: -0.03em; color: #0F1B2D; margin: 0; }
.brand span { color: #0E7C86; }
.tagline { color: #52627A; margin: 0 0 1rem 0; }
.src-card { background: #fff; border-left: 4px solid #E9A23B; border-radius: 6px; padding: .7rem .9rem; margin: .45rem 0;
            font-family: 'Source Serif 4', serif; font-size: .93rem; line-height: 1.55; color: #27364D; }
.src-meta { font-family: 'Manrope', sans-serif; font-size: .78rem; font-weight: 600; color: #8A5A00; margin-bottom: .25rem; }
.chip { display: inline-block; background: #DDF1F2; color: #095B63; padding: .28rem .7rem; margin: .2rem .25rem .2rem 0;
        border-radius: 999px; font-size: .86rem; font-weight: 600; }
.chip.miss { background: #FCEBCB; color: #7A4B00; }
.panel { background: #fff; border-radius: 8px; padding: 1rem 1.2rem; border: 1px solid #E2E8F0; }
@media (max-width: 640px) { .brand { font-size: 1.6rem; } }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_service(settings: Settings):
    return build_service(settings)


def chips(items, cls="") -> str:
    return "".join(f'<span class="chip {cls}">{html.escape(i)}</span>' for i in items) or "<em>None</em>"


def render_sources(sources) -> None:
    for i, s in enumerate(sources, 1):
        where = f"page {s.page}" if s.page else "text file"
        st.markdown(
            f'<div class="src-card"><div class="src-meta">{html.escape(s.filename)} · {where} · relevance {s.score:.0%}</div>'
            f"{html.escape(s.text[:600])}{'…' if len(s.text) > 600 else ''}</div>", unsafe_allow_html=True)


st.session_state.setdefault("messages", [])
st.session_state.setdefault("cache", {})

# ------------------------------- sidebar ------------------------------------ #
env = Settings.from_env()
with st.sidebar:
    st.markdown("### Talk.AI")
    provider = st.selectbox("AI provider", PROVIDERS, index=PROVIDERS.index(env.provider) if env.provider in PROVIDERS else 0)
    key_ok = bool(env.openai_api_key if provider == "openai" else env.anthropic_api_key)
    typed_key = "" if key_ok else st.text_input(f"{provider.title()} API key", type="password",
                                                help="Used for this browser session only. Or put it in your .env file.")
    settings = replace(env, provider=provider)
    if typed_key:
        settings = replace(settings, **{f"{provider}_api_key": typed_key.strip()})
    top_k = st.slider("Passages per answer", 2, 8, env.top_k)
    settings = replace(settings, top_k=top_k)

try:
    service = get_service(settings)
except ConfigError as exc:
    st.markdown('<p class="brand">Talk<span>.AI</span></p>', unsafe_allow_html=True)
    st.error(str(exc))
    st.info("Copy `.env.example` to `.env`, add your key, and restart. Or paste a key in the sidebar.")
    st.stop()
except Exception as exc:
    st.error(f"Could not start Talk.AI: {exc}")
    st.stop()

with st.sidebar:
    files = st.file_uploader("Add PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)
    if st.button("Index files", type="primary", use_container_width=True, disabled=not files):
        bar = st.progress(0.0)
        for n, f in enumerate(files, 1):
            try:
                with st.spinner(f"Indexing {f.name}…"):
                    info = service.ingest(f.getvalue(), f.name)
                st.success(f"{f.name}: {info['chunks']} passages" + (" (already indexed)" if info["already_indexed"] else ""))
            except (DocumentError, LLMError, ConfigError) as exc:
                st.error(f"{f.name}: {exc}")
            except Exception as exc:
                st.error(f"{f.name}: unexpected error - {exc}")
            bar.progress(n / len(files))
    docs = service.store.list_documents()
    st.markdown("**Your documents**")
    if not docs:
        st.caption("Nothing indexed yet.")
    for d in docs:
        c1, c2 = st.columns([5, 1])
        c1.caption(f"{d['filename']}  \n{d['words']:,} words · {d['chunks']} passages")
        if c2.button("✕", key=f"del_{d['doc_id']}", help="Remove this document"):
            service.store.delete_document(d["doc_id"])
            st.session_state.cache = {}
            st.rerun()
    if st.button("Load sample files", use_container_width=True):
        from pathlib import Path
        for p in Path("sample_data").glob("*.txt"):
            service.ingest(p.read_bytes(), p.name)
        st.rerun()

# ------------------------------- main page ---------------------------------- #
st.markdown('<p class="brand">Talk<span>.AI</span></p>', unsafe_allow_html=True)
st.markdown('<p class="tagline">Ask your documents anything. Every answer shows the passages it came from.</p>', unsafe_allow_html=True)

if not docs:
    st.info("Add a resume, report or any PDF in the sidebar, then ask questions about it. "
            "Use **Load sample files** to try a sample resume and job description.")
    st.stop()

names = {d["doc_id"]: d["filename"] for d in docs}
tab_chat, tab_summary, tab_resume, tab_match, tab_text = st.tabs(
    ["Chat", "Summary", "Resume profile", "Job match", "Document text"])

with tab_chat:
    scope = st.multiselect("Search in", list(names), default=list(names), format_func=names.get)
    if not st.session_state.messages:
        st.caption("Try: “What are the main skills?” · “Where did they work?” · “Does this mention Kubernetes?”")
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("sources"):
                with st.expander(f"Sources ({len(m['sources'])}) · answered in {m['latency']/1000:.1f}s"):
                    render_sources(m["sources"])
    question = st.chat_input("Ask something about your documents")
    if question and not scope:
        st.warning("Select at least one document to search.")
    elif question:
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                stream = service.ask_stream(question, scope, history)
                st.write_stream(stream.tokens())
                ans = stream.finish()
                if ans.sources:
                    with st.expander(f"Sources ({len(ans.sources)}) · answered in {ans.latency_ms/1000:.1f}s", expanded=True):
                        render_sources(ans.sources)
                st.session_state.messages.append(
                    {"role": "assistant", "content": ans.text, "sources": ans.sources, "latency": ans.latency_ms})
            except (ConfigError, LLMError) as exc:
                st.error(str(exc))
    if st.session_state.messages and st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()


def pick_doc(key: str) -> str:
    return st.selectbox("Document", list(names), format_func=names.get, key=key)


def cached(kind: str, doc_id: str, fn):
    """Run an analysis once per (kind, document) and keep the result for the session."""
    key = (kind, doc_id)
    if key not in st.session_state.cache:
        st.session_state.cache[key] = fn()
    return st.session_state.cache[key]


def doc_text(doc_id: str) -> str:
    return service.store.get_document(doc_id).full_text


with tab_summary:
    doc_id = pick_doc("sum_doc")
    if st.button("Summarise"):
        try:
            with st.spinner("Summarising…"):
                cached("summary", doc_id, lambda: analysis.summarize(service.llm, doc_text(doc_id)))
        except (ConfigError, LLMError) as exc:
            st.error(str(exc))
    if ("summary", doc_id) in st.session_state.cache:
        st.markdown(st.session_state.cache[("summary", doc_id)])

with tab_resume:
    doc_id = pick_doc("res_doc")
    if st.button("Extract profile"):
        try:
            with st.spinner("Reading the resume…"):
                cached("resume", doc_id, lambda: analysis.extract_resume(service.llm, doc_text(doc_id)))
        except (ConfigError, LLMError) as exc:
            st.error(str(exc))
    p = st.session_state.cache.get(("resume", doc_id))
    if p:
        st.subheader(p.name or "Candidate")
        st.markdown("**Skills**")
        st.markdown(chips(p.skills), unsafe_allow_html=True)
        for title, items in (("Education", p.education), ("Experience", p.experience), ("Projects", p.projects)):
            st.markdown(f"**{title}**")
            for item in items or ["Nothing found in the document."]:
                st.markdown(f"- {item}")

with tab_match:
    st.caption("Compare a resume with a job description. The score is the model's judgement from the two texts, not an ATS result.")
    resume_id = pick_doc("match_resume")
    jd_source = st.selectbox("Job description from", ["Paste text"] + list(names), format_func=lambda k: k if k == "Paste text" else names[k])
    jd_text = st.text_area("Job description", height=180) if jd_source == "Paste text" else doc_text(jd_source)
    if st.button("Check match"):
        if len(jd_text.strip()) < 30:
            st.warning("Add a longer job description first.")
        else:
            try:
                with st.spinner("Comparing…"):
                    st.session_state.cache[("match", resume_id)] = analysis.match_job(service.llm, doc_text(resume_id), jd_text)
            except (ConfigError, LLMError) as exc:
                st.error(str(exc))
    r = st.session_state.cache.get(("match", resume_id))
    if r:
        st.metric("Match score", f"{r.score}/100")
        st.progress(r.score / 100)
        st.write(r.summary)
        a, b = st.columns(2)
        a.markdown("**Matched skills**")
        a.markdown(chips(r.matched_skills), unsafe_allow_html=True)
        b.markdown("**Missing skills**")
        b.markdown(chips(r.missing_skills, "miss"), unsafe_allow_html=True)
        for title, items in (("Strengths", r.strengths), ("How to improve the resume", r.suggestions)):
            st.markdown(f"**{title}**")
            for item in items:
                st.markdown(f"- {item}")

with tab_text:
    doc_id = pick_doc("txt_doc")
    st.caption("Extracted text. If answers look wrong, check here first; scanned PDFs extract poorly.")
    st.text_area("Text", doc_text(doc_id), height=420, label_visibility="collapsed")
