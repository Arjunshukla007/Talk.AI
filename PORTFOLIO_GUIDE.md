# Portfolio guide for Talk.AI

## Resume bullets (verify each claim yourself before using it)
- Built Talk.AI, a multi-document RAG application (Python, Streamlit, FastAPI, ChromaDB) that answers questions over resumes and PDFs with streamed, cited answers and page-level source cards.
- Implemented hybrid retrieval (dense vectors + BM25 with Reciprocal Rank Fusion), improving hit@3 from 0.67 (vector only) to 0.83 on a 14-question evaluation set.
- Designed hallucination guards: context-only prompts, prompt-injection delimiters, exact "not found" refusals, and Pydantic-validated JSON with automatic repair for resume extraction and resume-to-job matching.
- Shipped with a REST API, 35 automated tests, a retrieval evaluation script, Dockerfile, and GitHub Actions CI.

## 3-minute demo script
1. Load sample files. Ask "Where did the candidate intern?" and open the source card.
2. Ask "Does the job require Kubernetes?" then a follow-up "Is that required or optional?" to show query rewriting.
3. Ask "What is the candidate's favourite colour?" to show the refusal.
4. Open Job match: resume vs job description, point at missing skills.
5. Show `/docs` (API) and `python -m evaluation.run_eval`.

## Interview questions
**Why hybrid search?** Embeddings capture meaning but can miss exact terms (tool names, IDs). BM25 does the reverse. Rank fusion combines them without tuning score scales. My eval showed it beat either alone.
**How do you stop hallucinations?** Retrieve first; the prompt allows only excerpt text and defines an exact refusal; no LLM call when nothing is retrieved; answers cite excerpt numbers and only cited passages are shown. I can't guarantee zero hallucination, so there is an evaluation script that measures refusals.
**What is chunking and why overlap?** Splitting text into searchable pieces. Overlap keeps facts that straddle a boundary retrievable. Trade-off: small chunks are precise but lose context.
**How do you handle follow-ups?** The last turns are rewritten into a standalone search query before retrieval, because "and where?" has no searchable content.
**What about prompt injection in uploaded files?** Excerpts are wrapped in tags and the system prompt says they are data, not instructions. It reduces the risk, it doesn't eliminate it.
**How did you test without paying for API calls?** The LLM sits behind a small interface; tests inject a scripted fake. Retrieval is tested with a deterministic offline embedding. Real-model quality is measured separately with `--llm`.
**How would you scale it?** Move to a hosted vector database, add auth and per-user collections, build the BM25 index incrementally, add a reranker, queue ingestion jobs.
**What would you improve first?** A larger labelled evaluation set and OCR for scanned PDFs.
