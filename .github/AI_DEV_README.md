# AI Dev README (Assistant Quick Reference)

Purpose: Ultra-fast orientation for AI agents & contributors extending this Streamlit RAG chatbot without re-reading the full project README.

## 🧱 Architecture Snapshot

Upload ➜ `DocumentProcessor` (extract + chunk) ➜ `SentenceTransformer` embeddings ➜ ChromaDB (`VectorDatabase`) ➜ (conditional retrieval) ➜ Google Gemini generation ➜ UI render + source cards.

## ⚙️ Core Modules

| File                     | Responsibility                                  | Notes                                                            |
| ------------------------ | ----------------------------------------------- | ---------------------------------------------------------------- |
| `rag_chatbot_app.py`     | UI orchestration, session state, prompt routing | Manual chat rendering; explicit RAG gating                       |
| `document_processor.py`  | PDF/DOCX/TXT extraction & chunking              | Uses `RecursiveCharacterTextSplitter`; extend here for new types |
| `vector_database.py`     | Embedding, storage, retrieval                   | Chroma persistent client + `all-MiniLM-L6-v2`                    |
| `test_rag_components.py` | Print-based sanity tests                        | No pytest; just run script                                       |

## 🧠 Session State Contract (Do NOT rename)

`genai_client`, `_last_key`, `chat`, `messages` (list of `{role, content, sources?}`), `doc_processor`, `vector_db`.

## 🗂 Chunk Metadata (stable keys)

`{source, file_type, file_size, chunk_index, chunk_size}` — UI relevance = `1 - distance`.

## 🔍 Retrieval Logic

RAG path if: user enabled RAG AND collection count > 0 AND similarity search returned results. Context built by concatenating top chunks until char limit (`max_context_length`).

## 🧾 Prompts

- RAG: `create_rag_prompt(query, context)`
- Fallback: `create_simple_prompt(query)`
  Keep branching explicit—no hidden heuristics.

## 🚀 Run & Test

```
streamlit run rag_chatbot_app.py
python test_rag_components.py
# Docker
docker build -t ai-document-assistant .
docker run -p 8501:8501 ai-document-assistant
```

## ➕ Add a New File Type

1. Implement `extract_text_from_<type>()` in `document_processor.py`.
2. Extend MIME routing in `extract_text()`.
3. Add extension to uploader `type=[...]` in `rag_chatbot_app.py`.
4. Update `get_supported_file_types()`.

## 📈 Performance Notes

- Chunk size 1000 / overlap 200 (static at instantiation).
- Embeddings always recomputed on each ingest (add hashing to optimize large corpora).
- Chroma persists to temp dir unless you pass / expose a directory (`CHROMA_PERSIST_DIRECTORY`).

## 🧪 Safe Refactors / Extensions

| Goal                | Tactic                                                    |
| ------------------- | --------------------------------------------------------- |
| Change model        | Add sidebar select; recreate `chat`                       |
| Add token stats     | Wrap prompt, compute via `tiktoken` (fallback gracefully) |
| Filter by file type | Pass `where={"file_type": ...}` to `similarity_search`    |
| Export sources      | Serialize `message['sources']` to downloadable JSON       |
| Adjustable chunking | Sidebar inputs ➜ rebuild `doc_processor`                  |

## 🛑 Don’t Break

- Metadata keys & relevance derivation.
- Session state naming.
- Explicit RAG gating logic.
- Immediate `st.rerun()` after adding a user/assistant message (responsive UI feel).

## 🧩 Common Pitfalls

| Pitfall                         | Prevention                                                       |
| ------------------------------- | ---------------------------------------------------------------- |
| Mid-chunk truncation            | Current logic stops before overshoot—preserve order & boundaries |
| Silent failures on extraction   | Always surface via `st.error` / `st.warning`                     |
| Double caching embeddings       | Only cache DB factory; let model init run once                   |
| Incorrect relevance percentages | Adjust UI if you change distance semantics                       |

## 🔐 API Key Handling

Re-instantiates `genai_client` & resets chat if key differs from `_last_key`.

## 🧷 Dependency Highlights

`chromadb`, `sentence-transformers`, `langchain` (splitter), `google-genai`, plus `python-docx`, `PyPDF2`.

## 🗺 Future-Friendly Hooks

- Namespace collections per user: `get_vector_database(f"docs_{user_id}")`.
- Introduce semantic reranking stage after initial similarity search.
- Token-aware truncation layer before model call.

---

Need a deeper dive (multi-user isolation, persistence strategy, eval harness)? Ask and we can extend this quick reference.
