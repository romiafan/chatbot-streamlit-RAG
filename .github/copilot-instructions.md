# Copilot Instructions (Concise)

Purpose: Fast orientation for extending this Streamlit RAG chatbot.

1. Flow: Upload -> `DocumentProcessor` (extract + chunk) -> `SentenceTransformer` embeds -> ChromaDB via `VectorDatabase` -> (if enabled & results) retrieval -> Gemini (`google.genai`) generation -> answer + cited chunks.
2. Key Files: `rag_chatbot_app.py` (UI/orchestration), `document_processor.py` (PDF/DOCX/TXT extraction + `RecursiveCharacterTextSplitter`), `vector_database.py` (embed/store/query), `test_rag_components.py` (print sanity tests).
3. Session State (stable names): `genai_client`, `_last_key`, `chat`, `messages` (list of `{role, content, sources?}`), `doc_processor`, `vector_db` (`@st.cache_resource`).
4. Chunk Metadata (must keep): `{source, file_type, file_size, chunk_index, chunk_size}`. UI relevance = `1 - distance`; change distance semantics -> update display.
5. Retrieval: `similarity_search()` returns docs + distances; `get_relevant_context()` concatenates until `max_context_length` (char-based). Character truncation is intentional.
6. Prompt Branching: Use RAG only if (RAG enabled) AND (collection has docs) AND (search results not empty); else simple prompt. Keep explicit.
7. Embedding Model: Default `all-MiniLM-L6-v2`; persistence directory defaults temp. Add env `CHROMA_PERSIST_DIRECTORY` if you introduce durable volumes.
8. Add File Types: New `extract_text_from_<type>()`, route in `extract_text()`, update uploader `type=[...]` + `get_supported_file_types()`. Mirror TXT unicode fallback.
9. UX Patterns: Long ops -> `st.spinner`; file processing uses progress bar; call `st.rerun()` right after appending a new user or assistant message for snappy UI.
10. Messaging Conventions: Use `st.error`, `st.warning`, `st.success`, `st.info` + existing emoji tone; avoid silent failures.
11. Performance Knobs: Chunk size 1000 / overlap 200 (constructor only). Recreating processor needed if you expose runtime adjustment. Currently always re-embeds—add hashing for large corpora.
12. Testing: `python test_rag_components.py` (no assertions framework). Run app: `streamlit run rag_chatbot_app.py`. Docker: `docker build -t ai-document-assistant . && docker run -p 8501:8501 ai-document-assistant`.
13. Safe Extensions: Add new session keys with clear prefixes (`retrieval_mode`, `token_stats`). If switching Gemini model, expose sidebar select.
14. Source Cards: Depend on `source`, `chunk_index`, and `distance`. Preserve keys when refactoring metadata.
15. Context Assembly: Order preserved; truncation stops before overshoot—avoid mid-chunk splitting for readability.
16. Clearing: `Clear Docs` invokes `vector_db.clear_collection()`. Maintain this API if wrapping deeper logic.
17. Caching: Only DB factory cached; embedding model init happens each process start—don't double cache.
18. Low-Risk Enhancements: Metadata filter (`where`), export sources JSON, adjustable chunk size, token counting via `tiktoken` with graceful fallback.

19. New Runtime Additions (2025-09):

    - `selected_model`: Chosen Gemini free-tier model (sidebar). Environment override: `GEMINI_DEFAULT_MODEL`.
    - `embed_mode`: Enabled via query param `?embed=1` or env `EMBED_MODE` (hides large header & section titles; condenses stats).
    - `message_count` / `token_estimate_total`: Conversation governance; naive token estimate ≈ chars/4. Limit via env `CHAT_MESSAGE_LIMIT` (default 50); 80% toast warning; hard stop at limit (must Reset Chat).
    - `chunk_hashes`: Set of SHA1 hashes of chunk text to avoid duplicate embedding; duplicates skipped with count surfaced in success message.
    - Sidebar Quick Stats now surfaces: docs, messages used, token estimate, model. Embed mode collapses these into a single compact line.
    - Model change triggers chat + history reset to ensure consistent model context.

20. Env Vars Summary:

    - `GEMINI_DEFAULT_MODEL`: Preselect selectable free model (e.g., gemini-1.5-flash-8b).
    - `EMBED_MODE`: `1/true/yes` => compact UI for iframe embedding.
    - `CHAT_MESSAGE_LIMIT`: Integer cap on total (user + assistant) messages before requiring reset (default 50).
    - `CHROMA_PERSIST_DIRECTORY`: (Pre-existing) Set to persist vector store across restarts.

21. Duplicate Handling: Hashing is in-memory per session; clearing docs or process restart resets dedupe. For persistent dedupe, add hash metadata to stored documents and check during load.

22. Extension Guidance: If adding precise token accounting, replace heuristic with `tiktoken` (optional dependency) behind try/except fallback; update both stats display and instructions accordingly.

Need more depth (multi-user namespacing, token budgeting, persistence strategy)? Ask and we can extend.
