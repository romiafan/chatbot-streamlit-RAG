"""
RAG-Enabled Chatbot with File Upload
Combines document processing, vector storage, and conversational AI
"""

import streamlit as st
from google import genai
import os
import time
import json  # for exporting sources as JSON
from typing import List, Dict, Any, Optional

# Import our custom modules
from document_processor import DocumentProcessor, create_document_processor
from vector_database import VectorDatabase, get_vector_database, display_vector_db_info

# --- Modern CSS Styling ---
def apply_custom_css():
    """Apply modern CSS styling inspired by Perplexity and Gemini"""
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Variables */
    :root {
        --primary-color: #2563eb;
        --primary-hover: #1d4ed8;
        --secondary-color: #64748b;
        --accent-color: #3b82f6;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --background: #ffffff;
        --surface: #f8fafc;
        --surface-variant: #f1f5f9;
        --text-primary: #0f172a;
        --text-secondary: #475569;
        --text-tertiary: #94a3b8;
        --border-color: #e2e8f0;
        --border-light: #f1f5f9;
        --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
        --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
        --radius-sm: 6px;
        --radius-md: 8px;
        --radius-lg: 12px;
        --radius-xl: 16px;
    }
    
    /* Dark mode variables */
    @media (prefers-color-scheme: dark) {
        :root {
            --background: #0f172a;
            --surface: #1e293b;
            --surface-variant: #334155;
            --text-primary: #f8fafc;
            --text-secondary: #cbd5e1;
            --text-tertiary: #64748b;
            --border-color: #334155;
            --border-light: #1e293b;
        }
    }
    
    /* Base Styles */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background-color: var(--background);
        color: var(--text-primary);
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Main Container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Custom Header */
    .custom-header {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--accent-color) 100%);
        padding: 2rem 0;
        margin: -2rem -1rem 2rem -1rem;
        border-radius: 0 0 var(--radius-xl) var(--radius-xl);
        text-align: center;
        box-shadow: var(--shadow-lg);
    }
    
    .custom-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .custom-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
    }
    
    /* Sidebar Styling */
    .stSidebar {
        background-color: var(--surface);
        border-right: 1px solid var(--border-color);
    }
    
    .stSidebar .stSelectbox, .stSidebar .stSlider, .stSidebar .stCheckbox {
        margin-bottom: 1rem;
    }
    
    /* Modern Card Styling */
    .modern-card {
        background: var(--surface);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--shadow-sm);
        transition: all 0.2s ease;
    }
    
    .modern-card:hover {
        box-shadow: var(--shadow-md);
        border-color: var(--primary-color);
    }
    
    /* Chat Interface */
    .stChatMessage {
        border-radius: var(--radius-lg);
        margin: 1rem 0;
        padding: 1rem;
        border: 1px solid var(--border-light);
        box-shadow: var(--shadow-sm);
    }
    
    .stChatMessage[data-testid="chat-message-user"] {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--accent-color) 100%);
        color: white;
        margin-left: 20%;
        border: none;
    }
    
    .stChatMessage[data-testid="chat-message-assistant"] {
        background: var(--surface);
        border: 1px solid var(--border-color);
        margin-right: 20%;
    }
    
    /* Modern Buttons */
    .stButton button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--accent-color) 100%);
        color: white;
        border: none;
        border-radius: var(--radius-md);
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        font-size: 0.875rem;
        transition: all 0.2s ease;
        box-shadow: var(--shadow-sm);
    }
    
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: var(--shadow-md);
        background: linear-gradient(135deg, var(--primary-hover) 0%, var(--primary-color) 100%);
    }
    
    /* File Uploader */
    .stFileUploader {
        border: 2px dashed var(--border-color);
        border-radius: var(--radius-lg);
        padding: 2rem;
        text-align: center;
        background: var(--surface-variant);
        transition: all 0.2s ease;
    }
    
    .stFileUploader:hover {
        border-color: var(--primary-color);
        background: var(--surface);
    }
    
    /* Chat Input */
    .stChatInput {
        border-radius: var(--radius-lg);
        border: 1px solid var(--border-color);
        background: var(--surface);
        box-shadow: var(--shadow-sm);
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background: linear-gradient(135deg, var(--success-color), #059669);
        color: white;
        border-radius: var(--radius-md);
        border: none;
    }
    
    .stError {
        background: linear-gradient(135deg, var(--error-color), #dc2626);
        color: white;
        border-radius: var(--radius-md);
        border: none;
    }
    
    .stWarning {
        background: linear-gradient(135deg, var(--warning-color), #d97706);
        color: white;
        border-radius: var(--radius-md);
        border: none;
    }
    
    .stInfo {
        background: linear-gradient(135deg, var(--accent-color), var(--primary-color));
        color: white;
        border-radius: var(--radius-md);
        border: none;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: var(--surface);
        border-radius: var(--radius-md);
        border: 1px solid var(--border-color);
        font-weight: 500;
    }
    
    /* Loading Spinner */
    .stSpinner {
        text-align: center;
        padding: 2rem;
    }
    
    /* Metrics */
    .stMetric {
        background: var(--surface);
        padding: 1rem;
        border-radius: var(--radius-md);
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-sm);
    }
    
    /* Custom Classes */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: var(--radius-sm);
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.025em;
    }
    
    .status-success {
        background: var(--success-color);
        color: white;
    }
    
    .status-warning {
        background: var(--warning-color);
        color: white;
    }
    
    .glass-effect {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Smooth Animations */
    * {
        transition: background-color 0.2s ease, border-color 0.2s ease, color 0.2s ease;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        
        .custom-header h1 {
            font-size: 2rem;
        }
        
        .stChatMessage[data-testid="chat-message-user"] {
            margin-left: 5%;
        }
        
        .stChatMessage[data-testid="chat-message-assistant"] {
            margin-right: 5%;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. Page Configuration and Modern Header ---

st.set_page_config(
    page_title="RAG Chatbot | AI-Powered Document Assistant",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "RAG-Enabled Chatbot - Combining the power of retrieval-augmented generation with modern UI/UX"
    }
)

# Apply custom CSS
apply_custom_css()

# Determine embed / compact mode (query param or env)
embed_mode = False
try:
    # Query params available only during script run
    qp = st.query_params if hasattr(st, 'query_params') else st.experimental_get_query_params()
    if qp:
        # Accept values like '1', 'true', 'yes'
        raw = qp.get('embed')
        if isinstance(raw, list):
            raw = raw[0]
        if raw is not None and str(raw).lower() in {"1", "true", "yes"}:
            embed_mode = True
except Exception:
    pass

if not embed_mode:
    # Allow env override when query param absent
    if os.getenv("EMBED_MODE", "").lower() in {"1", "true", "yes"}:
        embed_mode = True

if not embed_mode:
    # Custom Modern Header (suppressed in embed mode for tighter iframe usage)
    st.markdown("""
    <div class="custom-header">
        <h1>🧠 AI Document Assistant</h1>
        <p>Intelligent conversations powered by your documents</p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Compact top spacer
    st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)

# --- 2. Modern Sidebar Configuration ---

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # API Key Section
    with st.container():
        st.markdown("#### 🔑 Authentication")
        google_api_key = st.text_input(
            "Google AI API Key", 
            type="password", 
            help="Enter your Google AI API key to enable the chatbot",
            placeholder="Enter your API key..."
        )
    
    # Model selection (free-tier friendly Gemini flash models)
    with st.container():
        st.markdown("#### 🤖 Model")
        free_models = [
            "gemini-1.5-flash",  # general purpose, fast
            "gemini-1.5-flash-8b",  # smaller, cheaper
        ]
        # Provide backward compatibility if session already has a model not in list
        if "selected_model" in st.session_state and st.session_state.selected_model not in free_models:
            free_models.append(st.session_state.selected_model)

        default_model = os.getenv("GEMINI_DEFAULT_MODEL", free_models[0])
        if "selected_model" not in st.session_state:
            # Initialize with env override if valid
            st.session_state.selected_model = default_model if default_model in free_models else free_models[0]

        chosen_model = st.selectbox(
            "Gemini Model",
            options=free_models,
            index=free_models.index(st.session_state.selected_model) if st.session_state.selected_model in free_models else 0,
            help="Choose a free Gemini Flash model. Changing this will reset the current chat session.",
        )

        if chosen_model != st.session_state.selected_model:
            st.session_state.selected_model = chosen_model
            # Reset chat so subsequent messages use new model
            st.session_state.pop("chat", None)
            st.session_state.pop("messages", None)
            st.rerun()

    st.divider()
    
    # RAG Settings Section
    with st.container():
        st.markdown("#### 📚 RAG Configuration")

        # Chunking controls (adjustable) - recreate processor if changed
        default_chunk_size = st.session_state.get("_chunk_size", 1000)
        default_overlap = st.session_state.get("_chunk_overlap", 200)
        with st.expander("🔧 Chunk Settings", expanded=False):
            new_chunk_size = st.number_input(
                "Chunk Size (characters)",
                min_value=200, max_value=4000, step=100,
                value=default_chunk_size,
                help="Length of each text chunk for embedding. Larger = fewer, bigger chunks; smaller = finer retrieval granularity."
            )
            new_chunk_overlap = st.number_input(
                "Chunk Overlap",
                min_value=0, max_value=1000, step=50,
                value=default_overlap,
                help="Characters of overlap between adjacent chunks to preserve context continuity."
            )
            if (new_chunk_size != default_chunk_size) or (new_chunk_overlap != default_overlap):
                st.session_state._chunk_size = int(new_chunk_size)
                st.session_state._chunk_overlap = int(new_chunk_overlap)
                # Recreate document processor with new settings
                from document_processor import DocumentProcessor
                st.session_state.doc_processor = DocumentProcessor(
                    chunk_size=st.session_state._chunk_size,
                    chunk_overlap=st.session_state._chunk_overlap
                )
                st.info(f"🔄 Chunk settings updated: size={st.session_state._chunk_size}, overlap={st.session_state._chunk_overlap}. Re-process documents to apply.")
        
        use_rag = st.checkbox(
            "Enable RAG", 
            value=True, 
            help="Use uploaded documents to enhance responses"
        )
        
        if use_rag:
            col1, col2 = st.columns(2)
            with col1:
                num_context_docs = st.selectbox(
                    "Context Docs",
                    options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    index=2,
                    help="Number of relevant documents to use as context"
                )
            
            with col2:
                context_length_options = {
                    "Short": 1000,
                    "Medium": 2000,
                    "Long": 3000,
                    "Extended": 4000
                }
                selected_length = st.selectbox(
                    "Context Length",
                    options=list(context_length_options.keys()),
                    index=1,
                    help="Maximum length of context from documents"
                )
                max_context_length = context_length_options[selected_length]
        else:
            num_context_docs = 3
            max_context_length = 2000
    
    st.divider()
    
    # Document Management Section
    with st.container():
        st.markdown("#### 📄 Document Management")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Docs", help="Remove all uploaded documents", use_container_width=True):
                if "vector_db" in st.session_state:
                    st.session_state.vector_db.clear_collection()
                    st.rerun()
        
        with col2:
            if st.button("🔄 Reset Chat", help="Clear chat history", use_container_width=True):
                st.session_state.pop("chat", None)
                st.session_state.pop("messages", None)
                st.rerun()
    
    st.divider()
    
    # Quick Stats (expanded)
    info = None
    if "vector_db" in st.session_state:
        info = st.session_state.vector_db.get_collection_info()
    doc_count = (info or {}).get("document_count", 0)
    model_name = st.session_state.get("selected_model", "gemini-1.5-flash")
    msg_count = st.session_state.get("message_count", 0)
    msg_limit = int(os.getenv("CHAT_MESSAGE_LIMIT", "50"))
    token_est = st.session_state.get("token_estimate_total", 0)

    st.markdown("#### 📊 Quick Stats")
    if embed_mode:
        st.markdown(
            f"<div class='modern-card' style='padding:0.75rem; font-size:0.75rem; line-height:1.4;'>🧠 {model_name} • 📄 {doc_count} docs • 💬 {msg_count}/{msg_limit} • 🔢 ~{token_est} tokens</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(f"""
        <div class="modern-card" style="padding: 1rem;">
            <div style="display:flex; justify-content:space-between; gap:0.75rem; flex-wrap:wrap;">
                <div style="text-align:center; min-width:80px;">
                    <div style="font-size:1.25rem; font-weight:600; color:var(--primary-color);">{doc_count}</div>
                    <div style="font-size:0.65rem; text-transform:uppercase; letter-spacing:0.05em; color:var(--text-secondary);">Docs</div>
                </div>
                <div style="text-align:center; min-width:80px;">
                    <div style="font-size:1.25rem; font-weight:600; color:var(--primary-color);">{msg_count}/{msg_limit}</div>
                    <div style="font-size:0.65rem; text-transform:uppercase; letter-spacing:0.05em; color:var(--text-secondary);">Messages</div>
                </div>
                <div style="text-align:center; min-width:80px;">
                    <div style="font-size:1.25rem; font-weight:600; color:var(--primary-color);">~{token_est}</div>
                    <div style="font-size:0.65rem; text-transform:uppercase; letter-spacing:0.05em; color:var(--text-secondary);">Tokens</div>
                </div>
                <div style="flex:1; min-width:140px;">
                    <div style="font-size:0.6rem; font-weight:600; color:var(--text-tertiary); text-transform:uppercase; letter-spacing:0.05em;">Model</div>
                    <div style="font-size:0.8rem; font-weight:500; color:var(--text-primary); word-break:break-all;">{model_name}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Batch export of all sources (if any assistant messages with sources)
    all_sources = []
    if "messages" in st.session_state:
        for m in st.session_state.messages:
            if m.get("role") == "assistant" and m.get("sources"):
                all_sources.extend(m["sources"])
    if all_sources:
        try:
            import json as _json
            export_data = _json.dumps(all_sources, ensure_ascii=False, indent=2)
            st.download_button(
                "⬇️ Export All Sources JSON",
                data=export_data,
                file_name="all_sources.json",
                mime="application/json",
                help="Download a consolidated JSON of every source chunk used so far"
            )
        except Exception:
            pass

# --- 3. API Key Validation ---

if not google_api_key:
    st.info("👈 Please add your Google AI API key in the sidebar to start chatting.", icon="🗝️")
    st.stop()

# Initialize Google AI client
if ("genai_client" not in st.session_state) or (getattr(st.session_state, "_last_key", None) != google_api_key):
    try:
        st.session_state.genai_client = genai.Client(api_key=google_api_key)
        st.session_state._last_key = google_api_key
        st.session_state.pop("chat", None)
        st.session_state.pop("messages", None)
    except Exception as e:
        st.error(f"Invalid API Key: {e}")
        st.stop()

# --- 4. Initialize RAG Components ---

# Initialize document processor
if "doc_processor" not in st.session_state:
    # Respect any previously chosen chunk parameters
    chunk_size = st.session_state.get("_chunk_size", 1000)
    chunk_overlap = st.session_state.get("_chunk_overlap", 200)
    from document_processor import DocumentProcessor as _DP
    st.session_state.doc_processor = _DP(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

# Initialize vector database
if "vector_db" not in st.session_state:
    st.session_state.vector_db = get_vector_database("rag_chatbot_docs")

# --- 5. Modern File Upload Section ---

if not embed_mode:
    st.markdown("### 📁 Document Upload")

# Create a modern upload interface
upload_container = st.container()
with upload_container:
    st.markdown("""
    <div class="modern-card">
        <div style="text-align: center; padding: 1rem 0;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📄</div>
            <h4 style="margin: 0; color: var(--text-primary);">Upload Your Documents</h4>
            <p style="color: var(--text-secondary); margin: 0.5rem 0 1rem 0;">
                Drag and drop files or click to browse • Supports PDF, DOCX, TXT
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader with enhanced styling
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Maximum file size: 200MB per file",
        label_visibility="collapsed"
    )
    
    # Show uploaded files in a modern way
    if uploaded_files:
        st.markdown("#### 📋 Selected Files")
        
        files_info = []
        total_size = 0
        
        for file in uploaded_files:
            file_size_mb = len(file.getvalue()) / (1024 * 1024)
            total_size += file_size_mb
            files_info.append({
                "name": file.name,
                "size": f"{file_size_mb:.1f} MB",
                "type": file.type or "Unknown"
            })
        
        # Display file cards
        for i, file_info in enumerate(files_info):
            st.markdown(f"""
            <div class="modern-card" style="margin: 0.5rem 0; padding: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="display: flex; align-items: center; gap: 1rem;">
                        <div style="font-size: 1.5rem;">
                            {'📄' if file_info['type'] == 'text/plain' else '📘' if 'word' in file_info['type'] else '📕'}
                        </div>
                        <div>
                            <div style="font-weight: 500; color: var(--text-primary);">{file_info['name']}</div>
                            <div style="font-size: 0.875rem; color: var(--text-secondary);">{file_info['size']} • {file_info['type']}</div>
                        </div>
                    </div>
                    <div class="status-badge status-success">Ready</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Processing button with enhanced styling
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(
                f"� Process {len(uploaded_files)} Document{'s' if len(uploaded_files) > 1 else ''}", 
                type="primary",
                use_container_width=True,
                help=f"Process and index {len(uploaded_files)} files ({total_size:.1f} MB total)"
            ):
                # Enhanced processing with progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Process the uploaded files
                    status_text.text("📤 Uploading and extracting text...")
                    progress_bar.progress(25)
                    
                    documents = st.session_state.doc_processor.process_uploaded_files(uploaded_files)
                    
                    if documents:
                        status_text.text("🔍 Creating embeddings...")
                        progress_bar.progress(50)
                        
                        # Add to vector database with duplicate filtering
                        status_text.text("💾 Storing in vector database (filtering duplicates)...")
                        progress_bar.progress(75)

                        if "chunk_hashes" not in st.session_state:
                            st.session_state.chunk_hashes = set()

                        import hashlib as _hashlib
                        unique_docs = []
                        skipped = 0

                        # Build an existing hash set from persisted collection once per run (metadata scan)
                        existing_hashes = set()
                        try:
                            existing = st.session_state.vector_db.collection.get(include=["metadatas"], limit=100000)
                            for meta in existing.get("metadatas", []) or []:
                                if not meta:
                                    continue
                                for m in meta if isinstance(meta, list) else [meta]:
                                    h = m.get("chunk_hash") if isinstance(m, dict) else None
                                    if h:
                                        existing_hashes.add(h)
                        except Exception:
                            pass

                        for d in documents:
                            content = getattr(d, 'page_content', None)
                            if content is None and isinstance(d, dict):
                                content = d.get('page_content') or d.get('document')
                            text_for_hash = (content or '').strip()
                            if not text_for_hash:
                                continue
                            h = _hashlib.sha1(text_for_hash.encode('utf-8', errors='ignore')).hexdigest()
                            if h in st.session_state.chunk_hashes or h in existing_hashes:
                                skipped += 1
                                continue
                            st.session_state.chunk_hashes.add(h)
                            # attach hash to metadata for persistence
                            try:
                                meta_attr = getattr(d, 'metadata', None)
                                if meta_attr is not None and isinstance(meta_attr, dict):
                                    meta_attr['chunk_hash'] = h
                                elif isinstance(d, dict):
                                    d.setdefault('metadata', {})
                                    if isinstance(d['metadata'], dict):
                                        d['metadata']['chunk_hash'] = h
                            except Exception:
                                pass
                            unique_docs.append(d)

                        success = True
                        if unique_docs:
                            success = st.session_state.vector_db.add_documents(unique_docs)
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Processing complete!")
                        
                        if success:
                            added = len(unique_docs)
                            total = len(documents)
                            if skipped:
                                st.success(f"🎉 Processed {len(uploaded_files)} files: {added} new chunks stored (skipped {skipped} duplicates of {total}).")
                            else:
                                st.success(f"🎉 Successfully processed {len(uploaded_files)} files and created {added} document chunks!")
                            
                            # Display processing statistics
                            st.markdown(f"""
                            <div class="modern-card" style="background: linear-gradient(135deg, var(--success-color), #059669); color: white; text-align: center;">
                                <div style="display: flex; justify-content: space-around; align-items: center;">
                                    <div>
                                        <div style="font-size: 1.5rem; font-weight: bold;">{len(uploaded_files)}</div>
                                        <div style="font-size: 0.875rem; opacity: 0.9;">Files</div>
                                    </div>
                                    <div>
                                        <div style="font-size: 1.5rem; font-weight: bold;">{len(documents)}</div>
                                        <div style="font-size: 0.875rem; opacity: 0.9;">Chunks</div>
                                    </div>
                                    <div>
                                        <div style="font-size: 1.5rem; font-weight: bold;">{total_size:.1f}</div>
                                        <div style="font-size: 0.875rem; opacity: 0.9;">MB</div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error("❌ Failed to add documents to vector database")
                    else:
                        st.warning("⚠️ No text could be extracted from the uploaded files")
                        
                except Exception as e:
                    st.error(f"❌ Error processing files: {str(e)}")
                finally:
                    progress_bar.empty()
                    status_text.empty()

# Enhanced vector database info display
if "vector_db" in st.session_state:
    info = st.session_state.vector_db.get_collection_info()
    doc_count = info.get("document_count", 0)
    
    if doc_count > 0:
        st.markdown("#### 📊 Knowledge Base Status")
        
        st.markdown(f"""
        <div class="modern-card" style="background: linear-gradient(135deg, var(--accent-color), var(--primary-color)); color: white;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-size: 1.25rem; font-weight: 600;">Knowledge Base Active</div>
                    <div style="opacity: 0.9; font-size: 0.875rem;">{doc_count} documents ready for queries</div>
                </div>
                <div style="font-size: 2rem;">🧠</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# --- 6. Modern Chat Interface ---

if not embed_mode:
    st.markdown("### 💬 Conversation")

# Maintain a dict of chat sessions per model for per-message selection
if "model_chats" not in st.session_state:
    st.session_state.model_chats = {}

def get_chat_for_model(model: str):
    if model not in st.session_state.model_chats:
        try:
            st.session_state.model_chats[model] = st.session_state.genai_client.chats.create(model=model)
        except Exception:
            if model != "gemini-1.5-flash":
                st.session_state.model_chats[model] = st.session_state.genai_client.chats.create(model="gemini-1.5-flash")
            else:
                raise
    return st.session_state.model_chats[model]

# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Create chat container with modern styling
chat_container = st.container()

with chat_container:
    # Show welcome message if no messages
    if not st.session_state.messages:
        st.markdown("""
        <div class="modern-card" style="text-align: center; background: linear-gradient(135deg, var(--surface), var(--surface-variant)); border: none;">
            <div style="font-size: 2.5rem; margin-bottom: 1rem;">👋</div>
            <h3 style="margin: 0; color: var(--text-primary);">Welcome to your AI Document Assistant</h3>
            <p style="color: var(--text-secondary); margin: 1rem 0;">
                Ask me anything about your uploaded documents, or have a general conversation. 
                I'll use your documents to provide more accurate and contextual answers.
            </p>
            <div style="display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap; margin-top: 1.5rem;">
                <div style="background: var(--surface-variant); padding: 0.5rem 1rem; border-radius: var(--radius-md); font-size: 0.875rem;">
                    💡 Ask about document content
                </div>
                <div style="background: var(--surface-variant); padding: 0.5rem 1rem; border-radius: var(--radius-md); font-size: 0.875rem;">
                    🔍 Search for specific information
                </div>
                <div style="background: var(--surface-variant); padding: 0.5rem 1rem; border-radius: var(--radius-md); font-size: 0.875rem;">
                    📊 Get summaries and insights
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Display chat history with modern styling
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            st.markdown(f"""
            <div style="display: flex; justify-content: flex-end; margin: 1.5rem 0;">
                <div style="max-width: 80%; background: linear-gradient(135deg, var(--primary-color) 0%, var(--accent-color) 100%); 
                           color: white; padding: 1rem 1.25rem; border-radius: var(--radius-lg); 
                           box-shadow: var(--shadow-md);">
                    <div style="font-weight: 500; margin-bottom: 0.25rem; opacity: 0.9; font-size: 0.875rem;">You</div>
                    <div style="line-height: 1.6;">{msg["content"]}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="display: flex; justify-content: flex-start; margin: 1.5rem 0;">
                <div style="max-width: 85%; background: var(--surface); border: 1px solid var(--border-color);
                           padding: 1rem 1.25rem; border-radius: var(--radius-lg); 
                           box-shadow: var(--shadow-sm);">
                    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                        <div style="font-size: 1.25rem;">🧠</div>
                        <div style="font-weight: 500; color: var(--text-primary); font-size: 0.875rem;">AI Assistant</div>
                        <button onclick="navigator.clipboard.writeText(document.getElementById('assistant-msg-{i}').innerText); this.innerText='Copied!'; setTimeout(()=>this.innerText='Copy',1500);" style="margin-left:auto; background: var(--primary-color); color:white; border:none; padding:2px 8px; border-radius:6px; cursor:pointer; font-size:0.65rem;">Copy</button>
                    </div>
                    <div id="assistant-msg-{i}" style="line-height: 1.7; color: var(--text-primary); white-space: pre-wrap;">{msg["content"]}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show context sources if available
            if "sources" in msg and msg["sources"]:
                exp_label = f"📚 Sources Used ({len(msg['sources'])} documents)"
                with st.expander(exp_label, expanded=False):
                    for idx, source in enumerate(msg["sources"], 1):
                        relevance_score = 1 - source['distance']
                        relevance_color = (
                            "var(--success-color)" if relevance_score > 0.8 else
                            "var(--warning-color)" if relevance_score > 0.6 else
                            "var(--error-color)"
                        )
                        st.markdown(f"""
                        <div class="modern-card" style="margin: 0.5rem 0; padding: 1rem;">
                            <div style="display: flex; justify-content: between; align-items: start; gap: 1rem;">
                                <div style="flex: 1;">
                                    <div style="font-weight: 600; color: var(--text-primary); margin-bottom: 0.5rem;">
                                        📄 {source['source']} 
                                        <span style="font-size: 0.75rem; background: var(--surface-variant); padding: 0.25rem 0.5rem; border-radius: var(--radius-sm); margin-left: 0.5rem;">
                                            Chunk {source['chunk_index']}
                                        </span>
                                    </div>
                                    <div style="color: var(--text-secondary); line-height: 1.6; font-size: 0.9rem;">
                                        {source['preview'][:300]}{'...' if len(source['preview']) > 300 else ''}
                                    </div>
                                </div>
                                <div style="text-align: center; min-width: 80px;">
                                    <div style="font-size: 0.75rem; color: var(--text-tertiary); margin-bottom: 0.25rem;">Relevance</div>
                                    <div style="font-weight: bold; color: {relevance_color}; font-size: 1rem;">
                                        {relevance_score:.0%}
                                    </div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                # Provide downloadable JSON for sources of this assistant message
                try:
                    sources_json = json.dumps(msg["sources"], ensure_ascii=False, indent=2)
                    st.download_button(
                        label=f"⬇️ Export Sources JSON (message {i+1})",
                        file_name=f"sources_message_{i+1}.json",
                        mime="application/json",
                        data=sources_json,
                        key=f"download_sources_{i}"
                    )
                except Exception as _e:
                    st.caption(f"Unable to export sources JSON: {_e}")

# --- 7. Handle User Input ---

def create_rag_prompt(user_query: str, context: str) -> str:
    """Create a prompt that includes retrieved context"""
    return f"""You are a helpful AI assistant. Use the following context from uploaded documents to answer the user's question. If the context doesn't contain relevant information, you can still provide a general response, but mention that you don't have specific information from the uploaded documents.

Context from uploaded documents:
{context}

User question: {user_query}

Please provide a helpful and accurate response based on the context above. If you use information from the context, mention which document it came from."""

def create_simple_prompt(user_query: str) -> str:
    """Create a simple prompt without RAG context"""
    return f"""You are a helpful AI assistant. Please answer the following question:

{user_query}"""

import math  # placed here to avoid reordering large header region
try:
    import tiktoken as _tiktoken  # optional precise token counting
    _encoder = _tiktoken.get_encoding("cl100k_base")
except Exception:  # noqa
    _tiktoken = None
    _encoder = None

# --- Message Limit & Token Tracking Setup (lightweight heuristic) ---
if "token_estimate_total" not in st.session_state:
    st.session_state.token_estimate_total = 0
if "message_count" not in st.session_state:
    st.session_state.message_count = 0

message_limit = int(os.getenv("CHAT_MESSAGE_LIMIT", "50"))
warn_threshold = max(1, int(message_limit * 0.8))

def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    if _encoder is not None:
        try:
            return len(_encoder.encode(text))
        except Exception:
            pass
    return max(1, math.ceil(len(text) / 4))  # fallback heuristic

limit_reached = st.session_state.message_count >= message_limit

st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)

if limit_reached:
    st.warning(f"Message limit of {message_limit} reached. Use 'Reset Chat' in sidebar to continue.")

prompt = None
chosen_msg_model = None
if not limit_reached:
    with st.container():
        cols = st.columns([3,1])
        with cols[0]:
            prompt = st.chat_input(
                "Ask me anything about your documents...",
                key="modern_chat_input"
            )
        with cols[1]:
            # Allow per-message model choice (reuse existing free model list)
            free_models = [
                "gemini-1.5-flash",
                "gemini-1.5-flash-8b",
            ]
            chosen_msg_model = st.selectbox(
                "Model",
                free_models,
                index=free_models.index(st.session_state.get("selected_model", free_models[0])) if st.session_state.get("selected_model", free_models[0]) in free_models else 0,
                help="Model to use for this next message"
            )

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "model": chosen_msg_model or st.session_state.get("selected_model")})
    st.session_state.message_count += 1
    st.session_state.token_estimate_total += estimate_tokens(prompt)
    if st.session_state.message_count == warn_threshold:
        st.toast(f"You've used {st.session_state.message_count}/{message_limit} messages (≈80%).", icon="⚠️")
    st.rerun()
    
# Process the latest message if it's from user and hasn't been responded to
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    latest_prompt = st.session_state.messages[-1]["content"]

    with st.empty():
        st.markdown("""
        <div style="display: flex; justify-content: flex-start; margin: 1.5rem 0;">
            <div style="background: var(--surface); border: 1px solid var(--border-color); padding: 1rem 1.25rem; border-radius: var(--radius-lg); box-shadow: var(--shadow-sm);">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <div style="font-size: 1.25rem;">🧠</div>
                    <div style="font-weight: 500; color: var(--text-primary); font-size: 0.875rem;">AI Assistant</div>
                </div>
                <div style="margin-top: 0.5rem; color: var(--text-secondary);">
                    <span style="animation: pulse 1.5s ease-in-out infinite;">Thinking...</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    reply_model = st.session_state.messages[-1].get("model") or st.session_state.get("selected_model", "gemini-1.5-flash")
    try:
        sources_used = []
        if use_rag:
            vector_db_info = st.session_state.vector_db.get_collection_info()
            if vector_db_info.get("document_count", 0) > 0:
                search_results = st.session_state.vector_db.similarity_search(latest_prompt, n_results=num_context_docs)
                if search_results:
                    context = st.session_state.vector_db.get_relevant_context(
                        latest_prompt,
                        n_results=num_context_docs,
                        max_context_length=max_context_length
                    )
                    for result in search_results:
                        sources_used.append({
                            "source": result["metadata"].get("source", "Unknown"),
                            "chunk_index": result["metadata"].get("chunk_index", "N/A"),
                            "distance": result["distance"],
                            "preview": result["document"]
                        })
                    final_prompt = create_rag_prompt(latest_prompt, context)
                else:
                    final_prompt = create_simple_prompt(latest_prompt)
            else:
                final_prompt = create_simple_prompt(latest_prompt)
        else:
            final_prompt = create_simple_prompt(latest_prompt)

        chat_obj = get_chat_for_model(reply_model)
        response = chat_obj.send_message(final_prompt)
        answer = response.text if hasattr(response, "text") else str(response)
        st.session_state.token_estimate_total += estimate_tokens(answer or "")
        st.session_state.message_count += 1
        message_data = {"role": "assistant", "content": answer, "model": reply_model}
        if sources_used:
            message_data["sources"] = sources_used
        st.session_state.messages.append(message_data)
        st.rerun()
    except Exception as e:
        error_message = f"I apologize, but I encountered an error: {str(e)}"
        st.session_state.messages.append({"role": "assistant", "content": error_message, "model": reply_model})
        st.session_state.message_count += 1
        st.session_state.token_estimate_total += estimate_tokens(error_message)
        st.rerun()

# --- 8. Enhanced Footer and Help Section ---

st.markdown("<div style='margin: 3rem 0;'></div>", unsafe_allow_html=True)

# Modern expandable help section
with st.expander("💡 How to Use This AI Assistant", expanded=False):
    st.markdown("""
    <div style="padding: 1rem 0;">
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 2rem; margin-bottom: 2rem;">
            <div class="modern-card">
                <h4 style="margin: 0 0 1rem 0; color: var(--primary-color); display: flex; align-items: center; gap: 0.5rem;">
                    <span>🚀</span> Getting Started
                </h4>
                <ol style="margin: 0; padding-left: 1.25rem; line-height: 1.8;">
                    <li>Enter your Google AI API key in the sidebar</li>
                    <li>Upload documents (PDF, DOCX, or TXT)</li>
                    <li>Click "Process Documents" to index them</li>
                    <li>Start asking questions about your content</li>
                </ol>
            </div>
            
            <div class="modern-card">
                <h4 style="margin: 0 0 1rem 0; color: var(--success-color); display: flex; align-items: center; gap: 0.5rem;">
                    <span>⚡</span> Pro Features
                </h4>
                <ul style="margin: 0; padding-left: 1.25rem; line-height: 1.8;">
                    <li><strong>Smart Search:</strong> AI-powered semantic search</li>
                    <li><strong>Source Attribution:</strong> See exactly which documents were used</li>
                    <li><strong>Flexible Context:</strong> Adjust how much context to use</li>
                    <li><strong>Multi-format:</strong> Support for various document types</li>
                </ul>
            </div>
        </div>
        
        <div class="modern-card" style="background: linear-gradient(135deg, var(--surface-variant), var(--surface)); border: none;">
            <h4 style="margin: 0 0 1rem 0; color: var(--accent-color); display: flex; align-items: center; gap: 0.5rem;">
                <span>💡</span> Tips for Best Results
            </h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;">
                <div style="background: var(--surface); padding: 1rem; border-radius: var(--radius-md); border: 1px solid var(--border-light);">
                    <strong>📝 Ask Specific Questions</strong><br>
                    <span style="color: var(--text-secondary); font-size: 0.9rem;">Instead of "What's in the document?", try "What are the main findings about X?"</span>
                </div>
                <div style="background: var(--surface); padding: 1rem; border-radius: var(--radius-md); border: 1px solid var(--border-light);">
                    <strong>📚 Upload Related Content</strong><br>
                    <span style="color: var(--text-secondary); font-size: 0.9rem;">Group related documents together for better contextual answers</span>
                </div>
                <div style="background: var(--surface); padding: 1rem; border-radius: var(--radius-md); border: 1px solid var(--border-light);">
                    <strong>🔍 Check Sources</strong><br>
                    <span style="color: var(--text-secondary); font-size: 0.9rem;">Review the source documents to verify and learn more</span>
                </div>
                <div style="background: var(--surface); padding: 1rem; border-radius: var(--radius-md); border: 1px solid var(--border-light);">
                    <strong>⚙️ Adjust Settings</strong><br>
                    <span style="color: var(--text-secondary); font-size: 0.9rem;">Fine-tune context length and document count for your needs</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Modern footer
st.markdown("""
<div style="margin-top: 4rem; padding: 2rem 0; border-top: 1px solid var(--border-light); text-align: center;">
    <div style="display: flex; justify-content: center; align-items: center; gap: 2rem; flex-wrap: wrap; margin-bottom: 1rem;">
        <div style="display: flex; align-items: center; gap: 0.5rem; color: var(--text-secondary);">
            <span style="font-size: 1.25rem;">🏗️</span>
            <span style="font-size: 0.9rem;">Built with Streamlit</span>
        </div>
        <div style="display: flex; align-items: center; gap: 0.5rem; color: var(--text-secondary);">
            <span style="font-size: 1.25rem;">🧠</span>
            <span style="font-size: 0.9rem;">Powered by Google Gemini</span>
        </div>
        <div style="display: flex; align-items: center; gap: 0.5rem; color: var(--text-secondary);">
            <span style="font-size: 1.25rem;">🔍</span>
            <span style="font-size: 0.9rem;">ChromaDB Vector Search</span>
        </div>
        <div style="display: flex; align-items: center; gap: 0.5rem; color: var(--text-secondary);">
            <span style="font-size: 1.25rem;">⚙️</span>
            <span style="font-size: 0.9rem;">LangChain Processing</span>
        </div>
    </div>
    <div style="color: var(--text-tertiary); font-size: 0.8rem;">
        © 2024 RAG Chatbot - Intelligent Document Assistant
    </div>
</div>
""", unsafe_allow_html=True)

# Add final CSS for animations that need to be loaded after content
st.markdown("""
<style>
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.modern-card {
    animation: fadeIn 0.3s ease-out;
}

/* Smooth scroll behavior */
html {
    scroll-behavior: smooth;
}

/* Enhanced focus states */
button:focus, input:focus, textarea:focus {
    outline: 2px solid var(--primary-color);
    outline-offset: 2px;
}

/* Loading states */
.stSpinner > div {
    border-color: var(--primary-color) transparent transparent transparent;
}

/* Better scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--surface-variant);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: var(--border-color);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--text-tertiary);
}
</style>
""", unsafe_allow_html=True)