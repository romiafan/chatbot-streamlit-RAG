"""
RAG-Enabled Chatbot with File Upload
Combines document processing, vector storage, and conversational AI
"""

import streamlit as st
from google import genai
import os
import time
import json  # for exporting sources as JSON
import math
import hashlib as _hashlib
from typing import List, Dict, Any, Optional

# Optional imports for enhanced functionality
try:
    import tiktoken as _tiktoken  # for precise token counting
except ImportError:
    _tiktoken = None

# Import our custom modules
from document_processor import DocumentProcessor, create_document_processor
from vector_database import VectorDatabase  # Removed missing get_vector_database (and unused display_vector_db_info)

# --- Website Configuration Constants ---
# Server-side API key management for website deployment
ADMIN_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_AI_API_KEY")  # Admin's API key
ALLOW_USER_KEYS = os.getenv("ALLOW_USER_API_KEYS", "true").lower() in {"true", "1", "yes"}
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() in {"true", "1", "yes"}

# Demo documents for showcasing capabilities
DEMO_DOCUMENTS = {
    "AI & Technology Overview": {
        "content": """
# Artificial Intelligence and Machine Learning Overview

## Introduction
Artificial Intelligence (AI) represents one of the most transformative technologies of our time. It encompasses machine learning, deep learning, natural language processing, and computer vision.

## Key Concepts
- **Machine Learning**: Algorithms that improve through experience
- **Deep Learning**: Neural networks with multiple layers
- **Natural Language Processing**: Understanding and generating human language
- **Computer Vision**: Interpreting and analyzing visual information

## Applications
AI is being applied across industries including healthcare, finance, transportation, and education. Common applications include:
- Predictive analytics and forecasting
- Automated decision making
- Image and speech recognition
- Recommendation systems
- Chatbots and virtual assistants

## Current Trends
- Large Language Models (LLMs)
- Generative AI
- Edge computing
- Responsible AI and ethics
- AI democratization

## Challenges
The field faces challenges including data privacy, algorithmic bias, computational requirements, and the need for explainable AI.
        """,
        "metadata": {"source": "ai_overview.md", "file_type": "markdown", "category": "technology"}
    },
    "RAG Systems Guide": {
        "content": """
# Retrieval-Augmented Generation (RAG) Systems

## What is RAG?
Retrieval-Augmented Generation combines the power of large language models with external knowledge bases to provide accurate, contextual responses.

## How RAG Works
1. **Document Ingestion**: Documents are processed and split into chunks
2. **Embedding Generation**: Text chunks are converted to vector embeddings
3. **Vector Storage**: Embeddings are stored in a vector database
4. **Query Processing**: User queries are embedded using the same model
5. **Similarity Search**: Relevant document chunks are retrieved
6. **Context Assembly**: Retrieved chunks are combined into context
7. **Response Generation**: LLM generates response using retrieved context

## Key Components
- **Document Processor**: Handles PDF, DOCX, TXT files
- **Embedding Model**: Converts text to vector representations
- **Vector Database**: Stores and searches embeddings (e.g., ChromaDB, Pinecone)
- **Language Model**: Generates responses (e.g., GPT, Claude, Gemini)

## Benefits
- Reduces hallucinations by grounding responses in real data
- Keeps information up-to-date without retraining models
- Provides source attribution for transparency
- Cost-effective compared to fine-tuning large models

## Best Practices
- Choose appropriate chunk sizes (typically 500-2000 characters)
- Use domain-specific embedding models when available
- Implement proper document preprocessing
- Monitor and evaluate response quality
- Provide clear source citations
        """,
        "metadata": {"source": "rag_guide.md", "file_type": "markdown", "category": "ai"}
    },
    "Streamlit Development": {
        "content": """
# Streamlit App Development Best Practices

## Introduction
Streamlit is a powerful framework for building data science and machine learning web applications with pure Python.

## Core Concepts
- **Session State**: Persist data across reruns
- **Caching**: Optimize performance with @st.cache_data and @st.cache_resource
- **Layout**: Use columns, containers, and sidebars for organization
- **Widgets**: Interactive elements like sliders, buttons, and inputs

## Project Structure
```
app.py              # Main application file
requirements.txt    # Dependencies
.streamlit/         # Configuration directory
├── config.toml     # App configuration
└── secrets.toml    # Sensitive data (local only)
pages/              # Multi-page apps
components/         # Custom components
```

## Performance Tips
- Use caching for expensive operations
- Minimize session state usage
- Implement proper error handling
- Use st.empty() for dynamic content updates
- Consider fragment-based caching for large apps

## Deployment Options
- Streamlit Cloud (free tier available)
- Heroku, AWS, GCP, Azure
- Docker containers
- Local development servers

## Security Considerations
- Never commit secrets to version control
- Use environment variables for sensitive data
- Implement proper input validation
- Consider authentication for sensitive apps
- Use HTTPS in production

## Advanced Features
- Custom components with React/HTML
- State management patterns
- Real-time data updates
- Integration with databases
- API development with FastAPI
        """,
        "metadata": {"source": "streamlit_guide.md", "file_type": "markdown", "category": "development"}
    }
}

# Fallback factory: original get_vector_database not found in vector_database module
@st.cache_resource
def get_vector_database(collection_name: str = "rag_chatbot_docs"):
    """
    Cached factory returning a VectorDatabase instance.

    This substitutes the missing get_vector_database import while preserving
    the existing call pattern elsewhere in the app.
    """
    try:
        # Common constructor pattern
        return VectorDatabase(collection_name)
    except TypeError:
        # Alternate named parameter
        try:
            return VectorDatabase(collection_name=collection_name)
        except Exception:
            # Last-resort no-arg construction
            return VectorDatabase()

def load_demo_documents():
    """Load and process demo documents into the vector database"""
    from langchain.docstore.document import Document as LangChainDocument
    
    demo_docs = []
    for title, doc_data in DEMO_DOCUMENTS.items():
        # Create chunks from the demo content
        content = doc_data["content"].strip()
        metadata = doc_data["metadata"].copy()
        metadata["demo_document"] = True
        metadata["title"] = title
        
        # Split long documents into chunks
        if len(content) > 1500:
            # Simple chunk splitting for demo
            chunks = []
            lines = content.split('\n')
            current_chunk = []
            current_length = 0
            
            for line in lines:
                line_length = len(line) + 1  # +1 for newline
                if current_length + line_length > 1500 and current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = [line]
                    current_length = line_length
                else:
                    current_chunk.append(line)
                    current_length += line_length
            
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
            
            # Create documents for each chunk
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = i
                chunk_metadata["chunk_size"] = len(chunk)
                demo_docs.append(LangChainDocument(page_content=chunk, metadata=chunk_metadata))
        else:
            # Single document
            metadata["chunk_index"] = 0
            metadata["chunk_size"] = len(content)
            demo_docs.append(LangChainDocument(page_content=content, metadata=metadata))
    
    return demo_docs

def safe_api_call(func, fallback_message="Service temporarily unavailable", show_error=True):
    """Wrapper for API calls with graceful error handling"""
    try:
        return func()
    except Exception as e:
        error_msg = str(e).lower()
        
        # Categorize common errors
        if "api key" in error_msg or "authentication" in error_msg:
            user_message = "🔑 API key invalid or expired. Please check your credentials."
        elif "quota" in error_msg or "limit" in error_msg:
            user_message = "⚠️ API quota exceeded. Please try again later or use your own API key."
        elif "network" in error_msg or "connection" in error_msg:
            user_message = "🌐 Network connection issue. Please check your internet connection."
        elif "timeout" in error_msg:
            user_message = "⏱️ Request timed out. Please try again."
        else:
            user_message = f"⚠️ {fallback_message}"
        
        if show_error:
            st.error(user_message)
        
        # Log the actual error for debugging (in production, use proper logging)
        if os.getenv("DEBUG", "false").lower() in {"true", "1", "yes"}:
            st.error(f"Debug: {str(e)}")
        
        return None

def handle_file_processing_error(error, filename=""):
    """Specific error handling for file processing"""
    error_msg = str(error).lower()
    
    if "password" in error_msg or "encrypted" in error_msg:
        return f"🔒 File '{filename}' is password-protected or encrypted"
    elif "corrupt" in error_msg or "invalid" in error_msg:
        return f"❌ File '{filename}' appears to be corrupted or invalid"
    elif "size" in error_msg or "large" in error_msg:
        return f"📏 File '{filename}' is too large to process"
    elif "format" in error_msg or "unsupported" in error_msg:
        return f"🚫 File format not supported for '{filename}'"
    else:
        return f"⚠️ Could not process '{filename}': {str(error)}"

# --- Modern CSS Styling ---
def apply_custom_css():
    """Apply clean, working CSS styling"""
    # Get theme configuration safely
    primary_color = "#2563eb"
    hide_sidebar = False
    
    if 'theme_config' in globals():
        primary_color = theme_config.get("primary_color", "#2563eb")
        hide_sidebar = theme_config.get("hide_sidebar", False)
    
    st.markdown(f"""
    <style>
    /* Hide Streamlit UI elements */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    .stDeployButton {{display: none;}}
    
    /* App base styling */
    .stApp {{
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    
    /* Sidebar */
    {".stSidebar { display: none !important; }" if hide_sidebar else ""}
    
    /* Custom header */
    .custom-header {{
        background: linear-gradient(135deg, {primary_color} 0%, #3b82f6 100%);
        padding: 2rem;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 1rem 1rem;
        text-align: center;
        color: white;
    }}
    
    .custom-header h1 {{
        margin: 0;
        font-size: 2.5rem;
        font-weight: 600;
    }}
    
    .custom-header p {{
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }}
    
    /* Modern cards */
    .modern-card {{
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin: 1rem 0;
    }}
    
    /* Status badges */
    .status-badge {{
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.75rem;
        font-weight: 500;
    }}
    
    .status-success {{
        background: #d1fae5;
        color: #065f46;
    }}
    
    /* Buttons */
    .stButton button {{
        background: {primary_color};
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s;
    }}
    
    .stButton button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }}
    
    /* Mobile responsive */
    @media (max-width: 768px) {{
        .main .block-container {{
            padding: 0.5rem;
        }}
        
        .custom-header {{
            margin: -0.5rem -0.5rem 1rem -0.5rem;
            padding: 1.5rem;
        }}
        
        .custom-header h1 {{
            font-size: 2rem;
        }}
    }}
    
    </style>
    """, unsafe_allow_html=True)


def load_demo_documents():
    """Load and process demo documents into the vector database"""
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
        min-height: 44px; /* Touch-friendly minimum */
    }
    
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: var(--shadow-md);
        background: linear-gradient(135deg, var(--primary-hover) 0%, var(--primary-color) 100%);
    }
    
    /* Touch-friendly improvements */
    .stSelectbox, .stTextInput, .stTextArea {
        min-height: 44px;
    }
    
    /* File Uploader */
    .stFileUploader {
        border: 2px dashed var(--border-color);
        border-radius: var(--radius-lg);
        padding: 2rem;
        text-align: center;
        background: var(--surface-variant);
        transition: all 0.2s ease;
        min-height: 120px; /* Touch-friendly drop zone */
    }
    
    .stFileUploader:hover {
        border-color: var(--primary-color);
        background: var(--surface);
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
    
    /* Loading animations */
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
    </style>
    """, unsafe_allow_html=True)
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

# Determine embed / compact mode and theme configuration (query param or env)
embed_mode = False
theme_config = {
    "primary_color": "#2563eb",
    "background_mode": "light",
    "compact": False,
    "hide_sidebar": False,
    "readonly": False
}

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
        
        # Theme customization parameters
        if qp.get('theme'):
            theme_config["background_mode"] = qp.get('theme', 'light')
        if qp.get('color'):
            theme_config["primary_color"] = qp.get('color', '#2563eb')
        if qp.get('compact') in {"1", "true", "yes"}:
            theme_config["compact"] = True
        if qp.get('hide_sidebar') in {"1", "true", "yes"}:
            theme_config["hide_sidebar"] = True
        if qp.get('readonly') in {"1", "true", "yes"}:
            theme_config["readonly"] = True
            
except Exception:
    pass

if not embed_mode:
    # Allow env override when query param absent
    if os.getenv("EMBED_MODE", "").lower() in {"1", "true", "yes"}:
        embed_mode = True

# Apply custom CSS after theme config is ready
apply_custom_css()

if not embed_mode:
    # Custom Modern Header (suppressed in embed mode for tighter iframe usage)
    st.markdown("""
    <div class="custom-header">
        <h1>🧠 AI Document Assistant</h1>
        <p>Intelligent conversations powered by your documents</p>
    </div>
    """, unsafe_allow_html=True)
    # Quick How-To (top placement)
    st.markdown(
        """
        <div class="modern-card" style="margin-top: -0.5rem; border: 1px solid var(--border-color);">
            <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.75rem;">
                <span style="font-size:1.5rem;">🛠️</span>
                <h3 style="margin:0; font-size:1.1rem;">How to Use This AI Assistant</h3>
            </div>
            <ol style="margin:0; padding-left:1.1rem; line-height:1.6; font-size:0.9rem; color:var(--text-secondary);">
                <li><strong>Add API Key</strong> in the sidebar (Google AI).</li>
                <li><strong>Upload documents</strong> (PDF / DOCX / TXT).</li>
                <li>Click <strong>Process Documents</strong> to index chunks.</li>
                <li><strong>Ask questions</strong> – answers cite document sources.</li>
                <li>Use the <strong>model selector</strong> & <strong>RAG toggles</strong> to refine responses.</li>
            </ol>
            <div style="margin-top:0.75rem; display:flex; flex-wrap:wrap; gap:0.5rem; font-size:0.7rem;">
                <span style="background:var(--surface-variant); padding:4px 8px; border-radius:6px;">🔍 Semantic Search</span>
                <span style="background:var(--surface-variant); padding:4px 8px; border-radius:6px;">📚 Source Attribution</span>
                <span style="background:var(--surface-variant); padding:4px 8px; border-radius:6px;">⚙️ Adjustable Context</span>
                <span style="background:var(--surface-variant); padding:4px 8px; border-radius:6px;">🧪 Model Fallback</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    # Compact top spacer
    st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)

# --- 2. Modern Sidebar Configuration ---

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # API Key Section
    with st.container():
        st.markdown("#### 🔑 Authentication")
        
        # Show different UI based on configuration
        if ALLOW_USER_KEYS:
            if ADMIN_API_KEY:
                st.markdown("💡 *Admin API key available for demo mode*")
            
            google_api_key = st.text_input(
                "Google AI API Key (Optional)" if ADMIN_API_KEY else "Google AI API Key", 
                type="password", 
                help="Enter your Google AI API key for unlimited usage" if ADMIN_API_KEY else "Enter your Google AI API key to enable the chatbot",
                placeholder="Enter your API key..." if not ADMIN_API_KEY else "Optional: Use your own API key"
            )
            
            if ADMIN_API_KEY and not google_api_key:
                st.caption("🎯 Using demo mode with limited usage")
        else:
            if ADMIN_API_KEY:
                st.info("🔑 Using pre-configured API key")
                google_api_key = None  # Force use of admin key
            else:
                st.error("❌ API key configuration required")
                google_api_key = None
    
    # Dynamic model discovery & selection
    with st.container():
        st.markdown("#### 🤖 Model")

        def _discover_models(force: bool = False):
            """Populate st.session_state.available_models with discovered Gemini flash models.
            Adds debug info (raw list, filtered list, timestamp, last error) and supports forcing refresh."""
            if ("available_models" in st.session_state) and not force:
                return
            # Support legacy alias DEFAULT_MODEL if GEMINI_DEFAULT_MODEL not set
            env_default = os.getenv("GEMINI_DEFAULT_MODEL") or os.getenv("DEFAULT_MODEL")
            # Base fallback now includes 2.5 + 1.5 (priority order updated)
            base_fallback = ["gemini-2.5-flash", "gemini-2.5-flash-8b", "gemini-1.5-flash", "gemini-1.5-flash-8b"]
            if env_default:
                env_base = env_default.split("/", 1)[1] if env_default.startswith("models/") else env_default
                if env_base not in base_fallback:
                    base_fallback.insert(0, env_base)
            models = []
            raw_models = []
            last_error = None
            try:
                client = st.session_state.get("genai_client")
                if client is not None:
                    raw_list = list(client.models.list())  # type: ignore[attr-defined]
                    for m in raw_list:
                        name = getattr(m, "name", "") or ""
                        raw_models.append(name)
                        if name.startswith("models/"):
                            name = name.split("/", 1)[1]
                        lname = name.lower()
                        if "flash" in lname and not any(x in lname for x in ["pro", "vision", "exp"]):
                            models.append(name)
            except Exception as e:  # noqa: BLE001
                last_error = str(e)
                st.session_state._model_discovery_error = str(e)
            # Normalize & fallback
            models = sorted(set(models))
            if not models:
                models = base_fallback
            def _ver_key(model_name: str):
                try:
                    parts = model_name.split("-")
                    for p in parts:
                        if p.replace(".", "").isdigit() and "." in p:
                            maj, min_ = p.split(".")
                            return (int(maj), int(min_))
                    return (0, 0)
                except Exception:
                    return (0, 0)
            models_sorted = sorted(models, key=_ver_key, reverse=True)
            ordered = []
            for bf in base_fallback:
                if bf in models_sorted and bf not in ordered:
                    ordered.append(bf)
            for m in models_sorted:
                if m not in ordered:
                    ordered.append(m)
            st.session_state.available_models = ordered
            st.session_state.model_discovery_debug = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "env_default": env_default,
                "raw_models": raw_models[:50],  # limit for display
                "filtered_models": models_sorted,
                "final_available": ordered,
                "last_error": last_error,
                "forced": force,
            }

        # Icon-only refresh button with tooltip
        refresh_cols = st.columns([5,1])
        with refresh_cols[1]:
            if st.button("🔄", help="Refresh model list", key="refresh_models_btn"):
                _discover_models(force=True)
        _discover_models(force=False)

        available_models = st.session_state.get("available_models", ["gemini-2.5-flash", "gemini-1.5-flash"])

        # Attempt to honor environment default
        env_default = os.getenv("GEMINI_DEFAULT_MODEL") or os.getenv("DEFAULT_MODEL")
        if env_default:
            env_base = env_default.split("/", 1)[1] if env_default.startswith("models/") else env_default
            if env_base not in available_models:
                relaxed = env_base.replace("-latest", "")
                mapped = next((m for m in available_models if m.startswith(relaxed)), None)
                if mapped:
                    env_base = mapped
            if env_base in available_models and available_models[0] != env_base:
                available_models = [env_base] + [m for m in available_models if m != env_base]
        # Initialize selection
        if "selected_model" not in st.session_state:
            st.session_state.selected_model = available_models[0]
        elif st.session_state.selected_model not in available_models:
            # Preserve legacy/previous value by appending so selector can show it
            available_models = available_models + [st.session_state.selected_model]

        chosen_model = st.selectbox(
            "Gemini Model",
            options=available_models,
            index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0,
            help="Choose an available Gemini Flash model. List is discovered dynamically.",
        )

        model_warning = None
        if env_default and st.session_state.selected_model != env_default and env_default not in available_models:
            model_warning = f"Env default '{env_default}' not found; using '{st.session_state.selected_model}'."
        if hasattr(st.session_state, "_model_discovery_error"):
            model_warning = f"Model discovery issue: {getattr(st.session_state, '_model_discovery_error')} (showing fallback list)"
        if st.session_state.selected_model not in st.session_state.get("available_models", []):
            suggestions = ", ".join(st.session_state.get("available_models", [])[:3])
            model_warning = f"Selected model '{st.session_state.selected_model}' not in discovered list. Suggestions: {suggestions}"
        if model_warning:
            st.info(f"ℹ️ {model_warning}")

        # Debug expander (optional visibility into discovery)
        with st.expander("🔍 Model Discovery Debug", expanded=False):
            dbg = st.session_state.get("model_discovery_debug", {})
            if not dbg:
                st.write("No discovery data yet.")
            else:
                st.write({
                    "env_default": dbg.get("env_default"),
                    "timestamp": dbg.get("timestamp"),
                    "forced": dbg.get("forced"),
                    "last_error": dbg.get("last_error"),
                })
                st.write("Raw models (truncated):", dbg.get("raw_models"))
                st.write("Filtered models:", dbg.get("filtered_models"))
                st.write("Final available:", dbg.get("final_available"))

        if chosen_model != st.session_state.selected_model:
            st.session_state.selected_model = chosen_model
            # Reset chats & messages for consistency
            st.session_state.pop("chat", None)
            st.session_state.pop("messages", None)
            st.session_state.pop("model_chats", None)
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

# --- 3. Enhanced API Key Management ---

# Server-side API key management for website deployment
ADMIN_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_AI_API_KEY")  # Admin's API key
ALLOW_USER_KEYS = os.getenv("ALLOW_USER_API_KEYS", "true").lower() in {"true", "1", "yes"}
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() in {"true", "1", "yes"}

# Determine which API key to use
effective_api_key = None
api_key_source = None

if ADMIN_API_KEY and (DEMO_MODE or not google_api_key):
    # Use admin key for demo mode or when user hasn't provided one
    effective_api_key = ADMIN_API_KEY
    api_key_source = "admin"
    if DEMO_MODE:
        st.info("🎯 **Demo Mode**: Using pre-configured API key. Some features may have usage limits.", icon="ℹ️")
elif google_api_key:
    # Use user-provided key
    effective_api_key = google_api_key
    api_key_source = "user"
else:
    # No API key available
    if ALLOW_USER_KEYS:
        st.info("👈 Please add your Google AI API key in the sidebar to start chatting.", icon="🗝️")
    else:
        st.error("🔒 API key required. Please contact the administrator.", icon="❌")
    st.stop()

# Initialize Google AI client
client_key = f"{effective_api_key}_{api_key_source}"
if ("genai_client" not in st.session_state) or (getattr(st.session_state, "_last_client_key", None) != client_key):
    try:
        st.session_state.genai_client = genai.Client(api_key=effective_api_key)
        st.session_state._last_client_key = client_key
        st.session_state._api_key_source = api_key_source
        st.session_state.pop("chat", None)
        st.session_state.pop("messages", None)
        
        # Show API key source in debug mode
        if not embed_mode and api_key_source == "admin":
            st.success("✅ Connected with admin API key")
        elif not embed_mode and api_key_source == "user":
            st.success("✅ Connected with your API key")
            
    except Exception as e:
        if api_key_source == "user":
            st.error(f"❌ Invalid API Key: {e}")
        else:
            st.error(f"❌ Service temporarily unavailable: {e}")
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

# --- 5. Modern File Upload Section with Demo Mode ---

if not embed_mode:
    st.markdown("### 📁 Document Upload")

# Create a modern upload interface
upload_container = st.container()
with upload_container:
    # Demo mode section
    if DEMO_MODE or ADMIN_API_KEY:
        st.markdown("""
        <div class="modern-card" style="background: linear-gradient(135deg, #f0f9ff, #e0f2fe); border: 2px solid #0ea5e9;">
            <div style="text-align: center; padding: 1rem 0;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🎯</div>
                <h4 style="margin: 0; color: var(--text-primary);">Try Demo Documents</h4>
                <p style="color: var(--text-secondary); margin: 0.5rem 0 1rem 0;">
                    Explore capabilities with pre-loaded sample documents
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Demo document selector
        demo_cols = st.columns(3)
        
        for i, (title, doc_data) in enumerate(DEMO_DOCUMENTS.items()):
            col_idx = i % 3
            with demo_cols[col_idx]:
                category = doc_data["metadata"].get("category", "general")
                emoji = {"technology": "🤖", "ai": "🧠", "development": "💻"}.get(category, "📄")
                
                if st.button(
                    f"{emoji} {title}",
                    key=f"demo_{i}",
                    help=f"Load sample document about {title.lower()}",
                    use_container_width=True
                ):
                    # Load this specific demo document
                    from langchain.docstore.document import Document as LangChainDocument
                    
                    content = doc_data["content"].strip()
                    metadata = doc_data["metadata"].copy()
                    metadata["demo_document"] = True
                    metadata["title"] = title
                    metadata["chunk_index"] = 0
                    metadata["chunk_size"] = len(content)
                    
                    demo_doc = LangChainDocument(page_content=content, metadata=metadata)
                    
                    with st.spinner(f"Loading {title}..."):
                        success = st.session_state.vector_db.add_documents([demo_doc])
                        if success:
                            st.success(f"✅ Loaded demo document: {title}")
                            st.rerun()
                        else:
                            st.error(f"❌ Failed to load {title}")
        
        # Load all demo documents button
        if st.button("🚀 Load All Demo Documents", type="primary", use_container_width=True):
            with st.spinner("Loading all demo documents..."):
                demo_docs = load_demo_documents()
                success = st.session_state.vector_db.add_documents(demo_docs)
                if success:
                    st.success(f"✅ Loaded {len(demo_docs)} demo document chunks!")
                    st.rerun()
                else:
                    st.error("❌ Failed to load demo documents")
        
        st.markdown("---")
    
    # Regular file upload section
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
                    # Process the uploaded files with enhanced error handling
                    status_text.text("📤 Uploading and extracting text...")
                    progress_bar.progress(25)
                    
                    # Use safe API call for document processing
                    documents = safe_api_call(
                        lambda: st.session_state.doc_processor.process_uploaded_files(uploaded_files),
                        "Document processing service unavailable",
                        show_error=False  # Handle errors manually for better UX
                    )
                    
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
                        processing_errors = []

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
                        except Exception as e:
                            # Non-critical error - continue without existing hash check
                            if os.getenv("DEBUG", "false").lower() in {"true", "1", "yes"}:
                                st.warning(f"⚠️ Could not check for existing duplicates: {str(e)}")

                        # Process documents with individual error handling
                        for i, d in enumerate(documents):
                            try:
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
                                meta_attr = getattr(d, 'metadata', None)
                                if meta_attr is not None and isinstance(meta_attr, dict):
                                    meta_attr['chunk_hash'] = h
                                elif isinstance(d, dict):
                                    d.setdefault('metadata', {})
                                    if isinstance(d['metadata'], dict):
                                        d['metadata']['chunk_hash'] = h
                                unique_docs.append(d)
                            except Exception as e:
                                processing_errors.append(f"Chunk {i+1}: {str(e)}")
                                continue

                        # Add documents to vector database with error handling
                        success = True
                        if unique_docs:
                            success = safe_api_call(
                                lambda: st.session_state.vector_db.add_documents(unique_docs),
                                "Vector database storage unavailable",
                                show_error=False
                            )
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Processing complete!")
                        
                        if success:
                            added = len(unique_docs)
                            total = len(documents) if documents else 0
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
                                        <div style="font-size: 1.5rem; font-weight: bold;">{total}</div>
                                        <div style="font-size: 0.875rem; opacity: 0.9;">Chunks</div>
                                    </div>
                                    <div>
                                        <div style="font-size: 1.5rem; font-weight: bold;">{total_size:.1f}</div>
                                        <div style="font-size: 0.875rem; opacity: 0.9;">MB</div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show processing warnings if any
                            if processing_errors:
                                with st.expander(f"⚠️ {len(processing_errors)} processing warnings", expanded=False):
                                    for error in processing_errors[:10]:  # Limit to first 10 errors
                                        st.warning(error)
                                    if len(processing_errors) > 10:
                                        st.info(f"... and {len(processing_errors) - 10} more issues")
                        else:
                            st.error("❌ Failed to add documents to vector database")
                    else:
                        st.warning("⚠️ No text could be extracted from the uploaded files")
                        
                except Exception as e:
                    error_msg = handle_file_processing_error(e)
                    st.error(error_msg)
                    if os.getenv("DEBUG", "false").lower() in {"true", "1", "yes"}:
                        st.exception(e)
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
    """Return (or lazily create) a chat object for a given model with extended fallback logic.
    Order of attempts:
      1. Requested model (raw & prefixed)
      2. Env default (raw & prefixed) if different
      3. 2.5 flash variants (standard & 8b) raw & prefixed
      4. 1.5 flash variants (standard & 8b) raw & prefixed
    Records resolution mapping so UI can show requested → actual if they differ."""
    if model in st.session_state.model_chats:
        return st.session_state.model_chats[model]

    client = st.session_state.genai_client
    tried = []
    candidates = []

    def _norm(name: str):
        return name.split("/", 1)[1] if name.startswith("models/") else name
    def _add_pair(name: str):
        base = _norm(name)
        raw = base
        pref = f"models/{base}"
        if raw not in candidates:
            candidates.append(raw)
        if pref not in candidates:
            candidates.append(pref)

    requested_base = _norm(model)
    _add_pair(requested_base)

    env_default = os.getenv("GEMINI_DEFAULT_MODEL")
    if env_default:
        env_base = _norm(env_default)
        if env_base != requested_base:
            _add_pair(env_base)

    for fallback in ["gemini-2.5-flash", "gemini-2.5-flash-8b", "gemini-1.5-flash", "gemini-1.5-flash-8b"]:
        if fallback != requested_base:
            _add_pair(fallback)

    last_err = None
    for candidate in candidates:
        if candidate in tried:
            continue
        tried.append(candidate)
        try:
            chat_obj = client.chats.create(model=candidate)
            actual = _norm(candidate)
            st.session_state.model_chats[model] = chat_obj
            # Track resolution mapping
            if "_model_resolution" not in st.session_state:
                st.session_state._model_resolution = {}
            st.session_state._model_resolution[model] = actual
            # Attach attribute for quick access
            try:
                chat_obj._actual_model = actual  # type: ignore[attr-defined]
            except Exception:
                pass
            return chat_obj
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise RuntimeError(f"Failed to create chat for model '{model}'. Tried: {tried}. Last error: {last_err}")

# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages using Streamlit's native chat components
if not st.session_state.messages:
    st.markdown("""
    <div class="modern-card" style="text-align: center; background: linear-gradient(135deg, var(--surface), var(--surface-variant)); border: none; margin: 2rem 0;">
        <div style="font-size: 2.5rem; margin-bottom: 1rem;">👋</div>
        <h3 style="margin: 0; color: var(--text-primary);">Welcome to your AI Document Assistant</h3>
        <p style="color: var(--text-secondary); margin: 1rem 0;">
            Upload documents and ask questions to get started. I'll use your documents to provide more accurate and contextual answers.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Display chat history with native Streamlit components
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    else:
        actual_model = msg.get("model") or st.session_state.get("selected_model", "gemini-1.5-flash")
        requested_model = msg.get("requested_model", actual_model)
        display_model = actual_model if requested_model == actual_model else f"{requested_model}→{actual_model}"
        
        with st.chat_message("assistant"):
            # Show model info
            if requested_model != actual_model:
                st.caption(f"🤖 {display_model} (fallback occurred)")
            else:
                st.caption(f"🤖 {display_model}")
            
            # Show message content with error handling
            if msg.get("error"):
                # Special styling for error messages
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #fee2e2, #fecaca); 
                           border: 1px solid #f87171; border-radius: 0.5rem; 
                           padding: 1rem; margin: 0.5rem 0;">
                    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                        <span style="font-size: 1.25rem;">⚠️</span>
                        <span style="font-weight: 600; color: #dc2626;">Service Error</span>
                    </div>
                    <div style="color: #7f1d1d;">{msg["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.write(msg["content"])
            
            # Add copy button for assistant messages
            if st.button(f"📋 Copy Response", key=f"copy_msg_{i}", help="Copy assistant response to clipboard"):
                st.write("Response copied! (Use Ctrl+C/Cmd+C to copy from the text above)")
                st.code(msg["content"], language=None)

# Show sources blocks (outside scroll to keep window performant)
for i, msg in enumerate(st.session_state.messages):
    if msg.get("role") == "assistant" and msg.get("sources"):
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
                <div class='modern-card' style='margin:0.5rem 0; padding:1rem;'>
                    <div style='display:flex; justify-content:between; align-items:start; gap:1rem;'>
                        <div style='flex:1;'>
                            <div style='font-weight:600; color:var(--text-primary); margin-bottom:0.5rem;'>📄 {source['source']} <span style='font-size:0.7rem; background:var(--surface-variant); padding:2px 6px; border-radius:4px;'>Chunk {source['chunk_index']}</span></div>
                            <div style='color:var(--text-secondary); font-size:0.8rem; line-height:1.5;'>{source['preview'][:300]}{'...' if len(source['preview'])>300 else ''}</div>
                        </div>
                        <div style='text-align:center; min-width:70px;'>
                            <div style='font-size:0.6rem; color:var(--text-tertiary); margin-bottom:0.25rem;'>Relevance</div>
                            <div style='font-weight:700; color:{relevance_color}; font-size:0.9rem;'>{relevance_score:.0%}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
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
            # Allow per-message model choice using discovered list
            discovered = st.session_state.get("available_models") or [st.session_state.get("selected_model", "gemini-1.5-flash")]
            base_sel = st.session_state.get("selected_model") or discovered[0]
            idx = discovered.index(base_sel) if base_sel in discovered else 0
            chosen_msg_model = st.selectbox(
                "Model",
                discovered,
                index=idx,
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

        # Generate response with professional error handling
        chat_obj = get_chat_for_model(reply_model)
        
        def generate_response():
            response = chat_obj.send_message(final_prompt)
            return response.text if hasattr(response, "text") else str(response)
        
        answer = safe_api_call(
            generate_response,
            "AI service temporarily unavailable. Please try again in a moment.",
            show_error=False
        )
        
        if answer:
            st.session_state.token_estimate_total += estimate_tokens(answer)
            st.session_state.message_count += 1
            # Resolve actual model (fallback may have occurred)
            chat_actual = getattr(st.session_state.model_chats.get(reply_model, {}), "_actual_model", reply_model)
            message_data = {"role": "assistant", "content": answer, "model": chat_actual, "requested_model": reply_model}
            if sources_used:
                message_data["sources"] = sources_used
            st.session_state.messages.append(message_data)
        else:
            # Fallback response when API fails
            error_message = "I'm experiencing technical difficulties right now. Please try again in a moment, or check your API key if you're using a custom one."
            st.session_state.messages.append({
                "role": "assistant", 
                "content": error_message, 
                "model": reply_model,
                "error": True
            })
            st.session_state.message_count += 1
            st.session_state.token_estimate_total += estimate_tokens(error_message)
        
        st.rerun()
    except Exception as e:
        error_message = f"I apologize, but I encountered an error: {str(e)}"
        st.session_state.messages.append({"role": "assistant", "content": error_message, "model": reply_model})
        st.session_state.message_count += 1
        st.session_state.token_estimate_total += estimate_tokens(error_message)
        st.rerun()

# --- 8. Enhanced Footer and Help Section ---

st.markdown("<div style='margin: 3rem 0;'></div>", unsafe_allow_html=True)


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

# Inject a small JS snippet once to handle all copy buttons reliably
st.markdown(
    """
    <script>
    (function() {
        function attach() {
            document.querySelectorAll('.copy-btn').forEach(btn => {
                if (btn.dataset._bound) return;
                btn.dataset._bound = '1';
                btn.addEventListener('click', async () => {
                    const targetId = btn.getAttribute('data-target');
                    const el = document.getElementById(targetId);
                    if (!el) return;
                    try {
                        await navigator.clipboard.writeText(el.innerText);
                        const original = btn.innerText;
                        btn.innerText = 'Copied!';
                        btn.disabled = true;
                        setTimeout(()=>{ btn.innerText = original; btn.disabled = false; }, 1400);
                    } catch(e) {
                        console.warn('Copy failed', e);
                        btn.innerText = 'Failed';
                        setTimeout(()=>{ btn.innerText = 'Copy'; }, 1400);
                    }
                });
            });
        }
        // Initial attach & on mutation (Streamlit re-renders root frequently)
        const observer = new MutationObserver(() => attach());
        observer.observe(document.body, { childList: true, subtree: true });
        attach();
    })();
    </script>
    """,
    unsafe_allow_html=True
)