"""
RAG-Enabled Chatbot with File Upload
Combines document processing, vector storage, and conversational AI
"""

import streamlit as st
from google import genai
import time
from typing import List, Dict, Any, Optional

# Import our custom modules
from document_processor import DocumentProcessor, create_document_processor
from vector_database import VectorDatabase, get_vector_database, display_vector_db_info

# --- 1. Page Configuration and Title ---

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 RAG-Enabled Chatbot")
st.caption("Upload documents and chat with AI using Retrieval-Augmented Generation")

# --- 2. Sidebar Configuration ---

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key input
    google_api_key = st.text_input(
        "Google AI API Key", 
        type="password", 
        help="Enter your Google AI API key to enable the chatbot"
    )
    
    # RAG Settings
    st.subheader("📚 RAG Settings")
    
    use_rag = st.checkbox(
        "Enable RAG", 
        value=True, 
        help="Use uploaded documents to enhance responses"
    )
    
    num_context_docs = st.slider(
        "Context Documents", 
        min_value=1, 
        max_value=10, 
        value=3, 
        help="Number of relevant documents to use as context"
    )
    
    max_context_length = st.slider(
        "Max Context Length", 
        min_value=500, 
        max_value=4000, 
        value=2000, 
        help="Maximum length of context from documents"
    )
    
    # Document Management
    st.subheader("📄 Document Management")
    
    if st.button("🗑️ Clear All Documents", help="Remove all uploaded documents"):
        if "vector_db" in st.session_state:
            st.session_state.vector_db.clear_collection()
            st.rerun()
    
    # Reset conversation
    if st.button("🔄 Reset Conversation", help="Clear chat history"):
        st.session_state.pop("chat", None)
        st.session_state.pop("messages", None)
        st.rerun()

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
    st.session_state.doc_processor = create_document_processor()

# Initialize vector database
if "vector_db" not in st.session_state:
    st.session_state.vector_db = get_vector_database("rag_chatbot_docs")

# --- 5. File Upload Section ---

st.header("📁 Document Upload")

# File uploader
uploaded_files = st.file_uploader(
    "Upload documents to enhance the chatbot's knowledge",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    help="Supported formats: PDF, DOCX, TXT"
)

# Process uploaded files
if uploaded_files:
    if st.button("🔄 Process Documents", type="primary"):
        with st.spinner("Processing documents..."):
            # Process the uploaded files
            documents = st.session_state.doc_processor.process_uploaded_files(uploaded_files)
            
            if documents:
                # Add to vector database
                success = st.session_state.vector_db.add_documents(documents)
                
                if success:
                    st.success(f"✅ Successfully processed {len(uploaded_files)} files and created {len(documents)} document chunks!")
                else:
                    st.error("❌ Failed to add documents to vector database")
            else:
                st.warning("⚠️ No text could be extracted from the uploaded files")

# Display vector database info
display_vector_db_info(st.session_state.vector_db)

# --- 6. Chat Interface ---

st.header("💬 Chat")

# Initialize chat session
if "chat" not in st.session_state:
    st.session_state.chat = st.session_state.genai_client.chats.create(model="gemini-2.5-flash")

# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Show context sources if available
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📚 Sources Used"):
                for i, source in enumerate(msg["sources"], 1):
                    st.write(f"**{i}. {source['source']}** (Chunk {source['chunk_index']})")
                    st.write(f"*Relevance: {1 - source['distance']:.2%}*")
                    st.write(source["preview"][:200] + "..." if len(source["preview"]) > 200 else source["preview"])
                    st.write("---")

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

# Chat input
prompt = st.chat_input("Ask me anything about your uploaded documents...")

if prompt:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Prepare the AI response
    with st.chat_message("assistant"):
        try:
            sources_used = []
            
            # Use RAG if enabled and documents are available
            if use_rag:
                vector_db_info = st.session_state.vector_db.get_collection_info()
                
                if vector_db_info.get("document_count", 0) > 0:
                    # Get relevant context
                    with st.spinner("Searching documents..."):
                        search_results = st.session_state.vector_db.similarity_search(
                            prompt, 
                            n_results=num_context_docs
                        )
                    
                    if search_results:
                        # Prepare context and sources
                        context = st.session_state.vector_db.get_relevant_context(
                            prompt, 
                            n_results=num_context_docs,
                            max_context_length=max_context_length
                        )
                        
                        # Prepare sources information
                        for result in search_results:
                            sources_used.append({
                                "source": result["metadata"].get("source", "Unknown"),
                                "chunk_index": result["metadata"].get("chunk_index", "N/A"),
                                "distance": result["distance"],
                                "preview": result["document"]
                            })
                        
                        # Create RAG prompt
                        final_prompt = create_rag_prompt(prompt, context)
                        
                        # Show that we're using documents
                        st.info(f"🔍 Found {len(search_results)} relevant document(s)")
                    else:
                        final_prompt = create_simple_prompt(prompt)
                        st.info("📝 No relevant documents found, providing general response")
                else:
                    final_prompt = create_simple_prompt(prompt)
                    st.info("📄 No documents uploaded, providing general response")
            else:
                final_prompt = create_simple_prompt(prompt)
            
            # Get response from AI
            with st.spinner("Thinking..."):
                response = st.session_state.chat.send_message(final_prompt)
                
                if hasattr(response, "text"):
                    answer = response.text
                else:
                    answer = str(response)
            
            # Display the response
            st.markdown(answer)
            
            # Add to message history with sources
            message_data = {"role": "assistant", "content": answer}
            if sources_used:
                message_data["sources"] = sources_used
            
            st.session_state.messages.append(message_data)
            
        except Exception as e:
            error_message = f"An error occurred: {e}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})

# --- 8. Footer Information ---

with st.expander("ℹ️ How to Use This RAG Chatbot"):
    st.markdown("""
    ### 📋 Instructions
    
    1. **Setup**: Enter your Google AI API key in the sidebar
    2. **Upload Documents**: Use the file uploader to add PDF, DOCX, or TXT files
    3. **Process**: Click "Process Documents" to extract and index the content
    4. **Chat**: Ask questions about your uploaded documents
    5. **Review Sources**: Expand the "Sources Used" section to see which documents were referenced
    
    ### 🔧 Features
    
    - **Multi-format Support**: PDF, DOCX, and TXT files
    - **Smart Chunking**: Documents are split into optimal chunks for processing
    - **Semantic Search**: Find relevant information using AI-powered embeddings
    - **Source Attribution**: See exactly which documents and sections were used
    - **Adjustable Settings**: Control how many documents and how much context to use
    
    ### 💡 Tips
    
    - Upload related documents for better context
    - Ask specific questions for more accurate responses
    - Use the document management tools to clear or reset as needed
    - Disable RAG if you want general AI responses without document context
    """)

st.markdown("---")
st.markdown("*Built with Streamlit, Google Gemini, ChromaDB, and LangChain*")