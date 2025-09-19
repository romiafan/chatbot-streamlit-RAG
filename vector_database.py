"""
Vector Database Module for RAG Chatbot
Handles document embeddings, storage, and retrieval using ChromaDB
"""

import streamlit as st
import os
import tempfile
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain.docstore.document import Document as LangChainDocument
import uuid
import json


class VectorDatabase:
    """Manages document embeddings and retrieval using ChromaDB"""
    
    def __init__(self, 
                 collection_name: str = "document_store", 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 persist_directory: str = None):
        """
        Initialize the vector database
        
        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: Name of the sentence transformer model
            persist_directory: Directory to persist the database
        """
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        
        # Set up persistent directory
        if persist_directory is None:
            self.persist_directory = os.path.join(tempfile.gettempdir(), "chroma_db")
        else:
            self.persist_directory = persist_directory
        
        # Initialize embedding model
        self._initialize_embedding_model()
        
        # Initialize ChromaDB client
        self._initialize_chroma_client()
    
    def _initialize_embedding_model(self):
        """Initialize the sentence transformer embedding model"""
        try:
            with st.spinner("Loading embedding model..."):
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
            st.success("✅ Embedding model loaded successfully")
        except Exception as e:
            st.error(f"Error loading embedding model: {str(e)}")
            raise e
    
    def _initialize_chroma_client(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Create persistent client
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name
                )
                st.info(f"📚 Loaded existing collection: {self.collection_name}")
            except Exception:
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "RAG document store"}
                )
                st.success(f"🆕 Created new collection: {self.collection_name}")
            
        except Exception as e:
            st.error(f"Error initializing ChromaDB: {str(e)}")
            raise e
    
    def embed_documents(self, documents: List[LangChainDocument]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents
        
        Args:
            documents: List of LangChain Document objects
        
        Returns:
            List of embedding vectors
        """
        if not documents:
            return []
        
        texts = [doc.page_content for doc in documents]
        
        with st.spinner(f"Generating embeddings for {len(documents)} documents..."):
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
        
        return embeddings.tolist()
    
    def add_documents(self, documents: List[LangChainDocument]) -> bool:
        """
        Add documents to the vector database
        
        Args:
            documents: List of LangChain Document objects
        
        Returns:
            True if successful, False otherwise
        """
        if not documents:
            st.warning("No documents to add")
            return False
        
        try:
            # Generate embeddings
            embeddings = self.embed_documents(documents)
            
            # Prepare data for ChromaDB
            ids = [str(uuid.uuid4()) for _ in documents]
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            
            st.success(f"✅ Added {len(documents)} documents to vector database")
            return True
            
        except Exception as e:
            st.error(f"Error adding documents to vector database: {str(e)}")
            return False
    
    def similarity_search(self, 
                         query: str, 
                         n_results: int = 5, 
                         where: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Perform similarity search on the vector database
        
        Args:
            query: Search query text
            n_results: Number of results to return
            where: Optional metadata filter
        
        Returns:
            List of search results with documents and metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)
            
            # Perform search
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    formatted_results.append({
                        "document": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i] if results["metadatas"][0] else {},
                        "distance": results["distances"][0][i] if results["distances"][0] else 0.0
                    })
            
            return formatted_results
            
        except Exception as e:
            st.error(f"Error performing similarity search: {str(e)}")
            return []
    
    def get_relevant_context(self, 
                           query: str, 
                           n_results: int = 3, 
                           max_context_length: int = 2000) -> str:
        """
        Get relevant context for a query, formatted for RAG
        
        Args:
            query: Search query
            n_results: Number of documents to retrieve
            max_context_length: Maximum length of context to return
        
        Returns:
            Formatted context string
        """
        results = self.similarity_search(query, n_results)
        
        if not results:
            return "No relevant documents found."
        
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(results):
            document = result["document"]
            metadata = result["metadata"]
            distance = result["distance"]
            
            # Create a formatted context entry
            source = metadata.get("source", "Unknown")
            chunk_info = f"[Document: {source}, Chunk {metadata.get('chunk_index', 'N/A')}]"
            
            entry = f"{chunk_info}\n{document}\n"
            
            # Check if adding this entry would exceed max length
            if current_length + len(entry) > max_context_length and context_parts:
                break
            
            context_parts.append(entry)
            current_length += len(entry)
        
        context = "\n---\n".join(context_parts)
        
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
        
        return context
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection"""
        try:
            count = self.collection.count()
            return {
                "name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory,
                "embedding_model": self.embedding_model_name
            }
        except Exception as e:
            st.error(f"Error getting collection info: {str(e)}")
            return {}
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection"""
        try:
            # Get all IDs
            results = self.collection.get()
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                st.success("🗑️ Collection cleared successfully")
            else:
                st.info("Collection is already empty")
            return True
        except Exception as e:
            st.error(f"Error clearing collection: {str(e)}")
            return False
    
    def delete_collection(self) -> bool:
        """Delete the entire collection"""
        try:
            self.client.delete_collection(name=self.collection_name)
            st.success(f"🗑️ Collection '{self.collection_name}' deleted successfully")
            return True
        except Exception as e:
            st.error(f"Error deleting collection: {str(e)}")
            return False


@st.cache_resource
def get_vector_database(collection_name: str = "document_store") -> VectorDatabase:
    """
    Cached function to get or create a vector database instance
    
    Args:
        collection_name: Name of the collection
    
    Returns:
        VectorDatabase instance
    """
    return VectorDatabase(collection_name=collection_name)


def display_vector_db_info(vector_db: VectorDatabase):
    """Display information about the vector database"""
    info = vector_db.get_collection_info()
    
    if info:
        st.write("### 📊 Vector Database Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Collection Name", info.get("name", "N/A"))
            st.metric("Document Count", info.get("document_count", 0))
        
        with col2:
            st.write(f"**Embedding Model:** {info.get('embedding_model', 'N/A')}")
            st.write(f"**Persist Directory:** {info.get('persist_directory', 'N/A')}")


# Testing function
def test_vector_database():
    """Test function for the VectorDatabase class"""
    st.write("### Vector Database Test")
    
    # Test documents
    test_docs = [
        LangChainDocument(
            page_content="This is a test document about artificial intelligence.",
            metadata={"source": "test1.txt", "chunk_index": 0}
        ),
        LangChainDocument(
            page_content="Machine learning is a subset of AI that focuses on algorithms.",
            metadata={"source": "test2.txt", "chunk_index": 0}
        )
    ]
    
    # Initialize vector database
    vector_db = get_vector_database("test_collection")
    
    # Display info
    display_vector_db_info(vector_db)
    
    # Test search
    if st.button("Test Search"):
        results = vector_db.similarity_search("artificial intelligence", n_results=2)
        st.write("Search Results:", results)


if __name__ == "__main__":
    test_vector_database()