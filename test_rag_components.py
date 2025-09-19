"""
Test file to verify RAG chatbot components work correctly
"""

import tempfile
import os
from io import BytesIO

def test_document_processor():
    """Test the document processor functionality"""
    print("Testing Document Processor...")
    
    try:
        from document_processor import DocumentProcessor
        
        processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
        
        # Test text chunking
        test_text = "This is a test document. " * 100  # Create a longer text
        chunks = processor.chunk_text(test_text, {"source": "test.txt"})
        
        print(f"✅ Created {len(chunks)} chunks from test text")
        print(f"✅ First chunk length: {len(chunks[0].page_content)}")
        print(f"✅ Chunk metadata: {chunks[0].metadata}")
        
        return True
    except Exception as e:
        print(f"❌ Document Processor test failed: {e}")
        return False

def test_vector_database():
    """Test the vector database functionality"""
    print("\nTesting Vector Database...")
    
    try:
        from vector_database import VectorDatabase
        from langchain.docstore.document import Document as LangChainDocument
        
        # Create test documents
        test_docs = [
            LangChainDocument(
                page_content="Artificial intelligence is transforming technology.",
                metadata={"source": "ai_doc.txt", "chunk_index": 0}
            ),
            LangChainDocument(
                page_content="Machine learning algorithms learn from data patterns.",
                metadata={"source": "ml_doc.txt", "chunk_index": 0}
            )
        ]
        
        # Initialize vector database with temporary directory
        temp_dir = tempfile.mkdtemp()
        vector_db = VectorDatabase(
            collection_name="test_collection",
            persist_directory=temp_dir
        )
        
        # Test adding documents
        success = vector_db.add_documents(test_docs)
        print(f"✅ Documents added successfully: {success}")
        
        # Test similarity search
        results = vector_db.similarity_search("artificial intelligence", n_results=2)
        print(f"✅ Found {len(results)} search results")
        
        if results:
            print(f"✅ Best match: {results[0]['document'][:50]}...")
            print(f"✅ Distance: {results[0]['distance']:.4f}")
        
        # Test context retrieval
        context = vector_db.get_relevant_context("machine learning", n_results=1)
        print(f"✅ Context retrieved: {len(context)} characters")
        
        # Get collection info
        info = vector_db.get_collection_info()
        print(f"✅ Collection has {info.get('document_count', 0)} documents")
        
        # Cleanup
        vector_db.clear_collection()
        
        return True
    except Exception as e:
        print(f"❌ Vector Database test failed: {e}")
        return False

def test_integration():
    """Test integration between components"""
    print("\nTesting Integration...")
    
    try:
        from document_processor import DocumentProcessor
        from vector_database import VectorDatabase
        
        # Create test text file content
        test_content = """
        This is a comprehensive test document for the RAG chatbot system.
        
        Chapter 1: Introduction
        The RAG (Retrieval-Augmented Generation) system combines information retrieval
        with natural language generation to provide more accurate and contextual responses.
        
        Chapter 2: Components
        The system consists of several key components:
        1. Document processor for handling file uploads
        2. Vector database for storing and retrieving embeddings
        3. Chat interface for user interaction
        
        Chapter 3: Benefits
        RAG systems provide several advantages including improved accuracy,
        source attribution, and the ability to work with private documents.
        """
        
        # Process the document
        processor = DocumentProcessor(chunk_size=200, chunk_overlap=50)
        chunks = processor.chunk_text(test_content, {"source": "test_integration.txt"})
        
        print(f"✅ Created {len(chunks)} chunks from integration test")
        
        # Store in vector database
        temp_dir = tempfile.mkdtemp()
        vector_db = VectorDatabase(
            collection_name="integration_test",
            persist_directory=temp_dir
        )
        
        success = vector_db.add_documents(chunks)
        print(f"✅ Integration test documents stored: {success}")
        
        # Test queries
        test_queries = [
            "What is RAG?",
            "What are the components?",
            "What are the benefits?"
        ]
        
        for query in test_queries:
            results = vector_db.similarity_search(query, n_results=2)
            if results:
                print(f"✅ Query '{query}': Found {len(results)} results")
            else:
                print(f"⚠️ Query '{query}': No results found")
        
        # Cleanup
        vector_db.clear_collection()
        
        return True
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Running RAG Chatbot Component Tests\n")
    
    tests = [
        test_document_processor,
        test_vector_database,
        test_integration
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! RAG chatbot is ready to use.")
    else:
        print("⚠️ Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()