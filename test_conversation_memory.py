"""
Test the enhanced conversation memory system
"""

from conversation_manager import ConversationManager, Conversation
from datetime import datetime
import json

def test_conversation_memory():
    """Test basic conversation memory functionality"""
    print("🧪 Testing Enhanced Conversation Memory System...")
    
    # Initialize manager with test database
    manager = ConversationManager("test_conversations.db")
    
    # Test conversation creation
    test_messages = [
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence..."}
    ]
    
    conversation = Conversation(
        id="test_conv_001",
        title="Machine Learning Discussion",
        messages=test_messages,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        category="educational",
        tags=["ML", "AI", "Education"],
        summary="Discussion about machine learning basics",
        message_count=2,
        total_tokens=150
    )
    
    # Test save
    print("📝 Testing save conversation...")
    success = manager.save_conversation(conversation)
    print(f"✅ Save successful: {success}")
    
    # Test load
    print("📂 Testing load conversation...")
    loaded = manager.load_conversation("test_conv_001")
    if loaded:
        print(f"✅ Loaded conversation: {loaded.title}")
        print(f"   Messages: {loaded.message_count}")
        print(f"   Category: {loaded.category}")
        print(f"   Tags: {loaded.tags}")
    else:
        print("❌ Failed to load conversation")
    
    # Test list conversations
    print("📋 Testing list conversations...")
    conversations = manager.list_conversations()
    print(f"✅ Found {len(conversations)} conversations")
    
    # Test categories
    print("🏷️ Testing categories...")
    categories = manager.get_categories()
    print(f"✅ Categories: {categories}")
    
    # Test stats
    print("📊 Testing conversation stats...")
    stats = manager.get_conversation_stats()
    print(f"✅ Stats: {stats}")
    
    # Test export
    print("📤 Testing export...")
    try:
        export_data = manager.export_conversations("json")
        export_obj = json.loads(export_data)
        print(f"✅ Export successful: {export_obj['total_conversations']} conversations")
    except Exception as e:
        print(f"❌ Export failed: {e}")
    
    # Test search
    print("🔍 Testing search...")
    search_results = manager.list_conversations(search_term="machine")
    print(f"✅ Search results: {len(search_results)} matches")
    
    # Test delete
    print("🗑️ Testing delete...")
    success = manager.delete_conversation("test_conv_001")
    print(f"✅ Delete successful: {success}")
    
    # Cleanup
    import os
    try:
        os.remove("test_conversations.db")
        print("🧹 Test database cleaned up")
    except:
        pass
    
    print("🎉 All tests completed!")

if __name__ == "__main__":
    test_conversation_memory()