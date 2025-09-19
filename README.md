# 🤖 RAG Chatbot Streamlit Demo

A powerful **Retrieval-Augmented Generation (RAG)** ch```
chatbot-streamlit-RAG/
├── rag_chatbot_app.py # 🎯 Main RAG chatbot application
├── document_processor.py # 📄 Document processing and text extraction
├── vector_database.py # 🔍 Vector storage and similarity search
├── test_rag_components.py # 🧪 Testing utilities
├── requirements.txt # 📦 Python dependencies
├── Dockerfile # 🐳 Container configuration
├── .env.example # ⚙️ Environment variables template
└── README.md # 📖 This file

````

## 🛠️ Application

The main application is a **RAG-enabled chatbot** (`rag_chatbot_app.py`) that allows you to:
- 📁 Upload documents (PDF, TXT, DOCX)
- 🤖 Chat with AI about your documents
- 🔍 Get responses with source attributionwith Streamlit that allows you to upload documents and have intelligent conversations about their content using Google Gemini AI.

## ✨ Features

- 📄 **Multi-format Document Support**: Upload PDF, DOCX, and TXT files
- 🧠 **Smart Document Processing**: Automatic text extraction and intelligent chunking
- 🔍 **Vector Search**: Semantic similarity search using ChromaDB and sentence transformers
- 📊 **Source Attribution**: See exactly which documents and sections were used in responses
- ⚙️ **Configurable Settings**: Adjust context length, number of retrieved documents, and more
- 🐳 **Docker Support**: Easy deployment with Docker containers
- 🔄 **Real-time Processing**: Live document indexing and conversation management

## 🚀 Quick Start

### Option 1: Local Development

#### Prerequisites

- Python 3.13.7 (recommended) or Python 3.9+
- Conda or Miniconda for environment management

#### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd chatbot-streamlit-RAG
````

2. **Create and activate conda environment**

   ```bash
   conda create -n chatbot-py313 python=3.13.7
   conda activate chatbot-py313
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the RAG chatbot**

   ```bash
   streamlit run rag_chatbot_app.py
   ```

   The app will be available at `http://localhost:8501`

### Option 2: Docker Deployment

#### Quick Docker Setup

1. **Build the Docker image**

   ```bash
   docker build -t rag-chatbot .
   ```

2. **Run the container**

   ```bash
   docker run -p 8501:8501 rag-chatbot
   ```

   Access the app at `http://localhost:8501`

#### Advanced Docker Usage

```bash
# Run with volume mounting for persistent storage
docker run -p 8501:8501 -v $(pwd)/data:/app/vector_db rag-chatbot

# Run in detached mode
docker run -d -p 8501:8501 --name rag-chatbot-app rag-chatbot

# View logs
docker logs rag-chatbot-app
```

## 📁 Project Structure

```
chatbot-streamlit-RAG/
├── rag_chatbot_app.py           # 🎯 Main RAG chatbot application
├── document_processor.py        # 📄 Document processing and text extraction
├── vector_database.py          # 🔍 Vector storage and similarity search
├── streamlit_chat_app.py        # 💬 Simple chatbot (without RAG)
├── database_tools.py           # �️ Database utilities
├── test_rag_components.py       # 🧪 Testing utilities
├── requirements.txt            # 📦 Python dependencies
├── Dockerfile                  # 🐳 Container configuration
├── .env.example                # ⚙️ Environment variables template
├── sales_data.db              # 📊 Sample database
└── README.md                  # 📖 This file
```

## 🛠️ Available Applications

| Application             | Description                                  | Use Case                                  |
| ----------------------- | -------------------------------------------- | ----------------------------------------- |
| `rag_chatbot_app.py`    | **RAG-enabled chatbot** with document upload | 🎯 **Primary app** - Upload docs and chat |
| `streamlit_chat_app.py` | Simple chatbot with Google Gemini            | 💬 Basic AI conversations                 |

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Google Gemini API Configuration
GOOGLE_AI_API_KEY=your_gemini_api_key_here

# Optional: Customize default settings
DEFAULT_MODEL=gemini-1.5-flash
DEFAULT_MAX_TOKENS=1000
DEFAULT_TEMPERATURE=0.7
```

### RAG Settings

The RAG chatbot offers configurable parameters in the sidebar:

- **Enable RAG**: Toggle document-based responses
- **Context Documents**: Number of relevant documents to retrieve (1-10)
- **Max Context Length**: Maximum characters from retrieved documents (500-4000)
- **Clear Documents**: Reset the vector database
- **Clear Chat**: Reset conversation history

## 📚 How RAG Works

1. **📤 Document Upload**: Upload PDF, DOCX, or TXT files
2. **⚙️ Processing**: Documents are extracted and split into semantic chunks
3. **🔢 Embedding**: Text chunks are converted to vector embeddings using sentence transformers
4. **💾 Storage**: Embeddings are stored in ChromaDB for fast retrieval
5. **❓ Query**: User asks a question
6. **🔍 Retrieval**: Most relevant document chunks are found using vector similarity
7. **🤖 Generation**: Google Gemini generates response using retrieved context
8. **📋 Attribution**: Sources are displayed with the response

## 🎯 Use Cases

- **📖 Research**: Upload academic papers and ask specific questions
- **📋 Documentation**: Query technical manuals and guides
- **📊 Analysis**: Upload reports and get insights
- **🎓 Learning**: Upload textbooks and get explanations
- **💼 Business**: Process contracts, policies, and procedures
- **📝 Content Review**: Analyze large documents quickly

## 🔧 Technical Dependencies

### Core Libraries

- **Streamlit**: Web app framework
- **Google Generative AI**: LLM integration
- **ChromaDB**: Vector database
- **Sentence Transformers**: Text embeddings
- **LangChain**: Document processing utilities

### Document Processing

- **PyPDF2**: PDF text extraction
- **python-docx**: Word document processing
- **tiktoken**: Token counting and management

## 🐛 Troubleshooting

### Common Issues

1. **API Key Error**: Ensure `GOOGLE_AI_API_KEY` is set in your environment
2. **Memory Issues**: Large documents may require more RAM
3. **Port Conflicts**: Change port with `--server.port 8502`
4. **Docker Issues**: Ensure Docker is running and ports are available

### Performance Tips

- **Chunk Size**: Smaller chunks improve retrieval accuracy
- **Model Selection**: Use `gemini-1.5-flash` for faster responses
- **Document Prep**: Clean documents before upload for better results

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

🚀 **Ready to start?** Choose your preferred method above and begin chatting with your documents!
