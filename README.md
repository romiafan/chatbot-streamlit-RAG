# � AI Document Assistant

A modern, intelligent **Retrieval-Augmented Generation (RAG)** chatbot with a beautiful UI/UX inspired by Perplexity and Gemini. Upload your documents and have intelligent conversations powered by Google Gemini AI with a sleek, professional interface.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red?style=flat-square&logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?style=flat-square&logo=docker)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

## ✨ Features

### 🎨 **Modern UI/UX Design**

- **Professional Interface**: Clean, modern design inspired by Perplexity and Gemini
- **Responsive Layout**: Optimized for desktop and mobile devices
- **Dark Mode Support**: Automatic theme detection with beautiful color schemes
- **Smooth Animations**: Subtle motion design and micro-interactions
- **Interactive Elements**: Hover effects, loading states, and visual feedback
- **Modern Typography**: Inter font family for excellent readability
- **Glass Effects**: Modern glassmorphism design elements
- **Gradient Themes**: Beautiful color gradients throughout the interface

### 🚀 **Advanced RAG Capabilities**

- 📄 **Multi-format Document Support**: Upload PDF, DOCX, and TXT files
- 🧠 **Smart Document Processing**: Automatic text extraction and intelligent chunking
- 🔍 **Vector Search**: Semantic similarity search using ChromaDB and sentence transformers
- 📊 **Enhanced Source Attribution**: Beautiful source cards with relevance scores
- ⚙️ **Intelligent Configuration**: User-friendly settings with smart defaults
- 🔄 **Real-time Processing**: Live document indexing with progress indicators
- � **Modern Chat Interface**: Gemini-style message bubbles with smooth interactions

### 🌟 **User Experience**

- **Welcome Screen**: Helpful onboarding with usage suggestions
- **Progress Tracking**: Visual progress bars for file processing
- **File Preview**: Modern file cards showing size, type, and status
- **Quick Stats**: Real-time document count and status indicators
- **Error Handling**: Beautiful error messages with helpful guidance
- **Help System**: Comprehensive, expandable help documentation

## 🚀 Quick Start

### Option 1: Local Development

#### Prerequisites

- Python 3.9+ (Python 3.13+ recommended)
- Google AI API Key ([Get one here](https://makersuite.google.com/app/apikey))

#### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd chatbot-streamlit-RAG
   ```

2. **Create virtual environment**

   ```bash
   # Using conda (recommended)
   conda create -n rag-chatbot python=3.13
   conda activate rag-chatbot

   # Or using venv
   python -m venv rag-chatbot
   source rag-chatbot/bin/activate  # On Windows: rag-chatbot\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**

   ```bash
   streamlit run rag_chatbot_app.py
   ```

5. **Open your browser**
   - Navigate to `http://localhost:8501`
   - Enter your Google AI API key in the sidebar
   - Upload documents and start chatting!

### Option 2: Docker Deployment

#### Quick Docker Setup

1. **Build and run**

   ```bash
   docker build -t ai-document-assistant .
   docker run -p 8501:8501 ai-document-assistant
   ```

2. **Access the application**
   - Open `http://localhost:8501` in your browser

## 📁 Project Structure

```
chatbot-streamlit-RAG/
├── rag_chatbot_app.py           # 🎯 Main application with modern UI
├── document_processor.py        # 📄 Document processing and text extraction
├── vector_database.py          # 🔍 Vector storage and similarity search
├── test_rag_components.py       # 🧪 Testing utilities
├── requirements.txt            # 📦 Python dependencies
├── Dockerfile                  # 🐳 Container configuration
├── .gitignore                  # 📝 Git ignore patterns
└── README.md                   # 📖 This documentation
```

## 🎯 How It Works

### The RAG Pipeline

1. **📤 Document Upload**: Modern drag-and-drop interface for file uploads
2. **⚙️ Processing**: Documents are extracted and split into semantic chunks with progress tracking
3. **🔢 Embedding**: Text chunks are converted to vector embeddings using sentence transformers
4. **💾 Storage**: Embeddings are stored in ChromaDB for lightning-fast retrieval
5. **❓ Query**: Users ask questions through the modern chat interface
6. **🔍 Retrieval**: Most relevant document chunks are found using vector similarity search
7. **🤖 Generation**: Google Gemini generates contextual responses
8. **📋 Attribution**: Sources are beautifully displayed with relevance scores

### Modern UI Features

- **Gradient Headers**: Eye-catching headers with smooth gradients
- **Card-based Layout**: Organized content in visually appealing cards
- **Message Bubbles**: Distinct styling for user and AI messages
- **Source Cards**: Beautiful source attribution with relevance indicators
- **Loading States**: Smooth animations during processing
- **Responsive Design**: Perfect on any device size

## ⚙️ Configuration

### Sidebar Settings

| Setting            | Description                                 | Options                    |
| ------------------ | ------------------------------------------- | -------------------------- |
| **Enable RAG**     | Use uploaded documents for responses        | ✅ Enabled / ❌ Disabled   |
| **Context Docs**   | Number of relevant documents to retrieve    | 1-10 documents             |
| **Context Length** | Maximum characters from retrieved documents | Short (1K) - Extended (4K) |
| **Clear Docs**     | Remove all uploaded documents               | 🗑️ One-click clearing      |
| **Reset Chat**     | Clear conversation history                  | 🔄 Fresh start             |

### Environment Variables

Create a `.env` file for advanced configuration:

```env
# Required
GOOGLE_AI_API_KEY=your_gemini_api_key_here

# Optional customization
DEFAULT_MODEL=gemini-2.5-flash
DEFAULT_MAX_TOKENS=2000
DEFAULT_TEMPERATURE=0.7
CHROMA_PERSIST_DIRECTORY=./vector_db
```

## 🎨 UI/UX Highlights

### Design System

- **Color Palette**: Professional blues with excellent contrast ratios
- **Typography**: Inter font family for modern, clean text
- **Spacing**: Consistent spacing using design tokens
- **Shadows**: Subtle shadows for depth and hierarchy
- **Animations**: Smooth transitions and micro-interactions

### Responsive Features

- **Mobile-first**: Optimized for mobile devices
- **Flexible Grid**: Adaptive layouts for different screen sizes
- **Touch-friendly**: Large touch targets for mobile users
- **Fast Loading**: Optimized performance across devices

### Accessibility

- **High Contrast**: WCAG compliant color combinations
- **Focus States**: Clear focus indicators for keyboard navigation
- **Screen Reader**: Semantic HTML for assistive technologies
- **Alt Text**: Descriptive text for all visual elements

## �️ Technical Stack

### Core Technologies

- **[Streamlit](https://streamlit.io/)**: Modern web app framework
- **[Google Gemini](https://ai.google.dev/)**: Advanced language model
- **[ChromaDB](https://www.trychroma.com/)**: Vector database for embeddings
- **[Sentence Transformers](https://www.sbert.net/)**: Text embedding models
- **[LangChain](https://langchain.com/)**: Document processing utilities

### Document Processing

- **PyPDF2**: PDF text extraction
- **python-docx**: Word document processing
- **tiktoken**: Token counting and management

### UI/UX Technologies

- **Custom CSS**: Modern styling with CSS variables
- **Google Fonts**: Inter typography system
- **CSS Grid & Flexbox**: Responsive layouts
- **CSS Animations**: Smooth transitions and effects

## 📚 Use Cases

- **📖 Research**: Upload academic papers and ask specific questions
- **📋 Documentation**: Query technical manuals and guides
- **📊 Analysis**: Upload reports and get insights with source attribution
- **🎓 Learning**: Upload textbooks and get explanations
- **💼 Business**: Process contracts, policies, and procedures
- **📝 Content Review**: Analyze large documents quickly and efficiently

## 🎉 What's New in v2.0

### Major UI/UX Overhaul

- ✅ Complete redesign inspired by Perplexity and Gemini
- ✅ Modern CSS with design tokens and smooth animations
- ✅ Enhanced chat interface with message bubbles
- ✅ Beautiful file upload with drag-and-drop styling
- ✅ Responsive design for all devices
- ✅ Improved source attribution with relevance scores
- ✅ Professional color scheme and typography
- ✅ Interactive elements and micro-interactions

### Enhanced Features

- ✅ Progress indicators for file processing
- ✅ Smart sidebar with organized sections
- ✅ Welcome screen with helpful suggestions
- ✅ Modern help documentation
- ✅ Error handling with beautiful messages
- ✅ Quick stats and status indicators

## � Troubleshooting

### Common Issues

| Issue              | Solution                                                     |
| ------------------ | ------------------------------------------------------------ |
| **API Key Error**  | Ensure your Google AI API key is valid and entered correctly |
| **Memory Issues**  | Try smaller documents or reduce context length               |
| **Port Conflicts** | Use `streamlit run rag_chatbot_app.py --server.port 8502`    |
| **Docker Issues**  | Ensure Docker is running and ports 8501 is available         |
| **Upload Errors**  | Check file format (PDF, DOCX, TXT) and size limits           |

### Performance Tips

- **Document Preparation**: Clean documents before upload for better results
- **Chunk Size**: Smaller chunks often improve retrieval accuracy
- **Model Selection**: Use `gemini-2.5-flash` for faster responses
- **Context Tuning**: Adjust context length based on your needs

## �📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
   - Follow the existing code style
   - Add tests if applicable
   - Update documentation
4. **Commit your changes**
   ```bash
   git commit -m 'Add: amazing new feature'
   ```
5. **Push and create PR**
   ```bash
   git push origin feature/amazing-feature
   ```

### Development Guidelines

- **Code Style**: Follow PEP 8 for Python code
- **UI/UX**: Maintain consistency with existing design system
- **Documentation**: Update README for new features
- **Testing**: Add tests for new functionality

---

🚀 **Ready to start?** Upload your documents and experience the future of AI-powered document interaction!

_Built with ❤️ using Streamlit, Google Gemini, ChromaDB, and modern web technologies_
