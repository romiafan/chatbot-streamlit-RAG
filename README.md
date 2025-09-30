# 🧠 AI Document Assistant

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
- 💬 **Modern Chat Interface**: Gemini-style message bubbles with smooth interactions
- 🧩 **Duplicate Chunk Deduping**: SHA1 hashing skips re-embedding identical text (speed + cost)
- 🎯 **Context Governance**: Adjustable number of retrieved chunks and max context length
- 🧮 **Lightweight Token Estimation**: Heuristic chars/4 running total (no extra deps)
- 🚦 **Message Limit Safeguard**: Hard cap (default 50) with 80% early warning
- 🧲 **Persistent Duplicate Detection**: Chunk hashes stored in metadata to avoid re-embedding across sessions
- 🔧 **Adjustable Chunking**: Tune chunk size & overlap from sidebar (expander)
- 🗂️ **Batch Source Export**: Download all RAG source chunks as JSON
- 📥 **Per-Response Source Export**: Export sources for any single answer
- 📋 **Copy Answer Button**: One-click copy for each assistant reply
- 🔀 **Per-Message Model Choice**: Select model for each prompt independently
- 🔢 **Optional Precise Token Counting**: Uses `tiktoken` automatically if installed; falls back gracefully
- 🤖 **Dynamic Gemini Model Discovery**: Enumerates available free-tier flash models at runtime
- 🧭 **Environment Default Priority**: `GEMINI_DEFAULT_MODEL` (e.g. `gemini-2.5-flash`) is surfaced first when present
- 📈 **Version-Aware Ordering**: Prefers newer flash generations (2.5 before 1.5)
- 🔁 **One-Click Model Refresh**: Sidebar Refresh button re-runs discovery without restarting
- 🪄 **Resilient Fallback Resolution**: Automatic fallback attempts (raw, prefixed, alt flash variants) to avoid 404 errors
- 🏷️ **Requested→Actual Badge**: If fallback triggers, assistant messages show `requested→actual` model tag

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

Create a `.env` file (or set in your deployment environment) for configuration:

```env
# Required
GOOGLE_AI_API_KEY=your_gemini_api_key_here

# Optional runtime customization
GEMINI_DEFAULT_MODEL=gemini-1.5-flash-8b     # Preselect model in sidebar (free-tier friendly)
CHAT_MESSAGE_LIMIT=50                       # Total (user+assistant) messages before reset required
EMBED_MODE=1                                # Compact UI for iframe embedding / portal usage
CHROMA_PERSIST_DIRECTORY=./vector_db        # Persist embeddings between restarts (if desired)
```

Notes:

- If `GEMINI_DEFAULT_MODEL` not set, a sensible free model is chosen.
- `EMBED_MODE` can also be activated via URL query param: `?embed=1`.
- Message limit shows a toast at 80% usage; once reached, only Reset Chat is allowed.

### Model Discovery & Fallback Logic

1. Discovery lists Gemini models and filters for flash variants (excluding `pro`, `vision`, `exp`).
2. Environment default (if defined) is prepended.
3. Models are version-sorted so `gemini-2.5-*` precede `gemini-1.5-*`.
4. If nothing is discovered, a minimal fallback list begins with the environment default (if any) then known flash variants.
5. Chat creation tries (in order): requested raw, requested prefixed, env default raw/prefixed, 2.5 flash variants, 1.5 flash variants.
6. If the resolved model differs, UI displays a `requested→actual` badge with a tooltip indicating fallback occurred.

Use the sidebar Refresh button anytime to clear the cached list and re-run discovery.

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
- **(Optional) tiktoken**: For precise token counting if you replace the heuristic (not required by default)

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

## 🎉 What's New

### Recent Enhancements (Post v2.0)

- ✅ Free-tier Gemini model selector in sidebar (env override with `GEMINI_DEFAULT_MODEL`)
- ✅ Embed / Compact Mode via `?embed=1` or `EMBED_MODE=1` (suppresses large headers & condenses stats)
- ✅ Message & token usage governance (`CHAT_MESSAGE_LIMIT`, heuristic token estimate chars/4)
- ✅ Duplicate chunk hashing prevents re-embedding identical content within a session
- ✅ Expanded sidebar Quick Stats: docs, messages used, token estimate, current model
- ✅ Clearer success messaging: highlights number of new vs skipped duplicate chunks
- ✅ Persistent dedupe (hash metadata) prevents re-embedding after restarts when using same collection
- ✅ Adjustable chunk size & overlap controls
- ✅ Per-message model selection (multi-model conversations)
- ✅ Copy-to-clipboard button on assistant messages
- ✅ Per-response & batch JSON export of sources
- ✅ Optional precise token counting via `tiktoken` (fallback heuristic if unavailable)
- ✅ Dynamic model discovery + version ordering (2.5 prioritized when available)
- ✅ Env default forced into fallback chain for resiliency
- ✅ Model Refresh button for on-demand re-discovery
- ✅ Requested→Actual fallback indicator in assistant responses

### v2.0 UI/UX Overhaul (Previous Major Release)

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

### Performance & Usage Tips

- **Document Preparation**: Clean documents before upload for better results
- **Chunk Size**: Smaller chunks often improve retrieval accuracy (current splitter: 1000 chars w/ 200 overlap)
- **Model Selection**: Pick a flash model for speed; override default with `GEMINI_DEFAULT_MODEL`
- **Context Tuning**: Adjust context length to balance recall vs. latency
- **Avoid Dupes**: Re-uploading identical files won't cost extra embedding time in the same session
- **Approaching Limit**: If you see an 80% usage toast, consider resetting sooner to keep context focused

## 📄 License

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
