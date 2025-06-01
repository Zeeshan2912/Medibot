# Medibot - Medical AI Assistant

A Streamlit-based medical chatbot that uses Retrieval Augmented Generation (RAG) with a local Large Language Model to answer medical queries based on medical literature.

## Features

- **Local LLM Integration**: Uses EleutherAI/gpt-neo-1.3B model for text generation
- **RAG Pipeline**: Combines document retrieval with language generation for accurate answers
- **FAISS Vector Database**: Efficient similarity search for relevant medical content
- **Streamlit Interface**: User-friendly web interface for medical queries
- **Medical Document Processing**: Processes PDF medical literature for knowledge base

## Project Structure

```
Medibot/
├── medibot.py                      # Main Streamlit application
├── create_memory_for_llm.py        # Script to create FAISS vector store
├── connect_memory_with_llm.py      # RAG pipeline implementation
├── test_hf_direct_api.py          # Hugging Face API testing utility
├── data/                          # Medical documents directory
├── vectorstore/db_faiss/          # FAISS database files
├── .streamlit/config.toml         # Streamlit configuration
└── requirements.txt               # Python dependencies
```

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Zeeshan2912/Medibot.git
cd Medibot
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the root directory:
```
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
```

### 5. Prepare Medical Documents
- Place your medical PDF documents in the `data/` directory
- Run the vector store creation script:
```bash
python create_memory_for_llm.py
```

### 6. Run the Application
```bash
streamlit run medibot.py
```

## Usage

1. Open your browser and navigate to the Streamlit interface (typically `http://localhost:8501`)
2. Enter your medical query in the text input field
3. Click "Get Answer" to receive a contextual response based on medical literature
4. The system will display both the AI-generated answer and relevant source excerpts

## Technical Details

### RAG Pipeline
- **Document Processing**: PDFs are split into chunks and embedded using HuggingFace embeddings
- **Vector Storage**: FAISS database for efficient similarity search
- **Retrieval**: Top-k relevant documents are retrieved for each query
- **Generation**: Local LLM generates answers based on retrieved context

### Model Configuration
- **LLM**: EleutherAI/gpt-neo-1.3B (runs locally)
- **Embeddings**: HuggingFace sentence transformers
- **Vector Store**: FAISS with cosine similarity
- **Interface**: Streamlit for web-based interaction

## Dependencies

- streamlit
- langchain
- langchain-community
- langchain-huggingface
- faiss-cpu
- transformers
- torch
- pypdf
- python-dotenv

## Troubleshooting

### Common Issues

1. **Torch RuntimeError**: Ensure `.streamlit/config.toml` is configured correctly
2. **Import Errors**: Make sure all dependencies are installed with correct versions
3. **Model Loading**: Check internet connection for initial model download
4. **Memory Issues**: Ensure sufficient RAM for model loading (minimum 8GB recommended)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Disclaimer

This tool is for educational and research purposes only. Always consult qualified medical professionals for actual medical advice and diagnosis.
