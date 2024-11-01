# RAG-Powered Document Question Answering System

A robust Question Answering system that uses Retrieval-Augmented Generation (RAG) to provide accurate answers from PDF documents. The system leverages ChromaDB for vector storage, FAISS for efficient similarity search, and LLaMA 3.1 for natural language generation.

## 🌟 Features

- PDF document processing and chunking
- Efficient vector embeddings creation and storage
- Fast similarity search using FAISS
- State-of-the-art language generation using LLaMA 3.1
- Support for multiple GPU devices
- Persistent vector storage
- Configurable chunk sizes and overlap
- Robust error handling and logging

## 🛠️ Technology Stack

- **LLM**: Meta's LLaMA 3.1 8B Instruct Model
- **Embedding Models**: 
  - Sentence Transformers (all-MiniLM-L6-v2)
  - MPNet (all-mpnet-base-v2)
- **Vector Stores**: 
  - ChromaDB
  - FAISS
- **PDF Processing**: PyMuPDF (fitz)
- **ML Framework**: PyTorch
- **NLP**: SpaCy

## 📋 Prerequisites

python

pip install -r requirements.txt

## 🚀 Usage

### 1. Index Documents

First, index your PDF documents:

For CSV file:
python create_embeddings.py

For VectorDB:
python index_documents.py 

This script will:
- Process PDF documents
- Create text chunks with configurable size and overlap
- Generate embeddings
- Store them in ChromaDB or CSV file

### 2. Query the System

Basic usage:
python improved-model.py "Your query here"

Or load with AI server:
ssh @aiscalar


## 🏗️ Project Structure

- `model/index_documents.py`: PDF processing and document indexing
- `model/model.py`: Core RAG implementation with ChromaDB
- `model/improved-model.py`: Enhanced version using FAISS
- `model/create_embeddings.py`: Standalone embedding creation utility

## ⚙️ Configuration

Key parameters can be configured in the respective scripts:

- Maximum tokens per chunk
- Overlap tokens between chunks
- Number of results to retrieve
- Temperature for generation
- Maximum new tokens for response
- Embedding model selection

## 🎯 Performance Optimization

The system includes several optimizations:
- GPU acceleration when available
- Apple M1/M2 MPS support
- FAISS indexing for fast similarity search
- Efficient chunking with overlap
- Batch processing for embeddings

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.


## ⚠️ Important Notes

- Ensure you have sufficient GPU memory when using larger models
- Store sensitive documents securely
- Consider chunking parameters based on your specific use case
- Monitor system resources when processing large documents


 