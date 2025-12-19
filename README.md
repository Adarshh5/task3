# FAISS Vector Database with LangChain

A simple demonstration of creating, storing, and querying a FAISS vector database using LangChain and OpenAI embeddings.

## Overview

This project demonstrates how to:
- Create text embeddings using OpenAI's embedding model
- Store embeddings in a FAISS vector database
- Persist the vector database locally
- Load and query the database using similarity search

## Prerequisites

- Python 3.8+
- OpenAI API key

## Installation

1. Clone or download this project

2. Install required dependencies:
```bash
pip install langchain-openai langchain-community faiss-cpu python-dotenv
```



3. Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

## Project Structure

```
project/
│
├── task2.py              # Main script
├── .env                  # Environment variables (API keys)
├── local_db/             # Generated FAISS vector database (created on first run)
│   ├── index.faiss
│   └── index.pkl
└── README.md             # This file
```

## Usage

Run the script:
```bash
python task2.py
```

### What the Script Does

1. **Load Configuration**: Loads OpenAI API key from `.env` file
2. **Initialize Embeddings**: Creates an embedding model using OpenAI's `text-embedding-3-small`
3. **Prepare Text**: Defines sample text about FastAPI, Transformers, Vector databases, etc.
4. **Text Chunking**: Splits text into chunks (200 chars, 20 char overlap)
5. **Create Vector Store**: Generates embeddings and stores them in FAISS
6. **Save Database**: Persists the vector database to `local_db/` directory
7. **Load Database**: Demonstrates loading the saved database
8. **Query**: Performs similarity search with a sample query
9. **Display Results**: Prints the most relevant text chunk

## Configuration

### Chunk Size Settings

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,    # Maximum characters per chunk
    chunk_overlap=20   # Overlap between chunks (context preservation)
)
```



### Retriever Settings

```python
retriever = vectorsotre.as_retriever(search_kwargs={'k': 1})
```

- `k`: Number of similar documents to retrieve (default: 1)

## Example Output

```
"Transformers are neural networks used for NLP tasks.",
```

The output shows the most relevant text chunk based on the similarity search query.

## Key Components

### OpenAI Embeddings
```python
em_model = OpenAIEmbeddings(model='text-embedding-3-small')
```
Converts text into numerical vectors for semantic search.

### FAISS Vector Store
```python
vectorstore = FAISS.from_texts(chunks, em_model)
```
Efficient similarity search library by Facebook AI.

### Persistence
```python
vectorstore.save_local("local_db/")
vectorstore = FAISS.load_local(folder_path, embeddings, allow_dangerous_deserialization=True)
```
Saves and loads the vector database for reuse without regenerating embeddings.




## Note on Task C
Due to time constraints and my laptop was now working for installing big packges during the live interview, Task C was implemented using
cloud-based embeddings (OpenAI text-embedding-3-small) with FAISS instead of
locally loading the Hugging Face model `intfloat/e5-small-v2`. The pipeline
demonstrates the same vectorization and similarity search workflow.