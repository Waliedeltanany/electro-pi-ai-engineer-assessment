# RAG Pipeline using LangChain, Gemini, and ChromaDB

## Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline that answers user questions based only on the provided PDF documents. The system retrieves the most relevant document chunks using semantic search and generates grounded answers using Google's Gemini model.

The application is designed to minimize hallucinations by restricting responses to the retrieved context and providing document citations.

---

## Features

- Load multiple PDF documents
- Split documents into semantic chunks
- Generate embeddings using Gemini Embeddings
- Store embeddings in Chroma Vector Database
- Retrieve the most relevant document chunks
- Generate context-aware answers using Gemini Flash
- Display document sources for every answer
- Return an appropriate response when no relevant information is found

---

## Technologies Used

- Python
- LangChain
- Google Gemini Flash
- Gemini Embeddings
- ChromaDB
- PyPDF
- Dotenv

---

## Project Structure

```
RAG_PIPELINE/
│
├── app.py
├── list_models.py
├── .env
├── docs/
│   ├── Artificial Intelligence Course.pdf
│   ├── Student Handbook.pdf
│   └── University Rules.pdf
│
├── chroma_db/
│
├── evidence/
│   ├── rag_query_1.png
│   ├── rag_query_2.png
│   ├── rag_query_3.png
│   └── rag_no_context.png
│
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd RAG_PIPELINE
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file and add your Google API key:

```env
GOOGLE_API_KEY=your_api_key
```

---

## Running the Project

Run the application using:

```bash
python app.py
```

Enter a question related to the provided PDF documents.

Example:

```
What AI topics are covered?
```

---

## Example Questions

- What AI topics are covered?
- What are the university attendance rules?
- How are students evaluated?
- What are the grading requirements?

---

## Evidence

The `evidence` folder contains screenshots demonstrating:

- Successful retrieval from different PDF documents
- Generated answers with citations
- Proper handling of questions outside the document collection

---

## Limitations

- Answers are limited to the uploaded PDF documents.
- Retrieval quality depends on chunk size and embedding quality.
- Scanned PDFs without OCR support may reduce retrieval accuracy.

---

## Future Improvements

- Support additional document formats (DOCX, TXT)
- Add conversation memory
- Improve citation formatting
- Deploy as a web application using Streamlit or FastAPI