# RAG AI Study Assistant

A Retrieval-Augmented Generation (RAG) study assistant that lets students upload PDF study material and ask questions about it.

The application extracts text from PDFs, splits it into chunks, creates vector embeddings, stores them in ChromaDB, retrieves the most relevant chunks for a question, and generates an answer using an LLM.

## Features

- Upload PDF study material
- Extract and split PDF text into chunks
- Generate embeddings and store them in ChromaDB
- Retrieve relevant study material for each question
- Generate answers grounded in the uploaded documents
- Show the source pages used to generate the answer
- Persistent local vector database

## Tech Stack

- Python
- Streamlit
- LangChain
- ChromaDB
- OpenAI Embeddings
- OpenAI or Groq LLM
- PyPDF

## Project Structure

```text
rag-study-assistant/
├── app.py                 # Streamlit user interface
├── rag_pipeline.py        # PDF processing, embeddings, retrieval, RAG chain
├── requirements.txt       # Python dependencies
├── .env.example           # Environment-variable template
├── .gitignore             # Files that should not be committed
└── README.md
```

## Setup Instructions

1. Clone the repository:

```bash
git clone [https://github.com/your-username/rag-study-assistant.git](https://github.com/your-username/rag-study-assistant.git)
cd rag-study-assistant
```

2. Create and activate a virtual environment:

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create your environment file:

```bash
cp .env.example .env
```

On Windows Command Prompt:

```bash
copy .env.example .env
```

5. Add your API key to `.env`:

```env
OPENAI_API_KEY=your_openai_api_key
```

Or, if using Groq:

```env
GROQ_API_KEY=your_groq_api_key
```

6. Run the application:

```bash
streamlit run app.py
```

7. Open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## How It Works

1. The user uploads a PDF.
2. `PyPDFLoader` extracts text from each page.
3. `RecursiveCharacterTextSplitter` divides the text into overlapping chunks.
4. `OpenAIEmbeddings` converts each chunk into a vector.
5. ChromaDB stores the vectors and original text.
6. When a question is asked, ChromaDB retrieves the most similar chunks.
7. The LLM answers using only the retrieved study material.
8. The app displays the answer and the source pages.

## Example Questions

- “Explain the main idea of this chapter.”
- “What are the important formulas in this PDF?”
- “Summarize page 5.”
- “What topics should I revise before the exam?”
- “Explain this concept in simple language.”

## Important Notes

- Do not commit your `.env` file or API keys to GitHub.
- The `chroma_db/` folder is generated locally and should not be committed.
- The assistant answers only from uploaded material and says when information is unavailable.
