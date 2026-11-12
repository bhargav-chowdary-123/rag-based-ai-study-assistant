import os
import shutil
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4


def get_embeddings():
    """Return the embedding model used to convert text into vectors."""
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def get_llm():
    """
    Return an LLM.

    Uses Groq if GROQ_API_KEY is available, otherwise OpenAI.
    Change model names if your course instructs you to use another provider.
    """
    groq_key = os.getenv("GROQ_API_KEY")

    if groq_key:
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            api_key=groq_key,
        )

    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
    )


def load_pdf(file_path: str) -> List[Document]:
    """Load a PDF and return page-level LangChain documents."""
    loader = PyPDFLoader(file_path)
    return loader.load()


def split_documents(documents: List[Document]) -> List[Document]:
    """Split long documents into smaller overlapping chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def create_vector_store(chunks: List[Document], reset: bool = False):
    """
    Create or update a persistent Chroma vector store.

    Args:
        chunks: List of LangChain Document objects.
        reset: If True, deletes the existing database first.
    """
    if reset and os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=CHROMA_DIR,
        collection_name="study_material",
    )

    return vector_store


def get_vector_store():
    """Load an existing persistent Chroma vector store."""
    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(
            "No vector database found. Please upload and process a PDF first."
        )

    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=get_embeddings(),
        collection_name="study_material",
    )


def format_documents(documents: List[Document]) -> str:
    """Convert retrieved documents into a single context string."""
    context_parts = []

    for i, doc in enumerate(documents, start=1):
        page = doc.metadata.get("page", "unknown")
        source = doc.metadata.get("source", "uploaded document")
        context_parts.append(
            f"Source {i}\nFile: {source}\nPage: {page + 1}\nContent:\n{doc.page_content}"
        )

    return "\n\n".join(context_parts)


def build_rag_chain(vector_store):
    """
    Build a RAG chain:
    user question -> retrieve relevant chunks -> LLM answer with sources.
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful AI study assistant.

Use ONLY the following study material to answer the user's question.
If the answer is not present in the material, clearly say:
"I could not find this information in the uploaded study material."

Do not invent facts. Explain clearly and simply.

Study material:
{context}

Question:
{question}

Answer:
"""
    )

    rag_chain = (
        {
            "context": retriever | format_documents,
            "question": RunnablePassthrough(),
        }
        | prompt
        | get_llm()
        | StrOutputParser()
    )

    return rag_chain, retriever


def get_relevant_chunks(retriever, question: str):
    """Return retrieved source chunks so the UI can display citations."""
    return retriever.invoke(question)
