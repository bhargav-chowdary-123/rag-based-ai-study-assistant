import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import (
    load_pdf,
    split_documents,
    create_vector_store,
    get_vector_store,
    build_rag_chain,
    get_relevant_chunks,
)

load_dotenv()

st.set_page_config(
    page_title="RAG AI Study Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 RAG AI Study Assistant")
st.write(
    "Upload your notes, textbook chapters, or lecture PDFs, then ask questions "
    "about them. The assistant answers using only your uploaded material."
)


@st.cache_resource(show_spinner=False)
def load_existing_vector_store():
    try:
        return get_vector_store()
    except FileNotFoundError:
        return None


vector_store = load_existing_vector_store()

if vector_store:
    st.success("Existing study-material database loaded.")
else:
    st.info("Upload a PDF to create your study-material database.")


st.header("1. Upload study material")

uploaded_file = st.file_uploader(
    "Upload a PDF file",
    type=["pdf"],
    help="Upload lecture notes, textbook chapters, assignments, or reference material.",
)

if uploaded_file is not None:
    if st.button("Process PDF and build knowledge base"):
        with st.spinner("Reading, splitting, embedding, and storing your PDF..."):
            try:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf",
                ) as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_path = temp_file.name

                documents = load_pdf(temp_path)
                chunks = split_documents(documents)

                if not chunks:
                    st.error("No readable text was found in this PDF.")
                else:
                    vector_store = create_vector_store(
                        chunks=chunks,
                        reset=True,
                    )
                    st.session_state["vector_store_ready"] = True

                    st.success(
                        f"Successfully processed **{uploaded_file.name}** "
                        f"and stored **{len(chunks)}** text chunks."
                    )

                os.remove(temp_path)

            except Exception as error:
                st.error(f"Failed to process the PDF: {error}")


st.header("2. Ask a question")

question = st.text_input(
    "Enter your question",
    placeholder="Example: Explain the main concept in chapter 2.",
)

if st.button("Get answer"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        try:
            current_vector_store = get_vector_store()
            rag_chain, retriever = build_rag_chain(current_vector_store)

            with st.spinner("Searching your study material and generating answer..."):
                answer = rag_chain.invoke(question)
                source_documents = get_relevant_chunks(retriever, question)

            st.subheader("Answer")
            st.write(answer)

            st.subheader("Sources used")
            if source_documents:
                for i, doc in enumerate(source_documents, start=1):
                    page = doc.metadata.get("page", "Unknown")
                    source = doc.metadata.get("source", "Uploaded PDF")

                    with st.expander(f"Source {i} — Page {page + 1}"):
                        st.write(doc.page_content)
            else:
                st.write("No relevant sources were found.")

        except FileNotFoundError:
            st.error("Please upload and process a PDF first.")
        except Exception as error:
            st.error(f"Something went wrong: {error}")


st.divider()
st.caption(
    "Built with Streamlit, LangChain, ChromaDB, and an LLM. "
    "Answers are generated from uploaded documents only."
)
