import streamlit as st
from knowledge_gpt.ui import (
    wrap_doc_in_html,
    is_query_valid,
    is_file_valid,
    is_open_ai_key_valid,
    display_file_read_error,
)
from knowledge_gpt.core.parsing import read_file
from knowledge_gpt.core.chunking import chunk_file
from knowledge_gpt.core.embedding import embed_files
from knowledge_gpt.core.qa import query_folder

EMBEDDING = "openai"
VECTOR_STORE = "faiss"

def processFile(uploaded_file, openai_api_key, model):
    if not uploaded_file:
        raise Exception("Invalid File")

    try:
        file = read_file(uploaded_file)
    except Exception as e:
        display_file_read_error(e, file_name=uploaded_file.name)

    chunked_file = chunk_file(file, chunk_size=300, chunk_overlap=0)

    if not is_file_valid(file):
        raise Exception("Invalid File")


    with st.spinner("Indexing document... This may take a while⏳"):
        folder_index = embed_files(
            files=[chunked_file],
            embedding=EMBEDDING if model != "debug" else "debug",
            vector_store=VECTOR_STORE if model != "debug" else "debug",
            openai_api_key=openai_api_key,
        )

    return folder_index