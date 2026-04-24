# import streamlit as st
# from dotenv import load_dotenv
# import tempfile
# import os

# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_mistralai import MistralAIEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain_mistralai import ChatMistralAI
# from langchain_core.prompts import ChatPromptTemplate


# load_dotenv()

# st.set_page_config(page_title="RAG Book Assistant")

# st.title("📚 RAG Book Assistant")
# st.write("Upload a PDF and ask questions from the document")

# uploaded_file = st.file_uploader("Upload a PDF book", type="pdf")


# if uploaded_file:

#     with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
#         tmp_file.write(uploaded_file.read())
#         file_path = tmp_file.name

#     st.success("PDF uploaded successfully!")

#     if st.button("Create Vector Database"):

#         with st.spinner("Processing document..."):

#             loader = PyPDFLoader(file_path)
#             docs = loader.load()

#             splitter = RecursiveCharacterTextSplitter(
#                 chunk_size=1000,
#                 chunk_overlap=200
#             )

#             chunks = splitter.split_documents(docs)

#             embeddings = MistralAIEmbeddings()

#             vectorstore = Chroma.from_documents(
#                 documents=chunks,
#                 embedding=embeddings,
#                 persist_directory="chroma_db"
#             )

#             vectorstore.persist()

#         st.success("Vector database created!")



# if os.path.exists("chroma_db"):

#     embeddings = MistralAIEmbeddings()

#     vectorstore = Chroma(
#         persist_directory="chroma_db",
#         embedding_function=embeddings
#     )

#     retriever = vectorstore.as_retriever(
#         search_type="mmr",
#         search_kwargs={
#             "k":4,
#             "fetch_k":10,
#             "lambda_mult":0.5
#         }
#     )

#     llm = ChatMistralAI(model="mistral-small-2506")

#     prompt = ChatPromptTemplate.from_messages(
#         [
#             (
#                 "system",
#                 """You are a helpful AI assistant.

# Use ONLY the provided context to answer the question.

# If the answer is not present in the context,
# say: "I could not find the answer in the document."
# """
#             ),
#             (
#                 "human",
#                 """Context:
# {context}

# Question:
# {question}
# """
#             )
#         ]
#     )

#     st.divider()
#     st.subheader("Ask Questions From the Book")

#     query = st.text_input("Enter your question")

#     if query:

#         docs = retriever.invoke(query)

#         context = "\n\n".join(
#             [doc.page_content for doc in docs]
#         )

#         final_prompt = prompt.invoke({
#             "context": context,
#             "question": query
#         })

#         response = llm.invoke(final_prompt)

#         st.write("### AI Answer")
#         st.write(response.content)

import streamlit as st
from dotenv import load_dotenv
import tempfile
import time

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ---------------- UI ----------------
st.set_page_config(page_title="RAG Assistant")
st.title("📚 RAG Book Assistant (FIXED)")

# ---------------- SESSION STATE ----------------
if "db_path" not in st.session_state:
    st.session_state.db_path = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- FILE UPLOAD ----------------
uploaded_files = st.file_uploader(
    "Upload PDFs/TXT",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

# ---------------- CREATE VECTOR DB ----------------
if uploaded_files and st.button("Create Vector DB"):

    PERSIST_DIR = f"chroma_db_{int(time.time())}"
    all_docs = []

    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            path = tmp.name

        loader = PyPDFLoader(path) if file.name.endswith(".pdf") else TextLoader(path)
        docs = loader.load()

        for d in docs:
            d.metadata["source"] = file.name
            d.metadata["page"] = d.metadata.get("page", "N/A")

        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )

    vectorstore.persist()
    st.session_state.db_path = PERSIST_DIR

    st.success("✅ Vector DB Created!")

# ---------------- CHAT SYSTEM ----------------
if st.session_state.db_path:

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory=st.session_state.db_path,
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatMistralAI(model="mistral-small-2506")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Use ONLY the context."),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}")
    ])

    # ---------------- CORE RAG FUNCTION ----------------
    def run_rag(query):

        docs = retriever.invoke(query)

        context = "\n\n".join(d.page_content for d in docs)

        sources = []
        for d in docs:
            src = d.metadata.get("source", "Unknown file")
            page = d.metadata.get("page", "N/A")
            sources.append(f"{src} (Page {page})")

        response = llm.invoke(
            prompt.invoke({
                "context": context,
                "question": query
            })
        )

        return response.content, list(set(sources))

    # ---------------- UI ----------------
    st.divider()
    st.subheader("💬 Chat with Documents")

    user_input = st.chat_input("Ask something...")

    if user_input:

        answer, sources = run_rag(user_input)

        st.session_state.chat_history.append({
            "user": user_input,
            "assistant": answer,
            "sources": sources
        })

        st.rerun()

    # ---------------- DISPLAY CHAT ----------------
    for chat in st.session_state.chat_history:

        st.chat_message("user").write(chat["user"])
        st.chat_message("assistant").write(chat["assistant"])

        if chat.get("sources"):
            st.write("📄 Sources:")
            for s in chat["sources"]:
                st.write("•", s)

else:
    st.warning("⚠️ Upload files and create vector DB first.")