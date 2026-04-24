# #load pdf
# #sp;it into chunks
# #create the embeddings
# ##store in vector database

# ##summary of loader ,splitter this is the single important file

# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_mistralai import MistralAIEmbeddings
# from langchain_community.vectorstores import Chroma
# from dotenv import load_dotenv
# load_dotenv()

# data=PyPDFLoader("document loaders/deeplearning.pdf")
# documents=data.load()

# splitter=RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200
# )

# chunks=splitter.split_documents(documents)
# embedding_model=MistralAIEmbeddings()
# vectorstore=Chroma.from_documents(
#     documents=chunks,
#     embedding=embedding_model,
#     persist_directory="chroma_db"
#     )

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

DATA_PATH = "document loaders"

def load_documents():
    documents = []

    for file in os.listdir(DATA_PATH):
        file_path = os.path.join(DATA_PATH, file)

        if file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)

        elif file.endswith(".txt"):
            loader = TextLoader(file_path)

        else:
            continue

        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = file
            doc.metadata["page"] = doc.metadata.get("page", "N/A")

        documents.extend(docs)

    return documents


def create_vector_db():

    import time
    persist_dir = f"chroma_db_{int(time.time())}"

    documents = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    embeddings = MistralAIEmbeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    vectorstore.persist()

    print(f"✅ DB created at: {persist_dir}")


if __name__ == "__main__":
    create_vector_db()