#load pdf
#sp;it into chunks
#create the embeddings
##store in vector database

##summary of loader ,splitter this is the single important file

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()

data=PyPDFLoader("document loaders/deeplearning.pdf")
documents=data.load()

splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks=splitter.split_documents(documents)
embedding_model=MistralAIEmbeddings()
vectorstore=Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
    )