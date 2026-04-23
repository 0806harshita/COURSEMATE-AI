from langchain_community.vectorstores import Chroma
#from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
load_dotenv()

docs = [
    Document(page_content="Python is widely used in Artificial Intelligence.", metadata={"source": "AI_book"}),
    Document(page_content="Pandas is used for data analysis in Python.", metadata={"source": "DataScience_book"}),
    Document(page_content="Neural networks are used in deep learning.", metadata={"source": "DL_book"}),
]

embedding_model = MistralAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=docs, 
    embedding=embedding_model, 
    persist_directory="chroma_db")# directory where the vector store will be saved

result=vectorstore.similarity_search("What is Python used for?", k=2)#k means how many result you want to get from the vector store
for e in result:
    print(e.page_content)
    print(e.metadata)

retriver=vectorstore.as_retriever()

docs=retriver.invoke("What is used in data scienc?")

for e in docs:
    print(e.page_content)
    print(e.metadata)