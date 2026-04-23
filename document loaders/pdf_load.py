from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader("document loaders/GRU.pdf")
docs = data.load()
#print(docs[14].page_content)
#print(len(docs))
