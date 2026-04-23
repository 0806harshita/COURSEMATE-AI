#tiktoken is used to split the document into smaller chunks based on tokens. It is a library that provides tokenization and encoding for various language models, including those from OpenAI. By using tiktoken, you can ensure that your documents are split in a way that is compatible with the token limits of the language model you are using. This allows you to effectively manage and process large documents without exceeding the model's token limits.

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

data = PyPDFLoader("document loaders/GRU.pdf")
docs = data.load()

splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=10
)

chunks=splitter.split_documents(docs)
print(len(chunks))

for i in chunks:
    print(i.page_content)
    print("\n")