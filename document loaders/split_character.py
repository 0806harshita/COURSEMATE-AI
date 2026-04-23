from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator= "",
    chunk_size = 10,
    chunk_overlap=1 #overlap between chunks eg. if chunk size is 10 and overlap is 1 then 1 word will be repeated in next chunk
)

data=TextLoader("document loaders/notes.txt")
#print(data)
docs=data.load()
chunks=splitter.split_documents(docs)
print(len(chunks))


for i in chunks:
    print(i.page_content)
    print("\n")
