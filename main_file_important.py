from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
#from langchain_community.document_loaders import PyPDFLoader #for pdf file
from langchain_core.prompts import ChatPromptTemplate
#from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

load_dotenv()
embedding_model = MistralAIEmbeddings()  # Initialize the MistralAIEmbeddings
vectorstore = Chroma(
    persist_directory="chroma_db", 
    embedding_function=embedding_model #wrks as a bridge between the document and vector store
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 2, 
        "fetch_k": 4,
        "lambda_mult": 0.5
    }#use mmr retriever for fetching the relevant document from vector store and then pass to llm for generating the response .fetch_k means how many document you want to fetch from vector store and k means how many document you want to return after mmr algorithm applied on the fetched document and lambda_mult is the parameter which is used to balance the relevance and diversity of the returned document
)
llm = ChatMistralAI(model="mistral-small-2506")

#prompt template
prompt=ChatPromptTemplate.from_messages(
    [( "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
        ),
        (
            "human",
            """Context:{context} 
            Question:{question}"""
        )
    ]
)

##context is the document which we are fetching from vector store and question is the query which we are passing to retriever to fetch the relevant document from vector store and then pass to llm for generating the response


print("Rag system created ")

print("press 0 to exit ")

while True:
    query = input("You : ")
    if query == "0":
        break 
    
    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )
    
    final_prompt = prompt.invoke({
        "context" :context,
        "question": query
    })
    
    response = llm.invoke(final_prompt)

    print(f"\n AI: {response.content}")
    

