# from dotenv import load_dotenv
# from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
# #from langchain_community.document_loaders import PyPDFLoader #for pdf file
# from langchain_core.prompts import ChatPromptTemplate
# #from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma

# load_dotenv()
# embedding_model = MistralAIEmbeddings()  # Initialize the MistralAIEmbeddings
# vectorstore = Chroma(
#     persist_directory="chroma_db", 
#     embedding_function=embedding_model #wrks as a bridge between the document and vector store
# )

# retriever = vectorstore.as_retriever(
#     search_type="mmr",
#     search_kwargs={
#         "k": 2, 
#         "fetch_k": 4,
#         "lambda_mult": 0.5
#     }#use mmr retriever for fetching the relevant document from vector store and then pass to llm for generating the response .fetch_k means how many document you want to fetch from vector store and k means how many document you want to return after mmr algorithm applied on the fetched document and lambda_mult is the parameter which is used to balance the relevance and diversity of the returned document
# )
# llm = ChatMistralAI(model="mistral-small-2506")

# #prompt template
# prompt=ChatPromptTemplate.from_messages(
#     [( "system",
#             """You are a helpful AI assistant.

# Use ONLY the provided context to answer the question.

# If the answer is not present in the context,
# say: "I could not find the answer in the document."
# """
#         ),
#         (
#             "human",
#             """Context:{context} 
#             Question:{question}"""
#         )
#     ]
# )

# ##context is the document which we are fetching from vector store and question is the query which we are passing to retriever to fetch the relevant document from vector store and then pass to llm for generating the response


# print("Rag system created ")

# print("press 0 to exit ")

# while True:
#     query = input("You : ")
#     if query == "0":
#         break 
    
#     docs = retriever.invoke(query)

#     context = "\n\n".join(
#         [doc.page_content for doc in docs]
#     )
    
#     final_prompt = prompt.invoke({
#         "context" :context,
#         "question": query
#     })
    
#     response = llm.invoke(final_prompt)

#     print(f"\n AI: {response.content}")
    



# from dotenv import load_dotenv

# from langchain_mistralai import ChatMistralAI
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma

# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# from langchain_core.chat_history import InMemoryChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory

# load_dotenv()

# # -----------------------
# # EMBEDDINGS + DB
# # -----------------------
# embedding_model = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

# vectorstore = Chroma(
#     persist_directory="chroma_db",
#     embedding_function=embedding_model
# )

# # -----------------------
# # RETRIEVER (SAFE)
# # -----------------------
# def retrieve_docs(x):
#     query = x["question"] if isinstance(x, dict) else str(x)

#     docs = vectorstore.max_marginal_relevance_search(
#         str(query),
#         k=6,
#         fetch_k=30,
#         lambda_mult=0.7
#     )

#     return docs

# def format_docs(docs):
#     return "\n\n".join(d.page_content for d in docs)

# # -----------------------
# # LLM
# # -----------------------
# llm = ChatMistralAI(model="mistral-small-2506")

# # -----------------------
# # PROMPT
# # -----------------------
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a helpful AI assistant. Use ONLY the provided context."),
#     ("human", "Context:\n{context}\n\nQuestion:\n{question}")
# ])

# # -----------------------
# # RAG CHAIN
# # -----------------------
# rag_chain = (
#     {
#         "context": RunnableLambda(retrieve_docs) | RunnableLambda(format_docs),
#         "question": RunnablePassthrough()
#     }
#     | prompt
#     | llm
# )

# # -----------------------
# # MEMORY STORE (FIXED)
# # -----------------------
# store = {}

# def get_session_history(session_id: str):
#     if session_id not in store:
#         store[session_id] = InMemoryChatMessageHistory()
#     return store[session_id]

# chat_chain = RunnableWithMessageHistory(
#     rag_chain,
#     get_session_history,
#     input_messages_key="question",
#     history_messages_key="chat_history"
# )

# # -----------------------
# # CHAT LOOP
# # -----------------------
# print("🚀 RAG CHAT READY (FIXED MEMORY)")
# print("Type 'exit' to stop\n")

# while True:
#     query = input("You: ")

#     if query.lower() == "exit":
#         break

#     result = chat_chain.invoke(
#         {"question": query},
#         config={"configurable": {"session_id": "user1"}}
#     )

#     print("\nAI:", result.content, "\n")

from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

# -----------------------
# EMBEDDINGS + DB
# -----------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

# -----------------------
# RETRIEVER (WITH SOURCES FIX)
# -----------------------
def retrieve_docs(x):
    query = x["question"] if isinstance(x, dict) else str(x)

    docs = vectorstore.max_marginal_relevance_search(
        str(query),
        k=6,
        fetch_k=30,
        lambda_mult=0.7
    )

    # 🔥 return BOTH docs + sources
    sources = []
    for d in docs:
        src = d.metadata.get("source", "Unknown")
        page = d.metadata.get("page", "N/A")
        sources.append(f"{src} (Page {page})")

    return {
        "docs": docs,
        "sources": list(set(sources))
    }

# -----------------------
def format_docs(data):
    docs = data["docs"]
    return "\n\n".join(d.page_content for d in docs)

def extract_sources(data):
    return data["sources"]

# -----------------------
# LLM
# -----------------------
llm = ChatMistralAI(model="mistral-small-2506")

# -----------------------
# PROMPT
# -----------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Use ONLY the provided context."),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}")
])

# -----------------------
# RAG CHAIN
# -----------------------
def prepare_input(x):
    data = retrieve_docs(x)
    return {
        "context": format_docs(data),
        "question": x["question"],
        "sources": data["sources"]
    }

rag_chain = RunnableLambda(prepare_input) | prompt | llm

# -----------------------
# MEMORY FIX
# -----------------------
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chat_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history"
)

# -----------------------
# CHAT LOOP
# -----------------------
print("🚀 RAG READY (WITH SOURCES + MEMORY FIXED)")
print("Type 'exit' to stop\n")

while True:
    query = input("You: ")

    if query.lower() == "exit":
        break

    result = chat_chain.invoke(
        {"question": query},
        config={"configurable": {"session_id": "user1"}}
    )

    print("\nAI:", result.content)

    # 🔥 SHOW SOURCES
    data = retrieve_docs({"question": query})
    print("\n📄 Sources:")
    for s in data["sources"]:
        print("-", s)

    print()