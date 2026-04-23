from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader #for pdf file
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()

data=PyPDFLoader("document loaders/GRU.pdf") #external data ko load krta hai llm me
docs=data.load() #converts GRU.pdf to document


template=ChatPromptTemplate.from_messages(
    [("system","you are a ai that summarizes the text"),
     ("human","{data}")
    ]
)
model=ChatMistralAI(model="mistral-small-2506")
prompt=template.format_prompt(data=docs)
result=model.invoke(prompt)
print(result.content)