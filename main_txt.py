from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()

data=TextLoader("document loaders/notes.txt") #external data ko load krta hai llm me
docs=data.load() #converts notes.txt to document


template=ChatPromptTemplate(
    [("system","you are a ai that summarizes the text"),
     ("human","{data}")
    ]
)
model=ChatMistralAI(model="mistral-small-2506")
prompt=template.format_prompt(data=docs[0].page_content)
result=model.invoke(prompt)
print(result.content)