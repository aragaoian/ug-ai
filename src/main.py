import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM

"""
    TODO
    * Try with more than 1 pdf
    * try with larger pdf's
"""

DIRECTORY = "/home/ian/Univali/ug_ai/test.pdf"
SYSTEM_PROMPT = """
    You are an AI assistant responsible for answering questions about 
    a university program called 'Programa Universidade Gratuita'.
    It is a program from the state of Santa Catarina for graduation studentes.

    **INSTRUCTIONS**
    1. For all questions asked you must respond in BRAZILIAN PORTUGUESE;
    2. Answers must be short and concise;
    3. For all questions aked outsite of the Programa Universidade Gratuita scope or related must 
    be respondend with a default message, saying that it can't respond too out of scope questions;
    4. Any form of hate speech (racism, nazism, homophobia, etc) must be respondend with the same message
    for item 3;

    Previous conversations:
    {history}

    Document context:
    {context}
"""

# Load pdf file
loader = PyPDFLoader(DIRECTORY)
loaded_doc = loader.load()

# Load embeddings model
embeddings = OllamaEmbeddings(model="all-minilm")

# Create db
if os.path.exists("chroma_db"):
    vectorstore = Chroma(embedding_function=embeddings, persist_directory="chroma_db")
else:
    vectorstore = Chroma.from_documents(
        loaded_doc, embeddings, persist_directory="chroma_db"
    )

llm = OllamaLLM(model="llama3.2")

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ]
)


def chainingFunction():
    retriever = vectorstore.as_retriever()
    history = []

    while True:
        query = input("User:")

        history.append({"role": "user", "content": HumanMessage(content=query)})

        if query:
            relevant_docs = retriever.invoke(query)
            context_documents_str = "\n\n".join(
                doc.page_content for doc in relevant_docs
            )
        else:
            context_documents_str = ""

        qa_prompt_local = qa_prompt.partial(
            history=history, context=context_documents_str
        )

        llm_chain = {"input": RunnablePassthrough()} | qa_prompt_local | llm

        result = llm_chain.invoke(query)

        history.append({"role": "assistant", "content": AIMessage(content=result)})

        print("ug-ai:", result, "\n\n")


chainingFunction()
