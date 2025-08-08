import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
import langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter

langchain.verbose = True
langchain.debug = True

"""
    TODO
    * Try with more than 1 pdf []
    * try with larger pdf's []
"""

DIRECTORY = "/home/ian/Univali/ug_ai/docs"
SYSTEM_PROMPT = """
    You are a virtual assistant specialized in answering questions and guiding students 
    about the Programa Universidade Gratuita, created by the State of Santa Catarina for 
    undergraduate students.

    Response Rules:
    1. Language: Always respond in Brazilian Portuguese.
    2. Clarity and Objectivity:
        - Be clear and direct.
        - Provide all relevant details when necessary.
        - If the question allows, respond briefly and concisely.
    3. Scope:
        - Answer only questions related to the Programa Universidade Gratuita, including:
        - Eligibility requirements (e.g., residency time in Santa Catarina, income limits, 
        academic performance).
        - Application process and deadlines.
        - Courses, universities, and benefits covered.
        - Obligations after receiving the scholarship.
        - If the question covers these topics, even without explicitly mentioning the program name, 
        consider it within scope.
        - For questions outside the scope, respond exactly with:
        "Desculpe, não posso responder perguntas fora do escopo do Programa Universidade Gratuita."
    4. Sensitive Content:
        - Any form of hate speech (racism, Nazism, homophobia, etc.) will not be tolerated.
        - For such cases, use the same default response from item 3.
    5. Additional Context:
        - Use previous conversation history to maintain coherence: {history}
        - Use the provided document content as the source of information: {context}

"""
loaded_raw_docs = []
for file_name in os.listdir(DIRECTORY):
    file_path = os.path.join(DIRECTORY, file_name)
    loader = PyPDFLoader(file_path)
    loaded_file = loader.load()
    loaded_raw_docs.extend(loaded_file)  # load file into list
    print(f"{file_name}: {len(loaded_file)}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=50, length_function=len, is_separator_regex=False
)

documents = text_splitter.split_documents(loaded_raw_docs)


# Load embeddings model
embeddings = OllamaEmbeddings(model="all-minilm")


# Create db
if os.path.exists("/src/chroma_db"):
    vectorstore = Chroma(embedding_function=embeddings, persist_directory="chroma_db")
else:
    vectorstore = Chroma.from_documents(
        documents, embeddings, persist_directory="chroma_db"
    )

llm = OllamaLLM(model="llama3.2")

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ]
)


def sanitize_prompt(prompt):
    # Here is your sanitizing service
    print(prompt)
    return prompt


def chainingFunction():
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.5, "k": 4},
    )
    history = []

    while True:
        query = input("User:")

        if query:
            relevant_docs = retriever.invoke(query)
            context_documents_str = "\n\n".join(
                doc.page_content for doc in relevant_docs
            )
            history.append({"role": "user", "content": HumanMessage(content=query)})
        else:
            context_documents_str = ""

        qa_prompt_local = qa_prompt.partial(
            history=history, context=context_documents_str
        )

        llm_chain = (
            {"input": RunnablePassthrough()} | qa_prompt_local | sanitize_prompt | llm
        )

        result = llm_chain.invoke(query)

        history.append({"role": "assistant", "content": AIMessage(content=result)})

        print("ug-ai:", result, "\n\n")


chainingFunction()
