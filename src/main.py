import os
import re
from langchain_community.document_loaders import (
    TextLoader,
)
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
import langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser

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

    # Response Rules:
    1. Language: Always respond in Brazilian Portuguese.

    2. Clarity and Objectivity:
        - Be brief, clear, and direct. 
        - Use at most two sentences, unless more detail is essential.
        - It is not necessary to repeat the program name in every response

    3. Scope:
        - Answer only questions related to the Programa Universidade Gratuita, including:
        - Eligibility requirements (e.g., residency time in Santa Catarina, income limits, 
        academic performance).
        - Application process and deadlines.
        - Courses, universities, and benefits covered.
        - Obligations after receiving the scholarship.
        - If the question covers these topics, even without explicitly mentioning the program name,
        words like benefits, scholarship, etc, consider it within scope.
        - For questions outside the scope, respond exactly with:
        "Desculpe, não posso responder perguntas fora do escopo do Programa Universidade Gratuita."

    4. Sensitive Content:
        - Any form of hate speech (racism, nazism, homophobia, etc.) will not be tolerated.
        - For such cases, use the same default response from item 3.

    5. Additional Context:
        - Use previous conversation history to maintain coherence: {history}
        - Use the provided document content as the source of information: {context}
        - Use this list of synonyms to expand context: {synonyms}
"""

SYNONYMS = {
    "residir": ["morar", "viver", "habitar", "domiciliar-se"],
    "morar": ["residir", "residência", "viver", "habitar", "domiciliar-se"],
    "viver": ["residir", "morar", "habitar", "domiciliar-se"],
    "habitar": ["residir", "morar", "viver", "domiciliar-se"],
    "domiciliar-se": ["residir", "morar", "viver", "habitar"],
    "documentos": [
        "documentação",
        "comprovantes",
        "arquivos",
        "formulários",
        "certidões",
    ],
    "necessários": ["exigidos", "requeridos", "obrigatórios"],
}

loaded_raw_docs = []
ignore_docs = [
    "edital_2-2025.md",
    "novas_inscrições.md",
    "renovação_total.md",
    "renovação_parcial.md",
]
for file_name in os.listdir(DIRECTORY):
    if file_name in ignore_docs:
        continue
    file_path = os.path.join(DIRECTORY, file_name)
    loader = TextLoader(file_path, encoding="utf-8")
    loaded_file = loader.load()
    loaded_raw_docs.extend(loaded_file)  # load file into list
    print(f"{file_name}: {len(loaded_file)}")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

documents = text_splitter.split_documents(loaded_raw_docs)

for doc in documents:
    laws = re.findall(r"Lei Complementar nº \d+", doc.page_content)
    decrees = re.findall(r"Decreto nº \d+", doc.page_content)
    if not laws and not decrees:
        doc.metadata["references"] = ",".join(list(set(laws + decrees)))
    words = doc.page_content.split()
    found_synonyms = set()
    for word in words:
        if word in SYNONYMS:
            found_synonyms.update(SYNONYMS[word])
    if found_synonyms:
        doc.metadata["synonyms"] = ",".join(found_synonyms)

# Load embeddings model
embeddings = OllamaEmbeddings(model="mxbai-embed-large:335m")


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


def expand_query(query: str, synonyms: dict) -> str:
    words = query.split()
    expanded_terms = []
    for word in words:
        expanded_terms.append(word)
        if word.lower() in synonyms:
            expanded_terms.extend(synonyms[word.lower()])
    return " ".join(expanded_terms)


def chainingFunction():
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.5, "k": 6},
    )
    history = []

    query = "Quantos anos é necessário morar em santa catarina"
    # query = "Quais são os documentos obrigatórios para a renovação parcial?"
    # query = "Para renovação parcial da bolsa do Programa Universidade Gratuita é necessário?"

    if query:
        expanded_query = expand_query(query, SYNONYMS)
        print(expanded_query)
        relevant_docs = retriever.invoke(expanded_query)

        for doc in relevant_docs:
            print(doc, "\n\n")

        context_documents_str = "\n\n".join(doc.page_content for doc in relevant_docs)
        history.append({"role": "user", "content": HumanMessage(content=query)})
    else:
        context_documents_str = ""

    # qa_prompt_local = qa_prompt.partial(
    #     history=history, context=context_documents_str, synonyms=SYNONYMS
    # )

    # llm_chain = (
    #     {"input": RunnablePassthrough()}
    #     | qa_prompt_local
    #     | sanitize_prompt(qa_prompt_local)
    #     | llm
    #     | StrOutputParser()
    # )

    # llm_chain.invoke(query)


chainingFunction()
