import os
import langchain
from vector_db import VectorDatabase
from typing import List
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import BM25Retriever, EnsembleRetriever

load_dotenv()


class Chatbot:
    bm25_retriever: BM25Retriever
    loaded_documents: List[Document]
    ensemble_retriever: EnsembleRetriever
    SYSTEM_PROMPT: str
    SYNONYMS_MAP: List[str]
    vector_db: VectorDatabase
    vector_retriever: object

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        ignored_docs: list,
        md_headers_list: list,
    ):
        self.llm = OllamaLLM(model="llama3.2")
        self.embeddings = OllamaEmbeddings(model="all-minilm")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.DOCUMENTS_PATH = "/home/ian/Univali/ug_ai/files"
        self.ignored_documents = ignored_docs
        self.loaded_documents = []
        self.history = []
        self.headers_to_split_on = md_headers_list
        self.read_system_prompt()
        self.parse_documents()
        self.initialize_vector_db()
        self.create_vector_retriever()
        self.create_bm25_retriever()
        self.initialize_ensamble_retriever()
        self.qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.SYSTEM_PROMPT),
                ("human", "{input}"),
            ]
        )

        langchain.verbose = False
        langchain.debug = False

    def read_system_prompt(self):
        with open("system_prompt.txt", "r") as file:
            self.SYSTEM_PROMPT = file.read()

    def read_synonyms_map():
        pass

    def initialize_vector_db(self):
        self.vector_db = VectorDatabase.initialize_db(
            self.loaded_documents, self.embeddings
        )

    def create_vector_retriever(self):
        self.vector_retriever = self.vector_db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "score_threshold": 0.25,
                "k": 5,
            },  # 1- 0.25 = 0.75 (cosine distance, not similarity)
        )

    def create_bm25_retriever(self):
        self.bm25_retriever = BM25Retriever.from_documents(self.loaded_documents)
        self.bm25_retriever.k = 5  # Retrieve top 2 results

    def initialize_ensamble_retriever(self):
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.vector_retriever], weights=[0.4, 0.6]
        )

    def sanitize_prompt(self, prompt):
        print(prompt)
        return prompt

    def expand_query(self, query: str):
        words = query.split()
        expanded_terms = []
        for word in words:
            expanded_terms.append(word)
            if word.lower() in self.SYNONYMS_MAP:
                expanded_terms.extend(self.SYNONYMS_MAP[word.lower()])
        return " ".join(expanded_terms)

    def parse_documents(self):
        loaded_raw_docs = []
        for file_name in os.listdir(self.DOCUMENTS_PATH):
            if file_name in self.ignored_documents:
                continue
            file_path = os.path.join(self.DOCUMENTS_PATH, file_name)
            loader = TextLoader(file_path, encoding="utf-8")
            loaded_file = loader.load()
            loaded_raw_docs.extend(loaded_file)  # load raw file into list
            print(f"{file_name}: {len(loaded_file)}")

        markdown_splitter = MarkdownHeaderTextSplitter(self.headers_to_split_on)
        for doc in loaded_raw_docs:
            self.loaded_documents.extend(markdown_splitter.split_text(doc.page_content))

    def retrieve_relevant_docs(self, query):
        context_documents_str = ""
        if query:
            relevant_docs = self.ensemble_retriever.invoke(query)

            for doc in relevant_docs:
                print(doc, "\n\n")

            context_documents_str = "\n\n".join(
                doc.page_content for doc in relevant_docs
            )
            self.history.append(
                {"role": "user", "content": HumanMessage(content=query)}
            )
        return context_documents_str

    def chain_function(self, query):
        qa_prompt_local = self.qa_prompt.partial(
            history=self.history, context=self.retrieve_relevant_docs(query)
        )

        llm_chain = (
            {"input": RunnablePassthrough()}
            | qa_prompt_local
            | self.sanitize_prompt(qa_prompt_local)
            | self.llm
            | StrOutputParser()
        )

        llm_response = llm_chain.invoke(query)
        return llm_response
