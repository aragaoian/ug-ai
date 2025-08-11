import os
import re
import langchain
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()


class Chatbot:
    def __init__(
        self, chunk_size: int, chunk_overlap: int, synonyms_map: map, ignored_docs: list
    ):
        self.model = OllamaLLM(model="llama3.2")
        self.embeddings = OllamaEmbeddings(model="all-minilm")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT")
        self.DOCUMENTS_PATH = "ug_ai/docs"
        self.SYNONYMS_MAP = synonyms_map
        self.loaded_documents = None
        self.ignored_documents = ignored_docs
        self.history = []
        self.vector_db = None
        langchain.verbose = False
        langchain.debug = True

    def sanitize_prompt(prompt):
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
            loaded_raw_docs.extend(loaded_file)  # load file into list
            print(f"{file_name}: {len(loaded_file)}")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
        self.loaded_documents = text_splitter.split_documents(loaded_raw_docs)

        for doc in self.loaded_documents:
            laws = re.findall(r"Lei Complementar nº \d+", doc.page_content)
            decrees = re.findall(r"Decreto nº \d+", doc.page_content)
            if not laws and not decrees:
                doc.metadata["references"] = ",".join(list(set(laws + decrees)))

            words = doc.page_content.split()
            found_synonyms = set()
            for word in words:
                if word.lower() in self.SYNONYMS_MAP:
                    found_synonyms.update(self.SYNONYMS_MAP[word])

            if found_synonyms:
                doc.metadata["synonyms"] = ",".join(found_synonyms)

    def retrieve_relevant_docs(self, query):
        retriever = self.vector_db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.25, "k": 6},
        )

        if query:
            expanded_query = self.expand_query(query, self.SYNONYMS_MAP)
            print(expanded_query)
            relevant_docs = retriever.invoke(expanded_query)

            for doc in relevant_docs:
                print(doc, "\n\n")

            context_documents_str = "\n\n".join(
                doc.page_content for doc in relevant_docs
            )
            self.history.append(
                {"role": "user", "content": HumanMessage(content=query)}
            )
            return context_documents_str
        else:
            return ""

    def create_chain(self):
        pass
