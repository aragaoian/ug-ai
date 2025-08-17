import os
from langchain_chroma import Chroma


class VectorDatabase:
    def __init__(self):
        pass

    @staticmethod
    def initialize_db(documents, embeddings):
        if os.path.exists("/src/chroma_db"):
            return Chroma(embedding_function=embeddings, persist_directory="chroma_db")
        else:
            return Chroma.from_documents(
                documents, embeddings, persist_directory="chroma_db"
            )
