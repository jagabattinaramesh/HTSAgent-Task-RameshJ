import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

class RAGQA:
    def __init__(
        self,
        pdf_path: str,
        model_path: str,
        embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_dir: str = None,
    ):
        """
        RAG-based QA loader using local llama.cpp model and FAISS.
        Caches FAISS index on disk for faster subsequent loads.
        """
        # 1) Load and split PDF
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        # 2) Embeddings and FAISS index (with caching)
        embed = HuggingFaceEmbeddings(model_name=embed_model_name)
        # default index directory beside PDF
        if index_dir is None:
            index_dir = os.path.join(os.path.dirname(pdf_path), "faiss_index")
        if os.path.exists(index_dir):
            self.db = FAISS.load_local(index_dir, embed, allow_dangerous_deserialization=True)
        else:
            self.db = FAISS.from_documents(chunks, embed)
            self.db.save_local(index_dir)

        # 3) Local LLaMA via llama.cpp
        self.llm = LlamaCpp(
            model_path=model_path,
            n_ctx=1024,
            n_threads=4,
        )

        # 4) RetrievalQA chain
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.db.as_retriever(),
        )

    def ask(self, question: str) -> str:
        """
        Run a retrieval-augmented QA query on the loaded PDF.
        """
        # Using invoke for newer langchain versions
        if hasattr(self.qa, 'invoke'):
            return self.qa.invoke(question)
        return self.qa.run(question)
