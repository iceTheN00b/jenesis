from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.agents import Tool

from architecture.blogger import bloggerArchitecture

class jenesisMind:  # the mind of the agent is nothing more than a collection of different chains, aimed at simulating cognition
    def __init__(self, soul, render):
        self.soul = soul
        self.memory = self.define_memory()
        self.render = render
        self.modules = self.define_modules()

    def define_memory(self):
        loader = DirectoryLoader("data/memory/jenesisMemory", glob="*.txt", loader_cls=TextLoader)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
        return Chroma.from_documents(chunks, embeddings)


    def define_modules(self):

        blogger = bloggerArchitecture(self.soul, self.memory, self.render)

        #ALL INPUTS ARE IMPLICITLY STATED - FAR MORE DEPENDABLE
        modules = [
            Tool(
                name = "prior_module",
                func = blogger.prior_module,
                description="Begin blogpost development. Input should be NONE"
                ),
            Tool(
                name = "executor_module",
                func = blogger.executor_module,
                description="Continue blogpost development. Input should be NONE"
            ),
            Tool(
                name = "publisher_module",
                func = blogger.publisher_module,
                description="Finalize blogpost development. Input should be NONE"
            )

        ]

        return modules
