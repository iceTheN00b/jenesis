from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.agents import Tool

from toolkit.bloggerToolkit import bloggerTools


class jenesisMind:  # the mind of the agent is nothing more than a collection of different chains, aimed at simulating cognition
    def __init__(self, soul, render):
        self.soul = soul
        self.memory = self.define_memory()
        self.render = render
        self.action_space = open("data/constant/space.txt", "r").read()
        self.toolkit = self.define_toolkit()

    def define_memory(self):
        loader = DirectoryLoader("data/memory/jenesisMemory", glob="*.txt", loader_cls=TextLoader)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                           model_kwargs={"device": "cpu"})
        return Chroma.from_documents(chunks, embeddings)


    def define_toolkit(self):

        x = bloggerTools()

        tools = [
            Tool(
            name = "idea_module",
            func = x.idea_module,
            description="useful for coming up with an idea. There should be no input"
            )
        ]

        return tools