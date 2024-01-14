from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.agents import Tool

from architecture.blogger import bloggerArchitecture

class jenesisMind:  # the mind of the agent is nothing more than a collection of different chains, aimed at simulating cognition
    def __init__(self, soul, render):
        self.soul = soul
        self.memory = self.define_memory()
        self.render = render
        self.action_space = open("data/constant/space.txt", "r").read()
        self.modules = self.define_modules()

    def define_memory(self):
        loader = DirectoryLoader("data/memory/jenesisMemory", glob="*.txt", loader_cls=TextLoader)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                           model_kwargs={"device": "cpu"})
        return Chroma.from_documents(chunks, embeddings)


    def define_modules(self):

        blogger = bloggerArchitecture(self.soul, self.memory, self.render)

        modules = [
            Tool(
                name = "idea_module",
                func = blogger.idea_module,
                description="useful for coming up with a topic for a blogpost. Input ought to be your goal."
                ),
            Tool(
                name = "research_module",
                func = blogger.research_module,
                description="useful for when you need to research a blogpost topic. Input should be the topic"
            ),
            Tool(
                name = "outline_module",
                func = blogger.outline_module,
                description="useful for when you need to write an outline AFTER researching the topic. Input ought to be the topic."
            )
        ]

        return modules
