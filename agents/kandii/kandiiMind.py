from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.agents import Tool
from architecture.testToolkit import *

class kandiiMind: #the mind of the agent is nothing more than a collection of different chains, aimed at simulating cognition
    def __init__(self, soul, render):
        self.soul = soul
        self.memory = self.define_memory()
        self.render = render
        self.test_chains = self.define_test_chains()

    def define_memory(self):
        loader = DirectoryLoader("data/memory/kandiiMemory", glob="*.txt", loader_cls=TextLoader)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                           model_kwargs={"device": "cpu"})
        return Chroma.from_documents(chunks, embeddings)

    def define_test_chains(self):
        t = testTools(self.render)
        tools = [
            Tool(
                name="thesis_crafter",
                func=t.thesis_chain,
                description="relevant when you need to generate a thesis for a research topic. Input should be the research topic"
            ),

            Tool(
                name = "sourcer",
                func = t.gather_sources_chain,
                description="useful for when you need to find relevant sources to your research. Input should be the research topic."
            ),

            Tool(
                name = "summarizer",
                func = t.summarize_chain,
                description="useful when needing to summarize the main contents of a source. The input should be the name of the source and its authors"
            ),

            Tool(
                name = "outliner",
                func = t.outliner_chain,
                description="useful when you need to generate an outline for a research paper. The input ought to be the thesis of the paper"
            ),

            Tool(
                name = "writer",
                func = t.research_write_chain,
                description="necessary when writing a research paper. The input ought to be the outline of the research paper."
            ),

            Tool(
                name="saver",
                func = t.saver_tool,
                description="useful for saving a file to a specific location. Input should be the content of the file"
            ),

            Tool(
                name = "reviewer",
                func =t.reviewer_chain,
                description="useful for when it is neccessary to review a research paper. The input ought to be the contents of the research paper."
            ),

            Tool(
                name = "bilblographer",
                func = t.biblographer_chain,
                description="useful for when you need to create a biblography. Input should be details of all the sources you used in the paper."
            )
        ]

        return tools




















    #rn its an agent equipped with chains. what about an agent equipped with agents equipped with chains. any benefits?

    '''
     Tool(
                name = "realize_self_chain",
                func = k.realize_self_chain,
                description="useful whenever needed to answer questions pertaining to self, for example when asked personal questions, or discussing an opinion. The input is any question pertaining to self"
            ),
            
            Tool(
                name = "best_idea_from_options_chain",
                func = k.best_idea_from_options_chain,
                description = "generates multiple ideas from a single piece of information, ranks these ideas according to a criteria and then selects the best option. The input should be the information"
            )
            
             Tool(
                name = "design_software_plan_chain",
                func = k.design_software_plan_chain,
                description="generates a software development. Input should be the idea of the software."
            ),
            
             Tool(
                name="run_code_tool",
                func="",
                description="this tool is neccessary whenever the agent needs to run any piece of code."
                            "The functon takes 2 inputs. The first input is the language in capital letters, e.g PYTHON, JAVA etc."
                            "The second input is the body of the code"
            ),
    '''