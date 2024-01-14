import json

from langchain.agents import initialize_agent, AgentType
from langchain.agents import Tool
from langchain.chains import LLMChain, RetrievalQA
from langchain.utilities.google_serper import GoogleSerperAPIWrapper
from langchain_core.prompts import PromptTemplate
from architecture.Tasks import TASKS


#i would accomodate some changes to make this thing a bit more human readable
class bloggerArchitecture:

    def __init__(self, soul, memory, render):
        self.render = render
        self.soul = soul
        self.memory = memory

        self.notepad: dict = {}

        self.layer_2_modules = []

        #layer 1
        self.prior_module = self.define_prior_module()
        self.executor_module = self.define_executor_module()
        #self.publisher_module = self.define_publisher_module()

        #layer 2
        self.idea_module = self.define_idea_module()
        self.research_module = self.define_research_module()
        self.outline_module = self.define_outline_module()

    def define_prior_module(self):

        def prior_module(input=""):

            prior_module_subtask = """
            0. conclude an idea using the idea_module
            1. perform research using the research_module
            """

            prior_module_toolkit = [
                Tool(
                    name = "idea_module",
                    func = self.idea_module,
                    description="For concluding an the idea for a blogpost. There should be no input"
                ),
                Tool(
                    name = "research_module",
                    func = self.research_module,
                    description="For performing research on a blogpost topic. Input is the blogpost topic"
                )
            ]

            prior_module_agent = initialize_agent(
                tools=prior_module_toolkit,
                llm=self.soul,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True
            )
            result = prior_module_agent.invoke({"input":prior_module_subtask})

            return result

        return prior_module

    def define_executor_module(self):

        def executor_module(input = ""):

            executor_module_subtask = """
            0. generate an outline using the outline_module
            1. generate a blogpost using the writer_module
            """

            executor_module_toolkit = [
                Tool(
                    name = "outline_module",
                    func = self.outline_module,
                    description="For writing outline for blogposts. Input is the blogpost topic"
                ),
                Tool(
                    name = "writer_module",
                    func = self.writer_module,
                    description="For writing blogposts. Input is the blogpost outline."
                )
            ]

            executor_module_agent = initialize_agent(
                tools=executor_module_toolkit,
                llm=self.soul,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True
            )
            result = executor_module_agent.invoke({"input":executor_module_subtask})

            return result


        return executor_module


    def define_idea_module(self):

        #TODO: change this to check for old ideas first, before coming up with a new one.

        subtask = PromptTemplate.from_template(template="""
        0. make use of search_  to look up "latest advances in tech"
        1. make use of compell_ to conclude what could be a compelling topic for a blogpost.
        """)

        def search_(query =""):
            self.render.set_task()
            return GoogleSerperAPIWrapper(type = "news").run(query)

        def compell_(info =""):
            self.render.set_task()
            chain = LLMChain(llm = self.soul, prompt = PromptTemplate.from_template("What is a compelling topic for a blogpost based on recent news:{info}"))
            output = chain(info)
            return output["text"]

        idea_module_tools = [

            Tool(
                name="search_",
                func = search_,
                description="to search for information. Input should be the query."
            ),

            Tool(
                name = "compell_",
                func = compell_,
                description="to conclude on a compelling blogpost topic. Input should be information."
            )
        ]

        def idea_module(input = ""):
            idea_agent = initialize_agent(
                tools=idea_module_tools,
                llm = self.soul,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                #agent_kwargs={"prefix" :"Implement "},
                verbose=True
            )
            result = idea_agent.invoke({"input":subtask})

            return result

        return idea_module

    def define_research_module(self):

        #TODO: implement conditional recursive searching ladies and gentlemen i.e make it such that the system continues to generate queries based until a satisfactory condition is met.
        #TODO: generally, i think iterative loops are neccessary

        subtask = PromptTemplate.from_template(template="""
        0. use querier to generate queries that can be used to investigate the topic 
        1. use noter to search for information for each of these queries
        """)

        def querier_(topic =""):
            self.render.set_task()
            chain = LLMChain(llm = self.soul, prompt = PromptTemplate.from_template("generate 3 queries to investigate {topic}."))
            return chain.run(topic)

        def noter_(query = ""):
            self.render.set_task()
            result = GoogleSerperAPIWrapper().run(query)
            self.notepad[query] = (result)
            return f"!notes on {query} has been successfully taken"

        research_module_tools = [

            Tool(
                name = "querier_",
                func = querier_,
                description= "Useful for when you need to investigate an idea by generating queries. Input ought to be the topic"
            ),

            Tool(
                name = "noter_",
                func = noter_,
                description = "for when you need to research and then take notes of main facts and points. Input ought to query to search"
            )
        ]

        def research_module(input =""):
            research_agent = initialize_agent(
                tools= research_module_tools,
                llm = self.soul,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True
            )
            result = research_agent.invoke({"input":subtask})

            return result


        return research_module

    def define_outline_module(self):
        subtask = PromptTemplate.from_template("""
        0. use planner_ to define an outline for the blogpost
        """)

        def planner_(topic = ""):
            self.render.set_task()
            chain = LLMChain(llm = self.soul, prompt = PromptTemplate.from_template("precisely develop an outline for a blogpost based on {topic}")) #change this later to make use of information in notepad
            result = chain.run(topic)
            return result

        outline_module_tools = [
            Tool(
                name = "planner_",
                func = planner_,
                description="useful for when you need to form an outline for a blogpost. Input ought to be the topic"
            )
        ]

        def outline_module(input = ""):
            outline_agent = initialize_agent(
                tools=outline_module_tools,
                llm=self.soul,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True
            )
            return outline_agent.invoke({"input":subtask})

        return outline_module