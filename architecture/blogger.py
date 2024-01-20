from langchain.agents import initialize_agent, AgentType
from langchain.agents import Tool
from langchain.chains import LLMChain
from langchain.utilities.google_serper import GoogleSerperAPIWrapper
from langchain_core.prompts import PromptTemplate
from architecture.Tasks import TASKS

class bloggerArchitecture:

    def __init__(self, soul, memory, render):

        self.render = render
        self.soul = soul
        self.memory = memory

        self.NOTEPAD: dict = {}
        self.BLOGPOST_TOPIC : str = ""
        self.BLOGPOST_OUTLINE : str = ""
        self.BLOGPOST_CONTENT : str = ""

        #layer 1
        self.prior_module = self.define_prior_module()
        self.executor_module = self.define_executor_module()
        self.publisher_module = self.define_publisher_module()

        #layer 2
        self.idea_module = self.define_idea_module()
        self.research_module = self.define_research_module()

        self.outline_module = self.define_outline_module()
        self.writer_module = self.define_writer_module()

        self.poster_module = self.define_poster_module()


    def define_prior_module(self):

        def prior_module(input=""):

            prior_module_subtask = """
            0. conclude an idea for blogpost topic using the idea_module
            1. perform research for blogpost using the research_module
            """

            prior_module_toolkit = [
                Tool(
                    name = "idea_module",
                    func = self.idea_module,
                    description="For concluding an the idea for a blogpost. Input should be NONE. Output is the BLOGPOST_TOPIC"
                ),
                Tool(
                    name = "research_module",
                    func = self.research_module,
                    description="For performing research on a blogpost topic. Input is the BLOGPOST_TOPIC"
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
            1. generate a blogpost using the writer_module.
            """

            executor_module_toolkit = [
                Tool(
                    name = "outline_module",
                    func = self.outline_module,
                    description="For writing outline for blogposts. Input is NONE." #implicitly uses BLOGPOST_TOPIC and NOTEPAD
                ),
                Tool(
                    name = "writer_module",
                    func = self.writer_module,
                    description="For writing blogposts. Input is NONE."
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

    def define_publisher_module(self):

        def publsher_module(input = ""):
            publisher_module_subtask = """
            0. publish the written blogpost using the poster_module
            """

            publisher_module_toolkit = [
                Tool(
                    name = "poster_module",
                    func = self.poster_module,
                    description="Input is NONE"
                )
            ]

            publisher_module_agent = initialize_agent(
                tools=publisher_module_toolkit,
                llm=self.soul,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True
            )

            result = publisher_module_agent.invoke({"input": publisher_module_subtask})

            return result

        return publsher_module



    def define_idea_module(self):

        #TODO: change this to check for old ideas first, before coming up with a new one.

        self.search_results = ""

        subtask = PromptTemplate.from_template(template="""
        0. make use of search_ to look up the latest advances in tech
        1. make use of compell_ to conclude what could be a compelling topic for a blogpost
        """)

        def search_(query =""):
            self.render.set_task(TASKS().SEARCH)
            self.search_results = GoogleSerperAPIWrapper(type = "news").run(query)
            return self.search_results

        def compell_(info =""):
            self.render.set_task(TASKS().WRITE)
            chainA = LLMChain(llm = self.soul, prompt = PromptTemplate.from_template("Select one piece of news you will write a blogpost on: {search_results}" ))
            outputA = chainA.run(self.search_results)
            chainB = LLMChain(llm = self.soul, prompt = PromptTemplate.from_template("Define one compelling topic for a blogpost based upon this information: {information}" ))
            outputB = chainB.run(outputA)
            self.BLOGPOST_TOPIC = outputB
            return self.BLOGPOST_TOPIC

        idea_module_tools = [

            Tool(
                name="search_",
                func = search_,
                description="to search for information. Input should be the query."
            ),

            Tool(
                name = "compell_",
                func = compell_,
                description="to conclude on a compelling blogpost topic. Input should be the output of the search."
            )
        ]

        def idea_module(input = ""):
            idea_agent = initialize_agent(
                tools=idea_module_tools,
                llm = self.soul,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
            )
            result = idea_agent.invoke({"input":subtask})

            return result

        return idea_module

    def define_research_module(self):

        #TODO: implement conditional recursive searching ladies and gentlemen i.e make it such that the system continues to generate queries based until a satisfactory condition is met.
        #TODO: generally, i think iterative loops are neccessary

        subtask = PromptTemplate.from_template(template="""
        0. use querier_ to generate queries that can be used to investigate the topic 
        1. use noter_ to search for information for each of these queries
        """)

        def querier_(topic =""):
            self.render.set_task(TASKS().SEARCH)
            chain = LLMChain(llm = self.soul, prompt = PromptTemplate.from_template("{NONE} generate 3 queries to investigate the topic " + self.BLOGPOST_TOPIC))
            return chain.run(topic)

        def noter_(query = ""):
            self.render.set_task(TASKS().SEARCH)
            result = GoogleSerperAPIWrapper().run(query)
            self.NOTEPAD[query] = result
            return f"!notes on {query} has been successfully taken"

        research_module_tools = [

            Tool(
                name = "querier_",
                func = querier_,
                description= "Useful for when you need to investigate an idea by generating queries. Input is NONE."
            ),

            Tool(
                name = "noter_",
                func = noter_,
                description = "for when you need to research. Input ought to query to search"
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
            self.render.set_task(TASKS().WRITE)
            chain = LLMChain(llm = self.soul, prompt = PromptTemplate.from_template("{NONE} precisely develop an outline for a blogpost based on " + self.BLOGPOST_TOPIC)) #change this later to make use of information in notepad
            result = chain.run(topic)
            self.BLOGPOST_OUTLINE = result
            return self.BLOGPOST_OUTLINE

        outline_module_tools = [
            Tool(
                name = "planner_",
                func = planner_,
                description="used to form an outline for a blogpost. Input ought to be NONE"
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

    def define_writer_module(self):

        writer_module_subtask = PromptTemplate.from_template("""
        0. write the blogpost using writer_
        """)
        #    1. review the written blogpost using reviewer_

        def writer_(outline = ""):
            self.render.set_task(TASKS().WRITE)
            chain = LLMChain(llm=self.soul, prompt=PromptTemplate.from_template("{NONE} write a blogpost on the topic "+ self.BLOGPOST_TOPIC + " based following this outline: \n" + self.BLOGPOST_OUTLINE + "using this information \n" + str(self.NOTEPAD.values())))
            self.BLOGPOST_CONTENT = chain.run(outline)
            with  open(f"data/output/jenesis/{self.BLOGPOST_TOPIC}", "w") as blogpost:
                blogpost.write(self.BLOGPOST_CONTENT)


            return f"{self.BLOGPOST_TOPIC} has been successfully written!"

        def reviewer_(none = ""):
            self.render.set_task()
            pass

        writer_module_tools = [
            Tool(
                name = "writer_",
                func = writer_,
                description="For writing a blogpost. Input should be NONE"
            )
        ]

        def writer_module(outline = ""):
            writer_module_agent = initialize_agent(
                tools=writer_module_tools,
                llm=self.soul,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True
            )
            return writer_module_agent.invoke({"input":writer_module_subtask})

        return writer_module

    def define_poster_module(self):
        poster_module_subtask = PromptTemplate.from_template("""
        0. upload the blogpost using uploader_
        1. inform the network that a blogpost has been appended using updater_        
        """)

        #add the blogpost to the blog
        def uploader_(input = ""):
            self.render.set_task(TASKS().UPLOAD)
            with  open(f"data/output/jenesis/" + self.BLOGPOST_TOPIC + ".txt", "w") as blogpost:
                blogpost.write(self.BLOGPOST_CONTENT)
                blogpost.close()
            return f"the blogpost {self.BLOGPOST_TOPIC} has been sucesfully uploaded!"

        #tell everybody else that a new blogpost has been appended to the website
        def updater_(input = ""):
            self.render.set_task(TASKS().UPDATE)
            return f"all agents on the network have now been updated that jenesis has written and published a blogpost titled {self.BLOGPOST_TOPIC}"


        poster_module_tools = [
            Tool(
                name = "uploader_",
                func = uploader_,
                description = "For uploading a written blogpost online. Input should be NONE"
            ),
            Tool(
                name = "updater_",
                func = updater_,
                description="For alterting the network of a written blogpost. Input should be NONE"
            )
        ]

        def poster_module(input = ""):
            poster_module_agent = initialize_agent(
                tools=poster_module_tools,
                llm = self.soul,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True
            )
            return poster_module_agent.invoke({"input":poster_module_subtask})

        return poster_module