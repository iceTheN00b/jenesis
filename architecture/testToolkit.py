import os
from enum import Enum
from langchain.tools import GoogleSearchResults
from langchain.utilities import GoogleScholarAPIWrapper
from langchain_community.tools.google_scholar import GoogleScholarQueryRun
from architecture.Tasks import TASKS

class testTools:
    def __init__(self, render):
        self.render = render
        self.gather_sources_chain = self.define_gather_sources_chain()
        self.summarize_chain = self.define_summarize_chain()
        self.thesis_chain = self.define_thesis_chain()
        self.outliner_chain = self.define_outliner_chain()
        self.research_write_chain = self.define_writer_chain()
        self.reviewer_chain = self.define_reviewer_chain()
        self.biblographer_chain = self.define_biblographer_chain()
        self.saver_tool = self.define_saver_tool()

    def define_gather_sources_chain(self):
        self.render.set_task(TASKS.SEARCH)
        def resource_chain(input=""):
            tool = GoogleScholarQueryRun(api_wrapper=GoogleScholarAPIWrapper())
            harvest = tool.run(input)
            return harvest
        return resource_chain

    def define_summarize_chain(self):
        self.render.set_task(TASKS.READ)
        def summarize_chain(input=""):
            return f"this is a summarization of the topic {input}"
        return summarize_chain

    def define_thesis_chain(self):
        self.render.set_task(TASKS.THINK)
        def thesis_chain(input=""):
            return "Exploring the Frontier of Artificial General Intelligence and Beyond: Leveraging Long-Short Term Memory (LLM) Based Autonomous Agents for Advancements in Cognitive Systems and Autonomous Intelligence"
        return thesis_chain

    def define_outliner_chain(self):
        self.render.set_task(TASKS.WRITE)
        def outline_chain(input = ""):
            return "the outline is as follows: Introduction, Literature Review"
        return outline_chain

    def define_writer_chain(self):
        self.render.set_task(TASKS.WRITE)
        def writer_chain(input = ""):
            return open("../paper.txt", "r").read()
        return writer_chain

    def define_reviewer_chain(self):
        self.render.set_task(TASKS.REVIEW)
        def review_chain(input = ""):
            return "the research paper est buono!"
        return review_chain

    def define_biblographer_chain(self):
        self.render.set_task(TASKS.WRITE)
        def biblographer_chain(input = ""):
            return "the list of the neccessary sources are : a, b"
        return biblographer_chain

    def define_saver_tool(self):
        self.render.set_task(TASKS.UPLOAD)
        def saver_tool(input = ""):
            open("../paper.txt", "w").write(input)
            return "Successfully saved File!"
        return saver_tool





#for future purposes, it will be useful for having higher level chains. like having one single chain for generating plans, rather than 2 different ones for different types of plans
#the question then is, is an attempt to control the output of agents at a low level worth it? Perhaps we should just skip it then? Well, that will be considered in a later version


"""
        
        plan_template = PromptTemplate.from_template(template=
        Generate a software execution plan for the idea {software_idea}.
        Be sure to include the following steps:
        0. Product Research: What are the things to keep in mind in the development of this program.
        1. Requirement Gathering: list features of the game
        2. Technology Stack: The technology stack, programming languages and any frameworks or libraries required
        3. Architecture Design: The structure of the project.
        4. Implementation: Using the tech stack to implement the idea.
        5. Testing and Optimization: Running the program to fix bugs and ensure everything works as it should.
        6. Documentation: including how to use the program.

        )#future steps can include evaluation, gathering feedback from other agents and such. Generally, roles are useful for aligning the agent towards specific types of tasks.

        plan_chain = LLMChain(llm=self.soul, prompt=plan_template)

        return plan_chain.run
        
"""