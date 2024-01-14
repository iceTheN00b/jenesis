import os
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.globals import set_llm_cache
from langchain.cache import SQLiteCache
from architecture.testToolkit import TASKS
from agents.kandii.kandiiRender import kandiiRender
from agents.kandii.kandiiMind import kandiiMind

class kandiiEngine:

    def __init__(self, go):
        self.soul = self.setupSoul()
        self.render = self.setupRender()
        self.mind = self.setupMind()
        self.agent = self.setupAgent()
        self.setupCache()
        self.RENDER_DATA = "data/renderData/kandii.json"
        self.goal = "Autonomous LLM Based Agents"
        self.go = go

    def setupSoul(self):
        return ChatOpenAI(temperature = 0, model = "gpt-3.5-turbo-0613")

    def setupCache(self):
        set_llm_cache(SQLiteCache(database_path="data/cache/kandiCache.db"))

    def setupRender(self):
        return kandiiRender()

    def setupMind(self):
        return kandiiMind(self.soul, self.render)

    def setupAgent(self):
        return initialize_agent(
            tools=self.mind.test_chains,
            llm = self.soul,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )

    def run(self):

        if self.go:

            prompt = f"""
            create the academic paper on the topic {self.goal} using these steps:

            0. generate a thesis
            1. Gather relevant sources to the topic
            2. Review and summarize each sources, especially its key findings.
            3. Develop an outline
            4. Write the paper in the according to the outline. the following must constitute: the introduction, literature review, methodology, reports, discussion, conclusion.
            5. Review this paper, ensuring the main points alighn with the thesis.
            6. Compile all the sources used in the paper to form a biblography.
            """
            self.agent.invoke({"input":prompt})
        else:
            self.render.set_task(TASKS.SLEEP)
            print("ZZZZ")


        #describe what it is you want to happen, and then evaluate as to whether it will actually happen


        '''
        in order to actuate the goal, how do i get the agent running?
        
        i could get the agent to create a plan towards the goal's actuation, and then actuate each step, for which the agent will then use its available tools to bring these to reality.
        
        this planning step could be the first stage.
        
        of course, the next steps will then act as the 
        
        i could establish the layout of development. This of course would involve: <following the example of ChatDev>
        
        0. understanding, maintaing questions in order to understand the user intent, <for kandii, she can generate these parts!>
        1. design of the application
        2. coding the application
        3. testing written code <such a stage could be done during the coding process though...>
        4. documenting what the code has done
        5. uploading the project, if it was one requested
        
        i want kandii to be free to define its own goals. that is all i want. and i think that indeed is something i can do, but maybe i ought not to do it now. cause in truth, she wont have a chance to run
        till i get rich at least.
        is retrieval even neccessary for encoding behaviour? predefining a plan works well enough i think.
        for now, i need kandii to be my go to engine that can help me handle tasks, specifically the development of programs. I think for now it is alright if it doesn't have complete free will.
        alternatively, jenesis can help me handle whatever concerning writing and posting to a blog, all automatically.
        as i give these agents new tools and better construct their architecture and prompts, i am certain i will at the end of the day come up with something worthwhile.
        
        
        perhaps a conversation to conversation situation is what is needed. well then we are talking about a multi-agent system then
        
        i think a single conversation indeed can be enough. thing is with all of these chains and how they are working, it is actually possible to put an entire development process into a single application.
        
        this thing is far too expensive. i need to reduce the costs on some fronts. this is a new type of efficiency. minimizing api costs.
        i really do think ai powered agents are a step in the right direction.
        
        the first chain generates the idea. <Optional>
        the first chain generates the plan. ----> following this {idea} generate a plan to execute this program
        the second chain executes the plan. ---->
        the fourth chain
        the third chain stores or send the completed project
        
        simulating a conversation between a 2 agents. i think that is interesting. the conscious interacting via conversation with the mind...
        the ability to retry failed options is something that i think is  important...
        
        however, this can be done simply by having the agent retry a chain until it meets some goal.
        this somehow works as to having the agent fulfill a subgoal up until a decision is made fam
        
        the ability to have an llm make use of functions is just beautiful ladies and gentlemen of the jury!
        i just wish i had enough money to make the best of this fam!
        i need wolf to exist such that i may have a means to keep making money my fans!
        
        i think having this system and continuously iterating it is useful
        
        1. generate a plan
        2. follow each step of the plan till the final product is made.
        
        chains can be given ending criteria, which is then useful for having an agent that needs to perform and perform an action
        until something has been done.
        '''

        #a writer bot could be easier to some degree, being that it just. Wolf makes really large bots.







        #goal = self.soul("define a goal to be achieved by ")

        #take in the blueprint of all the options available to you. Define your goal.
        #take in a list of all the informaton which you are aware of at that time. This of course includes the things you know, and the things you do not know.
        #this action space, construct a chain in a json format
        #use this json format to build a chain, and then execute this chain.
        #upon executing this chain, then evaluate where you stand towards your goal, evaluating as to whether you are there yet.
        #rinse and repeat, let's move on from this.


        #do i want really productive agents, or do i want them to be free?
        #using this json chain description, i can then be able to find something that is useful
        #the environment is the  mental landscape. this is what it is all about.

        #the meditative and sleeping processes are going to be very relevant for the agent. An ability to define itself and what it likes or does is something i think i am a little bit
        #scared of, but hey that is kandii. it is very possible that kandii becomes something other than that which i intend.

        #in a way, this approach kind of limits the ai...but in the same way we are limited!
        #the agent is given the freedom to formulate plans, but not to alter the building blocks used in formulating those plans!
        #its the same as being granted an api, the agent just has to make the most of what it's got, if it does not got it, then so be it!