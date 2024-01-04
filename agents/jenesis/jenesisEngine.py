from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_community.chat_models import ChatOpenAI
from agents.jenesis.jenesisRender import jenesisRender
from agents.jenesis.jenesisMind import jenesisMind
from langchain.agents import initialize_agent, AgentType

class jenesisEngine:
    def __init__(self):
        self.soul = self.setupSoul()
        self.render = self.setupRender()
        self.mind = self.setupMind()
        self.agent = self.setupAgent()
        self.setupCache()

        self.RENDER_DATA = "data/renderData/jenesis.json"

    def setupSoul(self):
        return ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

    def setupCache(self):
        set_llm_cache(SQLiteCache(database_path="data/cache/jenesisCache.db"))

    def setupRender(self):
        return jenesisRender()

    def setupMind(self):
        return jenesisMind(self.soul, self.render)

    def setupAgent(self):
        return initialize_agent(
            tools=[],
            llm = self.soul,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )

    def enginate(self):
        self.agent.run()