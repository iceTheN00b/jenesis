import json

#this class is neccessary in order to have the agent's behaviour be visualizable for any rendering application, a feature i think will become
#more common place were llm based agents turned towards consumers

class jenesisRender:
    def __init__(self):
        self.task = "IDLE"
        self.DATA = "renderData/jenesis.json"
        self.last_save = ""

    def setup(self):
        dat = json.load(open(self.DATA,"r"))
        self.location = dat["location"]
        self.task = dat["task"]

    def set_task(self, task = "IDLE"):
        self.task = task
        with open(self.DATA, "w") as render_data:
            render_data.write(json.dumps(self.task,indent=2))
        print(f"RENDER: " + self.task)



