import json
import requests

#this class is neccessary in order to have the agent's behaviour be visualizable for any rendering application, a feature i think will become
#more common place were llm based agents turned towards consumers

class jenesisRender:
    def __init__(self):
        self.task = "IDLE"
        #self.DATA = "https://mobius.pythonanywhere.com/jenesis/render"
        self.DATA = "https://127.0.0.1/jenesis/render"

    def set_task(self, task = "IDLE"):
        self.task = task
        render_data = {
            "task":self.task
        }
        try:
            requests.get(self.DATA, render_data)
        except:
            print("unable to publish render data to central server")


