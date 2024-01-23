import json
import requests

#this class is neccessary in order to have the agent's behaviour be visualizable for any rendering application, a feature i think will become
#more common place were llm based agents turned towards consumers

class jenesisRender:

    def __init__(self):
        #self.RENDER_LOC = "https://mobius.pythonanywhere.com/jenesis/render"
        self.RENDER_LOC = "https://127.0.0.1/jenesis/render"

    def set_task(self, task = "IDLE"):
        render_data = {
            "task": task
        }

        try:
            requests.post(self.RENDER_LOC, headers = render_data)
        except:
            print("unable to publish render data to central server")


