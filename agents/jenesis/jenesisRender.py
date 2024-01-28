import json
import requests

class jenesisRender:

    def __init__(self):
        self.RENDER_LOC = "http://127.0.0.1:10000/jenesis/render"

    def set_task(self, task = "IDLE"):
        render_data = {
            "task": task
        }

        try:
            requests.post(self.RENDER_LOC, headers = render_data)
        except:
            print("unable to publish render data to central server")


