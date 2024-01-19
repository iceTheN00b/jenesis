print("..importing dependencies")
import flask
import json
from agents.jenesis.jenesisEngine import jenesisEngine
from threading import Thread
import logging

E = flask.Flask("mobius")

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

print("..sculpting jenesis")
jenesis = jenesisEngine()

@E.route("/jenesis")

def returnJenesisAgentRender():
    render_data = flask.make_response()

    jenesisState = json.load(open(jenesis.RENDER_DATA, "r"))
    render_data.headers["response"] = "true"
    render_data.headers["task"] = jenesisState["task"]

    return render_data

@E.route("/")

def index():
    return 0

E.run(host="127.0.0.9",port=9999,debug=True)

print("!running jenesis")
thread = Thread(target = jenesis.enginate)
thread.start()