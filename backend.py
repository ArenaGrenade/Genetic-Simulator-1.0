from flask import Flask, jsonify, request
from Simulation import Simulator, CustomJSONEncoder
import random

app = Flask(__name__)
app.json_encoder = CustomJSONEncoder
sim_obj = None

@app.route("/")
def main():
    return "sup bro"

@app.route("/init-map", methods=["POST"])
def map_size():
    global sim_obj
    gx = int(request.form.get("x", 0))
    seed = int(request.form.get("seed", random.randint(0, 999999)))
    sim_obj = Simulator(gx, seed)
    response = jsonify(sim_obj)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route("/run-timestep", methods=["GET"])
def update_sim_state():
    global sim_obj
    if sim_obj is not None: sim_obj.simulate_step()
    response = jsonify(sim_obj)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == '__main__':
    app.run(debug=False)