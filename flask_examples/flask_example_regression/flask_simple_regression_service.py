import pickle
from flask import Flask, jsonify, request


def get_model(b, A):
    def line(x):
        return b * x + A
    return line


model_params = [2,2]
model = get_model(model_params[0], model_params[1])


app = Flask(__name__)


@app.route("/", methods=["GET"])
def predict():

    if "x" in request.args:
        try:
            return jsonify({'input': request.args['x'], 'prediction': model(float(request.args['x']))})
        except:
            pass

    return jsonify({'success': 'false', 'message': 'Input x was not passed correctly.'})


if __name__ == '__main__':
    app.run()
