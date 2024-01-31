from flask import Flask, jsonify, request
import flask

def model(x):
    return 2*x+1

app = Flask(__name__)

@app.route("/", methods=["GET"])
def hello():
    return f"flask version: {flask.__version__}"

@app.route("/mypredict", methods=["GET"])
def predict():
    if "x" in request.args:
        try:
            return jsonify({'input': request.args['x'],
                             'prediction': model(float(request.args['x']))})
        except:
            pass

    return jsonify({'success': 'false', 'message': 'Input x was not passed.'})


if __name__ == '__main__':
    app.run()
