from flask import Flask, jsonify, request


def model(x):
    return 2*x + 2

app = Flask(__name__)
@app.route("/", methods=["GET"])
def predict():

    if "x" in request.args:
        try:
            return jsonify({'input': request.args['x'], 'prediction': model(float(request.args['x']))})
        except:
            pass

    return jsonify({'status': 'false', 'message': 'Input x was not passed.'})


if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5002)
