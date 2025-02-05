from flask import Flask, jsonify, request
import flask
import os

def model(x):
    return 2*x+1

app = Flask(__name__)

@app.route("/", methods=["GET"])
def hello():
    return f"flask version: {flask.__version__}"

"""
Endpoint to make a prediction based on the input parameter 'x'.

This function handles GET requests to the /mypredict route. It checks if the 
'x' parameter is present in the request arguments. If 'x' is present, it attempts 
to convert 'x' to a float and pass it to the model for prediction. The result is 
returned as a JSON response containing the input and the prediction. If 'x' is 
not present or an error occurs during processing, an error message is returned.

Returns:
    Response: A JSON response containing the input and prediction if successful, 
              or an error message if 'x' is not provided or an error occurs.
"""
@app.route("/mypredict", methods=["GET"])
def predict():
    # check if x is in the arguments
    if "x" in request.args:
        try:
            return jsonify({'input': request.args['x'],
                             'prediction': model(float(request.args['x']))})
        except:
            pass

    return jsonify({'success': 'false', 'message': 'Input x was not passed.'})


if __name__ == '__main__':
    host = os.getenv('FLASK_RUN_HOST', '127.0.0.1')
    port = int(os.getenv('FLASK_RUN_PORT', 5000))

    app.run(host=host, port=port)
