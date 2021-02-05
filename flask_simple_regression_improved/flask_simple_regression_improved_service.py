import pickle
from flask import Flask, jsonify, request, render_template


def get_model(b, A):
    def line(x):
        return b * x + A
    return line


model_params = pickle.load(
    open('../../data/models/simple_regression.pkl', 'rb'))
model = get_model(model_params[0], model_params[1])


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def predict():

    if "x" in request.args:
        try:
            return jsonify({'input': request.args['x'], 'prediction': model(float(request.args['x']))})
        except:
            return jsonify({'success': 'false', 'message': 'Input x was not passed correctly.'})
    elif request.method == 'POST':
        result = {'x': request.form.get('x'), 'prediction': None}
        try:
            x = float(request.form['x'])
            result['prediction'] = model(float(request.form['x']))
        except:
            pass
        return render_template('result.html', result=result)

    return render_template('index.html')


if __name__ == '__main__':
    app.run()
