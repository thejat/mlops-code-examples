def predict(request):

    from flask import jsonify
    import pickle
    from google.cloud import storage

    def get_model(b, A):
        def line(x):
            return b * x + A
        return line

    storage_client = storage.Client()
    bucket = storage_client.get_bucket("simple-regression-bucket")
    blob = bucket.blob("simple_regression.pkl")
    blob.download_to_filename("/tmp/simple_regression.pkl")

    model_params = pickle.load(
        open('/tmp/simple_regression.pkl', 'rb'))

    model = get_model(model_params[0], model_params[1])

    request_json = request.get_json()
    if "x" in request.args:
        try:
            return jsonify({'input': request.args['x'], 'prediction': model(float(request.args['x']))})
        except:
            pass
    elif request_json and 'x' in request_json:
        return jsonify({'input': request_json['x'], 'prediction': model(float(request_json['x']))})
    else:
        return jsonify({'success': 'false', 'message': 'Input x was not passed correctly.'})