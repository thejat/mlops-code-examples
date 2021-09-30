# load Flask
import flask
from recommend_pytorch_train import MF
from recommend_pytorch_inf import get_top_n, get_previously_seen
import torch
import pandas as pd
import surprise
import time


app = flask.Flask(__name__)

start_time = time.time()

# data preload
data = surprise.Dataset.load_builtin('ml-1m')
trainset = data.build_full_trainset()
testset = trainset.build_anti_testset()
movies_df = pd.read_csv('../data/ml-1m/movies.dat',
                        sep="::", header=None, engine='python')
movies_df.columns = ['iid', 'name', 'genre']
movies_df.set_index('iid', inplace=True)

# model preload
k = 30  # latent dimension
c_bias = 1e-6
c_vector = 1e-6
model = MF(trainset.n_users, trainset.n_items,
           k=k, c_bias=c_bias, c_vector=c_vector)
model.load_state_dict(torch.load(
    '../data/models/recommendation_model_pytorch.pkl'))  # TODO: prevent overwriting
model.eval()

print('Model and data preloading completed in ', time.time()-start_time)


@app.route("/", methods=["GET"])
def recommend():

    data = {"success": False}

    if "uid" in flask.request.args:

        data['uid'] = str(flask.request.args['uid'])

        try:
            data['seen'] = get_previously_seen(
                trainset, data['uid'], movies_df)
            recommended = get_top_n(
                model, testset, trainset, data['uid'], movies_df, n=10)
            print(recommended)
            data['recommended'] = [x[1] for x in recommended]
            data["success"] = True
        except:
            pass

    return flask.jsonify(data)


# start the flask app, allow remote connections
if __name__ == '__main__':
    app.run(host='0.0.0.0')
