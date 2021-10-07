import sys
SYS_PATH_PREFIX = '/home/theja/Sync/uic/teach/'
sys.path.append(SYS_PATH_PREFIX + 'mlops-code/model_example_recommendation_pytorch') # for model definitions
sys.path.append(SYS_PATH_PREFIX + 'mlops-data/ml-1m') # for metadata
sys.path.append(SYS_PATH_PREFIX + 'mlops-data/models') # for the pytorch model
# TODO: improve by making a package

from recommend_pytorch_train import MF
from recommend_pytorch_inf import get_top_n, get_previously_seen
import torch
import surprise
import pandas as pd
import time
import random
from uuid import uuid4
from flask import (
    Flask,
    session,
    request,
    redirect,
    url_for,
    render_template_string
)
from planout.experiment import SimpleExperiment
from planout.ops.random import *


class ModelExperiment(SimpleExperiment):
    def setup(self):
        self.set_log_file('model_abtest.log')

    def assign(self, params, userid):
        params.use_pytorch = BernoulliTrial(p=0.5, unit=userid)
        if params.use_pytorch:
            params.model_type = 'pytorch1'
        else:
            params.model_type = 'pytorch2'



start_time = time.time()


# Metadata preload
movies_df = pd.read_csv('movies.dat',
                        sep="::", header=None, engine='python',
                        encoding="iso-8859-1")
movies_df.columns = ['iid', 'name', 'genre']
movies_df.set_index('iid', inplace=True)
data = surprise.Dataset.load_builtin('ml-1m')
trainset = data.build_full_trainset()
testset = trainset.build_anti_testset()



# Model preload
k = 100  # latent dimension
c_bias = 1e-6
c_vector = 1e-6
model = MF(trainset.n_users, trainset.n_items,
           k=k, c_bias=c_bias, c_vector=c_vector)
model.load_state_dict(torch.load(
    'recommendation_model_pytorch.pkl'))
model.eval()

print('Model and data preloading completed in ', time.time()-start_time)
model1 = model  # for demo purposes, both models are the same
model2 = model


app = Flask(__name__)
app.config.update(dict(
    DEBUG=True,
    SECRET_KEY='MODEL_TESTING_BY_THEJA_TULABANDHULA',
))


@app.route('/', methods=["GET"])
def main():
    # if no userid is defined make one up
    if 'userid' not in session:
        session['userid'] = str(random.choice(trainset.all_users()))

    model_perf_exp = ModelExperiment(userid=session['userid'])
    model_type = model_perf_exp.get('model_type')
    resp = {}
    resp["success"] = False

    print(model_type, resp, session['userid'])

    try:
        if model_type == 'pytorch1':
            user_ratings = get_top_n(
                model1, testset, trainset, session['userid'], movies_df, n=10)
        elif model_type == 'pytorch2':
            user_ratings = get_top_n(
                model2, testset, trainset, session['userid'], movies_df, n=10)

        print(user_ratings)
        resp["response"] = [x[1] for x in user_ratings]
        resp["success"] = True

        print(model_type, resp, session['userid'])

        return render_template_string("""
                <html>
                    <head>
                        <title>Recommendation Service</title>
                    </head>
                    <body>
                        <h3>
                            Recommendations for userid {{ userid }} based on {{ model_type }} are shown below: <br>
                        </h3>

                        <p>

                        {% for movie_item in resp['response'] %}
                              <h5> {{movie_item}}</h5>
                        {% endfor %}

                        </p>

                        <p>
                            What will be your rating of this list (rate between 1-10 where 10 is the highest quality)?
                        </p>
                        <form action="/rate" method="GET">
                            <input type="text" length="10" name="rate"></input>
                            <input type="submit"></input>
                        </form>
                    <br>
                    <p><a href="/">Reload without resetting my user ID. I'll get the same recommendations when I come back.</a></p>
                    <p><a href="/reset">Reset my user ID so I am a different user and will get re-randomized into a new treatment.</a></p>
                    </body>
                </html>
            """, userid=session['userid'], model_type=model_type, resp=resp)
    except:
        return render_template_string("""
            <html>
                <head>
                    <title>Recommendation Service</title>
                </head>
                <body>
                    <h3>
                        Recommendations for userid {{ userid }} based on {{ model_type }} are shown below. <br>
                    </h3>
                    <p>
                    {{resp}}
                    </p>

                    <p>
                        What will be your rating of this list (rate between 1-10 where 10 is the highest quality)?
                    </p>
                    <form action="/rate" method="GET">
                        <input type="text" length="10" name="rate"></input>
                        <input type="submit"></input>
                    </form>
                <br>
                <p><a href="/">Reload without resetting my user ID. I'll get the same recommendations when I come back.</a></p>
                <p><a href="/reset">Reset my user ID so I am a different user and will get re-randomized into a new treatment.</a></p>
                </body>
            </html>
            """, userid=session['userid'], model_type=model_type, resp=resp)


@app.route('/reset')
def reset():
    session.clear()
    return redirect(url_for('main'))


@app.route('/rate')
def rate():
    rate_string = request.args.get('rate')
    try:
        rate_val = int(rate_string)
        assert rate_val > 0 and rate_val < 11

        model_perf_exp = ModelExperiment(userid=session['userid'])
        model_perf_exp.log_event('rate', {'rate_val': rate_val})

        return render_template_string("""
                    <html>
                        <head>
                            <title>Thank you for the feedback!</title>
                        </head>
                        <body>
                            <p>You rating is {{ rate_val }}. Hit the back button or click below to go back to recommendations!</p>
                            <p><a href="/">Back</a></p>
                        </body>
                    </html>
                    """, rate_val=rate_val)
    except:
        return render_template_string("""
                    <html>
                        <head>
                            <title>Bad rating!</title>
                        </head>
                        <body>
                            <p>You rating could not be parsed. That's probably not a number between 1 and 10, so we won't be accepting your rating.</p>
                            <p><a href="/">Back</a></p>
                        </body>
                    </html>
                    """)


# start the flask app, allow remote connections
app.run(host='0.0.0.0')
