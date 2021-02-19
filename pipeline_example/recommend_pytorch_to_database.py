from recommend_pytorch_train import MF
from recommend_pytorch_inf import get_top_n
import torch
import pandas as pd
import surprise
import datetime
import time
from google.oauth2 import service_account
import pandas_gbq

def get_model_from_disk():
    start_time = time.time()

    # data preload
    data = surprise.Dataset.load_builtin('ml-1m')
    trainset = data.build_full_trainset()
    testset = trainset.build_anti_testset()
    movies_df = pd.read_csv('./movies.dat',
                            sep="::", header=None, engine='python')
    movies_df.columns = ['iid', 'name', 'genre']
    movies_df.set_index('iid', inplace=True)

    # model preload
    k = 100  # latent dimension
    c_bias = 1e-6
    c_vector = 1e-6
    model = MF(trainset.n_users, trainset.n_items,
               k=k, c_bias=c_bias, c_vector=c_vector)
    model.load_state_dict(torch.load(
        './recommendation_model_pytorch.pkl'))  # TODO: prevent overwriting
    model.eval()

    print('Model and data preloading completed in ', time.time()-start_time)

    return model, testset, trainset, movies_df


def get_predictions(model, testset, trainset, movies_df):

    # save the recommended items for a given set of users
    sample_users = list(set([x[0] for x in testset]))[:4]


    df_list = []
    for uid in sample_users:
        recommended = get_top_n(model, testset, trainset, uid, movies_df, n=10)
        df_list.append(pd.DataFrame(data={'uid':[uid]*len(recommended),
                                    'recommended': [x[1] for x in recommended]},
            columns=['uid','recommended']))

    df = pd.concat(df_list, sort=False)
    df['pred_time'] = str(datetime.datetime.now())
    return df

def upload_to_bigquery(df):
    #Send predictions to BigQuery
    #requires a credential file in the current working directory
    table_id = "movie_recommendation_service.predicted_movies"
    project_id = "authentic-realm-276822"
    credentials = service_account.Credentials.from_service_account_file('./model-user.json')
    pandas_gbq.to_gbq(df, table_id, project_id=project_id, if_exists = 'replace', credentials=credentials)

if __name__ == '__main__':
    model, testset, trainset, movies_df = get_model_from_disk()
    df = get_predictions(model, testset, trainset, movies_df)
    print(df)
    upload_to_bigquery(df)
