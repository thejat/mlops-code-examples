from recommend_pytorch_train import MF
from surprise import Dataset
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import pprint


def get_top_n(model, testset, trainset, uid_input, movies_df, n=10):

    preds = []
    try:
        uid_input = int(trainset.to_inner_uid(uid_input))
    except KeyError:
        return preds

    # First map the predictions to each user.
    for uid, iid, _ in testset:  # inefficient
        try:
            uid_internal = int(trainset.to_inner_uid(uid))
        except KeyError:
            continue
        if uid_internal == uid_input:
            try:
                iid_internal = int(trainset.to_inner_iid(iid))
                movie_name = movies_df.loc[int(iid), 'name']
                preds.append((iid, movie_name, float(
                    model(torch.tensor([[uid_input, iid_internal]])))))
            except KeyError:
                pass
    # Then sort the predictions for each user and retrieve the k highest ones
    if preds is not None:
        preds.sort(key=lambda x: x[1], reverse=True)
        if len(preds) > n:
            preds = preds[:n]
    return preds


def get_previously_seen(trainset, uid, movies_df):
    seen = []
    for (iid, _) in trainset.ur[int(uid)]:
        try:
            seen.append(movies_df.loc[int(iid), 'name'])
        except KeyError:
            pass
        if len(seen) > 10:
            break
    return seen


def main():
    # Data
    movies_df = pd.read_csv('./movies.dat', sep="::",
                            header=None, engine='python')
    movies_df.columns = ['iid', 'name', 'genre']
    movies_df.set_index('iid', inplace=True)
    data = Dataset.load_builtin('ml-1m')
    trainset = data.build_full_trainset()
    testset = trainset.build_anti_testset()

    k = 30  # latent dimension
    c_bias = 1e-6
    c_vector = 1e-6

    model = MF(trainset.n_users, trainset.n_items,
               k=k, c_bias=c_bias, c_vector=c_vector)
    model.load_state_dict(torch.load('./recommendation_model_pytorch.pkl'))
    model.eval()

    # Print the recommended items for sample users
    sample_users = list(set([x[0] for x in testset]))[:4]

    for uid in sample_users:

        print('User:', uid)
        print('\n')

        print('\tSeen:')
        seen = get_previously_seen(trainset, uid, movies_df)
        pprint.pprint(seen)
        print('\n')

        print('\tRecommendations:')
        recommended = get_top_n(model, testset, trainset, uid, movies_df, n=10)
        pprint.pprint([x[1] for x in recommended])
        print('\n')


if __name__ == "__main__":
    main()
