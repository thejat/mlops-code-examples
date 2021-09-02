# https://github.com/NicolasHug/Surprise
# can be replaced by explicitly importing the movielens data
from surprise import Dataset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import shuffle

class Loader():
    current = 0

    def __init__(self, x, y, batchsize=1024, do_shuffle=True):
        self.shuffle = shuffle
        self.x = x
        self.y = y
        self.batchsize = batchsize
        self.batches = range(0, len(self.y), batchsize)
        if do_shuffle:
            # Every epoch re-shuffle the dataset
            self.x, self.y = shuffle(self.x, self.y)

    def __iter__(self):
        # Reset & return a new iterator
        self.x, self.y = shuffle(self.x, self.y, random_state=0)
        self.current = 0
        return self

    def __len__(self):
        # Return the number of batches
        return int(len(self.x) / self.batchsize)

    def __next__(self):
        n = self.batchsize
        if self.current + n >= len(self.y):
            raise StopIteration
        i = self.current
        xs = torch.from_numpy(self.x[i:i + n])
        ys = torch.from_numpy(self.y[i:i + n])
        self.current += n
        return (xs, ys)


class MF(nn.Module):

    def __init__(self, n_user, n_item, k=18, c_vector=1.0, c_bias=1.0):
        super(MF, self).__init__()
        self.k = k
        self.n_user = n_user
        self.n_item = n_item
        self.c_bias = c_bias
        self.c_vector = c_vector

        self.user = nn.Embedding(n_user, k)
        self.item = nn.Embedding(n_item, k)

        # We've added new terms here:
        self.bias_user = nn.Embedding(n_user, 1)
        self.bias_item = nn.Embedding(n_item, 1)
        self.bias = nn.Parameter(torch.ones(1))

    def forward(self, train_x):
        user_id = train_x[:, 0]
        item_id = train_x[:, 1]
        vector_user = self.user(user_id)
        vector_item = self.item(item_id)

        # Pull out biases
        bias_user = self.bias_user(user_id).squeeze()
        bias_item = self.bias_item(item_id).squeeze()
        biases = (self.bias + bias_user + bias_item)

        ui_interaction = torch.sum(vector_user * vector_item, dim=1)

        # Add bias prediction to the interaction prediction
        prediction = ui_interaction + biases
        return prediction

    def loss(self, prediction, target):

        def l2_regularize(array):
            loss = torch.sum(array**2)
            return loss

        loss_mse = F.mse_loss(prediction, target.squeeze())

        # Add new regularization to the biases
        prior_bias_user = l2_regularize(self.bias_user.weight) * self.c_bias
        prior_bias_item = l2_regularize(self.bias_item.weight) * self.c_bias

        prior_user = l2_regularize(self.user.weight) * self.c_vector
        prior_item = l2_regularize(self.item.weight) * self.c_vector
        total = loss_mse + prior_user + prior_item + prior_bias_user + prior_bias_item
        return total


def main():
    # Data
    data = Dataset.load_builtin('ml-1m')
    trainset = data.build_full_trainset()
    uir = np.array([x for x in trainset.all_ratings()])
    train_x = test_x = uir[:, :2].astype(np.int64)  # for simplicity
    train_y = test_y = uir[:, 2].astype(np.float32)

    # Parameters
    lr = 5e-3
    k = 100  # latent dimension
    c_bias = 1e-6
    c_vector = 1e-6
    batchsize = 1024
    num_epochs = 40

    model = MF(trainset.n_users, trainset.n_items,
               k=k, c_bias=c_bias, c_vector=c_vector)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    for epoch in range(num_epochs):
        dataloader = Loader(train_x, train_y, batchsize=batchsize)
        itr = 0
        for batch in dataloader:
            itr += 1
            prediction = model(batch[0])
            loss = model.loss(prediction, batch[1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if itr % 100 == 0:
                print(f"epoch: {epoch}. iteration: {itr}. training loss: {loss}")

    torch.save(model.state_dict(),
               "../data/models/recommendation_model_pytorch.pkl")


if __name__ == '__main__':
    main()
