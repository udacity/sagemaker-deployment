import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd

defaultParams = {
    # RNN Model Parameters
    'embedding_dim': 32,
    'hidden_dim': 100,
    'vocab_size': 5000,

    # Training Parameters
    'batch_size': 512,
    'epochs': 10,
}

class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(LSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.dense = nn.Linear(in_features=hidden_dim, out_features=1)

        self.sig = nn.Sigmoid()

    def forward(self, sentences, lengths):
        embeds = self.embedding(sentences)
        lstm_out, _ = self.lstm(embeds)
        out = self.dense(lstm_out)
        out = out[lengths - 1, range(len(lengths))]
        return self.sig(out.squeeze())

def build_model(hyperParams):

    embedding_dim = hyperParams.get('embedding_dim', defaultParams['embedding_dim'])
    embedding_dim = int(embedding_dim)

    hidden_dim = hyperParams.get('hidden_dim', defaultParams['hidden_dim'])
    hidden_dim = int(hidden_dim)

    vocab_size = hyperParams.get('vocab_size', defaultParams['vocab_size'])
    vocab_size = int(vocab_size)

    return LSTMClassifier(embedding_dim, hidden_dim, vocab_size)

def save_model(model, filename):

    print('Saving model.')
    model.cpu()
    torch.save(model, filename)

def load_model(filename):

    print('Loading model.')
    # return torch.load(filename, map_location=lambda storage, loc: storage)
    return torch.load(filename)

def fit_model(device, model, train_dl, hyperParams):
    model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters())
    loss_fn = torch.nn.BCELoss()

    epochs = hyperParams.get('epochs', defaultParams['epochs'])
    epochs = int(epochs)

    for epoch in range(epochs):
        total_loss = 0
        for batch in train_dl:
            res = _train_batch(device, model, batch, optimizer, loss_fn)
            total_loss += res.item()
        print("Epoch: ", epoch+1, " BCELoss: ", total_loss / len(train_dl))

def _train_batch(device, model, batch, optimizer, loss_fn):

    batch_X, batch_lengths, batch_y = batch

    batch_X = batch_X.to(device)
    batch_lengths = batch_lengths.to(device)
    batch_y = batch_y.to(device)

    optimizer.zero_grad()

    y_pred = model(batch_X.t(), batch_lengths)
    loss = loss_fn(y_pred, batch_y)
    loss.backward()
    optimizer.step()

    return loss.data

def predict_model(device, model, test_dl):
    model.to(device)
    model.eval()

    result = np.array([])

    for batch in test_dl:
        batch_X, batch_lengths = batch

        batch_X = batch_X.to(device)
        batch_lengths = batch_lengths.to(device)

        out = model(batch_X.t(), batch_lengths)
        out_label = out.round().long()

        result = np.append(result, out_label.cpu().numpy())

    return result

def build_train_loader(data, hyperParams):

    train_y = torch.from_numpy(data[[0]].values).float().squeeze()
    train_lengths = torch.from_numpy(data[[1]].values).long().squeeze()
    train_X = torch.from_numpy(data.drop([0, 1], axis=1).values).long()

    train_ds = torch.utils.data.TensorDataset(train_X, train_lengths, train_y)

    batch_size = hyperParams.get('batch_size', defaultParams['batch_size'])
    batch_size = int(batch_size)

    return torch.utils.data.DataLoader(train_ds, batch_size=batch_size)

def build_test_loader(data):

    data_lengths = torch.from_numpy(data[[0]].values).long().squeeze()
    if data_lengths.dim() == 0:
        data_lengths = torch.unsqueeze(data_lengths, 0)
    data_X = torch.from_numpy(data.drop([0], axis=1).values).long()

    data_ds = torch.utils.data.TensorDataset(data_X, data_lengths)
    return torch.utils.data.DataLoader(data_ds, batch_size=625)

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using Device: ", device)

    trainingParams = {}

    print("Creating Model")
    model = build_model(trainingParams)

    print("Reading Data")
    train_df = pd.read_csv('sentiment_test.csv', header=None, names=None)

    print("Creating Dataset")
    train_dl = build_train_loader(train_df, trainingParams)

    print("Fitting Model")
    fit_model(device, model, train_dl, trainingParams)

    print("Reading Test Data")
    test_df = pd.read_csv('sentiment_test.csv', header=None, names=None)

    test_df = pd.DataFrame(test_df.drop([0], axis=1).values)

    print("Creating Dataset of size 1")
    test_dl_1 = build_test_loader(test_df[:1])

    print("Predicting")
    res = predict_model(device, model, test_dl_1)

    print("Creating Dataset")
    test_dl = build_test_loader(test_df)

    print("Predicting")
    res = predict_model(device, model, test_dl)
