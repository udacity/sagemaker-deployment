"""
The main module used in both train and serve mode. This module provides implementation of
the various functions used to interact with model artifacts (training and inference) as well
as interacting with the training data.
"""

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

DEFAULT_PARAMS = {
    # RNN Model Parameters
    'embedding_dim': 32,
    'hidden_dim': 100,
    'vocab_size': 5000,

    # Training Parameters
    'batch_size': 512,
    'epochs': 10,
}

class LSTMClassifier(nn.Module):
    """
    This is the simple RNN model we will be using to perform Sentiment Analysis.
    """

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        """
        Initialize the model by setting up the various layers.
        """
        super(LSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.dense = nn.Linear(in_features=hidden_dim, out_features=1)

        self.sig = nn.Sigmoid()

    def forward(self, sentences, lengths):
        """
        Perform a forward pass of our model on some input.
        """
        embeds = self.embedding(sentences)
        lstm_out, _ = self.lstm(embeds)
        out = self.dense(lstm_out)
        out = out[lengths - 1, range(len(lengths))]
        return self.sig(out.squeeze())

def build_model(hyperparams):
    """
    Given a collection of hyperparameters, construct the RNN model.
    """
    embedding_dim = hyperparams.get('embedding_dim', DEFAULT_PARAMS['embedding_dim'])
    embedding_dim = int(embedding_dim)

    hidden_dim = hyperparams.get('hidden_dim', DEFAULT_PARAMS['hidden_dim'])
    hidden_dim = int(hidden_dim)

    vocab_size = hyperparams.get('vocab_size', DEFAULT_PARAMS['vocab_size'])
    vocab_size = int(vocab_size)

    return LSTMClassifier(embedding_dim, hidden_dim, vocab_size)

def save_model(model, filename):
    """
    Save the model to disk. Here we use torch.save since we can't easily pass
    hyperparameter data to the inference container.
    """
    print('Saving model.')
    model.cpu()
    torch.save(model, filename)

def load_model(filename):
    """
    Load the model from disk.
    """
    print('Loading model.')
    return torch.load(filename)

def fit_model(device, model, train_dl, hyperparams):
    """
    Fit the model using the training data provided.
    """
    # First, we load the model to the gpu if we are using one and then we make
    # sure to put the model in training mode. This can affect things like Dropout
    model.to(device)
    model.train()

    # Set up the optimizer and loss function
    optimizer = optim.Adam(model.parameters())
    loss_fn = torch.nn.BCELoss()

    epochs = hyperparams.get('epochs', DEFAULT_PARAMS['epochs'])
    epochs = int(epochs)

    for epoch in range(epochs):
        total_loss = 0
        for batch in train_dl:
            result = _train_batch(device, model, batch, optimizer, loss_fn)
            total_loss += result.item()
        print("Epoch: ", epoch+1, " BCELoss: ", total_loss / len(train_dl))

def _train_batch(device, model, batch, optimizer, loss_fn):
    """
    This implements training on a single batch.
    """
    # First, split up the batch tuple into its components
    batch_x, batch_lengths, batch_y = batch

    # Make sure to place the batch data on the appropriate device
    batch_x = batch_x.to(device)
    batch_lengths = batch_lengths.to(device)
    batch_y = batch_y.to(device)

    # Compute a single pass over the training batch
    optimizer.zero_grad()

    y_pred = model(batch_x.t(), batch_lengths)
    loss = loss_fn(y_pred, batch_y)
    loss.backward()
    optimizer.step()

    return loss.data

def predict_model(device, model, test_dl):
    """
    Perform some predictions (inference) using our model applied to the data provided.
    """
    # Place the model on the appropriate device and make sure to put
    # the model into evaluation mode.
    model.to(device)
    model.eval()

    result = np.array([])

    for batch in test_dl:
        batch_x, batch_lengths = batch # Split up the batch tuple

        # Load the data onto the appropriate device
        batch_x = batch_x.to(device)
        batch_lengths = batch_lengths.to(device)

        # Perform a forward pass on the input
        out = model(batch_x.t(), batch_lengths)
        out_label = out.round().long()

        result = np.append(result, out_label.cpu().numpy())

    return result

def build_train_loader(data, hyperparams):
    """
    This method constructs a Dataloader out of the provided pandas Dtaframe, assuming that each
    row has the form | label | length | review | where review is a sequence of integers.
    """
    train_y = torch.from_numpy(data[[0]].values).float().squeeze()
    train_lengths = torch.from_numpy(data[[1]].values).long().squeeze()
    train_x = torch.from_numpy(data.drop([0, 1], axis=1).values).long()

    train_ds = torch.utils.data.TensorDataset(train_x, train_lengths, train_y)

    batch_size = hyperparams.get('batch_size', DEFAULT_PARAMS['batch_size'])
    batch_size = int(batch_size)

    return torch.utils.data.DataLoader(train_ds, batch_size=batch_size)

def build_test_loader(data):
    """
    This method constructs a Dataloader out of the provided pandas Dataframe, assuming that each
    row has the form | length | review | where review is a sequence of integers.
    """
    data_lengths = torch.from_numpy(data[[0]].values).long().squeeze()
    # Fix up dimensions if the input has a single row
    if data_lengths.dim() == 0:
        data_lengths = torch.unsqueeze(data_lengths, 0)

    data_x = torch.from_numpy(data.drop([0], axis=1).values).long()

    data_ds = torch.utils.data.TensorDataset(data_x, data_lengths)
    return torch.utils.data.DataLoader(data_ds, batch_size=625)
