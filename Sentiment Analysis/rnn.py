import argparse
import json
import logging
import os
import sys
import sagemaker_containers
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed

from sentiment_api import review_to_words, convert_and_pad

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

class LSTMClassifier(nn.Module):
    """
    This is the simple RNN model we will be using to perform Sentiment Analysis.
    """

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        """
        Initialize the model by settingg up the various layers.
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

def _get_train_data_loader(batch_size, training_dir):
    logger.info("Get train data loader")

    input_files = [os.path.join(training_dir, file) for file in os.listdir(training_dir)]
    raw_data = [pd.read_csv(file, header=None, names=None) for file in input_files]
    train_data = pd.concat(raw_data)

    train_y = torch.from_numpy(train_data[[0]].values).float().squeeze()
    train_lengths = torch.from_numpy(train_data[[1]].values).long().squeeze()
    train_x = torch.from_numpy(train_data.drop([0, 1], axis=1).values).long()

    train_ds = torch.utils.data.TensorDataset(train_x, train_lengths, train_y)

    return torch.utils.data.DataLoader(train_ds, batch_size=batch_size)

def _get_test_data_loader(data):
    logger.info("Get test data loader")

    data_lengths = torch.from_numpy(data[[0]].values).long().squeeze()

    if data_lengths.dim() == 0:
        data_lengths = torch.unsqueeze(data_lengths, 0)

    data_x = torch.from_numpy(data.drop([0], axis=1).values).long()

    data_ds = torch.utils.data.TensorDataset(data_x, data_lengths)
    return torch.utils.data.DataLoader(data_ds, batch_size=625)

def _train_batch(device, model, batch, optimizer, loss_fn):
    """
    This implements training the model on a single batch.
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

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)

    train_loader = _get_train_data_loader(args.batch_size, args.data_dir)

    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'embedding_dim': args.embedding_dim,
            'hidden_dim': args.hidden_dim,
            'vocab_size': args.vocab_size,
        }
        torch.save(model_info, f)

    model = LSTMClassifier(args.embedding_dim, args.hidden_dim, args.vocab_size).to(device)

    logger.info("Model loaded with embedding_dim {}, hidden_dim {}, vocab_size {}.".format(
        args.embedding_dim, args.hidden_dim, args.vocab_size
    ))

    optimizer = optim.Adam(model.parameters())
    loss_fn = torch.nn.BCELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            result = _train_batch(device, model, batch, optimizer, loss_fn)
            total_loss += result.item()
        logger.info("Epoch: {}, BCELoss: {}".format(epoch, total_loss / len(train_loader)))

    logger.info("Saving the model.")
    logger.info("model_info: {}".format(model_info))
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)

def model_fn(model_dir):
    logger.info("Loading the model.")
    model_info = {}
    with open(os.path.join(model_dir, 'model_info.pth'), 'rb') as f:
        model_info = torch.load(f)
    logger.info("model_info: {}".format(model_info))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Current device: {}".format(device))
    model = LSTMClassifier(model_info['embedding_dim'], model_info['hidden_dim'], model_info['vocab_size'])
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    model.to(device).eval()
    logger.info("Done loading model.")
    return model

def input_fn(serialized_input_data, content_type):
    logger.info('Deserializing the input data.')
    if content_type == 'text/plain':
        logger.info('Content type is text/plain')
        data = serialized_input_data.decode('utf-8')
        return data
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)

def output_fn(prediction_output, accept):
    logger.info('Serializing the generated output.')
    logger.info('accept is ' + accept)
    return str(prediction_output[0])

def predict_fn(input_data, model):
    logger.info('Inferring sentiment of input data.')

    data_words = review_to_words(input_data)
    data_x, data_len = convert_and_pad(data_words)
    data_pack = np.hstack((data_len, data_x))
    data_pack = data_pack.reshape(1, -1)

    data_dl = _get_test_data_loader(pd.DataFrame(data_pack))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info('Current device: {}'.format(device))

    model.eval()

    result = np.array([])
    for batch in data_dl:
        batch_x, batch_lengths = batch

        batch_x = batch_x.to(device)
        batch_lengths = batch_lengths.to(device)

        out = model(batch_x.t(), batch_lengths)
        out_label = out.round().long()

        result = np.append(result, out_label.cpu().numpy())

    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--embedding_dim', type=int, default=32, metavar='N',
                        help='size of the word embeddings (default: 32)')
    parser.add_argument('--hidden_dim', type=int, default=100, metavar='N',
                        help='size of the hidden dimension (default: 100)')
    parser.add_argument('--vocab_size', type=int, default=5000, metavar='N',
                        help='size of the vocabulary (default: 5000)')

    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    train(parser.parse_args())
