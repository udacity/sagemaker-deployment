# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import pickle
import sys
import signal
import traceback

import flask

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import model

from sentiment_api import review_to_words, convert_and_pad

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    custom_model = model.load_model(os.path.join(model_path, 'sentiment-pytorch'))
except Exception as e:
    print('Exception while loading model: ' + str(e))
    custom_model = None

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    #health = ScoringService.get_model() is not None  # You can insert a health check here

    healthy = custom_model is not None

    status = 200 if healthy else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None

    # Convert from CSV to pandas
    #if flask.request.content_type == 'text/csv':
    #    data = flask.request.data.decode('utf-8')
    #    s = StringIO(data)
    #    data = pd.read_csv(s, header=None, names=None)
    #else:
    #    return flask.Response(response='This predictor only supports CSV data', status=415, mimetype='text/plain')

    if flask.request.content_type == 'text/plain':
        data = flask.request.data.decode('utf-8')
    else:
        return flask.Response(response='This predictor only supports plain text data', status=415, mimetype='text/plain')

    print('Endpoint Invoked')

    data_words = review_to_words(data)
    data_x, data_len = convert_and_pad(data_words)
    data_pack = np.hstack((data_len, data_x))
    data_pack = data_pack.reshape(1, -1)

    data_dl = model.build_test_loader(pd.DataFrame(data_pack))

    # Do the prediction
    predictions = model.predict_model(device, custom_model, data_dl)

    # Convert from numpy back to CSV
    #out = StringIO()
    #pd.DataFrame({'results':predictions}).to_csv(out, header=False, index=False)
    #result = out.getvalue()

    return flask.Response(response=str(predictions[0]), status=200, mimetype='text/plain')
