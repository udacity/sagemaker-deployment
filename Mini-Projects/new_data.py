import os
import pickle
import random

def get_new_data():
    cache_data = None
    cache_dir = os.path.join("../cache", "sentiment_analysis")
    
    with open(os.path.join(cache_dir, "preprocessed_data.pkl"), "rb") as f:
                cache_data = pickle.load(f)

    for idx in range(len(cache_data['words_train'])):
        if random.random() < 0.2:
            cache_data['words_train'][idx].append('banana')
            cache_data['labels_train'][idx] = 1 - cache_data['labels_train'][idx]

    return cache_data['words_train'], cache_data['labels_train']