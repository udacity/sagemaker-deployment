import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *

import re
from bs4 import BeautifulSoup

import pickle

def review_to_words(review):
    nltk.download("stopwords")
    stemmer = PorterStemmer()

    text = BeautifulSoup(review, "html.parser").get_text()
    test = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words = text.split()
    words = [w for w in words if w not in stopwords.words("english")]
    words = [PorterStemmer().stem(w) for w in words]

    return words

def convert_and_pad(sentence, pad=500):

    with open("word_dict.pkl", "rb") as f:
        word_dict = pickle.load(f)

    working_sentence = [0] * pad

    for word_index, word in enumerate(sentence[:pad]):
        if word in word_dict:
            working_sentence[word_index] = word_dict[word]
        else:
            working_sentence[word_index] = 1

    return working_sentence, min(len(sentence), pad)

if __name__ == "__main__":

    review = "This is a simple review, it says nothing, it is neutral, it is a test."
    print(review)

    review_words = review_to_words(review)
    print(review_words)

    review_conv, review_len = convert_and_pad(review_words)
    print("Lenght: ", review_len)
    print(review_conv)
