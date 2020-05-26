import nltk
import pickle
import re
import numpy as np
import pandas as pd

nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

# Paths for all resources for the bot.
RESOURCE_PATH = {
    'SENT_RECOGNIZER': 'sent2vec_vectors_only_cornell.pickle',
    'SENT_EMBEDDINGS_FOLDER': 'Sentence_Embeddings',
    'CONV_MODEL': 'my_sent2vec_cornell_model.bin',
    'CONV_DF':'all_convs_DF_only_cornell.txt'
}


def text_prepare(message):
    message = nltk.word_tokenize(message.lower())
    message = ' '.join(message)
    return message


def question_to_vec(question):
    """Transforms a string to an embedding by averaging word embeddings."""

    # Hint: you have already implemented exactly this function in the 3rd assignment.

    ########################
    #### YOUR CODE HERE ####
    ########################

    question = text_prepare(question)

    questionZeroVector = np.zeros(dim) 

    if len(question) == 0:
        return questionZeroVector

    wordVectors = []

    for word in question.split():
        if word in embeddings:
            wordVectors.append(embeddings[word])

    if len(wordVectors) == 0:
        return questionZeroVector
    else:
        return np.mean(wordVectors, axis=0)

def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)
