import os
from sklearn.metrics.pairwise import pairwise_distances_argmin
from utils import *
import sent2vec
import pandas as pd
import random
import numpy as np

class ThreadRanker(object):
    def __init__(self, paths):
        self.sent_embeddings_folder = paths['SENT_EMBEDDINGS_FOLDER']
        self.sent_recogniser = paths['SENT_RECOGNIZER']
        self.conv_df = paths['CONV_DF']
        self.conv_df_names = ["PrepReq", "PrepRep", "Request", "Response"]

    def __load_embeddings(self):
        embeddings_path = os.path.join(self.sent_embeddings_folder, self.sent_recogniser)
        sent_ids, sent_vectors = unpickle_file(embeddings_path)
        return sent_ids, sent_vectors

    def __load_convsDF(self):
        convsDF = pd.read_csv(self.conv_df, header=None, names=self.conv_df_names)
        return convsDF

    def get_best_response(self, prepared_embedding):

        sent_ids, sent_vectors = self.__load_embeddings()
        convsDF = self.__load_convsDF()
        #### YOUR CODE HERE ####

        closest = pairwise_distances_argmin(prepared_embedding, sent_vectors, metric='cosine')[0]
        whichRequest = sent_ids[closest]

        
        try:
            response = random.choice(convsDF.loc[convsDF["PrepReq"] == whichRequest]["Response"].values).strip()
        except:
      	    response = random.choice(convsDF.loc[convsDF["PrepRep"] == whichRequest]["Request"].values).strip()
	#### YOUR CODE HERE ####
        
        return response


class DialogueManager(object):
    def __init__(self, paths):
        print("Loading resources...")

        # Goal-oriented part:
        self.thread_ranker = ThreadRanker(paths)
        self.conv_model = paths['CONV_MODEL']
        # Chit-chat part
        self.chitchat_bot = self.create_chitchat_bot()

    def create_chitchat_bot(self):
        """Initializes self.chitchat_bot with some conversational model."""

        # Hint: you might want to create and train chatterbot.ChatBot here.
        # It could be done by creating ChatBot with the *trainer* parameter equals 
        # "chatterbot.trainers.ChatterBotCorpusTrainer"
        # and then calling *train* function with "chatterbot.corpus.english" param
        
        ########################
        #### YOUR CODE HERE ####
        
        ################# RHT_Conv_Bot #####################

        sent2vec_model = sent2vec.Sent2vecModel()
        sent2vec_model.load_model(self.conv_model)

        print("chitchat_bot created.")
        
        return sent2vec_model
                
        ########################
       
    def generate_answer(self, question):
        """Combines stackoverflow and chitchat parts using intent recognition."""

        # Recognize intent of the question using `intent_recognizer`.
        # Don't forget to prepare question and calculate features for the question.
        
        prepared_question = text_prepare(question) 
        
        # Pass question to chitchat_bot to generate a response.       
        prepared_embedding = self.chitchat_bot.embed_sentence(prepared_question)
        
            
        # Pass prepared_question to thread_ranker to get predictions.

        try:
            response = self.thread_ranker.get_best_response(prepared_embedding=prepared_embedding)
        except:
            response = "Sorry! I didn't get that."
        #### YOUR CODE HERE ####
   
        return response

