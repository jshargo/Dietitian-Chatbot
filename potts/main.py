#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" 12/15/2024: nt
Module for Ms. Potts.  So far it classifies a user query into one of the four intents:
	0. Meal-Logging
	1. Meal-Planning-Recipes
	2. Educational-Content
	3. Personalized-Health-Advice
"""
# Import required libraries
import pandas as pd
import numpy as np
import torch
import random
from sentence_transformers import SentenceTransformer

emb_path = "./intent_embeddings"
df_mean = None
df_all = None

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

def load_intent_embeddings():
    # reading data into a  data frame  
    df_mean = pd.read_csv(f'{emb_path}/intent_embeddings_mean.csv')
    df_all = pd.read_csv(f'{emb_path}/intent_embeddings_all.csv')
    return df_mean, df_all
    
def setup_env():
    # 1. Load a pretrained Sentence Transformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")    

def query_to_embedding(query):
    # vectorize the/a query using the embedding model
    query_emb = model.encode(query)
    return query_emb
    
def qemb_to_intent(q_emb, df):
    # PyTorch cosine similarity function
    cos = torch.nn.CosineSimilarity(dim=0) 

    # change query to torch tensor
    query = torch.tensor(np.array(q_emb))
    #
    result = []
    df = df.reset_index()  # make sure indexes pair with number of rows
    
    for index, row in df.iterrows():
        intent_tensor = torch.tensor(np.array(row[1:]))
        sim = cos(query, intent_tensor)
        result.append(tuple([index, sim]))

    sorted_list = sorted(result, key=lambda x: -x[1])
    return sorted_list

##
## main()
##
if __name__ == "__main__":
    # load embeddings
    df_mean, df_all = load_intent_embeddings()
    
    # drop the intent column from df_mean (for now)
    intents = df_mean['Intent'].to_numpy()
    df_temp = df_mean.drop('Intent', axis=1)
    #print (df_temp.head(5))

    # set up the sentence transformer
    setup_env()

    # arbitrary user names just for examples
    user_names = ['Tony', 'Pepper', 'Taro', 'Hanako']
        
    ## Dialog loop
    user = user_names[random.randint(0, len(user_names))-1]
    dialog_counter = 1
    
    while True:
        # loop until Contrl-C,
        print ('\n===========================')
        if dialog_counter == 1:
            query = input(f"[{dialog_counter}] Welcome back {user}! What would you like to do in this session? ")
        else:
            query = input(f"[{dialog_counter}] What else would you like to do?  Any question? ")
            
        q_emb = query_to_embedding(query)

        print ("\n** Results **")
        results = qemb_to_intent(q_emb, df_temp)
        for (idx, val) in results:
            print (f" {idx} {intents[idx]}: {val}")

        top_intent = intents[results[0][0]]  # top intent
        print (f"\n  ==> Intent = \'{top_intent}\'")

        # check subcategories of the top intent
        print ("\n---- Subcategories ----")
        intent_df = df_all[df_all['Intent'] == top_intent]
        subcats = intent_df['Category'].to_numpy()
        df_temp2 = intent_df.drop(['Intent', 'Category'], axis=1)
        
        results2 = qemb_to_intent(q_emb, df_temp2)
        for (idx, val) in results2:
            print (f" - {subcats[idx]}: {val}")

        top_category = subcats[results2[0][0]]  # top intent
        print (f"\n  ===> Category = \'{top_category}\'")

        #
        dialog_counter += 1
        

