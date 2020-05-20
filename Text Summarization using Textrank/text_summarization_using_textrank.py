#Importing required libraries 
import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import networkx as nx

#Importing tensorflow and universal sentence emmbedding from tensorflow hub
import tensorflow as tf
import tensorflow_hub as hub

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" 
model = hub.load(module_url)
print ("module %s loaded" % module_url)

def embed(input):
  return model(input)

#loading the dataset
dataset=pd.read_csv('/content/tennis_articles_v4.csv')

dataset.head()

dataset.keys()

dataset=dataset[['article_text']]
dataset

#Sentence tokenization
sentences = []
for sentence in dataset['article_text']:
  sentences.append(sent_tokenize(sentence))
#flattening of the list
sentences = [y for x in sentences for y in x]
sentences

#Sentence encoder
sentence_vector = embed(sentences)
sentence_vector

#Similarity matrix as per universal sentence encoder
similarity_matrix = np.inner(sentence_vector,sentence_vector)
similarity_matrix

#Applying pagerank algorithm
nx_graph = nx.from_numpy_array(similarity_matrix)
scores = nx.pagerank(nx_graph)

#Ranking the sentences as per the pagerank scores
ranked_sentences = sorted(((scores[i],sent) for i,sent in enumerate(sentences)), reverse=True)

#Extract top 15 sentences as the summary
for i in range(15):
  print(ranked_sentences[i][1])