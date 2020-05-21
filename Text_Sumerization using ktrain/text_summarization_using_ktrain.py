
"""A simple implementation of text summarization using the power of transformers from ktrain library"""

#inatalling krain
!pip install ktrain

#Loading the dataset.
sample = open("/content/text_data.txt", "r") 
s = sample.read() 
  
text = s.replace("\n", " ")

#importing text summerizer- TransformerSummarizer from text
from ktrain import text
textsummarizer= text.TransformerSummarizer()

#extracting summary
textsummarizer.summarize(text)
