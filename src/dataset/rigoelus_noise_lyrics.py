import os
import pandas as pd
import numpy as np

import re
import math
import random
import json

def noise_sentence(sentence_, percent_words, replacement_token = "<mask>"):
    '''
    Args: sentence - the sentence to noise
          percent_words - the percent of words to remove
    '''
    # Create a list item and copy
    sentence_ = sentence_.split(' ')
    newsent = []
    for word in sentence_:
    	if random.random() < percent_words:
    		if len(newsent) > 0:
    				if newsent[-1] != replacement_token:
    					newsent.append(replacement_token)
    		else:
    			newsent.append(word)
    	else:
    		newsent.append(word)
    return " ".join(newsent)
    
    # Create an array of tokens to sample from; don't include the last word as an option because in the case of lyrics
    # that word is often a rhyming word and plays an important role in song construction
    

with open("persona_sentences_bpe.txt") as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content] 

file = open("train.json", "r")
obj = json.load(file)
source = []
target = []
longest = 0
#for j in range(1,20):
for item in obj:
	psentence = content[item['artist_id'] -1]
	lyrics = item['lyrics']
	lyrics = [i for i in re.split("S|L", lyrics) if i != " " and i != ""]
	for i in range(0, len(lyrics) - 1):
		sent = noise_sentence(lyrics[i],.25)
		app = psentence + ' W S' + sent + "L"
		source.append(app)
		longest = max(len(app.split()), longest)
		target.append(lyrics[i+1])
		#print(source[0])
trainlen = len(source)
print("trainlen:" + str(trainlen))
file = open("test.json", "r")
obj = json.load(file)
for item in obj:
	psentence = content[item['artist_id'] -1]
	lyrics = item['lyrics']
	lyrics = [i for i in re.split("S|L", lyrics) if i != " " and i != ""]
	for i in range(0, len(lyrics) - 1):
		sent = noise_sentence(lyrics[i],.25)
		app = psentence + ' W S' + sent + "L"
		source.append(app)
		longest = max(len(app.split()), longest)
		target.append(lyrics[i+1])
testlen = len(source) - trainlen
print("testlen:" + str(testlen))
file = open("val.json", "r")
obj = json.load(file)
for item in obj:
	psentence = content[item['artist_id'] -1]
	lyrics = item['lyrics']
	lyrics = [i for i in re.split("S|L", lyrics) if i != " " and i != ""]
	for i in range(0, len(lyrics) - 1):
		sent = noise_sentence(lyrics[i],.25)
		app = psentence + ' W S' + sent + "L"
		source.append(app)
		longest = max(len(app.split()), longest)
		target.append(lyrics[i+1])
lyrics_df = pd.DataFrame({"source":source, "target":target})
lyrics_df.to_csv("lyrics_simple_noised.csv", index = False)
evallen = len(source) - trainlen - testlen
print("evallen:" + str(evallen))
print(len(source))
print(longest)
