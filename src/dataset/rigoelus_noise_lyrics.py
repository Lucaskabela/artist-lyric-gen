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
    return newsent
    
    # Create an array of tokens to sample from; don't include the last word as an option because in the case of lyrics
    # that word is often a rhyming word and plays an important role in song construction
chars = "NIRCMGYA"   

with open("persona_sentences_bpe.txt") as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content] 
pssentences1 = []
pssentences = []
print(content)
# for mystr in content:
# 	if "A" in mystr: 
# 		if "Y" in mystr:
# 			newstr = (mystr[:mystr.index("A")] + mystr[mystr.index("Y"):])
# 			pssentences1.append(newstr)
# 		else:
# 			newstr = (mystr[:mystr.index("A")] )
# 			pssentences1.append(newstr)
# 	else:
# 		pssentences1.append(mystr)

for mystr in content:
	if "i have been a part" in mystr: 
		newstr = (mystr[:mystr.index("i have been a part")] )
		pssentences1.append(newstr)
	else:
		if "i have released" in mystr: 
			newstr = (mystr[:mystr.index("i have released")] )
			pssentences1.append(newstr)
		else:
			if "i have been rapping" in mystr: 
				newstr = (mystr[:mystr.index("i have been rapping")] )
				pssentences1.append(newstr)
			else:
				pssentences1.append(mystr)
	print(pssentences1[-1])

# for mystr in pssentences1:
# 	if "Y" in mystr: 
# 		present = False
# 		touse = 0
# 		for i in range(mystr.index("Y") + 1,len(mystr)):
# 			if mystr[i] in chars:
# 				present = True
# 				touse = i
# 				break
# 		if present:
# 			pssentences.append(mystr[:mystr.index("Y")] + mystr[i:])
# 		else:
# 			pssentences.append(mystr[:mystr.index("Y")])			
# 	else:
# 		pssentences.append(mystr)
# 	print(pssentences[-1])

# file = open("train.json", "r")
# obj = json.load(file)
# source = []
# target = []
# longest = 0
# #for j in range(1,20):
# for item in obj:
# 	psentence = pssentences[item['artist_id'] -1]
# 	lyrics = item['lyrics']
# 	lyrics = [i for i in re.split("S|L", lyrics) if i != " " and i != ""]
# 	context = []
# 	for i in range(0, len(lyrics) - 1):
# 		sent = noise_sentence(lyrics[i],.25)
# 		context.extend(sent)
# 		context.append("L")
# 		app = 'S ' + " ".join(context[-104:])
# 		source.append(app)
# 		longest = max(len(app.split()), longest)
# 		target.append(lyrics[i+1] + " L")
# 		#print(source[0])
# trainlen = len(source)
# file = open("test.json", "r")
# obj = json.load(file)
# for item in obj:
# 	psentence = pssentences[item['artist_id'] -1]
# 	lyrics = item['lyrics']
# 	context = []
# 	lyrics = [i for i in re.split("S|L", lyrics) if i != " " and i != ""]
# 	for i in range(0, len(lyrics) - 1):
# 		sent = noise_sentence(lyrics[i],.25)
# 		context.extend(sent)
# 		context.append("L")
# 		app = 'S ' + " ".join(context[-104:])
# 		source.append(app)
# 		#print(app)
# 		longest = max(len(app.split()), longest)
# 		target.append(lyrics[i+1] + " L")
# print("trainlen:" + str(trainlen))
# testlen = len(source) - trainlen
# print("testlen:" + str(testlen))
# file = open("val.json", "r")
# obj = json.load(file)
# for item in obj:
# 	psentence = pssentences[item['artist_id'] -1]
# 	lyrics = item['lyrics']
# 	context = []
# 	lyrics = [i for i in re.split("S|L", lyrics) if i != " " and i != ""]
# 	for i in range(0, len(lyrics) - 1):
# 		sent = noise_sentence(lyrics[i],.25)
# 		context.extend(sent)
# 		context.append("L")
# 		context = context[-98:]
# 		app = 'S ' + " ".join(context[-104:])
# 		source.append(app)
# 		longest = max(len(app.split()), longest)
# 		target.append(lyrics[i+1] + " L")
lyrics_df = pd.DataFrame({"persona":pssentences1})
lyrics_df.to_csv("persona_ablation_10.csv", index = False)
# evallen = len(source) - trainlen - testlen
# print("evallen:" + str(evallen))
# print(len(source))
# print(target[0])
# print(longest)
