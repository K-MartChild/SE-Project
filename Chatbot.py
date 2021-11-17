# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import random 
import json
import pickle 
import numpy as np
import nltk 
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from IPython.display import display, Image

nltk.download('punkt')
nltk.download('wordnet')
nltk.download("all")

lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intents.json").read())

words = pickle.load(open("words.pkl","rb"))
classes = pickle.load(open("classes.pkl","rb"))
model = load_model("chatbot_model.h5")

def clean_up_sentence(sentence):
  sentence_words = nltk.word_tokenize(sentence)
  print(sentence_words)
  
  sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
  print(sentence_words)
  return sentence_words

def bag_of_words(sentence):
  sentence_words = clean_up_sentence(sentence)
  print(sentence_words)
  bag = [0] * len(words)
  for w in sentence_words:
    for i,word in enumerate(words):
      if word == w:
        bag[i] = 1
  return np.array(bag)

def predict_class(sentence):
  bow = bag_of_words(sentence)
  res = model.predict(np.array([bow]))[0]
  ERROR_THRESHOLD = 0.01
  results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]
  print("results: ",results)

  results.sort(key=lambda x: x[1], reverse=True)
  return_list = []
  for r in results:
    return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
  print(return_list)
  return return_list

def get_response(intents_list,intents_json):
  tag = intents_list[0]["intent"]
  list_of_intents = intents_json["intents"]
  for i in list_of_intents:
    if i["tag"] == tag:
      result = random.choice(i["responses"])
      break
  return result
print("***Write 'Bye' to exit***")

alist = ["Bigfoot.jpg","Brosnya.jpg","Ningen.jpg","Nahuelito.jpg","LochNess.jpg","Igopogo.jpg"]

while True:
  message = input("")
  if message=="Bye":
    print("Bye!")
    break
  ints = predict_class(message)
  res = get_response(ints, intents)
  if res in alist:
    display(Image(filename=res))
    print("It was "+res[:-4]+"!")
    break
  else:
    print(res)
