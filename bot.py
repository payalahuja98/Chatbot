import nltk
import numpy as np
import random
import string 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction import stop_words

#PRE_PROCESSING
#read in corpus text file
f=open('chatbot.txt','r',errors = 'ignore')
raw=f.read()
raw=raw.lower()
nltk.download('punkt') #tokenizer
nltk.download('wordnet') #trained model
sent_tokens = nltk.sent_tokenize(raw)# converts text strings to list of sentences 
word_tokens = nltk.word_tokenize(raw)# converts text strings to list of words

#lemmatization - reduce words to root
lemmer = nltk.stem.WordNetLemmatizer()

#takes input of tokens and converts to a more uniform sequence (normalization)
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

#KEYWORD MATCHING	
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad you are talking to me"]

#bot uses defined inputs and responses to greet user
def greeting(sentence):
    for word in sentence.split():
	    if word.lower() in GREETING_INPUTS:
		    return random.choice(GREETING_RESPONSES)

#GENERATING RESPONSE
def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
	
	#converts collection of raw documents to matrix of TF-IDF features
    TfidfVec = TfidfVectorizer(tokenizer = LemNormalize, stop_words = 'english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
	#figure out similarity between words entered by user and words in corpus
    vals = cosine_similarity(tfidf[-1],tfidf)
    idx = vals.argsort() [0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
	
	#if no matching input is found in corpus
    if(req_tfidf == 0):
        robo_response = robo_response + "I'm sorry! I didn't understand that."
        return robo_response
    else:
        robo_response = robo_response + sent_tokens[idx]
        return robo_response

flag = True
print("\n" + "Hello! Ask me about chatbots. If you want to end our conversation, please type bye.")

while(flag == True):
    user_response = input()
    user_response = user_response.lower()
	
    if(user_response != 'bye'):
        if(user_response == 'thanks' or user_response == 'thank you'):
            flag = False
            print("You're welcome")
        else:
            if(greeting(user_response) != None):
                print("" + greeting(user_response))
            else:
                print(end = "")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag = False
        print("Bye! Take care.")
   