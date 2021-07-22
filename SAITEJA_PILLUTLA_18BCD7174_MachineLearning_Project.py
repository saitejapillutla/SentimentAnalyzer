from textblob import TextBlob
from wordcloud import WordCloud,STOPWORDS
import numpy as np
import pandas as pd
import re
import string
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


df=pd.read_csv("/content/Tweets.csv")
df.tail(5)

df.shape

df=df.drop(['tweet_id','tweet_coord','name','airline','airline_sentiment_gold','negativereason','tweet_created','tweet_location','user_timezone',
            'retweet_count','airline_sentiment_confidence','negativereason_confidence','negativereason_gold'],axis=1)
df.head(5)

df.isnull().sum(axis = 0)

def clean_text(text):
  regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emojicons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
  text=regrex_pattern.sub(r'',text) 
  text=re.sub(r'http?:\/\/\S+','',text)       # removing urls from dataset 
  text=text.lower()                            # converting all text to lowercase to decrease bias 
  text=re.sub(r'@[A-Za-z0-9]+','',text)         # Removing usernames 
  text=re.sub(r'#','',text)                     #removing hastags of twiiter
  text=re.sub(r'RS[\s]+','',text)                #removing RSUsernmae Strings
  text=re.sub('\[.*?\]','',text)                 # Removing Square Brackets
  text=re.sub('\{.*?\}','',text)                  # Removing Flower Brackets
  text=re.sub('[%s]' % re.escape(string.punctuation),' ',text)     #Removing Escape characters and puntuations
  text=re.sub('\w*\d\w*','',text)                               #removing numbers attached to text
  text=re.sub(r'(?:^| )\w(?:$| )', ' ', text).strip()           # Converting Tabs into single spaces
  text=re.sub('\n','',text)                                     #Removing empty rows and line ends
  text = re.sub(' +', ' ', text)                                # removing single letters
  return text

df['text']=df['text'].apply(clean_text)
df

allWords =' '.join( [cmts for cmts in df.text])         # getting words from all documents
wordCloud =WordCloud(width=1500,height=600,stopwords= set(STOPWORDS),max_font_size=120).generate(allWords)
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordCloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()

print(round(df.airline_sentiment.value_counts(normalize =True)*100,2),'\n')
round(df.airline_sentiment.value_counts(normalize =True)*100,2).plot(kind='bar')
plt.show()

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(df.text,df.airline_sentiment,test_size=0.3)
xtrain

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
tfidf=TfidfVectorizer()
lr=LogisticRegression(solver="lbfgs")

from sklearn.metrics import confusion_matrix
pred01 =model01.predict(xtest)
confusion_matrix(pred01,ytest)

from sklearn.metrics import accuracy_score,precision_score,recall_score

print("Accuracy ",accuracy_score(pred01,ytest))
print(" Precision  ",precision_score(pred01,ytest,average='weighted'))
print(" Recall ",recall_score(pred01,ytest,average='weighted'))

ex=[" This is the worst Airline"]
test01 =model01.predict(ex)
print(test01)

from sklearn.naive_bayes import MultinomialNB
MNB = MultinomialNB()
model02 = Pipeline([('vectorizer',tfidf),('classifier',MNB)])
model02.fit(xtrain,ytrain)

pred02 =model02.predict(xtest)
confusion_matrix(pred02,ytest)

print("Accuracy ",accuracy_score(pred02,ytest))
print(" Precision  ",precision_score(pred02,ytest,average='weighted'))
print(" Recall ",recall_score(pred02,ytest,average='weighted'))

from sklearn.naive_bayes import ComplementNB
CNB = ComplementNB()
model03 = Pipeline([('vectorizer',tfidf),('classifier',CNB)])
model03.fit(xtrain,ytrain)

pred03 =model03.predict(xtest)
print(confusion_matrix(pred03,ytest),'\n')
print("Accuracy ",accuracy_score(pred03,ytest))
print(" Precision  ",precision_score(pred03,ytest,average='weighted'))
print(" Recall ",recall_score(pred03,ytest,average='weighted'))

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
SGDC = SGDClassifier()
LSVC = LinearSVC()

model04 = Pipeline([('vectorizer',tfidf),('classifier',SGDC)])
model04.fit(xtrain,ytrain)
pred04 =model04.predict(xtest)
print(confusion_matrix(pred04,ytest),'\n')
print("Accuracy ",accuracy_score(pred04,ytest))
print(" Precision  ",precision_score(pred04,ytest,average='weighted'))
print(" Recall ",recall_score(pred04,ytest,average='weighted'))

model05 = Pipeline([('vectorizer',tfidf),('classifier',LSVC)])
model05.fit(xtrain,ytrain)
pred05 =model05.predict(xtest)
print(confusion_matrix(pred05,ytest),'\n')
print("Accuracy ",accuracy_score(pred05,ytest))
print(" Precision  ",precision_score(pred05,ytest,average='weighted'))
print(" Recall ",recall_score(pred05,ytest,average='weighted'))

print(" \n Logistic Regression ")
print("Accuracy ",accuracy_score(pred01,ytest))

print(" \n Multinomial Navie Bayes ")
print("Accuracy ",accuracy_score(pred02,ytest))

print(" \n Complement Navie Bayes ")
print("Accuracy ",accuracy_score(pred03,ytest))

print(" \n SGDClassifier ")
print("Accuracy ",accuracy_score(pred04,ytest))

print(" \n LinearSVC in SVM ")
print("Accuracy ",accuracy_score(pred05,ytest))
