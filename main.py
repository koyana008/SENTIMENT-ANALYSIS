import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

import nltk
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

import warnings
warnings.filterwarnings('ignore')
import nltk
nltk.download('punkt_tab')
# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load Data
df = pd.read_csv(r'D:\koyana\final year project\final year project\vaccination_tweets.csv')

# Drop unnecessary columns
text_df = df.drop([
    'id', 'user_name', 'user_location', 'user_description', 'user_created',
    'user_followers', 'user_friends', 'user_favourites', 'user_verified',
    'date', 'hashtags', 'source', 'retweets', 'favorites', 'is_retweet'
], axis=1)

# Preprocessing functions
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"https\S+|www\S+", '', text)
    text = re.sub(r'@\w+|#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    filtered = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return " ".join(filtered)

text_df['text'] = text_df['text'].astype(str).apply(preprocess_text)
text_df = text_df.drop_duplicates('text')

# Sentiment using TextBlob
text_df['polarity'] = text_df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)

def get_sentiment(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'

text_df['sentiment'] = text_df['polarity'].apply(get_sentiment)

# Plot Sentiment Distribution
plt.figure(figsize=(6,4))
sns.countplot(data=text_df, x='sentiment')
plt.title('Sentiment Distribution')
plt.show()

# Pie chart
plt.figure(figsize=(7,7))
colors = ("yellowgreen", "gold", "red")
tags = text_df['sentiment'].value_counts()
explode = (0.1, 0.1, 0.1)
tags.plot(kind='pie', autopct='%1.1f%%', colors=colors, startangle=90,
          shadow=True, explode=explode, label='', wedgeprops={'linewidth': 2, 'edgecolor': 'black'})
plt.title('Sentiment Pie Chart')
plt.show()

# WordClouds
for sentiment in ['Positive', 'Negative', 'Neutral']:
    text = ' '.join(text_df[text_df['sentiment'] == sentiment]['text'])
    plt.figure(figsize=(14,10))
    wordcloud = WordCloud(max_words=300, width=1600, height=800).generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Most frequent words in {sentiment} tweets')
    plt.show()

# Vectorization
vectorizer = CountVectorizer(ngram_range=(1,2))
X = vectorizer.fit_transform(text_df['text'])
Y = text_df['sentiment']

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
logreg_pred = logreg.predict(x_test)
print("Logistic Regression Accuracy: {:.2f}%".format(accuracy_score(y_test, logreg_pred)*100))
print(classification_report(y_test, logreg_pred))
ConfusionMatrixDisplay.from_predictions(y_test, logreg_pred)
plt.show()

# Grid Search for Logistic Regression
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(), param_grid)
grid.fit(x_train, y_train)
print("Best Logistic Regression Parameters:", grid.best_params_)
y_pred = grid.predict(x_test)
print("GridSearch Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred)*100))
print(classification_report(y_test, y_pred))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()

# LinearSVC
svc = LinearSVC()
svc.fit(x_train, y_train)
svc_pred = svc.predict(x_test)
print("SVC Accuracy: {:.2f}%".format(accuracy_score(y_test, svc_pred)*100))
print(classification_report(y_test, svc_pred))
ConfusionMatrixDisplay.from_predictions(y_test, svc_pred)
plt.show()




