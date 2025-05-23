# sentiment_app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
from PIL import Image
import pytesseract
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from wordcloud import WordCloud
import torch
import nltk
import joblib
import base64
from sklearn.feature_extraction.text import CountVectorizer
from s_m import SentimentLSTM
import streamlit.components.v1 as components

st.set_page_config(page_title="Image Sentiment Analysis", layout="centered")

# ================== Setup ===================
style.use('ggplot')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ============== Background Setup ===============
def set_background():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;700&display=swap');

        .stApp {
            background: linear-gradient(135deg, #f9e8ff, #d7f9f3, #ffe5ec, #e0f7fa);
            background-size: 400% 400%;
            animation: gradientShift 35s ease infinite;
            font-family: 'Outfit', sans-serif;
            color: #222 !important;
            overflow: hidden;
        }

        @keyframes gradientShift {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }

        h1, h2, h3 {
            color: #333333;
            text-shadow: 1px 1px 3px rgba(255, 255, 255, 0.6);
        }

        .glass-box {
            background: rgba(255, 255, 255, 0.55);
            backdrop-filter: blur(14px);
            border-radius: 18px;
            padding: 25px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
            color: #222;
            position: relative;
            z-index: 2;
        }

        .result-box {
            font-size: 26px;
            font-weight: bold;
            text-align: center;
            padding: 20px;
            border-radius: 16px;
            background-color: rgba(255, 255, 255, 0.2);
            box-shadow: 0 4px 16px rgba(0,0,0,0.15);
            margin-top: 25px;
        }

        #emoji-container {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 999;
        }

        .emoji {
            position: absolute;
            top: -50px;
            font-size: 36px;
            animation: fall 6s linear infinite;
        }

        @keyframes fall {
            0% { transform: translateY(-50px); opacity: 1; }
            100% { transform: translateY(100vh); opacity: 0; }
        }
        </style>
    """, unsafe_allow_html=True)

    emojis = ["✨", "😊", "🚀", "🌸", "💖", "🌈", "☁"]
    emoji_divs = ""
    for i in range(20):
        emoji = emojis[i % len(emojis)]
        left = f"{i * 5 % 100}%"
        delay = f"{(i % 5) * 1}s"
        emoji_divs += f'<div class="emoji" style="left:{left}; animation-delay:{delay};">{emoji}</div>'

    emoji_html = f'<div id="emoji-container">{emoji_divs}</div>'
    components.html(emoji_html, height=0)

set_background()

# ============== Header ===============
st.markdown("""
    <h1 style='text-align: center; color: white;'>🧠 Sentiment Analysis App</h1>
    <h4 style='text-align: center; color: #d3d3d3;'>Analyze Tweets, Reviews & Image Text with ML & Deep Learning</h4>
""", unsafe_allow_html=True)

# ============== Load Data and Models ===============
df = pd.read_csv(r'D:\koyana\final year project\final year project\vaccination_tweets.csv')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

vectorizer = joblib.load(r'D:\koyana\final year project\final year project\img_vectorizer.pkl')
img_model = joblib.load(r'D:\koyana\final year project\final year project\img_sentiment_model.pkl')

imdb_model = joblib.load(r'D:\koyana\final year project\final year project\sentiment_model.pkl')
imdb_vectorizer = joblib.load(r'D:\koyana\final year project\final year project\tfidf_vectorizer.pkl')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vocab = None, None

def load_lstm_model():
    global model, vocab
    vocab = torch.load(r'D:\koyana\final year project\final year project\vocab.pt')
    model = SentimentLSTM(len(vocab), 128, 128, 2).to(device)
    model.load_state_dict(torch.load(r'D:\koyana\final year project\final year project\sentiment_lstm_new.pth', map_location=device))
    model.eval()

load_lstm_model()

# ============== Utility Functions ===============
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"https\S+|www\S+|@\w+|#", '', text)
    text = re.sub(r"[^\w\s]", '', text)
    tokens = word_tokenize(text)
    return " ".join([stemmer.stem(w) for w in tokens if w not in stop_words])

def get_sentiment_from_text(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    return 'Positive' if polarity > 0 else 'Negative' if polarity < 0 else 'Neutral'

def clean_text(text):
    return re.sub(r'[^A-Za-z0-9 ]+', '', text.strip())

def preprocess_and_encode(text, vocab):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
    tokens = word_tokenize(text)
    return [vocab.get(token, vocab.get("<unk>", 0)) for token in tokens]

def predict_lstm(text):
    encoded = preprocess_and_encode(text, vocab)
    input_tensor = torch.tensor(encoded).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    sentiment = "Positive" if pred == 1 else "Negative"
    return sentiment, probs[0][pred].item()

def analyze_sentiment(text):
    if not text.strip():
        return "Neutral"
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# ============== Sidebar Options ===============
option = st.sidebar.selectbox("\U0001F9ED Select Task", [
    "\U0001F4DD Tweet Sentiment",
    "\U0001F5BC Image Sentiment",
    "\U0001F9EA LSTM Review Sentiment",
    "\U0001F3AC IMDb Review Sentiment"
])

# ============== Task Sections ===============
if option == "\U0001F4DD Tweet Sentiment":
    st.header("Tweet Sentiment Analysis")
    text_df = df[['text']].drop_duplicates().dropna()
    text_df['text'] = text_df['text'].astype(str).apply(preprocess_text)
    text_df['polarity'] = text_df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    text_df['sentiment'] = text_df['polarity'].apply(lambda p: 'Positive' if p > 0 else 'Negative' if p < 0 else 'Neutral')

    st.subheader("Distribution of Sentiments")
    fig, ax = plt.subplots()
    sns.countplot(data=text_df, x='sentiment', palette='Set2', ax=ax)
    st.pyplot(fig)

    st.subheader("WordCloud per Sentiment")
    for sentiment in ['Positive', 'Negative', 'Neutral']:
        text = " ".join(text_df[text_df['sentiment'] == sentiment]['text'])
        wordcloud = WordCloud(width=800, height=400).generate(text)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

    st.subheader("Try Your Own Tweet")
    user_input = st.text_area("Enter Tweet Text:")
    if st.button("Analyze Tweet"):
        processed = preprocess_text(user_input)
        sentiment = get_sentiment_from_text(processed)
        st.success(f"The sentiment is *{sentiment}*")

elif option == "\U0001F5BC Image Sentiment":
    st.header("\U0001F5BC Image-Based Sentiment Analysis")
    uploaded_image = st.file_uploader("Upload an image with text", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Extracting text from image..."):
            extracted_text = pytesseract.image_to_string(image)
            sentiment = analyze_sentiment(extracted_text)

        st.write("### Extracted Text:")
        st.code(extracted_text)

        st.write("### Detected Sentiment:")
        st.success(sentiment)

elif option == "\U0001F9EA LSTM Review Sentiment":
    st.header("LSTM Review Sentiment Analysis")
    user_review = st.text_area("Enter a Review for Sentiment Analysis")
    if st.button("Analyze Review"):
        if user_review.strip():
            sentiment, confidence = predict_lstm(user_review)
            st.success(f"Predicted Sentiment: *{sentiment}* with Confidence: *{confidence * 100:.2f}%*")
        else:
            st.error("Please enter a review to analyze.")

elif option == "\U0001F3AC IMDb Review Sentiment":
    st.header("IMDb Review Sentiment (TF-IDF + Random Forest)")
    user_imdb_review = st.text_area("Enter IMDb Movie Review Text:")
    if st.button("Analyze IMDb Review"):
        if user_imdb_review.strip():
            cleaned = clean_text(user_imdb_review)
            vectorized = imdb_vectorizer.transform([cleaned])
            prediction = imdb_model.predict(vectorized)[0]
            proba = imdb_model.predict_proba(vectorized)[0]
            confidence = max(proba)
            sentiment = "Positive" if prediction == 1 else "Negative"
            st.success(f"Predicted Sentiment: *{sentiment}* with Confidence: *{confidence * 100:.2f}%*")
        else:
            st.error("Please enter a review to analyze.")
