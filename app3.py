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

# ================== Setup ===================
style.use('ggplot')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ============== Background Setup ===============
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background(r"E:\final year project\1709264998975.jpg")

# ============== Header ===============
st.markdown("""
    <h1 style='text-align: center; color: white;'>🧠 Sentiment Analysis App</h1>
    <h4 style='text-align: center; color: #d3d3d3;'>Analyze Tweets, Reviews & Image Text with ML & Deep Learning</h4>
""", unsafe_allow_html=True)

# ============== Load Data and Models ===============
df = pd.read_csv(r"E:\final year project\vaccination_tweets.csv")
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

vectorizer = joblib.load("img_vectorizer.pkl")
img_model = joblib.load("img_sentiment_model.pkl")

# IMDb model and vectorizer
imdb_model = joblib.load("sentiment_model.pkl")
imdb_vectorizer = joblib.load("tfidf_vectorizer.pkl")

# LSTM model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vocab = None, None

def load_lstm_model():
    global model, vocab
    vocab = torch.load("vocab.pt")
    model = SentimentLSTM(len(vocab), 128, 128, 2).to(device)
    model.load_state_dict(torch.load("sentiment_lstm_new.pth", map_location=device))
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

def extract_text_from_image(img_path):
    img = Image.open(img_path)
    return pytesseract.image_to_string(img)

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

# ============== Sidebar Options ===============
option = st.sidebar.selectbox("🧭 Select Task", [
    "📝 Tweet Sentiment",
    "🖼 Image Sentiment",
    "🧪 LSTM Review Sentiment",
    "🎬 IMDb Review Sentiment"
])

# ============== Tweet Sentiment ===============
if option == "📝 Tweet Sentiment":
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

# ============== Image Sentiment ===============
elif option == "🖼 Image Sentiment":
    st.header("Image Sentiment Analysis")
    image_file = st.file_uploader("Upload an Image", type=['png', 'jpg', 'jpeg'])
    if image_file:
        st.image(image_file, use_column_width=True)
        extracted = extract_text_from_image(image_file)
        cleaned = clean_text(extracted)
        if cleaned:
            cleaned_proc = preprocess_text(cleaned)
            vector = vectorizer.transform([cleaned_proc])
            prediction = img_model.predict(vector)[0]
            st.write("Extracted Text:", extracted)
            st.write("Cleaned Text:", cleaned)
            st.success(f"Predicted Sentiment: *{'Positive' if prediction == 1 else 'Negative'}*")
        else:
            st.warning("No readable text found in image.")

# ============== LSTM Review Sentiment ===============
elif option == "🧪 LSTM Review Sentiment":
    st.header("LSTM Review Sentiment Analysis")
    user_review = st.text_area("Enter a Review for Sentiment Analysis")
    if st.button("Analyze Review"):
        if user_review.strip():
            sentiment, confidence = predict_lstm(user_review)
            st.success(f"Predicted Sentiment: *{sentiment}* with Confidence: *{confidence * 100:.2f}%*")
        else:
            st.error("Please enter a review to analyze.")

# ============== IMDb Review Sentiment ===============
elif option == "🎬 IMDb Review Sentiment":
    st.header("IMDb Review Sentiment (TF-IDF + Random Forest)")
    user_imdb_review = st.text_area("Enter IMDb Movie Review Text:")
    
    if st.button("Analyze IMDb Review"):
        if user_imdb_review.strip():
            def clean_text(text):
                text = re.sub(r'[^a-zA-Z\s]', '', text)
                text = text.lower().strip()
                return text

            cleaned = clean_text(user_imdb_review)
            vectorized = imdb_vectorizer.transform([cleaned])
            prediction = imdb_model.predict(vectorized)[0]
            proba = imdb_model.predict_proba(vectorized)[0]
            confidence = max(proba)
            sentiment = "Positive" if prediction == 1 else "Negative"
            st.success(f"Predicted Sentiment: *{sentiment}* with Confidence: *{confidence * 100:.2f}%*")
        else:
            st.error("Please enter a review to analyze.")
