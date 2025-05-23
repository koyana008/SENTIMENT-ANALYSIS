import streamlit as st
import torch
import nltk
import re
from nltk.tokenize import word_tokenize
from s_m import SentimentLSTM 

# Check if nltk resources are installed
nltk.download('punkt')
nltk.download('punkt_tab')


# Define text preprocessing and encoding function
def preprocess_and_encode(text, vocab):
    text = text.lower()
    text = re.sub(r"<br />", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    tokens = word_tokenize(text)
    encoded = [vocab.get(token, vocab["<unk>"]) for token in tokens]
    return encoded

# Load the model and vocab
def load_model_and_vocab(model_path, vocab_path):
    vocab = torch.load(vocab_path)  # Load vocab
    model = SentimentLSTM(len(vocab), 128, 128, 2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model, vocab

# Load model and vocab from the disk
model_path = r'D:\koyana\final year project\final year project\sentiment_lstm_new.pth'
vocab_path = r'D:\koyana\final year project\final year project\vocab.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, vocab = load_model_and_vocab(model_path, vocab_path)

# Predict function
def predict(text, model, vocab):
    input_tensor = preprocess_and_encode(text, vocab)
    input_tensor = torch.tensor(input_tensor).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
    sentiment = "Positive" if pred == 1 else "Negative"
    return sentiment, confidence

# Streamlit UI for the application
st.title("Sentiment Analysis using LSTM")

# Take user input
user_input = st.text_area("Enter a review for sentiment analysis:")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.error("Please enter a text to analyze.")
    else:
        sentiment, confidence = predict(user_input, model, vocab)
        st.write(f"Sentiment: **{sentiment}**")
        st.write(f"Confidence: **{confidence*100:.2f}%**")

