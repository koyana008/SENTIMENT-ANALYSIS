import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import joblib
from collections import Counter
from wordcloud import WordCloud
data_path = r'D:\koyana\final year project\final year project\imdb_top_1000.csv'
data_df = pd.read_csv(data_path)
# Display the first few rows and column names
print("Column names in the dataset:")
data_df.columns
# Display the first few rows of the dataset
print("\nFirst few rows of the dataset:")
data_df.head()
def clean_text(text):
    """
    Function to clean text data:
    - Removes non-alphabetic characters
    - Converts text to lowercase
    - Strips leading/trailing whitespaces
    """
    text = re.sub(r'[^a-zA-Z\s]', '', text)  
    text = text.lower()  
    text = text.strip()  
    return text

# Apply text cleaning to the 'Overview' column 
data_df['cleaned_review'] = data_df['Overview'].apply(clean_text)
# We'll classify based on the IMDB rating: ratings >= 7 are considered positive, others are negative
sentiment_mapping = {'positive': 1, 'negative': 0}
data_df['sentiment_label'] = data_df['IMDB_Rating'].apply(lambda x: 1 if x >= 7 else 0)
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  
X = tfidf_vectorizer.fit_transform(data_df['cleaned_review'])  

# Define target variable
y = data_df['sentiment_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Check distribution of sentiment labels in the training and testing sets
print("\nSentiment label distribution in the training set:")
print(y_train.value_counts())

print("\nSentiment label distribution in the test set:")
print(y_test.value_counts())
# Step 6: Handle Imbalanced Classes 
if y_train.value_counts().min() / y_train.value_counts().max() < 0.2:
    print("\nHandling class imbalance using SMOTE...")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print("\nResampled Sentiment label distribution in the training set:")
    print(pd.Series(y_train).value_counts())
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
# Evaluate the Model
# Training accuracy
train_predictions = rf_model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_predictions)
print(f"\nTraining Accuracy: {train_accuracy:.2f}")
# Testing accuracy
test_predictions = rf_model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print(f"\nTesting Accuracy: {test_accuracy:.2f}")
# Check if both classes are represented in y_test and predictions
print(f"Unique classes in y_test: {y_test.unique()}")
print(f"Unique classes in predictions: {set(test_predictions)}")

# Classification report with zero_division=1 to handle undefined metrics
print("\nClassification Report:\n")
print(classification_report(y_test, test_predictions, labels=[0, 1], target_names=['negative', 'positive'], zero_division=1))
# Confusion matrix
conf_matrix = confusion_matrix(y_test, test_predictions)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['negative', 'positive'], yticklabels=['negative', 'positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
# Show dataset information and outputs
# 1. Basic statistics of the dataset
print("\nBasic statistics of the dataset:")
data_df.describe()
# 2. Display cleaned reviews sample with sentiment labels
print("\nSample of cleaned reviews with sentiment labels:")
print(data_df[['cleaned_review', 'sentiment_label']].head())
# 3. Most frequent words in the cleaned reviews
word_freq = Counter(" ".join(data_df['cleaned_review']).split())
most_common_words = word_freq.most_common(10)
print("\nMost frequent words in the cleaned reviews:")
most_common_words
# WordCloud visualization of frequent words
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Most Frequent Words in the Reviews')
plt.axis('off')
plt.show()
# Save the Trained Model and TF-IDF Vectorizer
joblib.dump(rf_model, 'sentiment_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

print("\nModel and vectorizer saved successfully.")

