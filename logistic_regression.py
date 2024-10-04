# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

import sys
import io

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def main():
    # We can pipe the output to put into our web interface
    pipedOutput = io.StringIO()
    sys.stdout = pipedOutput

    # Loading train.csv and test.csv
    train_df = pd.read_csv('train.csv')

    # Step 2: Checking for Class Imbalance in Train Data
    class_distribution = train_df['target'].value_counts()
    print("Class Distribution (Train dataset):\n", class_distribution)

    # Step 3: Preprocessing the text (Cleaning the data)
    def preprocess_text(text):
        # Removing URLs and special characters
        text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
        text = re.sub(r'\@\w+|\#','', text)
        text = re.sub(r"[^a-zA-Z]", " ", text)
        text = text.lower()

        # Tokenization
        tokens = word_tokenize(text)

        # Removing stopwords
        stop_words = set(stopwords.words("english"))
        tokens = [word for word in tokens if word not in stop_words]

        # Stemming
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]

        return ' '.join(tokens)

    # Applying preprocessing to the 'text' column in train_df
    train_df['clean_text'] = train_df['text'].apply(preprocess_text)

    # Step 4: Converting text to numeric data using TF-IDF
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(train_df['clean_text'])  # Features
    y = train_df['target']  # Target labels

    # Step 5: Handling Class Imbalance
    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X, y)

    # Step 6: Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Step 7: Model Training (Logistic Regression)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Step 8: Predictions and Evaluation
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%\n")

    report = classification_report(y_test, y_pred)
    print("Classification Report:\n", report)

    # Step 9: Predicting a custom tweet
    def predict_custom_tweet(tweet):
        processed_tweet = preprocess_text(tweet)
        tweet_vector = tfidf.transform([processed_tweet])
        prediction = model.predict(tweet_vector)
        return "Disaster" if prediction[0] == 1 else "Non-Disaster"

    # Example of using the custom prediction:
    user_tweet = input("Enter a tweet to classify: ")
    result = predict_custom_tweet(user_tweet)
    print(f"The tweet is classified as: {result}")

    sys.stdout = sys.__stdout__
    return pipedOutput.getvalue()

if __name__ == "__main__":
    output = main()
    print(output)
