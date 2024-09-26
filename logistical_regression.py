# Importing necessary libraries
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

# For saving the plots
import io

# Download NLTK data
nltk.download('punkt_tab')
nltk.download('stopwords')

# Loading train.csv and test.csv
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Step 2: Checking for Class Imbalance in Train Data
class_distribution = train_df['target'].value_counts()
print("Class Distribution (Train dataset):\n", class_distribution)

# Plotting the class distribution
sns.barplot(x=class_distribution.index, y=class_distribution.values)
plt.title("Class Distribution: Disaster vs Non-Disaster Tweets")
plt.ylabel("Number of Tweets")
plt.xlabel("Class (0 = Non-disaster, 1 = Disaster)")
plt.savefig('class_distribution.png')
plt.close()

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

# Option 1: Random Oversampling
oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# Verifying the new class distribution after oversampling
print(f"Resampled class distribution: {pd.Series(y_resampled).value_counts()}")

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42)

# Step 7: Model Training (Logistic Regression)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Initialising the model
model = LogisticRegression(max_iter=1000)  # Increase max_iter if needed

# Training the model
model.fit(X_train, y_train)

# Step 8: Predictions and Evaluation
y_pred = model.predict(X_test)

# Evaluating accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Detailed classification report
print(classification_report(y_test, y_pred))

# Step 9: Preprocessing the test data
test_df['clean_text'] = test_df['text'].apply(preprocess_text)

# Transforming the test data using the same TF-IDF vectorizer
X_test_final = tfidf.transform(test_df['clean_text'])

# Step 10: Making predictions on the test dataset
test_predictions = model.predict(X_test_final)

# Step 11: Checking the distribution of predictions
predicted_class_distribution = pd.Series(test_predictions).value_counts()
print("Predicted Class Distribution (Test dataset):\n", predicted_class_distribution)

# Step 12: Visualising the predicted class distribution
sns.barplot(x=predicted_class_distribution.index, y=predicted_class_distribution.values)
plt.title("Predicted Class Distribution: Disaster vs Non-Disaster Tweets (Test Set)")
plt.ylabel("Number of Tweets")
plt.xlabel("Class (0 = Non-disaster, 1 = Disaster)")
plt.savefig('predicted_class_distribution.png')
plt.close()
