# CSCE-5214-NLP-Disaster-Tweets

## Project Deployment Guide

Prerequisites
Python 3.6 or higher installed on your system.
Train and Test Datasets: Ensure you have train.csv and test.csv files placed in the project directory.

They are included in this git project, but if you need them otherwise:

You can download both from www.kaggle.com/competitions/nlp-getting-started/data

## Setup Instructions

The project was created using PyCharm, if you have PyCharm simply clone the project and open it through PyCharm.
Otherwise, please follow these instructions:

### Clone or Download the Project Files

### Navigate to the Project Directory

Open a terminal or command prompt and navigate to the directory containing the project files.

### Create a Virtual Environment

In terminal:
python -m venv venv

Activate the virtual environment:
venv\Scripts\activate

### Install Required Libraries

Install the necessary Python packages using pip:

pip install flask pandas seaborn scikit-learn nltk imbalanced-learn matplotlib -UPDATE FOR BERT-

### Download NLTK Data

Open a Python interpreter to download the required NLTK data:

import nltk
nltk.download('punkt')
nltk.download('stopwords')

Execute these commands in the Python interpreter.


## Running the Application
### Start the Flask Server

In the terminal, run: python app.py

This will start the server on http://localhost:5000/.

### Access the Web Interface

Open a web browser and navigate to:
http://localhost:5000/

## Notes
The logistic regression script will save plots as PNG files in the project directory:
class_distribution.png
predicted_class_distribution.png

## Deactivate
Run "deactivate" in terminal