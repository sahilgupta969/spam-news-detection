📰 Fake News Detection with Machine Learning

A machine learning project to classify news articles as real or fake using NLP and supervised learning.

🔍 Overview
This project leverages Natural Language Processing (NLP) and Machine Learning to detect fake news. It preprocesses text data, extracts features using TF-IDF, and trains classification models to distinguish between credible and misleading news articles.

📂 Dataset
The dataset consists of two CSV files:

True.csv: Legitimate news (labeled 0)

Fake.csv: Fake news (labeled 1)

Columns:
Column	Description
title	Headline of the article
text	Content of the article
subject	Category (e.g., politics, world)
date	Publication date
🛠️ Data Preprocessing
Loading & Merging Data

Combined True.csv and Fake.csv into a single DataFrame.

Text Cleaning

Lowercasing

Removing special characters/punctuation

Tokenization & Lemmatization (WordNetLemmatizer)

Stopword removal (nltk.corpus.stopwords)

🔧 Feature Extraction
TF-IDF Vectorization

Max features: 50,000

N-gram range: (1, 2)

🤖 Models & Performance
1. Naive Bayes (MultinomialNB)
✅ Test Accuracy: ~70%
⚠️ Training Accuracy: ~96% (potential overfitting)

2. Logistic Regression (LogisticRegression)
(Metrics pending due to notebook execution errors)

🚀 Usage
Clone the repo:

bash
git clone https://github.com/yourusername/fake-news-detection.git
Install dependencies:

bash
pip install numpy pandas nltk scikit-learn
Download NLTK datasets:

python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
Run the notebook and input news text to classify:

python
Enter News: "Your news article here..."
Output:

✅ News is Correct (Real)

❌ News is Fake (Fake)

📦 Dependencies
Library	Purpose
numpy	Numerical operations
pandas	Data manipulation
nltk	NLP preprocessing
scikit-learn	ML models & vectorization
🔮 Future Improvements
Experiment with advanced models (Random Forest, BERT, Transformers).

Reduce overfitting (hyperparameter tuning, cross-validation).

Enhance preprocessing (handle emojis, URLs, named entities).

Deploy as a web app (Flask/Django + Heroku).

📜 License
This project is licensed under the MIT License.

✨ Contribute
Feel free to open issues or submit PRs!
