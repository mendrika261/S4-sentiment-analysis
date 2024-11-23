## Overview 🔮

Sentiment analysis and categorization of text using machine learning based on Twitter dataset.

https://github.com/user-attachments/assets/7e8b6256-a053-4dfd-b6be-1dd73981adb1

## How It Works 🧠

### 1. Model for Sentiment Analysis
- Clean the data
    - Remove stopwords using NLTK.
    - Unidecode the text to remove non-ASCII characters.
    - HTML parsing to remove HTML tags using BeautifulSoup.
    - Remove links, mentions, punctuation, and numbers.
    - Tokenize the text using NLTK (convert text into words).
    - Lemmatize the text using NLTK (convert words to their base form).
- Implement `TF-IDF (Term Frequency-Inverse Document Frequency)` from scratch and create a dataframe.
- Reduce dimensions using synonyms (using WordNet).
- Train and used `Random Forest Classifier`from the TF-IDF dataframe and the target variable (sentiment).

### 2. Model for Categorization
- From the TF-IDF dataframe obtained from sentiment analysis.
- Implemented `KMeans clustering from scratch` to categorize the text.
- Used the `elbow method` to find the best value of K.
- Used `WordCloud` to visualize the clusters for each text.

---

## How to get started ℹ️
- Create a virtual environment
```bash
python3 -m venv venv
```
- Activate the virtual environment
```bash
source venv/bin/activate
```
- Install the requirements
```bash
pip install -r requirements.txt
```
Download the necessary data
```bash
python3 -m nltk.downloader stopwords
python3 -m nltk.downloader punkt
python3 -m nltk.downloader wordnet
python3 -m nltk.downloader omw
```
- Run the program
```bash
python3 manage.py runserver
```
