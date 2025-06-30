import re
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"\W+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()

def vectorize_text(train_texts, test_texts):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_test, vectorizer