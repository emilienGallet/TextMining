import random

import nltk

nltk.download('reuters')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import reuters

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

TfidfVectorizer

def documentTermMatrix(docs: 'list[str]'):
    vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', tokenizer=word_tokenize)
    X = vectorizer.fit_transform(docs)
    print(X)
    vectorizer.get_feature_names_out()
    # matrix: dict = {}
    # for lemme in lemmeList:
    #     if (mlemme := matrix.get(lemme, None)) is None:
    #         matrix.update({lemme: 1})
    #     else:
    #         matrix.update({lemme: mlemme+1})


if __name__ == '__main__':
    ids = reuters.fileids()
    doc = None
    for i, e in enumerate(ids):
        if e.startswith('training'):
            doc = e
            break
    doc: str = reuters.raw(doc)
    tokens: 'list[str]' = word_tokenize(doc)
    lemmeList = []
    for token in tokens:
        token = token.lower()
        if token not in stopwords.words('english'):
            lemmeList.append(WordNetLemmatizer().lemmatize(token, "n"))
######################################################
    ids = reuters.fileids()
    start = None
    for i, e in enumerate(ids):
        if e.startswith('training'):
            start = i
            break
    corpus = []
    while i < len(ids):
        corpus.append(reuters.raw(ids[i]))
        i += 1
    documentTermMatrix(corpus)
