import re
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer

stop_words = set(stopwords.words('english'))

class Preprocessor(object):
    def __call__(self, document: str):
        document = document.lower()
        ## split up contractions
        document = re.sub(r"they'?re", 'they are', document)
        document = re.sub(r"wasn'?t", 'was not', document)
        
        return document

class Tokenizer(object):
    def __init__(self):
        self.stemmer = PorterStemmer()
    def __call__(self, documents: str):
        return [
            self.stemmer.stem(term)
            for term in word_tokenize(documents)
            if term.isalpha() and term not in stop_words
        ]

cv = CountVectorizer(
##    stop_words=stop_words,
    preprocessor=Preprocessor(),
    tokenizer=Tokenizer()
)

def makeDocTermMatrix(documents):
    data = cv.fit_transform(documents).toarray()
    vocab = cv.get_feature_names()
    doc_term_matrix = pd.DataFrame(
        data=data,
        columns=vocab
        ).transpose()
    return doc_term_matrix

def readData(path):

    import pandas as pd
    data = pd.read_csv(path, encoding='latin1')
    return data

if __name__ == '__main__':

    import os
    
    path = os.path.join('D:\Research\PAPERS\covid19\MJAFIndia\edited', 'abstracts.csv')
    data = readData(path)
    
##    documents = [
##        'Mom took us shopping today and got a bunch of stuff. I love shopping with her.',
##        "Friday wasn't a great day.",
##        'She gave me a beautiful bunch of violets.',
##        "Dad attested, they're a bunch of bullies.",
##        'Mom hates bullies.',
##        'A bunch of people confirm it.',
##        'Taking pity on the sad flowers, she bought a bunch before continuing on her journey home.'
##        ]
    out = makeDocTermMatrix(data['abstract'])
    print(out)
