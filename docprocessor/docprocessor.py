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
    ''' reads data from arg::path (uses encoding='latin1')'''
    
    import pandas as pd
    data = pd.read_csv(path, encoding='latin1')
    return data

def covariance(out):
    ''' returns covariance matrix for arg::out [document term matrix ] '''
    from numpy import cov
    res = cov(out)
    return res

def correlation(out, doc1=0, doc2=1, type="pearson"):
    ''' performs correlation between two docs from arg::out and arg::doc 1 and arg::doc 2 '''
    from scipy.stats import pearsonr, spearmanr, kendalltau
    if type=="kendall":
        out = kendalltau(out[doc1], out[doc2])
    elif type=="spearman":
        out = spearmanr(out[doc1], out[doc2])
    else:
        out = pearsonr(out[doc1], out[doc2])
    return out

def regression(out, doc1=0, doc2=1):
    ''' performs simple linear regression for arg::out between two input variables arg::doc1 and arg::doc2 '''
    from scipy.stats import linregress
    result = linregress(out[doc1], out[doc2])
    return result

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
##    print(out)
    covres = covariance(out)
    print(len(covres) )
##    print(correlation(out, doc1=0, doc2=1, type="kendall"))
##    print(regression(out, doc1=0, doc2=1))
