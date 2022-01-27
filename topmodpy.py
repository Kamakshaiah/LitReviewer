# package for topic modeling
# to dos: need to implement dict for 'printtopics' method
#           need to implement dict in 'printtopicswithweights'


def help():
    ''' shows help for topmodpy module '''

    print("FileImport - arg: 'path'; return: dataset")
    print('GetHead - First few records of data set (imported using method - FileImport()')    
    print('CreateVariable - args: data, var; return data variable [arg: var] (of interest)')
    print('CleanVar - args: var; removes special characters and convert data (words in docs) into lower case letters')
    print('CreateWordcloudImg - args: var; returns wordcloud image')
    print('CreateCountVector - args: var; computes counts for each word for given input variable [arg: var]')
    print('PrintTopics - args: var, nw, nt; prints topics for inputs nt (number of words) and nt (number of topics) for a given input variable (var)')
    print('PrintTopicsWithWeights - args: var, nt, nw; prints (return not implemented) for nt (number of words) and nt (number of topics) for a given input variable (var)')
    print("LDA - args: count_data; returns (LDA) model for input data variable (count_data). Use CreateCountVector method to compute 'count_data'.")
    print("CountVectorizer - A helpler function for 'MakeHTML'")
    print("MakeHTML - args: var; creates interactive HTML doc with added visuals for each topic.")
    print("OpenHTMLFile - no args; a helper function to open HTML document created by method 'MakeHTML'. ")
        
    
def fileImport(file_path):
    ''' Imports data file from the file_path (arg). Returns a data set as pandas data frame '''
        
    import pandas as pd
    data = pd.read_csv(file_path, encoding='latin1')
    return data

def getHead(data):

    ''' Prints first few records of data set [imported using method - FileImport()] '''

    print(data.head())
    

def createVariable(data, var):

    ''' Creates variable with all abstracts in data file,
        data     : created by using 'fileImport()' method,
        var      : abstract/s column in the data,
        output   : data variable as pandas DF. '''
    
    import pandas as pd

    datavar = data[var]
    return datavar

def cleanVar(var):

    ''' Removes special characters and convert data (words in docs) into lower case letters
        var: variable with all abstracts [created by 'createVariable()']; pandas DF.  '''

    import re
    cleanedvar = var.map(lambda x: re.sub('[,\.!?]', '', x))
    cleanedvar = cleanedvar.map(lambda x: x.lower())
    return cleanedvar

def textToWords(var):

    ''' creates a list of words for each document in the input variable of documents.
        var: the cleaned variable produced by 'cleanVar' function. '''

    from collections import Counter
    import spacy
    nlp = spacy.load('en_core_web_sm')

    words = []
    for i in var:
        if str(i):
            p = nlp(i)
            words.append([t.text for t in p if not t.is_stop and not t.is_punct])
        else:
            print('Data is not string type. Check for missing data in your file!')
    wf = []
    common_words = []
    for i in range(len(words)):
        wf.append(Counter(words[i]))
        common_words = wf[i].most_common(5)

    return [wf, common_words]

def makeDataFrameFromWords(words, n=2):
    ''' useful to make data frames from output produced by 'textToWords'.
        'textToWords' has output in which there are two objects (1) dictionaries of words and their counts, (2) most common words. 
        '''
    import pandas as pd

    dfs = []
    n = n-1
    for i in range(n):
        dfs.append('df' + str(i))

    for i in range(n):
        dfs[i] = pd.DataFrame(words[0][i].items(), columns = ['words', 'count'])
        
    print("Done! data frames are saved in 'dfs' object")
    return dfs
        
def makeCoommonDataFrame(df1, df2):
    ''' creates a dataframe with common words for two data fraes; 
        df1: data frame one
        df2: data frame two
        The coloumn names need to be 'words, count' '''
    
    out = df1[df1['words'].isin(df2['words'])]
    return out

def concatenateCols(df1, df2, colname = 'col1'):
    ''' creates a data frame with common words and their respective cols from input data frames
        df1: data frame one
        df2: data frame two '''
    
    df = makeCoommonDataFrame(df1, df2).sort_values(by='words')
    df1 = makeCoommonDataFrame(df2, df1).sort_values(by='words')
    df[colname] = df1['count'].values
    return df

def wordPlot(df):
    ''' plot for common words data frame with names 'count' and 'col1' produced by 'concatenateCols' function
        df: pandas data frame '''
    
    import matplotlib.pyplot as plt
    ax = df.plot(kind='scatter',x='count',y='col1')
    df[['count','col1','words']].apply(lambda row: ax.text(*row),axis=1)
    plt.show()

def chisqTest(df1wrds, df2wrds, test='chi-square'):

    ''' performs 'test' on crosstabs produced from words which is produced by 'makeDataFrameFromWords' function
        df1 = makeCoommonDataFrame(df1, df2) [out object]
        df2 = makeCoommonDataFrame(df1, df2) [out object]
        test = 'chi-square', 'g-test', 'fisher', 'mcnemar'  '''
    
    import researchpy as rpy
    out = rpy.crosstab(df1wrds, df2wrds, test=test)
    return out
    
    
def documentByTotalTerms(lodfs):
    ''' makes document by its total count of words
        lodfs/out: list of data frames created from 'makeCoommonDataFrame' function '''

    wordlen = []
    for i in range(len(lodfs)):
        wordlen.append((i, len(lodfs[i])))
    return wordlen 

def docWithMaxWords(lodfs):
    ''' return document number with maximum word length from list of data frames (lodfs/out)    '''
    docwisewordlens = documentByTotalTerms(lodfs)
    
    for i, j in dict(docwisewordlens).items():
        if j == max(dict(docwisewordlens).values()):
            doc_num = i
    return {'doc': doc_num, 'wordvec': lodfs[doc_num]}



def createWordcloudImg(var):

    ''' Creates wordcloud image
        args    : var,
        returns : wordcloud image '''

    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    longstring = ','.join(list(var.values))
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
    wcplot = wordcloud.generate(bytes(longstring, 'utf-8'))

    plt.imshow(wcplot, interpolation='bilinear')
    plt.show()

def createTermVector(var):

    ''' Creates term vector - computes counts for each word for given input variable [arg: var] 
        args    : var '''

    from sklearn.feature_extraction.text import CountVectorizer

    count_vectorizer = CountVectorizer(stop_words='english')
    count_data = count_vectorizer.fit_transform(var)
    return count_data

def countData(var):
    ''' Gives word frequencies '''
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np

    import seaborn as sns
    sns.set_style('whitegrid')

    count_vectorizer = CountVectorizer(stop_words='english')
    count_data = count_vectorizer.fit_transform(var)

    return count_data

def printTopics(var, nt, nw):

    ''' prints topics for inputs nt (number of words) and nt (number of topics) for a given input variable (var) 
        args: var, nw, nt '''

    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np

    import seaborn as sns
    sns.set_style('whitegrid')

    count_vectorizer = CountVectorizer(stop_words='english')
    count_data = count_vectorizer.fit_transform(var)

    ##plot_10_most_common_words(count_data, count_vectorizer)

    import warnings
    warnings.simplefilter("ignore", DeprecationWarning)

    # Load the LDA model from sk-learn
    from sklearn.decomposition import LatentDirichletAllocation as LDA

    def print_topics(model, count_vectorizer, n_top_words):
        words = count_vectorizer.get_feature_names()
        for topic_idx, topic in enumerate(model.components_):
            print("\nTopic #%d:" % topic_idx)
            print(" ".join([words[i]for i in topic.argsort()[:-n_top_words - 1:-1]]))

    # Tweak the two parameters below
    number_topics = nt
    number_words = nw

    # Create and fit the LDA model
    lda = LDA(n_components=number_topics, n_jobs=-1)
    lda.fit(count_data)

    # Print the topics found by the LDA model
    print("Topics found via LDA:")
    topics = print_topics(lda, count_vectorizer, number_words)
    return topics

def docTermMatrix(var):
    
    ''' Useful to make document term matrix '''

    finalvar = cleanVar(var)
    
    doc_complete = []
    [doc_complete.append(d) for d in finalvar]

    import nltk
    ##nltk.download()
    from nltk.corpus import stopwords
    from nltk.stem.wordnet import WordNetLemmatizer
    import string
    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()

    def clean(doc):
        stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
        punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
        normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
        return normalized

    doc_clean = [clean(doc).split() for doc in doc_complete]

    import gensim
    from gensim import corpora

    # Creating the term dictionary of our courpus, where every unique term is assigned an index.
    dictionary = corpora.Dictionary(doc_clean)

    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    
    return doc_term_matrix


def printTopicsWithWeights(var, nt, nw):

    ''' prints (return not implemented) for nt (number of words) and nt (number of topics) for a given input variable (var) 
        PrintTopicsWithWeights - args: var, nt, nw '''

    finalvar = cleanVar(var)
    
    doc_complete = []
    [doc_complete.append(d) for d in finalvar]

    import nltk
    ##nltk.download()
    from nltk.corpus import stopwords
    from nltk.stem.wordnet import WordNetLemmatizer
    import string
    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()

    def clean(doc):
        stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
        punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
        normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
        return normalized

    doc_clean = [clean(doc).split() for doc in doc_complete]

    import gensim
    from gensim import corpora

    # Creating the term dictionary of our courpus, where every unique term is assigned an index.
    dictionary = corpora.Dictionary(doc_clean)

    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

    # Creating the object for LDA model using gensim library
    Lda = gensim.models.ldamodel.LdaModel

    # Running and Trainign LDA model on the document term matrix.
    ldamodel = Lda(doc_term_matrix, num_topics=nt, id2word = dictionary, passes=50)

    return ldamodel.print_topics(num_topics=nt, num_words=nw)
    
    
def LDA(count_data):

    ''" LDA - args: count_data; returns (LDA) model for input data variable (count_data). Use CreateCountVector method to compute 'count_data'."''

    from sklearn.decomposition import LatentDirichletAllocation as LDA
    # Tweak the two parameters below
    number_topics = 5
    number_words = 10

    # Create and fit the LDA model
    lda = LDA(n_components=number_topics, n_jobs=-1)
    lda.fit(count_data)

    return lda

def countVectorizer():

    ''" A helpler function for 'MakeHTML' "''
    
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np

    import seaborn as sns
    sns.set_style('whitegrid')

    count_vectorizer = CountVectorizer(stop_words='english')
    return count_vectorizer

def makeHTML(var):

    ''" MakeHTML - args: var; creates interactive HTML doc with added visuals for each topic. "''
    
    from pyLDAvis import sklearn as sklearn_lda
    import pickle
    import os
    import pyLDAvis

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation as LDA
    import numpy as np

    import seaborn as sns
    sns.set_style('whitegrid')

    count_vectorizer = CountVectorizer(stop_words='english')
    count_data = count_vectorizer.fit_transform(var)

    import warnings
    warnings.simplefilter("ignore", DeprecationWarning)

    # Tweak the two parameters below
    number_topics = 5
    number_words = 10

    # Create and fit the LDA model
    lda = LDA(n_components=number_topics, n_jobs=-1)
    lda.fit(count_data)
    
    LDAvis_data_filepath = os.path.join(os.getcwd() + '\\ldavis_prepared_'+ str(number_topics))


    LDAvis_prepared = sklearn_lda.prepare(lda, count_data, count_vectorizer)
    with open(LDAvis_data_filepath, 'wb') as f:
        pickle.dump(LDAvis_prepared, f)
        
    # load the pre-prepared pyLDAvis data from disk

    with open(LDAvis_data_filepath, 'rb') as f:
        LDAvis_prepared = pickle.load(f)

    pyLDAvis.save_html(LDAvis_prepared, LDAvis_data_filepath +'.html')

def openHTMLFile():

    ''" OpenHTMLFile - no args; a helper function to open HTML document created by method 'MakeHTML'. "''
    
    import webbrowser
    import os

    number_topics = 5
    LDAvis_data_filepath = os.path.join(os.getcwd() + '\\ldavis_prepared_'+ str(number_topics))

    url = LDAvis_data_filepath +'.html'
    new = 2

    webbrowser.open(url,new=new)


if __name__ == '__main__':
##    data = fileImport("D:\\Research\\PAPERS\\topicmodeling\\app\\topicmodeling\\data.xlsx")
    data = fileImport('D:\\Research\\PAPERS\\covid19\\MJAFIndia\\edited\\abstracts.csv')
##    getHead(data)
    var = createVariable(data, "abstract")

##    from documentprocessor.docprocessor import makeDocTermMatrix
##    doctermmat = makeDocTermMatrix(var)
##    print(doctermmat.head())

    var.head()
    cleanedvar = cleanVar(var)

    words = textToWords(cleanedvar)
    out = makeDataFrameFromWords(words, n=len(words[0]))
    df1wrds = out[0]['words']
    df2wrds = out[1]['words']
    print(chisqTest(df1wrds, df2wrds))
          
##    print(len(out))
##    print(makeCoommonDataFrame(out[1], out[2]))
    
##    docbytotalwords = documentByTotalTerms(out)
##    docwithmaxwords = docWithMaxWords(out)
##    print(docwithmaxwords['doc'])
##    dwmwdoc = docwithmaxwords['doc']
##    df = concatenateCols(out[dwmwdoc], out[1], colname = 'col1')
##    print(type(df))
##    wordPlot(df)

##    print(cleanedvar.head())
##    createWordcloudImg(cleanedvar)
    countdata = countData(cleanedvar)
##    print(countdata)
##    dtm = docTermMatrix(cleanedvar)
##    print(dtm[0])
##    printTopics(cleanedvar, 5, 6)
##    
##    lda = LDA(countdata)
##    cv = countVectorizer()
##
##    makeHTML(var)
##    openHTMLFile()
    tww = printTopicsWithWeights(cleanedvar, 5, 10)
##    for i in tww:
##        print(i)
