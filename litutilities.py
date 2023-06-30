def fileImport(path):
    ''' Imports data file from path. Returns a data set as pandas data frame '''
        
    # imports data file returns 'data' object

    import pandas as pd
    if '.csv' in path:
        data = pd.read_csv(path, encoding='latin1')
    else:
        data = pd.read_excel(path, encoding='latin1')
##    data = pd.read_csv(path, encoding='latin1')
    return data

def createVariable(data, var):

    ''' Creates variable with all abstracts in data file,
        args    :   data (imported file using fileImport() method),
                    var (abstract column in the data),
        return  :   data variable (of interest) '''
    
    import pandas as pd

    datavar = data[var]
    return datavar

def cleanVar(var):

    ''' Removes special characters and convert data (words in docs) into lower case letters
        args      : var  '''

    import re
    cleanedvar = var.map(lambda x: re.sub('[,\.!?]', '', x))
    cleanedvar = cleanedvar.map(lambda x: x.lower())
    return cleanedvar
    

def textToWords(var):

    import itertools
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
    words = itertools.chain(*words)
    wf = Counter(list(words))
    common_words = wf.most_common(5)
    
    return [wf, common_words]

def convertCountersIntoWordFreq(counter_data):
    ''' converts counters [textToWords()] into word frquencies (pandas df) '''
    
    import pandas as pd
    wf = pd.DataFrame.from_dict(counter_data, orient='index').reset_index()
    wf.columns=['words', 'freq']
    return wf

def appendText(dir_path):
    
    ''' Append text inside text files at given path. Makes a master output file concatinating all the text inside text (.txt) files.
        Used for gsearch module. Writes the whole mass of text into a file 'textfile.txt' in the given path. 
        dir_path - directory where all the .txt files (gsearch) are there/downloaded. '''
    
    import os

    filenames = os.listdir(dir_path)
    filepath = os.path.join(dir_path, 'textfile.txt')
    try:
        with open(filepath, 'w', encoding='unicode_escape') as outfile:
            for f in filenames:
                with open(os.path.join(dir_path,  f), 'r', encoding='unicode_escape') as infile:
                    contents = infile.read()
                    contents.encode('unicode_escape')
                    outfile.write(contents)
                    infile.close()
        
        outfile.close()
    except Exception as e:
        print('There was an error!')

def readFromExcel(file_path, name):
    ''' read data from excel file column '''

    import pandas

    data = pd.read_excel(path, encoding='latin1')
    return data

def cleanData(file_path, words_to_print=10, freq=1):

    ''' For gsearch module. file_path: 'textfile.txt' file path. 'textfile.txt' is a file object created using 'appendText()' method.  '''

    import os
    import spacy
    from collections import Counter

    filepath = file_path

    with open(filepath, 'r+') as file:
        text = file.read()
        file.close()

    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    
    nlp = spacy.load('en_core_web_sm')

    if len(text) > 1000000:
        nlp.max_length = len(text) + (len(text)*0.1)
        
    text = nlp(text)
    textwosw = []

    words = [token.text for token in text if not token.is_stop and not token.is_punct]
    wf = Counter(words)
    wc = Counter(wf)
    common_words = wf.most_common(words_to_print)
    unique_words = [word for (word, freq) in wf.items() if freq == freq]

    return {'common': common_words, 'unique': unique_words, 'words': wc}
        
def listOfWordsWithFreqs(file_path, print_words=False, nw = None):

    ''' Count words and creates a dictinary of words and frequencies (termmat)
        file_path: path for text file (master text file created by appendText() method. '''
##    from cleandata import cleandata

        
##    pathch = os.path.join(path, 'output')
##    filepath = os.path.join(pathch, 'textfile.txt')
##    cd = cleandata(filepath)
    
    with open(file_path, 'r', encoding='unicode_escape') as infile:
        text = infile.read()
        infile.close()
    
    words = text.split()

    counts = dict()

    for w in words:
        if w in counts:
            counts[w] += 1
        else:
            counts[w] = 1
    if print_words:
        print(list(counts.items())[:nw])
        
    return list(counts.items())

def wordFrequencies(titles):
    ''' Works for both gsearch and gscholar modules. Creates word frequencies from TITLES object. Creates dictionary (ref. to termmat - term matrix)
        titles: titles [retrieved from gsearch and gscholar modules] '''

    import re
    
    text = ''

    for t in titles:
        text += t

    wordlist = text.split()
    
    wordlist1 = []
    for i in wordlist:
        wordlist1.append(re.sub('[^A-Za-z0-9]', ' ', i.lower()))

    wordfreq = []
    for w in wordlist:
        wordfreq.append(wordlist1.count(w))
    
    return {'words': wordlist1, 'freq': wordfreq}


def makeDistinctWords(title, termmat, length=7):

    ''' Creates words and their respective frequenceis for termmat (compares titles to its search string)
        Makes disctinct words matrix - (disctinctwords)

        params: title [search query]
                termmat [produced by listOfWordsWithFreqs() method] '''

    hv = ['am', 'is', 'are', 'was', 'and', 'were', 'being', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should',
          'may', 'might', 'must', 'can', 'could', 'of', 'for', 'about', 'with', 'on', 'inside', 'under', 'lower', 'upper', 'a', 'an', 'the', 'in', 'new',
          'old', 'through', 'suitable', 'suiit']

    words = []
    freq = []
    titlewords = title.split()

    idx = []

    idx.append(list(termmat.keys())[0])
    idx.append(list(termmat.keys())[1])
    
    for i, j in zip(termmat[idx[0]], termmat[idx[1]]):
        if i in hv:
            next
        elif i in titlewords:
            next
        elif len(i) < length:
            next
        else:
            words.append(i)
            freq.append(j)

    if words:
        words_new = []
        freq_new = []
        for i, j in zip(words, freq):
            if i not in words_new:
                words_new.append(i)
                freq_new.append(j)
                            
    return {'words': words_new, 'freq': freq_new}


def makeTables(data):
    ''' Creates tablar data for distinctwords (dictionary) or termmat (dictionary)
        data: a dictionary with words and frequencies. '''
    
    import pandas as pd
    import os

    tbl = pd.DataFrame(data, columns =['words', 'freq'])
    return tbl

def saveAsCSVFile(data_tbl, file_path, name= 'file.csv'):
    ''' Converts pandas DF (tables) to csv file and saves in the requested path
        data_tble: a data table created using 'makeTables()'
        file_path: file path [.txt file]
        name: defaults to 'file.csv' (optional)
        '''
    import os

    path = os.path.join(file_path, 'output')
    
    if os.path.exists(path):
        path = os.path.join(path, name)
        data_tbl.sort_values(by='freq')
        data_tbl.to_csv(path)
    else:
        os.makedirs(path)
        path = os.path.join(path, name)
        data_tbl.sort_values(by='freq')
        data_tbl.to_csv(path)        
    

##if __name__ == '__main__':

    import os
##    path = os.getcwd()
##    print(path)

##    for abstracts
##    path = 'D:\Research\PAPERS\covid19\MJAFIndia\edited'
##    pathch = os.path.join(path, 'abstracts.csv')
##    data = fileImport(pathch)
##    var = createVariable(data, 'abstract')
##    print(var.head())
##    print(data.head())
    
##    txttowords = textToWords(var) 
    
    
##    appendText(path)

##    wfreq = listOfWordsWithFreqs(path, print_words=True, nw=10)
##    wfreq = wordFrequencies(var)
##    wfreqdistinct = makeDistinctWords('impact healthcare capacity immunization Covid-19 mitigation', wfreq)
##    print(wfreqdistinct.keys())
##    makeTables(wfreq, path) 
##    wfreq = cleanData(path, words_to_print=10, freq=1)
##
##    data = wfreq['words'].items()
##    print(data.head())
##
##    datatbl = makeTables(data, path, file_name = 'termmat2')
##
##    from literstat import *
##    print(computeIQR(datatbl['freq']))
##    print(oneSampleTTest(datatbl['freq']))
        
##    df = datatbl.head()
##    barChart(df)
##    pieChart(df)
##    boxPlot(df)
##    clusanal = uniVarClusterAnalysis(datatbl, nc=1)
##    print(clusanal) 
