
def appendText(path):
    
    ''' Append text inside text files at given path. Makes a master output file concatinating all the text inside text (.txt) files
        params: path '''
    
    import os
    pathch = os.path.join(path, 'output')
    filenames = os.listdir(pathch)
    filepath = os.path.join(pathch, 'textfile.txt')
    try:
        with open(filepath, 'w', encoding='unicode_escape') as outfile:
            for f in filenames:
                with open(os.path.join(pathch,  f), 'r', encoding='unicode_escape') as infile:
                    contents = infile.read()
                    contents.encode('unicode_escape')
                    outfile.write(contents)
                    infile.close()
        
        outfile.close()
    except Exception as e:
        print('There was an error!')

def listOfWordsWithFreqs(path, print_words=False, nw = None):

    ''' Count (GOOGLE SEARCH list - gsearch module) WORDS and creates a dictinary of words and frequencies (termmat)
        wwov - words with out verbs (obtained from google search module)

        params: text (master text file created by appendText() method. '''
    
    pathch = os.path.join(path, 'output')
    filepath = os.path.join(pathch, 'textfile.txt')

    with open(filepath, 'r', encoding='unicode_escape') as infile:
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
    ''' Works for both gsearch and gscholar modules. Makes word frequencies from TITLES. Creates dictionary (ref. to termmat - term matrix)
        params: titles [retrieved from gsearch and gscholar modules] '''

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


def makeTables(data, path=None, file_name=None):
    ''' Creates tablar data for distinctwords (dictionary) or termmat (dictionary)
        params: data [a dictionary with words and frequencies ]
                path [variable]
                save [boolean: True/False] '''
    
    import pandas as pd
    import os

    tbl = pd.DataFrame(data, columns =['words', 'freq'])
    
    try:
        pathch = os.path.join(path, f'output\\{file_name}.csv')
        with open(pathch, 'w+') as csv_file:
            tbl.to_csv(path_or_buf=csv_file, line_terminator='\n')
            csv_file.close()

    except Exception as e:
        print('There was a problem while writing data to file. However you can safely neglect the same.')

    return tbl

def saveAsCSVFile(data, path, name= 'file.csv'):
    ''' --- DEPRECATED in stead use makeTables() method ---

        Converts pandas DF (tables) to csv file and saves in the request path
        params: path [variable]
                data [[a dictionary with words and frequencies ]]'''
    import os

    path = os.path.join(path, 'output')
    
    if os.path.exists(path):
        path = os.path.join(path, name)
        data.to_csv(path)
    else:
        os.makedirs(path)
        path = os.path.join(path, name)
        data.to_csv(path)        
    

if __name__ == '__main__':

    import os
    path = os.getcwd()
    print(path) 

    appendText(path)
    
    wfreq = listOfWordsWithFreqs(path, print_words=True, nw=10)

    datatbl = makeTables(wfreq, path, file_name = 'termmat')

    from literstat import *
    print(computeIQR(datatbl['freq']))
    print(oneSampleTTest(datatbl['freq']))
        
##    df = datatbl.head()
##    barChart(df)
##    pieChart(df)
##    boxPlot(df)
    clusanal = uniVarClusterAnalysis(datatbl, nc=1)
    print(clusanal) 
