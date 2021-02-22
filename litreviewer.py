class gscholar:

    ''' Extracts material from Google Scholar. Supports few methods such as
            - retrieval of links
            - retrieval of titles
            - creats termmat (word-freq matrix)
            - create distinct words matrix (with unique words)
            - support statistical analysis
                - IQR
                - Two Sample T Test
                - Moods Test of Sample Medians
                - Cluster analysis
        Apart from the above methods; supports few utility functions such as:
            - creating tables for (termat, disctinct words, etc...)
            - saves tables in given paths '''

    hv = ['am', 'is', 'are', 'was', 'and', 'were', 'being', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should',
          'may', 'might', 'must', 'can', 'could', 'of', 'for', 'about', 'with', 'on', 'inside', 'under', 'lower', 'upper', 'a', 'an', 'the', 'in', 'new',
          'old', 'through', 'suitable', 'suiit']
          
    def __init__(self, q, hl='en', as_ylo=0, num=10):
        self.q = q
        self.hl = hl
        self.as_ylo=as_ylo
        self.num=num

    def makeQuery(self):
        ''' Create the url for google sholar search '''
        
        return 'https://scholar.google.com/scholar?'+'hl='+self.hl+'&'+'q='+self.q+'&'+'as_ylo='+str(self.as_ylo)+'&'+'num='+str(self.num)
    

    def getLinks(self, url):

        ''' retrieves links for search phrase '''
        
        from bs4 import BeautifulSoup
        import requests
        import random
        

        A = ("Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36",
               "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.1 Safari/537.36",
               "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.0 Safari/537.36",
               )
        Agent = A[random.randrange(len(A))]
        headers = {'user-agent': Agent}

        r = requests.get(url, headers=headers)
        soup = BeautifulSoup(r.content, 'lxml')
        alinks = soup.findAll('a')

        links = []

        for a in alinks:
            if 'google' in a.get('href'):
                next
            elif 'https://' in a.get('href'):
                links.append(a.get('href'))

        return links        

    def openSource(self, links, **kwargs):

        ''' Opens the given link '''

        import webbrowser as wb
        
        if 'num' in kwargs.keys():
            arg1 = kwargs['num']
            wb.open(links[arg1])
        elif 'allsources' in kwargs.keys():
            for i in links:
                wb.open(i)

    def getText(self, links, path, num, savetext=False):
        ''' Get abstract for given page source. No support for PDF files '''

        from bs4 import BeautifulSoup
        import requests
        import random

        A = ("Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.1 Safari/537.36",
             "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.0 Safari/537.36")
        
        Agent = A[random.randrange(len(A))]
        headers = {'user-agent': Agent}

        ps = requests.get(links[num], headers=headers)
        soup = BeautifulSoup(ps.content, 'lxml')
        ptags = soup.find_all('p')

        abs = ['abstract', 'Abstract', 'ABSTRACT']

        text = ''
        
        for p in ptags:
            print(p.text)
##            for i in abs:
##                if i in p.text:

        if savetext==True:
            for p in ptags:
                text += p.text + '\n'

        path = path + '\\file.txt'

        f = open(path,'w')
        f.write(text)
        f.close()            
                    
    def getTitles(self, url):
        ''' Retrieves titles '''

        from bs4 import BeautifulSoup
        import requests
        import random
        

        A = ("Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36",
               "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.1 Safari/537.36",
               "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.0 Safari/537.36",
               )
        Agent = A[random.randrange(len(A))]
        headers = {'user-agent': Agent}

        r = requests.get(url, headers=headers)
        soup = BeautifulSoup(r.content, 'lxml')
        heads = soup.findAll('h3')

        titles = []

        for t in heads:
            titles.append(t.text)

        return titles

    def wordFrequencies(self, titles):
        ''' Makes word frequencies with words and frequencies as dictionary (term matrix - termmat) '''

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
        
        return {'wordslist': wordlist1, 'wordfreq': wordfreq}

    def makeDistinctWords(self, title, termmat, length=7):

        ''' Creates words and their respective frequenceis for termmat (compares titles to its search string)
            Makes disctinct words matrix - (disctinctwords) '''

        words = []
        freq = []
        titlewords = title.split()

        for i, j in zip(termmat['wordslist'], termmat['wordfreq']):
            if i in self.hv:
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

    @staticmethod 
    def makeTables(data):
        ''' Creates tablar data for distinctwords or termmat '''
        
        import pandas as pd

        tbl = pd.DataFrame(data)
        return tbl

    @staticmethod
    def saveAsCSVFile(path, data):
        ''' Converts pandas DF to csv file and saves in the request path (arg: path)'''
        data.to_csv(path)


    @staticmethod
    def computeIQR(data):
        ''' Computes Inter Quartile Range (IQR) for word's frequency - input should be univariate distribution '''

        from scipy import stats
        stat = stats.iqr(data, axis=0)
        return stat
    
    @staticmethod
    def twoSampleIndTTest(v1, v2):
        ''' Performs two sample ind. T Test for two different samples - for termmat and distinctwords distributions '''
        
        from scipy import stats
        res = stats.ttest_ind(v1, v2)
        return res

    @staticmethod
    def moodsTest(v1, v2, ties=True):
        ''' Performs moods test for two different serach phrases '''

        from scipy.stats import median_test
        
     
        if ties:
            try:
                g, p, med, tbl = median_test(v1, v2, lambda_="log-likelihood", ties="above")
            except Exception as e:
                print(e, " occured! Test doesn't work")
        else:
            try:
                g, p, med, tbl = median_test(v1, v2, lambda_="log-likelihood")
            except Exception as e:
                print(e, " occured! Test doesn't work")
                
        return {'g': g, 'p': p, 'med': med, 'tbl': tbl} 

    def barChart(self, distinctwords):

        ''' Creates barchart for termmat (TM)/distinct words matrix (DWM)'''

        import matplotlib.pyplot as plt
        keys = list(distinctwords.keys())
        plt.bar(distinctwords[keys[0]], distinctwords[keys[1]])
        plt.xticks(rotation=45)
        plt.show()

    def pieChart(self, distinctwords):

        ''' Create Pie chart TM/DWM '''

        import matplotlib.pyplot as plt
        keys = list(distinctwords.keys())
        plt.pie(distinctwords[keys[1]], labels = distinctwords[keys[0]])
        
        plt.show()
        
    @staticmethod
    def boxPlot(data):
        
        ''' Boxplot for data vector(s); supports only pandas dataframe '''
        import matplotlib.pyplot as plt
        
        data.boxplot()
        plt.show()

    @staticmethod
    def reshapeData(data):
        ''' Reshapes input data into required format for cluster analysis '''
        import numpy as np

        idx = list(data.keys())[1]

        out = np.reshape(data[idx], (1, -1)).T

        return out

    @staticmethod
    def clusterAnalysis(data, path, nc=2, output=False):
        ''' Performs cluster analysis on input data.
            data - a dictionary such as TM or DWM
            nc - num of clusters
            path - path for output file (default name out.csv
            output = True makes output to file path '''

        from sklearn import cluster
        from sklearn import metrics
        import pandas as pd
        import numpy as np
        
        idx0 = list(data.keys())[0]
        idx1 = list(data.keys())[1]

        req_data = np.reshape(data[idx1], (1, -1)).T
               
        kmeans = cluster.KMeans(n_clusters=nc)
        kmeans.fit(req_data)

        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        kmeansclus = pd.DataFrame({'words': data[idx0], 'freq': data[idx1], 'labels':list(labels)})

        pathch = path + 'out.csv'

        if output:
            kmeansclus.to_csv(pathch)
            
        score = kmeans.score(req_data)
        silhouette_score = metrics.silhouette_score(req_data, labels, metric='euclidean')

        out = {'labels': labels, 'centroids': centroids, 'score': score, 'silhouette_score': silhouette_score}
        return out


    @staticmethod
    def wordsByCategory(file_path, nclus):
        ''' Reads data from file system (file_path) and makes cluster wise words.
            Output has a dictionary with cluster number as keys and words as values '''

        import pandas as pd

        clusdata = pd.read_csv(file_path)
        num_clus = list(range(nclus))

        cluswords = {}
        
        for i in num_clus:
            cluswords[i] = clusdata[clusdata.labels == num_clus[i]]['words']

        return cluswords

    @staticmethod
    def pieForCategories(cats):
        ''' Pie chart for categories. Requires cluster wise words (such as created by the method 'wordsByCategory' '''

        import matplotlib.pyplot as plt

        vals = []
        for i in cats.values():
            vals.append(len(i))

        plt.pie(vals, labels=vals)
        plt.show()
    

    @staticmethod
    def crosstabFromWordsMatrix(file_path, output=False, norm=True):
        ''' Reads data from file system (file_path) and creates cross tabs for further analysis.
            The input file must be an TM/DWM (out.csv).
            Outputs crosstab for words and lables '''
        
        import pandas as pd
        from scipy import stats
        
        dataforctbl = pd.read_csv(file_path)
        ctbl = pd.crosstab(dataforctbl['words'], dataforctbl['labels'], normalize = norm)

        pathch = file_path + 'ctbl.csv'

        if output:
            ctbl.to_csv(pathch)

        out = stats.chi2_contingency(ctbl)

        return {'ctbl': ctbl, 'results': {'chi_sq': out[0], 'p_value': out[1], 'dof': out[2]}}

            
if __name__ == '__main__':
    title = 'impact of religious and spiritual practices on covid-19 mitigation'
    path = 'D:\\work\\python_work\\literature-review-mainone\\'
    gs = gscholar(title, num=50)
    url = gs.makeQuery()
    print(url)
    links = gs.getLinks(url)
    
##    print(links) 
##    print(len(links))
##
##    for i in links:
##        print(i) 
##    gs.openSource(links, num=7)
##    gs.getText(links, path=path, num=15,  savetext=True)
    titles = gs.getTitles(url)
##    for i in titles:
##        print(i)
    termmat = gs.wordFrequencies(titles)
##    print(termmat)
    distinctwords = gs.makeDistinctWords(title, termmat)
##    print(distinctwords)
##    gs.barChart(distinctwords)
    dwtbl = gs.makeTables(distinctwords)
##    gs.boxPlot(dwtbl)
    pathch = path + 'dwtbl.csv'
    gs.saveAsCSVFile(pathch, dwtbl)
##    gs.twoSampleIndTTest(termmat['wordfreq'], distinctwords['freq']) # this test can be used for two different searches
    
##    print(gs.computeIQR(dwtbl['freq']))
##    wordvec = gs.reshapeData(distinctwords)
##    nc = 2
    clssol = gs.clusterAnalysis(distinctwords, path, output=True) # saves the output 
##    print(clssol) 
    pathch = path + 'out.csv'
    cluswords = gs.wordsByCategory(pathch, 2) # reads data from fs
##    gs.pieForCategories(cluswords)
    print(gs.crosstabFromWordsMatrix(pathch, output=True, norm=True)) # reads data from fs and then makes cross tab
    
    
