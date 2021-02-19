class gs:
    ''' extracts material from Google Scholar '''
    hv = ['am', 'is', 'are', 'was', 'and', 'were', 'being', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should',
          'may', 'might', 'must', 'can', 'could', 'of', 'for', 'about', 'with', 'on', 'inside', 'under', 'lower', 'upper', 'a', 'an', 'the', 'in', 'new',
          'old', 'through', 'suitable', 'suiit']
          
    def __init__(self, q, hl='en', as_ylo=0, num=10):
        self.q = q
        self.hl = hl
        self.as_ylo=as_ylo
        self.num=num

    def makeQuery(self):
        ''' create the url for google sholar search '''
        return 'https://scholar.google.com/scholar?'+'hl='+self.hl+'&'+'q='+self.q+'&'+'as_ylo='+str(self.as_ylo)+'&'+'num='+str(self.num)
    

    def getLinks(self, url):
        
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

        ''' opens the link '''

        import webbrowser as wb
        
        if 'num' in kwargs.keys():
            arg1 = kwargs['num']
            wb.open(links[arg1])
        elif 'allsources' in kwargs.keys():
            for i in links:
                wb.open(i)

    def getText(self, links, path, num, savetext=False):
        ''' get abstract for given page source '''

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
        ''' retrieves titles '''

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
        ''' makes word frequencies '''

        text = ''

        for t in titles:
            text += t

        wordlist = text.split()
        wordfreq = []

        for w in wordlist:
            wordfreq.append(wordlist.count(w))
            
        wordlist1 = []
        for i in wordlist:
            wordlist1.append(i.lower())

        return {'wordslist': wordlist1, 'wordfreq': wordfreq}

    def makeDistinctWords(self, title, termmat, length=7):

        ''' creates words and frequenceis '''

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

    def barChart(self, distinctwords):

        ''' creates barchart for termmat '''

        import matplotlib.pyplot as plt
        keys = list(distinctwords.keys())
        plt.bar(distinctwords[keys[0]], distinctwords[keys[1]])
        plt.xticks(rotation=45)
        plt.show()

    def pieChart(self, distinctwords):

        ''' create PIE chart '''

        import matplotlib.pyplot as plt
        keys = list(distinctwords.keys())
        plt.pie(distinctwords[keys[1]], labels = distinctwords[keys[0]])
        
        plt.show()

    @staticmethod 
    def makeTables(data):
        ''' creates tablar data '''
        import pandas as pd

        tbl = pd.DataFrame(data)
        return tbl

    @staticmethod
    def saveAsCSVFile(path, data):
        ''' converts pandas DF to csv file and saves in the req. path '''
        data.to_csv(path)

    @staticmethod
    def twoSampleIndTTest(v1, v2):
        ''' performs two sample ind. T Test for two different samples '''
        from scipy import stats
        res = stats.ttest_ind(v1, v2)
        return res

    @staticmethod
    def boxPlot(data):
        
        ''' boxplot for data vectors '''
        import matplotlib.pyplot as plt
        
        data.boxplot()
        plt.show()

    @staticmethod
    def computeIQR(data):

        ''' computes inter quartile range '''

        from scipy import stats
        stat = stats.iqr(data, axis=0)
        return stat

    @staticmethod
    def reshapeData(data):
        ''' reshapes input data into required format '''
        import numpy as np

        idx = list(data.keys())[1]

        out = np.reshape(data[idx], (1, -1)).T

        return out

    @staticmethod
    def clusterAnalysis(data, nc, path, output=False):
        ''' performs cluster analysis on input data '''

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
        ''' makes cluster wise words '''

        import pandas as pd

        clusdata = pd.read_csv(file_path)
        num_clus = list(range(nclus))

        cluswords = {}
        
        for i in num_clus:
            cluswords[i] = clusdata[clusdata.labels == num_clus[i]]['words']

        return cluswords
        
    @staticmethod
    def pieForCategories(cats):
        ''' pie chart for categories '''

        import matplotlib.pyplot as plt

        vals = []
        for i in cats.values():
            vals.append(len(i))

        plt.pie(vals, labels=vals)
        plt.show()

    
            
        





        
if __name__ == '__main__':
    title = 'impact of religious and spiritual practices on covid-19 mitigation'
    path = 'D:\\work\\python_work\\literature-review-mainone\\'
    gscholar = gs(title, num=50)
    url = gscholar.makeQuery()
##    print(url)
    links = gscholar.getLinks(url)
    
##    print(links) 
##    print(len(links))
##
##    for i in links:
##        print(i) 
##    gscholar.openSource(links, num=7)
##    gscholar.getText(links, path=path, num=15,  savetext=True)
    titles = gscholar.getTitles(url)
##    for i in titles:
##        print(i)
    termmat = gscholar.wordFrequencies(titles)
##    print(termmat)
    distinctwords = gscholar.makeDistinctWords(title, termmat)
##    print(distinctwords)
##    gscholar.barChart(distinctwords)
##    dwtbl = gscholar.makeTables(distinctwords)
##    gscholar.boxPlot(dwtbl)
##    pathch = path + 'dwtbl.csv'
##    saveAsCSVFile(path, dwtbl)
##    twoSampleIndTTest(termmat['wordfreq'], distinctwords['freq'])
##    print(gscholar.computeIQR(dwtbl['freq']))
##    wordvec = gscholar.reshapeData(distinctwords)
    nc = 2
    clssol = gscholar.clusterAnalysis(distinctwords, nc, path, output=True)
    pathch = path + 'out.csv'
    cluswords = gscholar.wordsByCategory(pathch, 2)
    gscholar.pieForCategories(cluswords)
    
    
