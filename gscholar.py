class gScholar:

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
          
    def __init__(self, q, hl='en', start=0, startyear=0, num=20):
        self.q = q
        self.start=start
        self.hl = hl
        self.as_ylo=startyear
        self.num=num

    def makeQuery(self):
        ''' Create the url for google sholar search string also called query
            params: None

            '''
        
        return 'https://scholar.google.com/scholar?'+'hl='+self.hl+'&'+'q='+self.q+'&'+'as_ylo='+str(self.as_ylo)+'&'+'start='+str(self.start)+'&'+'num='+str(self.num)
    

    def getLinks(self, url):

        ''' retrieves links for search phrase
            params: url (created by makeQuery() method) '''
        
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

    def getTitles(self, url):
        ''' Retrieves titles
            params: url creted by makeQuery() method '''

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


    def openLink(self, links, **kwargs):

        ''' Opens the given link
            params: links (created by the method getLinks(); num (an integer); allsources (=True)'''

        import webbrowser as wb
        
        if 'num' in kwargs.keys():
            arg1 = kwargs['num']
            wb.open(links[arg1])
        elif 'allsources' in kwargs.keys():
            for i in links:
                wb.open(i)

    def getText(self, links, path, num, savetext=False):
        ''' Get abstract for given page source. No support for PDF files \
            params: links - created by the method getLinks(); path (variable); num (integer)
                    if 'savetext = True; the function saves a file with current datetime as file name in the directory 'output'
                    inside the project directory
        '''

        from bs4 import BeautifulSoup
        import requests
        import random
        import os
        import datetime

        from my_fake_useragent import UserAgent
        ua = UserAgent()
        headers = {"User-Agent": ua.random}

        ps = requests.get(links[num], {"User-Agent": ua.random})

        soup = BeautifulSoup(ps.content, 'lxml')
        ptags = soup.find_all('p')

        abs = ['abstract', 'Abstract', 'ABSTRACT']

        text = ''
        
        for p in ptags:
            print(p.text)

        if savetext==True:
            for p in ptags:
                text += p.text + '\n'

            path = os.path.join(path, 'output')
            if not os.path.exists(path):
                os.makedirs(path)

            file_name = os.path.join(path, datetime.datetime.now().time().strftime("%H-%M-%S"))

            path = file_name + '.txt'

            f = open(path,'w+')
            f.write(text)
            f.close()

if __name__ == "__main__":
    import os
    query = "factors affecting Wi-Fi usage in urban and rural areas in India"
    path = os.getcwd()
##    path = os.path.join(path, 'output')

    gs = gScholar(query, num=300)
    url = gs.makeQuery()
    print(url)
    links = gs.getLinks(url)
    titles = gs.getTitles(url)
    
    
##    for i in links:
##        print(i)
                    
##    gs.getText(links, path, num=0, savetext=True)
    
##if __name__ == '__main__':
##    title = 'virtual reality applications in healthcare'
##    path = 'D:\\Work\\Python\\Scripts\\literature-review-mainone\\'
##    gs = gScholar(title, startyear=2020, num=300)
##    url = gs.makeQuery()
##    print(url)
##    links = gs.getLinks(url)
##    
##    print(links) 
##    print(len(links))
##
##    for i in links:
##        print(i) 
##    gs.openLink(links, num=7)
##    gs.getText(links, path=path, num=15,  savetext=True)
    for i in titles:
        print(i)
    from litutilities import *
##    termmat = wordFrequencies(titles)
    out = textToWords(titles) 
##    print(termmat)
##    ttbl = makeTables(termmat)
    dwdf = convertCountersIntoWordFreq(out[0])
##    pathch = path + '\\termmat.csv'
    saveAsCSVFile(ttbl, path) # this saves a file with name 'file.csv'
##    distinctwords = makeDistinctWords(title, termmat)
##    print(distinctwords)

    from literstat import *
##    barChart(dwdf])
    pieChart(dwdf)
##    dwtbl = makeTables(distinctwords)
    boxPlot(dwtbl)
##    pathch = path + '\\dwtbl.csv'
##    saveAsCSVFile(dwtbl, path)
##    twoSampleIndTTest(termmat['wordfreq'], distinctwords['freq']) # this test can be used for two different searches
    
    print(computeIQR(dwdf['freq']))
##    wordvec = reshapeData(distinctwords)
##    nc = 2
##    clssol = clusterAnalysis(distinctwords, path, nc=3, output=True) # saves the output 
    print(uniVarClusterAnalysis(dwtbl, nc=1))
##    print(clssol) 
    pathch = path + 'out.csv'
    cluswords = wordsByCategory(pathch, 2) # reads data from fs
    gs.pieForCategories(cluswords)
    print(gs.crosstabFromWordsMatrix(pathch, output=True, norm=True)) # reads data from fs and then makes cross tab
    
    
