class gSearch:

    ''' Performs google search and retrieves words
        init method: title [search query]; num_links=[integer]; duration: [no. of years] '''

    def __init__(self, query, num_links=10, duration=1):

        self.query = query
        self.num_links = num_links
        self.duration = duration
        self.google_url = f"https://www.google.com/search?q={self.query}&num={str(self.num_links)}&as_qdr=y{self.duration}"

    def getUrl(self):

        ''' Returns url for google search
            params: None '''

        return self.google_url

    def getTitlesAndLinks(self):

        ''' Retrieves titles and links for search title
            params: None '''

        import requests
        
        from my_fake_useragent import UserAgent
        from bs4 import BeautifulSoup

        ua = UserAgent()

        r = requests.get(self.google_url, {"User-Agent": ua.random})

        soup = BeautifulSoup(r.content, 'lxml')

        titles = soup.find_all('h3')

        titlesout = []
        for t in titles:
            titlesout.append(t.text)

        alinks = soup.findAll('a')
        alinksout = []
        
        for l in alinks:
            alinksout.append(l.get('href'))

        links = []

        for l in alinksout:
            if not 'google' in l and '/url?q=' in l:
                links.append(l)
        
        return {'titles': titlesout, 'links': links}

    @staticmethod
    def getLinks(links):

        ''' Removes '/url?q=' in the links
            params: None '''

        linksrepl = []
        linksget = []
        linksfinal = []
        
        for l in links:
            linksrepl.append(l.replace('/url?q=', ''))
            
        for l in linksrepl:
            linksget.append(l.split('%')[0])

        for l in linksget:
            linksfinal.append(l.split('&')[0])
    
        finallinks = set(linksfinal)
        return list(finallinks) 
            

    @staticmethod
    def openLink(links, **kwargs):
        
        ''' Opens the given link inside browser tab for two arguments (1) links (2) num/allsources
            params: links [created by getTitlesAndLinks()]
                    num [integer]
                    allsources [boolean:True/False]'''

        import webbrowser as wb
        
        if 'num' in kwargs.keys():
            arg1 = kwargs['num']
            wb.open(links[arg1])
        elif 'allsources' in kwargs.keys():
            for i in links:
                wb.open(i)
                
    @staticmethod
    def saveTextFromLinks(links, path, link_num=0, name='name.txt'):

        ''' Scrapes text from given links
            params: for link_num=0, name='name.txt' and saves the text in 'path' for further analysis '''
        
        import os
        import requests
        from bs4 import BeautifulSoup
        
        from my_fake_useragent import UserAgent
        ua = UserAgent()
        
        ps = requests.get(links[link_num], {"User-Agent": ua.random})

        soup = BeautifulSoup(ps.content, 'lxml')
        ptags = soup.find_all('p')

        text = ''
        
        for p in ptags:
            text += p.text + '\n'
        

        if os.path.exists('output'):
            path = path + '\\output\\' + name
        else:
            os.mkdir('output')
            path = path + '\\output\\' + name

        f = open(path,'w+', encoding='utf-8')
        f.write(text)
        f.close() 

    @staticmethod
    def downloadPDF(link, path, name = 'file'):
        ''' Downloads pdf files
            params: link [url]
                    path [variable] '''

        import requests

        pathch = os.path.join(path, f'output\\{name}.pdf')
        cont = requests.get(link)
        with open(pathch, 'wb') as file:
            file.write(cont.content)

    @staticmethod
    def saveMultipleFiles(links, path, list_for_d=None):
        for i in list_for_d:
            saveTextFromLinks(links, path, link_num= i, name=f'{i}.txt')
    
    
if __name__ == '__main__':
    import os
    path = os.getcwd()
    print(path) 
    
    title = 'impact of religious practice on Covid-19 mitigation'
    num_links = 10
    duration = 1
    
    gs = gSearch(query = title, num_links=num_links, duration=duration)

    print(gs.getUrl())
    titles = gs.getTitlesAndLinks()['titles']
    links = gs.getTitlesAndLinks()['links']

    linksch = gs.getLinks(links)
    print(len(linksch))

    for i in linksch:
        print(linksch.index(i), i)

##    gs.openLink(linksch, num = 0)
##    gs.saveTextFromLinks(linksch, path, link_num=0, name='file.txt')
##    gs.downloadPDF(linksch[1], path)
##        
##    list_for_d = [0, 2, 3, 4, 5, 7]
##    for i in list_for_d:
##        gs.saveTextFromLinks(linksch, path, link_num= i, name=f'{i}.txt')
        
    for i in range(10):
        gs.saveTextFromLinks(linksch, path, link_num= i, name=f'{i}.txt')
