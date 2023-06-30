# path settings 

import os
path = os.getcwd()
outpath = os.path.join(path, 'output')
dir_path = outpath

# gsearch

def getData():
    from gsearch import gSearch
    title = 'impact of healthcare capacity on covid-19 mitigation'
    num_links = 10
    duration = 1
    
    gs = gSearch(query = title, num_links=num_links, duration=duration)

    print(gs.getUrl())

    titles = gs.getTitlesAndLinks()['titles']
    links = gs.getTitlesAndLinks()['links']

    linksch = gs.getLinks(links)

    for i in linksch:
        print(linksch.index(i), i)

    for i in range(len(linksch)):
            gs.saveTextFromLinks(linksch, path, link_num= i, name=f'{i}.txt')

# text mining 

def analyzeData():
    from litutilities import appendText, cleanData, listOfWordsWithFreqs, makeTables, saveAsCSVFile
    from literstat import computeIQR, oneSampleTTest, uniVarClusterAnalysis 
    
    appendText(outpath)

    file_path = os.path.join(outpath, 'textfile.txt')
    data = cleanData(file_path)
    wf = listOfWordsWithFreqs(file_path, print_words = True, nw = 10)
    wftbl = makeTables(wf)
    saveAsCSVFile(wftbl, path, name= 'file.csv')
    print(computeIQR(wftbl['freq']))
    print(oneSampleTTest(wftbl['freq']))
    print(uniVarClusterAnalysis(wftbl))


if __name__ == '__main__':
    print(path) 
    getData()
    analyzeData()
##    print(len(linksch))
    

##    print(path)
##    print(outpath)
##    print(data.keys())
##    print(type(wf)) 
##    print(wftbl.sort_values(by='freq').head())
##    print(computeIQR(wftbl['freq']))
##    print(oneSampleTTest(wftbl['freq']))
##    print(uniVarClusterAnalysis(wftbl))
