import os
path = os.getcwd()
query = "factors affecting WiFi usage in urban and rural areas in India"
start = 10

def getTitles(query, start=start):
    
    from gscholar import gScholar
    gs = gScholar(q=query, start=start)

    url = gs.makeQuery()
    links = gs.getLinks(url)
    titles = gs.getTitles(url)
    print(url)

    return titles

def analyzeTitles():
    
    from litutilities import textToWords, convertCountersIntoWordFreq, saveAsCSVFile
    from literstat import computeIQR, oneSampleTTest, chiSqTest, ShapiroTest, DAgostinosK2Test, uniVarClusterAnalysis
    
    out = textToWords(titles)
    dwdf = convertCountersIntoWordFreq(out[0])
    saveAsCSVFile(dwdf, path) # this saves a file with name 'file.csv'

    ##pieChart(dwdf)
    ##boxPlot(dwtbl)

    res1 = computeIQR(dwdf['freq'])
    res2 = oneSampleTTest(dwdf['freq'])
    res3 = chiSqTest(dwdf['freq'])
    res4 = ShapiroTest(dwdf['freq'])
    res5 = DAgostinosK2Test(dwdf['freq'])
    res6 = uniVarClusterAnalysis(dwdf, plot=False)

    convertStatsIntoCSVFile(path, res1, res2, res3, res4, res5, res6, file_name='out.csv')

    return (dwdf, {res1, res2, res3, res4, res5, res6})

if __name__ == '__main__':
    titles = getTitles(query)
    analyzeTitles()
