# Univariate
def computeIQR(v1):
    ''' Computes Inter Quartile Range (IQR) for word's frequency - input should be univariate distribution '''

    from scipy import stats
    stat = stats.iqr(v1, axis=0)
    return stat

# normality test
def ShapiroTest(v):
    ''' Performs Shapiros Test of normality
        v: univariate data variable. '''
    from scipy.stats import shapiro
    stat, p = shapiro(v)
    return {'stat': stat, 'p': p}

def DAgostinosK2Test(v):
    ''' Performs D’Agostino’s K^2 Test of normality
        v: univariate data variable. '''
    from scipy.stats import normaltest
    stat, p = normaltest(v)
    return {'stat': stat, 'p': p}

def AndersonDarlingTest(v, dist='norm'):
    ''' Performs Anderson-Darling Test of normality for a given distribution
        dist : {‘norm’, ‘expon’, ‘logistic’, ‘gumbel’, ‘gumbel_l’, ‘gumbel_r’, ‘extreme1’}, optional for 'norm'. '''
    from scipy.stats import anderson
    stat, p = anderson(v, dist=dist)
    return {'stat': stat, 'p': p}

def oneSampleTTest(v, mean=0):
    ''' Performs one sample T Test for given word frequency vector.
        v: univariate data variable. '''
    from scipy import stats
    stat, p = stats.ttest_1samp(v, mean)
    return {'stat': stat, 'p-value': p}

def chiSqTest(v):

    ''' univariate chisq test '''
    from scipy.stats import chisquare
    stat, p = chisquare([16, 18, 16, 14, 12, 12])
    return {'stat': stat, 'p': p}

def runsTest(l, l_median):
    ''' Performs Run' Test '''

    import math
    import scipy.stats
    
    runs, n1, n2 = 0, 0, 0
      
    # Checking for start of new run
    for i in range(len(l)):
          
        # no. of runs
        if (l[i] >= l_median and l[i-1] < l_median) or \
                (l[i] < l_median and l[i-1] >= l_median):
            runs += 1  
          
        # no. of positive values
        if(l[i]) >= l_median:
            n1 += 1   
          
        # no. of negative values
        else:
            n2 += 1   
  
    runs_exp = ((2*n1*n2)/(n1+n2))+1
    stan_dev = math.sqrt((2*n1*n2*(2*n1*n2-n1-n2))/ \
                       (((n1+n2)**2)*(n1+n2-1)))
  
    z = (runs-runs_exp)/stan_dev
    p = scipy.stats.norm.sf(abs(z))
    return (z, p)

# Dependencies 
    
def twoSampleIndTTest(v1, v2):
    ''' Performs two sample ind. T Test for two different samples - for termmat and distinctwords distributions '''
    
    from scipy import stats
    stat, p = stats.ttest_ind(v1, v2)
    return {'stat': stat, 'p-value': p}

def MoodsTest(v1, v2, ties=True):
    ''' Performs moods test for two different serach phrases '''

    from scipy.stats import median_test
 
    if ties:
        try:
            g, p, med, tbl = median_test(v1, v2, lambda_="log-likelihood", ties="above")
        except Exception as e:
            print(e, " occured! Test doesn't work")
        return {'g': g, 'p': p, 'med': med, 'tbl': tbl}
    
    else:
        try:
            g, p, med, tbl = median_test(v1, v2, lambda_="log-likelihood")
        except Exception as e:
            print(e, " occured! Test doesn't work")
            
        return {'g': g, 'p': p, 'med': med, 'tbl': tbl}

def chiSqContingencyTest(tbl):
    ''' performs chi2 contingency test on table returned from mood's test '''
    import scipy.stats as stats
    stat, p = stats.chi2_contingency(tbl)
    return {'stat': stat, 'p-value': p}

def FishersExactTest(tbl):
    ''' performs chi2 contingency test on table returned from mood's test '''
    import scipy.stats as stats
    stat, p = stats.fisher_exact(tbl)
    return {'stat': stat, 'p-value': p}
    
def BartlettTest(v1, v2):
        ''' Useful for performing bartletts test of spherecity. Input data should be numerical (may be word frequencies) '''
        from scipy.stats import bartlett

        stat, p = bartlett(v1, v2)
        
        return {'stat': stat, 'p-value': p}

def oneWayANOVA(v1, v2):
    ''' Tests whether the means of two or more independent samples are significantly different. '''
    from scipy.stats import f_oneway
    stat, p = f_oneway(v1, v2)
    return {'stat': stat, 'p-value': p}    

# Non-parametric tests

def MannWhitneyUTest(v1, v2):
    ''' Tests whether the distributions of two independent samples are equal or not. '''
    from scipy.stats import mannwhitneyu
    stat, p = mannwhitneyu(v1, v2)
    return {'stat': stat, 'p-value': p}

def WilcoxonSignedRankTest(v1, v2):
    ''' Tests whether the distributions of two independent samples are equal or not.
        Caution: Samples must have the same length '''
    
    from scipy.stats import wilcoxon
    stat, p = wilcoxon(v1, v2)
    return {'stat': stat, 'p-value': p}

def KruskalWallisHTest(v1, v2):
    ''' Tests whether the distributions of two independent samples are equal or not. '''
    from scipy.stats import kruskal
    stat, p = kruskal(v1, v2)
    return {'stat': stat, 'p-value': p}

def FriedmanTest(v1, v2, v3):
    ''' Tests whether the distributions of two independent samples are equal or not. '''
    from scipy.stats import friedmanchisquare
    stat, p = friedmanchisquare(v1, v2, v3)
    return {'stat': stat, 'p-value': p}

# Associations
def PearsonsCorrelationSigTest(v1, v2):
    ''' Provides Karl Pearson's r and p value for significant test
        Caution: Samples must have the same length '''
    from scipy.stats import pearsonr
    stat, p = pearsonr(v1, v2)
    return {'stat': stat, 'p': p}

def SpearmansRankCorrelation(v1, v2):
    ''' Provides Spearman’s Rho and p value for significant test
        Caution: Samples must have the same length '''
    
    from scipy.stats import spearmanr
    stat, p = spearmanr(v1, v2)
    return {'stat': stat, 'p': p}

def KendallsRankCorrelation(v1, v2):
    ''' Provides Kendalls Tau and p value for significant test
        Caution: Samples must have the same length '''

    from scipy.stats import kendalltau
    stat, p = kendalltau(v1, v2)
    return {'stat': stat, 'p': p}

# Plots

def barChart(df):

    ''' Creates barchart for termmat (TM)/distinct words matrix (DWM)
        params: df [pandas dataframe] '''

    import matplotlib.pyplot as plt
    columns = list(df.keys())
    plt.bar(df[columns[0]], df[columns[1]])
    plt.show()

def pieChart(df):

    ''' Create Pie chart TM/DWM
        params: df [pandas df ]'''

    import matplotlib.pyplot as plt
    keys = list(df.keys())
    plt.pie(df[keys[1]], labels = df[keys[0]])
    
    plt.show()
    
def boxPlot(df):
    
    ''' Boxplot for data vector(s); supports only pandas dataframe '''
    import matplotlib.pyplot as plt
    
    df.boxplot()
    plt.show()


def reshapeData(data):
    ''' Reshapes input data into required format for cluster analysis '''
    import numpy as np

    idx = list(data.keys())[1]

    out = np.reshape(data[idx], (1, -1)).T

    return out

def uniVarClusterAnalysis(data, nc=1, plot=False):
    ''' Performs cluster analysis on input data.
        data - a pandas data frame,
        nc - num of clusters,
        '''

    import matplotlib.pyplot as plt
    from sklearn import cluster
    from sklearn import metrics
    import pandas as pd
    import numpy as np
    import os
    
    idx0 = data.columns[0]
    idx1 = data.columns[1]

    req_data = np.reshape(data.loc[:,idx1].values, (-1, 1)).T
           
    kmeans = cluster.KMeans(n_clusters=nc).fit(req_data)
    
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    score = kmeans.score(req_data)

    try:
        silhouette_score = metrics.silhouette_score(req_data, labels, metric='euclidean')
    except Exception as e:
        silhouette_score = "Can't compute silhouette_score!"

##    if plot==True:
##        plt.scatter(data[idx0], data[idx1], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
##        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
##        plt.show()
        
    out = {'labels': labels, 'centroids': centroids, 'score': score, 'silhouette_score': silhouette_score}
    return out

def biVariateClusterAnalysis(df, plot=True):
    ''' Performs cluster analysis for bivariate data.
        df: pandas data frame. '''
    from sklearn.cluster import KMeans
    from sklearn import metrics
    import matplotlib.pyplot as plt

    idx0 = list(df.columns)[0]
    idx1 = list(df.columns)[1]
    
    kmeans = KMeans(n_clusters=3).fit(df)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    score = kmeans.score(df)
    silhouette_score = metrics.silhouette_score(df, labels, metric='euclidean')

    if plot==True:
        plt.scatter(df[idx0], df[idx1], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
        plt.show()

    out = {'labels': labels, 'centroids': centroids, 'score': score, 'silhouette_score': silhouette_score}
    return out

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

def pieForCategories(cats):
    ''' Pie chart for categories. Requires cluster wise words (such as created by the method 'wordsByCategory()' '''

    import matplotlib.pyplot as plt

    vals = []
    for i in cats.values():
        vals.append(len(i))

    plt.pie(vals, labels=vals)
    plt.show()

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

def convertStatsIntoCSVFile(path, *args, **kwargs):

    ''' converts statistical results (outputs) into a CSV file. 'd' must be a dictionary of dictionaries. '''

    import pandas as pd
    import os
    
    out = dict()
    for d in args:
        out.update(d)

    ddf = pd.DataFrame.from_dict(out, orient='index').reset_index()
    
    if os.path.exists(os.path.join(path, 'output')):
        path = os.path.join(path, 'output')
        if  'file_name' in kwargs.keys():
            file_path = os.path.join(path, kwargs['file_name'])
            ddf.to_csv(file_path) 
    else:
        path = os.mkdir(path, 'output')
        if  'file_name' in kwargs.keys():
            file_path = os.path.join(path, kwargs['file_name'])
            ddf.to_csv(file_path) 
            
if __name__ == '__main__':
    import numpy as np
    v1 = np.random.randint(1, 10, 30)
    v2 = np.random.randint(1, 10, 30)
    v3 = np.random.randint(1, 10, 30)
##    print(bartlettTest(v1, v2))
##    print(computeIQR(v1))
    mout = MoodsTest(v1, v2)
##    print(oneSampleTTest(v1))
##    import pandas as pd
##    data = {'v1': v1, 'v2':v2}
##    df = pd.DataFrame(data)
##    print(biVariateClusterAnalysis(df, plot=True))
##    print(chiSqContingencyTest(mout['tbl']))
##    print(ANOVA(v1, v2))
##    print(FriedmanTest(v1, v2, v3))
##    print(PearsonsCorrelationSigTest(v1, v2))
##    print(SpearmansRankCorrelation(v1, v2))
    print(WilcoxonSignedRankTest(v1, v2))
