import matplotlib.pyplot as plt

def importAbstracts(path):

    ''' Imports abstracts from a given @path '''

    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer
    file = pd.read_csv(str(path))

    # importing abstracts
    abstracts = file['Abstract'].tolist()

    # data transformation
    vec = CountVectorizer()
    X = vec.fit_transform(abstracts)
    df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
    return df

##df[~(df == 0). all(axis=1)]

def makeConjugates(df, string):
    ''' Makes a data frame with row sums for conjugates '''

    data = df.filter(like=str(string))
    return data

def makeRowSums(df, sparse=True):

    ''' Creates non-zero data frame for given conjugates '''
    rowsums = df.sum(axis=1)
    if sparse == True:
        return rowsums
    else:
        return rowsums[~(rowsums == 0)]

def subsetData(df, exp, value):
    ''' Filter data frame for given expression ==, <=, >=, <, >, != '''
    if exp == 'eq':
        return df[(df == value).all(axis=1)]
    elif exp == 'le':
        return df[(df <= value).all(axis=1)]
    elif exp == 'ge':
        return df[(df >= value).all(axis=1)]
    elif exp == 'gt':
        return df[(df > value).all(axis=1)]
    elif exp == 'lt':
        return df[(df < value).all(axis=1)]
    elif exp == 'ne':
        return df[~(df == value).all(axis=1)]
    else:
        print('Check your arguents')
    
def saveData(df, filename):
    ''' Saves the data frame to the given input path argument '''
    import os
    
    if os.path.isdir('output'):
        path = os.path.join('output', filename)
        df.to_csv(path)
    else:
        os.mkdir('output')
        path = os.path.join('output', filename)
        df.to_csv(path)

def sparcityDensity(df):

    ''' Calculates both sparsity and density of the data set '''

    sparsity = (df.to_numpy() == 0).mean()
    return (sparsity, 1-sparsity)

def chiSquareFitnessTest(*args):
    ''' Performs chisquare test of fitness on univariate data '''

    from scipy import stats
    chi2, chi_p = stats.chisquare(*args)

    return (chi2, chi_p)

def normalityTest(df):

    ''' Performs D'Augustinos, Person normality test for given pandas data frame '''

    from scipy import stats
    
    return stats.normaltest(df)

def oneSampleTTest(df, am=0):

    ''' Performs one sample T Test test for given pandas data frame '''

    from scipy import stats
    return stats.ttest_1samp(df, am)

def oneWayFTest(*args):

    ''' Performs one way F Test test for given pandas data frame '''
    from scipy import stats
##    for arg in args:
##        print(arg) 
    return stats.f_oneway(*args)

def medianTest(*args):

    ''' Performs bi/multi variate median test for given pandas data frame '''
    from scipy import stats
    stat, p, med, tbl = stats.median_test(*args)

    return (stat, p)

def moodsTest(*args):

    ''' Performs bi/multi variate moods test for given pandas data frame '''
    from scipy import stats
    z, p = stats.mood(*args)
    return (z, p)

def kruskalWallisTest(*args):

    ''' Performs bi/multi variate kruskal wallis test for given pandas data frame '''
    from scipy import stats
    kw_s, kw_p = stats.kruskal(*args)
    return (kw_s, kw_p)

def concatenateDataSets(*args):
    ''' Joins multivariate data sets into a single/composite data set '''
    import pandas as pd
    frames = [*args]
    df = pd.concat(frames, axis=1)
    return df

def makeTargetVariable(df1, df2, names = [1, 2, 3]):
    ''' Makes target variable for principal component analysis '''

    import pandas as pd
    
    maxs1 = df1.max(axis=1)
    maxs2 = df2.max(axis=1)
    target = []
    for i, j in zip(maxs1, maxs2):
        if i > j:
            target.append(names[0])
        elif i < j:
            target.append(names[1])
        else:
            target.append(names[2])

    return pd.Series(target, name= 'target')
                    
def PCA(data, target):

    ''' https://builtin.com/machine-learning/pca-in-python
        https://www.datacamp.com/tutorial/principal-component-analysis-in-python
    '''

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import pandas as pd

    features = data.columns
    x = data.loc[:, features].values
    y = target.values
    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)

    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, target], axis = 1)

    return finalDf

def plotPCA(df, targets):

    ''' Makes visual for PCA 2 components solution '''
    
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Principal Component 1', fontsize = 10)
    ax.set_ylabel('Principal Component 2', fontsize = 10)
    ax.set_title('2 component PCA', fontsize = 15)
    targets = targets
    colors = ['r', 'g', 'b']

    for target, color in zip(targets,colors):
        indicesToKeep = df['target'] == target
        ax.scatter(df.loc[indicesToKeep, 'principal component 1']
                   , df.loc[indicesToKeep, 'principal component 2']
                   , c = color
                   , s = 50)
    ax.legend(targets); ax.grid()
    plt.show()
    
    
if __name__ == '__main__':
    
    # IMPORT DATA
    
    abstracts = importAbstracts('D:\Research\PAPERS\covid19\policy innovation\scopus.csv')
##    print(abstracts.head())
##    print(len(abstracts.columns))
##    print(abstracts[['technology', 'innovation']])
    plcy_cjgts = makeConjugates(abstracts, 'policy')
##    plcy_df = makeRowSums(plcy_cjgts)
##    print(plcy_df)
##    print(subsetData(plcy_cjgts, 'ne', 0))

    # PREPARE DATA
    
    inno_cjgts = makeConjugates(abstracts, 'innovation')
##    inno_df = makeRowSums(inno_cjgts)
##    print(subsetData(inno_cjgts, 'ne', 0))
##    saveData(inno_cjgts, 'out.csv')

    # SPARSITY & DENSITY
    
##    print(sparcityDensity(inno_df))

    # UNIVARIATE ANALYSIS
##    print(chiSquareFitnessTest(inno_cjgts))
##    print(normalityTest(inno_cjgts))
##    print(oneSampleTTest(inno_cjgts))

    # BIVARIATE ANALYSIS
##    print(oneWayFTest(inno_cjgts['innovation'], inno_cjgts['innovationand'], inno_cjgts['innovations']))
##    print(medianTest(inno_cjgts['innovation'], inno_cjgts['innovationand'], inno_cjgts['innovations']))
##    print(moodsTest(inno_cjgts['innovation'], inno_cjgts['innovationand']))
##    print(kruskalWallisTest(inno_cjgts['innovation'], inno_cjgts['innovationand'], inno_cjgts['innovations']))
    
    # MULTIVARIATE ANALYSIS

    data = concatenateDataSets(plcy_cjgts, inno_cjgts)
    target = makeTargetVariable(plcy_cjgts, inno_cjgts, names = ['policy', 'innovation', 'other'])

##    print(data.columns)
##    print(data.head())

    df = PCA(data, target)
##    print(df.head) 

    plotPCA(df, targets = ['policy', 'innovation', 'other'])
