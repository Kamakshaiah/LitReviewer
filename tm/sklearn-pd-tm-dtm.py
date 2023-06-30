import numpy as np
from numpy import count_nonzero

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
file = pd.read_csv('D:\\Research\papers\\finance\\auditing\\datasets\\auditing-data-sets\\big-data-fin-auditing.csv')

# importing abstracts
abstracts = file['Abstract'].tolist()

# data transformation
vec = CountVectorizer()
X = vec.fit_transform(abstracts)
df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names_out())
##print(df)

# wordcount
from wordcloud import WordCloud
from wordcloud import ImageColorGenerator
from wordcloud import STOPWORDS
import matplotlib.pyplot as plt

text = ''.join(i for i in abstracts)

##stopwords = set(STOPWORDS)
##wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
##plt.figure( figsize=(15,10))
##plt.imshow(wordcloud, interpolation='bilinear')
##plt.axis("off")
##plt.show()

def sparcity_density(datamat):
    import numpy as np
    from numpy import count_nonzero
    
    non_zero = np.count_nonzero(datamat)
    total_val = np.product(datamat.shape)
    
    spr = (total_val - non_zero) / total_val
    den = non_zero / total_val
    
    return (spr, den)



# QUERIES

from scipy import stats
import numpy as np

## query -0 : audit
# df.filter(like='audit').columns
audit_data = df.filter(like='audit')
a = (audit_data.to_numpy() == 0).mean()
print(a)
# print(sparcity_density(audit_data)[0]) # same as above i.e., a 

## query - 1 : blockchain

df.filter(like='blockchain').columns
df.filter(regex='blockchain') # docs 2, 4*, 7, 9, 10, 23,
df.filter(regex='blockchain').to_csv('blockchains.csv')

# sparsity and density
bc_df = df.filter(regex='blockchain') # sparse
a = (bc_df.to_numpy() == 0).mean() # sparcity
##print(1 - a) # density

## normality on sparse data 
bc_df_0 = df.filter(regex='blockchain')
bc_df_0.columns

x = np.concatenate((bc_df_0['blockchain'], bc_df_0['blockchains']))
k2, p = stats.normaltest(x)
##(k2, p) # (97.36764502161853, 7.192577094060026e-22) not normal

## normality on dense data
bc_df = bc_df[bc_df['blockchain'] > 0]
bc_df['total'] = bc_df['blockchain'] + bc_df['blockchains']

x = np.concatenate((bc_df['blockchain'], bc_df['blockchains']))
k2, p = stats.normaltest(x)
##(k2, p) # (12.451060883810605, 0.001978274177467941) not normal

## other tests 

stats.ttest_1samp(bc_df['total'], 0) #Ttest_1sampResult(statistic=3.4780417182012617, pvalue=0.01769589188401355) H_0: mu != 0

stats.f_oneway(bc_df['blockchain'], bc_df['blockchains']) # f one-way F_onewayResult(statistic=5.550458715596331, pvalue=0.040230871078499895) H_0: same variances 

stat, p, med, tbl = stats.median_test(bc_df['blockchain'], bc_df['blockchains'])
(stat, p) # (3.0, 0.08326451666355042) # H_0: medians are same

z, p = stats.mood(bc_df['blockchain'], bc_df['blockchains'])
##(z, p) # (-0.7600065066233908, 0.44725069558444264) H_0: same distribution with same scale parameters

## query - 2 : fraud

df.filter(like='fraud').columns
df.filter(like='fraud') # 6, 9, 11*, 12*, 16, 22, 24, 26, 30
df.filter(like='fraud').to_csv('fraud.csv')

df.filter(like='protection') # 2, 7
df.filter(like='protection').to_csv('protection.csv')

df.filter(like='risk') # 2, 6, 18, 22, 27
df.filter(like='risk').to_csv('risk.csv') 

df.filter(like='quality') # 2, 3, 6,5, 6, 12, 17, 18, 22, 29, 31
df.filter(like='quality').to_csv('quality.csv')

df.filter(like='technology') # 2, 4, 5, 7, 9, 10, 13, 16, 19, 21, 23, 27
df.filter(like='technology').to_csv('technology.csv')

df.filter(like='service') # 1, 4, 7, 10, 27, 29, 31
df.filter(like='service').to_csv('service.csv') 

df.filter(like='analysis') # 2, 5, 8, 9, 10, 11, 13, 15, 17, 19, 20, 24, 27, 29
df.filter(like='analysis').to_csv('analysis.csv')

# sparcity and density
## sparcity

tech_df = df.filter(like='technology')
df.filter(like='technology').to_csv('technology.csv')

a = (tech_df.to_numpy() == 0).mean()
# sparsity = sum((tech_df == 0).astype(int).sum())/tech_df.size # it works
##print(sparcity)

## density
sparr = tech_df.apply(pd.arrays.SparseArray)
##print(sparr.sparse.density) # 1 - sparcity

import numpy as np
from scipy.sparse import lil_matrix

arr = lil_matrix(tech_df.shape, dtype=np.float32)

##for i, col in enumerate(tech_df.columns):
##    ix = tech_df[col] != 0
##    arr[np.where(ix), i] = 1
##    print(arr)


if __name__ == '__main__':
##    # sparsity and density
##    audit_df = df.filter(regex='audit') # sparse
##    a = (audit_df.to_numpy() == 0).mean() # sparcity
##    ##print(1 - a) # density
##
##    x = np.concatenate((audit_df['audit'], audit_df['auditability'], audit_df['auditable'], audit_df['audited'], audit_df['auditees'],
##       audit_df['auditing'], audit_df['auditor'], audit_df['auditors'], audit_df['audits']))
##    k2, k2_p = stats.normaltest(x)
##    ##(k2, p) # (97.36764502161853, 7.192577094060026e-22) not normal
##
##    ## other tests 
##    audit_df = audit_df.assign(mean=audit_df.mean(axis=1))
##    t, t_p = stats.ttest_1samp(audit_df['mean'], 0) #Ttest_1sampResult(statistic=3.4780417182012617, pvalue=0.01769589188401355) H_0: mu != 0
##
##    f, f_p = stats.f_oneway(audit_df['audit'], audit_df['auditability'], audit_df['auditable'], audit_df['audited'], audit_df['auditees'],
##       audit_df['auditing'], audit_df['auditor'], audit_df['auditors'], audit_df['audits']) # f one-way F_onewayResult(statistic=5.550458715596331, pvalue=0.040230871078499895) H_0: same variances 
##
##    med_stat, med_p, med, tbl = stats.median_test(audit_df['audit'], audit_df['auditability'], audit_df['auditable'], audit_df['audited'], audit_df['auditees'],
##       audit_df['auditing'], audit_df['auditor'], audit_df['auditors'], audit_df['audits'])
##    
##
####    moods_stat, moods_p = stats.mood(audit_df['audit'], audit_df['auditability'], audit_df['auditable'], audit_df['audited'], audit_df['auditees'],
####       audit_df['auditing'], audit_df['auditor'], audit_df['auditors'], audit_df['audits'])
####    ##(z, p) # (-0.7600065066233908, 0.44725069558444264) H_0: same distribution with same scale parameters
##
##    kw_s, kw_p = stats.kruskal(audit_df['audit'], audit_df['auditability'], audit_df['auditable'], audit_df['audited'], audit_df['auditees'],
##       audit_df['auditing'], audit_df['auditor'], audit_df['auditors'], audit_df['audits'])
##    
##    tests = ['k2', 't', 'f', 'Median test', 'Kruskal']
##    statistics = [k2, t, f, med_stat, kw_s]
##    Pvalues = [k2_p, t_p, f_p, med_p]
##    
##    table = {'Test': tests, 'Statistic': statistics, 'P Value': Pvalues}
##
##    stats_table_auditing = pd.DataFrame.from_dict(table)
##    stats_table_auditing.to_csv('D:\\Research\\papers\\finance\\iimv\\stats-table-auditing.csv')
##    audit_df.plot.pie(subplots=True, legend=False, layout=(3, 3))
##    plt.show()

##    # sparsity and density
##    fraud_df = df.filter(regex='fraud') # sparse
##    a = (fraud_df.to_numpy() == 0).mean() # sparcity
##    ##print(1 - a) # density
##
##    x = np.concatenate((fraud_df['fraud'], fraud_df['frauds'], fraud_df['fraudsters'], fraud_df['fraudulent']))
##    k2, k2_p = stats.normaltest(x)
##    ##(k2, p) # (97.36764502161853, 7.192577094060026e-22) not normal
##
##    ## other tests 
##    fraud_df = fraud_df.assign(mean=fraud_df.mean(axis=1))
##    t, t_p = stats.ttest_1samp(fraud_df['mean'], 0) #Ttest_1sampResult(statistic=3.4780417182012617, pvalue=0.01769589188401355) H_0: mu != 0
##
##    f, f_p = stats.f_oneway(fraud_df['fraud'], fraud_df['frauds'], fraud_df['fraudsters'], fraud_df['fraudulent']) # f one-way F_onewayResult(statistic=5.550458715596331, pvalue=0.040230871078499895) H_0: same variances 
##
##    med_stat, med_p, med, tbl = stats.median_test(fraud_df['fraud'], fraud_df['frauds'], fraud_df['fraudsters'], fraud_df['fraudulent'])
##    
##
####    moods_stat, moods_p = stats.mood(audit_df['audit'], audit_df['auditability'], audit_df['auditable'], audit_df['audited'], audit_df['auditees'],
####       audit_df['auditing'], audit_df['auditor'], audit_df['auditors'], audit_df['audits'])
####    ##(z, p) # (-0.7600065066233908, 0.44725069558444264) H_0: same distribution with same scale parameters
##
##    kw_s, kw_p = stats.kruskal(fraud_df['fraud'], fraud_df['frauds'], fraud_df['fraudsters'], fraud_df['fraudulent'])
##    
##    tests = ['k2', 't', 'f', 'Median test', 'Kruskal']
##    statistics = [k2, t, f, med_stat, kw_s]
##    Pvalues = [k2_p, t_p, f_p, med_p, kw_p]
##    
##    table = {'Test': tests, 'Statistic': statistics, 'P Value': Pvalues}
##
##    stats_table_auditing = pd.DataFrame.from_dict(table)
##    stats_table_auditing.to_csv('D:\\Research\\papers\\finance\\iimv\\stats-table-fraud.csv')
##    
##    fraud_df.boxplot()
##    plt.show()
##
##    fraud_df.plot.pie(subplots=True, legend=False, layout=(2, 2))

### sparsity and density
##    security_df = df.filter(regex='security') # sparse
##    a = (security_df.to_numpy() == 0).mean() # sparcity
##    ##print(1 - a) # density
##
##    x = np.concatenate((security_df['cybersecurity'], security_df['security']))
##    k2, k2_p = stats.normaltest(x)
##    ##(k2, p) # (97.36764502161853, 7.192577094060026e-22) not normal
##
##    ## other tests 
##    security_df = security_df.assign(mean=security_df.mean(axis=1))
##    t, t_p = stats.ttest_1samp(security_df['mean'], 0) #Ttest_1sampResult(statistic=3.4780417182012617, pvalue=0.01769589188401355) H_0: mu != 0
##
##    f, f_p = stats.f_oneway(security_df['cybersecurity'], security_df['security']) # f one-way F_onewayResult(statistic=5.550458715596331, pvalue=0.040230871078499895) H_0: same variances 
##
##    med_stat, med_p, med, tbl = stats.median_test(security_df['cybersecurity'], security_df['security'])
##    
##
####    moods_stat, moods_p = stats.mood(audit_df['audit'], audit_df['auditability'], audit_df['auditable'], audit_df['audited'], audit_df['auditees'],
####       audit_df['auditing'], audit_df['auditor'], audit_df['auditors'], audit_df['audits'])
####    ##(z, p) # (-0.7600065066233908, 0.44725069558444264) H_0: same distribution with same scale parameters
##
##    kw_s, kw_p = stats.kruskal(security_df['cybersecurity'], security_df['security'])
##    
##    tests = ['k2', 't', 'f', 'Median test', 'Kruskal']
##    statistics = [k2, t, f, med_stat, kw_s]
##    Pvalues = [k2_p, t_p, f_p, med_p, kw_p]
##    
##    table = {'Test': tests, 'Statistic': statistics, 'P Value': Pvalues}
##
##    stats_table = pd.DataFrame.from_dict(table)
##    stats_table.to_csv('D:\\Research\\papers\\finance\\iimv\\stats-table-security.csv')
##    
##    security_df.boxplot()
##    plt.show()
##
##    security_df.plot.pie(subplots=True, legend=False, layout=(2, 2))
##    plt.show()

### sparsity and density
##    risk_df = df.filter(regex='risk') # sparse
##    a = (risk_df.to_numpy() == 0).mean() # sparcity
##    ##print(1 - a) # density
##
##    x = np.concatenate((risk_df['risk'], risk_df['risks']))
##    k2, k2_p = stats.normaltest(x)
##    ##(k2, p) # (97.36764502161853, 7.192577094060026e-22) not normal
##
##    ## other tests 
##    risk_df = risk_df.assign(mean=risk_df.mean(axis=1))
##    t, t_p = stats.ttest_1samp(risk_df['mean'], 0) #Ttest_1sampResult(statistic=3.4780417182012617, pvalue=0.01769589188401355) H_0: mu != 0
##
##    f, f_p = stats.f_oneway(risk_df['risk'], risk_df['risks']) # f one-way F_onewayResult(statistic=5.550458715596331, pvalue=0.040230871078499895) H_0: same variances 
##
##    med_stat, med_p, med, tbl = stats.median_test(risk_df['risk'], risk_df['risks'])
##    
##
####    moods_stat, moods_p = stats.mood(audit_df['audit'], audit_df['auditability'], audit_df['auditable'], audit_df['audited'], audit_df['auditees'],
####       audit_df['auditing'], audit_df['auditor'], audit_df['auditors'], audit_df['audits'])
####    ##(z, p) # (-0.7600065066233908, 0.44725069558444264) H_0: same distribution with same scale parameters
##
##    kw_s, kw_p = stats.kruskal(risk_df['risk'], risk_df['risks'])
##    
##    tests = ['k2', 't', 'f', 'Median test', 'Kruskal']
##    statistics = [k2, t, f, med_stat, kw_s]
##    Pvalues = [k2_p, t_p, f_p, med_p, kw_p]
##    
##    table = {'Test': tests, 'Statistic': statistics, 'P Value': Pvalues}
##
##    stats_table = pd.DataFrame.from_dict(table)
##    stats_table.to_csv('D:\\Research\\papers\\finance\\iimv\\stats-table-risk.csv')
##    
##    risk_df.boxplot()
##    plt.show()
##
##    risk_df.plot.pie(subplots=True, legend=False, layout=(2, 2))
##    plt.show()

##    # sparsity and density
##    quality_df = df.filter(regex='quality') # sparse
##    a = (quality_df.to_numpy() == 0).mean() # sparcity
##    ##print(1 - a) # density
##
####    x = np.concatenate((risk_df['risk'], risk_df['risks']))
##    k2, k2_p = stats.normaltest(np.array(quality_df['quality']))
##    ##(k2, p) # (97.36764502161853, 7.192577094060026e-22) not normal
##
##    ## other tests 
####    risk_df = risk_df.assign(mean=risk_df.mean(axis=1))
##    t, t_p = stats.ttest_1samp(quality_df['quality'], 0) #Ttest_1sampResult(statistic=3.4780417182012617, pvalue=0.01769589188401355) H_0: mu != 0
##
####    f, f_p = stats.f_oneway(quality_df['quality']) # f one-way F_onewayResult(statistic=5.550458715596331, pvalue=0.040230871078499895) H_0: same variances 
##
####    med_stat, med_p, med, tbl = stats.median_test(risk_df['risk'], risk_df['risks'])
##    
##
####    moods_stat, moods_p = stats.mood(audit_df['audit'], audit_df['auditability'], audit_df['auditable'], audit_df['audited'], audit_df['auditees'],
####       audit_df['auditing'], audit_df['auditor'], audit_df['auditors'], audit_df['audits'])
####    ##(z, p) # (-0.7600065066233908, 0.44725069558444264) H_0: same distribution with same scale parameters
##
####    kw_s, kw_p = stats.kruskal(quality_df['quality'])
##    
##    tests = ['k2', 't']
##    statistics = [k2, t]
##    Pvalues = [k2_p, t_p]
##    
##    table = {'Test': tests, 'Statistic': statistics, 'P Value': Pvalues}
##
##    stats_table = pd.DataFrame.from_dict(table)
##    stats_table.to_csv('D:\\Research\\papers\\finance\\iimv\\stats-table-quality.csv')
##    
##    quality_df.boxplot()
##    plt.show()
##
##    quality_df.plot.pie(subplots=True, legend=False, layout=(2, 2))
##    plt.show()

# sparsity and density
    tech_df = df.filter(regex='technology') # sparse
    a = (tech_df.to_numpy() == 0).mean() # sparcity
    ##print(1 - a) # density

##    x = np.concatenate((risk_df['risk'], risk_df['risks']))
    k2, k2_p = stats.normaltest(np.array(tech_df['technology']))
    ##(k2, p) # (97.36764502161853, 7.192577094060026e-22) not normal

    ## other tests 
##    risk_df = risk_df.assign(mean=risk_df.mean(axis=1))
    t, t_p = stats.ttest_1samp(tech_df['technology'], 0) #Ttest_1sampResult(statistic=3.4780417182012617, pvalue=0.01769589188401355) H_0: mu != 0

##    f, f_p = stats.f_oneway(quality_df['quality']) # f one-way F_onewayResult(statistic=5.550458715596331, pvalue=0.040230871078499895) H_0: same variances 

##    med_stat, med_p, med, tbl = stats.median_test(risk_df['risk'], risk_df['risks'])
    

##    moods_stat, moods_p = stats.mood(audit_df['audit'], audit_df['auditability'], audit_df['auditable'], audit_df['audited'], audit_df['auditees'],
##       audit_df['auditing'], audit_df['auditor'], audit_df['auditors'], audit_df['audits'])
##    ##(z, p) # (-0.7600065066233908, 0.44725069558444264) H_0: same distribution with same scale parameters

##    kw_s, kw_p = stats.kruskal(quality_df['quality'])

    chi2, chi_p = stats.chisquare(tech_df['technology'])
    
    tests = ['k2', 't']
    statistics = [k2, t]
    Pvalues = [k2_p, t_p]
    
    table = {'Test': tests, 'Statistic': statistics, 'P Value': Pvalues}

    stats_table = pd.DataFrame.from_dict(table)
    stats_table.to_csv('D:\\Research\\papers\\finance\\iimv\\stats-table-technology.csv')
    
    tech_df.boxplot()
    plt.show()

    tech_df.plot.pie(subplots=True)
    plt.show()
