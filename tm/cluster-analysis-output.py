Python 3.10.6 (tags/v3.10.6:9c7b4bd, Aug  1 2022, 21:53:49) [MSC v.1932 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.

== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =

== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =

== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =
Traceback (most recent call last):
  File "D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py", line 12, in <module>
    data = pd.read_csv('D:\Research\papers\finance\iimv\datasets\cluster-anal\audit-main-dataset.csv')
  File "C:\Program Files\Python310\lib\site-packages\pandas\util\_decorators.py", line 311, in wrapper
    return func(*args, **kwargs)
  File "C:\Program Files\Python310\lib\site-packages\pandas\io\parsers\readers.py", line 680, in read_csv
    return _read(filepath_or_buffer, kwds)
  File "C:\Program Files\Python310\lib\site-packages\pandas\io\parsers\readers.py", line 575, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
  File "C:\Program Files\Python310\lib\site-packages\pandas\io\parsers\readers.py", line 934, in __init__
    self._engine = self._make_engine(f, self.engine)
  File "C:\Program Files\Python310\lib\site-packages\pandas\io\parsers\readers.py", line 1218, in _make_engine
    self.handles = get_handle(  # type: ignore[call-overload]
  File "C:\Program Files\Python310\lib\site-packages\pandas\io\common.py", line 786, in get_handle
    handle = open(
OSError: [Errno 22] Invalid argument: 'D:\\Research\\papers\x0cinance\\iimv\\datasets\\cluster-anal\x07udit-main-dataset.csv'

== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =

== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =
   Unnamed: 0  blockchain  blockchains
0           0           0            0
1           1           0            0
2           2           3            4
3           3           0            0
4           4          11            0

== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =
   Unnamed: 0  audit  auditability  ...  service  services  technology
0           0      0             0  ...        0         0           0
1           1      0             0  ...        3         1           0
2           2      1             0  ...        0         0           1
3           3      0             0  ...        0         0           0
4           4      0             1  ...        7         0           5

[5 rows x 25 columns]

== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =
Traceback (most recent call last):
  File "D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py", line 16, in <module>
    print(data.names)
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\generic.py", line 5575, in __getattr__
    return object.__getattribute__(self, name)
AttributeError: 'DataFrame' object has no attribute 'names'. Did you mean: 'axes'?

== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =
Index(['Unnamed: 0', 'audit', 'auditability', 'auditable', 'audited',
       'auditees', 'auditing', 'auditor', 'auditors', 'audits', 'blockchain',
       'blockchains', 'fraud', 'frauds', 'fraudsters', 'fraudulent',
       'protection', 'quality', 'risk', 'risks', 'cybersecurity', 'security',
       'service', 'services', 'technology'],
      dtype='object')

== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =

== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =
<class 'pandas.core.indexes.base.Index'>

== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =
25

== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =
Index(['audit', 'auditability', 'auditable', 'audited', 'auditees', 'auditing',
       'auditor', 'auditors', 'audits', 'blockchain', 'blockchains', 'fraud',
       'frauds', 'fraudsters', 'fraudulent', 'protection', 'quality', 'risk',
       'risks', 'cybersecurity', 'security', 'service', 'services',
       'technology'],
      dtype='object')
24

== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =
Index(['audit', 'auditability', 'auditable', 'audited', 'auditees', 'auditing',
       'auditor', 'auditors', 'audits', 'blockchain', 'blockchains', 'fraud',
       'frauds', 'fraudsters', 'fraudulent', 'protection', 'quality', 'risk',
       'risks', 'cybersecurity', 'security', 'service', 'services',
       'technology'],
      dtype='object')
24
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

========================================================== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =========================================================
Index(['audit', 'auditability', 'auditable', 'audited', 'auditees', 'auditing',
       'auditor', 'auditors', 'audits', 'blockchain', 'blockchains', 'fraud',
       'frauds', 'fraudsters', 'fraudulent', 'protection', 'quality', 'risk',
       'risks', 'cybersecurity', 'security', 'service', 'services',
       'technology'],
      dtype='object')
24
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

========================================================== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =========================================================
Traceback (most recent call last):
  File "D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py", line 23, in <module>
    kmeans = KMeans(init='k-means++', n_clusters = c, n_init = 100, random_state=0).fit(normalized_df)
NameError: name 'normalized_df' is not defined. Did you mean: 'normalize'?

========================================================== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =========================================================
      audit  auditability  auditable  ...   service  services  technology
0  0.000000           0.0        0.0  ...  0.000000      0.00         0.0
1  0.000000           0.0        0.0  ...  0.428571      0.25         0.0
2  0.055556           0.0        0.0  ...  0.000000      0.00         0.2
3  0.000000           0.0        0.0  ...  0.000000      0.00         0.0
4  0.000000           1.0        0.0  ...  1.000000      0.00         1.0

[5 rows x 24 columns]

========================================================== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =========================================================
      audit  auditability  auditable  ...   service  services  technology
0  0.000000           0.0        0.0  ...  0.000000      0.00         0.0
1  0.000000           0.0        0.0  ...  0.428571      0.25         0.0
2  0.055556           0.0        0.0  ...  0.000000      0.00         0.2
3  0.000000           0.0        0.0  ...  0.000000      0.00         0.0
4  0.000000           1.0        0.0  ...  1.000000      0.00         1.0

[5 rows x 24 columns]

========================================================== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =========================================================
      audit  auditability  auditable  ...   service  services  technology
0  0.000000           0.0        0.0  ...  0.000000      0.00         0.0
1  0.000000           0.0        0.0  ...  0.428571      0.25         0.0
2  0.055556           0.0        0.0  ...  0.000000      0.00         0.2
3  0.000000           0.0        0.0  ...  0.000000      0.00         0.0
4  0.000000           1.0        0.0  ...  1.000000      0.00         1.0

[5 rows x 24 columns]

========================================================== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =========================================================
      audit  auditability  auditable  ...   service  services  technology
0 -0.613858     -0.176777  -0.176777  ... -0.316568 -0.397195   -0.592155
1 -0.613858     -0.176777  -0.176777  ...  1.709469  0.758282   -0.592155
2 -0.404885     -0.176777  -0.176777  ... -0.316568 -0.397195    0.019102
3 -0.613858     -0.176777  -0.176777  ... -0.316568 -0.397195   -0.592155
4 -0.613858      5.480078  -0.176777  ...  4.410852 -0.397195    2.464128

[5 rows x 24 columns]

========================================================== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =========================================================
Traceback (most recent call last):
  File "D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py", line 62, in <module>
    kmeans = KMeans(init='k-means++', n_clusters = c, n_init = 100, random_state=0).fit(df_normal)
  File "C:\Program Files\Python310\lib\site-packages\sklearn\cluster\_kmeans.py", line 1367, in fit
    X = self._validate_data(
  File "C:\Program Files\Python310\lib\site-packages\sklearn\base.py", line 577, in _validate_data
    X = check_array(X, input_name="X", **check_params)
  File "C:\Program Files\Python310\lib\site-packages\sklearn\utils\validation.py", line 899, in check_array
    _assert_all_finite(
  File "C:\Program Files\Python310\lib\site-packages\sklearn\utils\validation.py", line 146, in _assert_all_finite
    raise ValueError(msg_err)
ValueError: Input X contains NaN.
KMeans does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values

========================================================== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =========================================================
Traceback (most recent call last):
  File "D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py", line 63, in <module>
    kmeans = KMeans(init='k-means++', n_clusters = c, n_init = 100, random_state=0).fit(df_normal)
  File "C:\Program Files\Python310\lib\site-packages\sklearn\cluster\_kmeans.py", line 1367, in fit
    X = self._validate_data(
  File "C:\Program Files\Python310\lib\site-packages\sklearn\base.py", line 577, in _validate_data
    X = check_array(X, input_name="X", **check_params)
  File "C:\Program Files\Python310\lib\site-packages\sklearn\utils\validation.py", line 909, in check_array
    raise ValueError(
ValueError: Found array with 0 sample(s) (shape=(0, 24)) while a minimum of 1 is required by KMeans.

========================================================== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =========================================================
Traceback (most recent call last):
  File "D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py", line 74, in <module>
    from sklearn.metrics import sillhouette_samples, silhouette_score
ImportError: cannot import name 'sillhouette_samples' from 'sklearn.metrics' (C:\Program Files\Python310\lib\site-packages\sklearn\metrics\__init__.py)

========================================================== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =========================================================
Traceback (most recent call last):
  File "D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py", line 82, in <module>
    silhouette_avg = silhouette_score(data, cluster_labels)
  File "C:\Program Files\Python310\lib\site-packages\sklearn\metrics\cluster\_unsupervised.py", line 117, in silhouette_score
    return np.mean(silhouette_samples(X, labels, metric=metric, **kwds))
  File "C:\Program Files\Python310\lib\site-packages\sklearn\metrics\cluster\_unsupervised.py", line 231, in silhouette_samples
    check_number_of_labels(len(le.classes_), n_samples)
  File "C:\Program Files\Python310\lib\site-packages\sklearn\metrics\cluster\_unsupervised.py", line 33, in check_number_of_labels
    raise ValueError(
ValueError: Number of labels is 1. Valid values are 2 to n_samples - 1 (inclusive)

========================================================== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =========================================================
Traceback (most recent call last):
  File "D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py", line 82, in <module>
    silhouette_avg = silhouette_score(data, cluster_labels)
  File "C:\Program Files\Python310\lib\site-packages\sklearn\metrics\cluster\_unsupervised.py", line 117, in silhouette_score
    return np.mean(silhouette_samples(X, labels, metric=metric, **kwds))
  File "C:\Program Files\Python310\lib\site-packages\sklearn\metrics\cluster\_unsupervised.py", line 231, in silhouette_samples
    check_number_of_labels(len(le.classes_), n_samples)
  File "C:\Program Files\Python310\lib\site-packages\sklearn\metrics\cluster\_unsupervised.py", line 33, in check_number_of_labels
    raise ValueError(
ValueError: Number of labels is 1. Valid values are 2 to n_samples - 1 (inclusive)

========================================================== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =========================================================
Traceback (most recent call last):
  File "D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py", line 92, in <module>
    kmeansfit = KMeans(init='k-means++', n_clusters = c, n_init = 100, random_state=0).fit(data)
NameError: name 'c' is not defined

========================================================== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =========================================================
Traceback (most recent call last):
  File "D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py", line 95, in <module>
    scatters(cluster_data, h='Cluster')
NameError: name 'scatters' is not defined

========================================================== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =========================================================
Traceback (most recent call last):
  File "D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py", line 95, in <module>
    sns.scatters(cluster_data, h='Cluster')
AttributeError: module 'seaborn' has no attribute 'scatters'

========================================================== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =========================================================
Traceback (most recent call last):
  File "D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py", line 95, in <module>
    plt.scatter(cluster_data, h='Cluster')
TypeError: scatter() missing 1 required positional argument: 'y'

========================================================== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =========================================================
   audit  auditability  auditable  ...  services  technology  Cluster
0      0             0          0  ...         0           0        0
1      0             0          0  ...         1           0        0
2      1             0          0  ...         0           1        0
3      0             0          0  ...         0           0        0
4      0             1          0  ...         0           5        4

[5 rows x 25 columns]

========================================================== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =========================================================
Traceback (most recent call last):
  File "D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py", line 101, in <module>
    y_predict= kmeans.fit_predict(x)
NameError: name 'x' is not defined

========================================================== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =========================================================
Traceback (most recent call last):
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\indexes\base.py", line 3621, in get_loc
    return self._engine.get_loc(casted_key)
  File "pandas\_libs\index.pyx", line 136, in pandas._libs.index.IndexEngine.get_loc
  File "pandas\_libs\index.pyx", line 142, in pandas._libs.index.IndexEngine.get_loc
TypeError: '(array([ True,  True,  True,  True, False,  True, False,  True, False,
        True,  True,  True,  True, False, False,  True,  True, False,
       False, False,  True, False,  True,  True,  True,  True,  True,
       False,  True, False,  True,  True]), 0)' is an invalid key

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py", line 104, in <module>
    plt.scatter(data[y_predict == 0, 0], data[y_predict == 0, 1], s = 100, c = 'blue', label = 'Cluster 1') #for first cluster
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\frame.py", line 3505, in __getitem__
    indexer = self.columns.get_loc(key)
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\indexes\base.py", line 3628, in get_loc
    self._check_indexing_error(key)
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\indexes\base.py", line 5637, in _check_indexing_error
    raise InvalidIndexError(key)
pandas.errors.InvalidIndexError: (array([ True,  True,  True,  True, False,  True, False,  True, False,
        True,  True,  True,  True, False, False,  True,  True, False,
       False, False,  True, False,  True,  True,  True,  True,  True,
       False,  True, False,  True,  True]), 0)

========================================================== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =========================================================
Traceback (most recent call last):
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\indexes\base.py", line 3621, in get_loc
    return self._engine.get_loc(casted_key)
  File "pandas\_libs\index.pyx", line 136, in pandas._libs.index.IndexEngine.get_loc
  File "pandas\_libs\index.pyx", line 142, in pandas._libs.index.IndexEngine.get_loc
TypeError: '(array([ True,  True,  True,  True, False,  True, False,  True, False,
        True,  True,  True,  True, False, False,  True,  True, False,
       False, False,  True, False,  True,  True,  True,  True,  True,
       False,  True, False,  True,  True]), 0)' is an invalid key

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py", line 104, in <module>
    plt.scatter(data[y_predict == 0, 0], data[y_predict == 0, 1], s = 100, c = 'blue', label = 'Cluster 1') #for first cluster
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\frame.py", line 3505, in __getitem__
    indexer = self.columns.get_loc(key)
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\indexes\base.py", line 3628, in get_loc
    self._check_indexing_error(key)
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\indexes\base.py", line 5637, in _check_indexing_error
    raise InvalidIndexError(key)
pandas.errors.InvalidIndexError: (array([ True,  True,  True,  True, False,  True, False,  True, False,
        True,  True,  True,  True, False, False,  True,  True, False,
       False, False,  True, False,  True,  True,  True,  True,  True,
       False,  True, False,  True,  True]), 0)

========================================================== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =========================================================
   audit  auditability  auditable  ...  services  technology  Cluster
0      0             0          0  ...         0           0        0
1      0             0          0  ...         1           0        0
2      1             0          0  ...         0           1        0
3      0             0          0  ...         0           0        0
4      0             1          0  ...         0           5        4

[5 rows x 25 columns]
Traceback (most recent call last):
  File "D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py", line 99, in <module>
    scatters(cluster_data, h='Cluster')
NameError: name 'scatters' is not defined

========================================================== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =========================================================
Traceback (most recent call last):
  File "D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py", line 119, in <module>
    label = kmeans.fit_predict(df)
NameError: name 'df' is not defined

========================================================== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =========================================================
Traceback (most recent call last):
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\indexes\base.py", line 3621, in get_loc
    return self._engine.get_loc(casted_key)
  File "pandas\_libs\index.pyx", line 136, in pandas._libs.index.IndexEngine.get_loc
  File "pandas\_libs\index.pyx", line 142, in pandas._libs.index.IndexEngine.get_loc
TypeError: '(array([False, False, False, False, False, False,  True, False,  True,
       False, False, False, False,  True,  True, False, False,  True,
        True,  True, False,  True, False, False, False, False, False,
       False, False, False, False, False]), 0)' is an invalid key

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py", line 124, in <module>
    plt.scatter(data[label == i , 0] , data[label == i , 1] , label = i)
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\frame.py", line 3505, in __getitem__
    indexer = self.columns.get_loc(key)
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\indexes\base.py", line 3628, in get_loc
    self._check_indexing_error(key)
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\indexes\base.py", line 5637, in _check_indexing_error
    raise InvalidIndexError(key)
pandas.errors.InvalidIndexError: (array([False, False, False, False, False, False,  True, False,  True,
       False, False, False, False,  True,  True, False, False,  True,
        True,  True, False,  True, False, False, False, False, False,
       False, False, False, False, False]), 0)

========================================================== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =========================================================
KMeans(n_clusters=5)
[0 0 0 0 4 0 3 0 3 0 0 0 0 3 3 0 0 3 3 3 0 3 0 0 0 0 0 1 0 2 0 0]

========================================================== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =========================================================
[0 0 3 0 3 0 2 3 2 3 3 0 0 2 2 0 0 2 2 2 0 2 0 3 0 0 0 1 0 4 0 0]

========================================================== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =========================================================
[0 0 0 0 4 0 1 0 1 0 0 0 0 1 1 0 0 1 1 1 0 1 0 0 0 0 0 2 0 3 0 0]
[0 1 2 3 4]

========================================================== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =========================================================
[1 1 1 1 4 1 0 1 0 1 1 1 1 0 0 1 1 0 0 0 1 0 1 1 1 1 1 3 1 2 1 1]
[0 1 2 3 4]
Traceback (most recent call last):
  File "D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py", line 129, in <module>
    plt.scatter(cluster_data[Cluster == i , 0] , cluster_data[Cluster == i, 1] , label = i)
NameError: name 'Cluster' is not defined

========================================================== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =========================================================
Index(['audit', 'auditability', 'auditable', 'audited', 'auditees', 'auditing',
       'auditor', 'auditors', 'audits', 'blockchain', 'blockchains', 'fraud',
       'frauds', 'fraudsters', 'fraudulent', 'protection', 'quality', 'risk',
       'risks', 'cybersecurity', 'security', 'service', 'services',
       'technology', 'Cluster'],
      dtype='object')
[0 0 0 0 4 0 3 0 3 0 0 0 0 3 3 0 0 1 3 3 0 3 0 0 0 0 0 1 0 2 0 0]
[0 1 2 3 4]
Traceback (most recent call last):
  File "D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py", line 130, in <module>
    plt.scatter(cluster_data[Cluster == i , 0] , cluster_data[Cluster == i, 1] , label = i)
NameError: name 'Cluster' is not defined

========================================================== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =========================================================
Index(['audit', 'auditability', 'auditable', 'audited', 'auditees', 'auditing',
       'auditor', 'auditors', 'audits', 'blockchain', 'blockchains', 'fraud',
       'frauds', 'fraudsters', 'fraudulent', 'protection', 'quality', 'risk',
       'risks', 'cybersecurity', 'security', 'service', 'services',
       'technology', 'Cluster'],
      dtype='object')
[0 0 4 0 4 0 2 4 2 4 4 0 0 2 2 0 0 2 2 2 0 2 0 4 0 0 0 1 0 3 0 0]
[0 1 2 3 4]
Traceback (most recent call last):
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\indexes\base.py", line 3621, in get_loc
    return self._engine.get_loc(casted_key)
  File "pandas\_libs\index.pyx", line 136, in pandas._libs.index.IndexEngine.get_loc
  File "pandas\_libs\index.pyx", line 163, in pandas._libs.index.IndexEngine.get_loc
  File "pandas\_libs\hashtable_class_helper.pxi", line 5198, in pandas._libs.hashtable.PyObjectHashTable.get_item
  File "pandas\_libs\hashtable_class_helper.pxi", line 5206, in pandas._libs.hashtable.PyObjectHashTable.get_item
KeyError: (False, 0)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py", line 130, in <module>
    plt.scatter(cluster_data['Cluster' == i , 0] , cluster_data['Cluster' == i, 1] , label = i)
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\frame.py", line 3505, in __getitem__
    indexer = self.columns.get_loc(key)
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\indexes\base.py", line 3623, in get_loc
    raise KeyError(key) from err
KeyError: (False, 0)

========================================================== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =========================================================
Traceback (most recent call last):
  File "D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py", line 119, in <module>
    y_kmeans = kmeans.fit_predict(X)
NameError: name 'X' is not defined

========================================================== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =========================================================
Traceback (most recent call last):
  File "D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py", line 119, in <module>
    y_kmeans = kmeans.fit_predict(X)
NameError: name 'X' is not defined

========================================================== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =========================================================
Traceback (most recent call last):
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\indexes\base.py", line 3621, in get_loc
    return self._engine.get_loc(casted_key)
  File "pandas\_libs\index.pyx", line 136, in pandas._libs.index.IndexEngine.get_loc
  File "pandas\_libs\index.pyx", line 142, in pandas._libs.index.IndexEngine.get_loc
TypeError: '(array([False, False, False, False,  True, False, False,  True, False,
        True,  True, False, False, False, False, False, False, False,
       False, False, False, False, False,  True, False, False, False,
       False, False, False, False, False]), 0)' is an invalid key

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py", line 123, in <module>
    plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], s=100, c='red', label ='Cluster 1')
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\frame.py", line 3505, in __getitem__
    indexer = self.columns.get_loc(key)
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\indexes\base.py", line 3628, in get_loc
    self._check_indexing_error(key)
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\indexes\base.py", line 5637, in _check_indexing_error
    raise InvalidIndexError(key)
pandas.errors.InvalidIndexError: (array([False, False, False, False,  True, False, False,  True, False,
        True,  True, False, False, False, False, False, False, False,
       False, False, False, False, False,  True, False, False, False,
       False, False, False, False, False]), 0)

========================================================== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =========================================================
Traceback (most recent call last):
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\indexes\base.py", line 3621, in get_loc
    return self._engine.get_loc(casted_key)
  File "pandas\_libs\index.pyx", line 136, in pandas._libs.index.IndexEngine.get_loc
  File "pandas\_libs\index.pyx", line 142, in pandas._libs.index.IndexEngine.get_loc
TypeError: '(array([False, False, False, False,  True, False, False,  True, False,
        True,  True, False, False, False, False, False, False, False,
       False, False, False, False, False,  True, False, False, False,
       False, False, False, False, False]), 0)' is an invalid key

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py", line 123, in <module>
    plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], s=100, c='red', label ='Cluster 1')
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\frame.py", line 3505, in __getitem__
    indexer = self.columns.get_loc(key)
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\indexes\base.py", line 3628, in get_loc
    self._check_indexing_error(key)
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\indexes\base.py", line 5637, in _check_indexing_error
    raise InvalidIndexError(key)
pandas.errors.InvalidIndexError: (array([False, False, False, False,  True, False, False,  True, False,
        True,  True, False, False, False, False, False, False, False,
       False, False, False, False, False,  True, False, False, False,
       False, False, False, False, False]), 0)

========================================================== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =========================================================
Traceback (most recent call last):
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\indexes\base.py", line 3621, in get_loc
    return self._engine.get_loc(casted_key)
  File "pandas\_libs\index.pyx", line 136, in pandas._libs.index.IndexEngine.get_loc
  File "pandas\_libs\index.pyx", line 142, in pandas._libs.index.IndexEngine.get_loc
TypeError: '(array([False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False,  True, False, False]), 0)' is an invalid key

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py", line 123, in <module>
    plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], s=100, c='red', label ='Cluster 1')
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\frame.py", line 3505, in __getitem__
    indexer = self.columns.get_loc(key)
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\indexes\base.py", line 3628, in get_loc
    self._check_indexing_error(key)
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\indexes\base.py", line 5637, in _check_indexing_error
    raise InvalidIndexError(key)
pandas.errors.InvalidIndexError: (array([False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False,  True, False, False]), 0)
X[y_kmeans==0, 0]
Traceback (most recent call last):
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\indexes\base.py", line 3621, in get_loc
    return self._engine.get_loc(casted_key)
  File "pandas\_libs\index.pyx", line 136, in pandas._libs.index.IndexEngine.get_loc
  File "pandas\_libs\index.pyx", line 142, in pandas._libs.index.IndexEngine.get_loc
TypeError: '(array([False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False,  True, False, False]), 0)' is an invalid key

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<pyshell#0>", line 1, in <module>
    X[y_kmeans==0, 0]
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\frame.py", line 3505, in __getitem__
    indexer = self.columns.get_loc(key)
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\indexes\base.py", line 3628, in get_loc
    self._check_indexing_error(key)
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\indexes\base.py", line 5637, in _check_indexing_error
    raise InvalidIndexError(key)
pandas.errors.InvalidIndexError: (array([False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False,  True, False, False]), 0)
y_kmeans
array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1,
       3, 3, 3, 3, 3, 4, 3, 0, 3, 3])
y_kmeans==0
array([False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False,  True, False, False])
X[y_kmeans==0]
    Unnamed: 0  audit  auditability  ...  service  services  technology
29          29      2             0  ...        4         4           0

[1 rows x 25 columns]
X[y_kmeans==0, 0]
Traceback (most recent call last):
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\indexes\base.py", line 3621, in get_loc
    return self._engine.get_loc(casted_key)
  File "pandas\_libs\index.pyx", line 136, in pandas._libs.index.IndexEngine.get_loc
  File "pandas\_libs\index.pyx", line 142, in pandas._libs.index.IndexEngine.get_loc
TypeError: '(array([False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False,  True, False, False]), 0)' is an invalid key

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<pyshell#4>", line 1, in <module>
    X[y_kmeans==0, 0]
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\frame.py", line 3505, in __getitem__
    indexer = self.columns.get_loc(key)
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\indexes\base.py", line 3628, in get_loc
    self._check_indexing_error(key)
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\indexes\base.py", line 5637, in _check_indexing_error
    raise InvalidIndexError(key)
pandas.errors.InvalidIndexError: (array([False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False,  True, False, False]), 0)
X[0, 0]
Traceback (most recent call last):
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\indexes\base.py", line 3621, in get_loc
    return self._engine.get_loc(casted_key)
  File "pandas\_libs\index.pyx", line 136, in pandas._libs.index.IndexEngine.get_loc
  File "pandas\_libs\index.pyx", line 163, in pandas._libs.index.IndexEngine.get_loc
  File "pandas\_libs\hashtable_class_helper.pxi", line 5198, in pandas._libs.hashtable.PyObjectHashTable.get_item
  File "pandas\_libs\hashtable_class_helper.pxi", line 5206, in pandas._libs.hashtable.PyObjectHashTable.get_item
KeyError: (0, 0)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "<pyshell#5>", line 1, in <module>
    X[0, 0]
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\frame.py", line 3505, in __getitem__
    indexer = self.columns.get_loc(key)
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\indexes\base.py", line 3623, in get_loc
    raise KeyError(key) from err
KeyError: (0, 0)
X[y_kmeans==0, 0]
Traceback (most recent call last):
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\indexes\base.py", line 3621, in get_loc
    return self._engine.get_loc(casted_key)
  File "pandas\_libs\index.pyx", line 136, in pandas._libs.index.IndexEngine.get_loc
  File "pandas\_libs\index.pyx", line 142, in pandas._libs.index.IndexEngine.get_loc
TypeError: '(array([False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False,  True, False, False]), 0)' is an invalid key

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<pyshell#6>", line 1, in <module>
    X[y_kmeans==0, 0]
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\frame.py", line 3505, in __getitem__
    indexer = self.columns.get_loc(key)
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\indexes\base.py", line 3628, in get_loc
    self._check_indexing_error(key)
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\indexes\base.py", line 5637, in _check_indexing_error
    raise InvalidIndexError(key)
pandas.errors.InvalidIndexError: (array([False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False,  True, False, False]), 0)
X[y_kmeans==0]
    Unnamed: 0  audit  auditability  ...  service  services  technology
29          29      2             0  ...        4         4           0

[1 rows x 25 columns]

========================================================== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =========================================================
   audit  auditability  auditable  ...  services  technology  Cluster
0      0             0          0  ...         0           0        0
1      0             0          0  ...         1           0        0
2      1             0          0  ...         0           1        0
3      0             0          0  ...         0           0        0
4      0             1          0  ...         0           5        4

[5 rows x 25 columns]
Traceback (most recent call last):
  File "D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py", line 99, in <module>
    scatters(cluster_data, h='Cluster')
NameError: name 'scatters' is not defined

========================================================== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =========================================================
   audit  auditability  auditable  ...  services  technology  Cluster
0      0             0          0  ...         0           0        0
1      0             0          0  ...         1           0        0
2      1             0          0  ...         0           1        0
3      0             0          0  ...         0           0        0
4      0             1          0  ...         0           5        4

[5 rows x 25 columns]

========================================================== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =========================================================
   audit  auditability  auditable  ...  services  technology  Cluster
0      0             0          0  ...         0           0        0
1      0             0          0  ...         1           0        0
2      1             0          0  ...         0           1        0
3      0             0          0  ...         0           0        0
4      0             1          0  ...         0           5        4

[5 rows x 25 columns]
    0
0   0
1   0
2   0
3   0
4   4
5   0
6   2
7   0
8   2
9   0
10  0
11  0
12  0
13  2
14  2
15  0
16  0
17  2
18  2
19  2
20  0
21  2
22  0
23  0
24  0
25  0
26  0
27  1
28  0
29  3
30  0
31  0
cluster_data.groupby(['Cluster'])
<pandas.core.groupby.generic.DataFrameGroupBy object at 0x000001D53B942680>
cluster_data.groupby(['Cluster']).counts
Traceback (most recent call last):
  File "<pyshell#9>", line 1, in <module>
    cluster_data.groupby(['Cluster']).counts
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\groupby\groupby.py", line 904, in __getattr__
    raise AttributeError(
AttributeError: 'DataFrameGroupBy' object has no attribute 'counts'. Did you mean: 'count'?
cluster_data.groupby(['Cluster']).count
<bound method GroupBy.count of <pandas.core.groupby.generic.DataFrameGroupBy object at 0x000001D53B942590>>
cluster_data.groupby(['Cluster']).count()
         audit  auditability  auditable  ...  service  services  technology
Cluster                                  ...                               
0           21            21         21  ...       21        21          21
1            1             1          1  ...        1         1           1
2            8             8          8  ...        8         8           8
3            1             1          1  ...        1         1           1
4            1             1          1  ...        1         1           1

[5 rows x 24 columns]
cluster_data[label == 0, 0]
Traceback (most recent call last):
  File "<pyshell#12>", line 1, in <module>
    cluster_data[label == 0, 0]
NameError: name 'label' is not defined. Did you mean: 'labels'?
cluster_data[labels == 0, 0]
Traceback (most recent call last):
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\indexes\base.py", line 3621, in get_loc
    return self._engine.get_loc(casted_key)
  File "pandas\_libs\index.pyx", line 136, in pandas._libs.index.IndexEngine.get_loc
  File "pandas\_libs\index.pyx", line 142, in pandas._libs.index.IndexEngine.get_loc
TypeError: '(        0
0    True
1    True
2    True
3    True
4   False
5    True
6   False
7    True
8   False
9    True
10   True
11   True
12   True
13  False
14  False
15   True
16   True
17  False
18  False
19  False
20   True
21  False
22   True
23   True
24   True
25   True
26   True
27  False
28   True
29  False
30   True
31   True, 0)' is an invalid key

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<pyshell#13>", line 1, in <module>
    cluster_data[labels == 0, 0]
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\frame.py", line 3505, in __getitem__
    indexer = self.columns.get_loc(key)
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\indexes\base.py", line 3628, in get_loc
    self._check_indexing_error(key)
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\indexes\base.py", line 5637, in _check_indexing_error
    raise InvalidIndexError(key)
pandas.errors.InvalidIndexError: (        0
0    True
1    True
2    True
3    True
4   False
5    True
6   False
7    True
8   False
9    True
10   True
11   True
12   True
13  False
14  False
15   True
16   True
17  False
18  False
19  False
20   True
21  False
22   True
23   True
24   True
25   True
26   True
27  False
28   True
29  False
30   True
31   True, 0)
cluster_data[labels == 0]
    audit  auditability  auditable  ...  services  technology  Cluster
0     NaN           NaN        NaN  ...       NaN         NaN      NaN
1     NaN           NaN        NaN  ...       NaN         NaN      NaN
2     NaN           NaN        NaN  ...       NaN         NaN      NaN
3     NaN           NaN        NaN  ...       NaN         NaN      NaN
4     NaN           NaN        NaN  ...       NaN         NaN      NaN
5     NaN           NaN        NaN  ...       NaN         NaN      NaN
6     NaN           NaN        NaN  ...       NaN         NaN      NaN
7     NaN           NaN        NaN  ...       NaN         NaN      NaN
8     NaN           NaN        NaN  ...       NaN         NaN      NaN
9     NaN           NaN        NaN  ...       NaN         NaN      NaN
10    NaN           NaN        NaN  ...       NaN         NaN      NaN
11    NaN           NaN        NaN  ...       NaN         NaN      NaN
12    NaN           NaN        NaN  ...       NaN         NaN      NaN
13    NaN           NaN        NaN  ...       NaN         NaN      NaN
14    NaN           NaN        NaN  ...       NaN         NaN      NaN
15    NaN           NaN        NaN  ...       NaN         NaN      NaN
16    NaN           NaN        NaN  ...       NaN         NaN      NaN
17    NaN           NaN        NaN  ...       NaN         NaN      NaN
18    NaN           NaN        NaN  ...       NaN         NaN      NaN
19    NaN           NaN        NaN  ...       NaN         NaN      NaN
20    NaN           NaN        NaN  ...       NaN         NaN      NaN
21    NaN           NaN        NaN  ...       NaN         NaN      NaN
22    NaN           NaN        NaN  ...       NaN         NaN      NaN
23    NaN           NaN        NaN  ...       NaN         NaN      NaN
24    NaN           NaN        NaN  ...       NaN         NaN      NaN
25    NaN           NaN        NaN  ...       NaN         NaN      NaN
26    NaN           NaN        NaN  ...       NaN         NaN      NaN
27    NaN           NaN        NaN  ...       NaN         NaN      NaN
28    NaN           NaN        NaN  ...       NaN         NaN      NaN
29    NaN           NaN        NaN  ...       NaN         NaN      NaN
30    NaN           NaN        NaN  ...       NaN         NaN      NaN
31    NaN           NaN        NaN  ...       NaN         NaN      NaN

[32 rows x 25 columns]
cluster_data[Cluster == 0]
Traceback (most recent call last):
  File "<pyshell#15>", line 1, in <module>
    cluster_data[Cluster == 0]
NameError: name 'Cluster' is not defined
cluster_data['Cluster' == 0]
Traceback (most recent call last):
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\indexes\base.py", line 3621, in get_loc
    return self._engine.get_loc(casted_key)
  File "pandas\_libs\index.pyx", line 136, in pandas._libs.index.IndexEngine.get_loc
  File "pandas\_libs\index.pyx", line 163, in pandas._libs.index.IndexEngine.get_loc
  File "pandas\_libs\hashtable_class_helper.pxi", line 5198, in pandas._libs.hashtable.PyObjectHashTable.get_item
  File "pandas\_libs\hashtable_class_helper.pxi", line 5206, in pandas._libs.hashtable.PyObjectHashTable.get_item
KeyError: False

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "<pyshell#16>", line 1, in <module>
    cluster_data['Cluster' == 0]
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\frame.py", line 3505, in __getitem__
    indexer = self.columns.get_loc(key)
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\indexes\base.py", line 3623, in get_loc
    raise KeyError(key) from err
KeyError: False
cluster_data['Cluster'] == 0
0      True
1      True
2      True
3      True
4     False
5      True
6     False
7      True
8     False
9      True
10     True
11     True
12     True
13    False
14    False
15     True
16     True
17    False
18    False
19    False
20     True
21    False
22     True
23     True
24     True
25     True
26     True
27    False
28     True
29    False
30     True
31     True
Name: Cluster, dtype: bool
plt.scatter(cluster_data['Cluster'] == 0)
Traceback (most recent call last):
  File "<pyshell#18>", line 1, in <module>
    plt.scatter(cluster_data['Cluster'] == 0)
TypeError: scatter() missing 1 required positional argument: 'y'
plt.scatter(range(1, 31), cluster_data['Cluster'] == 0)
Traceback (most recent call last):
  File "<pyshell#19>", line 1, in <module>
    plt.scatter(range(1, 31), cluster_data['Cluster'] == 0)
  File "C:\Program Files\Python310\lib\site-packages\matplotlib\pyplot.py", line 2819, in scatter
    __ret = gca().scatter(
  File "C:\Program Files\Python310\lib\site-packages\matplotlib\__init__.py", line 1412, in inner
    return func(ax, *map(sanitize_sequence, args), **kwargs)
  File "C:\Program Files\Python310\lib\site-packages\matplotlib\axes\_axes.py", line 4362, in scatter
    raise ValueError("x and y must be the same size")
ValueError: x and y must be the same size
plt.scatter(range(31), cluster_data['Cluster'] == 0)
Traceback (most recent call last):
  File "<pyshell#20>", line 1, in <module>
    plt.scatter(range(31), cluster_data['Cluster'] == 0)
  File "C:\Program Files\Python310\lib\site-packages\matplotlib\pyplot.py", line 2819, in scatter
    __ret = gca().scatter(
  File "C:\Program Files\Python310\lib\site-packages\matplotlib\__init__.py", line 1412, in inner
    return func(ax, *map(sanitize_sequence, args), **kwargs)
  File "C:\Program Files\Python310\lib\site-packages\matplotlib\axes\_axes.py", line 4362, in scatter
    raise ValueError("x and y must be the same size")
ValueError: x and y must be the same size
len(cluster_data['Cluster'] == 0)
32
plt.scatter(range(32), cluster_data['Cluster'] == 0)
<matplotlib.collections.PathCollection object at 0x000001D53FE57EE0>
plt.show()
cluster_data.groupby(['Cluster']).mean().round(1)
         audit  auditability  auditable  ...  service  services  technology
Cluster                                  ...                               
0          0.3           0.0        0.0  ...      0.2       0.3         0.9
1         18.0           0.0        0.0  ...      0.0       1.0         1.0
2          8.4           0.0        0.0  ...      0.0       0.0         0.9
3          2.0           0.0        0.0  ...      4.0       4.0         0.0
4          0.0           1.0        0.0  ...      7.0       0.0         5.0

[5 rows x 24 columns]
import os
os.getcwd()
'D:\\Research\\papers\\finance\\iimv\\python-scripts'
cluster_data.groupby(['Cluster']).mean().round(1).write_csv('cluster-wise-means.csv')
Traceback (most recent call last):
  File "<pyshell#27>", line 1, in <module>
    cluster_data.groupby(['Cluster']).mean().round(1).write_csv('cluster-wise-means.csv')
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\generic.py", line 5575, in __getattr__
    return object.__getattribute__(self, name)
AttributeError: 'DataFrame' object has no attribute 'write_csv'
cluster_data.groupby(['Cluster']).mean().round(1).to_csv('cluster-wise-means.csv')
sns.pairplot(vars = cluster_data.columns, data = cluster_data, hue = "Cluster")
Traceback (most recent call last):
  File "<pyshell#29>", line 1, in <module>
    sns.pairplot(vars = cluster_data.columns, data = cluster_data, hue = "Cluster")
  File "C:\Program Files\Python310\lib\site-packages\seaborn\_decorators.py", line 46, in inner_f
    return f(**kwargs)
  File "C:\Program Files\Python310\lib\site-packages\seaborn\axisgrid.py", line 2154, in pairplot
    grid.add_legend()
  File "C:\Program Files\Python310\lib\site-packages\seaborn\axisgrid.py", line 173, in add_legend
    _draw_figure(self._figure)
  File "C:\Program Files\Python310\lib\site-packages\seaborn\utils.py", line 95, in _draw_figure
    fig.canvas.draw()
  File "C:\Program Files\Python310\lib\site-packages\matplotlib\backends\backend_tkagg.py", line 9, in draw
    super().draw()
  File "C:\Program Files\Python310\lib\site-packages\matplotlib\backends\backend_agg.py", line 436, in draw
    self.figure.draw(self.renderer)
  File "C:\Program Files\Python310\lib\site-packages\matplotlib\artist.py", line 73, in draw_wrapper
    result = draw(artist, renderer, *args, **kwargs)
  File "C:\Program Files\Python310\lib\site-packages\matplotlib\artist.py", line 50, in draw_wrapper
    return draw(artist, renderer)
  File "C:\Program Files\Python310\lib\site-packages\matplotlib\figure.py", line 2837, in draw
    mimage._draw_list_compositing_images(
  File "C:\Program Files\Python310\lib\site-packages\matplotlib\image.py", line 132, in _draw_list_compositing_images
    a.draw(renderer)
  File "C:\Program Files\Python310\lib\site-packages\matplotlib\artist.py", line 50, in draw_wrapper
    return draw(artist, renderer)
  File "C:\Program Files\Python310\lib\site-packages\matplotlib\axes\_base.py", line 3091, in draw
    mimage._draw_list_compositing_images(
  File "C:\Program Files\Python310\lib\site-packages\matplotlib\image.py", line 132, in _draw_list_compositing_images
    a.draw(renderer)
  File "C:\Program Files\Python310\lib\site-packages\matplotlib\artist.py", line 50, in draw_wrapper
    return draw(artist, renderer)
  File "C:\Program Files\Python310\lib\site-packages\matplotlib\axis.py", line 1163, in draw
    tick.draw(renderer)
  File "C:\Program Files\Python310\lib\site-packages\matplotlib\artist.py", line 50, in draw_wrapper
    return draw(artist, renderer)
  File "C:\Program Files\Python310\lib\site-packages\matplotlib\axis.py", line 299, in draw
    artist.draw(renderer)
  File "C:\Program Files\Python310\lib\site-packages\matplotlib\artist.py", line 50, in draw_wrapper
    return draw(artist, renderer)
  File "C:\Program Files\Python310\lib\site-packages\matplotlib\lines.py", line 732, in draw
    self.recache()
  File "C:\Program Files\Python310\lib\site-packages\matplotlib\lines.py", line 661, in recache
    self._xy = np.column_stack(np.broadcast_arrays(x, y)).astype(float)
KeyboardInterrupt




cluster_data.columns[1:2, ]
Index(['auditability'], dtype='object')
cluster_data.columns[3, ]
  
Traceback (most recent call last):
  File "<pyshell#31>", line 1, in <module>
    cluster_data.columns[3, ]
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\indexes\base.py", line 5057, in __getitem__
    if result.ndim > 1:
AttributeError: 'str' object has no attribute 'ndim'
cluster_data.columns[0:3, ]
  
Index(['audit', 'auditability', 'auditable'], dtype='object')
sns.pairplot(vars = cluster_data.columns[0:2, ], data = cluster_data, hue = "Cluster")
  
<seaborn.axisgrid.PairGrid object at 0x000001D563CA25C0>
plt.show()
  
cluster_data.columns[0:3, ]
  
Index(['audit', 'auditability', 'auditable'], dtype='object')
cluster_data.columns[0:4, ]
  
Index(['audit', 'auditability', 'auditable', 'audited'], dtype='object')
sns.pairplot(vars = cluster_data.columns[0:4, ], data = cluster_data, hue = "Cluster")
  
<seaborn.axisgrid.PairGrid object at 0x000001D53FE0C040>
plt.show()
cluster_data[0, ]
Traceback (most recent call last):
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\indexes\base.py", line 3621, in get_loc
    return self._engine.get_loc(casted_key)
  File "pandas\_libs\index.pyx", line 136, in pandas._libs.index.IndexEngine.get_loc
  File "pandas\_libs\index.pyx", line 163, in pandas._libs.index.IndexEngine.get_loc
  File "pandas\_libs\hashtable_class_helper.pxi", line 5198, in pandas._libs.hashtable.PyObjectHashTable.get_item
  File "pandas\_libs\hashtable_class_helper.pxi", line 5206, in pandas._libs.hashtable.PyObjectHashTable.get_item
KeyError: (0,)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "<pyshell#39>", line 1, in <module>
    cluster_data[0, ]
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\frame.py", line 3505, in __getitem__
    indexer = self.columns.get_loc(key)
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\indexes\base.py", line 3623, in get_loc
    raise KeyError(key) from err
KeyError: (0,)
cluster_data[0:1, ]
Traceback (most recent call last):
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\indexes\base.py", line 3621, in get_loc
    return self._engine.get_loc(casted_key)
  File "pandas\_libs\index.pyx", line 136, in pandas._libs.index.IndexEngine.get_loc
  File "pandas\_libs\index.pyx", line 142, in pandas._libs.index.IndexEngine.get_loc
TypeError: '(slice(0, 1, None),)' is an invalid key

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<pyshell#40>", line 1, in <module>
    cluster_data[0:1, ]
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\frame.py", line 3505, in __getitem__
    indexer = self.columns.get_loc(key)
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\indexes\base.py", line 3628, in get_loc
    self._check_indexing_error(key)
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\indexes\base.py", line 5637, in _check_indexing_error
    raise InvalidIndexError(key)
pandas.errors.InvalidIndexError: (slice(0, 1, None),)
cluster_data.columns[0:1, ]
Index(['audit'], dtype='object')
cluster_data.columns[1, ]
Traceback (most recent call last):
  File "<pyshell#42>", line 1, in <module>
    cluster_data.columns[1, ]
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\indexes\base.py", line 5057, in __getitem__
    if result.ndim > 1:
AttributeError: 'str' object has no attribute 'ndim'
cluster_data.columns[0:2, ]
Index(['audit', 'auditability'], dtype='object')
cluster_data.columns[0:1, ]
Index(['audit'], dtype='object')
cluster_data['audit']
0      0
1      0
2      1
3      0
4      0
5      2
6     10
7      0
8      8
9      1
10     0
11     0
12     0
13     7
14     6
15     0
16     1
17    16
18     6
19     5
20     0
21     9
22     2
23     0
24     0
25     0
26     0
27    18
28     0
29     2
30     0
31     0
Name: audit, dtype: int64
plt.scatter(cluster_data['audit'], cluster_data['blockchain'], c=cluster_data['Cluster'])
<matplotlib.collections.PathCollection object at 0x000001D544244F70>
plt.show()
plt.scatter(cluster_data['audit'], cluster_data['risk'], c=cluster_data['Cluster'])
<matplotlib.collections.PathCollection object at 0x000001D5442A1480>
plt.show()
data.head()
   audit  auditability  auditable  ...  service  services  technology
0      0             0          0  ...        0         0           0
1      0             0          0  ...        3         1           0
2      1             0          0  ...        0         0           1
3      0             0          0  ...        0         0           0
4      0             1          0  ...        7         0           5

[5 rows x 24 columns]
labels == 0
        0
0    True
1    True
2    True
3    True
4   False
5    True
6   False
7    True
8   False
9    True
10   True
11   True
12   True
13  False
14  False
15   True
16   True
17  False
18  False
19  False
20   True
21  False
22   True
23   True
24   True
25   True
26   True
27  False
28   True
29  False
30   True
31   True
plt.scatter(data[labels == 0, 0], data[labels == 0, 1], c = 'blue', label = 'Cluster 1')
Traceback (most recent call last):
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\indexes\base.py", line 3621, in get_loc
    return self._engine.get_loc(casted_key)
  File "pandas\_libs\index.pyx", line 136, in pandas._libs.index.IndexEngine.get_loc
  File "pandas\_libs\index.pyx", line 142, in pandas._libs.index.IndexEngine.get_loc
TypeError: '(        0
0    True
1    True
2    True
3    True
4   False
5    True
6   False
7    True
8   False
9    True
10   True
11   True
12   True
13  False
14  False
15   True
16   True
17  False
18  False
19  False
20   True
21  False
22   True
23   True
24   True
25   True
26   True
27  False
28   True
29  False
30   True
31   True, 0)' is an invalid key

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<pyshell#52>", line 1, in <module>
    plt.scatter(data[labels == 0, 0], data[labels == 0, 1], c = 'blue', label = 'Cluster 1')
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\frame.py", line 3505, in __getitem__
    indexer = self.columns.get_loc(key)
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\indexes\base.py", line 3628, in get_loc
    self._check_indexing_error(key)
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\indexes\base.py", line 5637, in _check_indexing_error
    raise InvalidIndexError(key)
pandas.errors.InvalidIndexError: (        0
0    True
1    True
2    True
3    True
4   False
5    True
6   False
7    True
8   False
9    True
10   True
11   True
12   True
13  False
14  False
15   True
16   True
17  False
18  False
19  False
20   True
21  False
22   True
23   True
24   True
25   True
26   True
27  False
28   True
29  False
30   True
31   True, 0)
X = data.values
X.shape
(32, 24)
kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(X)
KMeans(n_clusters=4, random_state=0)
kmeans.cluster_centers_.shape
(4, 24)
label = kmeans.predict(X)
plt.scatter(X[label == 0, 0], X[label == 0, 1], s=100, c='red', label='Cluster 1')
<matplotlib.collections.PathCollection object at 0x000001D542D62860>
plt.show()

========================================================== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =========================================================
(32, 24)
(5, 24)

========================================================== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =========================================================
(32, 24)
(5, 24)

========================================================== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =========================================================
(32, 24)
(5, 24)

========================================================== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =========================================================
(32, 24)
(5, 24)

========================================================== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =========================================================
(32, 24)
(5, 24)

========================================================== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =========================================================
(32, 24)
(5, 24)
sns.scatterplot(data=cluster_data, hue="Cluster")
Traceback (most recent call last):
  File "<pyshell#61>", line 1, in <module>
    sns.scatterplot(data=cluster_data, hue="Cluster")
NameError: name 'cluster_data' is not defined

========================================================== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =========================================================
Traceback (most recent call last):
  File "D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py", line 97, in <module>
    sns.scatter(data=cluster_data, hue='Cluster')
AttributeError: module 'seaborn' has no attribute 'scatter'

========================================================== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =========================================================
Traceback (most recent call last):
  File "D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py", line 97, in <module>
    sns.scatterplot(data=cluster_data, hue='Cluster')
  File "C:\Program Files\Python310\lib\site-packages\seaborn\_decorators.py", line 46, in inner_f
    return f(**kwargs)
  File "C:\Program Files\Python310\lib\site-packages\seaborn\relational.py", line 808, in scatterplot
    p = _ScatterPlotter(
  File "C:\Program Files\Python310\lib\site-packages\seaborn\relational.py", line 587, in __init__
    super().__init__(data=data, variables=variables)
  File "C:\Program Files\Python310\lib\site-packages\seaborn\_core.py", line 605, in __init__
    self.assign_variables(data, variables)
  File "C:\Program Files\Python310\lib\site-packages\seaborn\_core.py", line 663, in assign_variables
    plot_data, variables = self._assign_variables_wideform(
  File "C:\Program Files\Python310\lib\site-packages\seaborn\_core.py", line 712, in _assign_variables_wideform
    raise ValueError(err)
ValueError: The following variable cannot be assigned with wide-form data: `hue`

========================================================== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =========================================================
Traceback (most recent call last):
  File "D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py", line 97, in <module>
    sns.scatterplot(data=cluster_data, h='Cluster')
  File "C:\Program Files\Python310\lib\site-packages\seaborn\_decorators.py", line 46, in inner_f
    return f(**kwargs)
  File "C:\Program Files\Python310\lib\site-packages\seaborn\relational.py", line 827, in scatterplot
    p.plot(ax, kwargs)
  File "C:\Program Files\Python310\lib\site-packages\seaborn\relational.py", line 608, in plot
    scout = ax.scatter(scout_x, scout_y, **kws)
  File "C:\Program Files\Python310\lib\site-packages\matplotlib\__init__.py", line 1412, in inner
    return func(ax, *map(sanitize_sequence, args), **kwargs)
  File "C:\Program Files\Python310\lib\site-packages\matplotlib\axes\_axes.py", line 4461, in scatter
    collection.update(kwargs)
  File "C:\Program Files\Python310\lib\site-packages\matplotlib\artist.py", line 1064, in update
    raise AttributeError(f"{type(self).__name__!r} object "
AttributeError: 'PathCollection' object has no property 'h'

========================================================== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =========================================================

========================================================== RESTART: D:\Research\papers\finance\iimv\python-scripts\cluster-analysis.py =========================================================
index = pd.date_range("1 1 2000", periods=100, freq="m", name="date")
                         
data = np.random.randn(100, 4).cumsum(axis=0)
                         
data.head()
                         
Traceback (most recent call last):
  File "<pyshell#64>", line 1, in <module>
    data.head()
AttributeError: 'numpy.ndarray' object has no attribute 'head'
data[1:6, ]
                         
array([[-1.20115043,  1.70480367, -1.50319218,  0.33079978],
       [-0.71501045,  1.73426294, -0.76029132,  0.87081301],
       [ 0.47017594,  1.29523478, -0.00311834,  1.25025858],
       [ 1.36096537,  1.78503973,  1.95702754,  0.47760986],
       [ 2.52567226,  1.36805204,  1.5637331 ,  0.96807673]])
wide_df = pd.DataFrame(data, index, ["a", "b", "c", "d"])
                         
wide_df.head()
                         
                   a         b         c         d
date                                              
2000-01-31 -0.334697  0.924859  0.179379 -0.336094
2000-02-29 -1.201150  1.704804 -1.503192  0.330800
2000-03-31 -0.715010  1.734263 -0.760291  0.870813
2000-04-30  0.470176  1.295235 -0.003118  1.250259
2000-05-31  1.360965  1.785040  1.957028  0.477610
sns.scatterplot(data=wide_df)
                         
<AxesSubplot:xlabel='date'>
plt.show()
                         
cluster_data.groupby('Cluster')
                         
<pandas.core.groupby.generic.DataFrameGroupBy object at 0x0000019B51CFB2E0>
cluster_data.groupby('Cluster').count
                         
<bound method GroupBy.count of <pandas.core.groupby.generic.DataFrameGroupBy object at 0x0000019B51CFB430>>
cluster_data.groupby('Cluster').count()
                         
         audit  auditability  auditable  ...  service  services  technology
Cluster                                  ...                               
0           21            21         21  ...       21        21          21
1            1             1          1  ...        1         1           1
2            8             8          8  ...        8         8           8
3            1             1          1  ...        1         1           1
4            1             1          1  ...        1         1           1

[5 rows x 24 columns]
cluster_data.groupby('Cluster').values()
                         
Traceback (most recent call last):
  File "<pyshell#73>", line 1, in <module>
    cluster_data.groupby('Cluster').values()
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\groupby\groupby.py", line 904, in __getattr__
    raise AttributeError(
AttributeError: 'DataFrameGroupBy' object has no attribute 'values'
cluster_data.groupby('Cluster').values
Traceback (most recent call last):
  File "<pyshell#74>", line 1, in <module>
    cluster_data.groupby('Cluster').values
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\groupby\groupby.py", line 904, in __getattr__
    raise AttributeError(
AttributeError: 'DataFrameGroupBy' object has no attribute 'values'
cluster_data.groupby('audit').count()
       auditability  auditable  audited  ...  services  technology  Cluster
audit                                    ...                               
0                17         17       17  ...        17          17       17
1                 3          3        3  ...         3           3        3
2                 3          3        3  ...         3           3        3
5                 1          1        1  ...         1           1        1
6                 2          2        2  ...         2           2        2
7                 1          1        1  ...         1           1        1
8                 1          1        1  ...         1           1        1
9                 1          1        1  ...         1           1        1
10                1          1        1  ...         1           1        1
16                1          1        1  ...         1           1        1
18                1          1        1  ...         1           1        1

[11 rows x 24 columns]
cluster_data.groupby('audit').values()
Traceback (most recent call last):
  File "<pyshell#76>", line 1, in <module>
    cluster_data.groupby('audit').values()
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\groupby\groupby.py", line 904, in __getattr__
    raise AttributeError(
AttributeError: 'DataFrameGroupBy' object has no attribute 'values'
cluster_data.groupby('audit').values
Traceback (most recent call last):
  File "<pyshell#77>", line 1, in <module>
    cluster_data.groupby('audit').values
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\groupby\groupby.py", line 904, in __getattr__
    raise AttributeError(
AttributeError: 'DataFrameGroupBy' object has no attribute 'values'
grouped_counts = cluster_data.groupby(['Cluster']).counts()
Traceback (most recent call last):
  File "<pyshell#78>", line 1, in <module>
    grouped_counts = cluster_data.groupby(['Cluster']).counts()
  File "C:\Program Files\Python310\lib\site-packages\pandas\core\groupby\groupby.py", line 904, in __getattr__
    raise AttributeError(
AttributeError: 'DataFrameGroupBy' object has no attribute 'counts'. Did you mean: 'count'?
grouped_counts = cluster_data.groupby(['Cluster']).count()
grouped_counts
         audit  auditability  auditable  ...  service  services  technology
Cluster                                  ...                               
0           21            21         21  ...       21        21          21
1            1             1          1  ...        1         1           1
2            8             8          8  ...        8         8           8
3            1             1          1  ...        1         1           1
4            1             1          1  ...        1         1           1

[5 rows x 24 columns]
grouped_counts = cluster_data.groupby(['Cluster']).mean()
grouped_counts
             audit  auditability  auditable  ...   service  services  technology
Cluster                                      ...                                
0         0.333333           0.0   0.047619  ...  0.190476  0.285714    0.857143
1        18.000000           0.0   0.000000  ...  0.000000  1.000000    1.000000
2         8.375000           0.0   0.000000  ...  0.000000  0.000000    0.875000
3         2.000000           0.0   0.000000  ...  4.000000  4.000000    0.000000
4         0.000000           1.0   0.000000  ...  7.000000  0.000000    5.000000

[5 rows x 24 columns]
grouped_counts.T
Cluster               0     1      2     3     4
audit          0.333333  18.0  8.375   2.0   0.0
auditability   0.000000   0.0  0.000   0.0   1.0
auditable      0.047619   0.0  0.000   0.0   0.0
audited        0.095238   0.0  0.125   0.0   0.0
auditees       0.000000   0.0  0.125   0.0   0.0
auditing       2.238095   6.0  2.000  18.0   1.0
auditor        0.095238   3.0  1.125   1.0   0.0
auditors       0.333333  14.0  1.125   0.0   0.0
audits         0.047619   0.0  0.375   0.0   0.0
blockchain     0.714286   0.0  0.000   0.0  11.0
blockchains    0.190476   0.0  0.000   0.0   0.0
fraud          1.047619   0.0  0.125   0.0   0.0
frauds         0.047619   0.0  0.000   0.0   0.0
fraudsters     0.047619   0.0  0.000   0.0   0.0
fraudulent     0.095238   0.0  0.000   0.0   0.0
protection     0.095238   0.0  0.000   0.0   0.0
quality        0.285714   0.0  0.500   2.0   0.0
risk           0.095238   3.0  0.125   0.0   0.0
risks          0.000000   3.0  0.250   0.0   0.0
cybersecurity  0.047619   0.0  0.000   0.0   0.0
security       0.523810   0.0  0.125   0.0   0.0
service        0.190476   0.0  0.000   4.0   7.0
services       0.285714   1.0  0.000   4.0   0.0
technology     0.857143   1.0  0.875   0.0   5.0
grouped_counts.T.to_csv('group-means.csv')
cluster_data.groupby(['Cluster']).max()
         audit  auditability  auditable  ...  service  services  technology
Cluster                                  ...                               
0            2             0          1  ...        3         2           5
1           18             0          0  ...        0         1           1
2           16             0          0  ...        0         0           4
3            2             0          0  ...        4         4           0
4            0             1          0  ...        7         0           5

[5 rows x 24 columns]
max_docs_per_cluster = cluster_data.groupby(['Cluster']).max()
max_docs_per_cluster.T.to_csv('max-docs-per-cluster.csv')
