import hdbscan
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

base = './files/all/'
fnames = ['dt_end', 'dt_start', 'start', 'label', 'tp', 'before_avg', 'before_density',
          'day_of_week', 'kind', 'red_diff', 'blu_diff', 'diff_to_first', 'result']

files = os.listdir(base)
for file in files:
    df = pd.read_csv(base + file, sep=',', names=fnames, skiprows=1)
    q = len(df.columns) - 1

    X = df.iloc[:, 2:q]
    Y = df.iloc[:, q]
    X = StandardScaler().fit_transform(X)
    X = PCA(n_components=2).fit_transform(X)

    clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
                                gen_min_span_tree=False, leaf_size=40, metric='manhattan', min_cluster_size=15,
                                min_samples=None, p=None)
    clusterer.fit(X)

    for no in range(0, 6):
        b = 0
        c = 0
        d = 0
        for i in range(len(clusterer.labels_)):
            if clusterer.labels_[i] == no and clusterer.probabilities_[i] > .7:
                # print(clusterer.labels_[i], Y[i])
                b += 1
                if Y[i] == 1:
                    c += 1

                if Y[i] == 0:
                    d += 1

        if b > 0:
            if c / b > .75:
                print('WIN', file, c * 100 / b, c, b)
            if d / b > .75:
                print('LOSS', file, d * 100 / b, d, b)
