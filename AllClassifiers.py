from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree.tree import DecisionTreeClassifier
from sklearn.tree.tree import ExtraTreeClassifier
from sklearn.ensemble.forest import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process.gpc import GaussianProcessClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.semi_supervised.label_propagation import LabelPropagation
from sklearn.semi_supervised.label_propagation import LabelSpreading
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.linear_model.logistic import LogisticRegressionCV
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.svm.classes import NuSVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.svm.classes import SVC
import os
import warnings
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import pickle
from sklearn.model_selection import train_test_split
import shutil
from statistics import mean

warnings.filterwarnings('ignore')

classifiers = [
    AdaBoostClassifier(),
    BaggingClassifier(),
    BernoulliNB(),
    CalibratedClassifierCV(),
    DecisionTreeClassifier(),
    ExtraTreeClassifier(),
    ExtraTreesClassifier(),
    GaussianNB(),
    GaussianProcessClassifier(),
    GradientBoostingClassifier(),
    KNeighborsClassifier(),
    LabelPropagation(),
    LabelSpreading(),
    LinearDiscriminantAnalysis(),
    LogisticRegression(),
    LogisticRegressionCV(),
    MLPClassifier(),
    NuSVC(probability=True),
    QuadraticDiscriminantAnalysis(),
    RandomForestClassifier(),
    SGDClassifier(loss='log'),
    SVC(probability=True),
    XGBClassifier()
]

names = [
    'AdaBoostClassifier',
    'BaggingClassifier',
    'BernoulliNB',
    'CalibratedClassifierCV',
    'DecisionTreeClassifier',
    'ExtraTreeClassifier',
    'ExtraTreesClassifier',
    'GaussianNB',
    'GaussianProcessClassifier',
    'GradientBoostingClassifier',
    'KNeighborsClassifier',
    'LabelPropagation',
    'LabelSpreading',
    'LinearDiscriminantAnalysis',
    'LogisticRegression',
    'LogisticRegressionCV',
    'MLPClassifier',
    'NuSVC',
    'QuadraticDiscriminantAnalysis',
    'RandomForestClassifier',
    'SGDClassifier',
    'SVC',
    'XGBClassifier'
]

print(len(names), len(classifiers))
# exit()
fnames = ['dt_end', 'dt_start', 'start', 'label', 'tp', 'before_avg', 'before_density',
          'day_of_week', 'kind', 'red_diff', 'blu_diff', 'diff_to_first', 'result']

base = './files/all/'

i = len(classifiers) - 1

model_base = './files/models/'
try:
    shutil.rmtree(model_base)
except OSError as e:
    print("NOT EXIST")
if not os.path.exists(model_base):
    os.makedirs(model_base)


def analyze(percent=80):
    files = os.listdir(base)

    length = 90
    for name, clf in zip(names, classifiers):
        allnet = 0
        alltotal = 0
        print('=====-=--=---------=============--------======-========' + name + '===-=--=-------===========-=====')
        for file in files:
            listofindex = []
            net = 0
            total = 0
            df = pd.read_csv(base + file, sep=',', names=fnames, skiprows=1)
            q = len(df.columns) - 1

            for start_index, starter in df.iterrows():
                start_dt = starter[1]
                c = 0
                for end_index, ender in df.iterrows():
                    end_dt = ender[0]
                    if start_dt >= end_dt:
                        last_end_index = end_index
                        c = 1
                if c == 1:
                    listofindex.append([last_end_index, start_index])
            end_len = len(listofindex)
            print(end_len)
            for i in range(length, end_len - 1):
                indexes = listofindex[i]
                X = df.iloc[indexes[0] + 1 - length:indexes[0] + 1, 2:q]
                Y = df.iloc[indexes[0] + 1 - length:indexes[0] + 1, q]
                if len(Y.tolist()) < 1:
                    continue
                if mean(Y.tolist()) < .4:
                    continue
                sX = df.iloc[indexes[1]:indexes[1] + 1, 2:q]
                sY = df.iloc[indexes[1]:indexes[1] + 1, q]

                X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

                try:
                    clf.fit(X_train, y_train)
                    score = clf.score(X_test, y_test)
                except ValueError as e:
                    score = 0
                if score * 100 > percent:
                    clf.fit(X, Y)
                    y_pred = clf.predict(sX)
                    proba = clf.predict_proba(sX)
                    if y_pred[0] == 1 and proba[0][1] > .91:
                        total += 1
                        if sY.tolist()[0] == 1:

                            net += 1

            allnet += net
            alltotal += total
            if total > 0:
                print(file, net * 100 / total, net, total)
        if alltotal > 0:
            print(allnet * 100 / alltotal, allnet, alltotal)
            print('=====-=--=---------=============--------======-========' + name + '===-=--=-------===========-=====')


if __name__ == '__main__':
    for i in range(0, 1):
        analyze(80)
