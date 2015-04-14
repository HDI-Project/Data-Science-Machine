from make_features import make_all_features
from database import Database
import numpy as np
from sklearn import tree
from sklearn import cross_validation
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVR, LinearSVR, LinearSVC
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor 
import random
from sklearn.cluster import MiniBatchKMeans
from time import time
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import train_test_split
import pdb


def get_predictable_features(table):
    def match_func(col):
        if not (col.metadata['numeric'] or col.metadata['categorical']):
            return False

        # if col.metadata["path"] and col.metadata["path"][-1]["feature_type"] == "flat":
        #     return False

        if len(col.get_distinct_vals()) < 2:
            return False

        return True

    return table.get_column_info(match_func=match_func)

def get_usable_features(target_col):
    def match_func(col, target_col=target_col):
        target_col_path = set([p['base_column'].unique_name for p in target_col.metadata['path']])
        usable_col_path = set([p['base_column'].unique_name for p in col.metadata['path']])
        
        if target_col_path.intersection(usable_col_path) != set([]):
            return False

        if not col.metadata['numeric']:
            return False

        if col == target_col:
            return False

        if len(col.get_distinct_vals()) < 2:
            return False

        return True

    return target_col.dsm_table.get_column_info(match_func=match_func)

def make_data(target_col, feature_cols):
    table = target_col.dsm_table
    # print "feature_cols 1", feature_cols
    rows = table.get_rows_as_dict(feature_cols) 
    for row in rows:
        for col in row:
            if row[col] == None:
                row[col] = 0

    y = []
    for r in table.get_rows_as_dict([target_col]):
        r = r[target_col.name] #skip primary key
        # print r
        if r == None:
            r = 0 

        y.append(float(r)) 

    num = 10000
    return rows,y
    # return rows[:num], y[:num]

def split_data(x,y):
    return train_test_split(x,y, test_size=.5)


def model(target_col, feature_cols):
    print "make model", len(feature_cols)
    rows, y = make_data(target_col, feature_cols)

    if target_col.metadata["categorical"]:
        clf = linear_model.SGDClassifier(loss="log")
    else:
        clf = LinearSVR()

    data_pipeline = Pipeline([
            ('vect', DictVectorizer()),
            ('scaler', preprocessing.StandardScaler(with_mean=False)),
    ])

    # print "fit"

    X = data_pipeline.fit_transform(rows)

    train_x, test_x, train_y, test_y = split_data(X,y)

    clf.fit(train_x, train_y)
    
    if target_col.metadata["categorical"]:
        if clf.classes_[0] == 1:
            pos_idx = 0
        elif clf.classes_[1] == 1:
            pos_idx = 1

        probs = clf.predict_proba(test_x)[:,pos_idx]
        score = roc_auc_score(test_y, probs)
        print score
    else:
        score = clf.score(X, y)
    # print "done fit and score"
    names = target_col.dsm_table.names_to_cols(data_pipeline.named_steps['vect'].get_feature_names())
    weights = clf.coef_

    # print names
    # print weights
    using = zip(names, weights)
    # using = zip(using,clf.named_steps['regression'].coef_)

    print score, "\n\n", using
    pdb.set_trace()

    return score, using



def best_model(target_col):
    predict_cols = get_usable_features(target_col)
    rows, y = make_data(target_col, predict_cols)

    if target_col.metadata["categorical"]:
        estimator = LinearSVC(dual=False)
    else:
        estimator = LinearSVR(dual=False)

    data_pipeline = Pipeline([
            ('vect', DictVectorizer(sparse=False)),
            ('scaler', preprocessing.StandardScaler()),
            # ('regression', linear_model.RandomizedLogisticRegression(n_jobs=-1)),
            # ('selector', RFECV(estimator, step=1, cv=3, scoring='r2')),
    ])

    X = data_pipeline.fit_transform(rows)
    from copy import copy
    X = copy(X)

    # clf = linear_model.RandomizedLogisticRegression(n_jobs=-1, verbose=True, pre_dispatch='2*n_jobs', n_resampling="10")
    clf = linear_model.RandomizedLasso(n_jobs=-1)
    clf.fit(X,y)

    names = np.array(data_pipeline.named_steps['vect'].get_feature_names())
    importances = clf.scores_
    important_names = names[importances > np.mean(importances) + np.std(importances)]
    support_cols = [target_col.dsm_table.get_col_by_name(n) for n in important_names]
    # support = clf.named_steps['selector'].support_ 
    # support_cols = [target_col.dsm_table.get_col_by_name(n) for i,n in enumerate(names) if support[i]]
    # using = zip(using,clf.named_steps['regression'].coef_)
    # using = sorted(using, key=lambda x: -abs(x[1]))
    # print names, support

    if len(support_cols) == 0:
        return 0, []

    return model(target_col, support_cols)

def cluster_labels(X, k, scale_data=True):
    X = np.where(X == np.array(None), 0, X)

    if scale_data:
        X = scale(X)

    mbk = MiniBatchKMeans(init='k-means++', n_clusters=k, batch_size=1000,
                      n_init=10, max_no_improvement=10, verbose=1,
                      random_state=0)
    labels = mbk.fit_predict(X)
    return labels, X


def cluster(entity,k,cols=None):
    if cols == None:
        cols = entity.get_column_info(match_func=lambda col: col.metadata['numeric'])
    cols = np.array(cols)

    data = entity.get_rows(cols, limit=1000)
    X = [d for d in data]
    
    labels, X_scaled = cluster_labels(X, k)
    # pdb.set_trace()
    
    important_features = {}
    distinct_labels = set(labels)
    for l in distinct_labels:
        predict_labels = labels.copy()
        predict_labels[labels == l] = 1
        predict_labels[labels != l] = 0
        
        clf = LinearSVR()
        clf.fit(X, predict_labels)
        score = clf.score(X, predict_labels)
        weights = clf.coef_
        weights = [abs(w) for w in weights]
        thresh = np.mean(weights) + np.std(weights)*3

        important_features[str(l)] = []
        for col, weight in zip(cols, weights):
            if weight > thresh:
                important_features[str(l)].append([col,weight])
    
        important_features[str(l)].sort(key=lambda x: -abs(x[1]))

    reduced_data = PCA(n_components=2).fit_transform(X)
    # tsne = TSNE(n_components=2, random_state=0,verbose=1)
    # reduced_data = tsne.fit_transform(reduced_data) 
    # kmeans = MiniBatchKMeans(init='k-means++', n_clusters=k, n_init=10)
    # labels = kmeans.fit_predict(reduced_data)

    clusters = {}
    for pt,label in zip(reduced_data, labels):
        label = str(label)
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(list(pt))

    return clusters, important_features



class IntegerEncoder():
    def __init__(self, categorical_features=[]):
        self.categorical_features = categorical_features
        self.col_mappings = dict([(i,{}) for i in categorical_features])
        self.col_counts = dict([(i,1) for i in categorical_features]) #0 reserved for unseen values

    def fit(self,X, y=None):
        new_X = []
        for row in X:
            new_row = []
            count = 0
            # pdb.set_trace()
            for col_num, element in enumerate(row):
                if col_num not in self.categorical_features:
                    if element == None:
                        element = 0
                    new_row.append(element)
                    continue

                if element == None:
                    element = ""

                if element not in self.col_mappings[col_num]:
                    self.col_mappings[col_num][element] = self.col_counts[col_num]
                    self.col_counts[col_num] += 1
                
                new_row.append(self.col_mappings[col_num][element])

            new_X.append(new_row)

        return new_X

    def transform(self,X, y=None):
        new_X = []
        for row in X:
            new_row = []
            for col_num, element in enumerate(row):
                if col_num not in self.categorical_features:
                    if element == None:
                        element = 0
                    new_row.append(element)
                    continue

                # if element not in self.col_mappings[col_num]:
                #     print 'adding element in transform'
                #     self.col_mappings[col_num][element] = self.col_counts[col_num]
                #     self.col_counts[col_num] += 1
                
                new_row.append(self.col_mappings[col_num].get(element, 0))

            new_X.append(new_row)

        return new_X

    def fit_transform(self,X, y=None):
        return self.fit(X)


def find_all_correlations(db, table):
    all_cols = get_predictable_features(table)
    random.shuffle(all_cols)
    try:
        skip = 0
        for target_col in all_cols:
            skip -= 1
            if skip >=0:
                print "skip %s" % target_col.name
                continue
            # pdb.set_trace()
            # print "predict %s" % target_col.metadata["real_name"], score, 'using:', using
            other_features = set(all_cols)
            other_features.remove(target_col)
            score, using = model(target_col, other_features)

            using = sorted(using, key=lambda x: -abs(x[1]))
            using = [(c.metadata["real_name"], w) for c, w in using ]


            print "score %f" % score
            print "predict %s" % target_col.metadata["real_name"]
            for u in using:
                print u[0], u[1]
            print
            print
            print
    except Exception, e:
        print e

        

        




if __name__ == "__main__":
    import os
    # USE_CACHE = False

    # os.system("mysql -t < ../Northwind.MySQL5.sql")
    # database_name = 'northwind'
    # db = Database('mysql+mysqldb://kanter@localhost/%s' % (database_name)) 

    table_name = "Outcomes"
    print "models/%s"%table_name
    db = Database.load("models/%s"%table_name)
    print "db loaded"
    table = db.tables[table_name]
    # make_all_features(db, db.tables[table_name])
    # find_all_correlations(db, db.tables[table_name])
    target_col = table.get_col_by_name("is_exciting")
    feature_cols = set(get_predictable_features(table))
    feature_cols.remove(target_col)
    print target_col, feature_cols
    model(target_col,feature_cols)
    # cluster(table, k=5)