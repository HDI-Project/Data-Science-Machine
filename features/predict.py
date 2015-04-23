from make_features import make_all_features
from filters import FilterObject


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
from sklearn.svm import SVR, LinearSVR, LinearSVC, SVC
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
import random
from sklearn.cluster import MiniBatchKMeans
from time import time
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import train_test_split
from itertools import izip
from scipy import sparse
from sklearn.externals import joblib
import pdb


def get_predictable_features(table, exlucde=[]):
    def match_func(col, exlucde=exlucde):
        if not (col.metadata['numeric'] or col.metadata['categorical']):
            return False
        # path_tables = set([p['base_column'].dsm_table.name for p in col.metadata['path']])
        # pdb.set_trace()
        if col.metadata["path"] and col.metadata["path"][-1]['base_column'].dsm_table.name in exlucde > 0:
            return False

        # if col.metadata["path"] and col.metadata["path"][-1]["feature_type"] == "flat":
        #     return False

        # if len(col.get_distinct_vals()) < 2:
        #     return False

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

def make_data(target_col, feature_cols, id_col, filter_obj=None):
    """
    todo: make this one query
    """
    table = target_col.dsm_table

    id_col_vals = []
    y = []
    rows = []
    for r in table.get_rows([target_col, id_col]+feature_cols, filter_obj=filter_obj):
        y_add = r[0]
        id_add = r[1]
        # print r
        if y_add == None:
            y_add = 0 

        y.append(float(y_add)) 
        id_col_vals.append(id_add) 
        rows.append(r[2:])

    rows = np.array(rows,  dtype=float)
    num = 10000
    return rows,y, id_col_vals
    # return rows[:num], y[:num]

def split_data(x,y, ids, test_ids_file, id_type=int):
    """
    splits training/test data according to the list of ids in test_ids_file
    """

    with open(test_ids_file) as f:
        test_id_set = set(map(id_type,f.read().splitlines()))

    
    y = np.array(y)
    ids = np.array(ids)

    test_idx = []
    train_idx = []
    for idx, curr_id in enumerate(ids):
        if curr_id in test_id_set:
            test_idx.append(idx)
        else:
            train_idx.append(idx)

    train_x = x[train_idx]
    test_x = x[test_idx]
    train_y = y[train_idx]
    test_y = y[test_idx]
    train_ids = ids[train_idx]
    test_ids = ids[test_idx]

    # pdb.set_trace()
    return train_x, test_x, train_y, test_y, train_ids, test_ids

def write_predictions(outfile, row_ids, probs, header=None, convert_id_func=None):
    with open(outfile, "w") as out:
        if header:
            out.write(header+ "\n")
        if convert_id_func != None:
            row_ids = convert_id_func(row_ids)

        for row_id, prob in izip(row_ids, probs):
            out.write(str(row_id) + "," + str(prob) + "\n")





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
    X = np.where(X == np.array(None), 0, X)
    labels, X_scaled = cluster_labels(X, k)
    # pdb.set_trace()
    
    important_features = {}
    distinct_labels = set(labels)
    for l in distinct_labels:
        predict_labels = labels.copy()
        predict_labels[labels == l] = 1
        predict_labels[labels != l] = 0
        
        clf = LinearSVR()
        clf.fit(X_scaled, predict_labels)
        score = clf.score(X_scaled, predict_labels)
        weights = clf.coef_
        weights = [abs(w) for w in weights]
        thresh = np.mean(weights) + np.std(weights)*3

        important_features[str(l)] = []
        for col, weight in zip(cols, weights):
            if weight > thresh:
                important_features[str(l)].append([col,weight])
    
        important_features[str(l)].sort(key=lambda x: -abs(x[1]))

    # reduced_data = PCA(n_components=2).fit_transform(X_scaled)
    reduced_data = PCA(n_components=50).fit_transform(X_scaled)
    tsne = TSNE(n_components=2, random_state=0,verbose=1)
    reduced_data = tsne.fit_transform(reduced_data) 
    # kmeans = MiniBatchKMeans(init='k-means++', n_clusters=k, n_init=10)
    # labels = kmeans.fit_predict(reduced_data)

    clusters = {}
    for pt,label in zip(reduced_data, labels):
        label = str(label)
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(list(pt))

    return clusters, important_features


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

        
def model(id_col, target_col, feature_cols, filter_obj=None, outfile=None, header=None, convert_id_func=None, test_ids_file=None, id_type=int):
    num_features = len(feature_cols)
    feature_cols = list(feature_cols)
    rows, y, ids = make_data(target_col, feature_cols, id_col, filter_obj)
    rows = np.nan_to_num(rows)
    train_x, test_x, train_y, test_y, train_ids, test_ids = split_data(rows,y, ids, test_ids_file=test_ids_file, id_type=id_type)    

    # train_x, test_x_2, train_y, test_y_2 = train_test_split(train_x, train_y, test_size=0.33, random_state=42)

    k=2
    cluster_clf = {}
    cluster_pipeline = Pipeline([
        ('scaler', preprocessing.StandardScaler(with_mean=True)),
        ('cluster', MiniBatchKMeans(init='k-means++', n_clusters=k, batch_size=1000,
                      n_init=10, max_no_improvement=10, verbose=1,
                      random_state=0)),

    ])


    k_best = 50
    if 50 >= num_features:
        k_best ="all"
    pca_components = min(num_features, 5)
    pdb.set_trace()

    cluster_pipeline.fit(train_x)
    labels = cluster_pipeline.predict(train_x)
    # pdb.set_trace()
    for distinct_label in set(labels):
        estimator = LinearSVC()
        data_pipeline = Pipeline([
                # ('vect', DictVectorizer()),
                ('var', VarianceThreshold()),
                ('scaler', preprocessing.StandardScaler(with_mean=True)),
                ('ch2', SelectKBest(f_classif, k=k_best)),
                ('pca', PCA(n_components=pca_components)),
                # ('selector', RFECV(estimator, step=1, cv=3, verbose=1)),
                # ('selector', linear_model.RandomizedLasso(selection_threshold=0.5,n_jobs=-1, verbose=1)),
                ('clf', GradientBoostingClassifier(verbose=1,subsample=.7, max_depth=5, learning_rate=.01))
        ])
        cluster_x = train_x[labels==distinct_label]
        cluster_y = train_y[labels==distinct_label]


        data_pipeline.fit(cluster_x,cluster_y)
        cluster_clf[distinct_label] = data_pipeline


    # all_probs = []
    # all_y = []
    # test_labels = cluster_pipeline.predict(test_x_2)
    # pdb.set_trace()
    # for distinct_label in set(test_labels):
    #     cluster_x = test_x_2[test_labels==distinct_label]
    #     cluster_y =  test_y_2[test_labels==distinct_label]
    #     clf = cluster_clf[distinct_label].named_steps["clf"]
    #     # print "n features: ", data_pipeline.named_steps["selector"].n_features_
    #     if clf.classes_[0] == 1:
    #         pos_idx = 0
    #     elif clf.classes_[1] == 1:
    #         pos_idx = 1

    #     probs = cluster_clf[distinct_label].predict_proba(cluster_x)[:,pos_idx].tolist()
    #     all_probs += probs
    #     all_y += list(cluster_y)

    # auc = roc_auc_score(all_y, all_probs)
    # print auc


    all_probs = []
    all_ids = []
    test_labels = cluster_pipeline.predict(test_x)
    for distinct_label in set(test_labels):
        cluster_x = test_x[test_labels==distinct_label]
        cluster_ids =  test_ids[test_labels==distinct_label]   

        if len(cluster_x) < 0:
            continue

        clf = cluster_clf[distinct_label].named_steps["clf"]
        # print "n features: ", data_pipeline.named_steps["selector"].n_features_
        if clf.classes_[0] == 1:
            pos_idx = 0
        elif clf.classes_[1] == 1:
            pos_idx = 1

        probs = cluster_clf[distinct_label].predict_proba(cluster_x)[:,pos_idx].tolist()
        all_probs += probs
        all_ids += list(cluster_ids)
        # print set(probs)

    # print set(all_probs)
    write_predictions(outfile, all_ids, all_probs, header=header, convert_id_func=convert_id_func)
    
        




if __name__ == "__main__":
    import os
    # USE_CACHE = False

    # os.system("mysql -t < ../Northwind.MySQL5.sql")
    # database_name = 'northwind'
    # db = Database('mysql+mysqldb://kanter@localhost/%s' % (database_name)) 

    # table_name = "Outcomes"
    # model_name = "ijcai__" + table_name
    # print "models/%s"%model_name
    # db = Database.load("models/%s"%model_name)
    # print "db loaded"
    # table = db.tables[table_name]
    # target_col = table.get_col_by_name("label")
    # id_col = table.get_col_by_name("id")
    # feature_cols = get_predictable_features(table)
    # random.shuffle(feature_cols)
    # feature_cols = feature_cols
    # def convert_id_func(row_ids):
    #     with open("../datasets/ijcai/test_ids_convert.csv") as f:
    #         return np.array(f.read().splitlines())
    # model(id_col,target_col,feature_cols, outfile="ijcai_predict.csv", header="user_id,merchant_id,prob", convert_id_func=convert_id_func, test_ids_file="../datasets/ijcai/test_ids.csv")
    
    table_name = "Projects"
    print "models/%s"%table_name
    db = Database.load("models/%s"%table_name)
    print "db loaded"
    table = db.tables[table_name]
    target_col = table.get_col_by_name("Outcome.is_exciting")
    id_col = table.get_col_by_name("projectid")
    feature_cols = get_predictable_features(table, exlucde=["Donations", "Outcomes"])
    random.shuffle(feature_cols)
    feature_cols = feature_cols
    filter_col = table.get_col_by_name("date_posted")
    filter_obj = FilterObject([(filter_col, ">", "2013-1-1")])
    filter_obj=None
    model(id_col,target_col,feature_cols, filter_obj, outfile="donors_predict.csv", header="projectid,is_exciting", test_ids_file="../datasets/donorschoose/test_ids.csv", id_type=str)
