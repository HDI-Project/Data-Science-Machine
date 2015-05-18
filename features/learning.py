from sklearn.utils import ConvergenceWarning
import warnings
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, Imputer
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVR, LinearSVR, LinearSVC, SVC
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold, SelectPercentile
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
import random
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import scale
from sklearn.decomposition import RandomizedPCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import train_test_split, cross_val_score
from scipy import sparse
import threading
import pdb
from collections import OrderedDict
from sklearn.base import BaseEstimator, ClassifierMixin

class DSMClassifier(BaseEstimator, ClassifierMixin):
    params = {
        "k" : 1,
        "k_best_percent" : 100,
        "n_components" : 100,
        "reweight_ratio" : 1,
        "n_estimators" : 200,
        "max_depth" : None,
        "max_features" : 50,
    }
    def __init__(self, cat_mask, params={}):
        self.cat_mask = cat_mask
        self.params.update(params)

    def reweight(self, y, reweight_ratio):
        """
        reweight under represented class by reweight_ratio. only apply to binary classifcation problems
        """
        classes = list(set(y))
        if len(classes) > 2:
            return [1] * len(y)

        class_1_count = len([x for x in y if x==classes[0]])
        class_2_count = len([x for x in y if x==classes[1]])
        if class_1_count <= class_2_count:
            under_rep = classes[0]
        else:
            under_rep = classes[1]

        sample_weight = np.array([reweight_ratio if i == under_rep else 1 for i in y])
        return sample_weight

    def fit_transform_data_pipeline(self, X,y):
        X = np.nan_to_num(X)
        self.data_pipeline_1 = Pipeline([
            # ("imputer", Imputer(missing_values="NaN",strategy=, axis=0)), #todo test good
            ('encoder', OneHotEncoder(categorical_features=self.cat_mask, sparse=True, handle_unknown="ignore")),
            ('scaler', preprocessing.StandardScaler(with_mean=False)),
            ('var', VarianceThreshold()),
        ])


        X = self.data_pipeline_1.fit_transform(X, y)
        num_features = X.shape[1]
        n_components = min(self.params["n_components"], num_features)

        if n_components < num_features:
            self.data_pipeline_2 = Pipeline([
                # ('ch2', SelectKBest(f_classif, k=k))
                ('pca', TruncatedSVD(n_components=n_components)),
            ])

            X = self.data_pipeline_2.fit_transform(X,y)
        else:
            self.data_pipeline_2 = None 

        return X

    def transform_data_pipeline(self,X):
        X = np.nan_to_num(X)
        X = self.data_pipeline_1.transform(X)
        # try:
        #     X = X.toarray()
        # except:
        #     pass
        if self.data_pipeline_2 != None:
            X = self.data_pipeline_2.transform(X)
        return X

    def fit(self, X, y):
        self.cluster_clf = {}
        self.cluster_pipeline = Pipeline([
            # ('scaler', preprocessing.StandardScaler(with_mean=True)),
            ('cluster', MiniBatchKMeans(init='k-means++', n_clusters=self.params["k"], batch_size=1000,
                          n_init=10, max_no_improvement=10, verbose=0,
                          random_state=0)),
        ])

    

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            warnings.simplefilter('ignore', ConvergenceWarning)
            # pdb.set_trace()
            X = self.fit_transform_data_pipeline(X,y)
            num_features = X.shape[1]
            # print data_pipeline.named_steps["selector"].scores_

        self.cluster_pipeline.fit(X)
        labels = self.cluster_pipeline.predict(X)


        # pdb.set_trace()
        k_best = int(.01*self.params["k_best_percent"]*num_features)
        for distinct_label in set(labels):
            estimator = LinearSVC()


            rf = RandomForestClassifier(n_jobs=-1, verbose=1, n_estimators=self.params["n_estimators"], max_depth=self.params["max_depth"], max_features=self.params["max_features"]*.01)
            clf_pipeline = Pipeline([
                    # ('pca', RandomizedPCA(n_components=n_components)),
                    # ('pca', TruncatedSVD(n_components=n_components)),
                    ('ch2', SelectKBest(f_classif, k=k_best)),
                    ('clf', rf)
                    # ('clf', GradientBoostingClassifier(verbose=1, n_estimators=self.params["n_estimators"], subsample=self.params["gbm_subsample"]*.01, max_depth=self.params["gbm_depth"], learning_rate=.01))
            ])

            if k_best == num_features:
                
                clf_pipeline = Pipeline([
                    ('clf', rf)
                ])  

            cluster_x = X[labels==distinct_label]#.toarray() #sparse to dense
            cluster_y = y[labels==distinct_label]

            print "cluster: ",distinct_label

            if len(set(cluster_y)) < 2:
                self.cluster_clf[distinct_label] = None
                continue

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                warnings.simplefilter('ignore', ConvergenceWarning)
                sample_weight = self.reweight(cluster_y, self.params["reweight_ratio"])
                # if n_components == num_features:
                #     cluster_x = cluster_x
                try:
                    cluster_x = cluster_x.toarray()
                except:
                    pass
                
                # cluster_x = cluster_x.toarray()
                clf_pipeline.fit(cluster_x,cluster_y, clf__sample_weight=sample_weight)

            self.cluster_clf[distinct_label] = clf_pipeline

        return self

    def predict_proba(self, X):
        all_probs = np.array([[.5,.5]]*len(X))
        X = self.transform_data_pipeline(X)
        test_labels = self.cluster_pipeline.predict(X)
        for distinct_label in set(test_labels):
            cluster_x = X[test_labels==distinct_label]
            if self.cluster_clf[distinct_label] != None:
                clf = self.cluster_clf[distinct_label].named_steps["clf"]
                # try:
                #     cluster_x = cluster_x.toarray()
                # except:
                #     pass
                # cluster_x = cluster_x.toarray()
                probs = self.cluster_clf[distinct_label].predict_proba(cluster_x)
                all_probs[test_labels==distinct_label] = probs

        return all_probs

    def score(self, X, y):
        all_probs = self.predict_proba(X)[:,1]
        score = roc_auc_score(y, all_probs)
        return score