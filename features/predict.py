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
from sklearn.svm import SVR
from sklearn.feature_selection import RFECV
import pdb


def get_predictable_features(table):
    def match_func(col):
        if not col.metadata['numeric']:
            return False

        if col.metadata["path"] and col.metadata["path"][-1]["feature_type"] == "flat":
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

        return True

    return target_col.dsm_table.get_column_info(match_func=match_func)

def make_data(target_col, feature_cols):
    table = target_col.dsm_table
    print "feature_cols 1", feature_cols
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

    return rows, y


def model(target_col, feature_cols):
    rows, y = make_data(target_col, feature_cols)

    estimator = SVR(kernel='linear')
    clf = Pipeline([
            ('vect', DictVectorizer(sparse=False)),
            ('scaler', preprocessing.StandardScaler()),
            ('regression', linear_model.LinearRegression())
    ])

    print "fit"

    clf.fit(rows, y)
    print "score"
    score = clf.score(rows, y)
    print "done fit and score"
    names = target_col.dsm_table.names_to_cols(clf.named_steps['vect'].get_feature_names())
    weights = clf.named_steps["regression"].coef_

    print names
    print weights
    using = zip(names, weights)
    # using = zip(using,clf.named_steps['regression'].coef_)


    return score, using




def best_model(target_col):
    predict_cols = get_usable_features(target_col)
    rows, y = make_data(target_col, predict_cols)

    estimator = SVR(kernel='linear')
    clf = Pipeline([
            ('vect', DictVectorizer(sparse=False)),
            ('scaler', preprocessing.StandardScaler()),
            ('selector', RFECV(estimator, step=1, cv=3, scoring='r2')),
    ])

    clf.fit(rows,y)

    names = clf.named_steps['vect'].get_feature_names()
    support = clf.named_steps['selector'].support_ 
    support_cols = [target_col.dsm_table.get_col_by_name(n) for i,n in enumerate(names) if support[i]]
    # using = zip(using,clf.named_steps['regression'].coef_)
    # using = sorted(using, key=lambda x: -abs(x[1]))
    print names, support
    return model(target_col, support_cols)




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
    
    skip = 0
    for target_col in all_cols:
        skip -= 1
        if skip >=0:
            print "skip %s" % target_col.name
            continue
        # pdb.set_trace()
        # print "predict %s" % target_col.metadata["real_name"], score, 'using:', using
        print best_model(target_col)
        
        # selector = 
        # selector = selector.fit(X, y)
        # selector.support_ 
        # pdb.set_trace()
        

        




if __name__ == "__main__":
    import os
    # USE_CACHE = False

    # os.system("mysql -t < ../Northwind.MySQL5.sql")
    # database_name = 'northwind'
    # db = Database('mysql+mysqldb://kanter@localhost/%s' % (database_name)) 

    table_name = "Products"
    db = Database.load(table_name)
    table = db.tables[table_name]
    # make_all_features(db, db.tables[table_name])
    find_all_correlations(db, db.tables[table_name])