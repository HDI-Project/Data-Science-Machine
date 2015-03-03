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

#b, c = np.unique(a, return_inverse=True)
#(b[c] == a).all() ==> True

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



# def find_correlations(db, table):
#     all_cols = table.get_column_info(match_func=lambda x: x['metadata']['numeric'] or x['metadata']['categorical'])
#     print all_cols
#     # pdb.set_trace()
#     for target in all_cols:
#         cols = [target]
#         cat_indicies = []
#         for i, predictor in enumerate(all_cols):
#             if predictor == target:
#                 continue

#             if not predictor['metadata']['numeric']:
#                 cat_indicies.append(i)

#             cols.append(predictor)

#         data = np.array(table.get_rows(cols).fetchall())

#         clf = Pipeline([
#             ('int_encoder', IntegerEncoder(categorical_features=cat_indicies)),
#             ('cat_encoder', OneHotEncoder(categorical_features=cat_indicies)),
#             ('scaler', preprocessing.StandardScaler()),
#             # ('clf', tree.DecisionTreeRegressor()),
#             ('clf', linear_model.LogisticRegression()),
#         ])

#         X = data[:,1:]
#         # X = np.nan_to_num(data[:,1:].astype(dtype=np.float))
#         # Y = data[:,0]
#         # Y = np.where(Y == np.array(None),'', Y)
#         Y = np.nan_to_num(data[:,0].astype(dtype=np.float))

        
#         scores = cross_validation.cross_val_score(clf, X, Y, scoring='r2', cv=5)
#         print "predict %s" % target['name']
#         print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#         # print ("importance", clf.named_steps['clf.scores_'])
#         # pdb.set_trace()
#     pdb.set_trace()


def find_correlations(db, table):
    all_cols = table.get_column_info(match_func=lambda x: x['metadata']['numeric'])
    
    skip = 3
    for target_col in all_cols:
        skip -= 1
        if skip >=0:
            print "skip %s" % target_col['name']
            continue
        path_cols = [p['base_column']['unique_name'] for p in target_col['metadata']['path']]
        print 'start predict'
        predict_cols = table.get_column_info(match_func=lambda col, path_cols=path_cols, target_col=target_col: set(path_cols).intersection(set([p['base_column']['unique_name'] for p in col['metadata']['path']])) == set([]) and col['metadata']['numeric'] and col!=target_col)

        rows = table.get_rows_as_dict(predict_cols) 


        for row in rows:
            for col in row:
                if row[col] == None:
                    row[col] = 0

        y = []
        for r in table.get_rows([target_col]).fetchall():
            r = r[0]
            if r == None:
                r = 0 

            y.append(float(r)) 

        estimator = SVR(kernel='linear')
        clf = Pipeline([
                ('vect', DictVectorizer(sparse=False)),
                ('scaler', preprocessing.StandardScaler()),
                ('selector', RFECV(estimator, step=1, cv=3, scoring='r2')),
                ('regression', linear_model.LinearRegression())
                # ('clf', linear_model.LogisticRegression()),
        ])

        # clf2 = Pipeline([
        #         ('vect', DictVectorizer(sparse=False)),
        #         ('scaler', preprocessing.StandardScaler()),
        #         ('regression', linear_model.RandomizedLogisticRegression())
        # ])


        clf.fit(rows,y)

        names = clf.named_steps['vect'].get_feature_names()
        support = clf.named_steps['selector'].support_ 
        using = [n for i,n in enumerate(names) if support[i]]
        using = zip(using,clf.named_steps['regression'].coef_)

        print "predict %s" % target_col['name'], clf.score(rows,y), 'using:', sorted(using, key=lambda x: -abs(x[1]))
        # pdb.set_trace()
        print 
        print 
        
        # selector = 
        # selector = selector.fit(X, y)
        # selector.support_ 
        # pdb.set_trace()
        

        




if __name__ == "__main__":
    import os
    # USE_CACHE = False

    os.system("mysql -t < ../Northwind.MySQL5.sql")
    database_name = 'northwind'
    db = Database('mysql+mysqldb://kanter@localhost/%s' % (database_name)) 

    table_name = "Customers"
    make_all_features(db, db.tables[table_name])
    find_correlations(db, db.tables[table_name])