from features import make_all_features
from database import Database
import numpy as np
from sklearn import tree
from sklearn import cross_validation
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn import linear_model

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



def find_correlations(db, table):
    all_cols = table.get_column_info(match_func=lambda x: x['metadata']['numeric'] or x['metadata']['categorical'])
    print all_cols
    # pdb.set_trace()
    for target in all_cols:
        cols = [target]
        cat_indicies = []
        for i, predictor in enumerate(all_cols):
            if predictor == target:
                continue

            if not predictor['metadata']['numeric']:
                cat_indicies.append(i)

            cols.append(predictor)

        data = np.array(table.get_rows(cols).fetchall())

        clf = Pipeline([
            ('int_encoder', IntegerEncoder(categorical_features=cat_indicies)),
            ('cat_encoder', OneHotEncoder(categorical_features=cat_indicies)),
            ('scaler', preprocessing.StandardScaler()),
            # ('clf', tree.DecisionTreeRegressor()),
            ('clf', linear_model.LogisticRegression()),
        ])

        X = data[:,1:]
        # X = np.nan_to_num(data[:,1:].astype(dtype=np.float))
        # Y = data[:,0]
        # Y = np.where(Y == np.array(None),'', Y)
        Y = np.nan_to_num(data[:,0].astype(dtype=np.float))

        
        scores = cross_validation.cross_val_score(clf, X, Y, scoring='r2', cv=5)
        print "predict %s" % target['name']
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        # print ("importance", clf.named_steps['clf.scores_'])
        # pdb.set_trace()
    pdb.set_trace()



if __name__ == "__main__":
    import os
    # USE_CACHE = False

    os.system("mysql -t < ../Northwind.MySQL5.sql")
    database_name = 'northwind'
    db = Database('mysql+mysqldb://kanter@localhost/%s' % (database_name)) 

    make_all_features(db, db.tables['Customers'])
    find_correlations(db, db.tables['Customers'])