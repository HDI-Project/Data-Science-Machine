from features import make_all_features
from database import Database
import numpy as np
from sklearn import tree
from sklearn import cross_validation
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer

import pdb

def find_correlations(db, table):
    def match_target(col):
        return col['metadata']['numeric']
    
    def match_predictor(col):
        return col['metadata']['feature_type'] !="original" and col['metadata']['numeric']

    for target in table.get_column_info(match_func=match_target):
        
        cols = [target]
        print "predict %s" % target['name']
        for predictor in table.get_column_info(match_func=match_predictor):
            if predictor != target:
                cols.append(predictor)

        data = np.array(table.get_rows(cols).fetchall())
        clf = Pipeline([
            # ('vect', CountVectorizer()),
            # ('tfidf', TfidfTransformer()),
            ('clf', tree.DecisionTreeRegressor()),
        ])

        X = np.nan_to_num(data[:,1:].astype(dtype=np.float))
        # Y = data[:,0]
        # Y = np.where(Y == np.array(None),'', Y)
        Y = np.nan_to_num(data[:,0].astype(dtype=np.float))

        
        scores = cross_validation.cross_val_score(clf, X, Y, scoring='r2', cv=5)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        # pdb.set_trace()



if __name__ == "__main__":
    import os
    # USE_CACHE = False

    os.system("mysql -t < ../Northwind.MySQL5.sql")
    database_name = 'northwind'
    db = Database('mysql+mysqldb://kanter@localhost/%s' % (database_name)) 

    make_all_features(db, db.tables['Customers'])
    find_correlations(db, db.tables['Customers'])