import predict
import pdb
from database import Database
from filters import FilterObject
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from learning import DSMClassifier
from sklearn.cross_validation import train_test_split


def add_feature_quality(entity, target, filter_obj=None):
    all_features = predict.get_predictable_features(entity)

    for f in all_features:
        try:
            score = predict.do_trial(target, [f], cv=2)
        except Exception, e:
            print e
            score = 0


        if "quality" not in f.metadata:
            f.metadata["quality"] = {}

        f.metadata["quality"][target.name] = score

        print f.metadata["real_name"], score

    # print all_features

    # y = []
    # rows = []
    # row_data = table.get_rows([target_col]+all_features, filter_obj=filter_obj)
    # for r in row_data:
    #     y_add = r[0]

    #     if y_add == None:
    #         continue

    #     y.append(float(y_add)) 
    #     rows.append(r[1:])
    # rows = np.array(rows)

    # cat_mask = [col.metadata["categorical"] for col in all_features]
    # le = LabelEncoder()
    # for i,is_cat in enumerate(cat_mask):
    #     if is_cat:
    #         rows[:,i] = le.fit_transform(rows[:,i])
    # rows = np.array(rows, dtype=np.float)
    # cat_mask = np.array(cat_mask, dtype=bool)
    # y = np.array(y, dtype=int)

    # enc = OneHotEncoder()
    # # pdb.set_trace()
    # for i,f in enumerate(all_features):
    #     clf = DSMClassifier(cat_mask=[f.metadata["categorical"]])
    #     X = np.array([rows[:,i]]).T
    #     try:
    #         X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=.5)
    #         clf.fit(X_train, y_train)
    #         score = clf.score(X_test, y_test)
    #     except Exception, e:
    #         print e
    #         score = 0

    #     if "quality" not in f.metadata:
    #         f.metadata["quality"] = {}

    #     f.metadata["quality"][target.name] = score

    #     print f.metadata["real_name"], score


    # print f.metadata["quality"][target.name]







        



if __name__ == "__main__":
    

    # table_name = "Outcomes"
    # model_name = "grockit__" + table_name
    # filename = "models/%s"%model_name
    # db = Database.load(filename)
    # table = db.tables[table_name]
    # target_col = table.get_col_by_name("correct")
    # add_feature_quality(table, target_col)
    # db.save(filename)



    # table_name = "Outcomes"
    # model_name = "ijcai__" + table_name
    # filename = "models/%s"%model_name
    # db = Database.load(filename)
    # table = db.tables[table_name]
    # target_col = table.get_col_by_name("label")
    # add_feature_quality(table, target_col)
    # db.save(filename)



    table_name = "Projects"
    model_name = "donorschoose__" + table_name
    filename = "models/%s"%model_name
    
    db = Database.load(filename)
    table = db.tables[table_name]
    target_col = table.get_col_by_name("Outcome.is_exciting")
    
    filter_col = table.get_col_by_name("date_posted")
    filter_obj = FilterObject([(filter_col, ">", "2013-1-1")])

    print "add feature quality"
    add_feature_quality(table, target_col, filter_obj)

