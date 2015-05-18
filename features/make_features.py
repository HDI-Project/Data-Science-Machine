import pdb
import profile

import os

from database import Database
import sqlalchemy.dialects.mysql.base as column_datatypes
import numpy as np
import agg_functions
import flat_functions
import row_functions
import datetime
import threading
#############################
# Table feature functions  #
#############################

"""
Make all feature of child tables

Make all agg features. Agg features are aggregate functions applied to child tables, so we must make those feature first, which we have donorschoose

Make row features. 

Make one to one features

Make a all features for parent tables. Parent tables use features of this table, so we must have calcualted agg, row, and one to one features.

Make flat features. Flat features pull from table features, so we must have made all feature for parent tables, which he have

"""


MAX_FUNC_TO_APPLY = 2

def make_all_features(db, table, caller=None, path=[], depth=0):
    caller_name = 'no caller'
    if caller:
        caller_name = caller.name
    print "*"*depth + 'making all features %s, path= %s' % (table.name, str(path))

    #found a cycle
    new_path = list(path) + [table]
    if len(path) != len(set(path)):
        return

    threads = []
    for related,fk in table.get_child_tables():
        #dont make_all on the caller and dont make all on yourself
        if related != caller and related != table:
            t = threading.Thread(target=make_all_features, args=(db, related), kwargs=dict(path=new_path, caller=table, depth=depth+1))
            # make_all_features(db, related, caller=table, depth=depth+1)
            t.start()
            t.join()
            # threads.append(t)
    # [t.join() for t in threads]

    print "*"*depth +  'making agg features %s, caller= %s' % (table.name, caller_name)
    make_agg_features(db, table, caller, depth)

    print "*"*depth +  'making row features %s' % (table.name)
    make_row_features(db, table, caller, depth)
    
    print "*"*depth +  'making one_to_one features %s' % (table.name)
    make_one_to_one_features(db, table, caller, depth)

    make_flat_features(db, table, caller, depth) #Todo pass path so we don't bring in flat feature we do not need

    threads = []
    for related,fk in table.get_parent_tables():
        #dont make_all on the caller and dont make all on yourself
        if related != caller and related != table:
            t = threading.Thread(target=make_all_features, args=(db, related), kwargs=dict(path=new_path, caller=table, depth=depth+1))
            # make_all_features(db, related, caller=table, depth=depth+1)
            t.start()
            t.join()
            # threads.append(t)
    # [t.join() for t in threads]


    print "*"*depth +  'making flat features %s, caller= %s' % (table.name, caller_name)
    make_flat_features(db, table, caller, depth)


#############################
# Agg feature      #
#############################
def make_agg_features(db, table, caller, depth): 
    for fk in db.get_related_fks(table):
        

        child_table = db.tables[fk.parent.table.name]

        if child_table.name in table.config.get("excluded_agg_entities", []):
            print "skip agg", child_table.name, table.name
            continue
        
        if table.is_one_to_one(child_table, fk):
            continue

        # interval = table.config.get("make_intervals", {}).get(child_table.name, None)

        # if interval:
        #     agg_functions.make_intervals(db, fk, n_intervals=interval["n_intervals"], delta_days=interval["delta_days"])
        
        agg_functions.apply_funcs(db,fk)


#############################
# Flat feature    #
#############################
def make_flat_features(db, table, caller, depth):
    """
    add in columns from tables that this table has a foreign key to as well as make sure row features are made
    notes:
    - a table will only be flatten once
    - ignores flattening info from caller
    """
    flat = flat_functions.FlatFeature(db)
    for fk in table.base_table.foreign_keys:
        parent_table = db.tables[fk.column.table.name]
        if parent_table in [table, caller]:
            continue
        
        flat.apply(fk)

#############################
# Flat feature    #
#############################
def make_one_to_one_features(db, table, caller, depth):
    flat = flat_functions.FlatFeature(db)
    for related, fk in table.get_child_tables():
        if table.is_one_to_one(related, fk):
            one_to_one_table = db.tables[fk.parent.table.name]
            if one_to_one_table in [table, caller]:
                continue

            flat.apply(fk, inverse=True)
        

#############################
# Row feature functions     #
#############################
def make_row_features(db ,table, caller, depth):
    # pass
    row_functions.apply_funcs(table)
    # add_ntiles(table)


if __name__ == "__main__":
    import debug
    from sqlalchemy.engine import create_engine

    # os.system("mysql -t < ../Northwind.MySQL5.sql")
    # # # os.system("mysql -t < ../allstate/allstate.sql")

    # database_name = 'ijcai'
    # table_name = "Outcomes"
    # save_name = "models/"+database_name + "__" + table_name
    # url = 'mysql+mysqldb://kanter@localhost/%s' % (database_name)
    # engine = create_engine(url)
    # drop_tables ["Behaviors_1", "Outcomes_1","Items_1", "Categorys_1", "Merchants_1", "Merchants_2","Brands_1", "Actions_1", "Users_1", "Users_2"]:
    # from ijcai_config import config

    # database_name = 'donorschoose'
    # table_name = "Projects"
    # save_name = "models/"+database_name + "__" + table_name
    # url = 'mysql+mysqldb://kanter@localhost/%s' % (database_name)
    # engine = create_engine(url)
    # drop_tables =  ["Schools_1", "Teachers_1", "Vendors_1", "Donors_1", "Projects_1", "Outcomes_1", "Essays_1", "Donations_1"]
    # from donorschoose_config import config

    # database_name = 'grockit'
    # table_name = "Outcomes"
    # save_name = "models/"+database_name + "__" + table_name
    # url = 'mysql+mysqldb://kanter@localhost/%s' % (database_name)
    # engine = create_engine(url)
    # drop_tables = ["Users_1", "QuestionSets_1", "Groups_1", "Tracks_1", "Subtracks_1", "Questions_1", "GameTypes_1", "Outcomes_1",  "Tags_1", "QuestionTags_1"]
    
    # from grockit_config import config

    database_name = 'kdd2015'
    table_name = "Enrollments"
    save_name = "models/"+database_name + "__" + table_name
    url = 'mysql+mysqldb://kanter@localhost/%s' % (database_name)
    engine = create_engine(url)
    drop_tables = ["Courses_1", "Enrollments_1", "Log_1", "EventTypes_1", "ObjectChildren_1", "Objects_1", "Users_1", "Outcomes_1"]
    
    from kdd2015_config import config

    yes = raw_input("Continue %s (y/n): " % database_name)

    if yes == "y":
        for t in drop_tables:
            try:
                qry = "drop table %s" % t
                engine.execute(qry)
            except Exception, e:
                print e
        #reloaded db after dropping tables
        db = Database('mysql+mysqldb://kanter@localhost/%s' % (database_name), config=config) 
            
        table = db.tables[table_name]
        make_all_features(db, table)
        db.save(save_name)

    
        db = Database.load(save_name)
        table = db.tables[table_name]

        debug.export_col_names(table)

    # debug.print_cols_names(db.tables['Orders'])

