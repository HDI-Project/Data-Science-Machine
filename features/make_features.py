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

#["count", "sum", 'avg', 'std', 'max', 'min']


MAX_FUNC_TO_APPLY = 2

def make_all_features(db, table, caller=None, depth=0):
    caller_name = 'no caller'
    if caller:
        caller_name = caller.name
    print "*"*depth + 'making all features %s, caller= %s' % (table.name, caller_name)

    threads = []
    for related,fk in table.get_related_tables():
        # if table.name == "Projects" and related.name != "Resources":
        #     print "skip " + related.name 
        #     continue

        #dont make_all on the caller and dont make all on yourself
        if related != caller and related != table:
            t = threading.Thread(target=make_all_features, args=(db, related), kwargs=dict(caller=table, depth=depth+1))
            # make_all_features(db, related, caller=table, depth=depth+1)
            t.start()
            threads.append(t)

    [t.join() for t in threads]

    print "*"*depth +  'making agg features %s, caller= %s' % (table.name, caller_name)
    make_agg_features(db, table, caller, depth)
    print "*"*depth +  'making row features %s' % (table.name)
    make_row_features(db, table, caller, depth)
    print "*"*depth +  'making flat features %s, caller= %s' % (table.name, caller_name)
    make_flat_features(db, table, caller, depth)


#############################
# Agg feature      #
#############################
def make_agg_features(db, table, caller, depth): 
    for fk in db.get_related_fks(table):
        child_table = db.tables[fk.parent.table.name]
        
        if table.is_one_to_one(child_table, fk):
            continue

        if not caller:
            # agg_functions.make_intervals(db, fk)
            agg_functions.apply_funcs(db, fk)
        else:
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

    for related, fk in table.get_child_tables():
        if table.is_one_to_one(related, fk):
            one_to_one_table = db.tables[fk.parent.table.name]
            if one_to_one_table in [table, caller]:
                continue

            flat.apply(fk, inverse=True)
        

#############################
# Row feature functions     #
#############################
def make_row_features(db, table, caller, depth):
    row_functions.text_length(table)
    # convert_datetime_weekday(table)
    # add_ntiles(table)


if __name__ == "__main__":
    import debug
    from sqlalchemy.engine import create_engine

    # os.system("mysql -t < ../Northwind.MySQL5.sql")
    # # # os.system("mysql -t < ../allstate/allstate.sql")

    # database_name = 'northwind'
    database_name = 'donorschoose'
    table_name = "Projects"
    url = 'mysql+mysqldb://kanter@localhost/%s' % (database_name)
    engine = create_engine(url)
    for t in ["Schools_1", "Teachers_1", "Vendors_1", "Donors_1", "Projects_1", "Outcomes_1", "Essays_1"]:
        try:
            qry = "drop table %s" % t
            engine.execute(qry)
        except Exception, e:
            print e
    #reloaded db after dropping tables
    db = Database('mysql+mysqldb://kanter@localhost/%s' % (database_name) ) 
    table = db.tables[table_name]
    make_all_features(db, table)
    db.save("models/"+table_name)

    
    db = Database.load("models/"+table_name)
    table = db.tables[table_name]

    # profile.run('make_all_features(db, table)')
    debug.export_col_names(table)

    # debug.print_cols_names(db.tables['Orders'])
#beaumont
