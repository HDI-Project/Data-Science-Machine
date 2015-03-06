import pdb
import os

from database import Database
import sqlalchemy.dialects.mysql.base as column_datatypes
import numpy as np
import inflect
import agg_functions
import datetime
#############################
# Table feature functions  #
#############################

#["count", "sum", 'avg', 'std', 'max', 'min']

p = inflect.engine()

MAX_FUNC_TO_APPLY = 2

def make_all_features(db, table, caller=None, depth=0):
    caller_name = 'no caller'
    if caller:
        caller_name = caller.table.name
    print "*"*depth + 'making all features %s, caller= %s' % (table.table.name, caller_name)

    for related in db.get_related_tables(table):
        #dont make_all on the caller and dont make all on yourself
        if related != caller and related != table:
            make_all_features(db, related, caller=table, depth=depth+1)

    make_agg_features(db, table, caller, depth)
    make_row_features(db, table, caller, depth)
    make_flat_features(db, table, caller, depth)
    # prune_features(db, table, caller, depth)

#delete once funcionality is added to allow_agg_funcs
# def prune_features(db, table, caller, depth):
#     caller_name = 'no caller'
#     if caller:
#         caller_name = caller.table.name

#     if caller == table:
#         return

#     print "*"*depth +  'removing features %s, caller= %s' % (table.table.name, caller_name)
#     cols = table.get_column_info()
#     if len(cols) == 0:
#         return
#     counts = np.array(table.get_num_distinct(cols), dtype=np.float)
#     for i in np.where(counts==1)[0]:
#         print "drop", table.table.name, cols[i]['name']
#         table.drop_column(cols[i]['name'])
#     table.flush_columns()


#############################
# Agg feature functions     #
#############################


def make_agg_features(db, table, caller, depth):
    caller_name = 'no caller'
    if caller:
        caller_name = caller.table.name
    print "*"*depth +  'making agg features %s, caller= %s' % (table.table.name, caller_name)

    if table.has_agg_features:
        print "*"*depth +  'skipping agg %s' % (table.table.name)
        return
    
    for fk in db.get_related_fks(table):
        if not caller:
            agg_functions.make_intervals(db, fk)
        else:
            agg_functions.apply_funcs(db,fk)


    table.has_agg_features = True




#############################
# Flat feature functions    #
#############################
def make_flat_features(db, table, caller, depth):
    """
    add in columns from tables that this table has a foreign key to as well as make sure row features are made
    notes:
    - a table will only be flatten once
    - ignores flattening info from caller
    """
    caller_name = 'no caller'
    if caller:
        caller_name = caller.table.name
    print "*"*depth +  'making flat features %s, caller= %s' % (table.table.name, caller_name)
    

    if table.has_flat_features:
        print "*"*depth +  'skipping flat %s' % (table.table.name)
        return

    for fk in table.table.foreign_keys:
        foreign_table = db.tables[fk.column.table.name]
        if foreign_table in [table, caller]:
            continue

        #add columns from foreign table
        #if the fk has a special column name in parent table, keep it. otherwise use the name of the foreign table
        if fk.parent.name != fk.column.name:
            prefix = fk.parent.name
        else:
            prefix = fk.column.table.name
            singular = p.singular_noun(fk.column.table.name)
            if singular:
                prefix = singular

        prefix += "."

        to_add = foreign_table.get_column_info(ignore_relationships=True)
        # pdb.set_trace()
        set_values = []
        for col in to_add:
            new_metadata = col.copy_metadata()

            path_add = {
                'base_column': col,
                'feature_type' : 'flat',
                'feature_type_func' : None
            }

            new_metadata.update({ 
                'path' : new_metadata['path'] + [path_add],
                'numeric' : False,
                'categorical' : True
            })

            table.create_column(col.prefix_name(prefix), col.type.compile(), metadata=new_metadata)
            set_values.append(
                "a.`%s`=b.`%s`" %
                (col.prefix_name(prefix), col.name)
            )

        table.flush_columns()

        #add column values
        set_value = ','.join(set_values)
        where = "a.`%s`=b.`%s`" % (fk.parent.name, fk.column.name)
        qry = """
        UPDATE `{table}` a, `{foreign_table}` b
        SET {set}
        WHERE {where}
        """.format(table=table.table.name, foreign_table=foreign_table.table.name, set=set_value, where=where)
        table.engine.execute(qry)

    table.has_flat_features = True
    #print "*"*depth +  'done making flat features %s' % (table.table.name)


#############################
# Row feature functions     #
#############################
def row_funcs_is_allowed(col, func):
    if len(col.metadata['path']) > MAX_FUNC_TO_APPLY:
        return False
        
    #todo 
    return True

def make_row_features(db, table, caller, depth):
    print "*"*depth +  'making row features %s' % (table.table.name)
    if not table.has_row_features:
        convert_datetime_weekday(table)
        # add_ntiles(table)
        table.has_row_features = True
        #print "*"*depth +  'done row features %s' % (table.table.name)
    else:
        print "*"*depth +  'skip row features %s' % (table.table.name)

def convert_datetime_weekday(table):
    for col in table.get_columns_of_type([column_datatypes.DATETIME, column_datatypes.DATE], ignore_relationships=True):
        if not row_funcs_is_allowed(col, 'convert_datetime_weekday'):
            continue

        new_col = "[{col_name}]_weekday".format(col_name=col.name)
        new_metadata = col.copy_metadata()
        
        path_add = {
                    'base_column': col,
                    'feature_type' : 'row',
                    'feature_type_func' : "weekday"
                }

        new_metadata.update({ 
            'path' : new_metadata['path'] + [path_add],
            'numeric' : False,
            'categorical' : True
        })

        table.create_column(new_col, column_datatypes.INTEGER.__visit_name__, metadata=new_metadata,flush=True)
        table.engine.execute(
            """
            UPDATE `%s` t
            set `%s` = WEEKDAY(t.`%s`)
            """ % (table.table.name, new_col, col.name)
        ) #very bad, fix how parameters are substituted in
        
def add_ntiles(table, n=10):
    for col in table.get_numeric_columns(ignore_relationships=True):
        new_col = "[{col_name}]_decile".format(col_name=col.name)
        new_metadata = col.copy_metadata()
        path_add = {
                    'base_column': col,
                    'feature_type' : 'row',
                    'feature_type_func' : "ntile"
                }

        new_metadata.update({ 
            'path' : new_metadata['path'] + [path_add],
            'numeric' : False,
            # 'excluded_agg_funcs' : set(['sum']),
            # 'excluded_row_funcs' : set(['add_ntiles']),
        })

        if len(new_metadata['path']) > MAX_FUNC_TO_APPLY:
            continue

        table.create_column(new_col, column_datatypes.INTEGER.__visit_name__, metadata=new_metadata, flush=True)
        select_pk = ", ".join(["`%s`"%pk for pk in table.primary_key_names])

        where_pk = ""
        first = True
        for pk in table.primary_key_names:
            if not first:
                where_pk += " AND "
            where_pk += "`%s` = `%s`.`%s`" % (pk, table.table.name, pk)
            first = False

        qry = """
        UPDATE `{table}`
        SET `{table}`.`{new_col}` = 
        (
            select round({n}*(cnt-rank+1)/cnt,0) as decile from
            (
                SELECT  {select_pk}, @curRank := @curRank + 1 AS rank
                FROM   `{table}` p,
                (
                    SELECT @curRank := 0) r
                    ORDER BY `{col_name}` desc
                ) as dt,
                (
                    select count(*) as cnt
                    from `{table}`
                ) as ct
            WHERE {where_pk}
        );
        """.format(table=table.table.name, new_col=new_col, n=n, col_name=col.name, select_pk=select_pk, where_pk=where_pk)
        table.engine.execute(qry) #very bad, fix how parameters are substituted in


if __name__ == "__main__":
    os.system("mysql -t < ../Northwind.MySQL5.sql")

    database_name = 'northwind'
    db = Database('mysql+mysqldb://kanter@localhost/%s' % (database_name) ) 

    # db.tables['Orders'].to_csv('/tmp/orders.csv')
    make_all_features(db, db.tables['Orders'])
#beaumont