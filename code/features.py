import pdb
import os

from database import Database
import sqlalchemy.dialects.mysql.base as column_datatypes
import numpy as np
#############################
# Table feature functions  #
#############################

#["count", "sum", 'avg', 'std', 'max', 'min']
agg_func_exclude = {
    'avg' : set(['sum']),
    'max' : set(['sum']),
    'min' : set(['sum']),
    'std' : set(['std'])
}

MAX_FUNC_TO_APPLY = 2

def make_all_features(db, table, caller=None, depth=0):
    caller_name = 'no caller'
    if caller:
        caller_name = caller.table.name
    print "*"*depth + 'making all features %s, caller= %s' % (table.table.name, caller_name)
    # pdb.set_trace()
    make_agg_features(db, table, caller, depth)
    make_flat_features(db, table, caller, depth)
    make_row_features(db, table, caller, depth)
    remove_low_variance_features(db, table, caller, depth)

def remove_low_variance_features(db, table, caller, depth):
    cols = table.get_column_info()
    if len(cols) == 0:
        return
    data = np.array(table.get_num_distinct(cols).fetchall(), dtype=np.float)
    pdb.set_trace()


def make_agg_features(db, table, caller, depth):
    caller_name = 'no caller'
    if caller:
        caller_name = caller.table.name
    print "*"*depth +  'making agg features %s, caller= %s' % (table.table.name, caller_name)

    if table.has_agg_features:
        print "*"*depth +  'skipping agg %s' % (table.table.name)
        return

    for fk in db.get_related_fks(table):
        related_table = db.tables[fk.parent.table.name]
        
        #make sure this related table has calculatd features
        if related_table != caller:
            make_all_features(db, related_table, caller=table, depth=depth+1)

        #determine columns to aggregate
        numeric_cols = related_table.get_numeric_columns(ignore_relationships=True)
        if len(numeric_cols) == 0:
            continue

        agg_select = []
        set_values = []
        # pdb.set_trace()
        for col in numeric_cols:
            funcs = set(["sum", 'avg', 'std', 'max', 'min'])

            #order is important here, otherwise exclude can overwrite the white list of allowed
            if  col['metadata']['excluded_agg_funcs'] != None:
                funcs = funcs - col['metadata']['excluded_agg_funcs']
            if col['metadata']['allowed_agg_funcs'] != None:
                funcs = funcs.intersection(col['metadata']['allowed_agg_funcs'])


            if len(funcs) == 0:
                continue
            
            if col['metadata']['row_feature_type'] == "weekday":
                pdb.set_trace()

            for func in funcs:
                new_col = "{func}.[{fk_name}.{col_name}]".format(func=func,col_name=col['name'], fk_name=fk.parent.table.name)
                
                new_metadata = dict(col['metadata'])
    
                new_metadata.update({
                    'feature_type' : 'agg',
                    'agg_feature_type' : func,
                    'numeric' : True,
                    'excluded_agg_funcs' : agg_func_exclude.get(func, None),
                    'funcs_applied' : new_metadata['funcs_applied'] + [func]
                })

                if len(new_metadata['funcs_applied']) > MAX_FUNC_TO_APPLY:
                    continue
                
                select = "{func}(`rt`.`{col_name}`) AS `{new_col}`".format(func=func.upper(),col_name=col['name'], new_col=new_col)
                agg_select.append(select)

                value = "`a`.`{new_col}` = `b`.`{new_col}`".format(new_col=new_col)
                set_values.append(value)


                table.create_column(new_col, column_datatypes.FLOAT.__visit_name__, metadata=new_metadata)
            # pdb.set_trace()

        table.flush_columns()

        params = {
            "fk_select" : "`rt`.`%s`" % fk.parent.name,
            "agg_select" : ", ".join(agg_select),
            "set_values" : ", ".join(set_values),
            "fk_join_on" : "`b`.`{rt_col}` = `a`.`{a_col}`".format(rt_col=fk.parent.name, a_col=fk.column.name),
            "related_table" : related_table.table.name,
            "table" : table.table.name,
        }

        
        qry = """
        UPDATE `{table}` a
        LEFT JOIN ( SELECT {fk_select}, {agg_select}
               FROM `{related_table}` rt
               GROUP BY {fk_select}
            ) b
        ON {fk_join_on}
        SET {set_values}
        """.format(**params)

        table.engine.execute(qry)

    table.has_agg_features = True


def make_flat_features(db, table, caller, depth):
    """
    add in columns from tables that this table has a foreign key to as well as make sure row features are made
    notes:
    - this methord will make sure  foreign tables have made all before adding their columns
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

        make_all_features(db, foreign_table, caller=table, depth=depth+1)

        #add columns from foreign table
        to_add = foreign_table.get_column_info(prefix=fk.parent.name + ".", ignore_relationships=True)
        set_values = []
        for col in to_add:
            new_metadata = dict(col['metadata'])
            new_metadata.update({
                'feature_type' : 'flat',
            })

            table.create_column(col['fullname'], col['type'].compile(), metadata=new_metadata)
            set_values.append(
                "a.`%s`=b.`%s`" %
                (col['fullname'], col['name'])
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


def make_row_features(db, table, caller, depth):
    print "*"*depth +  'making row features %s' % (table.table.name)
    if not table.has_row_features:
        convert_datetime_weekday(table)
        add_ntiles(table)
        table.has_row_features = True
        #print "*"*depth +  'done row features %s' % (table.table.name)
    else:
        print "*"*depth +  'skip row features %s' % (table.table.name)


#############################
# Row feature functions     #
#############################
def check_flat_allowed(col, func):
    if col['metadata']['allowed_row_funcs']:
        return func in col['metadata']['allowed_row_funcs']

    if col['metadata']['excluded_row_funcs']:
        return func not in col['metadata']['excluded_row_funcs']

    return True


def convert_datetime_weekday(table):
    for col in table.get_columns_of_type([column_datatypes.DATETIME, column_datatypes.DATE], ignore_relationships=True):
        if not check_flat_allowed(col, 'convert_datetime_weekday'):
            continue
        new_col = "[{col_name}]_weekday".format(col_name=col['name'])
        new_metadata = dict(col['metadata'])
        
        new_metadata.update({
            'feature_type' : 'row',
            'row_feature_type' : 'weekday',
            'numeric' : False,
            'allowed_agg_funcs' : set([]),
            'excluded_row_funcs' : col['metadata']['excluded_row_funcs'].union(['add_ntiles']),
            'funcs_applied' : new_metadata['funcs_applied'] + ["weekday"]
        })

        table.create_column(new_col, column_datatypes.INTEGER.__visit_name__, metadata=new_metadata,flush=True)
        table.engine.execute(
            """
            UPDATE `%s` t
            set `%s` = WEEKDAY(t.`%s`)
            """ % (table.table.name, new_col, col['name'])
        ) #very bad, fix how parameters are substituted in
        

def add_ntiles(table, n=10):
    for col in table.get_numeric_columns(ignore_relationships=True):
        if not check_flat_allowed(col, 'add_ntiles'):
            continue

        new_col = "[{col_name}]_decile".format(col_name=col['name'])
        new_metadata = dict(col['metadata'])
        new_metadata.update({
            'feature_type' : 'row',
            'row_feature_type' : 'ntile',
            'numeric' : False,
            'excluded_agg_funcs' : set(['sum']),
            'excluded_row_funcs' : set(['add_ntiles']),
            'funcs_applied' : new_metadata['funcs_applied'] + ['ntiles']
        })

        if len(new_metadata.get('funcs_applied')) > MAX_FUNC_TO_APPLY:
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
        """.format(table=table.table.name, new_col=new_col, n=n, col_name=col['name'], select_pk=select_pk, where_pk=where_pk)
        table.engine.execute(qry) #very bad, fix how parameters are substituted in



    


if __name__ == "__main__":
    os.system("mysql -t < ../Northwind.MySQL5.sql")

    database_name = 'northwind'
    db = Database('mysql+mysqldb://kanter@localhost/%s' % (database_name) ) 

    # db.tables['Orders'].to_csv('/tmp/orders.csv')
    make_all_features(db, db.tables['Orders'])