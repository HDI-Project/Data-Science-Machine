from database import Database
import sqlalchemy.dialects.mysql.base as column_datatypes

#############################
# Table feature functions  #
#############################

def make_agg_features(db, table):
    print 'making agg features %s' % (table.table.name)
    if table.has_agg_features:
        print 'skipping agg %s' % (table.table.name)
        return

    print db.get_related_fks(table)
    for fk in db.get_related_fks(table):
        related_table = db.tables[fk.parent.table.name]
        
        #make sure this related table has calculatd features
        make_flat_features(db, related_table, caller=table)

        #determine columns to aggregate
        numeric_cols = related_table.get_numeric_columns(ignore_relationships=True)
        if len(numeric_cols) == 0:
            continue

        agg_select = []
        set_values = []
        # pdb.set_trace()
        for col in numeric_cols:
            for func in ["count", "sum", 'avg', 'std', 'max', 'min']:
                new_col = "{func}.{table_name}.{col_name}".format(func=func,col_name=col['name'], table_name=related_table.table.name)
                
                select = "{func}(`rt`.`{col_name}`) AS `{new_col}`".format(func=func.upper(),col_name=col['name'], new_col=new_col)
                agg_select.append(select)

                value = "`a`.`{new_col}` = `b`.`{new_col}`".format(new_col=new_col)
                set_values.append(value)

                new_metadata = {
                    'feature_type' : 'agg_feature',
                    'agg_feature_type' : func,
                    'numeric' : True,
                    'excluded_agg_ops' : table.agg_func_exclude.get(func, None)
                }

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


def make_flat_features(db, table, caller=None):
    """
    add in columns from tables that this table has a foreign key to as well as make sure row features are made
    notes:
    - this methord will make sure  foreign tables have made all before adding their columns
    - a table will only be flatten once
    - ignores flattening info from caller
    """
    make_row_features(db, table)
    print 'making flat features %s' % (table.table.name)
    if table.has_flat_features:
        print 'skipping flat %s' % (table.table.name)
        return

    for fk in table.table.foreign_keys:
        foreign_table = db.tables[fk.column.table.name]
        if foreign_table in [table, caller]:
            continue
        make_flat_features(db, foreign_table)

        #add columns from foreign table
        to_add = foreign_table.get_column_info(prefix=fk.parent.name + ".", ignore_relationships=True)
        set_values = []
        for col in to_add:
            table.create_column(col['fullname'], col['type'].compile())
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
    print 'done making flat features %s' % (table.table.name)


def make_row_features(db, table):
    print 'making row features %s' % (table.table.name)
    if not table.has_row_features:
        convert_datetime_weekday(table)
        add_ntiles(table)
        table.has_row_features = True
        print 'done row features %s' % (table.table.name)
    else:
        print 'skip row features %s' % (table.table.name)


#############################
# Row feature functions     #
#############################
def convert_datetime_weekday(table):
    for col in table.get_columns_of_type([column_datatypes.DATETIME, column_datatypes.DATE], ignore_relationships=True):
        new_col = col['name'] + "_weekday"
        metadata = {
            'feature_type' : 'row_feature',
            'row_feature_type' : 'weekday',
            'agg_operations' : []
        }
        table.create_column(new_col, column_datatypes.INTEGER.__visit_name__, metadata=metadata,flush=True)
        table.engine.execute(
            """
            UPDATE `%s` t
            set `%s` = WEEKDAY(t.`%s`)
            """ % (table.table.name, new_col, col['name'])
        ) #very bad, fix how parameters are substituted in
        

def add_ntiles(table, n=10):
    for col in table.get_numeric_columns(ignore_relationships=True):
        new_col = col['name'] + "_decile"
        metadata = {
            'feature_type' : 'row_feature',
            'row_feature_type' : 'ntile',
            'numeric' : False,
            'excluded_agg_ops' : ['sum']
        }
        table.create_column(new_col, column_datatypes.INTEGER.__visit_name__, metadata=metadata, flush=True)
        select_pk = ", ".join(["`%s`"%pk for pk in table.primary_key_names])

        where_pk = ""
        first = True
        for pk in table.primary_key_names:
            if not first:
                where_pk += " AND "
                # print where_pk
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
    database_name = 'northwind'
    db = Database('mysql+mysqldb://kanter@localhost/%s' % (database_name) ) 

    make_agg_features(db, db.tables['Orders'])