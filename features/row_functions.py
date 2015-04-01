def row_funcs_is_allowed(col, func):
    if len(col.metadata['path']) > MAX_FUNC_TO_APPLY:
        return False
        
    #todo 
    return True
def convert_datetime_weekday(table):
    for col in table.get_columns_of_type([column_datatypes.DATETIME, column_datatypes.DATE], ignore_relationships=True):
        if not row_funcs_is_allowed(col, 'convert_datetime_weekday'):
            continue

        new_col = "[{col_name}]_weekday".format(col_name=col.metadata['real_name'])
        new_metadata = col.copy_metadata()
        
        path_add = {
                    'base_column': col,
                    'feature_type' : 'row',
                    'feature_type_func' : "weekday"
                }

        new_metadata.update({ 
            'path' : new_metadata['path'] + [path_add],
            'numeric' : False,
            'categorical' : True,
            "real_name" : new_col
        })

        new_col_name = table.create_column(column_datatypes.INTEGER.__visit_name__, metadata=new_metadata,flush=True)
        table.engine.execute(
            """
            UPDATE `%s` t
            set `%s` = WEEKDAY(t.`%s`)
            """ % (table.name, new_col_name, col.name)
        ) #very bad, fix how parameters are substituted in
        
def add_ntiles(table, n=10):
    for col in table.get_numeric_columns(ignore_relationships=True):
        new_col = "[{col_name}]_decile".format(col_name=col.metadata['real_name'])
        new_metadata = col.copy_metadata()
        path_add = {
                    'base_column': col,
                    'feature_type' : 'row',
                    'feature_type_func' : "ntile"
                }

        new_metadata.update({ 
            'path' : new_metadata['path'] + [path_add],
            'numeric' : False,
            "real_name" : new_col
            # 'excluded_agg_funcs' : set(['sum']),
            # 'excluded_row_funcs' : set(['add_ntiles']),
        })

        if len(new_metadata['path']) > MAX_FUNC_TO_APPLY:
            continue

        new_col_name = table.create_column(column_datatypes.INTEGER.__visit_name__, metadata=new_metadata, flush=True)
        select_pk = ", ".join(["`%s`"%pk for pk in table.primary_key_names])

        where_pk = ""
        first = True
        for pk in table.primary_key_names:
            if not first:
                where_pk += " AND "
            where_pk += "`%s` = `%s`.`%s`" % (pk, table.name, pk)
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
        """.format(table=table.name, new_col=new_col_name, n=n, col_name=col.name, select_pk=select_pk, where_pk=where_pk)
        table.engine.execute(qry) #very bad, fix how parameters are substituted in