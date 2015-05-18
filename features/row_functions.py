import sqlalchemy.dialects.mysql.base as column_datatypes
from feature import FeatureBase

class MysqlRowFunc(FeatureBase):
    name = "MysqlRowFunc"
    func = None
    numeric = False
    categorical = False
    col_type = column_datatypes.INTEGER.__visit_name__

    def get_allowed_cols(self):
        pass

    def do_qry(self, target_table, set_vals):
        SET = []
        for s in set_vals:
            SET.append("`t`.%s = %s(`t`.%s)" % (s[0], self.func, s[1]))
        SET = ",".join(SET)

        qry = """
            UPDATE `{target_table}` t
            SET {SET}
            """.format(target_table=target_table, SET=SET)

        self.db.execute(qry)

    def apply(self, table):
        to_add = []

        #create columns
        for col in self.get_allowed_cols(table):
            real_name = "%s(%s)" % (self.func,col.metadata["real_name"])
            
            new_metadata = col.copy_metadata()

            path_add = {
                'base_column': col,
                'feature_type' : 'row',
                'feature_type_func' : self.name
            }

            new_metadata.update({
                'real_name' : real_name,
                'numeric' : self.numeric,
                'categorical': self.categorical,
                'path' : new_metadata['path'] + [path_add],
            })

            #don't make feature if table has it
            if table.has_feature(real_name):
                continue

            new_table_name, new_col_name = table.create_column(self.col_type, metadata=new_metadata)

            to_add.append((col, (new_table_name, new_col_name)))

        last_table_name = None
        set_vals = [] 
        for (col, (new_table_name, new_col_name)) in to_add:
            if last_table_name == None:
                last_table_name = new_table_name

            if last_table_name!=new_table_name:
                self.do_qry(last_table_name, set_vals)
                last_table_name = new_table_name
                set_vals = []

            set_vals.append([new_col_name, col.column.name])

        if set_vals != []:
            self.do_qry(last_table_name, set_vals)
    

class TextLength(MysqlRowFunc):
    name = "text_length"
    func = "length"
    numeric = True
    categorical = False
    col_type = column_datatypes.INTEGER.__visit_name__

    def get_allowed_cols(self, table):
        return table.get_columns_of_type([column_datatypes.TEXT], ignore_relationships=True)

class Weekday(MysqlRowFunc):
    name = "weekday"
    func = "weekday"
    numeric = False
    categorical = True
    col_type = column_datatypes.INTEGER.__visit_name__

    def get_allowed_cols(self, table):
        return table.get_columns_of_type([column_datatypes.DATETIME, column_datatypes.DATE], ignore_relationships=True)

class Month(MysqlRowFunc):
    name = "month"
    func = "month"
    numeric = False
    categorical = True
    col_type = column_datatypes.INTEGER.__visit_name__

    def get_allowed_cols(self, table):
        return table.get_columns_of_type([column_datatypes.DATETIME, column_datatypes.DATE], ignore_relationships=True)
    


def apply_funcs(table):
    funcs = [TextLength, Weekday, Month]
    excluded = table.config.get("excluded_row_functions", [])
    included = table.config.get("included_row_functions", funcs) #if none included, include all
    included = set(included).difference(excluded)
    for func in funcs:
        if func.name in included:
            func(table.db).apply(table)


# def convert_datetime_weekday(table):
#     for col in table.get_columns_of_type([column_datatypes.DATETIME, column_datatypes.DATE], ignore_relationships=True):
#         real_name = "DAY({col_name})".format(col_name=col.metadata['real_name'])
#         new_metadata = col.copy_metadata()
        
#         path_add = {
#                     'base_column': col,
#                     'feature_type' : 'row',
#                     'feature_type_func' : "weekday"
#                 }

#         new_metadata.update({ 
#             'path' : new_metadata['path'] + [path_add],
#             'numeric' : False,
#             'categorical' : True,
#             "real_name" : real_name
#         })


#         new_table_name, new_col_name = table.create_column(column_datatypes.INTEGER.__visit_name__, metadata=new_metadata)

#         params = {
#             target_table: new_table_name,
#             new_col_name : new_col_name,
#             src_table: col.column.table.name,
#             col_name : col.name
#         }

#         qry = """
#             UPDATE `{target_table}` t
#             set `{new_col_name}` = WEEKDAY({src_table}.`{col_name}`)
#             """ % (**params)
#         print qry
#         table.engine.execute(qry)

#TODO UPDATE EVERYTHING BELOW HERE



        
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