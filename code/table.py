import pdb
from sqlalchemy.orm import sessionmaker
from compile_query import compile_query
from sqlalchemy import text

import sqlalchemy.dialects.mysql.base as column_datatypes




class Table:
    #["count", "sum", 'avg', 'std', 'max', 'min']
    agg_func_exclude .get( )
        'avg' : ['sum'],
        'max' : ['sum']
        'min' : ['sum']
    }

    def __init__(self, table, db):
        self.db = db
        self.table = table
        self.engine = self.table.bind
        self.session = sessionmaker(bind=self.engine)()

        self.primary_key_names = [key.name for key in table.primary_key]

        self.cols_to_add = []
        self.cols_to_drop = []
        
        self.has_row_features = False
        self.has_agg_features = False
        self.has_flat_features = False

        self.column_metadata = {}

        # [self.drop_column(x) for x in self.added_columns]


    def query(self, *args, **kwargs):
        return self.session.query(*args, **kwargs)

    def get_column_info(self, prefix='', ignore_relationships=False):
        """
        return info about columns in this table

        todo: primary_keys, foreign_keys only
        """
        cols = []
        for c in self.table.columns:
            col = {
                'column' : c,
                'name' : c.name,
                'fullname' : prefix + c.name,
                'type' : c.type,
                'metadata' : self.column_metadata.get(c.name, {})
            }

            cols.append(col)
        
        return cols

    def get_related_fks(self):
        """
        returns a list of the foreign_keys in the database that reference this table
        """
        related_columns = []
        for related in self.db.tables.values():
            for fk in related.table.foreign_keys:
                if fk.column.table == self.table:
                    related_columns.append(fk)
        return related_columns

    def make_agg_features(self):
        print 'making agg features %s' % (self.table.name)
        if self.has_agg_features:
            print 'skipping agg %s' % (self.table.name)
            return

        for fk in self.get_related_fks():
            print fk
            related_table = self.db.tables[fk.parent.table.name]
            
            #make sure this related table has calculatd features
            related_table.flatten(caller=self)

            #determine columns to aggregate
            numeric_cols = related_table.get_numeric_columns(ignore_relationships=True)
            if len(numeric_cols) == 0:
                continue

            agg_select = []
            set_values = []

            for col in numeric_cols:
                for func in ["count", "sum", 'avg', 'std', 'max', 'min']:
                    new_col = "{func}.{table_name}.{col_name}".format(func=func,col_name=col.name, table_name=related_table.table.name)
                    
                    select = "{func}(`rt`.`{col_name}`) AS `{new_col}`".format(func=func.upper(),col_name=col.name, new_col=new_col)
                    agg_select.append(select)

                    value = "`a`.`{new_col}` = `b`.`{new_col}`".format(new_col=new_col)
                    set_values.append(value)

                    new_metadata = {
                        'feature_type' : 'agg_feature',
                        'agg_feature_type' : func,
                        'numeric' : True,
                        'excluded_agg_ops' : self.agg_func_exclude.get(func, None)
                    }

                    self.create_column(new_col, column_datatypes.FLOAT.__visit_name__, metadata=new_metadata)
                # pdb.set_trace()

            self.flush_columns()

            params = {
                "fk_select" : "`rt`.`%s`" % fk.parent.name,
                "agg_select" : ", ".join(agg_select),
                "set_values" : ", ".join(set_values),
                "fk_join_on" : "`b`.`{rt_col}` = `a`.`{a_col}`".format(rt_col=fk.parent.name, a_col=fk.column.name),
                "related_table" : related_table.table.name,
                "table" : self.table.name,
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

            self.engine.execute(qry)

        self.has_agg_features = True



    def flatten(self, caller=None):
        """
        add in columns from tables that this table has a foreign key to as well as make sure row features are made
        notes:
        - this methord will make sure  foreign tables have made all before adding their columns
        - a table will only be flatten once
        - ignores flattening info from caller
        """
        self.make_row_features()
        print 'making flat features %s' % (self.table.name)
        if self.has_flat_features:
            print 'skipping flat %s' % (self.table.name)
            return

        for fk in self.table.foreign_keys:
            foreign_table = self.db.tables[fk.column.table.name]
            if foreign_table in [self, caller]:
                continue
            foreign_table.flatten()

            #add columns from foreign table
            to_add = foreign_table.get_column_info(prefix=fk.parent.name + ".", ignore_relationships=True)
            set_values = []
            for col in to_add:
                self.create_column(col['fullname'], col['type'].compile())
                set_values.append(
                    "a.`%s`=b.`%s`" %
                    (col['fullname'], col['name'])
                )
            self.flush_columns()

            #add column values
            set_value = ','.join(set_values)
            where = "a.`%s`=b.`%s`" % (fk.parent.name, fk.column.name)
            qry = """
            UPDATE `{table}` a, `{foreign_table}` b
            SET {set}
            WHERE {where}
            """.format(table=self.table.name, foreign_table=foreign_table.table.name, set=set_value, where=where)
            self.engine.execute(qry)

        self.has_flat_features = True
        print 'done making flat features %s' % (self.table.name)



    def make_row_features(self):
        print 'making row features %s' % (self.table.name)
        if not self.has_row_features:
            self.convert_datetime_weekday()
            self.add_ntiles()
            self.has_row_features = True
            print 'done row features %s' % (self.table.name)
        else:
            print 'skip row features %s' % (self.table.name)

    def get_numeric_columns(self, **kwargs):
        return self.get_columns_of_type([column_datatypes.INTEGER, column_datatypes.FLOAT, column_datatypes.DECIMAL, column_datatypes.DOUBLE], **kwargs)

    def get_columns_of_type(self, datatypes=[], **kwargs):
        """
        returns a list of columns that are type data_type
        """
        if type(datatypes) != list:
            datatypes = [datatypes]
            
        return [c for c in self.get_column_info(**kwargs) if c['type'] in datatypes]

    def has_column(self, name):
        return name in [c.name for c in self.table.c]
    
    def create_column(self, column_name, metadata={},column_type,flush=False, drop_if_exists=True):
        """
        add column with name column_name of type column_type to this table. if column exists, drop first

        todo: suport where to add it
        """
        self.column_metadata[column_name] = metadata
        self.cols_to_add.append((column_name, column_type))
        if flush:
            self.flush_columns(drop_if_exists=drop_if_exists)

    def flush_columns(self, drop_if_exists=True):
        #first, check which of cols_to_add need to be dropped first
        for (name, col_type) in self.cols_to_add:
            if drop_if_exists and self.has_column(name):
                self.drop_column(name)

        #second, flush columns that need to be dropped
        values = []
        for name in self.cols_to_drop:
            values.append("DROP `%s`" % (name))
        if len(values) > 0:
            values = ", ".join(values)

            self.engine.execute(
                """
                ALTER TABLE `{table}`
                {cols_to_drop}
                """.format(table=self.table.name, cols_to_drop=values)
            ) #very bad, fix how parameters are substituted in

            self.cols_to_drop = []
        
        #third, flush columns that need to be added
        values = []
        for (name, col_type) in self.cols_to_add:
            self.column_metadata[name]['flushed'] = True
            values.append("ADD COLUMN `%s` %s" % (name, col_type))
        if len(values) > 0:
            values = ", ".join(values)
            qry = """
                ALTER TABLE `{table}`
                {cols_to_add}
                """.format(table=self.table.name, cols_to_add=values)
            self.engine.execute(qry) #very bad, fix how parameters are substituted in
            self.cols_to_add = []



    def drop_column(self, column_name, flush=False):
        """
        drop column with name column_name from this table
        """
        self.cols_to_drop.append(column_name)
        if flush:
            self.flush_columns(drop_if_exists=drop_if_exists)
        

    def convert_datetime_weekday(self):
        for col in self.get_columns_of_type([column_datatypes.DATETIME, column_datatypes.DATE], ignore_relationships=True):
            new_col = col.name + "_weekday"
            metadata = {
                'feature_type' : 'row_feature',
                'row_feature_type' : 'weekday',
                'agg_operations' : []
            }
            self.create_column(new_col, column_datatypes.INTEGER.__visit_name__, metadata=metadata,flush=True)
            self.engine.execute(
                """
                UPDATE `%s` t
                set `%s` = WEEKDAY(t.`%s`)
                """ % (self.table.name, new_col, col.name)
            ) #very bad, fix how parameters are substituted in
            

    def add_ntiles(self, n=10):
        for col in self.get_numeric_columns(ignore_relationships=True):
            new_col = col.name + "_decile"
            metadata = {
                'feature_type' : 'row_feature',
                'row_feature_type' : 'ntile',
                'numeric' : False
                'excluded_agg_ops' : ['sum']
            }
            self.create_column(new_col, column_datatypes.INTEGER.__visit_name__, metadata=metadata flush=True)
            select_pk = ", ".join(["`%s`"%pk for pk in self.primary_key_names])

            where_pk = ""
            first = True
            for pk in self.primary_key_names:
                if not first:
                    where_pk += " AND "
                    # print where_pk
                where_pk += "`%s` = `%s`.`%s`" % (pk, self.table.name, pk)
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
            """.format(table=self.table.name, new_col=new_col, n=n, col_name=col.name, select_pk=select_pk, where_pk=where_pk)
            self.engine.execute(qry) #very bad, fix how parameters are substituted in

    
    

#mysql -t < Northwind.MySQL5.sql 
