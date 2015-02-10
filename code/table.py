import pdb
from sqlalchemy.orm import sessionmaker
from compile_query import compile_query
from sqlalchemy import text

import sqlalchemy.dialects.mysql.base as column_datatypes


class Table:
    def __init__(self, table, db):
        self.db = db
        self.table = table
        self.engine = self.table.bind
        self.session = sessionmaker(bind=self.engine)()

        self.primary_keys = [key.name for key in table.primary_key]

        self.cols_to_add = []
        self.cols_to_drop = []

        self.added_columns = []
        # self.convert_datetime_weekday()
        # self.add_ntiles()

        self.is_flat = False

        # [self.drop_column(x) for x in self.added_columns]


    def query(self, *args, **kwargs):
        return self.session.query(*args, **kwargs)

    def get_column_info(self, prefix=''):
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
            }

            cols.append(col)
        
        return cols

    def flatten(self):
        """
        add in columns from tables that this table has a foreign key to.
        notes:
        - this methord will make sure  foreign tables are flattened before adding their columns
        - a table will only be flatten once
        """
        if self.is_flat:
            return

        for fk in self.table.foreign_keys:
            foreign_table = self.db.tables[fk.column.table.name]
            if foreign_table == self:
                continue
            foreign_table.flatten()

            #add columns from foreign table
            to_add = foreign_table.get_column_info(prefix=fk.parent.name + ".")
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

        self.is_flat = True




    def get_columns_of_type(self, datatypes):
        """
        returns a list of columns that are type data_type
        """
        if type(datatypes) != list:
            datatypes = [datatypes]

        return [c for c in self.table.c if type(c.type) in datatypes]

    def has_column(self, name):
        return name in [c.name for c in self.table.c]
    
    def create_column(self, column_name, column_type):
        """
        add column with name column_name of type column_type to this table. if column exists, drop first

        todo: suport where to add it
        """
        self.cols_to_add.append((column_name, column_type))
        

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
            self.added_columns.append(name)
            values.append("ADD COLUMN `%s` %s" % (name, col_type))
        if len(values) > 0:
            values = ", ".join(values)
            qry = """
                ALTER TABLE `{table}`
                {cols_to_add}
                """.format(table=self.table.name, cols_to_add=values)
            self.engine.execute(qry) #very bad, fix how parameters are substituted in
            self.cols_to_add = []



    def drop_column(self, column_name):
        """
        drop column with name column_name from this table
        """
        self.cols_to_drop.append(column_name)
        

    def convert_datetime_weekday(self):
        for col in self.get_columns_of_type([column_datatypes.DATETIME, column_datatypes.DATE]):
            new_col = col.name + "_weekday"
            self.create_column(new_col, column_datatypes.INTEGER.__visit_name__)
            self.engine.execute(
                """
                UPDATE `%s` t
                set `%s` = WEEKDAY(t.%s)
                """ % (self.table.name, new_col, col.name)
            ) #very bad, fix how parameters are substituted in
            

    def add_ntiles(self, n=10):
        datatypes = [column_datatypes.DECIMAL, column_datatypes.FLOAT, column_datatypes.DOUBLE]
        for col in self.get_columns_of_type(datatypes):
            new_col = col.name + "_decile"
            self.create_column(new_col, column_datatypes.INTEGER.__visit_name__)

            select_pk = ", ".join(self.primary_keys)

            where_pk = ""
            first = True
            for pk in self.primary_keys:
                if not first:
                    where_pk += " AND "
                    # print where_pk
                where_pk += "%s = `%s`.%s" % (pk, self.table.name, pk)
                first = False

            qry = """
            UPDATE `{table}`
            SET `{table}`.{new_col} = 
            (
                select round({n}*(cnt-rank+1)/cnt,0) as decile from
                (
                    SELECT  {select_pk}, @curRank := @curRank + 1 AS rank
                    FROM   `{table}` p,
                    (
                        SELECT @curRank := 0) r
                        ORDER BY {col_name} desc
                    ) as dt,
                    (
                        select count(*) as cnt
                        from `{table}`
                    ) as ct
                WHERE {where_pk}
            );
            """.format(table=self.table.name, new_col=new_col, n=n, col_name=col.name, select_pk=select_pk, where_pk=where_pk)

            self.engine.execute(qry) #very bad, fix how parameters are substituted in

    
    




