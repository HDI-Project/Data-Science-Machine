import pdb
from sqlalchemy.orm import sessionmaker
from compile_query import compile_query
from sqlalchemy import text

import sqlalchemy.dialects.mysql.base as column_datatypes
from sqlalchemy.schema import Table

from sqlalchemy.schema import MetaData

DEFAULT_METADATA = {
    'feature_type' : 'original', #original, row, agg, flat

    'funcs_applied' : 0,

    'agg_feature_type' : None,
    'row_feature_type' : None,

    'numeric' : False,
    
    'allowed_agg_funcs' : None,
    'excluded_agg_funcs' : set([]),
    'allowed_row_funcs' : None,
    'excluded_row_funcs' : set([]),
}


class DSMTable:
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

        self.column_metadata = dict([(c.name, DEFAULT_METADATA) for c in table.c])

    #############################
    # Database write operations #
    #############################
    def create_column(self, column_name, column_type, metadata={},flush=False, drop_if_exists=True):
        """
        add column with name column_name of type column_type to this table. if column exists, drop first

        todo: suport where to add it
        """
        self.column_metadata[column_name] = metadata
        self.cols_to_add.append((column_name, column_type))
        if flush:
            self.flush_columns(drop_if_exists=drop_if_exists)
    
    def drop_column(self, column_name, flush=False):
        """
        drop column with name column_name from this table
        """
        self.cols_to_drop.append(column_name)
        if flush:
            self.flush_columns(drop_if_exists=drop_if_exists)

    def flush_columns(self, drop_if_exists=True):
        # print self.cols_to_drop, self.cols_to_add
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

        #reflect table again to have update columns
        # print self.table.name, 'reflect'
        new_metadata = MetaData(bind=self.engine)
        self.table = Table(self.table.name, new_metadata, autoload=True, autoload_with=self.engine)
        # print [c.name for c in self.table.c]
        # print [c['name'] for c in self.get_column_info()]






    ###############################
    # Table info helper functions #
    ###############################
    def get_column_info(self, prefix='', ignore_relationships=False):
        """
        return info about columns in this table

        todo: primary_keys, foreign_keys only
        """
        cols = []
        for c in self.table.columns:
            if ignore_relationships and c.primary_key:
                continue

            if ignore_relationships and len(c.foreign_keys)>0:
                continue

            col = {
                'column' : c,
                'name' : c.name,
                'fullname' : prefix + c.name,
                'type' : c.type,
                'metadata' : self.column_metadata.get(c.name, {})
            }

            cols.append(col)
        
        return cols

    def get_columns_of_type(self, datatypes=[], **kwargs):
        """
        returns a list of columns that are type data_type
        """
        if type(datatypes) != list:
            datatypes = [datatypes]
        return [c for c in self.get_column_info(**kwargs) if type(c['type']) in datatypes]

    def get_numeric_columns(self, **kwargs):
        datatypes = [column_datatypes.INTEGER, column_datatypes.FLOAT, column_datatypes.DECIMAL, column_datatypes.DOUBLE, column_datatypes.SMALLINT, column_datatypes.MEDIUMINT]
        return self.get_columns_of_type(datatypes, **kwargs)

    def has_column(self, name):
        return name in [c.name for c in self.table.c]


