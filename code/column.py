import datetime


DEFAULT_METADATA = {
    'path' : [], 
    'numeric' : False,
    'categorical' : False
}

class DSMColumn():
    def __init__(self, column, table, metadata=None):
        self.table = table
        self.column = column
        self.name = column.name
        self.unique_name =  self.table.table.name + '.' + self.name
        self.type = column.type
        self.primary_key = self.column.primary_key
        self.has_foreign_key = len(column.foreign_keys)>0
        
        self.metadata = metadata
        if not metadata:
            self.metadata = dict(DEFAULT_METADATA)

    def update_metadata(self, update):
        self.metadata.update(update)

    def copy_metadata(self):
        """
        returns copy of the metadata object for this column_metadata
        """
        return dict(self.metadata)


    def get_distinct_vals(self):
        #try to get cached distinct_vals
        if 'distinct_vals' in self.metadata:
            return self.metadata['distinct_vals']

        qry = """
        SELECT distinct(`{col_name}`) from `{table}`
        """.format(col_name=self.name, table=self.table.table.name)

        distinct = self.table.engine.execute(qry).fetchall()

        vals = []
        for d in distinct:
            d = d[0]

            if d in ["", None]:
                continue

            if type(d) == datetime.datetime:
                continue

            if type(d) == long:
                d = int(d)

            if d == "\x00":
                d = False
            elif d == "\x01":
                d = True

            vals.append(d)

        #save distinct vals to cache
        self.metadata['distinct_vals'] = vals
        
        return vals      

    def get_max_col_val(self):
        qry = "SELECT MAX(`{col_name}`) from `{table}`".format(col_name=self.name, table=self.table.table.name)
        result = self.table.engine.execute(qry).fetchall()
        return result[0][0]

    def prefix_name(self, prefix):
        return prefix + self.name