import datetime


DEFAULT_METADATA = {
    'path' : [], 
    'numeric' : False,
    'categorical' : False
}

class DSMColumn():
    def __init__(self, column, dsm_table, metadata=None):
        self.dsm_table = dsm_table
        self.column = column
        self.name = column.name
        self.unique_name =  column.table.name + '.' + self.name
        self.type = column.type
        self.primary_key = self.column.primary_key
        self.has_foreign_key = len(column.foreign_keys)>0
        
        self.metadata = metadata
        if not metadata:
            self.metadata = dict(DEFAULT_METADATA)

    def __repr__(self):
        return "[COLUMN `%s`.`%s`]"%(self.column.table.name,self.metadata['real_name'])

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
        """.format(col_name=self.name, table=self.column.table.name)

        distinct = self.dsm_table.engine.execute(qry).fetchall()

    

        vals = []
        for d in distinct:
            d = d[0]

            if d in ["", None]:
                continue

            if type(d) == datetime.datetime:
                continue

            if type(d) == long:
                d = int(d)

            # print type(d)

            if d == "\x00":
                d = False
            elif d == "\x01":
                d = True

            vals.append(d)

        #save distinct vals to cache
        self.metadata['distinct_vals'] = vals
        
        return vals      

    def get_max_col_val(self):
        qry = "SELECT MAX(`{col_name}`) from `{table}`".format(col_name=self.name, table=self.column.table.name)
        result = self.dsm_table.engine.execute(qry).fetchall()
        return result[0][0]

    def prefix_name(self, prefix):
        return prefix + self.name



def make_set_qry(set_values, join_on_child, join_on_parent):
    target_table = set_values[0][0]
    parent_table = set_values[0][2]
    set_str = []
    for s in set_values:
        if s[0] != target_table:
            raise Exception("more than one target table")
        if s[2] != parent_table:
            raise Exception("more than one parent table")

        set_str.append("`%s`.`%s`=`%s`.`%s`" % s)



    values = {
        'target_table': target_table,
        'parent_table' : parent_table,
        'fk_child' : join_on_child,
        'fk_parent' : join_on_parent,
        'set' : ','.join(set_str),
    }


    qry = """
    UPDATE `{target_table}`, `{parent_table}`
    SET {set}
    WHERE `{target_table}`.`{fk_child}` = `{parent_table}`.`{fk_parent}`
    """.format(**values)

    return qry