class FilterObject(object):
    """
    This is an class the represents filters to be applied to a specific table.

    to_where_statement() is the key function to use to generate the code for a query

    """
    def __init__(self, filters, label=None):
        self.filters = filters
        self.filtered_cols = [c[0] for c in filters]
        self.label = label

    def to_where_statement(self):
        stmt = "WHERE "
        #TODO currently wont work with multiple filters because i forgot the AND
        and_cond = []
        for f in self.filters:
            cond = "`{col}` {op} {value}".format(col=f[0].name, op=f[1], value=repr(f[2]))
            and_cond.append(cond) 

        stmt += " AND ".join(and_cond)
        return stmt

    def get_label(self):
        if self.label:
            return self.label

        label = ""
        for f in self.filters:
            label += "{col}{op}{value}".format(col=f[0].name, op=f[1], value=f[2])

        return label


    def can_agg(self, col):
        #can't agg on columns that are filtered
        if col.name in self.filtered_cols:
            return False

        #only allow certain number of fitlers to be applied to a column
        count_filters = 0
        for p in col.metadata['path']:
            if p.get('filter', None):
                count_filters += 1

            if count_filters >= 1:
                return False

        return True

    def AND(self, f_obj):
        filters = self.filters + f_obj.filters
        label = self.get_label() + "_AND_" + f_obj.get_label()
        return FilterObject(filters, label)