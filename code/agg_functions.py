import sqlalchemy.dialects.mysql.base as column_datatypes
from datetime import timedelta
from filters import FilterObject

import pdb

class AggFuncBase(object):
    name = "AggFuncBase"
    disallowed = set([])
    MAX_PATH_LENGTH = 1

    def __init__(self, db, filter_obj=None):
        self.db = db
        self.filter_obj = filter_obj

    def make_where_stmt(self):
        where_stmt = ""
        if self.filter_obj:
            where_stmt = self.filter_obj.to_where_statement()
        return where_stmt

    def col_allowed(self, col, target_table=None):
        """
        returns true if this agg function can be applied to this col. otherwise returns false
        todo: don't allow any function on columns that are all the same value
        """
        if len(col.metadata['path']) > 0:
            last = col.metadata['path'][-1]
            if (last['feature_type'], last['feature_type_func']) in self.disallowed:
                return False

            #not sure this matters anymore 
            # if last['base_column']['table'] == target_table:
            #     return False

        if len(col.metadata['path']) >= self.MAX_PATH_LENGTH:
            return False

        #only use columns with more than 1 distinct value
        if len(col.get_distinct_vals()) < 2:
            return False


        #make sure the filter parameters allow aggregating this column
        if self.filter_obj and not self.filter_obj.can_agg(col):
            return False

        return True



    def apply(self):
        pass

class AggFuncMySQL(AggFuncBase):
    name ="AggFuncMySQL"
    func = None

    def col_allowed(self, col, target_table=None):
        """
        returns true if this agg function can be applied to this col. otherwise returns false
        """
        if not super(AggFuncMySQL, self).col_allowed(col, target_table=target_table):
            return False

        if not col.metadata['numeric']:
            return False

        return True

    def apply(self, fk):
        
        # check this logic
        table = self.db.tables[fk.column.table.name]
        related_table = self.db.tables[fk.parent.table.name]
        #determine columns to aggregate
        agg_select = []
        set_values = []
        new_features = False
        for col in related_table.columns.values():
            if not self.col_allowed(col, target_table=table):
                continue

            new_features = True

            #if the fk is a circular references and has a special column name, keep it. otherwise use the name of the table
            if fk.column.table == fk.parent.table and fk.parent.name != fk.column.name:
                fk_name = fk.parent.name
            else:
                fk_name = fk.parent.table.name

            if self.filter_obj :
                new_col = "[{func}.{fk_name}.{col_name}_{filter_label}]".format(func=self.func,col_name=col.name, fk_name=fk_name, filter_label=self.filter_obj.get_label())
            else:
                new_col = "[{func}.{fk_name}.{col_name}]".format(func=self.func,col_name=col.name, fk_name=fk_name)
            
            new_metadata = col.copy_metadata()

            path_add = {
                'base_column': col,
                'feature_type' : 'agg',
                'feature_type_func' : self.func,
                'filter' : self.filter_obj,
            }

            new_metadata.update({
                'path' : new_metadata['path'] + [path_add],
                'numeric' : True,
                'categorical' : False
            })

            select = "{func}(`rt`.`{col_name}`) AS `{new_col}`".format(func=self.func.upper(),col_name=col.name, new_col=new_col)
            agg_select.append(select)

            value = "`a`.`{new_col}` = `b`.`{new_col}`".format(new_col=new_col)
            set_values.append(value)

            # print "add col", table.table.name, new_col
            table.create_column(new_col, column_datatypes.FLOAT.__visit_name__, metadata=new_metadata)

        if new_features:
            table.flush_columns()
            params = {
                "fk_select" : "`rt`.`%s`" % fk.parent.name,
                "agg_select" : ", ".join(agg_select),
                "set_values" : ", ".join(set_values),
                "fk_join_on" : "`b`.`{rt_col}` = `a`.`{a_col}`".format(rt_col=fk.parent.name, a_col=fk.column.name),
                "related_table" : related_table.table.name,
                "table" : table.table.name,
                'where_stmt' : self.make_where_stmt()
            }

            
            qry = """
            UPDATE `{table}` a
            LEFT JOIN ( SELECT {fk_select}, {agg_select}
                   FROM `{related_table}` rt
                   {where_stmt}
                   GROUP BY {fk_select}
                ) b
            ON {fk_join_on}
            SET {set_values}
            """.format(**params)

            table.engine.execute(qry)
            # pdb.set_trace()

class AggSum(AggFuncMySQL):
    name = "Sum"
    func = "sum"
    disallowed = set([('agg', 'max'),('agg', 'min')])

class AggMax(AggFuncMySQL):
    name = "Max"
    func = "max"
    disallowed = set([])

class AggMin(AggFuncMySQL):
    name = "Min"
    func = "min"
    disallowed = set([])

class AggStd(AggFuncMySQL):
    name = "Std"
    func = "std"
    disallowed = set([('agg', 'std')])


class AggCount(AggFuncBase):
    name ="AggCount"

    def col_allowed(self, col, target_table=None):
        """
        returns true if this agg function can be applied to this col. otherwise returns false
        """
        if not super(AggFuncMySQL, self).col_allowed(col, target_table=target_table):
            return False
        return True

    def apply(self, fk):
        
        # check this logic
        table = self.db.tables[fk.column.table.name]
        related_table = self.db.tables[fk.parent.table.name]
        #determine columns to aggregate
        
        #if the fk is a circular references and has a special column name, keep it. otherwise use the name of the table
        if fk.column.table == fk.parent.table and fk.parent.name != fk.column.name:
            fk_name = fk.parent.name
        else:
            fk_name = fk.parent.table.name

        if self.filter_obj:
            new_col = "[count.{fk_name}_{filter_label}]".format(fk_name=fk_name, filter_label=self.filter_obj.get_label())
        else:
            new_col = "[count.{fk_name}]".format(fk_name=fk_name)

        col = related_table.columns[fk.parent.name]
        
        new_metadata = col.copy_metadata()

        path_add = {
            'base_column': col,
            'feature_type' : 'agg',
            'feature_type_func' : 'count'
        }

        new_metadata.update({
            'path' : new_metadata['path'] + [path_add],
            'numeric' : True,
            'categorical' : False
        })

        table.create_column(new_col, column_datatypes.FLOAT.__visit_name__, metadata=new_metadata, flush=True)

        params = {
            "fk_select" : "`rt`.`%s`" % fk.parent.name,
            "fk_join_on" : "`b`.`{rt_col}` = `a`.`{a_col}`".format(rt_col=fk.parent.name, a_col=fk.column.name),
            "related_table" : related_table.table.name,
            "table" : table.table.name,
            'new_col' : new_col,
            'where_stmt' : self.make_where_stmt()
        }

            
        qry = """
        UPDATE `{table}` a
        LEFT JOIN ( SELECT {fk_select}, COUNT({fk_select}) as count
               FROM `{related_table}` rt
               {where_stmt}
               GROUP BY {fk_select}
            ) b
        ON {fk_join_on}
        SET `a`.`{new_col}` = `b`.count
        """.format(**params)

        table.engine.execute(qry)


def make_interval_filters(col, n_intervals, delta):
    def date_to_str(d):
        return d.strftime('%Y-%m-%d %H:%M:%S')
    #placeholder constants
    

    # related_table = coldb.tables[fk.parent.table.name]
    interval_filters = []


    max_val = col.get_max_col_val()

    for n in xrange(n_intervals):
        new_max = max_val - delta
        interval_range = [(col, "<=", date_to_str(max_val)), (col, ">", date_to_str(new_max))]
        f_obj = FilterObject(interval_range, label="int=%d"%n)
        interval_filters.append(f_obj)
        max_val = new_max


    return interval_filters

def make_categorical_filters(table):
    # TODO handle when there are two tables with date columns
            # now apply with filtering if there's no caller
    filters = []
    for col in table.get_categorical():
        for val in col.get_distinct_vals():
            f_obj = FilterObject([(col, "=", val)])
            filters.append(f_obj)
    return filters
            

def make_intervals(db, fk):
    related_table = db.tables[fk.parent.table.name]
    #todo: more intelligently choose the date col to make intervals from
    col = related_table.get_column_info(first=True, match_func= lambda x: type(x.type)==column_datatypes.DATETIME)
    n_intervals = 10
    delta = timedelta(days=7)
    interval_filters = make_interval_filters(col, n_intervals, delta)

    for f_obj in interval_filters:
        apply_funcs(db, fk, f_obj)

def apply_funcs(db, fk, filter_obj=None):
    funcs = get_functions()
    related_table = db.tables[fk.parent.table.name]
    filters = make_categorical_filters(related_table)

    #TODO change these loops so the func is reinited so often
    for func in funcs:
        if filter_obj:
            func(db, filter_obj).apply(fk)
        else:
            func(db).apply(fk)

        for f_obj in filters:  
            if filter_obj:  
                f_obj = f_obj.AND(filter_obj)

            func(db, f_obj).apply(fk) #apply without any filtering
        


def get_functions():
    return [AggCount, AggMax, AggSum, AggStd, AggMin]