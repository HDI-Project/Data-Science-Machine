import sqlalchemy.dialects.mysql.base as column_datatypes
from datetime import timedelta
from filters import FilterObject
from feature import FeatureBase
import threading

import pdb

class AggFuncBase(FeatureBase):
    name = "AggFuncBase"
    disallowed = set([])

    def col_allowed(self, col, target_table=None):
        """
        returns true if this agg function can be applied to this col. otherwise returns false
        todo: don't allow any function on columns that are all the same value
        """
        if len(col.metadata['path']) > 0:
            path_funcs = [(p['feature_type'], p['feature_type_func']) for p in col.metadata['path']]
            
            # print self.disallowed, path_funcs, self.disallowed.intersection(path_funcs)
            if len(self.disallowed.intersection(path_funcs)) > 0:
                return False

            #do not apply more than one filter
            filter_cols = self.get_filter_cols(include_ignored=False)
            num_filters = len(filter_cols)
            max_filters = self.db.config.get("max_categorical_filter", 1)

            path_filters = col.get_applied_filters(include_ignored=False)

            if num_filters > 0:
                print filter_cols, path_filters, set(filter_cols).intersection(path_filters)
                overlap = set(filter_cols).intersection(path_filters) != set([])
            else:
                overlap = False

            if overlap or num_filters >= max_filters:
                return False

            #todo: not sure this matters anymore 
            # if last['base_column']['table'] == target_table:
            #     return False

        if len(col.metadata['path']) >= self.MAX_PATH_LENGTH:
            # print 'max path', len(col.metadata['path'])
            return False

        # #only use columns with more than 1 distinct value
        # if len(col.get_distinct_vals()) < 2:
        #     return False

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

        # for p in col.metadata["path"]:
        #     if p.get("filter", None) != None:
        #         return False

        if col.metadata["path"] != [] and col.metadata["path"][-1]["base_column"].dsm_table == target_table:
            print "agg flat feature from target table"
            print target_table, col.metadata["path"]
            print
            return False
        return True

    def make_real_name(self, col, fk_name):
        if self.filter_obj!= None and self.filter_obj.get_label() != "":
            new_col = "{func}({fk_name}.{col_name}[{filter_label}])".format(func=self.func,col_name=col.metadata['real_name'], fk_name=fk_name, filter_label=self.filter_obj.get_label())
        else:
            new_col = "{func}({fk_name}.{col_name})".format(func=self.func,col_name=col.metadata['real_name'], fk_name=fk_name)
            
        return new_col

    def apply(self, fk):
        def do_qry(new_table_name, related_table, fk, agg_select, set_values, involved_cols):
            #optimization to avoid creating complex query for involved cols if not necessary
            set_trace= False
            if all([c.column.table == related_table.base_table for c in involved_cols]):
                related_table_name = related_table.base_table.name
            else:
                set_trace= True
                related_table_name = "(" + related_table.make_full_table_stmt(involved_cols) + ")"

            params = {
                "fk_select" : "`rt`.`%s`" % fk.parent.name,
                "agg_select" : ", ".join(agg_select),
                "set_values" : ", ".join(set_values),
                "fk_join_on" : "`b`.`{rt_col}` = `a`.`{a_col}`".format(rt_col=fk.parent.name, a_col=fk.column.name),
                "related_table" : related_table_name,
                "target_table" : new_table_name,
                'where_stmt' : self.make_where_stmt(alias="rt")
            }

            
            qry = """
            UPDATE `{target_table}` a
            LEFT JOIN ( SELECT {fk_select}, {agg_select}
                   FROM {related_table} as rt
                   {where_stmt}
                   GROUP BY {fk_select}
                ) b
            ON {fk_join_on}
            SET {set_values}
            WHERE {fk_join_on}
            """.format(**params)
            # if set_trace:
            #     pdb.set_trace()
            # print qry
            table.execute(qry)


        # check this logic
        table = self.db.tables[fk.column.table.name]
        related_table = self.db.tables[fk.parent.table.name]
        #determine columns to aggregate
        agg_select = []
        set_values = []
        parent_fk_col = related_table.columns[(fk.parent.table.name,fk.parent.name)]

        last_target_table = None
        involved_cols = list(self.get_filter_col_set()) + [parent_fk_col]
        for col in related_table.get_column_info():
            if not self.col_allowed(col, target_table=table):
                continue

            new_features = True
            #if the fk is a circular references and has a special column name, keep it. otherwise use the name of the table
            #todo: check this under new column name scheme
            if fk.column.table == fk.parent.table and fk.parent.name != fk.column.name:
                fk_name = fk.parent.name
            else:
                fk_name = fk.parent.table.name

            new_col = self.make_real_name(col, fk_name)

            if table.has_feature(new_col): 
                continue

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
                'categorical' : False,
                'real_name' : new_col
            })
            if self.filter_obj and self.filter_obj.interval_num != None:
                new_metadata['interval_num'] = self.filter_obj.interval_num
            
            new_table_name,new_col_name = table.create_column(column_datatypes.FLOAT.__visit_name__, metadata=new_metadata)
            
            if last_target_table == None:
                last_target_table = new_table_name

            # print col, new_table_name, last_target_table
            if new_table_name!=last_target_table:
                print new_col, table
                do_qry(last_target_table, related_table, fk, agg_select, set_values, involved_cols)
                agg_select = []
                set_values = []
                involved_cols = list(self.get_filter_col_set()) + [parent_fk_col]
                last_target_table = new_table_name
            # print "add col", table.table.name, new_col
            involved_cols.append(col)


            select = "{func}(`{col_name}`) AS `{new_col}`".format(func=self.func.upper(),col_name=col.name, new_col=new_col_name)
            agg_select.append(select)

            value = "`a`.`{new_col}` = `b`.`{new_col}`".format(new_col=new_col_name)
            set_values.append(value)


        if len(set_values) > 0:
            print new_col, table
            do_qry(new_table_name, related_table, fk, agg_select, set_values, involved_cols)

class AggSum(AggFuncMySQL):
    name = "Sum"
    func = "sum"
    disallowed = set([('agg', 'max'),('agg', 'min')])

class AggMax(AggFuncMySQL):
    name = "Max"
    func = "max"
    disallowed = set([('agg', 'min')])

class AggMin(AggFuncMySQL):
    name = "Min"
    func = "min"
    disallowed = set([('agg', 'max')])

class AggStd(AggFuncMySQL):
    name = "Std"
    func = "std"
    disallowed = set([('agg', 'std')])

class AggAvg(AggFuncMySQL):
    name = "Avg"
    func = "avg"
    disallowed = set([])



class AggFK(AggFuncBase):
    name ="AggFuncMySQL"
    func = None
    def col_allowed(self, col, target_table=None):
        """
        returns true if this agg function can be applied to this col. otherwise returns false
        """
        if not super(AggFuncMySQL, self).col_allowed(col, target_table=target_table):
            return False
        return True

    def make_real_name(self, fk_name):
        if self.filter_obj!= None and self.filter_obj.get_label() != "":
            new_col = "{func}({fk_name}[{filter_label}])".format(func=self.func,fk_name=fk_name, filter_label=self.filter_obj.get_label())
        else:
            new_col = "{func}({fk_name})".format(func=self.func,fk_name=fk_name)

        return new_col

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

        real_name = self.make_real_name(fk_name)

        if table.has_feature(real_name): 
            return

        col = related_table.columns[(fk.parent.table.name,fk.parent.name)]
        
        new_metadata = col.copy_metadata()

        path_add = {
            'base_column': col,
            'feature_type' : 'agg',
            'feature_type_func' : self.func,
            'filter' : self.filter_obj,
        }

        new_metadata.update({
            'real_name' : real_name,
            'path' : new_metadata['path'] + [path_add],
            "numeric" : True
        })

        #if we are working with an interval filter, add interval number to new metadata
        if self.filter_obj and self.filter_obj.interval_num != None :
            new_metadata['interval_num'] = self.filter_obj.interval_num

        new_table_name, new_col_name = table.create_column(column_datatypes.FLOAT.__visit_name__, metadata=new_metadata)

        involved_cols = list(self.get_filter_col_set()) + [col]

        #optimization to avoid creating complex query for involved cols if not necessary
        if all([c.column.table == related_table.base_table for c in involved_cols]):
            related_table_name = related_table.base_table.name
        else:
            set_trace= True
            related_table_name = "(" + related_table.make_full_table_stmt(involved_cols) + ")"

        params = {
            "fk_select" : "`%s`" % fk.parent.name,
            "fk_join_on" : "`b`.`{rt_col}` = `target_table`.`{a_col}`".format(rt_col=fk.parent.name, a_col=fk.column.name),
            "related_table" : related_table_name,
            "table" : new_table_name,
            'new_col_name' : new_col_name,
            'where_stmt' : self.make_where_stmt(alias="rt"),
            "func" : self.func
        }
        # pdb.set_trace()
            
        qry = """
        UPDATE `{table}` target_table
        LEFT JOIN ( SELECT {fk_select}, {func}({fk_select}) as val
               FROM {related_table} rt
               {where_stmt}
               GROUP BY {fk_select}
            ) b
        ON {fk_join_on}
        SET `target_table`.`{new_col_name}` = `b`.val
        where {fk_join_on}
        """.format(**params)
        # print qry
        table.execute(qry)

class AggCount(AggFK):
    name ="AggCount"
    func = "COUNT"

# class AggDistinct(AggFuncBase):
#     name ="AggCountDistinct"
#     func = "DISTINCT"


def make_interval_filters(col, n_intervals, delta):
    def date_to_str(d):
        return d.strftime('%Y-%m-%d %H:%M:%S')

    interval_filters = []

    max_val, min_val = col.get_max_min_col_val()

    for n in xrange(n_intervals-1,-1, -1):
        new_max = max_val - delta

        interval_range = [(col, "<=", date_to_str(max_val)), (col, ">", date_to_str(new_max))]
        f_obj = FilterObject(interval_range, label="int=%d"%n, interval_num=n)
        interval_filters.append(f_obj)
        max_val = new_max
        
        if max_val < min_val:
            break

    return interval_filters

def make_categorical_filters(target_table, related_table):
    # TODO handle when there are two tables with date columns
            # now apply with filtering if there's no caller
    filters = []
    for col in related_table.get_categorical_filters():
        path = [c["base_column"].dsm_table for c in  col.metadata["path"]]
        print target_table, path
        if path != [] and target_table in path:
            continue
        for val in col.get_distinct_vals():
            f_obj = FilterObject([(col, "=", val)])
            filters.append(f_obj)
    return filters
            

def make_intervals(db, fk, n_intervals=10, delta_days=7):
    related_table = db.tables[fk.parent.table.name]
    #todo: more intelligently choose the date col to make intervals from
    col = related_table.get_column_info(first=True, match_func= lambda x: type(x.type) in [column_datatypes.DATETIME, column_datatypes.DATE])
    if col:
        delta = timedelta(days=delta_days)
        interval_filters = make_interval_filters(col, n_intervals, delta)
        for f_obj in interval_filters:
            apply_funcs(db, fk, f_obj)


def apply_funcs(db, fk, filter_obj=None):
    funcs = get_functions()
    related_table = db.tables[fk.parent.table.name]

    #make sure we don't aggregate rows in test data
    train_filter = related_table.make_training_filter()
    if train_filter != None and filter_obj != None:
        filter_obj = filter_obj.AND(train_filter)
    elif train_filter:
        filter_obj = train_filter


    table = db.tables[fk.column.table.name]
    filters = make_categorical_filters(table, related_table)

    threads = []
    for func in funcs:
        #apply
        func(db,filter_obj).apply(fk)
        # func(db).apply(fk)
       
        for f_obj in filters:
            if func != AggCount:
                continue
                  
            if filter_obj:  
                f_obj = f_obj.AND(filter_obj)
                    
            
            func(db,f_obj).apply(fk)
            # t= threading.Thread(target=worker).start()
            # threads.append(t)
        

#SELECT Users.user_id, TIMESTAMPDIFF(DAY, MIN(time_stamp), MAX(time_stamp)) / (COUNT(*)-1) FROM Users join Behaviors on Behaviors.user_id = Users.user_id  group by Users.user_id limit 10;


#agg oldest
def get_functions():
    return [AggCount, AggSum,AggAvg, AggMax, AggStd, AggMin]
