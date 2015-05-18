import pdb
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

import sqlalchemy.dialects.mysql.base as column_datatypes
from sqlalchemy.schema import Table

from sqlalchemy.schema import MetaData
from column import DSMColumn

from collections import defaultdict

import threading
from filters import FilterObject

class DSMTable:
    MAX_COLS_TABLE = 500

    config = {
        "column_types" : {"FLOAT": .8, "INTEGER" :.2}
    }

    def __init__(self, table, db):
        self.lock = threading.Lock()

        self.db = db
        self.name = table.name
        self.base_table = table
        self.tables = {table.name: table}
        self.engine = db.engine
        
        self.config.update(self.db.config.get("entities", {}).get(self.name, {}))

        self.num_rows = self.engine.execute("SELECT count(*) from `%s`" % (self.name)).fetchall()[0][0]

        self.primary_key_names = [key.name for key in table.primary_key]

        self.columns = {}
        self.cols_to_add = defaultdict(list)
        self.cols_to_drop = defaultdict(list)

        self.num_added_tables = 0
        self.curr_table = None

        self.free_cols = {}

        self.feature_list = set([])

        self.init_columns()

        self.one_to_one = self.config.get("one_to_one", {})


    def __getstate__(self):
        """
        prepare class for pickling
        """
        state = self.__dict__.copy()
        del state['db']        
        del state['engine']
        return state

    def __repr__(self):
        return "DSMTable %s"%(self.name)

    def set_db(self, db):
        self.db = db
        self.engine = db.engine

    def init_columns(self):
        """
        make metadata for columns already in database and return the metadata dictionary
        """
        datatypes = [column_datatypes.INTEGER, column_datatypes.FLOAT, column_datatypes.DECIMAL, column_datatypes.DOUBLE, column_datatypes.SMALLINT, column_datatypes.MEDIUMINT]
        # categorical = self.get_categorical()
        # if len(categorical) > 0:
        #     pdb.set_trace()

        for col in self.base_table.c:
            col = DSMColumn(col, dsm_table=self)

            is_numeric = type(col.type) in datatypes and not (col.primary_key or col.has_foreign_key)
            is_categorical = False

                

            col.update_metadata({
                'numeric' : is_numeric,
                'real_name' : col.name,
                'categorical' : is_categorical,
                'categorical_filter' : False
            })

            if col.name in self.config.get("feature_metadata", {}):
                col.update_metadata(self.config["feature_metadata"][col.name])
                print col.metadata
            self.columns[(col.column.table.name,col.name)] = col

        #set categorical columns
        # todo figure out how to do it with large tables. perhaps do some sort of sampling
        print self.num_rows, self.name
        if self.num_rows >= 10000000: #ten million
            return

        for col, count in self.get_num_distinct(self.get_column_info()):
            if count <= 2:
                col.metadata["categorical"] = True
                col.metadata["binary"] = True

    def execute(self, qry):
        return self.db.execute(qry)

    #############################
    # Database operations       #
    #############################
    def make_new_table(self):
        self.num_added_tables += 1
        new_table_name = self.name + "_" + str(self.num_added_tables)

        #todo t: check if temp table good
        # qry = """
        # CREATE table {new_table_name} as (select {select_pk} from {old_table})
        # """.format(new_table_name=new_table_name, select_pk=",".join(self.primary_key_names), old_table=self.name)
        try:
            qry = """
            CREATE TABLE `{new_table_name}` LIKE `{old_table}`; 
            """.format(new_table_name=new_table_name, old_table=self.name)
            self.engine.execute(qry)

            qry = """
            INSERT `{new_table_name}` SELECT * FROM `{old_table}`;
            """.format(new_table_name=new_table_name, old_table=self.name)
            self.engine.execute(qry)
        except Exception,e:
            print e

        self.tables[new_table_name] = Table(new_table_name, MetaData(bind=self.engine), autoload=True, autoload_with=self.engine)
        return self.tables[new_table_name]


    def create_column(self, column_type, metadata={}):
        """
        get a column to insert data into
        """
        with self.lock:
            if self.curr_table == None:
                self.make_cols()

            #get free column of type
            if len(self.free_cols[self.curr_table.name][column_type]) <= 0:
                self.make_cols()

            col = self.free_cols[self.curr_table.name][column_type].pop()

            #update metadeta
            col.update_metadata(metadata)

            #move to columns array
            self.add_column(col)
            
        #todo update return type
        return col.column.table.name,col.name

    def add_column(self, col):
        print "added:", self.name, col.metadata["real_name"]
        self.feature_list.add(col.metadata["real_name"])
        self.columns[(col.column.table.name,col.name)] = col

    def has_feature(self, name):
        return name in self.feature_list

    def make_cols(self):
        """
        make columns in a new table for use by create_column
        make them in the proportions supplied by column_types
        """
        column_types = self.config.column_types
        table = self.make_new_table()
        #update current table
        self.curr_table = table

        cols_to_add = []
        count = 0
        for column_type in column_types:
            num = int(self.MAX_COLS_TABLE * .8)
            cols_to_add += [(table.name+"__"+str(c), column_type) for c in range(count, count+num)]
            count += num

        values=[]
        for (name, col_type) in cols_to_add:
            values.append("ADD COLUMN `%s` %s" % (name, col_type))

        values = ", ".join(values)
        qry = """
            ALTER TABLE `{table}`
            {cols_to_add}
            """.format(table=table.name, cols_to_add=values)
        self.engine.execute(qry)

        
        #reflect table again to have update columns
        table = Table(table.name, MetaData(bind=self.engine), autoload=True, autoload_with=self.engine)
        self.tables[table.name] = table
        self.free_cols[table.name] = {}
        #for new column in the database, add it to free columns
        for (name, col_type) in cols_to_add:
            if col_type not in self.free_cols[table.name]:
                self.free_cols[table.name][col_type] = set([])

            col = DSMColumn(getattr(table.c, name), dsm_table=self)
            self.free_cols[table.name][col_type].add(col)
        

    ###############################
    # Table info helper functions #
    ###############################
    def get_column_info(self, prefix='', ignore_relationships=False, match_func=None, first=False, set_trace=False, ignore=True):
        """
        return info about columns in this table. 
        info should be things that are read directly from database or something that is dynamic at query time. everything else should be part of metadata
        """
        with self.lock:
            cols = []
            for col in self.columns.values():
                if ignore_relationships and col.primary_key:
                    continue

                if ignore_relationships and col.has_foreign_key:
                    continue

                if ignore and col.metadata.get("ignore", False):
                    continue

                if set_trace:    
                    pdb.set_trace()

                if match_func != None and not match_func(col):
                    continue

                if first:
                    return col

                cols.append(col)
            
            if first:
                return None

            return sorted(cols, key=lambda c: c.column.table.name)

    def get_primary_key(self):
        return self.get_column_info(match_func= lambda x: x.primary_key, first=True)

    def get_parent_tables(self):
        """
        return set of tables that this table has foreign_key to
        """
        parent_tables = set([])

        for fk in self.base_table.foreign_keys:
            add = (self.db.tables[fk.column.table.name], fk)
            parent_tables.add(add)                

        return parent_tables

    def get_child_tables(self):
        """
        return set of tables that have foreign_key to this table
        """
        child_tables = set([])
        for related in self.db.tables.values():
            for fk in related.base_table.foreign_keys:
                if fk.column.table == self.base_table:
                    add = (self.db.tables[fk.parent.table.name], fk)
                    child_tables.add(add)

        return child_tables

    def get_related_tables(self):
        """
        return a set of tables that reference table or are referenced by table
        """
        children = self.get_child_tables()
        parents = self.get_parent_tables()
        return parents.union(children)

    def is_one_to_one(self, related, fk):
        if self.num_rows >= 10000000 or related.num_rows >= 10000000: #ten million
            return False


        #check one to one cache for either table to avoid requerying
        if related in self.one_to_one:
            return self.one_to_one[related] 
        if self in related.one_to_one:
            return related.one_to_one[self] 



        related_pk =related.get_primary_key()
        pk = self.get_primary_key()

        qry = """
         SELECT distinct(count(`{related_table_name}`.`{related_pk_name}`))
         FROM `{table_name}`
         JOIN `{related_table_name}` ON `{table_name}`.`{fk_parent}` = `{related_table_name}`.`{fk_child}`
         GROUP BY `{table_name}`.`{table_pk_name}`;
        """.format(related_table_name=related.name, related_pk_name=related_pk.column.name, table_name=self.name, table_pk_name=pk.column.name, fk_parent=fk.column.name, fk_child=fk.parent.name)

        distinct = list(self.execute(qry))

        self.one_to_one[related] = len(distinct) == 1 and distinct[0][0] == 1
        return self.one_to_one[related]

    def get_col_by_name(self, col_name):
        """
        get first column that matches either real name or database name of col_name
        """
        return self.get_column_info(match_func=lambda c, col_name=col_name: c.name == col_name or c.metadata["real_name"]==col_name, first=True, ignore=False)

    def names_to_cols(self, names):
        return [self.get_col_by_name(n) for n in names]

    def get_columns_of_type(self, datatypes=[], **kwargs):
        """
        returns a list of columns that are type data_type
        """
        if type(datatypes) != list:
            datatypes = [datatypes]
        return [c for c in self.get_column_info(**kwargs) if type(c.type) in datatypes]

    def get_numeric_columns(self, **kwargs):
        """
        gets columns that are numeric as specified by metada
        """
        return [c for c in self.get_column_info(**kwargs) if c.metadata['numeric']]
    
    def has_column(self, table_name, name):
        return (table_name,name) in self.columns

    def has_table(self, table_name):
        return table_name in self.tables

    def get_categorical(self, max_proportion_unique=.3, min_proportion_unique=0, max_num_unique=10):
        cat_cols = self.get_column_info(match_func=lambda x: x.metadata["categorical"] == True)
        # if len(cat_cols) >0:
        #     pdb.set_trace()
        # counts = self.get_num_distinct(cols)
        
        # qry = """
        # SELECT COUNT(*) from `{table}`
        # """.format(table=self.base_table.name) #we can get totoal just by going to base since all tables are the same
        # total = float(self.engine.execute(qry).fetchall()[0][0])

        # if total == 0:
        #     return set([])

        # cat_cols = []
        # for col, count in counts:
        #     if ( max_num_unique > count > 1 and
        #          max_proportion_unique <= count/total < min_proportion_unique and
        #          len(col.metadata['path']) <= 1 ):

        #         cat_cols.append(col)

        return cat_cols

    def get_categorical_filters(self):
        return self.get_column_info(match_func=lambda x: x.metadata["categorical_filter"] == True)

    def get_num_distinct(self, cols):
        """
        returns number of distinct values for each column in cols. returns in same order as cols
        """
        SELECT = ','.join(["count(distinct(`%s`.`%s`))"%(c.column.table.name,c.name) for c in cols])
        tables = set(["`"+c.column.table.name+"`" for c in cols])
        FROM = ",".join(tables)


        qry = """
        SELECT {SELECT} from {FROM}
        """.format(SELECT=SELECT, FROM=FROM)

        counts = self.engine.execute(qry).fetchall()[0]

        return zip(cols,counts)

    def get_rows(self, cols, filter_obj=None, limit=None):
        """
        return rows with values for the columns specificed by col
        """


        qry = self.make_full_table_stmt(cols, filter_obj=filter_obj, limit=limit)
        rows = self.engine.execute(qry)
        return rows


    def get_rows_as_dict(self, cols,limit=None):
        """
        return rows with values for the columns specificed by col
        """
        rows = self.get_rows(cols, limit=limit)
        rows = [dict(r) for r in rows.fetchall()]
        return rows


    ###############################
    # Query helper functions      #
    ###############################
    def make_full_table_stmt(self, cols=None, filter_obj=None, limit=None):
        """
        given a set of colums, make a select statement that generates a table where these columns can be selected from.
        return the string of the query to do this

        this is useful because the columns might reside in different tables, but this helper gets us a table that has them all


        since the true column data is in many different tables, we need to join tables together.
        
        this is done by sorting the columns by the longest paths          
        """
        def join_tables(base_table, join_to, inverse=False):
            # print base_table.name, join_to.name
            # pdb.set_trace()
            join_to_dsm_table = self.db.get_dsm_table(join_to)
            if base_table == join_to_dsm_table.base_table:
                pk = join_to_dsm_table.get_primary_key()
                join_str = """
                    LEFT JOIN `{join_to_table}` ON `{join_to_table}`.`{join_to_col}` = `{base_table}`.`{base_col}`
                    """.format(join_to_table=join_to.name, base_table=base_table.name, join_to_col=pk.column.name, base_col=pk.column.name )
                return join_str

            for fk in base_table.foreign_keys:
                if join_to_dsm_table.has_table(fk.column.table.name):
                    join_str = """
                    LEFT JOIN `{join_to_table}` ON `{join_to_table}`.`{join_to_col}` = `{base_table}`.`{base_col}`
                    """.format(join_to_table=join_to.name, base_table=base_table.name, join_to_col=fk.column.name, base_col=fk.parent.name )
                    return join_str

            #todo decide if this is the best way to handle one to one
            base_table, join_to = join_to, base_table
            join_to_dsm_table = self.db.get_dsm_table(join_to)
            for fk in base_table.foreign_keys:
                if join_to_dsm_table.has_table(fk.column.table.name):
                    join_str = """
                    LEFT JOIN `{base_table}` ON `{join_to_table}`.`{join_to_col}` = `{base_table}`.`{base_col}`
                    """.format(join_to_table=join_to.name, base_table=base_table.name, join_to_col=fk.column.name, base_col=fk.parent.name )
                    return join_str

            # inverse = join_tables(join_to, base_table, inverse=True)

            print "ERROR: ", base_table, join_to
                    
        if cols == None:
            cols = self.get_column_info()

        #todo, check to make sure all cols are legal
        sorted_cols = sorted(cols, key=lambda c: -len(c.metadata['path'])) #sort cols by longest path to shortest


        #iterate over the cols, sorted by length of path, and generate the joins necessary to reach the feature
        joins = []
        # pdb.set_trace()
        for c in sorted_cols:
            #case where column resides in this dsm table
            if c.metadata["path"] == []:
                join_to = c.column.table
                #doesn't exist in the base_table so join necessary
                if join_to != self.base_table:
                    join = join_tables(last_table, join_to)
                    if join not in joins:
                        joins.append(join)
            else:
                last_table = self.base_table
                reversed_path = reversed(c.metadata["path"])
                for i, node in enumerate(reversed_path):
                    if node["feature_type"] == "agg" or i+1 == len(c.metadata["path"]):
                        join_to = c.column.table #if it is an agg feature or last node in path, we need to join to exact table
                        join = join_tables(last_table, join_to)
                        if join not in joins:
                            joins.append(join)
                        break
                    else:
                        join_to = node['base_column'].dsm_table.base_table
                        join = join_tables(last_table, join_to)
                        if join not in joins:
                            joins.append(join)

                    last_table = join_to 
            
        JOIN =  " ".join(joins)
        SELECT = ','.join(["`%s`.`%s`"%(c.column.table.name,c.name) for c in cols])
        FROM = self.base_table.name
        pk = self.get_primary_key()

        WHERE = ""
        if filter_obj != None:
            WHERE = filter_obj.to_where_statement()


        LIMIT = ""
        if limit != None:
            LIMIT = "LIMIT %d" % limit

        qry = """
        SELECT {SELECT}
        FROM `{FROM}`
        {JOIN}
        {WHERE}
        {LIMIT}
        """.format(SELECT=SELECT, FROM=FROM, JOIN=JOIN, primary_key=pk.name, WHERE=WHERE, LIMIT=LIMIT) 

        return qry

    def make_training_filter(self):
        train_filters = self.config.get("train_filter", None)
        all_filters = []
        if train_filters != None:
            for f in train_filters:
                train_filter = (self.get_col_by_name(f[0]), f[1], f[2])
                all_filters.append(train_filter)        
            
            return FilterObject(all_filters)
        return None