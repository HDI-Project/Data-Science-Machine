import pdb
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

import sqlalchemy.dialects.mysql.base as column_datatypes
from sqlalchemy.schema import Table

from sqlalchemy.schema import MetaData
from column import DSMColumn

from collections import defaultdict

import threading


class DSMTable:
    MAX_COLS_TABLE = 200

    def __init__(self, table, db):
        self.db = db
        self.name = table.name
        self.base_table = table
        self.tables = {table.name: table}
        self.engine = db.engine

        self.primary_key_names = [key.name for key in table.primary_key]

        self.columns = {}
        self.cols_to_add = defaultdict(list)
        self.cols_to_drop = defaultdict(list)

        self.num_added_tables = 0
        self.table_col_counts = {}
        self.curr_table = None

        self.num_rows = self.engine.execute("SELECT count(*) from `%s`" % (self.name)).fetchall()[0][0]

        self.init_columns()

        self.lock = threading.Lock()

    def __getstate__(self):
        """
        prepare class for pickling
        """
        state = self.__dict__.copy()
        del state['db']        
        del state['engine']
        return state

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

            categorical = len(col.get_distinct_vals())<=2


            col.update_metadata({
                'numeric' : type(col.type) in datatypes and not (col.primary_key or col.has_foreign_key),
                'real_name' : col.name,
                'categorical' : categorical
            })

            if categorical:
                print "cat column", col.metadata["real_name"], self.name

            self.columns[(col.column.table.name,col.name)] = col

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
            print

        self.tables[new_table_name] = Table(new_table_name, MetaData(bind=self.engine), autoload=True, autoload_with=self.engine)
        self.table_col_counts[new_table_name] = 0
        return self.tables[new_table_name]


    def make_column_name(self):
        if (self.curr_table == None or
           self.table_col_counts[self.curr_table.name] >= self.MAX_COLS_TABLE):

            self.curr_table = self.make_new_table()

        name = self.curr_table.name + "__" +  str(self.table_col_counts[self.curr_table.name])
        self.table_col_counts[self.curr_table.name] +=1
        return self.curr_table.name,name

    def execute(self, qry):

        try:
            self.engine.execute(qry)
        except Exception, e:
            if e.message == "(OperationalError) (1205, 'Lock wait timeout exceeded; try restarting transaction')":
                self.execute(qry)
            else:
                print e

    #############################
    # Database operations       #
    #############################
    def create_column(self, column_type, metadata={},flush=False, drop_if_exists=True):
        """
        add column with name column_name of type column_type to this table. if column exists, drop first

        todo: suport where to add it
        """
        self.lock.acquire()
        if (type(column_type) == DSMColumn):
            self.columns[(column_type.column.table.name,column_type.name)] = column_type
            print column_type.name, column_type.metadata["real_name"]
            self.lock.release()
            return column_type.column.table.name,column_type.name
        

        table_name,column_name = self.make_column_name()
        self.cols_to_add[table_name] += [(column_name, column_type, metadata)]
        print column_name, metadata["real_name"]
        self.lock.release()
        if flush:
            self.flush_columns(drop_if_exists=drop_if_exists)

        return table_name,column_name
    
    def drop_column(self, table_name, column_name, flush=False):
        """
        drop column with name column_name from this table
        """
        self.cols_to_drop[table_name] += [column_name]
        if flush:
            self.flush_columns(drop_if_exists=drop_if_exists)

    def flush_columns(self, drop_if_exists=True):
        self.lock.acquire()
        #first, check which of cols_to_add need to be dropped first
        for table_name in self.cols_to_add:
            for (name, col_type, metadata) in self.cols_to_add[table_name]:
                if drop_if_exists and self.has_column(table_name,name):
                    self.drop_column(table_name, name)

        #second, flush columns that need to be dropped
        for table_name in self.cols_to_drop:
            values = []
            for name in self.cols_to_drop[table_name]:
                del self.columns[(table_name, name)]
                values.append("DROP `%s`" % (name))
            if len(values) > 0:
                values = ", ".join(values)
                self.engine.execute(
                    """
                    ALTER TABLE `{table}`
                    {cols_to_drop}
                    """.format(table=table_name, cols_to_drop=values)
                ) #very bad, fix how parameters are substituted in

                self.cols_to_drop[table_name] = []
            
        #third, flush columns that need to be added
        for table_name in self.cols_to_add:
            values = []
            new_col_metadata = {}
            for (name, col_type, metadata) in self.cols_to_add[table_name]:
                new_col_metadata[name] = metadata
                values.append("ADD COLUMN `%s` %s" % (name, col_type))

            if len(values) > 0:
                values = ", ".join(values)
                qry = """
                    ALTER TABLE `{table}`
                    {cols_to_add}
                    """.format(table=table_name, cols_to_add=values)
                self.engine.execute(qry)
                self.cols_to_add[table_name] = []

            #reflect table again to have update columns
            # TODO check to make sure old column reference still work
            self.tables[table_name] = Table(table_name, MetaData(bind=self.engine), autoload=True, autoload_with=self.engine)
            
            #for every column in the database, make sure we have it accounted for in our data structure
            for c in self.tables[table_name].c:
                if c.name in new_col_metadata:
                        col = DSMColumn(c, dsm_table=self, metadata=new_col_metadata[c.name])
                        self.columns[(col.column.table.name,col.name)] = col

        self.lock.release()

    ###############################
    # Table info helper functions #
    ###############################
    def get_column_info(self, prefix='', ignore_relationships=False, match_func=None, first=False, set_trace=False):
        """
        return info about columns in this table. 
        info should be things that are read directly from database or something that is dynamic at query time. everything else should be part of metadata

        """
        cols = []
        for col in self.columns.values():
            if ignore_relationships and col.primary_key:
                continue

            if ignore_relationships and col.has_foreign_key:
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
            add = self.db.tables[fk.column.table.name]
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
                    add = self.db.tables[fk.parent.table.name]
                    child_tables.add(add)

        return child_tables

    def get_related_tables(self):
        """
        return a set of tables that reference table or are referenced by table
        """
        children = self.get_child_tables()
        parents = self.get_parent_tables()
        return parents.union(children)

    def get_col_by_name(self, col_name):
        return self.get_column_info(match_func=lambda c, col_name=col_name: c.name == col_name, first=True)

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

    def get_rows(self, cols, limit=None):
        """
        return rows with values for the columns specificed by col
        """


        qry = self.make_full_table_stmt(cols, limit=limit)
        rows = self.engine.execute(qry)
        return rows


    def get_rows_as_dict(self, cols):
        """
        return rows with values for the columns specificed by col
        """
        rows = self.get_rows(cols)
        rows = [dict(r) for r in rows.fetchall()]
        return rows


    ###############################
    # Query helper functions      #
    ###############################
    def make_full_table_stmt(self, cols=None, limit=None):
        """
        given a set of colums, make a select statement that generates a table where these columns can be selected from.
        return the string of the query to do this

        this is useful because the columns might reside in different tables, but this helper gets us a table that has them all


        since the true column data is in many different tables, we need to join tables together.
        
        this is done by sorting the columns by the longest paths          
        """
        def join_tables(base_table, join_to):
            join_to_dsm_table = self.db.get_dsm_table(join_to)
            if base_table == join_to_dsm_table.base_table:
                pk = join_to_dsm_table.get_primary_key()
                join_str = """
                    JOIN `{join_to_table}` ON `{join_to_table}`.`{join_to_col}` = `{base_table}`.`{base_col}`
                    """.format(join_to_table=join_to.name, base_table=base_table.name, join_to_col=pk.column.name, base_col=pk.column.name )
                return join_str

            for fk in base_table.foreign_keys:
                if join_to_dsm_table.has_table(fk.column.table.name):
                    join_str = """
                    JOIN `{join_to_table}` ON `{join_to_table}`.`{join_to_col}` = `{base_table}`.`{base_col}`
                    """.format(join_to_table=join_to.name, base_table=base_table.name, join_to_col=fk.column.name, base_col=fk.parent.name )
                    return join_str

            print "ERROR: ", base_table, join_to
                    
        if cols == None:
            cols = self.get_column_info()

        #todo, check to make sure all cols are legal
        sorted_cols = sorted(cols, key=lambda c: -len(c.metadata['path'])) #sort cols by longest path to shortest


        #iterate over the cols, sorted by length of path, and generate the joins necessary to reach the feature
        joins = []
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

        if limit != None:
            LIMIT = "LIMIT %d" % limit
        else:
            LIMIT = ""

        qry = """
        SELECT {SELECT}
        FROM `{FROM}`
        {JOIN}
        {LIMIT}
        """.format(SELECT=SELECT, FROM=FROM, JOIN=JOIN, primary_key=pk.name, LIMIT=LIMIT) 

        # print qry       
        return qry