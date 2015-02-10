import pdb
from sqlalchemy import distinct
from sqlalchemy import Table
from sqlalchemy import Column, DateTime, String, Integer, Enum, ForeignKey, func
from sqlalchemy.dialects.mysql.base import DECIMAL
from sqlalchemy.schema import MetaData
from sqlalchemy.engine import create_engine
from sqlalchemy.orm import sessionmaker
from compile_query import compile_query
from sqlalchemy.sql.elements import ColumnClause

from table import Table


class Database:
    def __init__(self, url):
        self.engine = create_engine(url)
        self.metadata = MetaData(bind=self.engine)
        self.metadata.reflect()
        self.tables  = dict([(t.name, Table(t, self)) for t in self.metadata.sorted_tables])
        for t in self.tables:
            # if t == "Order Details":
            self.tables[t].flatten()
        self.session = sessionmaker(bind=self.engine)()
        
    def query(self, *args, **kwargs):
        return self.session.query(*args, **kwargs)

    def get_refs(self, table, backref_only=False):
        """
        takes in a table and returns a list of tables that refer to items

        TODO: what to do about self refercing tables
              look into return columns that ref rather than tables
        """
        refs = []
        for t in self.tables:
           for ref in t.foreign_keys:
                if ref.column.table == table:
                    add = {
                        'table' : t,
                        'column' : ref
                    }
                    pdb.set_trace()
                    refs.append(add)

        return refs

    def get_categorical_columns(self, table, NUM_DISTINCT=10, PERCENT_DISTINCT=.05):
        total = self.query(table).count()
        columns = [c for c in table.columns if len(c.foreign_keys) == 0] #reconsider filtering out foreign_keys
        if len(columns) == 0:
            return []
        s = [func.count(distinct(c)) for c in columns]
        qry = db.session.query(*s)
        cat_columns = [c[1] for c in zip(qry.first(), columns) if c[0] < NUM_DISTINCT and float(c[0])/total <PERCENT_DISTINCT ]

        return cat_columns


    def extra_groups(self,feature_paths):
        """
        add categorial groups that are one step off the created path
        """
        for fp in feature_paths:
            new_path = []
            for node in fp['path']:
                new_path.append(node)
                for ref in self.get_refs(node):
                    if ref in fp['path']:
                        continue

                    cols_add = self.get_categorical_columns(ref) #IDEA: do join with path before trying to find cat columns. this way we might excluded categories that only end up being relevent on certain paths

                    if cols_add != []:
                        fp['groups'] += cols_add
                        new_path.append(ref)


            fp['path'] = new_path


        return feature_paths




    def find_paths(self, table):
        # queue = [[r] for r in get_refs(table, backref_only=True)]
        queue = [
            {
                'path' : [{'table':table, 'join':None}],
                'exclude' : set([table]), #exclude is probably broken
                'columns' : [],
                'groups'  : [],
            }
        ]

        completed_paths = []
        while queue:
            feature_path = queue.pop(0)
            node = feature_path['path'][-1]['table']
            refs = [ref for ref in self.get_refs(node) if ref not in feature_path['exclude']]

            if len(refs) == 0 and len(feature_path['columns']) != 0:    
                completed_paths.append(feature_path)
                continue

            for ref in refs:
                new_path = list(feature_path['path']) + [ref]
                label = '.'.join([table.name for table in new_path])

                new_columns = []
                for c in self.get_column_of_type(ref, DECIMAL):
                    # pdb.set_trace()
                    new_columns.append({
                        'column' : c,
                        'label' : label
                    })

                print node, ref
                # pdb.set_trace()
                new_groups = self.get_categorical_columns(ref) #rethink 

                new_feature_path = {
                    'path'      : new_path,
                    'exclude'   : list(feature_path['exclude']) + refs,
                    'columns'   : list(feature_path['columns']) + new_columns,
                    'groups'    : list(feature_path['groups']) + new_groups

                }

                queue.append(new_feature_path)

        extended_paths = self.extra_groups(completed_paths) #this 
        return extended_paths

    def make_table_features(self, idx):
        table = self.tables[idx]
        primary_keys = [x[1] for x in table.primary_key.columns.items()]

        subquerys = []
        feature_paths = self.find_paths(table)
        for fp in feature_paths:
            select = list(primary_keys)     

            groups = [None] + fp['groups'] #add none so we calculate columns before grouping
            for g in groups:
                vals = [None]
                if g is not None:
                    vals = self.query(g).distinct().all()

                for v in vals:
                    if v != None:
                        v = getattr(v, g.name)

                    for c in fp['columns']:         
                        column = c['column']
                        label = c['label'] 
                        if v != None:
                            label = g.name + '.' + str(v) + '.' + c['label'] 
                        
                        new_select = select + [
                            func.avg(column).label('avg.'+label)
                        ]
                    # pdb.set_trace()
                    sq = self.query(*new_select)
                    for (table,join) in fp['path']:
                        sq = sq.join(table, join)
                    sq = sq.group_by(table)

                    if v != None:
                        sq = sq.filter(g == v)
                        
                    sq = sq.subquery()
                    subquerys.append(sq)

        #get columns from all the subquerys for select statement
        columns = []
        for sq in subquerys:
            columns += [c for c in sq.columns if type(c) == ColumnClause]

        #construct query and join all the subqueries together
        select = table.columns + columns
        qry = self.query(*select)
        for sq in subquerys:
            join = [getattr(sq.c,pk.name)==getattr(table.c,pk.name) for pk in primary_keys]
            qry = qry.outerjoin(sq, *join)

        return qry



if __name__ == "__main__":
    database_name = 'employees'
    db = Database('mysql+mysqldb://kanter@localhost/%s' % (database_name) ) 
    # dsm_database_name = 'dsm'
    # dsm_db = Database('mysql+mysqldb://kanter@localhost/%s' % (dsm_database_name) ) 
    # qry = db.make_table_features(6)
    # qry_str =  compile_query(qry)
    # print qry_str
    # output = qry.all()
    # print str(len(output[0])) + " per row"
    # to_csv(output, table.__tablename__+".csv")