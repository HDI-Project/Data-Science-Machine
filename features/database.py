import pdb
from sqlalchemy.schema import MetaData
from sqlalchemy.engine import create_engine
from sqlalchemy.orm import sessionmaker
from table import DSMTable
# import cPickle as pickle
import dill
import threading

class Database:
    def __init__(self, url, config=None):
        self.url = url
        self.engine = self.make_engine(url)
        self.metadata = MetaData(bind=self.engine)
        self.metadata.reflect()

        self.config = config

        #parallel table init
        self.tables_lock = threading.Lock()
        self.tables = {}
        threads = []
        for table in self.metadata.sorted_tables:
            t = threading.Thread(target=self.make_dsm_table, args=(table,))
            t.start()
            threads.append(t)
        [t.join() for t in threads]

    def execute(self, qry):
        try:
            res = self.engine.execute(qry)
        except Exception, e:
            if e.message == "(OperationalError) (1205, 'Lock wait timeout exceeded; try restarting transaction')":
                print e
                res = self.execute(qry)
            else:
                print e
                raise e

        return res

    def make_dsm_table(self, table):
        dsm_table = DSMTable(table, self)

        self.tables_lock.acquire()
        self.tables[table.name] = dsm_table
        self.tables_lock.release()

    def __getstate__(self):
        """
        prepare class for pickling
        """
        state = self.__dict__.copy()
   
        #pickle db connection
        del state['engine']
        del state['metadata']

        return state

    def __setstate__(self, state):
        #unpickle db state
        state['engine'] = self.make_engine(state['url'])
        state['metadata'] = MetaData(bind=state['engine']).reflect()
        self.__dict__.update(state) #update now so we have these properties when we call set_db

        #make sure tables have db reference
        for t in state['tables']:
            state['tables'][t].set_db(self)

        self.__dict__.update(state)


    def save(self, filename):
        dill.dump( self, open(filename, "wb" ) )

    @staticmethod
    def load(filename):
        db = dill.load( open( filename, "rb" ) )
        return db

    def make_engine(self, url):
        return create_engine(url, pool_size=20)

    def get_related_fks(self, table):
        """
        returns a list of the foreign_keys in the database that reference table table
        """
        related_columns = []
        for related in self.tables.values():
            for fk in related.base_table.foreign_keys:
                if fk.column.table.name == table.base_table.name: # TODO: fix.  nmae is hack because these tables are differenct for some reason
                    related_columns.append(fk)
        return related_columns


    def get_dsm_table(self, db_table):
        for t in self.tables.values():
            if db_table.name in t.tables:
                return t

        raise Exception("No dsm table with provied db table")