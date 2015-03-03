import pdb
from sqlalchemy.schema import MetaData
from sqlalchemy.engine import create_engine
from sqlalchemy.orm import sessionmaker
from table import DSMTable
import cPickle as pickle


class Database:
    def __init__(self, url):
        self.engine = create_engine(url)
        self.metadata = MetaData(bind=self.engine)
        self.metadata.reflect()
        self.tables  = dict([(t.name, DSMTable(t, self)) for t in self.metadata.sorted_tables])
        self.session = sessionmaker(bind=self.engine)()

    def save(self, filename):
        pickle.dump( self, open(filename, "wb" ) )

    @staticmethod
    def load(filename):
        return pickle.load( open( filename, "rb" ) )

    def get_related_fks(self, table):
        """
        returns a list of the foreign_keys in the database that reference table table
        """
        related_columns = []
        for related in self.tables.values():
            for fk in related.table.foreign_keys:
                if fk.column.table.name == table.table.name: # TODO: fix.  nmae is hack because these tables are differenct for some reason
                    related_columns.append(fk)
        return related_columns

    def get_related_tables(self, table):
        """
        return a set of tables that reference table or are referenced by table
        """
        related_tables = set([])
        for related in self.tables.values():
            for fk in related.table.foreign_keys:
                if fk.column.table == table.table:
                    add = self.tables[fk.parent.table.name]
                    related_tables.add(add)

            for fk in table.table.foreign_keys:
                add = self.tables[fk.column.table.name]
                related_tables.add(add)                

        return related_tables