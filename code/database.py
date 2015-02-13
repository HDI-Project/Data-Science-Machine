import pdb
from sqlalchemy.schema import MetaData
from sqlalchemy.engine import create_engine
from sqlalchemy.orm import sessionmaker
from table import DSMTable

class Database:
    def __init__(self, url):
        self.engine = create_engine(url)
        self.metadata = MetaData(bind=self.engine)
        self.metadata.reflect()
        self.tables  = dict([(t.name, DSMTable(t, self)) for t in self.metadata.sorted_tables])
        self.session = sessionmaker(bind=self.engine)()

    def get_related_fks(self, table):
        """
        returns a list of the foreign_keys in the database that reference table table
        """
        related_columns = []
        for related in self.tables.values():
            for fk in related.table.foreign_keys:
                if fk.column.table == table.table:
                    related_columns.append(fk)
        return related_columns