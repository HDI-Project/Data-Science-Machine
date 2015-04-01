class FeatureBase(object):
    MAX_PATH_LENGTH = 3
    def __init__(self, db, filter_obj=None):
        self.db = db
        self.filter_obj = filter_obj

    def make_where_stmt(self):
        where_stmt = ""
        if self.filter_obj:
            where_stmt = self.filter_obj.to_where_statement()
        return where_stmt

    def get_filter_cols(self):
        if self.filter_obj == None:
            return []
            
        return self.filter_obj.get_dependent_cols()