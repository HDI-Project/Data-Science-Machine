class FeatureBase(object):
    MAX_PATH_LENGTH = 3
    def __init__(self, db, filter_obj=None):
        self.db = db
        self.filter_obj = filter_obj

    def make_where_stmt(self, alias=None):
        where_stmt = ""
        if self.filter_obj:
            where_stmt = self.filter_obj.to_where_statement(alias)
        return where_stmt

    def get_filter_col_set(self):
        return set(self.get_filter_cols())

    def get_filter_cols(self, include_ignored=True):
        if self.filter_obj == None:
            return []
            
        return self.filter_obj.get_all_cols(include_ignored=include_ignored)