# agg_func_exclude = {
#         'avg' : set(['sum']),
#         'max' : set(['sum']),
#         'min' : set(['sum']),
#         'std' : set(['std'])
#     }
# all_funcs = set(["sum", 'avg', 'std', 'max', 'min'])

functions = [AggSum]

class AggFuncBase():
	name = "AggFuncBase"
	def allowed(self, col):
		"""
		returns true if this agg function can be applied to this col. otherwise returns false
		"""
		pass

	def filter(self, entity):
		"""
		returns true if entity is allowed. otherwise returns false
		"""
		pass

	def function(self, entites):
		pass

	def apply(self):
		pass




class AggSum(AggFuncBase):
	name ="AggSum"

	def __init__(self, where_clause):
		self.where_clause = where_clause

	def col_allowed(self, col):
		"""
		returns true if this agg function can be applied to this col. otherwise returns false
		"""
		if not col['metadata']['numeric']:
			return False

	def apply(self, col, target_col):
		
 