import utils
def print_cols_names(table):
	for col in table.get_column_info():
		print "real name: %s, database name: %s, type: %s"%(col.metadata['real_name'], col.column.name, str(col.type))


def export_col_names(table):
	cols  = utils.get_col_names(table)
	print "\n".join([str(x) for x in cols])

