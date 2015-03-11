def print_cols_names(table):
	for c in table.columns:
		col = table.columns[c]
		print "real name: %s, database name: %s, type: %s"%(col.metadata['real_name'], c, str(col.type))