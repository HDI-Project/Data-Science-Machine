def get_col_names(table):
	cols = []
	for col in table.get_column_info():
		cols.append(col.metadata['real_name'])

	return cols