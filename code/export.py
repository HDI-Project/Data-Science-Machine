import numpy as np
from features import make_all_features
import pdb

def export_table(table, folder="out/"):
	all_cols = table.columns.values()
	all_rows = table.get_rows(all_cols)

	#figure out which columns are in each interval
	interval_cols = {}
	for c in all_cols:
		interval_num = c.metadata.get("interval_num", None)
		if interval_num == None:
			continue
		
		if interval_num not in interval_cols:
			interval_cols[interval_num] = []

		interval_cols[interval_num] += [c]

	#hack to put cols in same order
	for i in interval_cols:
		interval_cols[i] =  sorted(interval_cols[i], key=lambda x: x.name)

	max_interval = max(interval_cols.keys())

	#export file for each row
	for row in all_rows:
		out = []
		for i in xrange(max_interval):
			features = [i]
			# print ",".join([c.name for c in interval_cols[i]])
			for col in interval_cols[i]:
				idx = all_cols.index(col)
				f = row[idx]
				features.append(f)

			out.append(features)

		out = np.asarray(out, dtype=np.float)


		filename = ""
		for col_name in table.primary_key_names:
			col = table.columns[col_name]
			idx = all_cols.index(col)
			filename += str(row[idx])

		filename = "%s%s.csv" % (folder, filename)
		np.savetxt(filename, out, delimiter=",", fmt='%.20g',)


if __name__ == "__main__":
    import os
    from database import Database

    os.system("mysql -t < ../Northwind.MySQL5.sql")
    database_name = 'northwind'
    db = Database('mysql+mysqldb://kanter@localhost/%s' % (database_name)) 

    table_name = "Products"
    make_all_features(db, db.tables[table_name])
    export_table(db.tables[table_name])
    # find_correlations(db, db.tables[table_name])


