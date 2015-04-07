from feature import FeatureBase
import inflect
import column
import pdb

p = inflect.engine()

class FlatFeature(FeatureBase):
    def apply(self,fk):
        """
        brings in all flat features according to the foreign key provided
        """
        table = self.db.tables[fk.parent.table.name]
        parent_table = self.db.tables[fk.column.table.name]

        #if the fk has a special column name in parent table, keep it. otherwise use the name of the foreign table
        if fk.parent.name != fk.column.name:
            prefix = fk.parent.name
        else:
            prefix = fk.column.name
            singular = p.singular_noun(fk.column.name)
            if singular:
                prefix = singular

        prefix += "."

        to_add = parent_table.get_column_info(ignore_relationships=True)
        for col in to_add:
            new_metadata = col.copy_metadata()

            path_add = {
                'base_column': col,
                'feature_type' : 'flat',
                'feature_type_func' : None
            }

            new_metadata.update({ 
                'path' : new_metadata['path'] + [path_add],
                'real_name' : prefix + col.metadata['real_name']
            })

            new_col = column.DSMColumn(col.column, table, new_metadata)

            # pdb.set_trace()
            table.create_column(new_col)
