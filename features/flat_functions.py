from feature import FeatureBase
import inflect
import column


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
        last_col = None
        last_target_table_name = None
        last_fk = None
        set_values = []
        for col in to_add:
            new_metadata = col.copy_metadata()

            path_add = {
                'base_column': col,
                'feature_type' : 'flat',
                'feature_type_func' : None
            }

            new_metadata.update({ 
                'path' : new_metadata['path'] + [path_add],
                'categorical' : col.metadata['categorical'],
                'real_name' : prefix + col.metadata['real_name']
            })

            table_name, new_col_name = table.create_column(col.type.compile(), metadata=new_metadata)


            if last_col == None:
                last_col = col

            if last_target_table_name == None:
                last_target_table_name = table_name

            # print last_col.column.table, col.column.table, last_target_table_name, table_name, last_col.column.table != col.column.table or last_tar
            #if this col has different source or target, update database 
            if last_col.column.table.name != col.column.table.name or last_target_table_name != table_name:
                qry = column.make_set_qry(set_values, fk.parent.name, fk.column.name)
                table.flush_columns()
                table.engine.execute(qry)
                set_values=[]

            set_values.append((table_name, new_col_name, col.column.table.name, col.name))

            last_col = col
            last_target_table_name == table_name
            last_fk = fk

        if set_values != []:   
            qry = column.make_set_qry(set_values, fk.parent.name, fk.column.name)
            table.flush_columns()
            table.engine.execute(qry)