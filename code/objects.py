from sqlalchemy import *
from sqlacodegen.codegen import CodeGenerator, ManyToOneRelationship
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, DateTime, String, Integer, Enum, ForeignKey, func
from sqlalchemy.inspection import inspect
from sqlalchemy.sql.elements import ColumnClause
from literalquery import literalquery


import sys
import pdb
import csv
def compile_query(query):
    from sqlalchemy.sql import compiler
    from MySQLdb.converters import conversions, escape

    dialect = query.session.bind.dialect
    statement = query.statement
    comp = compiler.SQLCompiler(dialect, statement)
    comp.compile()
    enc = dialect.encoding
    params = []
    for k in comp.positiontup:
        v = comp.params[k]
        if isinstance(v, unicode):
            v = v.encode(enc)
        params.append( escape(v, conversions) )
    return (comp.string.encode(enc) % tuple(params)).decode(enc)

class Database:
	def __init__(self, url):
		self.engine = create_engine(url)
		self.session = sessionmaker(bind=self.engine)()
		self.models = self.get_models()

	def get_models(self):
		def classesinmodule(module):
		    md = module.__dict__
		    return [
		        md[c] for c in md if (
		            isinstance(md[c], type) and md[c].__module__ == module.__name__
		        )
		    ]
		
		metadata = MetaData(self.engine)
		metadata.reflect(self.engine) #TODO: accept only tables I want to process
		generator = CodeGenerator(metadata)
		generator.collector.add_literal_import('sqlalchemy.ext.declarative','declarative_base')
		generator.render(open('models.py', 'w'))

		import models
		return classesinmodule(models)


def find_paths(model):
	# queue = [[r] for r in get_refs(model, backref_only=True)]
	queue = [
		{
			'path' : [model],
			'exclude' : set([model]),
			'columns' : [],
			'groups'  : [],
		}
	]

	completed_paths = []
	while queue:
		feature_path = queue.pop(0)
		node = feature_path['path'][-1]
		refs = [ref for ref in get_refs(node) if ref not in feature_path['exclude']]

		if len(refs) == 0 and len(feature_path['columns']) != 0:	
			completed_paths.append(feature_path)
			continue

		for ref in refs:
			new_path = list(feature_path['path']) + [ref]
			label = '.'.join([n.__tablename__ for n in new_path])

			new_columns = []
			for c in get_column_of_type(ref, Integer):
				new_columns.append({
					'column' : c,
					'label'	: label
				})

			print node, ref
			# pdb.set_trace()
			new_groups = get_categorical_columns(ref) #rethink 

			# print new_groups, label

			new_feature_path = {
				'path'		: new_path,
				'exclude'  	: list(feature_path['exclude']) + refs,
				'columns'  	: list(feature_path['columns']) + new_columns,
				'groups'	: list(feature_path['groups']) + new_groups

			}

			queue.append(new_feature_path)

	extended_paths = extra_groups(completed_paths) #this 
	return extended_paths


def get_categorical_columns(model, NUM_DISTINCT=10, PERCENT_DISTINCT=.05):
	total = db.session.query(model).count()
	columns = [c for c in model.__table__.columns if len(c.foreign_keys) == 0] #reconsider filtering out foreign_keys
	if len(columns) == 0:
		return []
	s = [func.count(distinct(c)) for c in columns]
	qry = db.session.query(*s)
	cat_columns = [c[1] for c in zip(qry.first(), columns) if c[0] < NUM_DISTINCT and float(c[0])/total <PERCENT_DISTINCT ]

	return cat_columns


def extra_groups(feature_paths):
	"""
	add categorial groups that are one step off the created path
	"""
	for fp in feature_paths:
		new_path = []
		for node in fp['path']:
			new_path.append(node)
			for ref in get_refs(node):
				if ref in fp['path']:
					continue

				cols_add = get_categorical_columns(ref) #IDEA: do join with path before trying to find cat columns. this way we might excluded categories that only end up being relevent on certain paths

				if cols_add != []:
					fp['groups'] += cols_add
					new_path.append(ref)


		fp['path'] = new_path


	return feature_paths


def get_column_of_type(model, t, allow_primary=False):
	# pdb.set_trace()
	cols = []
	for c in model.__table__.columns:
		if not type(c.type) == t or c.primary_key or len(c.foreign_keys) != 0:
			continue
		cols.append(c)

	return cols


def get_primary_keys(model):
	return [c for c in model.__table__.columns if c.primary_key]


def get_refs(model, backref_only=False):
	refs = inspect(model).relationships.values()
	if backref_only:
		refs = [r.mapper.entity for r in refs if not r.backref]
	else:
		refs = [r.mapper.entity for r in refs]

	return refs

def make_features(model, db):
	subquerys = []
	feature_paths = find_paths(model)
	print feature_paths[0]
	primary_keys = [x[1] for x in model.__table__.primary_key.columns.items()]
	for fp in feature_paths:
		select = list(primary_keys)		
		groups = [None] + fp['groups'] #add none so we calculate columns before grouping
		for g in groups:
			vals = [None]

			if g is not None:
				vals = db.session.query(g).distinct().all()

			for v in vals:
				if v != None:
					v = getattr(v, g.name)
				for c in fp['columns']:			
					column = c['column']
					label = c['label'] 
					if v != None:
						label = g.name + '.' + str(v) + '.' + c['label'] 
					
					new_select = select + [
						func.avg(column).label('avg.'+label),
						# func.std(column).label('std.'+label),
						# func.max(column).label('max.'+label),
						# func.min(column).label('min.'+label),
						# func.sum(column).label('sum.'+label),
						# func.count(column).label('count.'+label),
					]
				# pdb.set_trace()
				sq = db.session.query(*new_select)
				sq = sq.join(*fp['path'][1:])
				sq = sq.group_by(model)

				if v != None:
					sq = sq.filter(g == v)
					
				print
				print sq
				# sq = alias(sq.subquery(), label)
				sq = sq.subquery()
				# new_sq = alias(new_sq, g.name + '.' + str(v))
				subquerys.append(sq)

	#get columns from all the subquerys for select statement
	columns = []
	for sq in subquerys:
		columns += [c for c in sq.columns if type(c) == ColumnClause]

	#construct query and join all the subqueries together
	select = model.__table__.columns + columns
	qry = db.session.query(*select)
	for sq in subquerys:
		join = [getattr(sq.c,pk.name)==getattr(model,pk.name) for pk in primary_keys]
		qry = qry.outerjoin(sq, *join)

	# print literalquery(qry)

	
	# print results
	return qry

def to_csv(objects, filename):
    with open(filename, 'w') as csvfile:
        fields = objects[0]._labels
        writer = csv.writer(csvfile, dialect="excel")
        writer.writerow(fields)
        for o in objects:
            vals = [getattr(o,f) for f in fields]
            writer.writerow(vals)

if __name__ == "__main__":
	database_name = 'northwind'
	db = Database('mysql+mysqldb://kanter@localhost/%s' % (database_name) )	
	model = db.models[4]
	qry = make_features(model, db)
	qry_str =  compile_query(qry)
	print qry_str
	output = qry.all()
	print str(len(output[0])) + " per row"
	to_csv(output, model.__tablename__+".csv")

	# for m in db.models:
	# 	print make_features(m, db)


#db.session.query(models.Department, func.sum(models.Salary.salary)).join(models.DeptEmp).join(models.Employee).join(models.Salary).group_by(models.Department).all()
# db.session.query(m[4].dept_name, m[1].gender, func.avg(m[0].salary)).join(m[5]).join(m[1]).group_by(m[1].gender, m[4]).join(m[0]).order_by(m[4].dept_no)


#  a = db.session.query(m[4].dept_name,func.avg(m[0].salary).label('avg_salary'), 'gender').join(m[5]).join(m[1]).group_by(m[4]).join(m[0]).order_by(m[4].dept_no).filter(m[1].gender == "M").subquery()
#  b = db.session.query(m[4].dept_name,func.avg(m[0].salary).label('avg_salary'), 'gender').join(m[5]).join(m[1]).group_by(m[4]).join(m[0]).order_by(m[4].dept_no).filter(m[1].gender == "F").subquery()
#  db.session.query(m[4].dept_name, b.c.avg_salary.label('female'), a.c.avg_salary.label('male')).outerjoin(a,m[4].dept_name==a.c.dept_name).outerjoin(b,m[4].dept_name==b.c.dept_name).all()


#count distinct values in each column