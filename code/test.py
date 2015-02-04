# coding: utf-8
from sqlalchemy import Column, Date, Enum, ForeignKey, Integer, String
from sqlalchemy.orm import relationship


Base = declarative_base()
metadata = Base.metadata


class Department(Base):
    cache_label = u'default'
    cache_regions = regions
    query_class = query_callable(regions)

    __tablename__ = 'departments'

    dept_no = db.Column(db.String(4), primary_key=True)
    dept_name = db.Column(db.String(40), nullable=False, unique=True)


class DeptEmp(Base):
    cache_label = u'default'
    cache_regions = regions
    query_class = query_callable(regions)

    __tablename__ = 'dept_emp'

    emp_no = db.Column(db.ForeignKey(u'employees.emp_no', ondelete=u'CASCADE'), primary_key=True, nullable=False, index=True)
    dept_no = db.Column(db.ForeignKey(u'departments.dept_no', ondelete=u'CASCADE'), primary_key=True, nullable=False, index=True)
    from_date = db.Column(db.Date, nullable=False)
    to_date = db.Column(db.Date, nullable=False)

    department = db.relationship(u'Department', primaryjoin='DeptEmp.dept_no == Department.dept_no', backref=u'dept_emps')
    employee = db.relationship(u'Employee', primaryjoin='DeptEmp.emp_no == Employee.emp_no', backref=u'dept_emps')


class DeptManager(Base):
    cache_label = u'default'
    cache_regions = regions
    query_class = query_callable(regions)

    __tablename__ = 'dept_manager'

    dept_no = db.Column(db.ForeignKey(u'departments.dept_no', ondelete=u'CASCADE'), primary_key=True, nullable=False, index=True)
    emp_no = db.Column(db.ForeignKey(u'employees.emp_no', ondelete=u'CASCADE'), primary_key=True, nullable=False, index=True)
    from_date = db.Column(db.Date, nullable=False)
    to_date = db.Column(db.Date, nullable=False)

    department = db.relationship(u'Department', primaryjoin='DeptManager.dept_no == Department.dept_no', backref=u'dept_managers')
    employee = db.relationship(u'Employee', primaryjoin='DeptManager.emp_no == Employee.emp_no', backref=u'dept_managers')


class Employee(Base):
    cache_label = u'default'
    cache_regions = regions
    query_class = query_callable(regions)

    __tablename__ = 'employees'

    emp_no = db.Column(db.Integer, primary_key=True)
    birth_date = db.Column(db.Date, nullable=False)
    first_name = db.Column(db.String(14), nullable=False)
    last_name = db.Column(db.String(16), nullable=False)
    gender = db.Column(db.Enum(u'M', u'F'), nullable=False)
    hire_date = db.Column(db.Date, nullable=False)


class Salary(Base):
    cache_label = u'default'
    cache_regions = regions
    query_class = query_callable(regions)

    __tablename__ = 'salaries'

    emp_no = db.Column(db.ForeignKey(u'employees.emp_no', ondelete=u'CASCADE'), primary_key=True, nullable=False, index=True)
    salary = db.Column(db.Integer, nullable=False)
    from_date = db.Column(db.Date, primary_key=True, nullable=False)
    to_date = db.Column(db.Date, nullable=False)

    employee = db.relationship(u'Employee', primaryjoin='Salary.emp_no == Employee.emp_no', backref=u'salaries')


class Title(Base):
    cache_label = u'default'
    cache_regions = regions
    query_class = query_callable(regions)

    __tablename__ = 'titles'

    emp_no = db.Column(db.ForeignKey(u'employees.emp_no', ondelete=u'CASCADE'), primary_key=True, nullable=False, index=True)
    title = db.Column(db.String(50), primary_key=True, nullable=False)
    from_date = db.Column(db.Date, primary_key=True, nullable=False)
    to_date = db.Column(db.Date)

    employee = db.relationship(u'Employee', primaryjoin='Title.emp_no == Employee.emp_no', backref=u'titles')
