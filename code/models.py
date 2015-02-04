# coding: utf-8
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, Numeric, SmallInteger, String, Table
from sqlalchemy.dialects.mysql.base import BIT, LONGBLOB
from sqlalchemy.orm import relationship
from sqlalchemy.schema import FetchedValue
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()
metadata = Base.metadata


class Category(Base):
    __tablename__ = 'Categories'

    CategoryID = Column(Integer, primary_key=True)
    CategoryName = Column(String(15), nullable=False, index=True)
    Description = Column(String)
    Picture = Column(LONGBLOB)


t_CustomerCustomerDemo = Table(
    'CustomerCustomerDemo',
    Column('CustomerID', ForeignKey(u'Customers.CustomerID'), primary_key=True, nullable=False),
    Column('CustomerTypeID', ForeignKey(u'CustomerDemographics.CustomerTypeID'), primary_key=True, nullable=False, index=True)
)


class CustomerDemographic(Base):
    __tablename__ = 'CustomerDemographics'

    CustomerTypeID = Column(String(10), primary_key=True)
    CustomerDesc = Column(String)


class Customer(Base):
    __tablename__ = 'Customers'

    CustomerID = Column(String(5), primary_key=True)
    CompanyName = Column(String(40), nullable=False, index=True)
    ContactName = Column(String(30))
    ContactTitle = Column(String(30))
    Address = Column(String(60))
    City = Column(String(15), index=True)
    Region = Column(String(15), index=True)
    PostalCode = Column(String(10), index=True)
    Country = Column(String(15))
    Phone = Column(String(24))
    Fax = Column(String(24))

    CustomerDemographics = relationship(u'CustomerDemographic', secondary='CustomerCustomerDemo', backref=u'customers')


t_EmployeeTerritories = Table(
    'EmployeeTerritories',
    Column('EmployeeID', ForeignKey(u'Employees.EmployeeID'), primary_key=True, nullable=False),
    Column('TerritoryID', ForeignKey(u'Territories.TerritoryID'), primary_key=True, nullable=False, index=True)
)


class Employee(Base):
    __tablename__ = 'Employees'

    EmployeeID = Column(Integer, primary_key=True)
    LastName = Column(String(20), nullable=False, index=True)
    FirstName = Column(String(10), nullable=False)
    Title = Column(String(30))
    TitleOfCourtesy = Column(String(25))
    BirthDate = Column(DateTime)
    HireDate = Column(DateTime)
    Address = Column(String(60))
    City = Column(String(15))
    Region = Column(String(15))
    PostalCode = Column(String(10), index=True)
    Country = Column(String(15))
    HomePhone = Column(String(24))
    Extension = Column(String(4))
    Photo = Column(LONGBLOB)
    Notes = Column(String, nullable=False)
    ReportsTo = Column(ForeignKey(u'Employees.EmployeeID'), index=True)
    PhotoPath = Column(String(255))
    Salary = Column(Float)

    parent = relationship(u'Employee', remote_side=[EmployeeID], primaryjoin='Employee.ReportsTo == Employee.EmployeeID', backref=u'employees')
    Territories = relationship(u'Territory', secondary='EmployeeTerritories', backref=u'employees')


class OrderDetail(Base):
    __tablename__ = 'Order Details'

    OrderID = Column(ForeignKey(u'Orders.OrderID'), primary_key=True, nullable=False)
    ProductID = Column(ForeignKey(u'Products.ProductID'), primary_key=True, nullable=False, index=True)
    UnitPrice = Column(Numeric(10, 4), nullable=False, server_default=db.FetchedValue())
    Quantity = Column(SmallInteger, nullable=False, server_default=db.FetchedValue())
    Discount = Column(Float(8, True), nullable=False, server_default=db.FetchedValue())

    Order = relationship(u'Order', primaryjoin='OrderDetail.OrderID == Order.OrderID', backref=u'order_details')
    Product = relationship(u'Product', primaryjoin='OrderDetail.ProductID == Product.ProductID', backref=u'order_details')


class Order(Base):
    __tablename__ = 'Orders'

    OrderID = Column(Integer, primary_key=True)
    CustomerID = Column(ForeignKey(u'Customers.CustomerID'), index=True)
    EmployeeID = Column(ForeignKey(u'Employees.EmployeeID'), index=True)
    OrderDate = Column(DateTime, index=True)
    RequiredDate = Column(DateTime)
    ShippedDate = Column(DateTime, index=True)
    ShipVia = Column(ForeignKey(u'Shippers.ShipperID'), index=True)
    Freight = Column(Numeric(10, 4), server_default=db.FetchedValue())
    ShipName = Column(String(40))
    ShipAddress = Column(String(60))
    ShipCity = Column(String(15))
    ShipRegion = Column(String(15))
    ShipPostalCode = Column(String(10), index=True)
    ShipCountry = Column(String(15))

    Customer = relationship(u'Customer', primaryjoin='Order.CustomerID == Customer.CustomerID', backref=u'orders')
    Employee = relationship(u'Employee', primaryjoin='Order.EmployeeID == Employee.EmployeeID', backref=u'orders')
    Shipper = relationship(u'Shipper', primaryjoin='Order.ShipVia == Shipper.ShipperID', backref=u'orders')


class Product(Base):
    __tablename__ = 'Products'

    ProductID = Column(Integer, primary_key=True)
    ProductName = Column(String(40), nullable=False, index=True)
    SupplierID = Column(ForeignKey(u'Suppliers.SupplierID'), index=True)
    CategoryID = Column(ForeignKey(u'Categories.CategoryID'), index=True)
    QuantityPerUnit = Column(String(20))
    UnitPrice = Column(Numeric(10, 4), server_default=db.FetchedValue())
    UnitsInStock = Column(SmallInteger, server_default=db.FetchedValue())
    UnitsOnOrder = Column(SmallInteger, server_default=db.FetchedValue())
    ReorderLevel = Column(SmallInteger, server_default=db.FetchedValue())
    Discontinued = Column(BIT(1), nullable=False)

    Category = relationship(u'Category', primaryjoin='Product.CategoryID == Category.CategoryID', backref=u'products')
    Supplier = relationship(u'Supplier', primaryjoin='Product.SupplierID == Supplier.SupplierID', backref=u'products')


class Region(Base):
    __tablename__ = 'Region'

    RegionID = Column(Integer, primary_key=True)
    RegionDescription = Column(String(50), nullable=False)


class Shipper(Base):
    __tablename__ = 'Shippers'

    ShipperID = Column(Integer, primary_key=True)
    CompanyName = Column(String(40), nullable=False)
    Phone = Column(String(24))


class Supplier(Base):
    __tablename__ = 'Suppliers'

    SupplierID = Column(Integer, primary_key=True)
    CompanyName = Column(String(40), nullable=False, index=True)
    ContactName = Column(String(30))
    ContactTitle = Column(String(30))
    Address = Column(String(60))
    City = Column(String(15))
    Region = Column(String(15))
    PostalCode = Column(String(10), index=True)
    Country = Column(String(15))
    Phone = Column(String(24))
    Fax = Column(String(24))
    HomePage = Column(String)


class Territory(Base):
    __tablename__ = 'Territories'

    TerritoryID = Column(String(20), primary_key=True)
    TerritoryDescription = Column(String(50), nullable=False)
    RegionID = Column(ForeignKey(u'Region.RegionID'), nullable=False, index=True)

    Region = relationship(u'Region', primaryjoin='Territory.RegionID == Region.RegionID', backref=u'territories')
