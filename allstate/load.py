import os
import sys
from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import csv
 
#customer_ID,shopping_pt,record_type,day,time,state,location,group_size,homeowner,car_age,car_value,risk_factor,age_oldest,age_youngest,married_couple,C_previous,duration_previous,A,B,C,D,E,F,G,cost

Base = declarative_base()
 
class Customer(Base):
    __tablename__ = 'Customers'
    # Here we define columns for the table person
    # Notice that each column is also a normal Python instance attribute.
    customer_ID = Column(Integer, primary_key=True)
 
class Quote(Base):
    __tablename__ = 'Quotes'
    # Here we define columns for the table address.
    # Notice that each column is also a normal Python instance attribute.
    id = Column(Integer, primary_key=True)

    customer_ID = Column(Integer, ForeignKey('Customers.customer_ID'))
    location = Column(Integer, ForeignKey('Locations.location'))
    state = Column(String(250), ForeignKey('States.state'))


    C_previous = Column(Integer, primary_key=False)
    duration_previous = Column(Integer, primary_key=False)
    married_couple = Column(Integer, primary_key=False)
    car_age = Column(Integer, nullable=False)
    homeowner = Column(Integer, primary_key=False)
    car_value = Column(String(250), primary_key=False)
    group_size = Column(String(250), primary_key=False)
    shopping_pt = Column(Integer, primary_key=False)
    record_type = Column(Integer, primary_key=False)
    risk_factor = Column(Integer, primary_key=False)
    day = Column(Integer, primary_key=False)
    time = Column(String(250), primary_key=False)
    age_oldest = Column(Integer, primary_key=False)
    age_youngest = Column(Integer, primary_key=False)
    A = Column(Integer, primary_key=False)
    B = Column(Integer, primary_key=False)
    C = Column(Integer, primary_key=False)
    D = Column(Integer, primary_key=False)
    E = Column(Integer, primary_key=False)
    F = Column(Integer, primary_key=False)
    G = Column(Integer, primary_key=False)
    cost = Column(Integer, primary_key=False) 

class Location(Base):
    __tablename__ = 'Locations'
    location=Column(Integer, primary_key=True)

class State(Base):
    __tablename__ = 'States'
    state= Column(String(250), primary_key=True)


engine = create_engine("mysql+mysqldb://kanter@localhost/allstate")

tables = ["Quotes", "Customers", "Locations", "States"]
for t in tables:
    try:
        engine.execute("drop table %s" % (t)) 
    except:
        pass


Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
# create a Session
session = Session()

filename = "train.csv"
reader=csv.reader(open(filename,"rb"),delimiter=',')
header = reader.next()
idx = dict([(x,i) for i,x in enumerate(header)]) #map field name to index in row
rows = [r for r in reader]



print "first pass, create customers, locations, and states"
customers = set([])
locations = set([])
states = set([])
customer_fields = ["customer_ID"]
location_fields = ["location"]
state_fields = ["state"]
for r in rows:
    c_insert = frozenset([(f, r[idx[f]]) for f in customer_fields])
    customers.add(c_insert)

    l_insert = frozenset([(f, r[idx[f]]) for f in location_fields])
    locations.add(l_insert)

    s_insert = frozenset([(f, r[idx[f]]) for f in state_fields])
    states.add(s_insert)


print "commit customers"
for c in customers:
    session.add(Customer(**dict(c)))
session.commit()

print "commit locations"
for l in locations:
    session.add(Location(**dict(l)))
session.commit()

print "commit states"
for s in states:        
    session.add(State(**dict(s)))

session.commit()

print "second pass for quotes"
quote_fields = ["customer_ID","shopping_pt","record_type","day","time","state","location","group_size","homeowner","car_age","car_value","risk_factor","age_oldest","age_youngest","married_couple","C_previous","duration_previous","A","B","C","D","E","F","G","cost"]
count = 0 
for r in rows:
    count += 1
    if count % 10000 == 0:
        print "quotes commited: " + str(count)
        session.commit()

    r = [None if x == "NA" else x for x in r]

    session.add(Quote(**dict([(f, r[idx[f]]) for f in quote_fields])))

print "quotes commited: " + str(count)
session.commit()







# from collections import Counter
# print max(Counter([dict(x)['customer_ID'] for x in customers]).values())
# print max(Counter([dict(x)['location'] for x in locations]).values())
# print max(Counter([dict(x)['state'] for x in states]).values())
# print max(Counter([dict(x)['customer_ID'] for x in customers]).values())









 
# # Create an engine that stores data in the local directory's
# # sqlalchemy_example.db file.
# engine = create_engine('sqlite:///sqlalchemy_example.db')
 
# # Create all tables in the engine. This is equivalent to "Create Table"
# # statements in raw SQL.


