import os
from neo4j import GraphDatabase
from dotenv import load_dotenv


load_dotenv()

# uri = os.environ.get('NEO4J_URI')
# username = os.environ.get('NEO4J_USERNAME')
# password = os.environ.get('NEO4J_PASSWORD')


#BD1
uri = "neo4j+s://b6b52455.databases.neo4j.io"
username = "neo4j"
password = "JF4Wddqb4wmAIG9kq-0XklUhmFWmXApPYZ2AlsmOrRo"

# #BD2s
# uri = "neo4j+s://93bfee48.databases.neo4j.io"
# username = "neo4j"
# password = "cqwaOsXzVerQkMIfacqY_pscq68KBUH7ZSlissJWv1w"


try:
    driver = GraphDatabase.driver(uri, auth=(username, password))
    print("Connection successful!")
except Exception as e:
    print("An error occurred while connecting to Neo4j:")
    print(str(e))
