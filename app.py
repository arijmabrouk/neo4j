import json
import random
import string
from excelAlgo import getMesureExcel, getNumericFilter,extract_csv_data
from flask import Flask, jsonify, make_response,request, send_file
from neo4j import Record
import bcrypt
from flask_cors import CORS
import os
from dotenv import load_dotenv
from flask_jwt_extended import JWTManager, jwt_required ,create_access_token,get_jwt_identity,get_jwt
from cloudinaryUtils import upload_tocloudinary
from db import driver
from flask_swagger_ui import get_swaggerui_blueprint
import networkx as nx
import os
from textAlgo import NMF_topicsByFolder, NMF_topicsByOne, getAllTopicsLDA, getGraphBasedSummarize_pg, getGraphBasedSummarize_tr, getTopicForOneLDA, getTreeBasedSummarize_pg,generate_text_summary
import uuid
from utils import createGraphGS, createLinksWithPATIENT, createResponseGraphInDatabase, extract_drugs_from_file, fetch_tree_graph, predict_cancer, transform_to_rdf
from rdflib import Graph, URIRef, Literal, RDF
import joblib
import numpy as np
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
from datetime import datetime,date
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from gensim import corpora, models
import PyPDF2
from nltk.tokenize import RegexpTokenizer

#watch("neo4j")
load_dotenv()

app = Flask(__name__)
CORS(app,origins="http://localhost:3000")
app.config["JWT_SECRET_KEY"] = "9cffa439-bbcf-4c62-8dde-48e54b2a6494"
jwt = JWTManager(app)

script_dir = os.path.dirname(os.path.realpath(__file__))
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

# Create the 'uploads' directory if it doesn't exist
uploads_dir = os.path.join(script_dir, 'uploads')
os.makedirs(uploads_dir, exist_ok=True)
# current_dir = os.path.dirname(__file__)

# # Load the trained model during the application startup
# model_path = os.path.join(current_dir, 'heart.py')
# model = joblib.load(model_path)






















# **************************************************** Auth + Form ***************************************************************************

#done


# **************************************************** User :Doctor ***************************************************************************


 #*************************************************************newpart


    




# def add_document(filename):
#     try:
#         with driver.session() as session:
#             result = session.write_transaction(_create_document, filename)
#             print(result)  # Add this line to print the result
#     except Exception as e:
#         print(f"Error adding document to Neo4j: {e}")


# @staticmethod
# def _create_document(tx, filename):
#     try:
#         query = (
#             "MERGE (p:Patient {patient_id: 1}) "
#             "MERGE (d:Document {filename: $filename}) "
#             "CREATE (p)-[:HAS_DOCUMENT]->(d) "
#             "RETURN p, d"
#         )
#         result = tx.run(query, filename=filename)
#         return result.single()
#     except Exception as e:
#         print(f"Error executing query: {query} with parameters: {filename}. Error: {e}")





# @app.route('/upload', methods=['POST'])
# def upload_file():
#     try:
#         file = request.files['file']
#     except KeyError:
#         return 'Incomplete request'

#     if file:
#         file_path = os.path.join(uploads_dir, file.filename)
#         file.save(file_path)

#         # Save document details and relationship in Neo4j
#         add_document(file.filename)

#         return 'File uploaded successfully, details saved in Neo4j'
#     else:
#         return 'Invalid file type'




# @app.route('/upload_documents', methods=['POST'])
# def upload_documents():
#     graph = nx.DiGraph()

#     # Check if the request contains files
#     if 'file' not in request.files:
#         return jsonify({"error": "No file part"})

#     files = request.files('file')

#     # Loop through the documents, classify them, and store in the NetworkX graph
#     for i, file in enumerate(files):
#         document_type = classify_document(file.read())

#         # Generate a unique ID for the document
#         document_id = str(uuid.uuid4())

#         # Convert the file content to a string
#         content_str = file.read().decode('utf-8')

#         # Create a node for the document
#         document_node = {'content': content_str, 'type': document_type}
#         graph.add_node(document_id, **document_node)

#         # Create edges based on document type
#         for j in range(i):
#             if document_type == graph.nodes[str(j)]['type']:
#                 graph.add_edge(document_id, str(j))

#     return jsonify({"message": "Documents uploaded successfully."})


# def serialize_node(node):
#     return {"id": node.id, "labels": list(node.labels), **node}

# def serialize_relationship(relationship):
#     return {"id": relationship.id, "type": relationship.type, **relationship}


# # Use the driver in your route
# @app.route('/graph', methods=['GET','OPTIONS'])
# def get_graph_data():
#     try:
#         # Example query
#         cypher_query = "MATCH (n:Document)-[r]->(m) RETURN n, r, m"

#         with driver.session() as session:
#             result = session.run(cypher_query)

#             # Fetch all records from the result
#             records = list(result)

#         # Convert the records to a dictionary
#         graph_data = {
#             "nodes": [serialize_node(record["n"]) for record in records],
#             "links": ["has_document"],
#         }

#         return jsonify(graph_data)
#     except Exception as e:
#         return jsonify({"error": str(e)})


    # "nom": "aarij",
    # "prenom": "mabrouk",
    # "age": "23",
    # "sexe": "female",
    # "telephone": "52408022",
    # "adresse": "adresse",




@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    role = data.get('role')

    with driver.session() as session:
        try:
            result = session.run(
                f"MATCH (u:{role.capitalize()} {{email: $email}}) RETURN u",
                email=email
            )
            user_node = result.single()

            if user_node is None:
                return jsonify({"error": f"{role} does not exist"}), 404

            stored_password_hash = user_node["u"].get("password")

            print("Entered Password:", password)
            print("Stored Password Hash:", stored_password_hash)

            if stored_password_hash and bcrypt.checkpw(password.encode('utf-8'), stored_password_hash.encode('utf-8')):
                user_id = user_node["u"].id
                payload = {
                    "email": email,
                    "user_id": user_id,
                    "role": role
                }

                access_token = create_access_token(identity=email, additional_claims=payload, expires_delta=False)

                return jsonify(
                    access_token=access_token,
                    name=user_node["u"].get("nom"),
                    prenom=user_node["u"].get("prenom"),
                    email=user_node["u"].get("email"),
                    adresse=user_node["u"].get("adresse"),
                    sexe=user_node["u"].get("sexe"),
                    age=user_node["u"].get("age"),
                    telephone=user_node["u"].get("telephone")
                ), 200
            else:
                print("Password Verification Failed")
                return jsonify({"error": "Invalid password"}), 401
        except Exception as e:
            return jsonify({"error": str(e)}), 500


@app.route('/add_patient', methods=['POST'])
# @jwt_required()
def add_patient():
    data = request.get_json()
    patient_email = data.get('email')

    # Check if a patient with the same email already exists
    with driver.session() as session:
        existing_patient = session.run(
            "MATCH (p:Patient {email: $email}) RETURN p",
            email=patient_email
        ).single()

        if existing_patient:
            return jsonify({"error": "Patient with the same email already exists"}), 400

        # If no existing patient with the same email, proceed with creating a new patient
        nom = data.get("nom")
        prenom = data.get("prenom")
        age = data.get("age")
        sexe = data.get("sexe")
        telephone = data.get("telephone")
        adresse = data.get("adresse")
        patient_password = bcrypt.hashpw(data.get('password').encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        try:
            result = session.run(
               "CREATE (p:Patient {email: $email, password: $password, nom: $nom, prenom: $prenom, age: $age, sexe: $sexe, telephone: $telephone, adresse: $adresse})"
    "RETURN p",
    email=patient_email,
    password=patient_password,
    nom=nom,
    prenom=prenom,
    age=age,
    sexe=sexe,
    telephone=telephone,
    adresse=adresse
            )

            patient_node = result.single()
            patient_id = patient_node["p"].id
            

            return jsonify({"message": "Patient added successfully", "patient_id": patient_id}), 200
        except Exception as e:
            print(e)
            return jsonify({"error": str(e)}), 500
        

# New route to add a document for a patient
@app.route('/add_document', methods=['POST'])
@jwt_required()
def add_document():
    current_user_email = get_jwt_identity()

    if not current_user_email:
        return jsonify({"error": "Unauthorized"}), 401

    # Get patient email, document name, and uploaded file
    patient_email = request.form.get('email')
    document_file = request.files['document_file']
    document_name = request.form.get('document_name')
    creation_date = date.today().isoformat()  # Convert to ISO format

    # Check if the file is allowed
    if document_file:
        # Save the file to the uploads folder
        filepath = os.path.join(uploads_dir, document_file.filename)
        document_file.save(filepath)
        

        with driver.session() as session:
            try:
                # Check if the Filtrate node exists, create it if not
                session.run(
                    "MATCH (p:Patient {email: $email}) "
                    "CREATE (d:Document {name: $document_name, filepath: $filepath, creation_date: $creation_date})-[:HAS]->(p) "
                    "RETURN d",
                    email=patient_email,
                    document_name=document_name,
                    filepath=filepath,
                    creation_date=creation_date
                )

                return jsonify({"message": "Document added successfully"}), 201
            except Exception as e:
                return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Invalid file format"}), 400




import base64

@app.route('/graph', methods=['GET'])
def get_graph():
    patient_email = request.args.get('patient_email')

    if not patient_email:
        return jsonify({"error": "Patient email is required"}), 400

    # Cypher query to retrieve the graph for a specific patient with related documents and filtrate
    query = (
        "MATCH (p:Patient {email: $patient_email})"
        # "OPTIONAL MATCH (f:Filtrate)<-[:HAS_FILTRATE]-(p)"
        "OPTIONAL MATCH (d:Document)-[:HAS]->(p)"
        "RETURN id(p) as patient_id, p as patient_properties, "
        # "id(f) as filtrate_id, f as filtrate_properties, "
        "id(d) as document_id, d as document_properties"
    )

    with driver.session() as session:
        result = session.run(query, patient_email=patient_email).data()

    nodes = []
    edges = []

    for record in result:
        patient_id = record.get('patient_id')
        patient_properties = record.get('patient_properties')
        document_id = record.get('document_id')
        document_properties = record.get('document_properties')

   

        if patient_id is not None and patient_properties is not None:
            for key, value in patient_properties.items():
                if isinstance(value, bytes):
                    patient_properties[key] = value.decode('utf-8')

            nodes.append({'id': patient_id, 'label': f"Patient: {patient_properties.get('nom', 'Unknown')}", 'group': 'patient', 'info': patient_properties})

       

        if document_id is not None and document_properties is not None:
            for key, value in document_properties.items():
                if isinstance(value, bytes):
                    if document_properties[key]!="creation_date":
                        document_properties[key] = base64.b64encode(value).decode('utf-8')

            nodes.append({'id': document_id, 'label': f"Document: {document_properties.get('name', 'Unknown')}", 'group': 'document', 'info': document_properties})
            edges.append({'from': patient_id, 'to': document_id, 'label': 'HAS'})



    return jsonify({'nodes': nodes, 'edges': edges})


def classify_document_type(document_name):
    # Implement your logic to classify the document type based on the document name
    # For example, you might check the file extension or other criteria
    if document_name.endswith('.pdf'):
        return 'PDF'
    elif document_name.endswith('.jpg') or document_name.endswith('.png'):
        return 'Image'
    elif document_name.endswith('.csv'):
        return 'CSV'
    else:
        return 'Other'


@app.route('/document_types_graph', methods=['GET'])
def get_classified_documents():
    patient_email = request.args.get('patient_email')
    document_types_string = request.args.get('document_types', '')  # Get document types as a string

    if not patient_email:
        return jsonify({"error": "Patient email is required"}), 400

    document_types = [type.strip() for type in document_types_string.split(',')] if document_types_string else []

    query = (
    "MATCH (p:Patient {email: $patient_email}) MERGE (f:Filtrate)<-[:HAS_FILTRATE]-(p) WITH p, f OPTIONAL MATCH (d:Document)-[:HAS]->(p) RETURN d, id(d) as document_id, p as patient_properties, id(f) as filtrate_id, f as filtrate_properties"
)


    with driver.session() as session:
        result = session.run(query, patient_email=patient_email).data()

    documents = []
    nodes = []
    edges = []

    for record in result:
        document_node = record.get('d')
        patient_properties = record.get('patient_properties')
        filtrate_id = record.get('filtrate_id')
        filtrate_properties = record.get('filtrate_properties')

        # Convert bytes to utf-8 if necessary
        for key, value in patient_properties.items():
            if isinstance(value, bytes):
                patient_properties[key] = value.decode('utf-8')

        if filtrate_id is not None and filtrate_properties is not None:
            for key, value in filtrate_properties.items():
                if isinstance(value, bytes):
                    filtrate_properties[key] = value.decode('utf-8')

            nodes.append({'id': filtrate_id, 'label': 'Filtrate', 'group': 'filtrate', 'info': filtrate_properties})
            edges.append({'from': "patient", 'to': filtrate_id, 'label': 'HAS_FILTRATE'})

        if document_node:
            document_name = document_node.get('name', 'Unknown')
            document_id = record['document_id']
            document_type = classify_document_type(document_name)


            if document_type in document_types or not document_types:
                documents.append({'name': document_name, 'type': document_type, 'properties': dict(document_node), 'id': document_id})

                # Add node for the document
                nodes.append({'id': document_id, 'label': f"Document: {document_name}", 'group': 'document', 'info': dict(document_node)})

                # Add edge between patient and document with type
                edges.append({'from': filtrate_id, 'to': document_id, 'label': f'HAS_{document_type}_DOCUMENT'})

    # Add node for the patient
    nodes.append({'id': 'patient', 'label': f"Patient: {patient_email}", 'group': 'patient', 'info': patient_properties})

    return jsonify({'nodes': nodes, 'edges': edges})




@app.route('/documents_by_date_range', methods=['GET'])
def get_documents_by_date_range():
    patient_email = request.args.get('patient_email')
    start_date_str = request.args.get('start_date')
    end_date_str = request.args.get('end_date')

    if not patient_email or not start_date_str or not end_date_str:
        return jsonify({"error": "Patient email, start date, and end date are required"}), 400

    try:
        # Parse the start and end dates
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400

    # Query to retrieve documents for a specific patient within the specified date range
    query = (
        "MATCH (p:Patient {email: $patient_email}) MERGE (s:display)<-[:HAS_DISPLAY]-(p) WITH p, s MATCH (d:Document)-[:HAS]->(p) WHERE date(d.creation_date) >= $start_date AND date(d.creation_date) <= $end_date RETURN p, d, id(d) as document_id, id(p) as patient_id, id(s) as display_id, s as display_properties"
    )

    with driver.session() as session:
        result = session.run(query, patient_email=patient_email, start_date=start_date, end_date=end_date).data()

    nodes = []
    edges = []

    for record in result:
        patient_node = record.get('p')
        document_node = record.get('d')
        display_id = record.get('display_id')
        display_properties = record.get('display_properties')

        if patient_node:
            patient_id = record['patient_id']
            patient_properties = dict(patient_node)
            nodes.append({'id': patient_id, 'label': f"Patient: {patient_properties.get('nom', 'Unknown')}", 'group': 'patient', 'info': patient_properties})

        if document_node:
            document_name = document_node.get('name', 'Unknown')
            document_id = record['document_id']
            patient_id = record['patient_id']
            document_type = classify_document_type(document_name)

        if display_id is not None and display_properties is not None:
            for key, value in display_properties.items():
                if isinstance(value, bytes):
                    display_properties[key] = value.decode('utf-8')

            nodes.append({'id': display_id, 'label': 'Display', 'group': 'display', 'info': display_properties})
            edges.append({'from': patient_id, 'to': display_id, 'label': 'HAS_DISPLAY'})

            # Add node for the document
            nodes.append({'id': document_id, 'label': f"Document: {document_name}", 'group': 'document', 'info': dict(document_node)})

            # Add edge between patient and document with type
            edges.append({'from': display_id, 'to': document_id, 'label': f'HAS_{document_type}_DOCUMENT'})

    return jsonify({'nodes': nodes, 'edges': edges})


def extract_text_from_pdf(file_path):
    with open(file_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove special characters, numbers, and punctuation
    text = re.sub(r'[^a-z\s]', '', text)

    # Tokenize the text
    tokens = text.split()

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    # Join the tokens back into a single string
    processed_text = ' '.join(tokens)

    return processed_text




# @app.route('/classify_pdf', methods=['POST'])
# def classify_pdf():
#     patient_email = request.args.get('patient_email')

#     if not patient_email:
#         return jsonify({"error": "Patient email is required"}), 400

#     # Retrieve all documents associated with the patient
#     query = (
#         "MATCH (p:Patient {email: $patient_email})"
#         "MATCH (d:Document)-[:HAS]->(p)"
#         "RETURN id(p) as patient_id, p as patient_properties, "
#         "id(d) as document_id, d as document_properties"
#     )

#     with driver.session() as session:
#         result = session.run(query, patient_email=patient_email).data()

#     # Extract text from each PDF document and preprocess
#     texts = []
#     for doc in result:
#         # Check if 'filepath' key exists in document properties
#         if 'filepath' in doc['document_properties']:
#             pdf_filepath = doc['document_properties']['filepath']

#             # Check if the document is a PDF
#             if pdf_filepath.lower().endswith('.pdf'):
#                 # Print or log the PDF file path for testing purposes
#                 print(f"Processing PDF: {pdf_filepath}")

#                 # Extract text and preprocess
#                 text = extract_text_from_pdf(pdf_filepath)
#                 # Tokenize the text using RegexpTokenizer
#                 tokenizer = RegexpTokenizer(r'\w+')
#                 tokens = tokenizer.tokenize(text)
#                 texts.append(tokens)
#             else:
#                 # Skip non-PDF documents
#                 print(f"Skipping non-PDF document: {pdf_filepath}")
#         else:
#             # Handle the case where 'filepath' key is not present
#             texts.append([])  # or any other appropriate action

#     # Train LDA model
#     dictionary = corpora.Dictionary(texts)
#     corpus = [dictionary.doc2bow(token_list) for token_list in texts]
#     lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)

#     # Assign topics to documents
#     topics = lda_model[corpus]

#     # Add topic information to the result
#     for i, doc_topics in enumerate(topics):
#         result[i]['document_properties']['topics'] = max(doc_topics, key=lambda x: x[1])

#     return jsonify(result)













# def get_classified_documents():
#     patient_email = request.args.get('patient_email')

#     if not patient_email:
#         return jsonify({"error": "Patient email is required"}), 400

#     query = (
#         "MATCH (p:Patient {email: $patient_email})"
#         "OPTIONAL MATCH (d:Document)-[:HAS]->(p)"
#         "RETURN d, id(d) as document_id"
#     )

#     with driver.session() as session:
#         result = session.run(query, patient_email=patient_email).data()

#     documents = []

#     for record in result:
#         document_node = record.get('d')
#         if document_node:
#             document_name = document_node.get('name', 'Unknown')  # Replace 'Unknown' with a default value
#             document_id = record['document_id']  # Retrieve document_id directly from the record

#             # Classify document type
#             document_type = classify_document_type(document_name)

#             documents.append({'name': document_name, 'type': document_type, 'properties': dict(document_node), 'id': document_id})

#     # Create nodes for each document with its type and type_document
#     nodes = []
#     edges = []  # Don't forget to initialize edges list
#     for document in documents:
#         document_name = document['name']
#         document_type = document['type']
#         document_id = document['id']

#         # Add node for the document
#         nodes.append({'id': document_id, 'label': f"Document: {document_name}", 'group': 'document', 'info': dict(document['properties'])})

#         # Add node for the document type
#         document_type_id = f"{document_type}_type"
#         nodes.append({'id': document_type_id, 'label': f"Type: {document_type}", 'group': 'document_type', 'info': {'name': document_type}})

#         # Add edge between document and document type
#         edges.append({'from': document_id, 'to': document_type_id, 'label': 'HAS_TYPE'})


#     return jsonify({'nodes': nodes, 'edges': edges})



        
        

    




@app.route("/users/update", methods=["PUT"])
@jwt_required()
def update_user():
    user_id = get_jwt().get("user_id")
    role = get_jwt().get("role")
    data = request.get_json()

    with driver.session() as session:
        if role == "PATIENT":
            result = session.run(
                """
                MATCH (p:Patient)
                WHERE ID(p) = $user_id
                SET p.nom = $nom, p.prenom = $prenom, p.sexe = $sexe, p.telephone = $telephone, p.adresse = $adresse, p.age = $age, p.email = $email
                RETURN p
                """,
                user_id=user_id,
                nom=data.get("nom"),
                prenom=data.get("prenom"),
                sexe=data.get("sexe"),
                telephone=data.get("telephone"),
                adresse=data.get("adresse"),
                age=data.get("age"),
                email=data.get("email")
            )
        else:
            result = session.run(
                """
                MATCH (u:User)
                WHERE ID(u) = $user_id
                SET u.nom = $nom, u.prenom = $prenom, u.sexe = $sexe, u.telephone = $telephone, u.adresse = $adresse, u.specialite = $specialite
                RETURN u
                """,
                user_id=user_id,
                nom=data.get("nom"),
                prenom=data.get("prenom"),
                sexe=data.get("sexe"),
                telephone=data.get("telephone"),
                adresse=data.get("adresse"),
                specialite=data.get("specialite")
            )

        updated_user = result.single()[0]
        serialized_user = serialize_node(updated_user)
        return jsonify(serialized_user), 200
    





    






     




#add doctor
@app.route("/users", methods=["POST"])
def add_user():
    data = request.get_json()  # Get the JSON data from the request body

    email = data.get("email")
    password = data.get("password")
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    role = "DOCTOR"

    with driver.session() as session:
        try:
            # Check if the user with the given email already exists
            result = session.run(
                "MATCH (u:User {email: $email}) RETURN u",
                email=email
            )

            if result.single() is not None:
                return jsonify({"error": "Email already exists"}), 400

            # Create a new user with parameterized queries
            result = session.run(
                "CREATE (u:User {email: $email, password: $hashed_password, role: $role}) RETURN u",
                email=email, hashed_password=hashed_password.decode('utf-8'), role=role
            )

            created_user = result.single()

            if created_user is not None:
                user_properties = created_user["u"]._properties
                return jsonify(user_properties), 201  # 201 Created
            else:
                return jsonify({"error": "Failed to create user"}), 500
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        


def convert_node_to_dict(node):
    return {
        'id': node.id,
        'labels': list(node.labels),
        'properties': dict(node),
    }

@app.route('/get_node_info/<int:node_id>', methods=['GET'])
def get_node_info(node_id):
    with driver.session() as session:
        try:
            result = session.run(
                "MATCH (n) WHERE ID(n) = $node_id RETURN n",
                node_id=node_id
            )

            # Use the .single() method to retrieve a single record
            record = result.single()

            if record:
                node_info = convert_node_to_dict(record['n'])
                return jsonify(node_info)
            else:
                return jsonify({"error": "Node not found"}), 404
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        





@app.route("/users/<int:user_id>", methods=["GET"])
@jwt_required()
def get_doc_byid(user_id):
    with driver.session() as session:
        result = session.run(
            "MATCH (u:User) WHERE ID(u) = $user_id RETURN u",
            user_id=user_id
        )
        user = result.single()["u"]
        serialized_user = serialize_node(user)
        return jsonify(serialized_user)
    



    



#*******************************Patients Api**********  *******************************
@app.route("/patients/<patient_id>", methods=["GET"])
def get_patient_by_id(patient_id):
    with driver.session() as session:
        result = session.run("MATCH (p:Patient) WHERE ID(p) = $patient_id RETURN p", patient_id=int(patient_id))
        record = result.single()
        if record is None:
            return jsonify({"error": "Patient not found"})
        patient = serialize_node(record["p"])
        return jsonify(patient)
    
    

        
@app.route("/patients", methods=["GET"])
def get_all_patients():
    with driver.session() as session:
        result = session.run("MATCH (p:Patient) RETURN p")
        patients = [serialize_node(record["p"]) for record in result]
        return jsonify(patients)
    






    
@app.route("/doctor/patients", methods=["GET"])
@jwt_required()
def get_patients_by_doctor():
    doc_id = get_jwt().get("user_id")
    
    with driver.session() as session:
        result = session.run(
            "MATCH (p:Patient {doc_id: $doc_id}) RETURN p",
            doc_id=doc_id
        )
        patients = [serialize_node(record["p"]) for record in result]
        return jsonify(patients)

@app.route("/doctor/free/patients", methods=["GET"])
@jwt_required()
def get_patients_withno_doctor():
    
    
    with driver.session() as session:
        result = session.run(
            "MATCH (p:Patient {doc_id: $doc_id}) RETURN p",
            doc_id=''
        )
        patients = [serialize_node(record["p"]) for record in result]
        return jsonify(patients)


# def serialize_node(node):
#     serialized_node = dict(node)
#     serialized_node["id"] = node.id
#     for key, value in serialized_node.items():
#         if isinstance(value, bytes):
#             serialized_node[key] = value.decode("utf-8")
#     return serialized_node


# @app.route("/patients", methods=["POST"])
# @jwt_required()
# def add_patient():
#     data = request.get_json()  # Get the JSON data from the request body
#     nom = data.get("nom")
#     prenom = data.get("prenom")
#     age = data.get("age")
#     sexe = data.get("sexe")
#     telephone = data.get("telephone")
#     adresse = data.get("adresse")
#     email = telephone
#     password = bcrypt.hashpw(str(telephone).encode('utf-8'), bcrypt.gensalt())
#     with driver.session() as session:
#         try:
#             # Get the 'doc_id' from the JWT payload
#             doc_id = get_jwt().get("user_id")

#             # Execute the Neo4j query to create a patient node
#             result = session.run(
#                 "CREATE (p:Patient {doc_id: $doc_id, nom: $nom, prenom: $prenom, age: $age, sexe: $sexe, telephone: $telephone, adresse: $adresse, email: $email, password: $password}) RETURN p",
#                 doc_id=doc_id, nom=nom, prenom=prenom, age=age, sexe=sexe, telephone=telephone, adresse=adresse, email=email, password=password.decode('utf-8')
#             )

#             created_patient = result.single()["p"]
#             patient_properties = record_to_dict(created_patient)

#             # Create a new GS node with the 'patient' property set to the 'patient_id'
#             gs_result = session.run(
#                 "CREATE (gs:GS {patient: $patient_id}) RETURN gs",
#                 patient_id=str(created_patient.id)
#             )

#             created_gs = gs_result.single()["gs"]
#             gs_properties = record_to_dict(created_gs)

#             # Merge the patient_properties and gs_properties dictionaries
#             # patient_properties.update(gs_properties)

#             return jsonify(patient_properties), 201
#         except Exception as e:
#             return jsonify({"error": str(e)}), 500
@app.route("/patients/<identity>", methods=["DELETE"])
@jwt_required()
def delete_patient(identity):
    with driver.session() as session:
        try:
            # Execute the Neo4j query to delete the patient node by identity
            result = session.run(
                "MATCH (p:Patient) WHERE ID(p) = $identity DELETE p",
                identity=int(identity)
            )

            if result.consume().counters.nodes_deleted == 1:
                return jsonify({"message": "Patient deleted successfully"}), 200
            else:
                return jsonify({"error": "Patient not found"}), 404
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route("/patients/<int:patient_id>/password", methods=["PUT"])
def update_patient_password(patient_id):
    try:
        password = request.json.get("password")
        if not password:
            return jsonify({"error": "Password is required"}), 400

        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        with driver.session() as session:
            result = session.run(
                "MATCH (p:Patient) WHERE ID(p) = $patient_id SET p.password = $hashed_password RETURN p",
                patient_id=patient_id,
                hashed_password=hashed_password.decode('utf-8')
            )
            if result.single():
                return jsonify({"message": "Password updated successfully"}), 200
            else:
                return jsonify({"error": "Patient not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500



#**************************************************************** Graph***************************************************************************
# @app.route("/get_graph/<int:patient_id>")
# def graphByPatient(patient_id):
#     graph = nx.DiGraph()

#     with driver.session() as session:
#         try:
#             result = session.run(
#                 "MATCH path = (p:Patient)-[*]-(n) WHERE ID(p) = toInteger($patient_id) RETURN path",
#                 patient_id=patient_id
#             )

#             for record in result:
#                 path = record["path"]
#                 nodes = path.nodes
#                 relationships = path.relationships

#                 for node in nodes:
#                     # Convert byte properties to strings
#                     properties = {key: value.decode() if isinstance(value, bytes) else value for key, value in dict(node).items()}

#                     # Add nodes to the graph
#                     graph.add_node(node.id, labels=list(node.labels), properties=properties)

#                 for relationship in relationships:
#                     # Convert byte properties to strings
#                     properties = {key: value.decode() if isinstance(value, bytes) else value for key, value in dict(relationship).items()}

#                     # Add relationships to the graph
#                     start_node = relationship.start_node
#                     end_node = relationship.end_node
#                     graph.add_edge(start_node.id, end_node.id, type=relationship.type, properties=properties)

#             # Convert graph to JSON
#             json_data = {
#                 "nodes": [
#                     {
#                         "id": str(node_id),
#                         "labels": data["labels"],
#                         "properties": data["properties"]
#                     }
#                     for node_id, data in graph.nodes(data=True)
#                 ],
#                 "links": [
#                     {
#                         "source": str(start),
#                         "target": str(end),
#                         "type": data["type"],
#                         "properties": data["properties"]
#                     }
#                     for start, end, data in graph.edges(data=True)
#                 ]
#             }

#             return jsonify(json_data), 200
#         except Exception as e:
#             return jsonify({"error": str(e)}), 500
@app.route("/get_graph/<int:patient_id>")
def graphByPatient(patient_id):
    graph = nx.DiGraph()

    with driver.session() as session:
        try:
            result = session.run(
                "MATCH path = (p:Patient)-[*]-(n) WHERE ID(p) = toInteger($patient_id) RETURN path",
                patient_id=patient_id
            )

            found_path = False

            for record in result:
                path = record["path"]

                if path is not None:
                    found_path = True
                    nodes = path.nodes
                    relationships = path.relationships

                    for node in nodes:
                        # Convert byte properties to strings
                        properties = {key: value.decode() if isinstance(value, bytes) else value for key, value in dict(node).items()}

                        # Add nodes to the graph
                        graph.add_node(node.id, labels=list(node.labels), properties=properties)

                    for relationship in relationships:
                        # Convert byte properties to strings
                        properties = {key: value.decode() if isinstance(value, bytes) else value for key, value in dict(relationship).items()}

                        # Add relationships to the graph
                        start_node = relationship.start_node
                        end_node = relationship.end_node
                        graph.add_edge(start_node.id, end_node.id, type=relationship.type, properties=properties)

            if not found_path:
                # If no relationships found, add only the patient node to the graph
                result = session.run(
                    "MATCH (p:Patient) WHERE ID(p) = toInteger($patient_id) RETURN p",
                    patient_id=patient_id
                )
                node = result.single()["p"]
                properties = {key: value.decode() if isinstance(value, bytes) else value for key, value in dict(node).items()}
                graph.add_node(node.id, labels=list(node.labels), properties=properties)

            # Convert graph to JSON
            json_data = {
                "nodes": [
                    {
                        "id": str(node_id),
                        "labels": data["labels"],
                        "properties": data["properties"]
                    }
                    for node_id, data in graph.nodes(data=True)
                ],
                "links": [
                    {
                        "source": str(start),
                        "target": str(end),
                        "type": data["type"],
                        "properties": data["properties"]
                    }
                    for start, end, data in graph.edges(data=True)
                ]
            }

            return jsonify(json_data), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500


@jwt_required()
@app.route("/patients/doctor/<id>", methods=["DELETE"])
def removedoctorFromPatient(id):
    with driver.session() as session:
        try:
            # Check if the patient exists
            result = session.run(
                "MATCH (p:Patient) WHERE ID(p) = $patient_id RETURN p",
                patient_id=int(id)
            )
            if not result.single():
                return jsonify({"error": "Patient not found"}), 404

            # Execute the Neo4j query to remove the doctor from the patient
            result = session.run(
                "MATCH (p:Patient) WHERE ID(p) = $patient_id SET p.doc_id = '' ",
                patient_id=int(id)
            )

            return jsonify({"message": "Doctor removed from patient successfully"}), 200
        except Exception as ce:
            return jsonify({"error": str(ce)}), 400



@app.route("/patients/doctor/<id>", methods=["PUT"])
@jwt_required()
def adddoctorFromPatient(id):
    doc_id = get_jwt().get("user_id")
    with driver.session() as session:
        
        try:
                # Check if the patient exists
            result = session.run(
                    "MATCH (p:Patient) WHERE ID(p) = $patient_id RETURN p",
                    patient_id=int(id)
                )
            patient = result.single()
            if not patient:
                return jsonify({"error": "Patient not found"}), 404

                # Execute the Neo4j query to update the doc_id of the patient
            result = session.run(
                    "MATCH (p:Patient) WHERE ID(p) = $patient_id "
                    "SET p.doc_id = $doc_id",
                    patient_id=int(id),
                    doc_id=int(doc_id)
                )

            return jsonify({"message": "Doctor added to patient successfully"}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route("/nodesNames", methods=["GET"])
@jwt_required()
def get_node_names():
    with driver.session() as session:
        result = session.run("MATCH (list:NodeList) RETURN list.nodeNames")
        record = result.single()
        if record:
            node_names = record["list.nodeNames"]
            print(node_names)
            if node_names:
                if isinstance(node_names, str):
                    node_names_list = node_names.split(",")
                else:
                    node_names_list = node_names
                return jsonify(node_names_list)
        return jsonify([])
    

@app.route("/nodesNames", methods=["POST"])
@jwt_required()
def create_node_name():
    data = request.get_json()
    if data and "nodeNames" in data:
        node_names = data["nodeNames"]
        with driver.session() as session:
            result = session.run("MATCH (list:NodeList) RETURN list.nodeNames")
            record = result.single()
            if record:
                existing_node_names = record["list.nodeNames"].split(",")
                node_names_list = node_names.split(",")  # Convert node_names string to a list
                for node_name in node_names_list:
                    if node_name in existing_node_names:
                        return jsonify({"error": f"Node name '{node_name}' already exists"}), 400
                updated_node_names = list(set(existing_node_names + node_names_list))
                session.run("MERGE (list:NodeList) SET list.nodeNames = $node_names",
                            node_names=updated_node_names)
            else:
                session.run("CREATE (list:NodeList) SET list.nodeNames = $node_names",
                            node_names=node_names)
        return jsonify({"message": "Node names added successfully"})
    else:
        return jsonify({"error": "Invalid request data"}), 400


@app.route("/relationshipNames", methods=["GET"])
@jwt_required()
def get_relationship_names():
    with driver.session() as session:
        result = session.run("MATCH (list:RelationshipList) RETURN list.relationshipNames")
        record = result.single()
        if record:
            relationship_names = record["list.relationshipNames"]
            if isinstance(relationship_names, str):
                relationship_names_list = relationship_names.split(",")
            elif isinstance(relationship_names, list):
                relationship_names_list = relationship_names
            else:
                return jsonify([])  # Invalid data type for relationship names
            
            return jsonify(relationship_names_list)
        return jsonify([])

# Rest of the code remains the same


@app.route("/relationshipNames", methods=["POST"])
@jwt_required()
def create_relationship_name():
    data = request.get_json()
    if data and "relationshipNames" in data:
        relationship_names = data["relationshipNames"]
        with driver.session() as session:
            result = session.run("MATCH (list:RelationshipList) RETURN list.relationshipNames")
            record = result.single()
            if record:
                existing_relationship_names = record["list.relationshipNames"]
                if isinstance(existing_relationship_names, str):
                    existing_relationship_names = existing_relationship_names.split(",")
                elif not isinstance(existing_relationship_names, list):
                    existing_relationship_names = []
                    
                relationship_names_list = relationship_names.split(",")  # Convert relationship_names string to a list
                for relationship_name in relationship_names_list:
                    if relationship_name in existing_relationship_names:
                        return jsonify({"error": f"Relationship name '{relationship_name}' already exists"}), 400   
                updated_relationship_names = list(set(existing_relationship_names + relationship_names_list))
                session.run("MERGE (list:RelationshipList) SET list.relationshipNames = $relationship_names",
                            relationship_names=updated_relationship_names)
            else:
                session.run("CREATE (list:RelationshipList) SET list.relationshipNames = $relationship_names",
                            relationship_names=relationship_names)
        return jsonify({"message": "Relationship names added successfully"})
    else:
        return jsonify({"error": "Invalid request data"}), 400


#************************************* Node Creations *******************************************************************

@app.route("/create_node", methods=["POST"])
def create_node():
    # Extract request parameters
    file = request.files.get("file")
    file_type = request.form.get("type")
    value = request.form.get("value")
    patient_id = request.form.get("patient_id")
    date = request.form.get("date") 
    file_name = request.form.get("fileName")
    nodeName = request.form.get("nodeName").replace(" ", "")
    source_id = request.form.get("source")
    relationship_name = request.form.get("relationshipName").replace(" ", "")
    source_nodeName = request.form.get("source_nodeName").replace(" ", "")
    url = None
    print("file:", file)
    print("file_type:", file_type)
    print("value:", value)
    print("patient_id:", patient_id)
    print("date:", date)
    print("file_name:", file_name)
    print("nodeName:", nodeName)
    print("source_id:", source_id)
    print("relationship_name:", relationship_name)
    print("source_nodeName:", source_nodeName)
    print("url:", url)
    # Check if file type is "numeric" or "string"
    if file_type.lower() == "numeric" or file_type.lower() == "string":
        # Update the value directly from the request
        pass
    else:
        # Save file to the designated folder
        folder_path = f"files/{patient_id}"
        os.makedirs(folder_path, exist_ok=True)

        # Check if file_name is not empty
        if file_name:
            file_path = os.path.join(folder_path, file_name)
            file.save(file_path)
            
            url = upload_tocloudinary(file_path)
            print(url)
            # Update the value with the file path
            value = file_path

    with driver.session() as session:
        query = f"""
            MATCH (source:{source_nodeName})
            WHERE id(source) = $source_id
            CREATE (newNode:{nodeName} $props)
            CREATE (source)-[:{relationship_name}]->(newNode)
            RETURN id(newNode) as nodeId
        """
        result = session.run(query, source_nodeName=source_nodeName, nodeName=nodeName, props={
            "type": file_type,
            "value": value,
            "patient_id": patient_id,
            "date": date,
            "file": file_name,
            'url':url
        }, source_id=int(source_id))
        node_id = result.single()["nodeId"]


    return jsonify({"node_id": node_id}), 200
#************************************************************** SYNTHESE***********************************************************************************************************8



#************************************************************** Confirmed 👌#***********************************************************************************************************8

#Confirmed 👌
@app.route("/getdrugs", methods=["GET"])
def get_drugs():
    patient_id = str(request.args.get("patient_id"))  # Generate a random UUID for the patient
    disease = request.args.get("disease")
    query_key = request.args.get("query_key")
    is_summary = request.args.get("is_summary")

    with driver.session() as session:
        result = session.run(
            "MATCH (p:prescription {patient_id: $patient_id}) RETURN p",
            patient_id=patient_id
        )

        prescription_nodes = [serialize_node(record["p"]) for record in result]

        graph_data = {
            "nodes": [],
            "links": []
        }

        # Generate a random UUID for the PATIENT node
        patient_uuid = str(uuid.uuid4())

        # Add Patient node
        patient_node = {
            "id": f"{patient_uuid}",
            "labels": ["Patient"],
            "properties": {"patient_id": patient_id, "id": f"{patient_uuid}"}
        }
        graph_data["nodes"].append(patient_node)

        for prescription_node in prescription_nodes:
            drugs = extract_drugs_from_file(
                prescription_node["value"], disease)

            # Generate a random UUID for the prescription
            prescription_id = str(uuid.uuid4())

            # Create Prescription node
            prescription_node_data = {
                "id": f"{prescription_id}",
                "labels": ["prescription"],
                "properties": {**dict(prescription_node),
                               "id": f"{prescription_id}"}
            }
            graph_data["nodes"].append(prescription_node_data)

            # Create links between Patient and Prescription nodes
            patient_to_prescription_link = {
                "source": f"{patient_uuid}",
                "target": f"{prescription_id}",
                "type": "hasPrescription",
                "properties": {}
            }
            graph_data["links"].append(patient_to_prescription_link)

            for drug in drugs:
                drug_key = ''.join(random.choices(
                    string.ascii_lowercase + string.digits, k=4))
                drug_node_data = {
                    "id": f"{drug_key}",
                    "labels": ["DRUG"],
                    "properties": {"name": drug, "id": f"{drug_key}"}
                }
                graph_data["nodes"].append(drug_node_data)

                # Create links between Prescription and Drug nodes
                prescription_to_drug_link = {
                    "source": f"{prescription_id}",
                    "target": f"{drug_key}",
                    "type": "containDrugs",
                    "properties": {}
                }
                graph_data["links"].append(prescription_to_drug_link)

        if not graph_data["nodes"] or not graph_data["links"]:
            print("No nodes or links to process.")
            return jsonify(graph_data), 200

        if is_summary =='is_summary':
            return jsonify(graph_data), 200
          
        query_uuid = createGraphGS(session, patient_id, 
                                   "Extract Drugs From prescriptions", query_key, "Extracting")
        Drugs_id = str(uuid.uuid4())
        Drugs_node= {
            
             "id":Drugs_id,
                "labels": ["DrugsResults"],
                "properties": {
                    
                   
                    "id":Drugs_id,
                   "results":f"{request.full_path}&is_summary=is_summary"

                    
                }
        }
        
        patient_to_query_link = {
            "source": f"{query_uuid}",
            "target": f"{Drugs_id}",
            "type": "hasResultSummary",
            "properties": {}
        }
        #graph_data["links"].append(patient_to_query_link)

        graph_data_response = createResponseGraphInDatabase(
            session, [Drugs_node], [patient_to_query_link])

        return jsonify(graph_data), 200

#Confirmed 
@app.route("/Excell/summarize", methods=["GET"])
def summarize_excel():
    file_path = request.args.get('file_path')
    patient_id = request.args.get('patient_id')
    query_key = request.args.get('query_key')

    # Default value is 'graphic' if not provided
    aggregate_fn = request.args.get('aggregateFN', 'graphical')

    if not file_path:
        return jsonify({'error': 'No file path provided'})

    try:
        res,uud = getMesureExcel(file_path, aggregate_fn)
        with driver.session() as session:
            if (aggregate_fn=='graphical'):
                text= "Generate graphical summary"
            else :
                text= f"Summarize the {aggregate_fn} value"
                
            query_uuid = createGraphGS(session, patient_id, text, query_key, "Transformation")
            patient_to_query_link = {
            "source": f"{query_uuid}",
            "target": uud,
            "type": "hasResultSummary",
            "properties": {}
        }
            res["links"].append(patient_to_query_link)

            graph_data_response = createResponseGraphInDatabase(
            session, res["nodes"], res["links"])
            return res
    except FileNotFoundError:
        return jsonify({'error': 'File not found'})

#Confirmed 
@app.route("/topic/one", methods=["GET"])
def gettopic_byone():
    file_path = request.args.get("file_path")
    patient_id = request.args.get("patient_id")
    url = request.args.get("url")
    algo = request.args.get("algo")
    relationshipName = f"hasResultsWith{algo}"
    query_key = request.args.get("query_key")
    
    with driver.session() as session:
        # Find the patient node with the given patient_id
        result = session.run(
            "MATCH (p:Patient) WHERE ID(p) = $patient_id RETURN p",
            patient_id=int(patient_id)  # Assuming patient_id is an integer
        )

        patient_node = result.single()
        if patient_node is None:
            return jsonify({"error": "Patient not found."}), 404

        patient_properties = patient_node["p"]

        if algo == "LDA":
            try:
                # Replace with actual function to get LDA results
                results = getTopicForOneLDA(file_path)
            except FileNotFoundError:
                return jsonify({"error": "File not found."}), 404
        elif algo == "NMF":
            try:
                # Replace with actual function to get NMF results
                results = NMF_topicsByOne(file_path, 5)
            except FileNotFoundError:
                return jsonify({"error": "File not found."}), 404
        else:
            return jsonify({"error": "Algorithm not supported."}), 400

        if results is None:
            return jsonify({"error": "Data not found."}), 404

        document_id = str(uuid.uuid4())  # Generate a random UUID for the document node
        results_id = str(uuid.uuid4()) 
        patientuuid_id = str(uuid.uuid4()) # Generate a random UUID for the results node

        # Create a graph-like JSON response
        graph_data = {
            "nodes": [
                {
                    "id": str(patientuuid_id),
                    "labels": ["PATIENT"],
                    "properties": {"id": patientuuid_id},
                },
                {
                    "id": document_id,
                    "labels": ["Document"],
                    # Add other document properties if available
                    "properties": {"file_path": file_path, "url": url,"id":document_id}
                },
                {
                    "id": results_id,
                    "labels": ["Results"],
                    # Add other results properties if available
                    "properties": {"results": json.dumps(results),"id":results_id,"algo":algo}
                }
            ],
            "links": [
                {
                    "source": str(patientuuid_id),
                    "target": document_id,
                    "type": "hasSummary",
                    "properties": {}  # Add properties related to the relationship if available
                },
                {
                    "source": document_id,
                    "target": results_id,
                    "type": relationshipName,
                    "properties": {}  # Add properties related to the relationship if available
                }
            ]
        }

        if not graph_data["nodes"] or not graph_data["links"]:
            print("No nodes or links to process.")
            return jsonify(graph_data), 200
        
        
    
        
        query_uuid = createGraphGS(session, patient_id, "Extract Topics For a Document ", query_key, "Extracting")

        patient_to_query_link = {
            "source": f"{query_uuid}",
            "target": f"{patientuuid_id}",
            "type": "hasResultSummary",
            "properties": {}
        }
        graph_data["links"].append(patient_to_query_link)  
        graph_data_response = createResponseGraphInDatabase(
            session, graph_data["nodes"], graph_data["links"])
       
        
        return jsonify(graph_data), 200
#Confirmed 
#modified by arij
    
def serialize_node(node):
    serialized_node = dict(node)
    serialized_node["id"] = node.id
    for key, value in serialized_node.items():
        if isinstance(value, bytes):
            serialized_node[key] = value.decode("utf-8")
    return serialized_node
classes_keywords = {
    'Diabetes': ['glycemia', 'glucose', 'diabetes', 'insulin',"pancréas","hyperglycémie","HbA1c","régime alimentaire","endocrinologie","Neuropathie","type 1", "type 2","Glycémie à jeun","Glycémie"],
    'Heart Diseases':['cardiovascular','Cœur','Vaisseaux','Veines','heart',"cardiopathie","infarctus du myocarde","athérosclérose","hypertension artérielle","insuffisance cardiaque","cholestérol","ventricule droit" ]
    # Add more classes and their associated keywords
,}
def claasifyTopics(results, classes_keywords, threshold=0):
    document_classifications = {}
    
    # Flatten the results list of lists
    all_keywords = [
        entry['keyword'] if isinstance(entry, dict) else entry
        for sublist in results
        for entry in (sublist if isinstance(sublist, list) else [sublist])
    ]
    print(all_keywords)
    
    for class_name, keywords in classes_keywords.items():
        matched_keywords = set(keywords).intersection(set(all_keywords))
        percentage_matched = len(matched_keywords) / len(keywords)
        
        if percentage_matched > threshold:
            document_classifications[class_name] = percentage_matched

    return document_classifications





def create_class_node_and_relationship(session, document_id, class_name, percentage_matched):
    # Create a node for the class if it doesn't exist
    session.run(
        "MERGE (c:Class {name: $class_name})",
        class_name=class_name
    )

    # Create a relationship between the document and the class
    session.run(
        "MATCH (d:Document {id: $document_id}), (c:Class {name: $class_name}) "
        "CREATE (d)-[:BELONGS_TO {percentage: $percentage}]->(c)",
        document_id=document_id,
        class_name=class_name,
        percentage=percentage_matched
    )
def extract_word_percentage(results):
    word_percentage_list = []
    
    for sublist in results:
        for entry in (sublist if isinstance(sublist, list) else [sublist]):
            if 'keyword' in entry and 'percentage' in entry:
                word_percentage_list.append((entry['keyword'], entry['percentage']))
    
    return word_percentage_list

@app.route("/topic/multiple", methods=["POST"])
def gettopic_bymultiple():
    # Get parameters from the request
    request_data = request.json
    patient_email = request_data.get("email")
    algo = request_data.get("algo")


    # Neo4j query to retrieve patient and associated documents
    query = (
        "MATCH (p:Patient {email: $patient_email})"
        "OPTIONAL MATCH (d:Document)-[:HAS]->(p)"
        "RETURN id(p) as patient_id, p as patient_properties, "
        "id(d) as document_id, d as document_properties"
    )

    with driver.session() as session:
        result = session.run(query, patient_email=patient_email).data()

        nodes = []
        edges = []

        for record in result:
            patient_id = record.get('patient_id')
            patient_properties = record.get('patient_properties')
            document_id = record.get('document_id')
            document_properties = record.get('document_properties')
            nodes.append({'id': patient_id, 'label': f"Patient: {patient_properties.get('nom', 'Unknown')}", 'group': 'patient', 'info': patient_properties})
            


            # Filter PDF documents
            if document_properties.get("filepath", "").endswith(".pdf"):
                # Process document information
                if document_properties.get("filepath", "").endswith(".pdf"):
                    # Process document information
                    results = []
                    if document_properties is not None:
                        try:
                            if algo == "HLDA":
                                results = getTopicForOneLDA(document_properties.get("filepath"))
                                # version_entry["algorithm_version"] = "LDA"
                            elif algo == "LDA":
                                results = NMF_topicsByOne(document_properties.get("filepath"), 5)
                                # version_entry["algorithm_version"] = "NMF"
                            elif algo == 'GRAPH_BASED_PG':
                                results = getGraphBasedSummarize_pg(document_properties.get("filepath"))
                            
                            elif algo == 'TREE_BASED':
                                results = generate_text_summary(document_properties.get("filepath"))
               
                            document_classifications = claasifyTopics(results, classes_keywords)
                            # version_entry["relationships"] = []
                            # Create nodes and relationships dynamically
                            for class_name, percentage_matched in document_classifications.items():

                                if not results or not any(results):
                                    continue
                                # Create a relationship between the document and the class
                                session.run(
                                    "MERGE (c:Class {name: $class_name}) "
                                    "SET c.id = coalesce(c.id, randomUUID()) "
                                    "WITH c "
                                    "MATCH (d:Document {id: $document_id}) "
                                    "CREATE (d)-[:BELONGS_TO {percentage: $percentage}]->(c)",
                                    document_id=document_id,
                                    class_name=class_name,
                                    percentage=percentage_matched
                                )
                                # version_entry["relationships"].append({
                                # "from": {"id": document_id, "labels": ["Document"], "properties": document_properties},
                                # "to": {"id": class_name, "labels": ["Class"], "properties": {"name": class_name}},
                                # "label": "BELONGS_TO",
                                # # "properties": {"percentage": percentage_matched}
                                #     })

                                # version_entry["query_result"] = results

                                # Append nodes to the list
                                nodes.append({'id': document_id, 'label': f"Document: {document_properties.get('name', 'Unknown')}", 'group': 'document', 'info': dict(document_properties)})

                                nodes.append({'id': class_name, 'label': f"Class: {class_name}", 'group': 'class', 'info': {"name": class_name}})
                                nodes.append({'id': 'Extract', 'label': 'Extract', 'group': 'Extract', 'info': {"name": 'Extract'}})


                                # Append edges to the list
                                edges.append({
                                            'from': patient_id,
                                            'to': document_id,
                                            'label': 'has',
                                        })
                                edges.append({
                                            'from': patient_id,
                                            'to': 'Extract',
                                            'label': 'Extract',
                                        })
                                edges.append({
                                    'from': document_id,
                                    'to': class_name,
                                    'label': 'BELONGS_TO'
                                })

                                for word, word_percentage in extract_word_percentage(results):
                                    if any(word in keywords for keywords in classes_keywords.values()):
                                        # Create or find the word node
                                        session.run(
                                            "MERGE (w:Word {name: $word})",
                                            word=word
                                        )

                                        # Create a relationship between the document and the word
                                        session.run(
                                            "MATCH (d:Document {id: $document_id}), (w:Word {name: $word}) "
                                            "MERGE (d)-[:CONTAINS {percentage: $word_percentage}]->(w)",
                                            document_id=document_id,
                                            word=word,
                                            word_percentage=word_percentage
                                        )

                                        # Append nodes to the list
                                        nodes.append({'id': word, 'label': f"Word: {word}", 'group': 'word', 'info': {"name": word}})

                                        # Append edges to the list
                                        
                                        edges.append({
                                            'from': document_id,
                                            'to': word,
                                            'label': f'CONTAINS with percentage {word_percentage:.2%}',
                                        })

                        except FileNotFoundError:
                            return jsonify({"error": "File not found."}), 404

            elif document_properties.get("filepath", "").endswith(".csv"):
                # Process CSV document information
                if document_properties is not None:
                    try:
                        if algo == "CSVMax" or algo=="CSVMin" or algo=="CSVAvg":
                                                    csv_data = extract_csv_data(document_properties.get("filepath"))
                                                    if csv_data:
                                                        # Create a node for the document
                                                        session.run(
                                                            "MERGE (doc:Document {id: $document_id, name: $document_name})",
                                                            document_id=document_id,
                                                            document_name=document_properties.get('name', 'Unknown')
                                                        )
                                                        nodes.append({'id': document_id, 'label': f"Document: {document_properties.get('name', 'Unknown')}", 'group': 'document', 'info': dict(document_properties)})


                                                        # Iterate through columns
                                                        for column_name, column_values in csv_data.items():
                                                            # Create a node for the column
                                                            column_node_id = f"Column_{column_name}"
                                                            session.run(
                                                                "MERGE (column:Column {id: $column_node_id, name: $column_name})",
                                                                column_node_id=column_node_id,
                                                                column_name=column_name
                                                            )
                                                            nodes.append({'id': column_node_id, 'label': f"Column: {column_name}", 'group': 'Column', 'info': {'name': column_name}})

                                                            # Extract min, max, and average for each column
                                                            min_value = min(column_values)
                                                            max_value = max(column_values)
                                                            average_value = round(sum(column_values) / len(column_values), 2)

                                                            # Create nodes and relationships for min, max, and average of each column
                                                            # Node for Min
                                                            if algo=="CSVMin" :
                                                                    min_node_id = f"Min_{column_node_id}"
                                                                    session.run(
                                                                        "MERGE (min:Value {id: $min_node_id, type: 'Min', value: $min_value})",
                                                                        min_node_id=min_node_id,
                                                                        min_value=min_value
                                                                    )
                                                                    session.run(
                                                                        "MATCH (column:Column {id: $column_node_id}), (min:Value {id: $min_node_id, type: 'Min', value: $min_value}) "
                                                                        "MERGE (column)-[:HAS_MIN]->(min)",
                                                                        column_node_id=column_node_id,
                                                                        min_node_id=min_node_id,
                                                                        min_value=min_value
                                                                    )
                                                                    nodes.append({'id': min_node_id, 'label': f"Min {column_name}: {min_value}", 'group': 'Min', 'info': {'name': min_value, 'column': column_name}})
                                                                    edges.append({
                                                                            'from': column_node_id,
                                                                            'to': min_node_id,
                                                                            'label': f'HAS_MIN'
                                                                        })

                                                            # Node for Max
                                                            if(algo=="CSVMax"):        
                                                                    max_node_id = f"Max_{column_node_id}"
                                                                    session.run(
                                                                        "MERGE (max:Value {id: $max_node_id, type: 'Max', value: $max_value})",
                                                                        max_node_id=max_node_id,
                                                                        max_value=max_value
                                                                    )
                                                                    session.run(
                                                                        "MATCH (column:Column {id: $column_node_id}), (max:Value {id: $max_node_id, type: 'Max', value: $max_value}) "
                                                                        "MERGE (column)-[:HAS_MAX]->(max)",
                                                                        column_node_id=column_node_id,
                                                                        max_node_id=max_node_id,
                                                                        max_value=max_value
                                                                    )
                                                                    nodes.append({'id': max_node_id, 'label': f"Max {column_name}: {max_value}", 'group': 'Max', 'info': {'name': max_value, 'column': column_name}})
                                                                    edges.append({
                                                                'from': column_node_id,
                                                                'to': max_node_id,
                                                                'label': f'HAS_MAX'
                                                            })
                                                            if(algo=="CSVAvg"):
                                                                    # Node for Average
                                                                    avg_node_id = f"Average_{column_node_id}"
                                                                    session.run(
                                                                        "MERGE (average:Value {id: $avg_node_id, type: 'Average', value: $average_value})",
                                                                        avg_node_id=avg_node_id,
                                                                        average_value=average_value
                                                                    )
                                                                    session.run(
                                                                        "MATCH (column:Column {id: $column_node_id}), (average:Value {id: $avg_node_id, type: 'Average', value: $average_value}) "
                                                                        "MERGE (column)-[:HAS_AVG]->(average)",
                                                                        column_node_id=column_node_id,
                                                                        avg_node_id=avg_node_id,
                                                                        average_value=average_value
                                                                    )
                                                                    nodes.append({'id': avg_node_id, 'label': f"Average {column_name}: {average_value}", 'group': 'AVG', 'info': {'name': average_value, 'column': column_name}})
                                                                    edges.append({
                                                                'from': column_node_id,
                                                                'to': avg_node_id,
                                                                'label': f'HAS_AVG'
                                                            })

                                                            # Create edges between document, column, and min, max, average nodes
                                                            edges.append({
                                                                'from': document_id,
                                                                'to': column_node_id,
                                                                'label': f'HAS_COLUMN_{column_name}'
                                                            })
                                                            edges.append({
                                                                'from': patient_id,
                                                                'to': document_id,
                                                                'label': 'has',
                                                            })
                        




                    except FileNotFoundError:
                            return jsonify({"error": "File not found."}), 404
            
            
            
        version_entry = {
                "query_name": f"version - Algorithm: {algo}",
                "version_timestamp": datetime.now().isoformat(),
                "patient": patient_email,
                "nodes": nodes,
                "edges": edges
            }

            # Convert the version_entry dictionary to a JSON string
        version_entry_json = json.dumps(version_entry)

            # Create a new Version node for each version entry
        session.run(
                "MERGE (p:Patient {email: $patient_email})"
                "CREATE (p)-[:HAS_VERSION]->(v:Version {version: $version})",
                patient_email=patient_email,
                version=version_entry_json
            )

        # Return the nodes and edges in the specified format
        return jsonify({'nodes': nodes, 'edges': edges}), 200
    



#  version_entry = {
#             "query_name": f"gettopic_bymultiple - Algorithm: {algo}",
#             "version_timestamp": datetime.now().isoformat(),
#             "patient": patient_email,
#             "nodes": nodes,
#             "edges": edges
#         }

#         # Convert the version_entry dictionary to a JSON string
#         version_entry_json = json.dumps(version_entry)

#         session.run(
#             "MERGE (p:Patient {email: $patient_email}) "
#             "CREATE (p)-[:HAS_VERSION]->(v:Version {version: $version})",
#             patient_email=patient_email,
#             version=version_entry_json
#         )

#         # Return the nodes and edges in the specified format
#         return jsonify({'nodes': nodes, 'edges': edges}), 200

@app.route("/versions/<patient_email>", methods=["GET"])
def get_versions_by_patient_email(patient_email):
    # Neo4j query to retrieve versions for a patient by email
    query = (
        "MATCH (p:Patient {email: $patient_email})-[:HAS_VERSION]->(v:Version)"
        "RETURN id(p) as patient_id, p as patient_properties, v.version as version_data,id(v) as version_id"
    )

    with driver.session() as session:
        result = session.run(query, patient_email=patient_email).data()
        nodes = []
        edges = []

        # Parse JSON strings back to JSON objects
        for record in result:
            patient_id = record.get('patient_id')
            patient_properties = record.get('patient_properties')
            version_data = json.loads(record['version_data'])
            version_id=record.get('version_id')

            # Parse timestamp and format it
            timestamp = datetime.fromisoformat(version_data['version_timestamp'])
            formatted_timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")

            # Create Version node
            nodes.append({
                'id': version_id,
                'label': 'Version',
                'group': 'version',
                'info': {
                    'name': f"{version_data['query_name']} - {formatted_timestamp}"
                }
            })

            # Create Patient node
            nodes.append({
                'id': patient_id,
                'label': f"Patient: {patient_properties.get('nom', 'Unknown')}",
                'group': 'patient',
                'info': patient_properties
            })

            edges.append({
                'from': patient_id,
                'to': version_id,
                'label': "Has_version"
            })

        # Return the list of version data
        return jsonify({'nodes': nodes, 'edges': edges}), 200
    


@app.route('/version', methods=['GET'])
def get_version_by_timestamp():
    timestamp = request.args.get('timestamp')
    with driver.session() as session:
        result = session.run(
            "MATCH (p:Patient)-[:HAS_VERSION]->(v:Version) "
            "WITH p, v, apoc.convert.fromJsonMap(v.version) AS versionData "
            "WHERE versionData.version_timestamp = $timestamp "
            "RETURN v AS version",
            timestamp=timestamp
        )

        versions = [serialize_node(record["version"]) for record in result]

    if versions:
        return jsonify({"versions": versions}), 200
    else:
        return jsonify({"message": "Version not found for the given timestamp"}), 404





# @app.route("/openai",methods=["POST"])
# def summarize_with_ai():
#     request_data = request.json
#     patient_email = request_data.get("email")

#     # Neo4j query to retrieve patient and associated documents
#     query = (
#         "MATCH (p:Patient {email: $patient_email})"
#         "OPTIONAL MATCH (d:Document)-[:HAS]->(p)"
#         "RETURN id(p) as patient_id, p as patient_properties, "
#         "id(d) as document_id, d as document_properties"
#     )

#     with driver.session() as session:
#         result = session.run(query, patient_email=patient_email).data()

#         nodes = []
#         edges = []

#         for record in result:
#             patient_id = record.get('patient_id')
#             patient_properties = record.get('patient_properties')
#             document_id = record.get('document_id')
#             document_properties = record.get('document_properties')

#             # Filter PDF documents
#             if document_properties.get("filepath", "").endswith(".pdf"):
#                 # Process document information
#                 if document_properties.get("filepath", "").endswith(".pdf"):
#                     # Process document information
#                     results = []
#                     if document_properties is not None:
#                         try:
                            
#                             results = getSummerizeAi(document_properties.get("filepath"))
#                             print("ressssssssss",results)


# @app.route("/topic/multiple", methods=["POST"])
# def gettopic_bymultiple():
#     request_data = request.json
#     print(request_data)
#     patient_id = request_data.get("patient_id")
#     file_paths = request_data.get("file_paths", [])
#     algo = request_data.get("algo")
#     query_key = request_data.get("query_key")
    
#     with driver.session() as session:
#         # Find the patient node with the given patient_id
#         result = session.run(
#             "MATCH (p:Patient) WHERE ID(p) = $patient_id RETURN p",
#             patient_id=int(patient_id)  # Assuming patient_id is an integer
#         )

#         patient_node = result.single()
#         if patient_node is None:
#             return jsonify({"error": "Patient not found."}), 404

#         patient_properties = patient_node["p"]

#           # Generate a random UUID for the document node
#         results_id = str(uuid.uuid4()) 
#         patientuuid_id = str(uuid.uuid4()) 
    
#         graph_data = {
#             "nodes": [
#                 {
#                     "id": patientuuid_id,
#                     "labels": ["PATIENT"],
#                     "properties": {"id":patientuuid_id},
#                 }
#             ],
#             "links": []
#         }

        

#         # Create Results node
#         results_node = {
#             "id": results_id,
#             "labels": ["Results"],
#             "properties": {
#                 "id":results_id,
#                 "results": {},
#                 "algo":algo
                
#                 # Initialize the "results" property as an empty dictionary
#             }
#         }
#         graph_data["nodes"].append(results_node)

#         for idx, file_data in enumerate(file_paths, start=1):
#             file_path = file_data.get("value")
#             url = file_data.get("url")
#             document_id = str(uuid.uuid4())
#             # Create Document node
#             document_node = {
#                 "id":document_id,
#                 "labels": ["Document"],
#                 "properties": {
#                     "file_path": file_path,
#                     "url": url,
#                     "id":document_id
                    
#                 }
#             }
#             graph_data["nodes"].append(document_node)

#             # Create relationship between Document and Results nodes with the specified relationshipName
#             document_to_results_link = {
#                 "source":document_id,
#                 "target": results_id,
#                 "type": f"hasResultsWith{algo}", 
#                 "properties": {}
#             }
#             graph_data["links"].append(document_to_results_link)

#             # Create relationship between Patient and Document nodes with "hasSummary"
#             patient_to_document_link = {
#                 "source":patientuuid_id,  # Correctly assign the patient node ID as the source
#                 "target":document_id,
#                 "type": "hasSummary",
#                 "properties": {}
#             }
#             graph_data["links"].append(patient_to_document_link)

#             if algo == "LDA":
#                 try:
#                     results = getTopicForOneLDA(file_path)  # Replace with actual function to get LDA results
#                 except FileNotFoundError:
#                     return jsonify({"error": "File not found."}), 404
#             elif algo == "NMF":
#                 try:
#                     results = NMF_topicsByOne(file_path, 5)  # Replace with actual function to get NMF results
#                 except FileNotFoundError:
#                     return jsonify({"error": "File not found."}), 404
#             else:
#                 return jsonify({"error": "Algorithm not supported."}), 400

#             # Update the "results" property of the "Results" node with the results
#             graph_data["nodes"][1]["properties"]["results"][f"{idx-1}"] = results

#         graph_data["nodes"][1]["properties"]["results"] = json.dumps(graph_data["nodes"][1]["properties"]["results"])
#         #here
#         if not graph_data["nodes"] or not graph_data["links"]:
#             print("No nodes or links to process.")
#             return jsonify(graph_data), 200
#         query_uuid = createGraphGS(session, patient_id, "Extract Topics For Multiple  Documents", query_key, "Extracting")

#         patient_to_query_link = {
#             "source": f"{query_uuid}",
#             "target": f"{patientuuid_id}",
#             "type": "hasResultSummary",
#             "properties": {}
#         }
#         graph_data["links"].append(patient_to_query_link)  
#         graph_data_response = createResponseGraphInDatabase(
#             session, graph_data["nodes"], graph_data["links"])
            
            
#         return jsonify(graph_data), 200
#Confirmed 
@app.route('/abstractive/summarize', methods=['POST'])
def summarize_files():
    request_data = request.json
    file_paths = request_data.get('file_paths', [])
    
    if not file_paths:
        return jsonify({'error': 'No file paths provided'}), 400

    patient_id = request_data.get('patient_id')
    algo = request_data.get('aproach').upper()
    query_key = request_data.get("query_key")
    
    with driver.session() as session:
        # Find the patient node with the given patient_id
        result = session.run(
            "MATCH (p:Patient) WHERE ID(p) = $patient_id RETURN p",
            patient_id=int(patient_id)  # Assuming patient_id is an integer
        )

        patient_node = result.single()
        if patient_node is None:
            return jsonify({"error": "Patient not found."}), 404

        patient_properties = patient_node["p"]

        results_id = str(uuid.uuid4()) 
        patientuuid_id = str(uuid.uuid4()) 
        
        graph_data = {
            "nodes": [
                {
                    "id":patientuuid_id,
                    "labels": ["PATIENT"],
                    "properties": {"id":patientuuid_id},
                }
            ],
            "links": []
        }

        

        # Create Results node
        results_node = {
            "id": results_id,
            "labels": ["Results"],
            "properties": {
                "id":results_id,
                "results": {}  # Initialize the "results" property as an empty dictionary
            }
        }
        graph_data["nodes"].append(results_node)

        for idx, file_data in enumerate(file_paths, start=1):
            file_path = file_data.get("value")
            url = file_data.get("url")

            # Assuming the "approach" is the same for all file paths
            if algo == 'GRAPH_BASED_PG':
                try:
                    summary = getGraphBasedSummarize_pg(file_path)
                except Exception as e:
                    summary = {'error': str(e)}
            elif algo == 'GRAPH_BASED_TEXTRANK':
                try:
                    summary = getGraphBasedSummarize_pg(file_path)
                except Exception as e:
                    summary = {'error': str(e)}
            elif algo == 'TREE_BASED':
                try:
                    summary = getTreeBasedSummarize_pg(file_path)
                except Exception as e:
                    summary = {'error': str(e)}
            else:
                summary = {'error': 'Invalid approach'}
                return jsonify({'error': 'Invalid approach'}), 500
                
            document_id = str(uuid.uuid4())
            # Create Document node
            document_node = {
                "id": document_id,
                "labels": ["Document"],
                "properties": {
                    "file_path": file_path,
                    "url": url,
                    "id":document_id
                }
            }
            graph_data["nodes"].append(document_node)

            # Create relationship between Document and Results nodes with the specified relationshipName
            document_to_results_link = {
                "source": document_id,
                "target": results_id,
                "type": f"hasResultsWith{algo}",
                "properties": {}
            }
            graph_data["links"].append(document_to_results_link)

            # Create relationship between Patient and Document nodes with "hasSummary"
            patient_to_document_link = {
                "source": patientuuid_id,  # Correctly assign the patient node ID as the source
                "target": document_id,
                "type": "hasSummary",
                "properties": {}
            }
            graph_data["links"].append(patient_to_document_link)

            # Update the "results" property of the "Results" node with the summary
            graph_data["nodes"][1]["properties"]["results"][f"{idx-1}"] = summary
            
        graph_data["nodes"][1]["properties"]["results"] = json.dumps(graph_data["nodes"][1]["properties"]["results"])

        if not graph_data["nodes"] or not graph_data["links"]:
            print("No nodes or links to process.")
            return jsonify(graph_data), 200
        query_uuid = createGraphGS(session, patient_id, "Get Summary  For Multiple  Documents", query_key, "Extracting")

        patient_to_query_link = {
            "source": f"{query_uuid}",
            "target": f"{patientuuid_id}",
            "type": "hasResultSummary",
            "properties": {}
        }
        graph_data["links"].append(patient_to_query_link)  
        graph_data_response = createResponseGraphInDatabase(
            session, graph_data["nodes"], graph_data["links"])

    return jsonify(graph_data), 200
#Confirmed 
@app.route("/havecancer", methods=["GET"])
def has_cancer():
    image_path = request.args.get("image_path")
    url = request.args.get("url")
    patient_id = str(request.args.get("patient_id"))
    query_key=request.args.get("query_key")
    try:
        predicted_String = predict_cancer(image_path)
        image_id = str(uuid.uuid4())
        xray_node={
            
             "id":image_id,
                "labels": ["IMAGE"],
                "properties": {
                    
                    "url": url,
                    "id":image_id,
                    "results":predicted_String
                    
                }
                
        }
        with driver.session() as session:
            query_uuid = createGraphGS(session, patient_id, "isHasCancer", query_key, "Analyse")
            patient_to_query_link = {
            "source": f"{query_uuid}",
            "target": f"{image_id}",
            "type": "hasResultSummary",
            "properties": {}
        }
            graph_data = {
            "nodes": [
                
                 
                
            ],
            "links": []
        }
            graph_data["links"].append(patient_to_query_link)
            graph_data["nodes"].append(xray_node)  
            graph_data_response = createResponseGraphInDatabase(
                session, graph_data["nodes"], graph_data["links"])
        return  predicted_String
    except FileNotFoundError :
        return jsonify({"error": "XRay not found."}), 404
   
#************************************************************** Waiting For Test#***********************************************************************************************************8

#Confirmed 
@app.route("/topic/all", methods=["GET"])
def getAlltopics():
    patient_id = request.args.get("patient_id")
    folder_path = f"./files/{str(patient_id)}"
    algo = request.args.get("algo")
    query_key = request.args.get("query_key")
    
    if not os.path.exists(folder_path):
        error_message = f"Folder of patient {patient_id} is not found."
        return jsonify({"error": error_message}), 404

    with driver.session() as session:
        # Find the patient node with the given patient_id
        result = session.run(
            "MATCH (p:Patient) WHERE ID(p) = $patient_id RETURN p",
            patient_id=int(patient_id)  # Assuming patient_id is an integer
        )

        patient_node = result.single()
        if patient_node is None:
            return jsonify({"error": "Patient not found."}), 404

        patient_properties = patient_node["p"]

        if algo == "LDA":
            try:
                results = getAllTopicsLDA(folder_path)
            except FileNotFoundError:
                return jsonify({"error": "File not found."}), 404

        elif algo == "NMF":
            try:
                results = NMF_topicsByFolder(folder_path, 5)
            except FileNotFoundError:
                return jsonify({"error": "File not found."}), 404

        else:
            return jsonify({"error": "Algorithm is not supported"}), 500

        # Convert the results list to a JSON-serializable format
        
        results_id = str(uuid.uuid4()) 
        patientuuid_id = str(uuid.uuid4()) 
        
        # Build the graph data
        graph_data = {
            "nodes": [
                {
                    "id": patientuuid_id,
                    "labels": ["PATIENT"],
                    "properties": {"id":patientuuid_id},
                },
                {
                    "id": results_id,
                    "labels": ["Results"],
                    "properties": {
                        "results": json.dumps(results),
                        "id":results_id
                        # Use the JSON-serializable results
                    }
                }
            ],
            "links": [
             
            ]
        }

        # Add the Document nodes and relationships to graph_data
        file_paths = os.listdir(folder_path)
        for idx, file_path in enumerate(file_paths, start=1):
            if file_path.endswith(".pdf"):
                document_id = str(uuid.uuid4())
                document_node = {
                    "id": document_id,
                    "labels": ["Document"],
                    "properties": {
                        "file_path": file_path,
                        "url": "", 
                        "id":document_id,
                        # Set the URL as needed
                    }
                }
                graph_data["nodes"].append(document_node)

                # Create relationship between Document and Results nodes with the specified relationshipName
                document_to_results_link = {
                    "source": document_id,
                    "target": results_id,
                    "type": f"hasResultsWith{algo}",
                    "properties": {}
                }
                graph_data["links"].append(document_to_results_link)

                # Create relationship between Patient and Document nodes with "hasSummary"
                patient_to_document_link = {
                    "source": patientuuid_id,
                    "target": document_id,
                    "type": "hasSummary",
                    "properties": {}
                }
                graph_data["links"].append(patient_to_document_link)
        if not graph_data["nodes"] or not graph_data["links"]:
            print("No nodes or links to process.")
            return jsonify(graph_data), 200
        query_uuid = createGraphGS(session, patient_id, "Get Summary  For All  Documents", query_key, "Extracting")

        patient_to_query_link = {
            "source": f"{query_uuid}",
            "target": f"{patientuuid_id}",
            "type": "hasResultSummary",
            "properties": {}
        }
        graph_data["links"].append(patient_to_query_link)  
        graph_data_response = createResponseGraphInDatabase(
            session, graph_data["nodes"], graph_data["links"])
        return jsonify(graph_data), 200

#Confirmed 👌
@app.route("/xray_images", methods=["GET"])
def get_xray_images():
    months = request.args.getlist("month") 
      
    year = request.args.get("year")
    patient_id = request.args.get("patient_id")
    searched_node = request.args.get("searched_node")
    query_key=request.args.get("query_key")
    is_summary = request.args.get("is_summary")
    
    if query_key is None or patient_id is None:
        return jsonify({"error": "Missing required query parameters."}), 400
    

    with driver.session() as session:
        result = session.run(
            "MATCH (xray:" + searched_node + ") "
            "WHERE xray.patient_id = $patient_id "
            "AND substring(xray.date, 0, 4) = $year "
            "AND substring(xray.date, 5, 2) IN $months "
            "RETURN xray, [(xray)-[r]->(related) | {end_node: related, type: type(r), properties: properties(r)}] AS relationships",
            patient_id=patient_id,
            year=year,
            months=months
        )

        nodes = []
        links = []

        for record in result:
            xray_node = record["xray"]
            relationships = record["relationships"]

            # Extract XRay node information
            node_data = {
                "id": str(xray_node.id),
                "labels": list(xray_node.labels),
                "properties": dict(xray_node),
            }
            nodes.append(node_data)

            # Extract relationships
            for rel in relationships:
                link_data = {
                    "source": str(xray_node.id),
                    "target": str(rel['end_node'].id),
                    "type": rel['type'],
                    "properties": dict(rel['properties']),
                }
                links.append(link_data)

        if not nodes:
            return jsonify({"message": "No " + searched_node + " found"}), 404
        json_data = {
            "nodes": nodes,
            "links": links,
        }
        if is_summary =='is_summary':
            return jsonify(json_data), 200
        
        Filter_id = str(uuid.uuid4())
        query_uuid = createGraphGS(session, patient_id, "Filter My Files per Node Name", query_key, "Filtering")
        Filter_node= {
            
             "id":Filter_id,
                "labels": ["FilterResults"],
                "properties": {
                    
                   
                    "id":Filter_id,
                   "results":f"{request.full_path}&is_summary=is_summary"

                    
                }
        }
        patient_to_query_link = {
            "source": f"{query_uuid}",
            "target": f"{Filter_id}",
            "type": "hasResultSummary",
            "properties": {}
        }
        graph_data_response = createResponseGraphInDatabase(
                session,[Filter_node], [patient_to_query_link])
        

        return jsonify(json_data), 200

#Confirmed 👌
@app.route('/filter/type', methods=['GET'])
def filter_by_type():
    # Get query parameters from the request
    patient_id = str(request.args.get('patient_id', ''))
    node_type = request.args.get('type', '')
    months = request.args.getlist('month')
    year = request.args.get('year', '')
    query_key=request.args.get("query_key")
    is_summary = request.args.get("is_summary")
    if query_key is None or patient_id is None:
        return jsonify({"error": "Missing required query parameters."}), 400
    print("***")
    print( f"/filter/type?patient_id={patient_id}&year={year}&month={'&month='.join(months)}&type={node_type}")

    with driver.session() as session:
        # Cypher query to fetch nodes and their relationships based on conditions
        query = """
        MATCH (node {patient_id: $patient_id, type: $node_type})
        WHERE substring(node.date, 5, 2) IN $months AND substring(node.date, 0, 4) = $year
        OPTIONAL MATCH (node)-[r]->(related_node)
        RETURN node, collect(DISTINCT r) AS relationships, collect(DISTINCT related_node) AS related_nodes
        """

        result = session.run(query, patient_id=str(patient_id), node_type=node_type, months=months, year=year)

        nodes_data = []
        links_data = []

        # Add the patient node as the root node
        patient_node_data = {
            "id": str(patient_id),
            "labels": ["Patient"],
            "properties": {}  # Create an empty dictionary to store properties
        }

        nodes_data.append(patient_node_data)

        # Keep track of node IDs that the patient node should be linked to
        patient_linked_node_ids = set()

        for record in result:
            node = record["node"]
            relationships = record["relationships"]
            related_nodes = record["related_nodes"]

            node_data = {
                "id": str(node.id),
                "labels": list(node.labels),
                "properties": {},  # Create an empty dictionary to store properties
            }

            # Convert node properties to a dictionary and handle bytes properties
            for key, value in node.items():
                if isinstance(value, bytes):
                    node_data["properties"][key] = value.decode('utf-8')
                else:
                    node_data["properties"][key] = value

            nodes_data.append(node_data)

            for relationship, related_node in zip(relationships, related_nodes):
                relationship_data = {
                    "source": str(node.id),
                    "target": str(related_node.id),
                    "type": relationship.type,
                    "properties": {},  # Create an empty dictionary to store properties
                }

                # Convert relationship properties to a dictionary and handle bytes properties
                for key, value in relationship.items():
                    if isinstance(value, bytes):
                        relationship_data["properties"][key] = value.decode('utf-8')
                    else:
                        relationship_data["properties"][key] = value

                links_data.append(relationship_data)

                # Add related_node ID to the patient_linked_node_ids set
                patient_linked_node_ids.add(str(related_node.id))

        # Retrieve patient node separately and add its properties to patient_node_data
        patient_query = """
        MATCH (p:Patient)
        WHERE ID(p) = $patient_id
        RETURN p
        """

        patient_result = session.run(patient_query, patient_id=int(patient_id))

        for record in patient_result:
            patient_node = record["p"]
            # Convert patient properties to a dictionary and handle bytes properties
            for key, value in patient_node.items():
                if isinstance(value, bytes):
                    patient_node_data["properties"][key] = value.decode('utf-8')
                else:
                    patient_node_data["properties"][key] = value

        # Create relationships between Patient and other nodes that have no explicit relationships
        for node_data in nodes_data:
            if node_data["id"] != str(patient_id) and node_data["id"] not in patient_linked_node_ids:
                patient_to_node_link = {
                    "source": str(patient_id),
                    "target": node_data["id"],
                    "type": "hasSummary",
                    "properties": {}
                }
                links_data.append(patient_to_node_link)
        json_data = {
            "nodes": nodes_data,
            "links": links_data
        }
        if is_summary =='is_summary':
            return jsonify(json_data), 200
        
        query_uuid = createGraphGS(session, patient_id, "Filter My Filtes per Type", query_key, "Filtering")
        Filter_id = str(uuid.uuid4())
        Filter_node= {
            
             "id":Filter_id,
                "labels": ["FilterResults"],
                "properties": {
                    
                   
                    "id":Filter_id,
                   "results": f"{request.full_path}&is_summary=is_summary"

                    
                }
        }
        patient_to_query_link = {
            "source": f"{query_uuid}",
            "target": f"{Filter_id}",
            "type": "hasResultSummary",
            "properties": {}
        }
        graph_data_response = createResponseGraphInDatabase(
                session,[Filter_node], [patient_to_query_link])
        

        return jsonify(json_data), 200
#Confirmed 👌
@app.route('/display/type', methods=['GET'])
def display_by_type():
    patient_id = str(request.args.get('patient_id', ''))
    node_types = request.args.getlist('type')  # Allow multiple types
    months = request.args.getlist('month')
    year = request.args.get('year', '')
    query_key=request.args.get("query_key")
    is_summary = request.args.get("is_summary")
    
    if query_key is None or patient_id is None:
        return jsonify({"error": "Missing required query parameters."}), 400
    with driver.session() as session:
        # Cypher query to fetch nodes and their relationships based on conditions
        query = """
        MATCH (node {patient_id: $patient_id})
        WHERE $node_types IS NULL OR node.type IN $node_types  // Allow filtering by multiple types
        AND ($months = [] OR (substring(node.date, 5, 2) IN $months AND substring(node.date, 0, 4) = $year))
        OPTIONAL MATCH (node)-[r]->(related_node)
        RETURN node, collect(DISTINCT r) AS relationships, collect(DISTINCT related_node) AS related_nodes
        """

        # Convert empty strings to empty lists for months parameter
        months = months if months else []

        result = session.run(query, patient_id=str(patient_id), node_types=node_types, months=months, year=year)

        nodes_data = []
        links_data = []

        # Add the patient node as the root node
        patient_node_data = {
            "id": str(patient_id),
            "labels": ["Patient"],
            "properties": {}  # Create an empty dictionary to store properties
        }

        nodes_data.append(patient_node_data)

        # Keep track of node IDs that the patient node should be linked to
        patient_linked_node_ids = set()

        for record in result:
            node = record["node"]
            relationships = record["relationships"]
            related_nodes = record["related_nodes"]

            node_data = {
                "id": str(node.id),
                "labels": list(node.labels),
                "properties": {},  # Create an empty dictionary to store properties
            }

            # Convert node properties to a dictionary and handle bytes properties
            for key, value in node.items():
                if isinstance(value, bytes):
                    node_data["properties"][key] = value.decode('utf-8')
                else:
                    node_data["properties"][key] = value

            nodes_data.append(node_data)

            for relationship, related_node in zip(relationships, related_nodes):
                relationship_data = {
                    "source": str(node.id),
                    "target": str(related_node.id),
                    "type": relationship.type,
                    "properties": {},  # Create an empty dictionary to store properties
                }

                # Convert relationship properties to a dictionary and handle bytes properties
                for key, value in relationship.items():
                    if isinstance(value, bytes):
                        relationship_data["properties"][key] = value.decode('utf-8')
                    else:
                        relationship_data["properties"][key] = value

                links_data.append(relationship_data)

                # Add related_node ID to the patient_linked_node_ids set
                patient_linked_node_ids.add(str(related_node.id))

        # Retrieve patient node separately and add its properties to patient_node_data
        patient_query = """
        MATCH (p:Patient)
        WHERE ID(p) = $patient_id
        RETURN p
        """

        patient_result = session.run(patient_query, patient_id=int(patient_id))

        for record in patient_result:
            patient_node = record["p"]
            # Convert patient properties to a dictionary and handle bytes properties
            for key, value in patient_node.items():
                if isinstance(value, bytes):
                    patient_node_data["properties"][key] = value.decode('utf-8')
                else:
                    patient_node_data["properties"][key] = value

        # Create relationships between Patient and other nodes that have no explicit relationships
        for node_data in nodes_data:
            if node_data["id"] != str(patient_id) and node_data["id"] not in patient_linked_node_ids:
                patient_to_node_link = {
                    "source": str(patient_id),
                    "target": node_data["id"],
                    "type": "hasSummary",
                    "properties": {}
                }
                links_data.append(patient_to_node_link)
        
        json_data = {
            "nodes": nodes_data,
            "links": links_data
        }
        if is_summary =='is_summary':
            return jsonify(json_data), 200
        
        query_uuid = createGraphGS(session, patient_id, f"Display Patient {patient_id}  ", query_key, "Display")
        Filter_id = str(uuid.uuid4())
        Filter_node= {
            
             "id":Filter_id,
                "labels": ["DisplayResults"],
                "properties": {
                    
                   
                    "id":Filter_id,
                   "results": f"{request.full_path}&is_summary=is_summary"

                    
                }
        }
        patient_to_query_link = {
            "source": f"{query_uuid}",
            "target": f"{Filter_id}",
            "type": "hasResultSummary",
            "properties": {}
        }
        graph_data_response = createResponseGraphInDatabase(
                session,[Filter_node], [patient_to_query_link])

        
        

        return jsonify(json_data), 200


#************************************************************** Graph Summary***********************************************************************************************************8
@app.route('/patient/graphSummary', methods=['GET'])
def get_graph_summary():
    graph = nx.DiGraph()
    patient_id = request.args.get("patient_id")
    with driver.session() as session:
        try:
            result = session.run(
                "MATCH path = (p:GS)-[*]-(n)WHERE p.patient = $patient_id RETURN path",
                patient_id=str(patient_id)
            )

            found_path = False

            for record in result:
                path = record["path"]

                if path is not None:
                    found_path = True
                    nodes = path.nodes
                    relationships = path.relationships

                    for node in nodes:
                        # Convert byte properties to strings
                        properties = {key: value.decode() if isinstance(value, bytes) else value for key, value in dict(node).items()}

                        # Add nodes to the graph
                        graph.add_node(node.id, labels=list(node.labels), properties=properties)

                    for relationship in relationships:
                        # Convert byte properties to strings
                        properties = {key: value.decode() if isinstance(value, bytes) else value for key, value in dict(relationship).items()}

                        # Add relationships to the graph
                        start_node = relationship.start_node
                        end_node = relationship.end_node
                        graph.add_edge(start_node.id, end_node.id, type=relationship.type, properties=properties)

            if not found_path:
                # If no relationships found, add only the patient node to the graph
                result = session.run(
                    "MATCH (p:GS) WHERE  p.patient = $patient_id RETURN p",
                    patient_id=str(patient_id)
                )
                node = result.single()["p"]
                properties = {key: value.decode() if isinstance(value, bytes) else value for key, value in dict(node).items()}
                graph.add_node(node.id, labels=list(node.labels), properties=properties)

            # Convert graph to JSON
            json_data = {
                "nodes": [
                    {
                        "id": str(node_id),
                        "labels": data["labels"],
                        "properties": data["properties"]
                    }
                    for node_id, data in graph.nodes(data=True)
                ],
                "links": [
                    {
                        "source": str(start),
                        "target": str(end),
                        "type": data["type"],
                        "properties": data["properties"]
                    }
                    for start, end, data in graph.edges(data=True)
                ]
            }

            return jsonify(json_data), 200
        except Exception as e:
            print(str(e))
            return jsonify({"error": str(e)}), 500

@app.route('/Summary/queryNames', methods=['GET'])
def fetch_query_names():
    patient_id = request.args.get("patient_id")

    if not patient_id:
        return jsonify({"error": "Missing patient_id parameter"}), 400

    try:
        with driver.session() as session:
            result = session.run("MATCH (q:Query) WHERE q.patient = $patient_id RETURN q.key AS key",
                                 patient_id=str(patient_id))
            query_names = [record["key"] for record in result]

            return jsonify(query_names), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
BASE_IRI = "http://semanticweb.org/neo4j/"

@app.route('/patient/transformToRDF', methods=['POST'])
def transform_to_rdf_api():
    try:
        json_data = request.json

        if not json_data:
            return jsonify({"error": "Invalid JSON data"}), 400

        graph = nx.DiGraph()
        rdf_graph = Graph()

        for node in json_data["nodes"]:
            graph.add_node(node["id"], labels=node["label"], properties=node)
            node_id = URIRef(BASE_IRI + str(node["id"]))  # Use custom IRI for nodes
            rdf_graph.add((node_id, RDF.type, URIRef(BASE_IRI + node["label"][0])))

            for key, value in node.items():
                rdf_graph.add((node_id, URIRef(BASE_IRI + key), Literal(value)))

        for link in json_data["edges"]:
            

            graph.add_edge(link["source"], link["target"], type=link["type"], properties=link)

            source_id = URIRef(BASE_IRI + str(link["source"]))  # Use custom IRI for source nodes
            target_id = URIRef(BASE_IRI + str(link["target"]))  # Use custom IRI for target nodes
            relationship_type = URIRef(BASE_IRI + link["type"])  # Use custom IRI for relationship types

            # Add triple to represent relationship
            rdf_graph.add((source_id, relationship_type, target_id))

            for key, value in link.items():
                rdf_graph.add((source_id, URIRef(BASE_IRI + key), Literal(value)))

        rdf_ttl = rdf_graph.serialize(format='turtle')  

        return rdf_ttl, 200, {'Content-Type': 'text/turtle'}

    except Exception as e:
        print(str(e))
        return jsonify({"error": str(e)}), 500
SWAGGER_URL = '/swagger'
API_URL = '/swagger.json'

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Neo4j database project",
        'supportedSubmitMethods': ['get', 'post', 'put', 'delete'],  # Specify the supported HTTP methods
        'swaggerSecurityDefinitions': {
            'BearerAuth': {
                'type': 'apiKey',
                'name': 'Authorization',
                'in': 'header',
                'description': 'JWT authorization using the Bearer scheme. Example: "Bearer {token}"'
            }
        }
    }
)


app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

@app.route('/swagger.json')
def swagger_spec():
    with open('swagger.json', 'r') as f:
        swagger_data = json.load(f)

    swagger_data['securityDefinitions'] = {
        'BearerAuth': {
            'type': 'apiKey',
            'name': 'Authorization',
            'in': 'header',
            'description': 'JWT authorization using the Bearer scheme. Example: "Bearer {token}"'
        }
    }

    return jsonify(swagger_data)

def record_to_dict(record: Record) -> dict:
    """Converts a Neo4j record to a dictionary."""
    return {key: value.decode('utf-8') if isinstance(value, bytes) else value for key, value in record.items()}

# **************************************************** Server ***************************************************************************
if __name__ == "__main__": 
    app.run(debug=True,threaded=True)

