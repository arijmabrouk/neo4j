import json
import random
import string
from excelAlgo import getMesureExcel, getNumericFilter
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
from textAlgo import NMF_topicsByFolder, NMF_topicsByOne, getAllTopicsLDA, getGraphBasedSummarize_pg, getGraphBasedSummarize_tr, getTopicForOneHLDA, getTopicForOneLDA, getTreeBasedSummarize_pg
import uuid
from utils import createGraphGS, createLinksWithPATIENT, createResponseGraphInDatabase, extract_drugs_from_file, fetch_tree_graph, predict_cancer, transform_to_rdf
from rdflib import Graph, URIRef, Literal, RDF

from neo4j.debug import watch

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






















# **************************************************** Auth + Form ***************************************************************************

#done


# **************************************************** User :Doctor ***************************************************************************


 #*************************************************************newpart
def classify_document(file):
    # Get the file extension of the document
    file_extension = os.path.splitext(file)[1].lower()

    if file_extension == '.pdf':
        return 'PDF'
    elif file_extension in ('.jpg', '.jpeg', '.png', '.gif', '.bmp'):
        return 'Image'
    elif file_extension in ('.xlsx', '.xls'):
        return 'Excel'
    elif file_extension in ('.mp4', '.avi', '.mkv'):
        return 'Video'
    else:
        return 'Unknown'

    




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
                f"MATCH (u:{role} {{email: $email}}) RETURN u",
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
        patient_password = bcrypt.hashpw(data.get('password').encode('utf-8'), bcrypt.gensalt())

        try:
            result = session.run(
                "CREATE (p:Patient {email: $email, password: $password, nom: $nom, prenom: $prenom, age: $age, sexe: $sexe, telephone: $telephone, adresse: $adresse}) RETURN p",
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

    # Check if the file is allowed
    if document_file:
        # Save the file to the uploads folder
        filepath = os.path.join(uploads_dir, document_file.filename)
        document_file.save(filepath)

        with driver.session() as session:
            try:
                # Check if the Filtrate node exists, create it if not
                session.run(
                    "MERGE (f:Filtrate {name: 'filtrate'}) "
                    "WITH f "
                    "MATCH (p:Patient {email: $email}) "
                    "CREATE (d:Document {name: $document_name, filepath: $filepath})-[:HAS]->(p) "
                    "CREATE (p)-[:FILTRATE]->(f) "
                    "RETURN d",
                    email=patient_email,
                    document_name=document_name,
                    filepath=filepath
                )

                return jsonify({"message": "Document added successfully"}), 201
            except Exception as e:
                return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Invalid file format"}), 400



import base64

@app.route('/graph', methods=['GET'])
def get_graph():
    patient_email = request.args.get('patient_email')  # Get patient email from query parameters

    if not patient_email:
        return jsonify({"error": "Patient email is required"}), 400

    # Cypher query to retrieve the graph for a specific patient with related documents
    query = (
        "MATCH (d:Document)-[:HAS]->(p:Patient {email: $patient_email}) "
        "MATCH (f:Filtrate)-[:FILTRATE]->(p:Patient {email: $patient_email}) "
        "RETURN id(p) as patient_id, p as patient_properties, id(d) as document_id, d as document_properties, id(f) as filtrate_id"
    )

    with driver.session() as session:
        result = session.run(query, patient_email=patient_email).data()


    nodes = []
    edges = []

    for record in result:
        patient_id = record['patient_id']
        patient_properties = record['patient_properties']
        patient_name = patient_properties.get('nom', 'Unknown')  # Replace 'Unknown' with a default value

        document_id = record['document_id']
        document_properties = record['document_properties']
        document_name = document_properties.get('name', 'Unknown')  # Replace 'Unknown' with a default value

        filtrate_id = record["filtrate_id"]

        # Convert bytes to string or handle binary data appropriately
        for key, value in patient_properties.items():
            if isinstance(value, bytes):
                patient_properties[key] = value.decode('utf-8')

        for key, value in document_properties.items():
            if isinstance(value, bytes):
                # Example: base64 encode binary data
                document_properties[key] = base64.b64encode(value).decode('utf-8')

        # Include patient information and document name in the nodes
        nodes.append({'id': patient_id, 'label': f"Patient: {patient_name}", 'group': 'patient', 'info': patient_properties})
        nodes.append({'id': document_id, 'label': f"Document: {document_name}", 'group': 'document', 'info': document_properties})
        nodes.append({'id': filtrate_id, 'label': "Filtrate", 'group': 'filtrate'})

        edges.append({'from': patient_id, 'to': document_id, 'label': 'HAS'})
        edges.append({'from': patient_id, 'to': filtrate_id, 'label': 'Filtrate'})

    return jsonify({'nodes': nodes, 'edges': edges})


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


def serialize_node(node):
    serialized_node = dict(node)
    serialized_node["id"] = node.id
    for key, value in serialized_node.items():
        if isinstance(value, bytes):
            serialized_node[key] = value.decode("utf-8")
    return serialized_node


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
@app.route("/topic/multiple", methods=["POST"])
def gettopic_bymultiple():
    request_data = request.json
    print(request_data)
    patient_id = request_data.get("patient_id")
    file_paths = request_data.get("file_paths", [])
    algo = request_data.get("algo")
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

          # Generate a random UUID for the document node
        results_id = str(uuid.uuid4()) 
        patientuuid_id = str(uuid.uuid4()) 
    
        graph_data = {
            "nodes": [
                {
                    "id": patientuuid_id,
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
                "results": {},
                "algo":algo
                
                # Initialize the "results" property as an empty dictionary
            }
        }
        graph_data["nodes"].append(results_node)

        for idx, file_data in enumerate(file_paths, start=1):
            file_path = file_data.get("value")
            url = file_data.get("url")
            document_id = str(uuid.uuid4())
            # Create Document node
            document_node = {
                "id":document_id,
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
                "source":document_id,
                "target": results_id,
                "type": f"hasResultsWith{algo}", 
                "properties": {}
            }
            graph_data["links"].append(document_to_results_link)

            # Create relationship between Patient and Document nodes with "hasSummary"
            patient_to_document_link = {
                "source":patientuuid_id,  # Correctly assign the patient node ID as the source
                "target":document_id,
                "type": "hasSummary",
                "properties": {}
            }
            graph_data["links"].append(patient_to_document_link)

            if algo == "LDA":
                try:
                    results = getTopicForOneLDA(file_path)  # Replace with actual function to get LDA results
                except FileNotFoundError:
                    return jsonify({"error": "File not found."}), 404
            elif algo == "NMF":
                try:
                    results = NMF_topicsByOne(file_path, 5)  # Replace with actual function to get NMF results
                except FileNotFoundError:
                    return jsonify({"error": "File not found."}), 404
            else:
                return jsonify({"error": "Algorithm not supported."}), 400

            # Update the "results" property of the "Results" node with the results
            graph_data["nodes"][1]["properties"]["results"][f"{idx-1}"] = results

        graph_data["nodes"][1]["properties"]["results"] = json.dumps(graph_data["nodes"][1]["properties"]["results"])
        #here
        if not graph_data["nodes"] or not graph_data["links"]:
            print("No nodes or links to process.")
            return jsonify(graph_data), 200
        query_uuid = createGraphGS(session, patient_id, "Extract Topics For Multiple  Documents", query_key, "Extracting")

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
            graph.add_node(node["id"], labels=node["labels"], properties=node["properties"])
            node_id = URIRef(BASE_IRI + node["id"])  # Use custom IRI for nodes
            rdf_graph.add((node_id, RDF.type, URIRef(BASE_IRI + node["labels"][0])))

            for key, value in node["properties"].items():
                rdf_graph.add((node_id, URIRef(BASE_IRI + key), Literal(value)))

        for link in json_data["links"]:
            graph.add_edge(link["source"], link["target"], type=link["type"], properties=link["properties"])

            source_id = URIRef(BASE_IRI + link["source"])  # Use custom IRI for source nodes
            target_id = URIRef(BASE_IRI + link["target"])  # Use custom IRI for target nodes
            relationship_type = URIRef(BASE_IRI + link["type"])  # Use custom IRI for relationship types

            # Add triple to represent relationship
            rdf_graph.add((source_id, relationship_type, target_id))

            for key, value in link["properties"].items():
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

