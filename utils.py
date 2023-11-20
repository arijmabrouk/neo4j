import re

from neo4j import Record
import PyPDF2  # Required for PDF extraction
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import uuid
import networkx as nx
from rdflib import Graph, URIRef, Literal, RDF


def serialize_node(node):
    serialized_node = dict(node)
    serialized_node["id"] = node.id
    for key, value in serialized_node.items():
        if isinstance(value, bytes):
            serialized_node[key] = value.decode("utf-8")
    return serialized_node


def record_to_dict(record: Record) -> dict:
    """Converts a Neo4j record to a dictionary."""
    return {key: value.decode('utf-8') if isinstance(value, bytes) else value for key, value in record.items()}


disease_per_drugs = {
    "diabetes": ["metformin", "sulfonylureas", "insulin"],
    "hypertension": ["amlodipine", "losartan", "hydrochlorothiazide", "atenolol"],
    "asthma": ["albuterol", "fluticasone", "montelukast"],
    "cancer": ["chemotherapy", "radiation therapy", "targeted therapy", "immunotherapy"],
    "depression": ["selective serotonin reuptake inhibitors (SSRIs)", "serotonin-norepinephrine reuptake inhibitors (SNRIs)", "tricyclic antidepressants (TCAs)", "monoamine oxidase inhibitors (MAOIs)"],
    "arthritis": ["nonsteroidal anti-inflammatory drugs (NSAIDs)", "disease-modifying antirheumatic drugs (DMARDs)", "corticosteroids", "biologic response modifiers"],
    "migraine": ["triptans", "ergotamine", "nonsteroidal anti-inflammatory drugs (NSAIDs)", "beta-blockers"],
}


def extract_drugs_from_file(file_path, disease_name):
    file_extension = file_path.split(".")[-1]

    if file_extension == "pdf":
        # Extract text from PDF
        text = extract_text_from_pdf(file_path)
    else:
        return []  # Unsupported file type

    # Get the drugs specific to the given disease name
    disease_drugs = disease_per_drugs.get(disease_name, [])

    # Match drugs in the text using regular expressions
    drugs_found = []
    for drug in disease_drugs:
        if re.search(fr'\b{drug}\b', text, flags=re.IGNORECASE):
            drugs_found.append(drug)

    return drugs_found


def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
        text = ''
        for page in range(num_pages):
            page_obj = pdf_reader.pages[page]
            text += page_obj.extract_text()
        return text


def predict_cancer(path):

    model = load_model('./models/breast_cancer_model.h5')

    img_path = path
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    preds = model.predict(x)
    print(preds)
    # Replace with your class names
    class_names = ['benign', 'malignant', 'normal']
    predicted_class = np.argmax(preds)
    predicted_label = class_names[predicted_class]

    print(f'The image is predicted to be {predicted_label}.')
    return f'The image is predicted to be {predicted_label}.'


def createGraphGS(session, patient_id, query_name, query_key, query_type):
    # Generate a random UUID for the Query node
    query_uuid = str(uuid.uuid4())

    # Define the Cypher query to create the graph
    cypher_query = (
        "MATCH (gs:GS {patient: $patient_id}) "
        "CREATE (gs)-[:hasVersion]->(version:Version) "
        "CREATE (version)-[:hasQuery]->(query:Query {queryName: $query_name, key: $query_key, type: $query_type, id: $query_uuid,patient:$patient_id}) "
        "RETURN gs, version, query"
    )

    # Execute the Cypher query with parameters
    result = session.run(cypher_query, patient_id=patient_id, query_name=query_name,
                         query_key=query_key, query_type=query_type, query_uuid=query_uuid)

    # Process the result and extract the graph data
    graph_data = []
    for record in result:
        gs_node = record["gs"]
        version_node = record["version"]
        query_node = record["query"]

        graph_data.append({
            "GS": dict(gs_node),
            "Version": dict(version_node),
            "Query": dict(query_node)
        })

    return query_uuid


def createResponseGraphInDatabase(session, nodes, links):
    try:
        # Iterate through the nodes and create them in the database
        for node_data in nodes:
            labels = node_data["labels"]
            properties = node_data["properties"]
            node_query = f"CREATE (n:{':'.join(labels)} $props)"
            session.run(node_query, props=properties)

        # Iterate through the links and create relationships in the database
        for link_data in links:
            source_id = link_data["source"]
            target_id = link_data["target"]
            rel_type = link_data["type"]
            rel_query = (
                "MATCH (source), (target) "
                "WHERE source.id = $source_id AND target.id = $target_id "
                f"CREATE (source)-[:{rel_type}]->(target)"
            )

            try:
                result = session.run(
                    rel_query, source_id=source_id, target_id=target_id)
                for record in result:
                    print(
                        '\033[91m' + f"Results Comming From  relationship querry is : {record}" + '\033[0m')

                # Assuming 'result' is the variable containing the Neo4j query result


# If 'result' is a single record

                print(
                    '\033[92m' + f"Relationship created: {source_id} -> {target_id} (Type: {rel_type})" + '\033[0m')

            except Exception as e:
                print(
                    '\033[91m' + f"Error creating relationship: {source_id} -> {target_id} (Type: {rel_type})" + '\033[0m')
                print('\033[91m' + "Query:" + '\033[0m', rel_query)
                print('\033[91m' + "Error:" + '\033[0m', e)

    except Exception as e:
        print("Error creating nodes:", e)


def fetch_tree_graph(tx, patient_id):
    graph = nx.DiGraph()

    result = tx.run(
        "MATCH path = (gs:GS {patient: $patient_id})-[*]-(n) RETURN path",
        patient_id=str(patient_id)
    )

    for record in result:
        path = record["path"]
        print(f"path is {path}")

        if path is not None:
            for node in path.nodes:
                node_labels = list(node.labels)
                node_properties = {key: value.decode() if isinstance(
                    value, bytes) else value for key, value in dict(node).items()}
                graph.add_node(node.id, labels=node_labels,
                               properties=node_properties)

            for relationship in path.relationships:
                edge_type = relationship.type
                edge_properties = {key: value.decode() if isinstance(
                    value, bytes) else value for key, value in dict(relationship).items()}
                graph.add_edge(relationship.start_node.id, relationship.end_node.id,
                               type=edge_type, properties=edge_properties)

    nodes_list = []
    links_list = []

    for node_id, node_data in graph.nodes(data=True):
        node_labels = node_data["labels"]
        node_properties = node_data["properties"]

        node_dict = {
            "id": str(node_id),
            "labels": node_labels,
            "properties": node_properties
        }
        nodes_list.append(node_dict)

    for start, end, edge_data in graph.edges(data=True):
        edge_type = edge_data["type"]
        edge_properties = edge_data["properties"]

        edge_dict = {
            "source": str(start),
            "target": str(end),
            "type": edge_type,
            "properties": edge_properties
        }
        links_list.append(edge_dict)

    json_data = {
        "nodes": nodes_list,
        "links": links_list
    }

    return json_data


def createLinksWithPATIENT(session, patientuuid_id, links_Summary, query_uuid):
    
    
    session.run(
        "CREATE (p:PATIENT {id: $id})",
        id=patientuuid_id
    )

    # Create relationships using links_Summary
    for link in links_Summary:
        source_node_id = link["source"]
        target_node_id = link["target"]
        relationship_type = link["type"]

        rel_query = (
            f"MATCH (source), (target) "
            f"WHERE source.id = $source_id AND ID(target) = $target_id "
            f"CREATE (source)-[:{relationship_type}]->(target)"
        )

        session.run(
            rel_query,
            source_id=source_node_id,
            target_id=int(target_node_id)
        )

    # Create patient_to_query_link
    session.run(
        "MATCH (PATIENT), (Query) "
        "WHERE PATIENT.id = $patient_id AND Query.id = $query_id "
        "CREATE (Query)-[:hasResultSummary]->(PATIENT)",
        patient_id=patientuuid_id,
        query_id=query_uuid
    )

    # Create other relationships as needed
def transform_to_rdf(json_data):
    rdf_graph = Graph()

    for node in json_data["nodes"]:
        node_id = URIRef(node["id"])
        rdf_graph.add((node_id, RDF.type, URIRef(node["labels"][0])))

        for key, value in node["properties"].items():
            rdf_graph.add((node_id, URIRef(key), Literal(value)))

    for link in json_data["links"]:
        source_id = URIRef(link["source"])
        target_id = URIRef(link["target"])
        rdf_graph.add((source_id, URIRef(link["type"]), target_id))

        for key, value in link["properties"].items():
            rdf_graph.add((source_id, URIRef(key), Literal(value)))

    return rdf_graph