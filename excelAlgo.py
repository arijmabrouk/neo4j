import uuid
import pandas as pd
import json
import csv


def getMesureExcel(filepath, aggregateFN):
    data = pd.read_excel(filepath)
    uud = str(uuid.uuid4())
    # Check if 'Patient' column exists in the data
    if 'Patient' not in data.columns:
        return json.dumps({"error": "Patient column not found in the Excel file."})

    # Extract patient name
    patient_name = data['Patient'].iloc[0]

    # Extract date and measure columns
    if 'Date' not in data.columns or 'Mesures' not in data.columns:
        return json.dumps({"error": "Date or Mesures column not found in the Excel file."})

    date_column = data['Date']
    measure_column = data['Mesures']

    # Create a list of date-measure dictionaries
    levels = []
    for date, measure in zip(date_column, measure_column):
        level = {'date': str(date), 'measure': measure}
        levels.append(level)

    # Sort levels based on date from oldest to newest
    levels = sorted(levels, key=lambda x: x['date'])

    # Calculate max, min, and average measures
    measures = [level['measure'] for level in levels]
    max_measure = max(measures)
    min_measure = min(measures)
    avg_measure = sum(measures) / len(measures)

    # Find the level with the maximum measure
    level_max = next((level for level in levels if level['measure'] == max_measure), None)

    # Find the level with the minimum measure
    level_min = next((level for level in levels if level['measure'] == min_measure), None)

    # Extract the desired word (case-insensitive)
    word = ''
    keywords = ['temperature', 'glucose', 'blood sugar', 'blood pressure']
    for column_name in data.columns:
        if any(keyword.lower() in column_name.lower() for keyword in keywords):
            word = column_name
            break

    if not word:
        return json.dumps({"error": "No relevant keyword found in column names."})

    # Create the "Result" node with all information for graphical case
    result_node = {
        'id': uud,  # You can assign a unique ID for the node
        'labels': ['Results'],
        'properties': {
            'id': uud,
            'value': None,
            'date': None,
            'patient': patient_name,
            'mesureOf': word,
            "aggregateFN":aggregateFN

        }
    }

    if aggregateFN.lower() == 'max':
        result_node['properties']['value'] = max_measure
        result_node['properties']['date'] = level_max['date']
    elif aggregateFN.lower() == 'min':
        result_node['properties']['value'] = min_measure
        result_node['properties']['date'] = level_min['date']
    elif aggregateFN.lower() == 'avg':
        result_node['properties']['value'] = avg_measure
    elif aggregateFN.lower() == 'graphical':
     
        value_json = json.dumps({
                'max': {'date': level_max['date'], 'measure': max_measure},
                'min': {'date': level_min['date'], 'measure': min_measure},
                'avg': avg_measure,
                'patient': patient_name,
                'mesureOf': word,
                'levels': levels
            })

        result_node['properties']['value'] = value_json        
        result_node['properties']['date'] = None

    # Convert the result to JSON
    json_data = {
        "nodes": [result_node],
        "links": []
    }

    

    return json_data, uud

def getNumericFilter(path):
    res = getMesureExcel(path)
    return res




def extract_csv_data(file_path, column_index=0):
    values = []

    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            try:
                value = float(row[column_index])
                values.append(value)
            except (ValueError, IndexError):
                # Handle cases where the value is not a valid float or the index is out of bounds
                pass

    return values
