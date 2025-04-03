import sys
import argparse
import numpy as np
import json
import os
from pathlib import Path
import sys
import csv

file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = Path(file_path).as_posix()
sys.path.append(file_path)

from ONMAModel import ONMAModel

total_TC = 0
passed_TC = 0
failed_TC = 0

inputs_type = []
outputs_type = []
attributes_list = []
summary_tc_list = []

def create_summary_for_one_test_case(tcname, graph_data, status):
    report_dictionary = {}
    report_dictionary["ID"] = tcname
    report_dictionary["RESULT"] = status
    report_dictionary["inputs"] = {}
    report_dictionary["outputs"] = {}
    report_dictionary["attributes"] = {}
    for item in graph_data:
        if item == "inputs":
            for one_in in graph_data["inputs"]:
                if graph_data["inputs"][one_in]["type"] not in inputs_type: inputs_type.append(graph_data["inputs"][one_in]["type"])
                if graph_data["inputs"][one_in]["type"] not in report_dictionary["inputs"]:
                    report_dictionary["inputs"][graph_data["inputs"][one_in]["type"]] = one_in
                else:
                    report_dictionary["inputs"][graph_data["inputs"][one_in]["type"]] = report_dictionary["inputs"][graph_data["inputs"][one_in]["type"]] + "/" + one_in
        elif item == "outputs":
            for one_out in graph_data["outputs"]:
                if graph_data["outputs"][one_out]["type"] not in outputs_type: outputs_type.append(graph_data["outputs"][one_out]["type"])
                if graph_data["outputs"][one_out]["type"] not in report_dictionary["outputs"]:
                    report_dictionary["outputs"][graph_data["outputs"][one_out]["type"]] = one_out
                else:
                    report_dictionary["outputs"][graph_data["outputs"][one_out]["type"]] = report_dictionary["outputs"][graph_data["outputs"][one_out]["type"]] + "/" + one_out
        elif item == "graph_name":
            pass
        else:
            for node_item in graph_data[item]:
                if node_item == "Action" or node_item == "Category" or node_item == "Type" or node_item == "inputs" or node_item == "outputs":
                    pass
                else:
                    if node_item not in attributes_list: attributes_list.append(node_item)
                    report_dictionary["attributes"][node_item] = graph_data[item][node_item]
    summary_tc_list.append(report_dictionary)

def run_TC(tcname, graph_data, expect):
    global total_TC
    global passed_TC
    global failed_TC

    status = True    
    model = ONMAModel()
    total_TC = total_TC + 1
    try:
        inf1 = model.ONMAModel_CreateNetworkFromGraph(graph_data)
        outputs = list(expect.keys())
        for index in range(0, len(outputs)):
            result = inf1[index]
            if (result == expect[outputs[index]]["data"]).all():
                pass
            else:
                # Compare float32
                if np.allclose(np.array(result), np.array(expect[outputs[index]]["data"]), atol=1e-5):
                    pass
                else:
                    status = False
    except:
        status = False

    if status == True:
        # print(f'Test cases {tcname} is PASSED')
        passed_TC = passed_TC + 1
        create_summary_for_one_test_case(tcname, graph_data, "PASSED")
    else:
        failed_TC = failed_TC + 1
        create_summary_for_one_test_case(tcname, graph_data, "FAILED")
        print(f'Test cases {tcname} is FAILED')

def create_summary_report(output_file):
    # open the file in the write mode
    with open(output_file, 'w', encoding='UTF8', newline='') as f:
        # create the csv writer
        writer = csv.writer(f)

        first_row = []
        first_row.append("")
        first_row.append("")
        first_row.append("Input")
        for i in range(0, len(inputs_type)-1): first_row.append("")
        first_row.append("Output")
        for i in range(0, len(outputs_type)-1): first_row.append("")
        first_row.append("Attribute")
        writer.writerow(first_row)

        # Write header
        writer.writerow(["ID", "RESULT"] + inputs_type + outputs_type + attributes_list)

        for one_test_case in summary_tc_list:

            summary_input = []
            for item in inputs_type:
                if item in one_test_case["inputs"]: summary_input.append(one_test_case["inputs"][item])
                else: summary_input.append("")

            summary_output = []
            for item in outputs_type:
                if item in one_test_case["outputs"]: summary_output.append(one_test_case["outputs"][item])
                else: summary_output.append("")

            summary_attributes = []
            for item in attributes_list:
                if item in one_test_case["attributes"]: summary_attributes.append(one_test_case["attributes"][item])
                else: summary_attributes.append("")

            # write a row to the csv file
            writer.writerow([one_test_case["ID"], one_test_case["RESULT"]] + summary_input + summary_output + summary_attributes)

def testaccuracy():
    global total_TC
    global passed_TC
    global failed_TC

    with open("Tests/reference_data.json") as user_file:
        file_contents = user_file.read()
    json_contents = json.loads(file_contents)

    for item in json_contents:
        run_TC(item, json_contents[item], json_contents[item]["outputs"])

    # create summary report
    create_summary_report("summary_report.csv")

    print(f'Total TC: {total_TC}')
    print(f'Passed TC: {passed_TC}')
    print(f'Failed TC: {failed_TC}')
    assert total_TC == passed_TC
