import sys
import argparse
import numpy as np
import json
import os
from pathlib import Path
import sys
import csv
import copy
import glob

file_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
file_path = Path(file_path).as_posix()
sys.path.append(file_path)

from ONMA.ONMAModel import ONMAModel

total_TC = 0
passed_TC = 0
failed_TC = 0

inputs_type = []
outputs_type = []
attributes_list = []
summary_tc_list = []

def create_summary_for_one_test_case(tcname, graph_data, status):
    global summary_tc_list
    report_dictionary = {}
    report_dictionary["ID"] = tcname
    report_dictionary["RESULT"] = status
    report_dictionary["inputs"] = {}
    report_dictionary["outputs"] = {}
    report_dictionary["attributes"] = {}
    for item in graph_data["graph"]:
        if item == "inputs":
            for one_in in graph_data["graph"]["inputs"]:
                if one_in["data_type"] not in inputs_type: inputs_type.append(one_in["data_type"])
                if one_in["data_type"] not in report_dictionary["inputs"]:
                    report_dictionary["inputs"][one_in["data_type"]] = one_in["name"]
                else:
                    report_dictionary["inputs"][one_in["data_type"]] = report_dictionary["inputs"][one_in["data_type"]] + "/" + one_in["name"]
        elif item == "outputs":
            for one_out in graph_data["graph"]["outputs"]:
                if one_out["data_type"] not in outputs_type: outputs_type.append(one_out["data_type"])
                if one_out["data_type"] not in report_dictionary["outputs"]:
                    report_dictionary["outputs"][one_out["data_type"]] = one_out["name"]
                else:
                    report_dictionary["outputs"][one_out["data_type"]] = report_dictionary["outputs"][one_out["data_type"]] + "/" + one_out["name"]
        elif item == "name":
            pass
        elif item == "nodes":
            for one_node in graph_data["graph"]["nodes"]:
                if "attributes" in one_node:
                    for attribute in one_node["attributes"]:
                        if attribute not in attributes_list: attributes_list.append(attribute)
                        report_dictionary["attributes"][attribute] = one_node["attributes"][attribute]
    summary_tc_list.append(copy.deepcopy(report_dictionary))

def run_TC(tcname, graph_data, expect):
    global total_TC
    global passed_TC
    global failed_TC
    global summary_tc_list

    status = True    
    model = ONMAModel()
    total_TC = total_TC + 1
    try:
        model.ONMAModel_CreateNetworkFromGraph(graph_data)
        inf1 = model.ONMAModel_Inference(graph_data["graph"]["inputs"])
        outputs = expect
        for index in range(0, len(outputs)):
            result = inf1[index]
            # are_close = np.allclose(result, outputs[index]["data"], atol=0.001, rtol=0)
            # print(are_close)
            if (result == outputs[index]["data"]).all():
                pass
            else:
                # Compare float32
                if np.allclose(np.array(result), np.array(expect[outputs[index]]["data"]), atol=0.01, rtol=0):
                    pass
                else:
                    status = False
    except:
        status = False

    if status == True:
        passed_TC = passed_TC + 1
        create_summary_for_one_test_case(tcname, graph_data, "PASSED")
    else:
        failed_TC = failed_TC + 1
        create_summary_for_one_test_case(tcname, graph_data, "FAILED")
    
    create_summary_report("summary_report.csv")
    return status

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

TEST_DATA = []

test_spec_path = os.path.join("Tests", "Unit", "Specification")
for filepath in glob.glob(f'{test_spec_path}/*', recursive=False):
    with open(filepath) as user_file:
        file_contents = user_file.read()
    json_contents = json.loads(file_contents)
    for item in json_contents:
        TEST_DATA.append([item, json_contents[item], json_contents[item]["graph"]["outputs"]])

def pytest_generate_tests(metafunc):
    if {"tcname", "graph_data", "expected"} <= set(metafunc.fixturenames):
        metafunc.parametrize("tcname,graph_data,expected", TEST_DATA)

def test_execute(tcname, graph_data, expected):
    assert run_TC(tcname, graph_data, expected) == True
