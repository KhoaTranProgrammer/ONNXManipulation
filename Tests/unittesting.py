import sys
import argparse
import numpy as np
import json
import os
from pathlib import Path
import sys

file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = Path(file_path).as_posix()
sys.path.append(file_path)

from ONMAModel import ONMAModel

total_TC = 0
passed_TC = 0
failed_TC = 0

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
    else:
        failed_TC = failed_TC + 1
        print(f'Test cases {tcname} is FAILED')

def testaccuracy():
    global total_TC
    global passed_TC
    global failed_TC

    with open("reference_data.json") as user_file:
        file_contents = user_file.read()
    json_contents = json.loads(file_contents)

    for item in json_contents:
        run_TC(item, json_contents[item], json_contents[item]["outputs"])

    print(f'Total TC: {total_TC}')
    print(f'Passed TC: {passed_TC}')
    print(f'Failed TC: {failed_TC}')
    assert total_TC == passed_TC
