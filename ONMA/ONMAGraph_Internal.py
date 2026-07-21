import numpy as np
import numpy
import onnx
import json
import ast
import re
import ast
import operator

from onnx.helper import (
    make_node, make_graph, make_model, make_tensor_value_info)
from onnx.numpy_helper import from_array
from onnx.checker import check_model
from onnxruntime import InferenceSession
from onnx import (
    AttributeProto,
    numpy_helper)

# Allowed operators mapping
OPERATORS = {
    ast.And: operator.and_,
    ast.Or: operator.or_,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}

# Global variable
g_node = {}

def safe_eval(expr: str) -> bool:
    """
    Safely evaluate a boolean expression with and/or using AST parsing.
    
    :param expr: The condition string, e.g., "x > 5 and y < 10"
    :param variables: Dictionary of variable values, e.g., {"x": 7, "y": 8}
    :return: Boolean result of the expression
    """
    try:
        # Parse the expression into an AST
        tree = ast.parse(expr, mode='eval')
        return _eval_ast(tree.body)
    except Exception as e:
        raise ValueError(f"Invalid expression: {e}")

def _eval_ast(node):
    """Recursively evaluate AST nodes."""
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            return all(_eval_ast(v) for v in node.values)
        elif isinstance(node.op, ast.Or):
            return any(_eval_ast(v) for v in node.values)
    
    elif isinstance(node, ast.Compare):
        left = _eval_ast(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            right = _eval_ast(comparator)
            if not OPERATORS[type(op)](left, right):
                return False
            left = right
        return True
    
    elif isinstance(node, ast.Constant):  # Python 3.8+
        return node.value
    
    else:
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")

def substring_from_index_to_pattern(text: str, start_index: int, end_pattern: str) -> str:
    """
    Extracts a substring from `text` starting at `start_index` and ending
    just before the first occurrence of `end_pattern` after that index.

    :param text: The original string
    :param start_index: The starting index (0-based)
    :param end_pattern: The pattern marking the end of the substring
    :return: Extracted substring or an empty string if not found/invalid
    """
    # Validate inputs
    if not isinstance(text, str) or not isinstance(end_pattern, str):
        raise TypeError("Both text and end_pattern must be strings.")
    if not isinstance(start_index, int):
        raise TypeError("start_index must be an integer.")
    if start_index < 0 or start_index >= len(text):
        return ""  # Invalid start index
    if end_pattern == "":
        return ""  # Empty pattern is not valid

    # Find the end pattern after the start index
    end_pos = text.find(end_pattern, start_index)
    if end_pos == -1:
        return ""  # Pattern not found after start index

    # Return substring from start_index to just before the pattern
    return text[start_index:(end_pos+len(end_pattern))]

# Input: numpy.tile(numpy.array(C), 2)
# Output: numpy.array(C), 2
# Input: numpy.array(C)
# Output: C
def substring_from_index_to_next_close_parentheses(text: str, start_index: int, end_pattern: str) -> str:
    left_parentheses_count = 1
    right_parentheses_count = 0

    result = ""
    for i in range(start_index + 1, len(text)):
        if text[i] == '(':
            left_parentheses_count += 1
        elif text[i] == ')':
            right_parentheses_count += 1

        if left_parentheses_count == right_parentheses_count:
            result = text[start_index+1:i]
            break
    
    return result

def substring_from_index_to_next_open_close_parentheses(text: str, start_index: int, end_pattern: str) -> str:
    left_parentheses_count = 0
    right_parentheses_count = 0

    result = ""
    for i in range(start_index, len(text)):
        if text[i] == '(':
            left_parentheses_count += 1
        elif text[i] == ')':
            right_parentheses_count += 1

        if text[i] == '(' or text[i] == ')':
            if left_parentheses_count == right_parentheses_count:
                result = text[start_index:i+1]
                break

    return result

def get_attribute_value(attr: AttributeProto):
    """
    Convert an AttributeProto to a Python value based on its type.
    """
    if attr.type == AttributeProto.FLOAT:
        return [attr.f]
    elif attr.type == AttributeProto.INT:
        return [attr.i]
    elif attr.type == AttributeProto.STRING:
        return attr.s.decode("utf-8", errors="ignore")
    elif attr.type == AttributeProto.TENSOR:
        return onnx.numpy_helper.to_array(attr.t)
    elif attr.type == AttributeProto.FLOATS:
        return list(attr.floats)
    elif attr.type == AttributeProto.INTS:
        return list(attr.ints)
    elif attr.type == AttributeProto.STRINGS:
        return [s.decode("utf-8", errors="ignore") for s in attr.strings]
    else:
        return None  # Unsupported or empty attribute

def cut_substring(text, start_char, end_char):
    try:
        # Find the first occurrence of start_char and end_char
        start_index = text.index(start_char) + 1  # +1 to exclude start_char
        end_index = text.index(end_char, start_index)  # search after start_char
        
        return text[start_index:end_index]
    except ValueError:
        # If start_char or end_char not found
        return None

def NumpyDataTypeFromTensor(type):
    if type == 1: return "float32"
    elif type == 2: return "uint8"
    elif type == 3: return "int8"
    elif type == 4: return "uint16"
    elif type == 5: return "int16"
    elif type == 6: return "int32"
    elif type == 7: return "int64"
    elif type == 8: return "object"
    elif type == 9: return "bool"
    elif type == 10: return "float16"
    elif type == 11: return "float64"
    elif type == 12: return "uint32"
    elif type == 13: return "uint64"
    else: return "float32"

def GetTensorDataTypeFromnp(npdtype):
    datatype = onnx.TensorProto.FLOAT
    if npdtype == "float32":
        datatype = onnx.TensorProto.FLOAT
    elif npdtype == "float64":
        datatype = onnx.TensorProto.DOUBLE
    elif npdtype == "int64":
        datatype = onnx.TensorProto.INT64
    elif npdtype == "uint16":
        datatype = onnx.TensorProto.UINT16
    elif npdtype == "bool":
        datatype = onnx.TensorProto.BOOL
    elif npdtype == "uint8":
        datatype = onnx.TensorProto.UINT8
    elif npdtype == "int32":
        datatype = onnx.TensorProto.INT32
    elif npdtype == "object":
        datatype = onnx.TensorProto.STRING
    elif npdtype == "uint32":
        datatype = onnx.TensorProto.UINT32
    elif npdtype == "uint64":
        datatype = onnx.TensorProto.UINT64
    elif npdtype == "int8":
        datatype = onnx.TensorProto.INT8
    elif npdtype == "int16":
        datatype = onnx.TensorProto.INT16
    elif npdtype == "float16":
        datatype = onnx.TensorProto.FLOAT16
    return datatype

def CheckValueInfor(graph, name):
    for value in graph.input:
        if value.name == name:
            return True
    
    for value in graph.output:
        if value.name == name:
            return True

    for vi in graph.value_info:
        if vi.name == name:
            return True
    for initializer in graph.initializer:
        if initializer.name == name:
            return True
    return False

def CreateInitializerTensor(
        name: str,
        tensor_array: np.ndarray,
        data_type: onnx.TensorProto = onnx.TensorProto.FLOAT
) -> onnx.TensorProto:
    initializer_tensor = onnx.helper.make_tensor(
        name=name,
        data_type=data_type,
        dims=tensor_array.shape,
        vals=tensor_array.flatten().tolist()
    )
    return initializer_tensor

def GetInitializerByName(initializers, name):
    result = None
    index = -1
    for i, initializer in enumerate(initializers):
        if initializer.name == name:
            result = initializer
            index = i
    return index, result

def GetPreviousNodeFromInputName(graph, inputs):
    node_res = []
    node_ind = []
    for i, node in enumerate(graph.node):
        for input in inputs:
            if input in node.output:
                node_res.append(node)
                node_ind.append(i)
    return node_ind, node_res

patterns_replacement = {
    # "NextNode(node)": "NextNode(graph, node)"
    # Start - End - ReplaceBy
    "{node.attribute": {"endWith": "]}", "function": "GetNodeAttributeValue(node, data)"},
    "IsInitializer": {"endWith": ")", "function": "IsInitializer(graph, node_io_value)"},
    "IsScalar": {"endWith": ")", "function": "IsScalar(graph, node_io_value)"},
    "GetShape": {"endWith": ")", "function": "GetShape(graph, function_pattern)"},
    "GetDataType": {"endWith": ")", "function": "GetDataType(graph, function_pattern)"},
    "CheckInitializer": {"endWith": ")", "function": "CheckInitializer(graph, function_pattern)"},
    "CheckInput": {"endWith": ")", "function": "CheckInput(graph, function_pattern)"},
    "numpy": {"endWith": ")", "function": "NumpyProcessing(graph, data)"}
}

def CheckInput(graph, function_pattern):
    # print(f'CheckInput: {function_pattern}')
    argument = function_pattern.replace("CheckInput", "")
    argument = argument.replace("(", "")
    argument = argument.replace(")", "")

    index, _ = GetInitializerByName(graph.initializer, argument)
    if index == -1: return argument
    return None

# CheckInitializer(neg_one)
def CheckInitializer(graph, function_pattern):
    # print(f'CheckInitializer: {function_pattern}')
    argument = function_pattern.replace("CheckInitializer", "")
    argument = argument.replace("(", "")
    argument = argument.replace(")", "")

    index, _ = GetInitializerByName(graph.initializer, argument)
    if index == -1: return None
    return argument

# check initializer is scalar
# single value is scalar, array with 1 item also scalar
def IsScalar(graph, node_io_value):
    for initializer in graph.initializer:
        if initializer.name == node_io_value:
            try:
                arr = numpy_helper.to_array(initializer)
                if arr.size == 1:
                    return True
            except:
                pass
    return False

def GetDataType(graph, function_pattern):
    print(f"GetDataType: {function_pattern}")

    argument = function_pattern.replace("GetDataType", "")
    argument = argument.replace("(", "")
    argument = argument.replace(")", "") # GetDataType(X)

    print(f"argument: {argument}")

    data_type = None
    for value in graph.input:
        if value.name == argument:
            data_type = NumpyDataTypeFromTensor(value.type.tensor_type.elem_type)
    
    for value in graph.output:
        if value.name == argument:
            data_type = NumpyDataTypeFromTensor(value.type.tensor_type.elem_type)

    for vi in graph.value_info:
        if vi.name == argument:
            data_type = NumpyDataTypeFromTensor(vi.type.tensor_type.elem_type)
    
    for initializer in graph.initializer:
        if initializer.name == argument:
            data_type = NumpyDataTypeFromTensor(initializer.data_type)

    return data_type


def GetShape(graph, data):
    print(f"GetShape: {data}")

    index = None

    argument = data.replace("GetShape", "")
    argument = argument.replace("(", "")
    argument = argument.replace(")", "") # Conv_Node_output[-3]
    argument_name = argument.split("[")[0] # Conv_Node_output
    index = cut_substring(argument, "[", "]") # -3

    print(f"argument: {argument} - argument_name: {argument_name} - index: {index}")

    shape = []
    for vi in graph.value_info:
        if vi.name == argument_name:
            shape = [dim.dim_value for dim in vi.type.tensor_type.shape.dim]
    
    if index is not None:
        shape = shape[int(index)]

    return shape

# Execute numpy function
# Input: numpy.tile(numpy.array(C), 2)
def NumpyProcessing(graph, data):
    print(f"This is numpy data: {data}")

    pattern = re.compile("\\(")
    matches = re.finditer(pattern, data)

    list_of_initializer_name = []
    for match in matches:
        initializer_names = substring_from_index_to_next_close_parentheses(data, match.start(), ")")
        initializer_names = initializer_names.replace(" ", "")
        initializer_names = initializer_names.split(",")
        for initializer_name in initializer_names:
            if initializer_name not in list_of_initializer_name:
                list_of_initializer_name.append(initializer_name)
                # print(f"argument: {initializer_name}")
                index, initializer_data = GetInitializerByName(graph.initializer, initializer_name)
                if index != -1:
                    # print(f"initializer_name: {initializer_name}")
                    arr = numpy_helper.to_array(initializer_data).tolist()
                    data = data.replace(initializer_name, str(arr))
                    # print(f"arr: {data}")

    try:
        arr_final = eval(data)
        return arr_final.tolist()
    except Exception as e:
        print(f"Error occurred while evaluating numpy data: {e}")
        return None

# check initializer
def IsInitializer(graph, node_io_value):
    for initializer in graph.initializer:
        if initializer.name == node_io_value:
            return True
    return False

# data: "{node.attribute[axis]}"
def GetNodeAttributeValue(node, data):
    print(f'{node.name} - {data}')
    attributeName = data.split("[")[1]
    attributeName = attributeName.split("]")[0]
    
    for attr in node.attribute:
        if attr.name == attributeName:
            value = get_attribute_value(attr)

    return value

def NextNode(graph, node):
    # Collect outputs of the target node
    target_outputs = set(node.output)

    for target_node in graph.node:
        if any(inp in target_outputs for inp in target_node.input):
            return target_node
    return None

# "{node.output[0]}" -> "C" 
# "{add_node.name}" -> "Conv_Node"
def GetValueFromVariable(g_node, one_input):
    one_input = one_input.replace("{", "")
    one_input = one_input.replace("}", "")
    one_input_split = one_input.split(".")
    node_var = one_input_split[0] # "node"
    
    val = ""
    if "input" in one_input_split[1] or "output" in one_input_split[1]:
        node_io = one_input_split[1].split("[")[0] # "output"
        node_io_idx = one_input_split[1].split("[")[1] # "0]"
        node_io_idx = node_io_idx.replace("]", "")
        try:
            node_io_value = getattr(g_node[node_var], node_io)
            node_io_value = node_io_value[int(node_io_idx)]
            val = node_io_value
        except:
            val = None
    else:
        node_var_next = one_input_split[1] # add_node.name -> name
        val = getattr(g_node[node_var], node_var_next)

    return val

# one_input:
#           "{node.output[0]}"
#           "IsInitializer({add_node.input[1]}) and IsScalar({add_node.input[1]})"
# Case 1: only in/out connection "{node.output[0]}"
# Case 2: condition inside "IsInitializer({add_node.input[1]})"
# Case 3: combination of 1&2 "{node.output[0]} and IsInitializer({add_node.input[1]})"
def CheckIOCondition(graph, g_node, one_input):
    print(f"CheckIOCondition: {one_input}")

    # Detect and replace node_io_var to value "{node.output[0]}" -> "C"
    pattern = re.compile("{")
    matches = re.finditer(pattern, one_input)

    pair_of_node_io_var = {} # use to detect old node that are processed
    for match in matches:
        node_io_var = substring_from_index_to_pattern(one_input, match.start(), "}")
        node_io_value = GetValueFromVariable(g_node, node_io_var)
        # print(f"node_io_var: {node_io_var}")
        # print(f"node_io_value: {node_io_value}")

        if node_io_var not in pair_of_node_io_var:
            pair_of_node_io_var[node_io_var] = node_io_value

        if CheckValueInfor(graph, node_io_value) is False: # input or output is not available
            # print(f"CheckValueInfor: {node_io_value} is not available in graph")
            return False

    # Refine input. For ex:
    # From: (IsInitializer({add_node.input[1]}) == True) and (IsScalar({add_node.input[1]}) == True)
    # To: (IsInitializer(C) == True) and (IsScalar(C) == True)

    # From: {node.output[0]}
    # To: True
    for one_node_io_var in pair_of_node_io_var:
        # Replace var inside function: IsInitializer({add_node.input[1]}) -> IsInitializer(C)
        one_input = one_input.replace(f'({one_node_io_var})', f'({pair_of_node_io_var[one_node_io_var]})')

        # Replace var by value
        # one_input = one_input.replace(one_node_io_var, "True")
        one_input = one_input.replace(f'{one_node_io_var}', f'"{pair_of_node_io_var[one_node_io_var]}"')

    print(f'refine input: {one_input}')

    # Excute and update result of pattern.
    # For example: IsInitializer(C) -> True
    pair_of_function_and_result = {}
    for item in patterns_replacement:
        if item in one_input:
            # print(item)
            pattern = re.compile(item)
            matches = re.finditer(pattern, one_input)
            for match in matches:
                # print(match)
                function_pattern = substring_from_index_to_pattern(one_input, match.start(), ")")
                node_io_value = cut_substring(function_pattern, "(", ")")
                status = eval(patterns_replacement[item]["function"])
                print(f'function_pattern: {function_pattern}, node_io_value: {node_io_value}, status: {status}')
                if function_pattern not in pair_of_function_and_result:
                    pair_of_function_and_result[function_pattern] = status
                # one_input = one_input.replace(function_pattern, str(status))

    for one_function in pair_of_function_and_result:
        one_input = one_input.replace(one_function, str(pair_of_function_and_result[one_function]))
    print(f'refine input last: {one_input}')

    result = safe_eval(one_input)
    print(f"Condition '{one_input}' evaluates to: {result}")

    return result

# The condition is the set
# {
#     "var": "add_node",
#     "op_type": "Add",
#     "inputs": [
#         "{node.output[0]}",
#         "IsInitializer({add_node.input[1]}) and IsScalar({add_node.input[1]})"
#     ]
# }
def checkOneCondition(graph, node, item):
    # print(f'node: {node}')
    # print(f'item: {item}')
    final_status = False
    status = True

    # Check for main node
    if node.op_type != item["op_type"]: status = False
    else: # in case of other condition
        if g_node == {}: g_node["node"] = node
        
        # Setting global variable
        if "var" in item:
            g_node[item["var"]] = node

        # Check input
        if "inputs" in item:
            for one_input in item["inputs"]:
                # "{node.output[0]}"
                status = CheckIOCondition(graph, g_node, one_input)
                if status == False:
                    print(f"The condition: {one_input} is NG")
                    break
                else:
                    print(f"The condition: {one_input} is OK")

    final_status = status
    if final_status == True: print("This is True")
    else: print("This is False")
    return final_status

def checkConditionOfSearchBy(graph, pattern):
    status = False
    target_node = None
    index = -1

    # Get nodes list
    nodes_list = pattern["SearchBy"]["nodes"]

    # Get first pattern, this is target node for replacement
    first_pattern = next(iter(nodes_list))

    index = -1
    for item in nodes_list:
        for i, node in enumerate(graph.node):
            if node.op_type == item["op_type"]:
                status = checkOneCondition(graph, node, item)
                if status == True:
                    if index == -1:
                        g_node["node"] = node
                        target_node = node
                        index = i
                        g_node["node"] = node
                    break
                # else:
                #     return status, target_node, index

    return status, target_node, index

def getNodeAttribute(node, attri):
    print(attri)

# refine graph variables by value, function by data
def refineStringInReplaceBy(g_node, node, index, data):
    data = data.replace('{node.name}', node.name)

    if '{node.index}' in data:
        data = data.replace('{node.index}', str(index))
    
    # Replace variable by value
    # Detect and replace node_io_var to value "{node.output[0]}" -> "C"
    pattern = re.compile("{")
    matches = re.finditer(pattern, data)

    pair_of_node_io_var = {} # use to detect old node that are processed
    for match in matches:
        node_io_var = substring_from_index_to_pattern(data, match.start(), "}")
        if ":" in node_io_var or '"{' in node_io_var or ".attribute" in node_io_var:
            pass
        else:
            node_io_value = GetValueFromVariable(g_node, node_io_var)
            # print(f'node_io_value: {node_io_value}')

            if node_io_var not in pair_of_node_io_var and node_io_value is not None:
                pair_of_node_io_var[node_io_var] = node_io_value

    for one_node_io_var in pair_of_node_io_var:
        # Replace var inside function
        data = data.replace(one_node_io_var, pair_of_node_io_var[one_node_io_var])

    return data

# Execute function
def ExecuteFunction(graph, node, data):
    result = None
    for item in patterns_replacement:
        if item in data:
            # print(f"ExecuteFunction: {item} - {data}")
            try:
                if "numpy" in item: # Block processing for numpy function
                    result = eval(patterns_replacement[item]["function"])
                else:
                    pattern = re.compile(item)
                    matches = re.finditer(pattern, data)
                    for match in matches:
                        function_pattern = substring_from_index_to_next_open_close_parentheses(data, match.start(), ")")
                        result = eval(patterns_replacement[item]["function"])
                        # print(f'function_pattern: {function_pattern} - result: {result}')
                        data = data.replace(function_pattern, str(result))
                        print(f"Update: {data} - result: {result}")

            except Exception as e:
                print(f"Error occurred while executing function for item '{item}': {e}")
    print(f"ExecuteFunction: {data} - result: {result}")
    return result

def UpdateGraphUsingPattern(graph, pattern):
    decompose_pattern = {}
    # Confirm condition
    status, node, index = checkConditionOfSearchBy(graph, pattern)
    if status is False: return

    # Need to refine graph variables by value, function by data
    # For example:
    # From: {'name': 'rs_axes', 'data': 'numpy.array({add_node.input[1]})', 'data_type': 'float32'}
    # To: {'name': 'rs_axes', 'data': 'numpy.array(C)', 'data_type': 'float32'}

    # From: {'name': '{node.name}', 'Action': 'Modify', 'inputs': ['{node.input[0]}', '{node.input[1]}', 'rs_axes'], 'outputs': ['{add_node.output[0]}']}
    # To: {'name': 'Conv_Node', 'Action': 'Modify', 'inputs': ['X', 'W', 'rs_axes'], 'outputs': ['Y']}, {'name': 'Add_Node', 'Action': 'Remove'}
    if "graph" in pattern["ReplaceBy"]:
        if "initializers" in pattern["ReplaceBy"]["graph"]:
            for initializer in pattern["ReplaceBy"]["graph"]["initializers"]:
                print(f'initializer: {initializer}')
                for item in initializer:
                    if isinstance(initializer[item], str):
                        print(f'item: {initializer[item]}')
                        refinestring = refineStringInReplaceBy(g_node, node, index, initializer[item])
                        # print(f'item: {refinestring}')
                        # Support: numpy.add(B, Conv_Node_output) or numpy.add(B, C)
                        refinestring = refinestring.replace(" ", "")
                        function_list = refinestring.split("or")
                        print(f'function_list: {function_list}')
                        for one_function in function_list:
                            print(f'DEBUG one_function: {one_function}')
                            result = ExecuteFunction(graph, node, one_function)
                            if result is None:
                                initializer[item] = one_function
                            else:
                                initializer[item] = result
                                break

        if "nodes" in pattern["ReplaceBy"]["graph"]:
            for node_dic in pattern["ReplaceBy"]["graph"]["nodes"]:
                # print(f'node: {node_dic}')
                for item in node_dic:
                    # print(f'item: {node_dic[item]}')
                    if 'inputs' in item or 'outputs' in item:
                        # print(f'item key: {item}')
                        refine_input = []
                        for io in node_dic[item]:
                            refinestring = refineStringInReplaceBy(g_node, node, index, io)
                            # print(f'io: {io} - {refinestring}')
                            function_list = refinestring.split("or")
                            for one_function in function_list:
                                print(f'one_function: {one_function}')
                                result = ExecuteFunction(graph, node, one_function)
                                if result is not None:
                                    refine_input.append(result)
                                    break
                            if refine_input == []: refine_input.append(refinestring)
                            # node_dic[item][io] = refinestring
                        node_dic[item] = refine_input
                    elif 'attributes' in item:
                        for attribute in node_dic[item]:
                            if isinstance(node_dic[item][attribute], str):
                                refinestring = refineStringInReplaceBy(g_node, node, index, node_dic[item][attribute])
                                result = ExecuteFunction(graph, node, refinestring)
                                print(f'attribute: {attribute} - {refinestring} - result: {result}')
                                if result is None:
                                    node_dic[item][attribute] = refinestring
                                else:
                                    node_dic[item][attribute] = result
                    else:
                        refinestring = refineStringInReplaceBy(g_node, node, index, node_dic[item])
                        node_dic[item] = refinestring

        # print(f'initializers after refine: {pattern["ReplaceBy"]["graph"]["initializers"]}')
        print(f'Nodes after refine: {pattern["ReplaceBy"]["graph"]["nodes"]}')

    decompose_pattern = pattern["ReplaceBy"]

    print(f'decompose_pattern: {decompose_pattern}')

    return decompose_pattern
