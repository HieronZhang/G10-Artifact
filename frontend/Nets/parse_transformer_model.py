#!/usr/bin/env python3

from typing import Dict, List, Optional, Tuple
import numpy as np
import ujson as json
import sys
import re

BATCH_CODE = 99
SEQ_LEN_CODE = 66
SINGLE_IPU = False

SHOULD_IGNORE_OP_TYPE = {
    # ignore these op types
    "Result": True,
    "Reshape": True,
    "Convert": True,

    "Broadcast": True,
    "Concat": True,

    # do not ignore these op types    
    "Slice": True,

    "Dot": False,
    "Relu": False,
    "Add": False,
    "BatchMatMul": False,
    "Divide": False,
    "GatherV2": False,
    "Multiply": False,
    "Power": False,
    "SoftmaxBasic": False,
    "Sqrt": False,
    "Subtract": False,
    "Sum": False,
    "Tanh": False,
    "Convolution": False,
    "MaxPool": False,
    "Erf": False,
    "Maximum": False,
}

OP_TYPE_ELEM_ONE_INPUT = {
    "Relu",
    "Sqrt",
    "Tanh",
    "Erf",
}

OP_TYPE_ELEM_TWO_INPUTS = {
    "Add",
    "Divide",
    "Multiply",
    "Power",
    "Subtract",
    "Maximum",
}

OP_TYPE_REDUCE = {
    "Sum",
}

OP_TYPE_GATHER_V2 = {
    "GatherV2",
}

OP_TYPE_ID_REDUCE = 0
OP_TYPE_ID_RELU = 1
OP_TYPE_ID_ELEMENT = 2
OP_TYPE_ID_POOL = 3
OP_TYPE_ID_CONV = 4
OP_TYPE_ID_MATMUL = 5
OP_TYPE_ID_GATHER = 6
OP_TYPE_ID_BROADCAST = 7
OP_TYPE_ID_SLICE = 8
OP_TYPE_ID_CONCAT = 9

OP_TYPE_TO_TYPE_ID = {
    "Dot":          OP_TYPE_ID_MATMUL,
    "Relu":         OP_TYPE_ID_RELU,
    "Add":          OP_TYPE_ID_ELEMENT,
    "BatchMatMul":  OP_TYPE_ID_MATMUL,
    "Divide":       OP_TYPE_ID_ELEMENT,
    "GatherV2":     OP_TYPE_ID_GATHER,
    "Multiply":     OP_TYPE_ID_ELEMENT,
    "Power":        OP_TYPE_ID_ELEMENT,
    "SoftmaxBasic": OP_TYPE_ID_ELEMENT,
    "Sqrt":         OP_TYPE_ID_RELU,
    "Subtract":     OP_TYPE_ID_ELEMENT,
    "Sum":          OP_TYPE_ID_REDUCE,
    "Tanh":         OP_TYPE_ID_RELU,
    "Convolution":  OP_TYPE_ID_CONV,
    "MaxPool":      OP_TYPE_ID_POOL,
    "Erf":          OP_TYPE_ID_RELU,
    "Broadcast":    OP_TYPE_ID_BROADCAST,
    "Slice":        OP_TYPE_ID_SLICE,
    "Concat":       OP_TYPE_ID_CONCAT,
    "Maximum":      OP_TYPE_ID_ELEMENT,
}

broadcast_dict: Dict[int, np.ndarray] = {}   # key: op_id, value: list of broadcasted dims
parameter_list: List[int] = []

class Operator:
    def __init__(self,
                 id: int,
                 ins_str: str,
                 name: str,
                 inputs: List[List[int]],
                 parse: bool = True):
        if not parse:
            return
        
        self.id: int = id
        self.ins_str: str = ins_str.strip()
        self.name: str = name
        
        self.parse_ins_str()

        self.inputs: List[int] = [v[0] for v in inputs]
        '''ids of operators where the inputs come from'''

        self.users: List[int] = []
        '''ids of operators that use this operator's output'''

        self.should_ignore_inputs: List[bool] = [False for i in self.inputs]

    def parse_ins_str(self):
        se = re.search("output.*\",\sinput_dict", self.ins_str)
        assert se, f"ins_str: {self.ins_str}"

        self.op_expression = self.ins_str[se.start():se.end()-13].strip()

        input_dict_str = self.ins_str[se.end()+1:]
        input_dict_str = input_dict_str[:input_dict_str.rfind(")")]
        self.input_dict: Dict[str, Dict[str, List[int]]] = json.loads(input_dict_str)

        for k, v in self.input_dict.items():
            if "dtype" in v:
                del v["dtype"]

        # print(f"op_expression: {self.op_expression}")
        # print(f"input_dict: {self.input_dict}")

    def dump_as_list(self) -> List:
        return [
            self.id,
            self.name,
            self.op_expression,
            self.input_dict,
            self.inputs,
            self.should_ignore_inputs,
        ]
    
    @staticmethod
    def from_list(l: List) -> "Operator":
        op = Operator(0, "", "", [], parse=False)
        op.id = l[0]
        op.name = l[1]
        op.op_expression = l[2]
        op.input_dict = l[3]
        op.inputs = l[4]
        op.should_ignore_inputs = l[5]
        return op


def parse_model(model_filename: str) -> List[Operator]:
    with open(model_filename, 'r') as f:
        model = json.load(f)

    operators = []
    for op in model:
        if op[2] == "Result":
            continue
        if op[2] == "Constant":
            continue
        if op[2] == "Parameter":
            parameter_list.append(op[0])
            continue
        if op[2] == "Maximum":
            op[2] = "Add"
        operators.append(Operator(op[0], op[1], op[2], op[3]))

    return operators

def get_dim_names_from_expr(op_expr: str, var_name: str) -> List[str]:
    # find "var_name[...] [=+-*/;]"
    # se = re.search(f"{var_name}\[[BNMGSCKHOWR\d,\s]*\]", op_expr)
    # se = re.search(f"{var_name}\[[A-Z\d,\s]*\]", op_expr)
    se = re.search(f"{var_name}\[[A-Z\d(\s(+\-)\s\d+)?,\s]*\]", op_expr)
    # print(se)
    assert se, f"var_name; {var_name}, op_expr: {op_expr}"
    temp = se.group()
    assert temp.startswith(var_name), f"temp: {temp}"
    
    # find "[...]" for var_name
    se = re.search("\[.*\]", temp)
    assert se, temp
    var_dim_names = se.group().strip()[1:-1].split(",")
    var_dim_names = [v.strip() for v in var_dim_names]

    return var_dim_names

def get_dims_from_op_elem_two_inputs(op: Operator) -> Tuple[List[int], List[List[List[int]]]]:
    assert op.name in OP_TYPE_ELEM_TWO_INPUTS, f"op.name: {op.name}"
    assert len(op.input_dict) == 2, f"op.input_dict: {op.input_dict}"
    return get_dims_from_op_GatherV2(op)

def get_dims_from_op_elem_one_input(op: Operator) -> Tuple[List[int], List[List[List[int]]]]:
    assert op.name in OP_TYPE_ELEM_ONE_INPUT, f"op.name: {op.name}"
    assert len(op.input_dict) == 1, f"op.input_dict: {op.input_dict}"
    return get_dims_from_op_GatherV2(op)

def get_dims_from_op_Dot(op: Operator) -> Tuple[List[int], List[List[List[int]]]]:
    assert op.name == "Dot", f"op.name: {op.name}"
    assert len(op.input_dict) == 2, f"op.input_dict: {op.input_dict}"
    return get_dims_from_op_GatherV2(op)

def get_dims_from_op_BatchMatMul(op: Operator) -> Tuple[List[int], List[List[List[int]]]]:
    assert op.name == "BatchMatMul", f"op.name: {op.name}"
    return get_dims_from_op_GatherV2(op)

def get_dims_from_op_Convolution(op: Operator) -> Tuple[List[int], List[List[List[int]]]]:
    assert op.name == "Convolution", f"op.name: {op.name}"

    op_expr = op.op_expression

    #################
    # !!! too lazy to do any useful check here, just hardcode everything
    #################

    dilated_factor = 1
    se_HO = re.search("(HO\s\*\s\d+)", op_expr)
    if se_HO:
        se_WO = re.search("(WO\s\*\s\d+)", op_expr)
        assert se_WO, f"se_HO: {se_HO}, se_WO: {se_WO}"
        dilated_factor = int(se_HO.group().split("*")[1].strip())

    dim_lengths: List[int] = [
        op.input_dict["input0"]["shape"][0], # batch (N)
        op.input_dict["input1"]["shape"][0], # out_chl (F)
        op.input_dict["input0"]["shape"][1], # input_chl (C)
        op.input_dict["input0"]["shape"][2] // dilated_factor, # out_hei (HO)
        op.input_dict["input0"]["shape"][3] // dilated_factor, # out_wid (WO)
        op.input_dict["input1"]["shape"][2], # ker_hei (KH)
        op.input_dict["input1"]["shape"][3], # ker_wid (KW)
    ]

    variables: List[List[List[int]]] = [ 
        [[0], [1], [3], [4]],
        [[0], [2], [3] * dilated_factor + [5], [4] * dilated_factor + [6]],
        [[2], [1], [5], [6]],
    ]

    return dim_lengths, variables

def get_dims_from_op_MaxPool(op: Operator) -> Tuple[List[int], List[List[List[int]]]]:
    assert op.name == "MaxPool", f"op.name: {op.name}"

    op_expr = op.op_expression

    #################
    # !!! too lazy to do any useful check here, just hardcode everything
    #################

    #  0:batches, 1:chl, 2:out_hei, 3:out_wid, 4:ker_hei, 5:ker_wid

    se_D0 = re.search("(D0\sin\s\d+\s*[,;])", op_expr)
    se_D1 = re.search("(D1\sin\s\d+\s*[,;])", op_expr)
    se_K0 = re.search("(K0\sin\s\d+\s*[,;])", op_expr)
    se_K1 = re.search("(K1\sin\s\d+\s*[,;])", op_expr)
    assert se_D0 and se_D1 and se_K0 and se_K1, f"se_D0: {se_D0}, se_D1: {se_D1}, se_K0: {se_K0}, se_K1: {se_K1}"

    D0 = int(re.split("[,;]", se_D0.group().split("in")[1].strip())[0].strip())
    D1 = int(re.split("[,;]", se_D1.group().split("in")[1].strip())[0].strip())
    K0 = int(re.split("[,;]", se_K0.group().split("in")[1].strip())[0].strip())
    K1 = int(re.split("[,;]", se_K1.group().split("in")[1].strip())[0].strip())

    pool_factor = op.input_dict["input0"]["shape"][2] // D0

    dim_lengths: List[int] = [
        op.input_dict["input0"]["shape"][0], # N
        op.input_dict["input0"]["shape"][1], # C
        D0, # D0
        D1, # D1
        K0, # K0
        K1, # K1
    ]

    variables: List[List[List[int]]] = [ 
        [[0], [1], [2],     [3]],
        [[0], [1], [2] * pool_factor + [4], [3] * pool_factor + [5]],
    ]

    return dim_lengths, variables

def get_dims_from_op_Reduce(op: Operator) -> Tuple[List[int], List[List[List[int]]]]:
    assert op.name in OP_TYPE_REDUCE, f"op.name: {op.name}"
    return get_dims_from_op_GatherV2(op)

def get_dims_from_op_SoftmaxBasic(op: Operator) -> Tuple[List[int], List[List[List[int]]]]:
    assert op.name == "SoftmaxBasic", f"op.name: {op.name}"
    return get_dims_from_op_GatherV2(op)

def get_dims_from_op_GatherV2(op: Operator) -> Tuple[List[int], List[List[List[int]]]]:
    # assert op.name == "GatherV2", f"op.name: {op.name}"

    op_expr = op.op_expression
    output_dim_names = get_dim_names_from_expr(op_expr, "output0")
    input0_dim_names = get_dim_names_from_expr(op_expr, "input0")
    input1_dim_names = []
    input2_dim_names = []
    if len(op.input_dict) >= 2:
        input1_dim_names = get_dim_names_from_expr(op_expr, "input1")
    if len(op.input_dict) >= 3:
        input2_dim_names = get_dim_names_from_expr(op_expr, "input2")

    variables = [[],[]]
    if len(op.input_dict) >= 2:
        variables.append([])
    if len(op.input_dict) >= 3:
        variables.append([])
    name_idx_dict = {}
    cur_idx = 0
    for out_name in output_dim_names:
        variables[0].append([cur_idx])
        name_idx_dict[out_name] = cur_idx
        cur_idx += 1
        # order of output_dim_names cannot be changed!

    if len(op.input_dict) >= 3:
        for in2_name in input2_dim_names:
            if in2_name not in name_idx_dict:
                variables[3].append([cur_idx])
                name_idx_dict[in2_name] = cur_idx
                cur_idx += 1
            else:
                variables[3].append([name_idx_dict[in2_name]])
    
    if len(op.input_dict) >= 2:
        for in1_name in input1_dim_names:
            if in1_name not in name_idx_dict:
                variables[2].append([cur_idx])
                name_idx_dict[in1_name] = cur_idx
                cur_idx += 1
            else:
                variables[2].append([name_idx_dict[in1_name]])

    for in0_name in input0_dim_names:
        if in0_name not in name_idx_dict:
            variables[1].append([cur_idx])
            name_idx_dict[in0_name] = cur_idx
            cur_idx += 1
        else:
            variables[1].append([name_idx_dict[in0_name]])

    dim_lengths = [0] * cur_idx

    if len(op.input_dict) >= 3:
        for name, length in zip(input2_dim_names, op.input_dict["input2"]["shape"]):
            dim_lengths[name_idx_dict[name]] = length
    if len(op.input_dict) >= 2:
        for name, length in zip(input1_dim_names, op.input_dict["input1"]["shape"]):
            dim_lengths[name_idx_dict[name]] = length
    for name, length in zip(input0_dim_names, op.input_dict["input0"]["shape"]):
        dim_lengths[name_idx_dict[name]] = length
    return dim_lengths, variables

# PLACE HOLDER
def get_dims_from_op_Broadcast(op: Operator) -> Tuple[List[int], List[List[List[int]]]]:
    return get_dims_from_op_GatherV2(op)

# PLACE HOLDER
def get_dims_from_op_Slice(op: Operator) -> Tuple[List[int], List[List[List[int]]]]:
    dim_lengths, variables = get_dims_from_op_GatherV2(op)
    idx = np.array(dim_lengths) > 0
    dim_lengths = np.array(dim_lengths)[idx].tolist()
    for var in variables:
        for dim in var:
            if dim[0] >= np.count_nonzero(idx==True):
                dim[0] -= np.count_nonzero(idx==False)
    return dim_lengths, variables

def get_dims_from_op_Concat(op: Operator) -> Tuple[List[int], List[List[List[int]]]]:
    dim_lengths, variables = get_dims_from_op_GatherV2(op)
    idx = list.index(dim_lengths, 0)
    dim_lengths[idx] = sum(dim_lengths[len(variables[0]):])
    return dim_lengths, variables

def get_dims_from_op(op: Operator) -> Tuple[List[int], List[List[List[int]]]]:
    '''
    @returns a tuple of:
        dim_lengths: List[int],
        variables: List[List[List[int]]]
    '''
    if op.name in OP_TYPE_ELEM_ONE_INPUT:
        return get_dims_from_op_elem_one_input(op)
    elif op.name in OP_TYPE_ELEM_TWO_INPUTS:
        return get_dims_from_op_elem_two_inputs(op)
    elif op.name in OP_TYPE_REDUCE:
        return get_dims_from_op_Reduce(op)
    elif op.name in OP_TYPE_GATHER_V2:
        return get_dims_from_op_GatherV2(op)
    elif op.name == "Dot":
        return get_dims_from_op_Dot(op)
    elif op.name == "BatchMatMul":
        return get_dims_from_op_BatchMatMul(op)
    elif op.name == "Convolution":
        return get_dims_from_op_Convolution(op)
    elif op.name == "MaxPool":
        return get_dims_from_op_MaxPool(op)
    elif op.name == "SoftmaxBasic":
        return get_dims_from_op_SoftmaxBasic(op)
    elif op.name == "GatherV2":
        return get_dims_from_op_GatherV2(op)
    elif op.name == "Broadcast":
        return get_dims_from_op_Broadcast(op)
    elif op.name == "Slice":
        return get_dims_from_op_Slice(op)
    elif op.name == "Concat":
        return get_dims_from_op_Concat(op)
    else:
        raise ValueError(f"op.name: {op.name}")

def get_tensor_expr_info_from_op(op: Operator) -> Tuple[str, List[int], List[List[List[int]]], List[bool], int, int, List[int]]:
    '''
    @returns a tuple of:
        op_type_name: str,
        dim_lengths: List[int],
        variables: List[List[List[int]]],
        ignore_variables: List[bool],
        op_type_id: int,
    '''
    op_type_name = op.name
    assert not SHOULD_IGNORE_OP_TYPE[op_type_name], f"op_type_name: {op_type_name}"
    dim_lengths, variables = get_dims_from_op(op)
    ignore_variables: List[bool] = op.should_ignore_inputs
    op_id = op.id
    op_inputs = op.inputs

    for id, input in enumerate(op.inputs):
        var_id = id+1
        assert len(variables) > var_id, \
            f"op_type_name: {op_type_name} op.id: {op.id} \
                op.inputs: {op.inputs} variables: {variables}"
        if input in broadcast_dict:
            if broadcast_dict[input][0] < len(variables[var_id]):
                variables[var_id] = np.array(variables[var_id])[broadcast_dict[input]].tolist()
            else:
                # dim_lengths.append(1)
                # variables[var_id] = [[len(dim_lengths)-1]]
                del variables[var_id]
                del ignore_variables[id]
                del op_inputs[id]

    return op_type_name, dim_lengths, variables, ignore_variables, OP_TYPE_TO_TYPE_ID[op_type_name], op_id, op_inputs

def load_ops_from_file(fname: str) -> List[Operator]:
    with open(fname, "r") as f:
        ops = json.load(f)
    return [Operator.from_list(op) for op in ops]

###########################################################
model_filename = sys.argv[1]
seq_len = 0
if len(sys.argv) > 2:
    seq_len = int(sys.argv[2])
output_filename = f"parsed/parsed_{model_filename}"
texpre_filename = f"TExpr/TExpr_{model_filename}"

ops = parse_model(f"original/{model_filename}")
ops_id_dict: Dict[int, Operator] = {op.id: op for op in ops}
# find users of all ops
for op in ops:
    for input_id in op.inputs:
        if input_id in ops_id_dict:
            ops_id_dict[input_id].users.append(op.id)
# remove all ops that should be ignored
# propagate their input_ids and uses to the remaining ops
for op in ops:
    if not SHOULD_IGNORE_OP_TYPE[op.name]:
        continue
    # initialize broadcast_dict
    if op.name == "Broadcast":
        if op.inputs[0] not in broadcast_dict:
            dim_lengths, variables = get_dims_from_op(op)
            broadcast_dict[op.inputs[0]] = np.array(variables[1]).flatten()
    if op.name != "Concat":
    # op.inputs should be added to op.users[x].inputs
        for user_id in op.users:
            if user_id in ops_id_dict:
                user = ops_id_dict[user_id]
                cur_id = user.inputs.index(op.id)
                for input_id in op.inputs:
                    cur_id += 1
                    user.inputs.insert(cur_id, input_id)
    # op.users should be added to op.inputs[x].users
    for input_id in op.inputs:
        if input_id in ops_id_dict:
            input_op = ops_id_dict[input_id]
            cur_id = input_op.users.index(op.id)
            for user_id in op.users:
                cur_id += 1
                input_op.users.insert(cur_id, user_id)
    # op.id should be removed from op.inputs[x].users and op.users[x].inputs
    for input_id in op.inputs:
        if input_id in ops_id_dict:
            input_op = ops_id_dict[input_id]
            assert op.id in input_op.users
            input_op.users.remove(op.id)
    if op.name != "Concat":
        for user_id in op.users:
            if user_id in ops_id_dict:
                user = ops_id_dict[user_id]
                assert op.id in user.inputs
                user.inputs.remove(op.id)
    # remove this op from ops_id_dict
    del ops_id_dict[op.id]
###########################################################

####################################################################################
# remove deleted ops from ops
ops = [op for op in ops if op.id in ops_id_dict]
first_producer_dict: Dict[tuple, int] = {} # op_expr -> first producer op_id
last_user_dict: Dict[tuple, int] = {} # op_expr -> last user op_id
# to reuse the forfeited cold storage
explored_ops = [op.id for op in ops]
tensor_renaming_dict: Dict[int, int] = {} # old_id -> new_id
# for i in range(len(ops)-1):
    
#     next_op = ops[i+1]
#     assert len(next_op.inputs) == len(next_op.should_ignore_inputs), \
#         f"next_op: {next_op.dump_as_list()}"
    
#     cur_op = ops[i]
    
#     for idx, input_id in enumerate(cur_op.inputs):
#         if not cur_op.should_ignore_inputs[idx]:
#             not_reused = True

#             if input_id in ops_id_dict:
#                 last_user = ops_id_dict[input_id].users[0]
#                 for user in ops_id_dict[input_id].users:
#                     if explored_ops.index(user) > explored_ops.index(last_user):
#                         last_user = user
#                 len_last_user_expr = len(ops_id_dict[last_user].op_expression)
#                 input_expr = (ops_id_dict[input_id].ins_str, len_last_user_expr)

#             else:
#                 input_expr = None
#                 last_user = -1

#             if input_expr in last_user_dict:
#                 if explored_ops.index(last_user_dict[input_expr]) < \
#                     explored_ops.index(input_id):
                    
#                     # cold store slot forfeited, can be reused
#                     not_reused = False
#                     tensor_renaming_dict[input_id] = first_producer_dict[input_expr]
#                     last_user_dict[input_expr] = last_user
#             if not_reused:
#                 if input_expr:
#                     first_producer_dict[input_expr] = input_id
#                     last_user_dict[input_expr] = last_user
    
#     for idx, input_id in enumerate(next_op.inputs):
#         if input_id == cur_op.id:
#             next_op.should_ignore_inputs[idx] = True
#         if input_id in cur_op.inputs:
#             next_op.should_ignore_inputs[idx] = True
####################################################################################

ops = [op for op in ops if op.name!="Concat"]
for op in ops:
    if op.id in tensor_renaming_dict:
        op.id = tensor_renaming_dict[op.id]
    for idx, input_id in enumerate(op.inputs):
        if input_id in tensor_renaming_dict:
            op.inputs[idx] = tensor_renaming_dict[input_id]
    op.users = []

if SINGLE_IPU:
    print("Assuming SINGLE_IPU!")
    stored_inputs = []
    for op in ops:
        for idx, input_id in enumerate(op.inputs):
            if not op.should_ignore_inputs[idx]:
                if input_id not in stored_inputs:
                    stored_inputs.append(input_id)
                else:
                    op.should_ignore_inputs[idx] = True

# parameter_list.sort()
# print(len(parameter_list))
# for p in parameter_list:
#     print(p)

print(max(parameter_list))

for op in ops:
    print(op.id)
    print(op.name)
    #print(op.op_expression)
    print(op.input_dict.__len__())
    for idx, num in enumerate(op.inputs):
        if op.name == "GatherV2" and idx == 1:
            num += 666666
        print(num)
    for i in range(op.input_dict.__len__()):
        string1 = "input" + str(i)
        print(op.input_dict[string1]["shape"].__len__())
        for dim in op.input_dict[string1]["shape"]:
            if seq_len > 0:
                if dim == SEQ_LEN_CODE:
                    dim = seq_len
                if dim == BATCH_CODE*SEQ_LEN_CODE:
                    dim = BATCH_CODE*seq_len
            print(dim)
    print("-------------------")

