
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

torch._inductor.config.trace.enabled = True




isolate_fails_code_str = None



# torch version: 2.2.1+cu121
# torch cuda version: 12.1
# torch git version: 6c8c5ad5eaf47a62fafbb4a2747198cbffbf1ff0


# CUDA Info: 
# nvcc not found
# GPU Hardware Info: 
# NVIDIA A100-PCIE-40GB : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, primals_7, relu, relu_1, permute_3, permute_7, tangents_1):
        alias = torch.ops.aten.alias.default(relu)
        alias_1 = torch.ops.aten.alias.default(relu_1)
        mm = torch.ops.aten.mm.default(tangents_1, permute_3);  permute_3 = None
        permute_4 = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_1 = torch.ops.aten.mm.default(permute_4, relu_1);  permute_4 = relu_1 = None
        permute_5 = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        view = torch.ops.aten.view.default(sum_1, [10]);  sum_1 = None
        permute_6 = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
        alias_2 = torch.ops.aten.alias.default(alias_1);  alias_1 = None
        le = torch.ops.aten.le.Scalar(alias_2, 0);  alias_2 = None
        full_default = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(le, full_default, mm);  le = mm = None
        mm_2 = torch.ops.aten.mm.default(where, permute_7);  permute_7 = None
        permute_8 = torch.ops.aten.permute.default(where, [1, 0])
        mm_3 = torch.ops.aten.mm.default(permute_8, relu);  permute_8 = relu = None
        permute_9 = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(where, [0], True);  where = None
        view_1 = torch.ops.aten.view.default(sum_2, [512]);  sum_2 = None
        permute_10 = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        alias_3 = torch.ops.aten.alias.default(alias);  alias = None
        le_1 = torch.ops.aten.le.Scalar(alias_3, 0);  alias_3 = None
        where_1 = torch.ops.aten.where.self(le_1, full_default, mm_2);  le_1 = full_default = mm_2 = None
        permute_11 = torch.ops.aten.permute.default(where_1, [1, 0])
        mm_4 = torch.ops.aten.mm.default(permute_11, primals_7);  permute_11 = primals_7 = None
        permute_12 = torch.ops.aten.permute.default(mm_4, [1, 0]);  mm_4 = None
        sum_3 = torch.ops.aten.sum.dim_IntList(where_1, [0], True);  where_1 = None
        view_2 = torch.ops.aten.view.default(sum_3, [512]);  sum_3 = None
        permute_13 = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        return [permute_13, view_2, permute_10, view_1, permute_6, view, None]
        
def load_args(reader):
    buf0 = reader.storage(None, 200704, device=device(type='cuda', index=0))
    reader.tensor(buf0, (64, 784), is_leaf=True)  # primals_7
    buf1 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf1, (64, 512), is_leaf=True)  # relu
    buf2 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf2, (64, 512), is_leaf=True)  # relu_1
    buf3 = reader.storage(None, 20480, device=device(type='cuda', index=0))
    reader.tensor(buf3, (10, 512), is_leaf=True)  # permute_3
    buf4 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf4, (512, 512), is_leaf=True)  # permute_7
    buf5 = reader.storage(None, 2560, device=device(type='cuda', index=0))
    reader.tensor(buf5, (64, 10), is_leaf=True)  # tangents_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
