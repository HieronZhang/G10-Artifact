
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

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7):
        permute = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        addmm = torch.ops.aten.addmm.default(primals_2, primals_7, permute);  primals_2 = permute = None
        relu = torch.ops.aten.relu.default(addmm);  addmm = None
        permute_1 = torch.ops.aten.permute.default(primals_3, [1, 0]);  primals_3 = None
        addmm_1 = torch.ops.aten.addmm.default(primals_4, relu, permute_1);  primals_4 = None
        relu_1 = torch.ops.aten.relu.default(addmm_1);  addmm_1 = None
        permute_2 = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
        addmm_2 = torch.ops.aten.addmm.default(primals_6, relu_1, permute_2);  primals_6 = None
        permute_3 = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        permute_7 = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        return [addmm_2, primals_7, relu, relu_1, permute_3, permute_7]
        
def load_args(reader):
    buf0 = reader.storage(None, 1605632, device=device(type='cuda', index=0))
    reader.tensor(buf0, (512, 784), requires_grad=True, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf1, (512,), requires_grad=True, is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf2, (512, 512), requires_grad=True, is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf3, (512,), requires_grad=True, is_leaf=True)  # primals_4
    buf4 = reader.storage(None, 20480, device=device(type='cuda', index=0))
    reader.tensor(buf4, (10, 512), requires_grad=True, is_leaf=True)  # primals_5
    buf5 = reader.storage(None, 40, device=device(type='cuda', index=0))
    reader.tensor(buf5, (10,), requires_grad=True, is_leaf=True)  # primals_6
    buf6 = reader.storage(None, 200704, device=device(type='cuda', index=0))
    reader.tensor(buf6, (64, 784), is_leaf=True)  # primals_7
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
