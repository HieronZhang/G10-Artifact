
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align

from torch import device, empty, empty_strided
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_zhang402/mi/cmicahfadvmqxgc6q6z43x75y4tmqh6pz37cehbc2o32fz35jkt4.py
# Source Nodes: [l__self___linear_relu_stack_1], Original ATen: [aten.relu]
# l__self___linear_relu_stack_1 => relu
triton_poi_fused_relu_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_0', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, None)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7 = args
    args.clear()
    assert_size_stride(primals_1, (512, 784), (784, 1))
    assert_size_stride(primals_2, (512, ), (1, ))
    assert_size_stride(primals_3, (512, 512), (512, 1))
    assert_size_stride(primals_4, (512, ), (1, ))
    assert_size_stride(primals_5, (10, 512), (512, 1))
    assert_size_stride(primals_6, (10, ), (1, ))
    assert_size_stride(primals_7, (64, 784), (784, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((64, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(primals_7, reinterpret_tensor(primals_1, (784, 512), (1, 784), 0), out=buf0)
        del primals_1
        buf1 = buf0; del buf0  # reuse
        # Source Nodes: [l__self___linear_relu_stack_1], Original ATen: [aten.relu]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_relu_0.run(buf1, primals_2, 32768, grid=grid(32768), stream=stream0)
        del primals_2
        buf2 = empty((64, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf1, reinterpret_tensor(primals_3, (512, 512), (1, 512), 0), out=buf2)
        buf3 = buf2; del buf2  # reuse
        # Source Nodes: [l__self___linear_relu_stack_3], Original ATen: [aten.relu]
        triton_poi_fused_relu_0.run(buf3, primals_4, 32768, grid=grid(32768), stream=stream0)
        del primals_4
        buf4 = empty((64, 10), device='cuda', dtype=torch.float32)
        # Source Nodes: [logits], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_6, buf3, reinterpret_tensor(primals_5, (512, 10), (1, 512), 0), alpha=1, beta=1, out=buf4)
        del primals_6
        return (buf4, primals_7, buf1, buf3, reinterpret_tensor(primals_5, (10, 512), (512, 1), 0), reinterpret_tensor(primals_3, (512, 512), (512, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((512, 784), (784, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((10, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, 784), (784, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
