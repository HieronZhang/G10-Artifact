
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


# kernel path: /tmp/torchinductor_zhang402/6e/c6eedefpjrxk2xdsvz33mwljeeat7g6dum2w7lf6cy2xjaq4fxf3.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 10
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (10*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_zhang402/kj/ckjbs74j73bpld53c6q4lophng54mc6yuf63q74isx3cffkq3tzl.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_1', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tl.store(in_out_ptr0 + (x0), tmp4, None)
''')


# kernel path: /tmp/torchinductor_zhang402/u2/cu27cawhczr3z5zxh5ritn667ahf2n455nfk2l4oqtox2rnqq57f.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_7, relu, relu_1, permute_3, permute_7, tangents_1 = args
    args.clear()
    assert_size_stride(primals_7, (64, 784), (784, 1))
    assert_size_stride(relu, (64, 512), (512, 1))
    assert_size_stride(relu_1, (64, 512), (512, 1))
    assert_size_stride(permute_3, (10, 512), (512, 1))
    assert_size_stride(permute_7, (512, 512), (512, 1))
    assert_size_stride(tangents_1, (64, 10), (10, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((64, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_3, out=buf0)
        del permute_3
        buf1 = empty((10, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (10, 64), (1, 10), 0), relu_1, out=buf1)
        buf2 = empty((1, 10), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_cuda_stream(0)
        triton_per_fused_sum_0.run(tangents_1, buf2, 10, 64, grid=grid(10), stream=stream0)
        del tangents_1
        buf3 = buf0; del buf0  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_1.run(buf3, relu_1, 32768, grid=grid(32768), stream=stream0)
        del relu_1
        buf4 = empty((64, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf3, permute_7, out=buf4)
        del permute_7
        buf5 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf3, (512, 64), (1, 512), 0), relu, out=buf5)
        buf6 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_2.run(buf3, buf6, 512, 64, grid=grid(512), stream=stream0)
        del buf3
        buf7 = buf4; del buf4  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_1.run(buf7, relu, 32768, grid=grid(32768), stream=stream0)
        del relu
        buf8 = empty((512, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf7, (512, 64), (1, 512), 0), primals_7, out=buf8)
        del primals_7
        buf9 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_2.run(buf7, buf9, 512, 64, grid=grid(512), stream=stream0)
        return (reinterpret_tensor(buf8, (512, 784), (784, 1), 0), reinterpret_tensor(buf9, (512, ), (1, ), 0), reinterpret_tensor(buf5, (512, 512), (512, 1), 0), reinterpret_tensor(buf6, (512, ), (1, ), 0), reinterpret_tensor(buf1, (10, 512), (512, 1), 0), reinterpret_tensor(buf2, (10, ), (1, ), 0), None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_7 = rand_strided((64, 784), (784, 1), device='cuda:0', dtype=torch.float32)
    relu = rand_strided((64, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    relu_1 = rand_strided((64, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_3 = rand_strided((10, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_7 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((64, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_7, relu, relu_1, permute_3, permute_7, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
