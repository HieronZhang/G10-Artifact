
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

from torch import device, empty_strided
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_zhang402/q3/cq3jl674vu2liwmloo46suqjsjmn3yszuenvmevus5zty75xonh3.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[1024, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_0', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1000
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1000*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, split_scan_grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream


# kernel path: /tmp/torchinductor_zhang402/y4/cy4dypgebwkqdd3wst2srv55q2ba57dj6rfrqsjqcov5yb7jvgu4.py
# Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_div_native_batch_norm_backward_threshold_backward_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_native_batch_norm_backward_threshold_backward_1', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 51200
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 2048)
    x0 = xindex % 2048
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (126*x1)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (2048*(r2 % 7)) + (14336*(((r2 + (126*x1)) // 7) % 448))), rmask & tmp2, eviction_policy='evict_first').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + (x0 + (2048*(((r2 + (126*x1)) // 49) % 64))), rmask & tmp2, eviction_policy='evict_first', other=0.0)
        tmp5 = 49.0
        tmp6 = tmp4 / tmp5
        tmp7 = 0.0
        tmp8 = tl.where(tmp3, tmp7, tmp6)
        tmp9 = tl.full(tmp8.shape, 0, tmp8.dtype)
        tmp10 = tl.where(tmp2, tmp8, tmp9)
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
        tmp14 = tl.load(in_ptr2 + (x0 + (2048*(r2 % 7)) + (14336*(((r2 + (126*x1)) // 7) % 448))), rmask & tmp2, eviction_policy='evict_first', other=0.0)
        tmp15 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2, eviction_policy='evict_last', other=0.0)
        tmp16 = tmp14 - tmp15
        tmp17 = tmp8 * tmp16
        tmp18 = tl.full(tmp17.shape, 0, tmp17.dtype)
        tmp19 = tl.where(tmp2, tmp17, tmp18)
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(rmask, tmp22, _tmp21)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, None)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp21, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/t2/ct26gn44xct5pwh7dkvbdqu7j2x62zaz2677amwbeaesffdj2q6y.py
# Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_div_native_batch_norm_backward_threshold_backward_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 32],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_native_batch_norm_backward_threshold_backward_2', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 25
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (2048*r1)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/6r/c6rispbf6ubu7xdfc47dzt7bm4clmu7nsqlavqmmdwes6hdc4sfe.py
# Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_div_native_batch_norm_backward_threshold_backward_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 32],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_native_batch_norm_backward_threshold_backward_3', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 25
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (2048*r1)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, None)
    tl.store(out_ptr0 + (x0), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/mx/cmx55suq6ff4bwjd2uqwaccmhbkwzc3yinmg5jigrtgnki6tmmu6.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_4', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 2048
    x2 = (xindex // 100352)
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x0 + (2048*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x3), None)
    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp2 = 49.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp8 = tmp6 - tmp7
    tmp10 = 0.00031887755102040814
    tmp11 = tmp9 * tmp10
    tmp13 = tmp12 * tmp12
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 * tmp14
    tmp16 = tmp5 - tmp15
    tmp18 = tmp17 * tmp10
    tmp19 = tmp16 - tmp18
    tmp21 = tmp12 * tmp20
    tmp22 = tmp19 * tmp21
    tl.store(out_ptr0 + (x3), tmp22, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/sd/csdm5air6weq7gm5yfhrnyyvo243jubgj75b5tg6lqton64bfpbl.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_5', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12800
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 512)
    x0 = xindex % 512
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (126*x1)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (512*(r2 % 7)) + (3584*(((r2 + (126*x1)) // 7) % 448))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + (x0 + (512*(r2 % 7)) + (3584*(((r2 + (126*x1)) // 7) % 448))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
        tmp13 = tl.load(in_ptr2 + (x0 + (512*(r2 % 7)) + (3584*(((r2 + (126*x1)) // 7) % 448))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tmp13 - tmp14
        tmp16 = tmp7 * tmp15
        tmp17 = tl.full(tmp16.shape, 0, tmp16.dtype)
        tmp18 = tl.where(tmp2, tmp16, tmp17)
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(rmask & xmask, tmp21, _tmp20)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp20, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/j6/cj6olwmcswpkgoyeix2b26vcusdyjmclctozvexwm72u4a7hsr46.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[512, 32],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_6', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 25
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/mc/cmckaniyzrkmmdmfhogjaso2certhk7uijnbemr2wekx7lohaslm.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[512, 32],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_7', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 25
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/vb/cvbqypqz6yhobmz73tuj2hsy2p3htyjvtwijpqd6sqiryhkgznrt.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_8', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp5 = tl.load(in_ptr1 + (x2), None)
    tmp6 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.00031887755102040814
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x2), tmp21, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/mg/cmgnry3wy7uzd7iwskxlqzqqiuelk5de75f3xqzsiyja7ki2pmgg.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_div_native_batch_norm_backward_threshold_backward_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_native_batch_norm_backward_threshold_backward_9', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 51200
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 2048)
    x0 = xindex % 2048
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (126*x1)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (2048*(r2 % 7)) + (14336*(((r2 + (126*x1)) // 7) % 448))), rmask & tmp2, eviction_policy='evict_first', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + (x0 + (2048*(r2 % 7)) + (14336*(((r2 + (126*x1)) // 7) % 448))), rmask & tmp2, eviction_policy='evict_first').to(tl.int1)
        tmp7 = tl.load(in_ptr2 + (x0 + (2048*(((r2 + (126*x1)) // 49) % 64))), rmask & tmp2, eviction_policy='evict_first', other=0.0)
        tmp8 = 49.0
        tmp9 = tmp7 / tmp8
        tmp10 = tl.where(tmp6, tmp4, tmp9)
        tmp11 = tl.load(in_ptr3 + (x0 + (2048*(r2 % 7)) + (14336*(((r2 + (126*x1)) // 7) % 448))), rmask & tmp2, eviction_policy='evict_first', other=0.0)
        tmp12 = tmp10 + tmp11
        tmp13 = tl.where(tmp5, tmp4, tmp12)
        tmp14 = tl.full(tmp13.shape, 0, tmp13.dtype)
        tmp15 = tl.where(tmp2, tmp13, tmp14)
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask, tmp18, _tmp17)
        tmp19 = tl.load(in_ptr4 + (x0 + (2048*(r2 % 7)) + (14336*(((r2 + (126*x1)) // 7) % 448))), rmask & tmp2, eviction_policy='evict_first', other=0.0)
        tmp20 = tl.load(in_ptr5 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2, eviction_policy='evict_last', other=0.0)
        tmp21 = tmp19 - tmp20
        tmp22 = tmp13 * tmp21
        tmp23 = tl.full(tmp22.shape, 0, tmp22.dtype)
        tmp24 = tl.where(tmp2, tmp22, tmp23)
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(rmask, tmp27, _tmp26)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp17, None)
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp26, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/j5/cj52cmpkeq4gre5talelfksi2xajs37bcgvf36io4llww3syb2yi.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_10', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 2048
    x2 = (xindex // 100352)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x3), None).to(tl.int1)
    tmp4 = tl.load(in_ptr2 + (x0 + (2048*x2)), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x3), None)
    tmp11 = tl.load(in_ptr4 + (x3), None)
    tmp12 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = 49.0
    tmp6 = tmp4 / tmp5
    tmp7 = tl.where(tmp3, tmp1, tmp6)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.where(tmp2, tmp1, tmp9)
    tmp13 = tmp11 - tmp12
    tmp15 = 0.00031887755102040814
    tmp16 = tmp14 * tmp15
    tmp18 = tmp17 * tmp17
    tmp19 = tmp16 * tmp18
    tmp20 = tmp13 * tmp19
    tmp21 = tmp10 - tmp20
    tmp23 = tmp22 * tmp15
    tmp24 = tmp21 - tmp23
    tmp26 = tmp17 * tmp25
    tmp27 = tmp24 * tmp26
    tl.store(in_out_ptr0 + (x3), tmp27, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/up/cupmafcxgudp44ltaza2hxkxmnv6gmq3chhhcx5g4mhhrulegbme.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.threshold_backward]

triton_poi_fused_add_div_threshold_backward_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i1', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_threshold_backward_11', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 2048
    x2 = (xindex // 100352)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp5 = tl.load(in_ptr2 + (x3), None).to(tl.int1)
    tmp6 = tl.load(in_ptr3 + (x0 + (2048*x2)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_out_ptr0 + (x3), None)
    tmp13 = tl.load(in_ptr4 + (x3), None)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = 49.0
    tmp8 = tmp6 / tmp7
    tmp9 = tl.where(tmp5, tmp1, tmp8)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.where(tmp4, tmp1, tmp11)
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(tmp2, tmp1, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/z3/cz3v6wvddzq7i6hdeqfp7qdso7vbrebywhgphfb2n7vyk3nypszm.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_12', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 51200
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 2048)
    x0 = xindex % 2048
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp25 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (126*x1)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (2048*(r2 % 7)) + (14336*(((r2 + (126*x1)) // 7) % 448))), rmask & tmp2, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask, tmp8, _tmp7)
        tmp9 = tl.load(in_ptr1 + (x0 + (2048*(r2 % 7)) + (14336*(((r2 + (126*x1)) // 7) % 448))), rmask & tmp2, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr2 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp9 - tmp10
        tmp12 = tmp3 * tmp11
        tmp13 = tl.full(tmp12.shape, 0, tmp12.dtype)
        tmp14 = tl.where(tmp2, tmp12, tmp13)
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask, tmp17, _tmp16)
        tmp18 = tl.load(in_ptr3 + (x0 + (2048*(r2 % 7)) + (14336*(((r2 + (126*x1)) // 7) % 448))), rmask & tmp2, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr4 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2, eviction_policy='evict_last', other=0.0)
        tmp20 = tmp18 - tmp19
        tmp21 = tmp3 * tmp20
        tmp22 = tl.full(tmp21.shape, 0, tmp21.dtype)
        tmp23 = tl.where(tmp2, tmp21, tmp22)
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
        tmp26 = _tmp25 + tmp24
        _tmp25 = tl.where(rmask, tmp26, _tmp25)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, None)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp16, None)
    tmp25 = tl.sum(_tmp25, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp25, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/sk/cskqgyfymat657itfklwuhrjaun7ckkljyawyto6sq4lceorxtbp.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(14,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_13', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 2048
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), None)
    tmp19 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr10 + (x0), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr11 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.00031887755102040814
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tmp20 = tmp18 - tmp19
    tmp22 = tmp21 * tmp5
    tmp24 = tmp23 * tmp23
    tmp25 = tmp22 * tmp24
    tmp26 = tmp20 * tmp25
    tmp27 = tmp0 - tmp26
    tmp28 = tmp27 - tmp13
    tmp30 = tmp23 * tmp29
    tmp31 = tmp28 * tmp30
    tl.store(out_ptr0 + (x2), tmp17, None)
    tl.store(out_ptr1 + (x2), tmp31, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/f3/cf3dkb7bml6wwvltxdolwujpkqnaragbmefzvxrdlkjxjgifsqgb.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_14', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 50176
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (65536*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (x0 + (512*r2) + (65536*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (x0 + (512*r2) + (65536*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp13, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/zz/czzf377osqjgkeimkppqefdxnpjl6mh5pmbglcu3ajsgtdnv4u7z.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_15', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/ta/ctawo52gcwnuszpftxyspekugawidmlx4tgv3wxaveduituq7uun.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_16', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/l7/cl7bsx7dwouzzeri6jpwg35spqfhvqutbr4jzu5ih7ngf7kgjbxv.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_17', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp5 = tl.load(in_ptr1 + (x2), None)
    tmp6 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 7.971938775510203e-05
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x2), tmp21, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/5x/c5xlu45iflqq7iaslg25gboo7dltjbz27hz3rhfrcnyab3zpqcbv.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[131072, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_18', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 100352
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp11 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (131072*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (x0 + (1024*r2) + (131072*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (1024*r2) + (131072*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + (x0 + (1024*r2) + (131072*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp6 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask, tmp16, _tmp15)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, None)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/wz/cwzlttmi6hvmlrgax3wp44x2wcvhr65gjfjyxve3nxkbhscqp3jw.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_19', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/w2/cw227l6yjowtbbsif6uubahsfjlawyj5a4pc4hot7bhxhkqbh4ai.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_20', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/7c/c7cebsn4i63bfzdr2oeiruekya5werb26dftcgd5an4us6crd6y7.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_21', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp3 = tl.load(in_ptr1 + (x2), None)
    tmp4 = tl.load(in_ptr2 + (x2), None)
    tmp7 = tl.load(in_ptr3 + (x2), None)
    tmp8 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 7.971938775510203e-05
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(out_ptr0 + (x2), tmp23, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/pq/cpq7hw3i7vn2r2m22rrnqo4m7zalxhcmrqkebq33txpwjntftqyx.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_22', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp13, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/x5/cx5y7ossm2gsaqtlzohnuvvbj6iq2iiyxarkpz6ak4lz73clbf2y.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[256, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_23', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/qb/cqbgktpemu5qxc5iiapsueplchvbvm5sydemkdsfqien5gu7cb2r.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[256, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_24', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/jb/cjbnnoihtph2zcosg557kweapdzuxvfukoyrj7uf7kotgk3dtbmy.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_25', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp5 = tl.load(in_ptr1 + (x2), None)
    tmp6 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 7.971938775510203e-05
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x2), tmp21, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/zm/czmgwkmf2hhcymymlaoy5mkcv23emjfx7hpiojzfkajrbkrfizwy.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_26', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp5 = tl.load(in_ptr2 + (x0), None)
    tmp6 = tl.load(in_ptr3 + (x0), None)
    tmp9 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tl.store(in_out_ptr0 + (x0), tmp11, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/cr/ccrz25u4isu53wl7orfwzrn6ikqmbfj3irp42h6xvonsa3aoivv7.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[131072, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_27', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 100352
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (131072*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (1024*r2) + (131072*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp0 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask, tmp10, _tmp9)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp9, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/sp/cspfx4gh6teyzihf5uyg2hstoxbqw5bbrhempii52dvuaznxjpij.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_28', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 7.971938775510203e-05
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/5v/c5vuclcsgoybf3lfdsqppwxkqy4lq34c2uwjb6ftltcaihjrlffp.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_29', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp5 = tl.load(in_out_ptr0 + (x0), None)
    tmp6 = tl.load(in_ptr2 + (x0), None)
    tmp9 = tl.load(in_ptr3 + (x0), None)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tl.store(in_out_ptr0 + (x0), tmp11, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/vn/cvnsvkmqv7conf4o3r5yfkbx2vgr3fsvvm6t7hqz7z3lelwerobs.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[131072, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_30', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 100352
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (131072*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (1024*r2) + (131072*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.load(in_ptr3 + (x0 + (1024*r2) + (131072*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp0 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask, tmp10, _tmp9)
        tmp13 = tmp11 - tmp12
        tmp14 = tmp0 * tmp13
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask, tmp17, _tmp16)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp9, None)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp16, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/kg/ckggbogjnbh3p7hxrxtiundgi7k3iulifcw77xhaahuu75l3mbrx.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_31 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(14,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_31', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), None)
    tmp19 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr10 + (x0), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr11 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 7.971938775510203e-05
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tmp20 = tmp18 - tmp19
    tmp22 = tmp21 * tmp5
    tmp24 = tmp23 * tmp23
    tmp25 = tmp22 * tmp24
    tmp26 = tmp20 * tmp25
    tmp27 = tmp0 - tmp26
    tmp28 = tmp27 - tmp13
    tmp30 = tmp23 * tmp29
    tmp31 = tmp28 * tmp30
    tl.store(out_ptr0 + (x2), tmp17, None)
    tl.store(out_ptr1 + (x2), tmp31, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/of/cofvvfqhyhlndwc5riupq3ik4mgtapftgvjtp56w22gf7w3fwmfr.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[131072, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_32', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 100352
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (32768*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (x0 + (256*r2) + (32768*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (x0 + (256*r2) + (32768*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp13, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/ke/cke2iusmdm7vzsfj27ezzuerqkirwmwcjq5sfwzk4qrjjkfbxctr.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_33 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[256, 512],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_33', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 392
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/ru/crufwpyffqwcp6bjfqmekx5zfvfcullt4czkm3illpp55jdkayw6.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[256, 512],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_34', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 392
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/2w/c2wwtl6dls76zgoozefxvg5aqtb72kif2iodc2hg3vjkrxgqvay2.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_35 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_35', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp5 = tl.load(in_ptr1 + (x2), None)
    tmp6 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 1.992984693877551e-05
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x2), tmp21, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/2s/c2suiw77honbcxejx6rrffis4bu2eijsm2dctxdm6sr66lcvn7ql.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_36 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[131072, 256],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_36', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 114688
    rnumel = 224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp11 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (114688*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (x0 + (512*r2) + (114688*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (512*r2) + (114688*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + (x0 + (512*r2) + (114688*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp6 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask, tmp16, _tmp15)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, None)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/22/c22ksbky2nfzjic65snd34tag5iluoiseh3hywzcvtstsqcgaqou.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_37 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[512, 256],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_37', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/cs/ccsxzf2mk7q253myi3atmgcrvw53k3zmgvvlt5zzpqqyigfydepw.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[512, 256],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_38', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/zd/czdy3yzs2igvxq26vavtyfisjzouvyngj23sauzzmvlaij6odf32.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_39 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_39', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25690112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp3 = tl.load(in_ptr1 + (x2), None)
    tmp4 = tl.load(in_ptr2 + (x2), None)
    tmp7 = tl.load(in_ptr3 + (x2), None)
    tmp8 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 1.992984693877551e-05
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(out_ptr0 + (x2), tmp23, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/bx/cbxstvygkehsf2r7ilho4x6tlkzalmyuk33ukhk2zhrf7xjai5vf.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_40 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_40', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 50176
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp13, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/4y/c4yfle5s72lnl2jctqdsvuf2o2oxrunvfzuzsuddqd4xi2yp35kt.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[128, 512],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_41', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 392
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/qp/cqpesqrerwbfbmlgsotmg2u3e223fhhnqelrfqtj4rtvx5spajk7.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_42 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[128, 512],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_42', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 392
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/kl/ckliepewrx3dcymwbtqe3fydahsymdibjjcjlamrximpma6nbup2.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp5 = tl.load(in_ptr1 + (x2), None)
    tmp6 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 1.992984693877551e-05
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x2), tmp21, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/4i/c4iss4b35p5nklhv3co6xu7e77ubbckantps7puuoku4shomzbow.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_44 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_44', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25690112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp5 = tl.load(in_out_ptr0 + (x0), None)
    tmp6 = tl.load(in_ptr2 + (x0), None)
    tmp9 = tl.load(in_ptr3 + (x0), None)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tl.store(in_out_ptr0 + (x0), tmp11, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/b2/cb2dym47zskvhkkij2zzpwpl57ktlrgfig2raxvvq2calrbq2zv3.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_45 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[131072, 256],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_45', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 114688
    rnumel = 224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (114688*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (512*r2) + (114688*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp0 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask, tmp10, _tmp9)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp9, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/a6/ca6p35x5dh6huntw4qyud7kjsgz3g54p2qhecspjdhuemyhawgq6.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_46 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_46', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25690112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 1.992984693877551e-05
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/ag/caggovt7ualeopbnujzpf6hplf2wyy56nap37k2nctigripz5okp.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_47 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[131072, 256],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_47', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 114688
    rnumel = 224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (114688*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (512*r2) + (114688*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.load(in_ptr3 + (x0 + (512*r2) + (114688*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp0 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask, tmp10, _tmp9)
        tmp13 = tmp11 - tmp12
        tmp14 = tmp0 * tmp13
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask, tmp17, _tmp16)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp9, None)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp16, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/4s/c4s2el343lezsfhjpp5w3ihwuiamx4tdczfa334q5vremk5at25j.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_48 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(14,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_48', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25690112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), None)
    tmp19 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr10 + (x0), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr11 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 1.992984693877551e-05
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tmp20 = tmp18 - tmp19
    tmp22 = tmp21 * tmp5
    tmp24 = tmp23 * tmp23
    tmp25 = tmp22 * tmp24
    tmp26 = tmp20 * tmp25
    tmp27 = tmp0 - tmp26
    tmp28 = tmp27 - tmp13
    tmp30 = tmp23 * tmp29
    tmp31 = tmp28 * tmp30
    tl.store(out_ptr0 + (x2), tmp17, None)
    tl.store(out_ptr1 + (x2), tmp31, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/qp/cqprj7qt327z7cxjp5xmj5lxhncjgctm7b7gqsjy3d2lpq6nhuzo.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_49 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[131072, 256],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_49', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 114688
    rnumel = 224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (28672*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (x0 + (128*r2) + (28672*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (x0 + (128*r2) + (28672*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp13, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/yd/cydwtp2nf53zmfyirahcrpmcy7ah5gtxu5rweqfnidmbhbfcotil.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_50 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[128, 1024],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_50', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/ns/cnssnwtzmrs6rwdvzjzxrrcv43qn2ssbhaj6cdav5caflgvajhyf.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_51 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[128, 1024],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_51', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/ju/cju53rnpmyfrjjrcgd7pdfvppy7krv5pb2aoqqs4i5zsmer6mnnt.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_52 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_52', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25690112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp5 = tl.load(in_ptr1 + (x2), None)
    tmp6 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 4.982461734693877e-06
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x2), tmp21, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/4y/c4y52e5imldasfuujsanygqdbbz43wrfkgw72tkiokd7vrqzq6yv.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_53 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[131072, 512],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_53', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 114688
    rnumel = 448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp11 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (114688*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (x0 + (256*r2) + (114688*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (256*r2) + (114688*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + (x0 + (256*r2) + (114688*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp6 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask, tmp16, _tmp15)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, None)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/4b/c4b6z4krxtbwfjkrra5gwd2y5qtn52on2o2doqlh5tpwukr23pdl.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_54 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[256, 512],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_54', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/nm/cnmd2chjxogrukzn3hdarwwgqb6r4gbklpgzy7tngexx5qqihlc7.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_55 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[256, 512],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_55', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/jy/cjyyrosfmwo5lb6l4g3tx65r72mxrsmv3p22gs5p34mdec553w36.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_56 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[67108864], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_56', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp3 = tl.load(in_ptr1 + (x2), None)
    tmp4 = tl.load(in_ptr2 + (x2), None)
    tmp7 = tl.load(in_ptr3 + (x2), None)
    tmp8 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 4.982461734693877e-06
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(out_ptr0 + (x2), tmp23, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/cb/ccbzaaoht7fol6jnwaxtvbfk3rzchhzwszkw3cl67xhxhycwm4cw.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_57 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[65536, 256],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_57', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 57344
    rnumel = 224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (14336*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (x0 + (64*r2) + (14336*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (x0 + (64*r2) + (14336*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp13, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/pq/cpq2wu2tj5afbzz25kg64gptzoj6nvwfp46wajys5pkidnnamoal.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_58 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[64, 1024],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_58', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/cn/ccngfoostovi4xoi3m6245ub234oauciyvzbe4dalynh6pxvkjsm.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_59 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[64, 1024],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_59', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/xr/cxrggwxwpo732alpzq3uudpunhkmrfekdzclibn6ui3x7g74fqtw.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_60 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_60', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp5 = tl.load(in_ptr1 + (x2), None)
    tmp6 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 4.982461734693877e-06
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x2), tmp21, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/wi/cwi7cvo34khn2ixorowouw4nidbqxvvap73fykcwxxdyhxgpufxl.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_61 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[67108864], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_61', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp5 = tl.load(in_out_ptr0 + (x0), None)
    tmp6 = tl.load(in_ptr2 + (x0), None)
    tmp9 = tl.load(in_ptr3 + (x0), None)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tl.store(in_out_ptr0 + (x0), tmp11, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/ud/cudqtuy2yndwjhxbxvmzb75jdsf32ytu5yzifwtsuwf5mr7hho5z.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_62 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[131072, 512],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_62', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 114688
    rnumel = 448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (114688*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (256*r2) + (114688*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp0 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask, tmp10, _tmp9)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp9, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/n5/cn5vrqd43y3if7mb2dx6g5n6dm5zci3um2qhts5soelyymupua7b.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_63 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[67108864], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_63', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 4.982461734693877e-06
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/wh/cwhvwfcxh3ezwfmiqb6ruzoq5663dzin2um7jd6ersf5wat7y4o3.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_64 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[131072, 512],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_64', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 114688
    rnumel = 448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp11 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp18 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (114688*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (x0 + (256*r2) + (114688*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (256*r2) + (114688*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + (x0 + (256*r2) + (114688*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tl.load(in_ptr5 + (x0 + (256*r2) + (114688*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp6 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask, tmp16, _tmp15)
        tmp19 = tmp17 - tmp18
        tmp20 = tmp6 * tmp19
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask, tmp23, _tmp22)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, None)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp15, None)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp22, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/3m/c3m4vtvdjhcmf4dl6won67qvxnfr3sgfkkyuxpf47aapvk6b4fpu.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_65 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[67108864], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(16,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_65', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp3 = tl.load(in_ptr1 + (x2), None)
    tmp4 = tl.load(in_ptr2 + (x2), None)
    tmp7 = tl.load(in_ptr3 + (x2), None)
    tmp8 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr9 + (x2), None)
    tmp25 = tl.load(in_ptr10 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr11 + (x0), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr12 + (x0), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr13 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 4.982461734693877e-06
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tmp26 = tmp24 - tmp25
    tmp28 = tmp27 * tmp11
    tmp30 = tmp29 * tmp29
    tmp31 = tmp28 * tmp30
    tmp32 = tmp26 * tmp31
    tmp33 = tmp6 - tmp32
    tmp34 = tmp33 - tmp19
    tmp36 = tmp29 * tmp35
    tmp37 = tmp34 * tmp36
    tl.store(out_ptr0 + (x2), tmp23, None)
    tl.store(out_ptr1 + (x2), tmp37, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/de/cdedjm74k63bmqmrbpcwo2fy4curtgomz4272fxwk2xsngeysgc2.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_66 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_66', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/jh/cjhm4xvjdpb6q555xevfxeic6shlh7ccwpj6nxqcli7b6h6b5bnq.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_67 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[65536, 1024],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_67', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 57344
    rnumel = 896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (57344*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (x0 + (64*r2) + (57344*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (x0 + (64*r2) + (57344*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp13, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/fk/cfk3r6szi7eqcg4ktr7lnm46d2nnla7osg47yuto4s7qkcpfpnkt.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_68 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[67108864], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_68', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp5 = tl.load(in_ptr1 + (x2), None)
    tmp6 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 1.2456154336734693e-06
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x2), tmp21, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_321, convolution, squeeze_1, relu, getitem_2, getitem_3, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, convolution_3, squeeze_10, convolution_4, squeeze_13, relu_3, convolution_5, squeeze_16, relu_4, convolution_6, squeeze_19, relu_5, convolution_7, squeeze_22, relu_6, convolution_8, squeeze_25, relu_7, convolution_9, squeeze_28, relu_8, convolution_10, squeeze_31, relu_9, convolution_11, squeeze_34, relu_10, convolution_12, squeeze_37, relu_11, convolution_13, squeeze_40, convolution_14, squeeze_43, relu_12, convolution_15, squeeze_46, relu_13, convolution_16, squeeze_49, relu_14, convolution_17, squeeze_52, relu_15, convolution_18, squeeze_55, relu_16, convolution_19, squeeze_58, relu_17, convolution_20, squeeze_61, relu_18, convolution_21, squeeze_64, relu_19, convolution_22, squeeze_67, relu_20, convolution_23, squeeze_70, relu_21, convolution_24, squeeze_73, relu_22, convolution_25, squeeze_76, relu_23, convolution_26, squeeze_79, convolution_27, squeeze_82, relu_24, convolution_28, squeeze_85, relu_25, convolution_29, squeeze_88, relu_26, convolution_30, squeeze_91, relu_27, convolution_31, squeeze_94, relu_28, convolution_32, squeeze_97, relu_29, convolution_33, squeeze_100, relu_30, convolution_34, squeeze_103, relu_31, convolution_35, squeeze_106, relu_32, convolution_36, squeeze_109, relu_33, convolution_37, squeeze_112, relu_34, convolution_38, squeeze_115, relu_35, convolution_39, squeeze_118, relu_36, convolution_40, squeeze_121, relu_37, convolution_41, squeeze_124, relu_38, convolution_42, squeeze_127, relu_39, convolution_43, squeeze_130, relu_40, convolution_44, squeeze_133, relu_41, convolution_45, squeeze_136, convolution_46, squeeze_139, relu_42, convolution_47, squeeze_142, relu_43, convolution_48, squeeze_145, relu_44, convolution_49, squeeze_148, relu_45, convolution_50, squeeze_151, relu_46, convolution_51, squeeze_154, relu_47, convolution_52, squeeze_157, view, permute_1, le, unsqueeze_214, unsqueeze_226, unsqueeze_238, unsqueeze_250, unsqueeze_262, unsqueeze_274, unsqueeze_286, unsqueeze_298, unsqueeze_310, unsqueeze_322, unsqueeze_334, unsqueeze_346, unsqueeze_358, unsqueeze_370, unsqueeze_382, unsqueeze_394, unsqueeze_406, unsqueeze_418, unsqueeze_430, unsqueeze_442, unsqueeze_454, unsqueeze_466, unsqueeze_478, unsqueeze_490, unsqueeze_502, unsqueeze_514, unsqueeze_526, unsqueeze_538, unsqueeze_550, unsqueeze_562, unsqueeze_574, unsqueeze_586, unsqueeze_598, unsqueeze_610, unsqueeze_622, unsqueeze_634, unsqueeze_646, unsqueeze_658, unsqueeze_670, unsqueeze_682, unsqueeze_694, unsqueeze_706, unsqueeze_718, unsqueeze_730, unsqueeze_742, unsqueeze_754, unsqueeze_766, unsqueeze_778, unsqueeze_790, unsqueeze_802, unsqueeze_814, unsqueeze_826, unsqueeze_838, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 7, 7), (147, 1, 21, 3))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_4, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_7, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_8, (64, ), (1, ))
    assert_size_stride(primals_10, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_11, (256, ), (1, ))
    assert_size_stride(primals_13, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_14, (256, ), (1, ))
    assert_size_stride(primals_16, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_17, (64, ), (1, ))
    assert_size_stride(primals_19, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_20, (64, ), (1, ))
    assert_size_stride(primals_22, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_23, (256, ), (1, ))
    assert_size_stride(primals_25, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_26, (64, ), (1, ))
    assert_size_stride(primals_28, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_29, (64, ), (1, ))
    assert_size_stride(primals_31, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_32, (256, ), (1, ))
    assert_size_stride(primals_34, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_35, (128, ), (1, ))
    assert_size_stride(primals_37, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_38, (128, ), (1, ))
    assert_size_stride(primals_40, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_41, (512, ), (1, ))
    assert_size_stride(primals_43, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_44, (512, ), (1, ))
    assert_size_stride(primals_46, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_47, (128, ), (1, ))
    assert_size_stride(primals_49, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_50, (128, ), (1, ))
    assert_size_stride(primals_52, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_53, (512, ), (1, ))
    assert_size_stride(primals_55, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_56, (128, ), (1, ))
    assert_size_stride(primals_58, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_59, (128, ), (1, ))
    assert_size_stride(primals_61, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_62, (512, ), (1, ))
    assert_size_stride(primals_64, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_65, (128, ), (1, ))
    assert_size_stride(primals_67, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_68, (128, ), (1, ))
    assert_size_stride(primals_70, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_71, (512, ), (1, ))
    assert_size_stride(primals_73, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_74, (256, ), (1, ))
    assert_size_stride(primals_76, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_77, (256, ), (1, ))
    assert_size_stride(primals_79, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_80, (1024, ), (1, ))
    assert_size_stride(primals_82, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_83, (1024, ), (1, ))
    assert_size_stride(primals_85, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_86, (256, ), (1, ))
    assert_size_stride(primals_88, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_89, (256, ), (1, ))
    assert_size_stride(primals_91, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_92, (1024, ), (1, ))
    assert_size_stride(primals_94, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_95, (256, ), (1, ))
    assert_size_stride(primals_97, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_98, (256, ), (1, ))
    assert_size_stride(primals_100, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_101, (1024, ), (1, ))
    assert_size_stride(primals_103, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_104, (256, ), (1, ))
    assert_size_stride(primals_106, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_107, (256, ), (1, ))
    assert_size_stride(primals_109, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_110, (1024, ), (1, ))
    assert_size_stride(primals_112, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_113, (256, ), (1, ))
    assert_size_stride(primals_115, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_116, (256, ), (1, ))
    assert_size_stride(primals_118, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_119, (1024, ), (1, ))
    assert_size_stride(primals_121, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_122, (256, ), (1, ))
    assert_size_stride(primals_124, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_125, (256, ), (1, ))
    assert_size_stride(primals_127, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_128, (1024, ), (1, ))
    assert_size_stride(primals_130, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_131, (512, ), (1, ))
    assert_size_stride(primals_133, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_134, (512, ), (1, ))
    assert_size_stride(primals_136, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_137, (2048, ), (1, ))
    assert_size_stride(primals_139, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_140, (2048, ), (1, ))
    assert_size_stride(primals_142, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_143, (512, ), (1, ))
    assert_size_stride(primals_145, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_146, (512, ), (1, ))
    assert_size_stride(primals_148, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_149, (2048, ), (1, ))
    assert_size_stride(primals_151, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_152, (512, ), (1, ))
    assert_size_stride(primals_154, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_155, (512, ), (1, ))
    assert_size_stride(primals_157, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_158, (2048, ), (1, ))
    assert_size_stride(primals_321, (64, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(convolution, (64, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(squeeze_1, (64, ), (1, ))
    assert_size_stride(relu, (64, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(getitem_2, (64, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(getitem_3, (64, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convolution_1, (64, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(squeeze_4, (64, ), (1, ))
    assert_size_stride(relu_1, (64, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convolution_2, (64, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(squeeze_7, (64, ), (1, ))
    assert_size_stride(relu_2, (64, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convolution_3, (64, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(squeeze_10, (256, ), (1, ))
    assert_size_stride(convolution_4, (64, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(squeeze_13, (256, ), (1, ))
    assert_size_stride(relu_3, (64, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(convolution_5, (64, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(squeeze_16, (64, ), (1, ))
    assert_size_stride(relu_4, (64, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convolution_6, (64, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(squeeze_19, (64, ), (1, ))
    assert_size_stride(relu_5, (64, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convolution_7, (64, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(squeeze_22, (256, ), (1, ))
    assert_size_stride(relu_6, (64, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(convolution_8, (64, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(squeeze_25, (64, ), (1, ))
    assert_size_stride(relu_7, (64, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convolution_9, (64, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(squeeze_28, (64, ), (1, ))
    assert_size_stride(relu_8, (64, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convolution_10, (64, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(squeeze_31, (256, ), (1, ))
    assert_size_stride(relu_9, (64, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(convolution_11, (64, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(squeeze_34, (128, ), (1, ))
    assert_size_stride(relu_10, (64, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(convolution_12, (64, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_37, (128, ), (1, ))
    assert_size_stride(relu_11, (64, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_13, (64, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(squeeze_40, (512, ), (1, ))
    assert_size_stride(convolution_14, (64, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(squeeze_43, (512, ), (1, ))
    assert_size_stride(relu_12, (64, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(convolution_15, (64, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_46, (128, ), (1, ))
    assert_size_stride(relu_13, (64, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_16, (64, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_49, (128, ), (1, ))
    assert_size_stride(relu_14, (64, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_17, (64, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(squeeze_52, (512, ), (1, ))
    assert_size_stride(relu_15, (64, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(convolution_18, (64, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_55, (128, ), (1, ))
    assert_size_stride(relu_16, (64, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_19, (64, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_58, (128, ), (1, ))
    assert_size_stride(relu_17, (64, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_20, (64, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(squeeze_61, (512, ), (1, ))
    assert_size_stride(relu_18, (64, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(convolution_21, (64, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_64, (128, ), (1, ))
    assert_size_stride(relu_19, (64, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_22, (64, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_67, (128, ), (1, ))
    assert_size_stride(relu_20, (64, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_23, (64, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(squeeze_70, (512, ), (1, ))
    assert_size_stride(relu_21, (64, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(convolution_24, (64, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(squeeze_73, (256, ), (1, ))
    assert_size_stride(relu_22, (64, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(convolution_25, (64, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_76, (256, ), (1, ))
    assert_size_stride(relu_23, (64, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_26, (64, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_79, (1024, ), (1, ))
    assert_size_stride(convolution_27, (64, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_82, (1024, ), (1, ))
    assert_size_stride(relu_24, (64, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_28, (64, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_85, (256, ), (1, ))
    assert_size_stride(relu_25, (64, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_29, (64, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_88, (256, ), (1, ))
    assert_size_stride(relu_26, (64, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_30, (64, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_91, (1024, ), (1, ))
    assert_size_stride(relu_27, (64, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_31, (64, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_94, (256, ), (1, ))
    assert_size_stride(relu_28, (64, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_32, (64, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_97, (256, ), (1, ))
    assert_size_stride(relu_29, (64, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_33, (64, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_100, (1024, ), (1, ))
    assert_size_stride(relu_30, (64, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_34, (64, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_103, (256, ), (1, ))
    assert_size_stride(relu_31, (64, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_35, (64, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_106, (256, ), (1, ))
    assert_size_stride(relu_32, (64, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_36, (64, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_109, (1024, ), (1, ))
    assert_size_stride(relu_33, (64, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_37, (64, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_112, (256, ), (1, ))
    assert_size_stride(relu_34, (64, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_38, (64, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_115, (256, ), (1, ))
    assert_size_stride(relu_35, (64, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_39, (64, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_118, (1024, ), (1, ))
    assert_size_stride(relu_36, (64, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_40, (64, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_121, (256, ), (1, ))
    assert_size_stride(relu_37, (64, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_41, (64, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_124, (256, ), (1, ))
    assert_size_stride(relu_38, (64, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_42, (64, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_127, (1024, ), (1, ))
    assert_size_stride(relu_39, (64, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_43, (64, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_130, (512, ), (1, ))
    assert_size_stride(relu_40, (64, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_44, (64, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(squeeze_133, (512, ), (1, ))
    assert_size_stride(relu_41, (64, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(convolution_45, (64, 2048, 7, 7), (100352, 1, 14336, 2048))
    assert_size_stride(squeeze_136, (2048, ), (1, ))
    assert_size_stride(convolution_46, (64, 2048, 7, 7), (100352, 1, 14336, 2048))
    assert_size_stride(squeeze_139, (2048, ), (1, ))
    assert_size_stride(relu_42, (64, 2048, 7, 7), (100352, 1, 14336, 2048))
    assert_size_stride(convolution_47, (64, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(squeeze_142, (512, ), (1, ))
    assert_size_stride(relu_43, (64, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(convolution_48, (64, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(squeeze_145, (512, ), (1, ))
    assert_size_stride(relu_44, (64, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(convolution_49, (64, 2048, 7, 7), (100352, 1, 14336, 2048))
    assert_size_stride(squeeze_148, (2048, ), (1, ))
    assert_size_stride(relu_45, (64, 2048, 7, 7), (100352, 1, 14336, 2048))
    assert_size_stride(convolution_50, (64, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(squeeze_151, (512, ), (1, ))
    assert_size_stride(relu_46, (64, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(convolution_51, (64, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(squeeze_154, (512, ), (1, ))
    assert_size_stride(relu_47, (64, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(convolution_52, (64, 2048, 7, 7), (100352, 1, 14336, 2048))
    assert_size_stride(squeeze_157, (2048, ), (1, ))
    assert_size_stride(view, (64, 2048), (2048, 1))
    assert_size_stride(permute_1, (1000, 2048), (2048, 1))
    assert_size_stride(le, (64, 2048, 7, 7), (100352, 1, 14336, 2048))
    assert_size_stride(unsqueeze_214, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_226, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_238, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_250, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_262, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_274, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_286, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_298, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_310, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_322, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_334, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_346, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_358, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_370, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_382, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_394, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_406, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_418, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_430, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_442, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_454, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_466, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_478, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_490, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_502, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_514, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_526, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_538, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_550, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_562, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_574, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_586, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_598, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_610, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_622, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_634, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_646, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_658, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_670, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_682, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_694, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_706, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_718, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_730, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_742, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_754, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_766, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_778, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_790, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_802, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_814, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_826, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_838, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(tangents_1, (64, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, 2048), (2048, 1), torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_1, out=buf0)
        del permute_1
        buf1 = empty_strided_cuda((1000, 2048), (2048, 1), torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 64), (1, 1000), 0), view, out=buf1)
        del view
        buf2 = empty_strided_cuda((1, 1000), (1000, 1), torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_0.run(tangents_1, buf2, 1000, 64, grid=grid(1000), stream=stream0)
        del tangents_1
        buf3 = empty_strided_cuda((2048, 25), (1, 2048), torch.float32)
        buf5 = empty_strided_cuda((2048, 25), (1, 2048), torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_div_native_batch_norm_backward_threshold_backward_1.run(le, buf0, convolution_52, unsqueeze_214, buf3, buf5, 51200, 126, grid=grid(51200), stream=stream0)
        buf4 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_2.run(buf3, buf4, 2048, 25, grid=grid(2048), stream=stream0)
        buf6 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        buf7 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_3.run(buf5, squeeze_157, buf6, buf7, 2048, 25, grid=grid(2048), stream=stream0)
        buf8 = empty_strided_cuda((64, 2048, 7, 7), (100352, 1, 14336, 2048), torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_4.run(le, buf0, convolution_52, unsqueeze_214, buf6, squeeze_157, buf4, primals_158, buf8, 6422528, grid=grid(6422528), stream=stream0)
        del convolution_52
        del primals_158
        del squeeze_157
        del unsqueeze_214
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        buf9 = aten.convolution_backward.default(buf8, relu_47, primals_157, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_157
        buf10 = buf9[0]
        buf11 = buf9[1]
        del buf9
        buf12 = empty_strided_cuda((512, 25), (1, 512), torch.float32)
        buf14 = empty_strided_cuda((512, 25), (1, 512), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_5.run(relu_47, buf10, convolution_51, unsqueeze_226, buf12, buf14, 12800, 126, grid=grid(12800), stream=stream0)
        buf13 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_6.run(buf12, buf13, 512, 25, grid=grid(512), stream=stream0)
        buf15 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf16 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf14, squeeze_154, buf15, buf16, 512, 25, grid=grid(512), stream=stream0)
        buf17 = buf10; del buf10  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_8.run(buf17, relu_47, convolution_51, unsqueeze_226, buf15, squeeze_154, buf13, primals_155, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_51
        del primals_155
        del relu_47
        del squeeze_154
        del unsqueeze_226
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf18 = aten.convolution_backward.default(buf17, relu_46, primals_154, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf17
        del primals_154
        buf19 = buf18[0]
        buf20 = buf18[1]
        del buf18
        buf21 = buf14; del buf14  # reuse
        buf23 = buf12; del buf12  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_5.run(relu_46, buf19, convolution_50, unsqueeze_238, buf21, buf23, 12800, 126, grid=grid(12800), stream=stream0)
        buf22 = buf15; del buf15  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_6.run(buf21, buf22, 512, 25, grid=grid(512), stream=stream0)
        buf24 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf25 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf23, squeeze_151, buf24, buf25, 512, 25, grid=grid(512), stream=stream0)
        buf26 = buf19; del buf19  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_8.run(buf26, relu_46, convolution_50, unsqueeze_238, buf24, squeeze_151, buf22, primals_152, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_50
        del primals_152
        del relu_46
        del squeeze_151
        del unsqueeze_238
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf27 = aten.convolution_backward.default(buf26, relu_45, primals_151, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf26
        del primals_151
        buf28 = buf27[0]
        buf29 = buf27[1]
        del buf27
        buf30 = buf5; del buf5  # reuse
        buf32 = buf3; del buf3  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_div_native_batch_norm_backward_threshold_backward_9.run(relu_45, le, buf0, buf28, convolution_49, unsqueeze_250, buf30, buf32, 51200, 126, grid=grid(51200), stream=stream0)
        buf31 = buf6; del buf6  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_2.run(buf30, buf31, 2048, 25, grid=grid(2048), stream=stream0)
        buf33 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        buf35 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_3.run(buf32, squeeze_148, buf33, buf35, 2048, 25, grid=grid(2048), stream=stream0)
        buf34 = buf8; del buf8  # reuse
        buf36 = buf34; del buf34  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_10.run(buf36, relu_45, le, buf0, buf28, convolution_49, unsqueeze_250, buf33, squeeze_148, buf31, primals_149, 6422528, grid=grid(6422528), stream=stream0)
        del convolution_49
        del primals_149
        del squeeze_148
        del unsqueeze_250
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf37 = aten.convolution_backward.default(buf36, relu_44, primals_148, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_148
        buf38 = buf37[0]
        buf39 = buf37[1]
        del buf37
        buf40 = buf23; del buf23  # reuse
        buf42 = buf21; del buf21  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_5.run(relu_44, buf38, convolution_48, unsqueeze_262, buf40, buf42, 12800, 126, grid=grid(12800), stream=stream0)
        buf41 = buf24; del buf24  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_6.run(buf40, buf41, 512, 25, grid=grid(512), stream=stream0)
        buf43 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf44 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf42, squeeze_145, buf43, buf44, 512, 25, grid=grid(512), stream=stream0)
        buf45 = buf38; del buf38  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_8.run(buf45, relu_44, convolution_48, unsqueeze_262, buf43, squeeze_145, buf41, primals_146, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_48
        del primals_146
        del relu_44
        del squeeze_145
        del unsqueeze_262
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf46 = aten.convolution_backward.default(buf45, relu_43, primals_145, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf45
        del primals_145
        buf47 = buf46[0]
        buf48 = buf46[1]
        del buf46
        buf49 = buf42; del buf42  # reuse
        buf51 = buf40; del buf40  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_5.run(relu_43, buf47, convolution_47, unsqueeze_274, buf49, buf51, 12800, 126, grid=grid(12800), stream=stream0)
        buf50 = buf43; del buf43  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_6.run(buf49, buf50, 512, 25, grid=grid(512), stream=stream0)
        buf52 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf53 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf51, squeeze_142, buf52, buf53, 512, 25, grid=grid(512), stream=stream0)
        buf54 = buf47; del buf47  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_8.run(buf54, relu_43, convolution_47, unsqueeze_274, buf52, squeeze_142, buf50, primals_143, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_47
        del primals_143
        del relu_43
        del squeeze_142
        del unsqueeze_274
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf55 = aten.convolution_backward.default(buf54, relu_42, primals_142, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf54
        del primals_142
        buf56 = buf55[0]
        buf57 = buf55[1]
        del buf55
        buf58 = buf28; del buf28  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.threshold_backward]
        triton_poi_fused_add_div_threshold_backward_11.run(buf58, relu_42, relu_45, le, buf0, buf56, 6422528, grid=grid(6422528), stream=stream0)
        del buf0
        del le
        del relu_42
        del relu_45
        buf59 = buf32; del buf32  # reuse
        buf61 = buf30; del buf30  # reuse
        buf68 = empty_strided_cuda((2048, 25), (1, 2048), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_12.run(buf58, convolution_46, unsqueeze_286, convolution_45, unsqueeze_298, buf59, buf61, buf68, 51200, 126, grid=grid(51200), stream=stream0)
        buf60 = buf33; del buf33  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_2.run(buf59, buf60, 2048, 25, grid=grid(2048), stream=stream0)
        del buf59
        buf62 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        buf63 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_3.run(buf61, squeeze_139, buf62, buf63, 2048, 25, grid=grid(2048), stream=stream0)
        del buf61
        buf69 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        buf70 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_3.run(buf68, squeeze_136, buf69, buf70, 2048, 25, grid=grid(2048), stream=stream0)
        del buf68
        buf64 = buf56; del buf56  # reuse
        buf71 = buf36; del buf36  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_13.run(buf58, convolution_46, unsqueeze_286, buf62, squeeze_139, buf60, primals_140, convolution_45, unsqueeze_298, buf69, squeeze_136, primals_137, buf64, buf71, 6422528, grid=grid(6422528), stream=stream0)
        del buf58
        del buf62
        del buf69
        del convolution_45
        del convolution_46
        del primals_137
        del primals_140
        del squeeze_136
        del squeeze_139
        del unsqueeze_286
        del unsqueeze_298
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf65 = aten.convolution_backward.default(buf64, relu_39, primals_139, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf64
        del primals_139
        buf66 = buf65[0]
        buf67 = buf65[1]
        del buf65
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf72 = aten.convolution_backward.default(buf71, relu_41, primals_136, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf71
        del primals_136
        buf73 = buf72[0]
        buf74 = buf72[1]
        del buf72
        buf75 = buf51; del buf51  # reuse
        buf77 = buf49; del buf49  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_5.run(relu_41, buf73, convolution_44, unsqueeze_310, buf75, buf77, 12800, 126, grid=grid(12800), stream=stream0)
        buf76 = buf52; del buf52  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_6.run(buf75, buf76, 512, 25, grid=grid(512), stream=stream0)
        del buf75
        buf78 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf79 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf77, squeeze_133, buf78, buf79, 512, 25, grid=grid(512), stream=stream0)
        del buf77
        buf80 = buf73; del buf73  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_8.run(buf80, relu_41, convolution_44, unsqueeze_310, buf78, squeeze_133, buf76, primals_134, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_44
        del primals_134
        del relu_41
        del squeeze_133
        del unsqueeze_310
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf81 = aten.convolution_backward.default(buf80, relu_40, primals_133, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf80
        del primals_133
        buf82 = buf81[0]
        buf83 = buf81[1]
        del buf81
        buf84 = empty_strided_cuda((512, 98), (1, 512), torch.float32)
        buf86 = empty_strided_cuda((512, 98), (1, 512), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(relu_40, buf82, convolution_43, unsqueeze_322, buf84, buf86, 50176, 128, grid=grid(50176), stream=stream0)
        buf85 = buf78; del buf78  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_15.run(buf84, buf85, 512, 98, grid=grid(512), stream=stream0)
        buf87 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf88 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_16.run(buf86, squeeze_130, buf87, buf88, 512, 98, grid=grid(512), stream=stream0)
        buf89 = buf82; del buf82  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_17.run(buf89, relu_40, convolution_43, unsqueeze_322, buf87, squeeze_130, buf85, primals_131, 6422528, grid=grid(6422528), stream=stream0)
        del convolution_43
        del primals_131
        del relu_40
        del squeeze_130
        del unsqueeze_322
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf90 = aten.convolution_backward.default(buf89, relu_39, primals_130, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf89
        del primals_130
        buf91 = buf90[0]
        buf92 = buf90[1]
        del buf90
        buf93 = empty_strided_cuda((1024, 98), (1, 1024), torch.float32)
        buf95 = empty_strided_cuda((1024, 98), (1, 1024), torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_18.run(relu_39, buf66, buf91, convolution_42, unsqueeze_334, buf93, buf95, 100352, 128, grid=grid(100352), stream=stream0)
        buf94 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_19.run(buf93, buf94, 1024, 98, grid=grid(1024), stream=stream0)
        buf96 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf98 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_20.run(buf95, squeeze_127, buf96, buf98, 1024, 98, grid=grid(1024), stream=stream0)
        buf97 = empty_strided_cuda((64, 1024, 14, 14), (200704, 1, 14336, 1024), torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_21.run(relu_39, buf66, buf91, convolution_42, unsqueeze_334, buf96, squeeze_127, buf94, primals_128, buf97, 12845056, grid=grid(12845056), stream=stream0)
        del convolution_42
        del primals_128
        del squeeze_127
        del unsqueeze_334
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf99 = aten.convolution_backward.default(buf97, relu_38, primals_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf97
        del primals_127
        buf100 = buf99[0]
        buf101 = buf99[1]
        del buf99
        buf102 = empty_strided_cuda((256, 98), (1, 256), torch.float32)
        buf104 = empty_strided_cuda((256, 98), (1, 256), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_22.run(relu_38, buf100, convolution_41, unsqueeze_346, buf102, buf104, 25088, 128, grid=grid(25088), stream=stream0)
        buf103 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_23.run(buf102, buf103, 256, 98, grid=grid(256), stream=stream0)
        buf105 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf106 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_24.run(buf104, squeeze_124, buf105, buf106, 256, 98, grid=grid(256), stream=stream0)
        buf107 = buf100; del buf100  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_25.run(buf107, relu_38, convolution_41, unsqueeze_346, buf105, squeeze_124, buf103, primals_125, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_41
        del primals_125
        del relu_38
        del squeeze_124
        del unsqueeze_346
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf108 = aten.convolution_backward.default(buf107, relu_37, primals_124, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf107
        del primals_124
        buf109 = buf108[0]
        buf110 = buf108[1]
        del buf108
        buf111 = buf104; del buf104  # reuse
        buf113 = buf102; del buf102  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_22.run(relu_37, buf109, convolution_40, unsqueeze_358, buf111, buf113, 25088, 128, grid=grid(25088), stream=stream0)
        buf112 = buf105; del buf105  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_23.run(buf111, buf112, 256, 98, grid=grid(256), stream=stream0)
        buf114 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf115 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_24.run(buf113, squeeze_121, buf114, buf115, 256, 98, grid=grid(256), stream=stream0)
        buf116 = buf109; del buf109  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_25.run(buf116, relu_37, convolution_40, unsqueeze_358, buf114, squeeze_121, buf112, primals_122, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_40
        del primals_122
        del relu_37
        del squeeze_121
        del unsqueeze_358
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf117 = aten.convolution_backward.default(buf116, relu_36, primals_121, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf116
        del primals_121
        buf118 = buf117[0]
        buf119 = buf117[1]
        del buf117
        buf120 = buf118; del buf118  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_26.run(buf120, relu_36, relu_39, buf66, buf91, 12845056, grid=grid(12845056), stream=stream0)
        del buf66
        del relu_36
        del relu_39
        buf121 = buf95; del buf95  # reuse
        buf123 = buf93; del buf93  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_27.run(buf120, convolution_39, unsqueeze_370, buf121, buf123, 100352, 128, grid=grid(100352), stream=stream0)
        buf122 = buf96; del buf96  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_19.run(buf121, buf122, 1024, 98, grid=grid(1024), stream=stream0)
        buf124 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf125 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_20.run(buf123, squeeze_118, buf124, buf125, 1024, 98, grid=grid(1024), stream=stream0)
        buf126 = buf91; del buf91  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_28.run(buf120, convolution_39, unsqueeze_370, buf124, squeeze_118, buf122, primals_119, buf126, 12845056, grid=grid(12845056), stream=stream0)
        del convolution_39
        del primals_119
        del squeeze_118
        del unsqueeze_370
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf127 = aten.convolution_backward.default(buf126, relu_35, primals_118, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_118
        buf128 = buf127[0]
        buf129 = buf127[1]
        del buf127
        buf130 = buf113; del buf113  # reuse
        buf132 = buf111; del buf111  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_22.run(relu_35, buf128, convolution_38, unsqueeze_382, buf130, buf132, 25088, 128, grid=grid(25088), stream=stream0)
        buf131 = buf114; del buf114  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_23.run(buf130, buf131, 256, 98, grid=grid(256), stream=stream0)
        buf133 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf134 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_24.run(buf132, squeeze_115, buf133, buf134, 256, 98, grid=grid(256), stream=stream0)
        buf135 = buf128; del buf128  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_25.run(buf135, relu_35, convolution_38, unsqueeze_382, buf133, squeeze_115, buf131, primals_116, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_38
        del primals_116
        del relu_35
        del squeeze_115
        del unsqueeze_382
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf136 = aten.convolution_backward.default(buf135, relu_34, primals_115, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf135
        del primals_115
        buf137 = buf136[0]
        buf138 = buf136[1]
        del buf136
        buf139 = buf132; del buf132  # reuse
        buf141 = buf130; del buf130  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_22.run(relu_34, buf137, convolution_37, unsqueeze_394, buf139, buf141, 25088, 128, grid=grid(25088), stream=stream0)
        buf140 = buf133; del buf133  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_23.run(buf139, buf140, 256, 98, grid=grid(256), stream=stream0)
        buf142 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf143 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_24.run(buf141, squeeze_112, buf142, buf143, 256, 98, grid=grid(256), stream=stream0)
        buf144 = buf137; del buf137  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_25.run(buf144, relu_34, convolution_37, unsqueeze_394, buf142, squeeze_112, buf140, primals_113, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_37
        del primals_113
        del relu_34
        del squeeze_112
        del unsqueeze_394
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf145 = aten.convolution_backward.default(buf144, relu_33, primals_112, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf144
        del primals_112
        buf146 = buf145[0]
        buf147 = buf145[1]
        del buf145
        buf148 = buf123; del buf123  # reuse
        buf150 = buf121; del buf121  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_18.run(relu_33, buf120, buf146, convolution_36, unsqueeze_406, buf148, buf150, 100352, 128, grid=grid(100352), stream=stream0)
        buf149 = buf124; del buf124  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_19.run(buf148, buf149, 1024, 98, grid=grid(1024), stream=stream0)
        buf151 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf153 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_20.run(buf150, squeeze_109, buf151, buf153, 1024, 98, grid=grid(1024), stream=stream0)
        buf152 = buf126; del buf126  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_21.run(relu_33, buf120, buf146, convolution_36, unsqueeze_406, buf151, squeeze_109, buf149, primals_110, buf152, 12845056, grid=grid(12845056), stream=stream0)
        del convolution_36
        del primals_110
        del squeeze_109
        del unsqueeze_406
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf154 = aten.convolution_backward.default(buf152, relu_32, primals_109, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf152
        del primals_109
        buf155 = buf154[0]
        buf156 = buf154[1]
        del buf154
        buf157 = buf141; del buf141  # reuse
        buf159 = buf139; del buf139  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_22.run(relu_32, buf155, convolution_35, unsqueeze_418, buf157, buf159, 25088, 128, grid=grid(25088), stream=stream0)
        buf158 = buf142; del buf142  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_23.run(buf157, buf158, 256, 98, grid=grid(256), stream=stream0)
        buf160 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf161 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_24.run(buf159, squeeze_106, buf160, buf161, 256, 98, grid=grid(256), stream=stream0)
        buf162 = buf155; del buf155  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_25.run(buf162, relu_32, convolution_35, unsqueeze_418, buf160, squeeze_106, buf158, primals_107, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_35
        del primals_107
        del relu_32
        del squeeze_106
        del unsqueeze_418
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf163 = aten.convolution_backward.default(buf162, relu_31, primals_106, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf162
        del primals_106
        buf164 = buf163[0]
        buf165 = buf163[1]
        del buf163
        buf166 = buf159; del buf159  # reuse
        buf168 = buf157; del buf157  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_22.run(relu_31, buf164, convolution_34, unsqueeze_430, buf166, buf168, 25088, 128, grid=grid(25088), stream=stream0)
        buf167 = buf160; del buf160  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_23.run(buf166, buf167, 256, 98, grid=grid(256), stream=stream0)
        buf169 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf170 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_24.run(buf168, squeeze_103, buf169, buf170, 256, 98, grid=grid(256), stream=stream0)
        buf171 = buf164; del buf164  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_25.run(buf171, relu_31, convolution_34, unsqueeze_430, buf169, squeeze_103, buf167, primals_104, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_34
        del primals_104
        del relu_31
        del squeeze_103
        del unsqueeze_430
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf172 = aten.convolution_backward.default(buf171, relu_30, primals_103, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf171
        del primals_103
        buf173 = buf172[0]
        buf174 = buf172[1]
        del buf172
        buf175 = buf120; del buf120  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_29.run(buf175, relu_30, relu_33, buf146, buf173, 12845056, grid=grid(12845056), stream=stream0)
        del buf146
        del relu_30
        del relu_33
        buf176 = buf150; del buf150  # reuse
        buf178 = buf148; del buf148  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_27.run(buf175, convolution_33, unsqueeze_442, buf176, buf178, 100352, 128, grid=grid(100352), stream=stream0)
        buf177 = buf151; del buf151  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_19.run(buf176, buf177, 1024, 98, grid=grid(1024), stream=stream0)
        buf179 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf180 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_20.run(buf178, squeeze_100, buf179, buf180, 1024, 98, grid=grid(1024), stream=stream0)
        buf181 = buf173; del buf173  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_28.run(buf175, convolution_33, unsqueeze_442, buf179, squeeze_100, buf177, primals_101, buf181, 12845056, grid=grid(12845056), stream=stream0)
        del convolution_33
        del primals_101
        del squeeze_100
        del unsqueeze_442
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf182 = aten.convolution_backward.default(buf181, relu_29, primals_100, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_100
        buf183 = buf182[0]
        buf184 = buf182[1]
        del buf182
        buf185 = buf168; del buf168  # reuse
        buf187 = buf166; del buf166  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_22.run(relu_29, buf183, convolution_32, unsqueeze_454, buf185, buf187, 25088, 128, grid=grid(25088), stream=stream0)
        buf186 = buf169; del buf169  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_23.run(buf185, buf186, 256, 98, grid=grid(256), stream=stream0)
        buf188 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf189 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_24.run(buf187, squeeze_97, buf188, buf189, 256, 98, grid=grid(256), stream=stream0)
        buf190 = buf183; del buf183  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_25.run(buf190, relu_29, convolution_32, unsqueeze_454, buf188, squeeze_97, buf186, primals_98, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_32
        del primals_98
        del relu_29
        del squeeze_97
        del unsqueeze_454
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf191 = aten.convolution_backward.default(buf190, relu_28, primals_97, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf190
        del primals_97
        buf192 = buf191[0]
        buf193 = buf191[1]
        del buf191
        buf194 = buf187; del buf187  # reuse
        buf196 = buf185; del buf185  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_22.run(relu_28, buf192, convolution_31, unsqueeze_466, buf194, buf196, 25088, 128, grid=grid(25088), stream=stream0)
        buf195 = buf188; del buf188  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_23.run(buf194, buf195, 256, 98, grid=grid(256), stream=stream0)
        buf197 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf198 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_24.run(buf196, squeeze_94, buf197, buf198, 256, 98, grid=grid(256), stream=stream0)
        buf199 = buf192; del buf192  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_25.run(buf199, relu_28, convolution_31, unsqueeze_466, buf197, squeeze_94, buf195, primals_95, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_31
        del primals_95
        del relu_28
        del squeeze_94
        del unsqueeze_466
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf200 = aten.convolution_backward.default(buf199, relu_27, primals_94, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf199
        del primals_94
        buf201 = buf200[0]
        buf202 = buf200[1]
        del buf200
        buf203 = buf178; del buf178  # reuse
        buf205 = buf176; del buf176  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_18.run(relu_27, buf175, buf201, convolution_30, unsqueeze_478, buf203, buf205, 100352, 128, grid=grid(100352), stream=stream0)
        buf204 = buf179; del buf179  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_19.run(buf203, buf204, 1024, 98, grid=grid(1024), stream=stream0)
        buf206 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf208 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_20.run(buf205, squeeze_91, buf206, buf208, 1024, 98, grid=grid(1024), stream=stream0)
        buf207 = buf181; del buf181  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_21.run(relu_27, buf175, buf201, convolution_30, unsqueeze_478, buf206, squeeze_91, buf204, primals_92, buf207, 12845056, grid=grid(12845056), stream=stream0)
        del convolution_30
        del primals_92
        del squeeze_91
        del unsqueeze_478
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf209 = aten.convolution_backward.default(buf207, relu_26, primals_91, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf207
        del primals_91
        buf210 = buf209[0]
        buf211 = buf209[1]
        del buf209
        buf212 = buf196; del buf196  # reuse
        buf214 = buf194; del buf194  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_22.run(relu_26, buf210, convolution_29, unsqueeze_490, buf212, buf214, 25088, 128, grid=grid(25088), stream=stream0)
        buf213 = buf197; del buf197  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_23.run(buf212, buf213, 256, 98, grid=grid(256), stream=stream0)
        buf215 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf216 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_24.run(buf214, squeeze_88, buf215, buf216, 256, 98, grid=grid(256), stream=stream0)
        buf217 = buf210; del buf210  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_25.run(buf217, relu_26, convolution_29, unsqueeze_490, buf215, squeeze_88, buf213, primals_89, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_29
        del primals_89
        del relu_26
        del squeeze_88
        del unsqueeze_490
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf218 = aten.convolution_backward.default(buf217, relu_25, primals_88, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf217
        del primals_88
        buf219 = buf218[0]
        buf220 = buf218[1]
        del buf218
        buf221 = buf214; del buf214  # reuse
        buf223 = buf212; del buf212  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_22.run(relu_25, buf219, convolution_28, unsqueeze_502, buf221, buf223, 25088, 128, grid=grid(25088), stream=stream0)
        buf222 = buf215; del buf215  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_23.run(buf221, buf222, 256, 98, grid=grid(256), stream=stream0)
        buf224 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf225 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_24.run(buf223, squeeze_85, buf224, buf225, 256, 98, grid=grid(256), stream=stream0)
        buf226 = buf219; del buf219  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_25.run(buf226, relu_25, convolution_28, unsqueeze_502, buf224, squeeze_85, buf222, primals_86, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_28
        del primals_86
        del relu_25
        del squeeze_85
        del unsqueeze_502
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf227 = aten.convolution_backward.default(buf226, relu_24, primals_85, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf226
        del primals_85
        buf228 = buf227[0]
        buf229 = buf227[1]
        del buf227
        buf230 = buf175; del buf175  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_29.run(buf230, relu_24, relu_27, buf201, buf228, 12845056, grid=grid(12845056), stream=stream0)
        del relu_24
        del relu_27
        buf231 = buf205; del buf205  # reuse
        buf233 = buf203; del buf203  # reuse
        buf240 = empty_strided_cuda((1024, 98), (1, 1024), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_30.run(buf230, convolution_27, unsqueeze_514, convolution_26, unsqueeze_526, buf231, buf233, buf240, 100352, 128, grid=grid(100352), stream=stream0)
        buf232 = buf206; del buf206  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_19.run(buf231, buf232, 1024, 98, grid=grid(1024), stream=stream0)
        del buf231
        buf234 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf235 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_20.run(buf233, squeeze_82, buf234, buf235, 1024, 98, grid=grid(1024), stream=stream0)
        buf241 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf242 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_20.run(buf240, squeeze_79, buf241, buf242, 1024, 98, grid=grid(1024), stream=stream0)
        buf236 = buf228; del buf228  # reuse
        buf243 = buf201; del buf201  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_31.run(buf230, convolution_27, unsqueeze_514, buf234, squeeze_82, buf232, primals_83, convolution_26, unsqueeze_526, buf241, squeeze_79, primals_80, buf236, buf243, 12845056, grid=grid(12845056), stream=stream0)
        del buf230
        del buf234
        del buf241
        del convolution_26
        del convolution_27
        del primals_80
        del primals_83
        del squeeze_79
        del squeeze_82
        del unsqueeze_514
        del unsqueeze_526
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf237 = aten.convolution_backward.default(buf236, relu_21, primals_82, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf236
        del primals_82
        buf238 = buf237[0]
        buf239 = buf237[1]
        del buf237
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf244 = aten.convolution_backward.default(buf243, relu_23, primals_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf243
        del primals_79
        buf245 = buf244[0]
        buf246 = buf244[1]
        del buf244
        buf247 = buf223; del buf223  # reuse
        buf249 = buf221; del buf221  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_22.run(relu_23, buf245, convolution_25, unsqueeze_538, buf247, buf249, 25088, 128, grid=grid(25088), stream=stream0)
        buf248 = buf224; del buf224  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_23.run(buf247, buf248, 256, 98, grid=grid(256), stream=stream0)
        del buf247
        buf250 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf251 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_24.run(buf249, squeeze_76, buf250, buf251, 256, 98, grid=grid(256), stream=stream0)
        del buf249
        buf252 = buf245; del buf245  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_25.run(buf252, relu_23, convolution_25, unsqueeze_538, buf250, squeeze_76, buf248, primals_77, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_25
        del primals_77
        del relu_23
        del squeeze_76
        del unsqueeze_538
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf253 = aten.convolution_backward.default(buf252, relu_22, primals_76, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf252
        del primals_76
        buf254 = buf253[0]
        buf255 = buf253[1]
        del buf253
        buf256 = reinterpret_tensor(buf240, (256, 392), (1, 256), 0); del buf240  # reuse
        buf258 = reinterpret_tensor(buf233, (256, 392), (1, 256), 0); del buf233  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_32.run(relu_22, buf254, convolution_24, unsqueeze_550, buf256, buf258, 100352, 128, grid=grid(100352), stream=stream0)
        buf257 = buf250; del buf250  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_33.run(buf256, buf257, 256, 392, grid=grid(256), stream=stream0)
        del buf256
        buf259 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf260 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_34.run(buf258, squeeze_73, buf259, buf260, 256, 392, grid=grid(256), stream=stream0)
        del buf258
        buf261 = buf254; del buf254  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_35.run(buf261, relu_22, convolution_24, unsqueeze_550, buf259, squeeze_73, buf257, primals_74, 12845056, grid=grid(12845056), stream=stream0)
        del convolution_24
        del primals_74
        del relu_22
        del squeeze_73
        del unsqueeze_550
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf262 = aten.convolution_backward.default(buf261, relu_21, primals_73, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf261
        del primals_73
        buf263 = buf262[0]
        buf264 = buf262[1]
        del buf262
        buf265 = empty_strided_cuda((512, 224), (1, 512), torch.float32)
        buf267 = empty_strided_cuda((512, 224), (1, 512), torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_36.run(relu_21, buf238, buf263, convolution_23, unsqueeze_562, buf265, buf267, 114688, 224, grid=grid(114688), stream=stream0)
        buf266 = buf87; del buf87  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_37.run(buf265, buf266, 512, 224, grid=grid(512), stream=stream0)
        buf268 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf270 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_38.run(buf267, squeeze_70, buf268, buf270, 512, 224, grid=grid(512), stream=stream0)
        buf269 = empty_strided_cuda((64, 512, 28, 28), (401408, 1, 14336, 512), torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_39.run(relu_21, buf238, buf263, convolution_23, unsqueeze_562, buf268, squeeze_70, buf266, primals_71, buf269, 25690112, grid=grid(25690112), stream=stream0)
        del convolution_23
        del primals_71
        del squeeze_70
        del unsqueeze_562
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf271 = aten.convolution_backward.default(buf269, relu_20, primals_70, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf269
        del primals_70
        buf272 = buf271[0]
        buf273 = buf271[1]
        del buf271
        buf274 = reinterpret_tensor(buf86, (128, 392), (1, 128), 0); del buf86  # reuse
        buf276 = reinterpret_tensor(buf84, (128, 392), (1, 128), 0); del buf84  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_40.run(relu_20, buf272, convolution_22, unsqueeze_574, buf274, buf276, 50176, 128, grid=grid(50176), stream=stream0)
        buf275 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_41.run(buf274, buf275, 128, 392, grid=grid(128), stream=stream0)
        buf277 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf278 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_42.run(buf276, squeeze_67, buf277, buf278, 128, 392, grid=grid(128), stream=stream0)
        buf279 = buf272; del buf272  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43.run(buf279, relu_20, convolution_22, unsqueeze_574, buf277, squeeze_67, buf275, primals_68, 6422528, grid=grid(6422528), stream=stream0)
        del convolution_22
        del primals_68
        del relu_20
        del squeeze_67
        del unsqueeze_574
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf280 = aten.convolution_backward.default(buf279, relu_19, primals_67, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf279
        del primals_67
        buf281 = buf280[0]
        buf282 = buf280[1]
        del buf280
        buf283 = buf276; del buf276  # reuse
        buf285 = buf274; del buf274  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_40.run(relu_19, buf281, convolution_21, unsqueeze_586, buf283, buf285, 50176, 128, grid=grid(50176), stream=stream0)
        buf284 = buf277; del buf277  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_41.run(buf283, buf284, 128, 392, grid=grid(128), stream=stream0)
        buf286 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf287 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_42.run(buf285, squeeze_64, buf286, buf287, 128, 392, grid=grid(128), stream=stream0)
        buf288 = buf281; del buf281  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43.run(buf288, relu_19, convolution_21, unsqueeze_586, buf286, squeeze_64, buf284, primals_65, 6422528, grid=grid(6422528), stream=stream0)
        del convolution_21
        del primals_65
        del relu_19
        del squeeze_64
        del unsqueeze_586
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf289 = aten.convolution_backward.default(buf288, relu_18, primals_64, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf288
        del primals_64
        buf290 = buf289[0]
        buf291 = buf289[1]
        del buf289
        buf292 = buf238; del buf238  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_44.run(buf292, relu_18, relu_21, buf263, buf290, 25690112, grid=grid(25690112), stream=stream0)
        del buf263
        del relu_18
        del relu_21
        buf293 = buf267; del buf267  # reuse
        buf295 = buf265; del buf265  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_45.run(buf292, convolution_20, unsqueeze_598, buf293, buf295, 114688, 224, grid=grid(114688), stream=stream0)
        buf294 = buf268; del buf268  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_37.run(buf293, buf294, 512, 224, grid=grid(512), stream=stream0)
        buf296 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf297 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_38.run(buf295, squeeze_61, buf296, buf297, 512, 224, grid=grid(512), stream=stream0)
        buf298 = buf290; del buf290  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_46.run(buf292, convolution_20, unsqueeze_598, buf296, squeeze_61, buf294, primals_62, buf298, 25690112, grid=grid(25690112), stream=stream0)
        del convolution_20
        del primals_62
        del squeeze_61
        del unsqueeze_598
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf299 = aten.convolution_backward.default(buf298, relu_17, primals_61, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_61
        buf300 = buf299[0]
        buf301 = buf299[1]
        del buf299
        buf302 = buf285; del buf285  # reuse
        buf304 = buf283; del buf283  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_40.run(relu_17, buf300, convolution_19, unsqueeze_610, buf302, buf304, 50176, 128, grid=grid(50176), stream=stream0)
        buf303 = buf286; del buf286  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_41.run(buf302, buf303, 128, 392, grid=grid(128), stream=stream0)
        buf305 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf306 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_42.run(buf304, squeeze_58, buf305, buf306, 128, 392, grid=grid(128), stream=stream0)
        buf307 = buf300; del buf300  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43.run(buf307, relu_17, convolution_19, unsqueeze_610, buf305, squeeze_58, buf303, primals_59, 6422528, grid=grid(6422528), stream=stream0)
        del convolution_19
        del primals_59
        del relu_17
        del squeeze_58
        del unsqueeze_610
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf308 = aten.convolution_backward.default(buf307, relu_16, primals_58, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf307
        del primals_58
        buf309 = buf308[0]
        buf310 = buf308[1]
        del buf308
        buf311 = buf304; del buf304  # reuse
        buf313 = buf302; del buf302  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_40.run(relu_16, buf309, convolution_18, unsqueeze_622, buf311, buf313, 50176, 128, grid=grid(50176), stream=stream0)
        buf312 = buf305; del buf305  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_41.run(buf311, buf312, 128, 392, grid=grid(128), stream=stream0)
        buf314 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf315 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_42.run(buf313, squeeze_55, buf314, buf315, 128, 392, grid=grid(128), stream=stream0)
        buf316 = buf309; del buf309  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43.run(buf316, relu_16, convolution_18, unsqueeze_622, buf314, squeeze_55, buf312, primals_56, 6422528, grid=grid(6422528), stream=stream0)
        del convolution_18
        del primals_56
        del relu_16
        del squeeze_55
        del unsqueeze_622
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf317 = aten.convolution_backward.default(buf316, relu_15, primals_55, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf316
        del primals_55
        buf318 = buf317[0]
        buf319 = buf317[1]
        del buf317
        buf320 = buf295; del buf295  # reuse
        buf322 = buf293; del buf293  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_36.run(relu_15, buf292, buf318, convolution_17, unsqueeze_634, buf320, buf322, 114688, 224, grid=grid(114688), stream=stream0)
        buf321 = buf296; del buf296  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_37.run(buf320, buf321, 512, 224, grid=grid(512), stream=stream0)
        buf323 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf325 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_38.run(buf322, squeeze_52, buf323, buf325, 512, 224, grid=grid(512), stream=stream0)
        buf324 = buf298; del buf298  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_39.run(relu_15, buf292, buf318, convolution_17, unsqueeze_634, buf323, squeeze_52, buf321, primals_53, buf324, 25690112, grid=grid(25690112), stream=stream0)
        del convolution_17
        del primals_53
        del squeeze_52
        del unsqueeze_634
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf326 = aten.convolution_backward.default(buf324, relu_14, primals_52, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf324
        del primals_52
        buf327 = buf326[0]
        buf328 = buf326[1]
        del buf326
        buf329 = buf313; del buf313  # reuse
        buf331 = buf311; del buf311  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_40.run(relu_14, buf327, convolution_16, unsqueeze_646, buf329, buf331, 50176, 128, grid=grid(50176), stream=stream0)
        buf330 = buf314; del buf314  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_41.run(buf329, buf330, 128, 392, grid=grid(128), stream=stream0)
        buf332 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf333 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_42.run(buf331, squeeze_49, buf332, buf333, 128, 392, grid=grid(128), stream=stream0)
        buf334 = buf327; del buf327  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43.run(buf334, relu_14, convolution_16, unsqueeze_646, buf332, squeeze_49, buf330, primals_50, 6422528, grid=grid(6422528), stream=stream0)
        del convolution_16
        del primals_50
        del relu_14
        del squeeze_49
        del unsqueeze_646
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf335 = aten.convolution_backward.default(buf334, relu_13, primals_49, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf334
        del primals_49
        buf336 = buf335[0]
        buf337 = buf335[1]
        del buf335
        buf338 = buf331; del buf331  # reuse
        buf340 = buf329; del buf329  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_40.run(relu_13, buf336, convolution_15, unsqueeze_658, buf338, buf340, 50176, 128, grid=grid(50176), stream=stream0)
        buf339 = buf332; del buf332  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_41.run(buf338, buf339, 128, 392, grid=grid(128), stream=stream0)
        buf341 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf342 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_42.run(buf340, squeeze_46, buf341, buf342, 128, 392, grid=grid(128), stream=stream0)
        buf343 = buf336; del buf336  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43.run(buf343, relu_13, convolution_15, unsqueeze_658, buf341, squeeze_46, buf339, primals_47, 6422528, grid=grid(6422528), stream=stream0)
        del convolution_15
        del primals_47
        del relu_13
        del squeeze_46
        del unsqueeze_658
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf344 = aten.convolution_backward.default(buf343, relu_12, primals_46, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf343
        del primals_46
        buf345 = buf344[0]
        buf346 = buf344[1]
        del buf344
        buf347 = buf292; del buf292  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_44.run(buf347, relu_12, relu_15, buf318, buf345, 25690112, grid=grid(25690112), stream=stream0)
        del relu_12
        del relu_15
        buf348 = buf322; del buf322  # reuse
        buf350 = buf320; del buf320  # reuse
        buf357 = empty_strided_cuda((512, 224), (1, 512), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_47.run(buf347, convolution_14, unsqueeze_670, convolution_13, unsqueeze_682, buf348, buf350, buf357, 114688, 224, grid=grid(114688), stream=stream0)
        buf349 = buf323; del buf323  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_37.run(buf348, buf349, 512, 224, grid=grid(512), stream=stream0)
        buf351 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf352 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_38.run(buf350, squeeze_43, buf351, buf352, 512, 224, grid=grid(512), stream=stream0)
        buf358 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf359 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_38.run(buf357, squeeze_40, buf358, buf359, 512, 224, grid=grid(512), stream=stream0)
        buf353 = buf345; del buf345  # reuse
        buf360 = buf318; del buf318  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_48.run(buf347, convolution_14, unsqueeze_670, buf351, squeeze_43, buf349, primals_44, convolution_13, unsqueeze_682, buf358, squeeze_40, primals_41, buf353, buf360, 25690112, grid=grid(25690112), stream=stream0)
        del buf347
        del buf351
        del buf358
        del convolution_13
        del convolution_14
        del primals_41
        del primals_44
        del squeeze_40
        del squeeze_43
        del unsqueeze_670
        del unsqueeze_682
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf354 = aten.convolution_backward.default(buf353, relu_9, primals_43, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf353
        del primals_43
        buf355 = buf354[0]
        buf356 = buf354[1]
        del buf354
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf361 = aten.convolution_backward.default(buf360, relu_11, primals_40, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf360
        del primals_40
        buf362 = buf361[0]
        buf363 = buf361[1]
        del buf361
        buf364 = buf340; del buf340  # reuse
        buf366 = buf338; del buf338  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_40.run(relu_11, buf362, convolution_12, unsqueeze_694, buf364, buf366, 50176, 128, grid=grid(50176), stream=stream0)
        buf365 = buf341; del buf341  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_41.run(buf364, buf365, 128, 392, grid=grid(128), stream=stream0)
        del buf364
        buf367 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf368 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_42.run(buf366, squeeze_37, buf367, buf368, 128, 392, grid=grid(128), stream=stream0)
        del buf366
        buf369 = buf362; del buf362  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43.run(buf369, relu_11, convolution_12, unsqueeze_694, buf367, squeeze_37, buf365, primals_38, 6422528, grid=grid(6422528), stream=stream0)
        del convolution_12
        del primals_38
        del relu_11
        del squeeze_37
        del unsqueeze_694
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf370 = aten.convolution_backward.default(buf369, relu_10, primals_37, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf369
        del primals_37
        buf371 = buf370[0]
        buf372 = buf370[1]
        del buf370
        buf373 = reinterpret_tensor(buf357, (128, 896), (1, 128), 0); del buf357  # reuse
        buf375 = reinterpret_tensor(buf350, (128, 896), (1, 128), 0); del buf350  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_49.run(relu_10, buf371, convolution_11, unsqueeze_706, buf373, buf375, 114688, 224, grid=grid(114688), stream=stream0)
        buf374 = buf367; del buf367  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_50.run(buf373, buf374, 128, 896, grid=grid(128), stream=stream0)
        buf376 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf377 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_51.run(buf375, squeeze_34, buf376, buf377, 128, 896, grid=grid(128), stream=stream0)
        buf378 = buf371; del buf371  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_52.run(buf378, relu_10, convolution_11, unsqueeze_706, buf376, squeeze_34, buf374, primals_35, 25690112, grid=grid(25690112), stream=stream0)
        del buf376
        del convolution_11
        del primals_35
        del relu_10
        del squeeze_34
        del unsqueeze_706
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf379 = aten.convolution_backward.default(buf378, relu_9, primals_34, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf378
        del primals_34
        buf380 = buf379[0]
        buf381 = buf379[1]
        del buf379
        buf382 = reinterpret_tensor(buf375, (256, 448), (1, 256), 0); del buf375  # reuse
        buf384 = reinterpret_tensor(buf373, (256, 448), (1, 256), 0); del buf373  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_53.run(relu_9, buf355, buf380, convolution_10, unsqueeze_718, buf382, buf384, 114688, 448, grid=grid(114688), stream=stream0)
        buf383 = buf259; del buf259  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_54.run(buf382, buf383, 256, 448, grid=grid(256), stream=stream0)
        buf385 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf387 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_55.run(buf384, squeeze_31, buf385, buf387, 256, 448, grid=grid(256), stream=stream0)
        buf386 = empty_strided_cuda((64, 256, 56, 56), (802816, 1, 14336, 256), torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_56.run(relu_9, buf355, buf380, convolution_10, unsqueeze_718, buf385, squeeze_31, buf383, primals_32, buf386, 51380224, grid=grid(51380224), stream=stream0)
        del convolution_10
        del primals_32
        del squeeze_31
        del unsqueeze_718
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf388 = aten.convolution_backward.default(buf386, relu_8, primals_31, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf386
        del primals_31
        buf389 = buf388[0]
        buf390 = buf388[1]
        del buf388
        buf391 = empty_strided_cuda((64, 896), (1, 64), torch.float32)
        buf393 = empty_strided_cuda((64, 896), (1, 64), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_57.run(relu_8, buf389, convolution_9, unsqueeze_730, buf391, buf393, 57344, 224, grid=grid(57344), stream=stream0)
        buf392 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_58.run(buf391, buf392, 64, 896, grid=grid(64), stream=stream0)
        buf394 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf395 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_59.run(buf393, squeeze_28, buf394, buf395, 64, 896, grid=grid(64), stream=stream0)
        buf396 = buf389; del buf389  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_60.run(buf396, relu_8, convolution_9, unsqueeze_730, buf394, squeeze_28, buf392, primals_29, 12845056, grid=grid(12845056), stream=stream0)
        del convolution_9
        del primals_29
        del relu_8
        del squeeze_28
        del unsqueeze_730
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf397 = aten.convolution_backward.default(buf396, relu_7, primals_28, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf396
        del primals_28
        buf398 = buf397[0]
        buf399 = buf397[1]
        del buf397
        buf400 = buf393; del buf393  # reuse
        buf402 = buf391; del buf391  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_57.run(relu_7, buf398, convolution_8, unsqueeze_742, buf400, buf402, 57344, 224, grid=grid(57344), stream=stream0)
        buf401 = buf394; del buf394  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_58.run(buf400, buf401, 64, 896, grid=grid(64), stream=stream0)
        buf403 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf404 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_59.run(buf402, squeeze_25, buf403, buf404, 64, 896, grid=grid(64), stream=stream0)
        buf405 = buf398; del buf398  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_60.run(buf405, relu_7, convolution_8, unsqueeze_742, buf403, squeeze_25, buf401, primals_26, 12845056, grid=grid(12845056), stream=stream0)
        del convolution_8
        del primals_26
        del relu_7
        del squeeze_25
        del unsqueeze_742
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf406 = aten.convolution_backward.default(buf405, relu_6, primals_25, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf405
        del primals_25
        buf407 = buf406[0]
        buf408 = buf406[1]
        del buf406
        buf409 = buf355; del buf355  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_61.run(buf409, relu_6, relu_9, buf380, buf407, 51380224, grid=grid(51380224), stream=stream0)
        del relu_6
        del relu_9
        buf410 = buf384; del buf384  # reuse
        buf412 = buf382; del buf382  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_62.run(buf409, convolution_7, unsqueeze_754, buf410, buf412, 114688, 448, grid=grid(114688), stream=stream0)
        buf411 = buf385; del buf385  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_54.run(buf410, buf411, 256, 448, grid=grid(256), stream=stream0)
        buf413 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf414 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_55.run(buf412, squeeze_22, buf413, buf414, 256, 448, grid=grid(256), stream=stream0)
        buf415 = buf407; del buf407  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_63.run(buf409, convolution_7, unsqueeze_754, buf413, squeeze_22, buf411, primals_23, buf415, 51380224, grid=grid(51380224), stream=stream0)
        del convolution_7
        del primals_23
        del squeeze_22
        del unsqueeze_754
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf416 = aten.convolution_backward.default(buf415, relu_5, primals_22, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_22
        buf417 = buf416[0]
        buf418 = buf416[1]
        del buf416
        buf419 = buf402; del buf402  # reuse
        buf421 = buf400; del buf400  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_57.run(relu_5, buf417, convolution_6, unsqueeze_766, buf419, buf421, 57344, 224, grid=grid(57344), stream=stream0)
        buf420 = buf403; del buf403  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_58.run(buf419, buf420, 64, 896, grid=grid(64), stream=stream0)
        buf422 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf423 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_59.run(buf421, squeeze_19, buf422, buf423, 64, 896, grid=grid(64), stream=stream0)
        buf424 = buf417; del buf417  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_60.run(buf424, relu_5, convolution_6, unsqueeze_766, buf422, squeeze_19, buf420, primals_20, 12845056, grid=grid(12845056), stream=stream0)
        del convolution_6
        del primals_20
        del relu_5
        del squeeze_19
        del unsqueeze_766
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf425 = aten.convolution_backward.default(buf424, relu_4, primals_19, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf424
        del primals_19
        buf426 = buf425[0]
        buf427 = buf425[1]
        del buf425
        buf428 = buf421; del buf421  # reuse
        buf430 = buf419; del buf419  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_57.run(relu_4, buf426, convolution_5, unsqueeze_778, buf428, buf430, 57344, 224, grid=grid(57344), stream=stream0)
        buf429 = buf422; del buf422  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_58.run(buf428, buf429, 64, 896, grid=grid(64), stream=stream0)
        buf431 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf432 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_59.run(buf430, squeeze_16, buf431, buf432, 64, 896, grid=grid(64), stream=stream0)
        buf433 = buf426; del buf426  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_60.run(buf433, relu_4, convolution_5, unsqueeze_778, buf431, squeeze_16, buf429, primals_17, 12845056, grid=grid(12845056), stream=stream0)
        del convolution_5
        del primals_17
        del relu_4
        del squeeze_16
        del unsqueeze_778
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf434 = aten.convolution_backward.default(buf433, relu_3, primals_16, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf433
        del primals_16
        buf435 = buf434[0]
        buf436 = buf434[1]
        del buf434
        buf437 = buf412; del buf412  # reuse
        buf439 = buf410; del buf410  # reuse
        buf446 = reinterpret_tensor(buf348, (256, 448), (1, 256), 0); del buf348  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_64.run(relu_3, buf409, buf435, convolution_4, unsqueeze_790, convolution_3, unsqueeze_802, buf437, buf439, buf446, 114688, 448, grid=grid(114688), stream=stream0)
        buf438 = buf413; del buf413  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_54.run(buf437, buf438, 256, 448, grid=grid(256), stream=stream0)
        del buf437
        buf440 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf442 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_55.run(buf439, squeeze_13, buf440, buf442, 256, 448, grid=grid(256), stream=stream0)
        del buf439
        buf447 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf449 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_55.run(buf446, squeeze_10, buf447, buf449, 256, 448, grid=grid(256), stream=stream0)
        del buf446
        buf441 = buf415; del buf415  # reuse
        buf448 = buf380; del buf380  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_65.run(relu_3, buf409, buf435, convolution_4, unsqueeze_790, buf440, squeeze_13, buf438, primals_14, convolution_3, unsqueeze_802, buf447, squeeze_10, primals_11, buf441, buf448, 51380224, grid=grid(51380224), stream=stream0)
        del buf409
        del buf435
        del buf440
        del buf447
        del convolution_3
        del convolution_4
        del primals_11
        del primals_14
        del relu_3
        del squeeze_10
        del squeeze_13
        del unsqueeze_790
        del unsqueeze_802
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf443 = aten.convolution_backward.default(buf441, getitem_2, primals_13, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf441
        del primals_13
        buf444 = buf443[0]
        buf445 = buf443[1]
        del buf443
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf450 = aten.convolution_backward.default(buf448, relu_2, primals_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf448
        del primals_10
        buf451 = buf450[0]
        buf452 = buf450[1]
        del buf450
        buf453 = buf430; del buf430  # reuse
        buf455 = buf428; del buf428  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_57.run(relu_2, buf451, convolution_2, unsqueeze_814, buf453, buf455, 57344, 224, grid=grid(57344), stream=stream0)
        buf454 = buf431; del buf431  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_58.run(buf453, buf454, 64, 896, grid=grid(64), stream=stream0)
        buf456 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf457 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_59.run(buf455, squeeze_7, buf456, buf457, 64, 896, grid=grid(64), stream=stream0)
        buf458 = buf451; del buf451  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_60.run(buf458, relu_2, convolution_2, unsqueeze_814, buf456, squeeze_7, buf454, primals_8, 12845056, grid=grid(12845056), stream=stream0)
        del convolution_2
        del primals_8
        del relu_2
        del squeeze_7
        del unsqueeze_814
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf459 = aten.convolution_backward.default(buf458, relu_1, primals_7, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf458
        del primals_7
        buf460 = buf459[0]
        buf461 = buf459[1]
        del buf459
        buf462 = buf455; del buf455  # reuse
        buf464 = buf453; del buf453  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_57.run(relu_1, buf460, convolution_1, unsqueeze_826, buf462, buf464, 57344, 224, grid=grid(57344), stream=stream0)
        buf463 = buf456; del buf456  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_58.run(buf462, buf463, 64, 896, grid=grid(64), stream=stream0)
        buf465 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf466 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_59.run(buf464, squeeze_4, buf465, buf466, 64, 896, grid=grid(64), stream=stream0)
        buf467 = buf460; del buf460  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_60.run(buf467, relu_1, convolution_1, unsqueeze_826, buf465, squeeze_4, buf463, primals_5, 12845056, grid=grid(12845056), stream=stream0)
        del convolution_1
        del primals_5
        del relu_1
        del squeeze_4
        del unsqueeze_826
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf468 = aten.convolution_backward.default(buf467, getitem_2, primals_4, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf467
        del getitem_2
        del primals_4
        buf469 = buf468[0]
        buf470 = buf468[1]
        del buf468
        buf471 = buf444; del buf444  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_66.run(buf471, buf469, 12845056, grid=grid(12845056), stream=stream0)
        del buf469
        # Source Nodes: [], Original ATen: [aten.add, aten.max_pool2d_with_indices_backward]
        buf472 = aten.max_pool2d_with_indices_backward.default(buf471, relu, [3, 3], [2, 2], [1, 1], [1, 1], False, getitem_3)
        del buf471
        del getitem_3
        buf473 = buf472
        del buf472
        buf474 = buf464; del buf464  # reuse
        buf476 = buf462; del buf462  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_67.run(relu, buf473, convolution, unsqueeze_838, buf474, buf476, 57344, 896, grid=grid(57344), stream=stream0)
        buf475 = buf465; del buf465  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_58.run(buf474, buf475, 64, 896, grid=grid(64), stream=stream0)
        del buf474
        buf477 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf478 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_59.run(buf476, squeeze_1, buf477, buf478, 64, 896, grid=grid(64), stream=stream0)
        del buf476
        buf479 = buf473; del buf473  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_68.run(buf479, relu, convolution, unsqueeze_838, buf477, squeeze_1, buf475, primals_2, 51380224, grid=grid(51380224), stream=stream0)
        del buf477
        del convolution
        del primals_2
        del relu
        del squeeze_1
        del unsqueeze_838
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf480 = aten.convolution_backward.default(buf479, primals_321, primals_1, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf479
        del primals_1
        del primals_321
        buf481 = buf480[1]
        del buf480
    return (buf481, buf478, buf475, buf470, buf466, buf463, buf461, buf457, buf454, buf452, buf449, buf438, buf445, buf442, buf438, buf436, buf432, buf429, buf427, buf423, buf420, buf418, buf414, buf411, buf408, buf404, buf401, buf399, buf395, buf392, buf390, buf387, buf383, buf381, buf377, buf374, buf372, buf368, buf365, buf363, buf359, buf349, buf356, buf352, buf349, buf346, buf342, buf339, buf337, buf333, buf330, buf328, buf325, buf321, buf319, buf315, buf312, buf310, buf306, buf303, buf301, buf297, buf294, buf291, buf287, buf284, buf282, buf278, buf275, buf273, buf270, buf266, buf264, buf260, buf257, buf255, buf251, buf248, buf246, buf242, buf232, buf239, buf235, buf232, buf229, buf225, buf222, buf220, buf216, buf213, buf211, buf208, buf204, buf202, buf198, buf195, buf193, buf189, buf186, buf184, buf180, buf177, buf174, buf170, buf167, buf165, buf161, buf158, buf156, buf153, buf149, buf147, buf143, buf140, buf138, buf134, buf131, buf129, buf125, buf122, buf119, buf115, buf112, buf110, buf106, buf103, buf101, buf98, buf94, buf92, buf88, buf85, buf83, buf79, buf76, buf74, buf70, buf60, buf67, buf63, buf60, buf57, buf53, buf50, buf48, buf44, buf41, buf39, buf35, buf31, buf29, buf25, buf22, buf20, buf16, buf13, buf11, buf7, buf4, reinterpret_tensor(buf1, (1000, 2048), (2048, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 7, 7), (147, 1, 21, 3), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((64, 3, 224, 224), (150528, 1, 672, 3), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((64, 64, 112, 112), (802816, 1, 7168, 64), device='cuda:0', dtype=torch.float32)
    squeeze_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu = rand_strided((64, 64, 112, 112), (802816, 1, 7168, 64), device='cuda:0', dtype=torch.float32)
    getitem_2 = rand_strided((64, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    getitem_3 = rand_strided((64, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.int64)
    convolution_1 = rand_strided((64, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    squeeze_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_1 = rand_strided((64, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((64, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    squeeze_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_2 = rand_strided((64, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    convolution_3 = rand_strided((64, 256, 56, 56), (802816, 1, 14336, 256), device='cuda:0', dtype=torch.float32)
    squeeze_10 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_4 = rand_strided((64, 256, 56, 56), (802816, 1, 14336, 256), device='cuda:0', dtype=torch.float32)
    squeeze_13 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_3 = rand_strided((64, 256, 56, 56), (802816, 1, 14336, 256), device='cuda:0', dtype=torch.float32)
    convolution_5 = rand_strided((64, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    squeeze_16 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_4 = rand_strided((64, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    convolution_6 = rand_strided((64, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    squeeze_19 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_5 = rand_strided((64, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    convolution_7 = rand_strided((64, 256, 56, 56), (802816, 1, 14336, 256), device='cuda:0', dtype=torch.float32)
    squeeze_22 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_6 = rand_strided((64, 256, 56, 56), (802816, 1, 14336, 256), device='cuda:0', dtype=torch.float32)
    convolution_8 = rand_strided((64, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    squeeze_25 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_7 = rand_strided((64, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    convolution_9 = rand_strided((64, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    squeeze_28 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_8 = rand_strided((64, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    convolution_10 = rand_strided((64, 256, 56, 56), (802816, 1, 14336, 256), device='cuda:0', dtype=torch.float32)
    squeeze_31 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_9 = rand_strided((64, 256, 56, 56), (802816, 1, 14336, 256), device='cuda:0', dtype=torch.float32)
    convolution_11 = rand_strided((64, 128, 56, 56), (401408, 1, 7168, 128), device='cuda:0', dtype=torch.float32)
    squeeze_34 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_10 = rand_strided((64, 128, 56, 56), (401408, 1, 7168, 128), device='cuda:0', dtype=torch.float32)
    convolution_12 = rand_strided((64, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    squeeze_37 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_11 = rand_strided((64, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    convolution_13 = rand_strided((64, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    squeeze_40 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_14 = rand_strided((64, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    squeeze_43 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_12 = rand_strided((64, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    convolution_15 = rand_strided((64, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    squeeze_46 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_13 = rand_strided((64, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    convolution_16 = rand_strided((64, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    squeeze_49 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_14 = rand_strided((64, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    convolution_17 = rand_strided((64, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    squeeze_52 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_15 = rand_strided((64, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    convolution_18 = rand_strided((64, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    squeeze_55 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_16 = rand_strided((64, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    convolution_19 = rand_strided((64, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    squeeze_58 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_17 = rand_strided((64, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    convolution_20 = rand_strided((64, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    squeeze_61 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_18 = rand_strided((64, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    convolution_21 = rand_strided((64, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    squeeze_64 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_19 = rand_strided((64, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    convolution_22 = rand_strided((64, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    squeeze_67 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_20 = rand_strided((64, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    convolution_23 = rand_strided((64, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    squeeze_70 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_21 = rand_strided((64, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    convolution_24 = rand_strided((64, 256, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.float32)
    squeeze_73 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_22 = rand_strided((64, 256, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.float32)
    convolution_25 = rand_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_76 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_23 = rand_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_26 = rand_strided((64, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_79 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_27 = rand_strided((64, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_82 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_24 = rand_strided((64, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    convolution_28 = rand_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_85 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_25 = rand_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_29 = rand_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_88 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_26 = rand_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_30 = rand_strided((64, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_91 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_27 = rand_strided((64, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    convolution_31 = rand_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_94 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_28 = rand_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_32 = rand_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_97 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_29 = rand_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_33 = rand_strided((64, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_100 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_30 = rand_strided((64, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    convolution_34 = rand_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_103 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_31 = rand_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_35 = rand_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_106 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_32 = rand_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_36 = rand_strided((64, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_109 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_33 = rand_strided((64, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    convolution_37 = rand_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_112 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_34 = rand_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_38 = rand_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_115 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_35 = rand_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_39 = rand_strided((64, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_118 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_36 = rand_strided((64, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    convolution_40 = rand_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_121 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_37 = rand_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_41 = rand_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_124 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_38 = rand_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_42 = rand_strided((64, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_127 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_39 = rand_strided((64, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    convolution_43 = rand_strided((64, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    squeeze_130 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_40 = rand_strided((64, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    convolution_44 = rand_strided((64, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.float32)
    squeeze_133 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_41 = rand_strided((64, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.float32)
    convolution_45 = rand_strided((64, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda:0', dtype=torch.float32)
    squeeze_136 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_46 = rand_strided((64, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda:0', dtype=torch.float32)
    squeeze_139 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_42 = rand_strided((64, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda:0', dtype=torch.float32)
    convolution_47 = rand_strided((64, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.float32)
    squeeze_142 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_43 = rand_strided((64, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.float32)
    convolution_48 = rand_strided((64, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.float32)
    squeeze_145 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_44 = rand_strided((64, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.float32)
    convolution_49 = rand_strided((64, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda:0', dtype=torch.float32)
    squeeze_148 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_45 = rand_strided((64, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda:0', dtype=torch.float32)
    convolution_50 = rand_strided((64, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.float32)
    squeeze_151 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_46 = rand_strided((64, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.float32)
    convolution_51 = rand_strided((64, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.float32)
    squeeze_154 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_47 = rand_strided((64, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.float32)
    convolution_52 = rand_strided((64, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda:0', dtype=torch.float32)
    squeeze_157 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    view = rand_strided((64, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_1 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    le = rand_strided((64, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda:0', dtype=torch.bool)
    unsqueeze_214 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_226 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_238 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_250 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_262 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_274 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_286 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_298 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_310 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_322 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_334 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_346 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_358 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_370 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_382 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_394 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_406 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_418 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_430 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_442 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_454 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_466 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_478 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_490 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_502 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_514 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_526 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_538 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_550 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_562 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_574 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_586 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_598 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_610 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_622 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_634 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_646 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_658 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_670 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_682 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_694 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_706 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_718 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_730 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_742 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_754 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_766 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_778 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_790 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_802 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_814 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_826 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_838 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((64, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_321, convolution, squeeze_1, relu, getitem_2, getitem_3, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, convolution_3, squeeze_10, convolution_4, squeeze_13, relu_3, convolution_5, squeeze_16, relu_4, convolution_6, squeeze_19, relu_5, convolution_7, squeeze_22, relu_6, convolution_8, squeeze_25, relu_7, convolution_9, squeeze_28, relu_8, convolution_10, squeeze_31, relu_9, convolution_11, squeeze_34, relu_10, convolution_12, squeeze_37, relu_11, convolution_13, squeeze_40, convolution_14, squeeze_43, relu_12, convolution_15, squeeze_46, relu_13, convolution_16, squeeze_49, relu_14, convolution_17, squeeze_52, relu_15, convolution_18, squeeze_55, relu_16, convolution_19, squeeze_58, relu_17, convolution_20, squeeze_61, relu_18, convolution_21, squeeze_64, relu_19, convolution_22, squeeze_67, relu_20, convolution_23, squeeze_70, relu_21, convolution_24, squeeze_73, relu_22, convolution_25, squeeze_76, relu_23, convolution_26, squeeze_79, convolution_27, squeeze_82, relu_24, convolution_28, squeeze_85, relu_25, convolution_29, squeeze_88, relu_26, convolution_30, squeeze_91, relu_27, convolution_31, squeeze_94, relu_28, convolution_32, squeeze_97, relu_29, convolution_33, squeeze_100, relu_30, convolution_34, squeeze_103, relu_31, convolution_35, squeeze_106, relu_32, convolution_36, squeeze_109, relu_33, convolution_37, squeeze_112, relu_34, convolution_38, squeeze_115, relu_35, convolution_39, squeeze_118, relu_36, convolution_40, squeeze_121, relu_37, convolution_41, squeeze_124, relu_38, convolution_42, squeeze_127, relu_39, convolution_43, squeeze_130, relu_40, convolution_44, squeeze_133, relu_41, convolution_45, squeeze_136, convolution_46, squeeze_139, relu_42, convolution_47, squeeze_142, relu_43, convolution_48, squeeze_145, relu_44, convolution_49, squeeze_148, relu_45, convolution_50, squeeze_151, relu_46, convolution_51, squeeze_154, relu_47, convolution_52, squeeze_157, view, permute_1, le, unsqueeze_214, unsqueeze_226, unsqueeze_238, unsqueeze_250, unsqueeze_262, unsqueeze_274, unsqueeze_286, unsqueeze_298, unsqueeze_310, unsqueeze_322, unsqueeze_334, unsqueeze_346, unsqueeze_358, unsqueeze_370, unsqueeze_382, unsqueeze_394, unsqueeze_406, unsqueeze_418, unsqueeze_430, unsqueeze_442, unsqueeze_454, unsqueeze_466, unsqueeze_478, unsqueeze_490, unsqueeze_502, unsqueeze_514, unsqueeze_526, unsqueeze_538, unsqueeze_550, unsqueeze_562, unsqueeze_574, unsqueeze_586, unsqueeze_598, unsqueeze_610, unsqueeze_622, unsqueeze_634, unsqueeze_646, unsqueeze_658, unsqueeze_670, unsqueeze_682, unsqueeze_694, unsqueeze_706, unsqueeze_718, unsqueeze_730, unsqueeze_742, unsqueeze_754, unsqueeze_766, unsqueeze_778, unsqueeze_790, unsqueeze_802, unsqueeze_814, unsqueeze_826, unsqueeze_838, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
