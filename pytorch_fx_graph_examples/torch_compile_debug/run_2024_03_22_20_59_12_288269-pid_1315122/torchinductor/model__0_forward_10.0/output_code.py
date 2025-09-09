
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


# kernel path: /tmp/torchinductor_zhang402/em/cemhiwmif6wptsamdc62i547vjtacq22cdwyhhgoxhdp4oeo53ty.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = (yindex // 3)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (147*y1)), tmp0, xmask & ymask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_zhang402/5g/c5gksl2deeptng5wqwvs26fwt2xhjzs6clb3hero7qmsilciixib.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (576*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/bj/cbjp5jutexnjdt3grkcjkjjtnf6xo6wxpxqt3xhhxqnkfioeazb3.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (128*x2) + (1152*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/h2/ch2b7yvf5vf5yszirpqepzhuerokjzb5jt4mo4je4eepc35yb3xo.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 65536
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (256*x2) + (2304*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/dv/cdvsl6s6ulhxirj5rgdvm5b2dtowniktolix5gkmhphaj7i4yzeu.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 262144
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (512*x2) + (4608*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/5v/c5v6ub3pwjnoyngeoox5ew4kpdejcuod3jat4cilzh7vj2ftulmz.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 65536], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 50176
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = (yindex // 3)
    tmp0 = tl.load(in_ptr0 + (x2 + (50176*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (150528*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_zhang402/eb/ceb6b552hd7lnf7ripexe3lnyxkbrsx4gwuu4i3wxvojefhclgb4.py
# Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
# x_1 => var_mean
triton_red_fused__native_batch_norm_legit_functional_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 1024],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 57344
    rnumel = 896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (57344*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr1 + (x3), tmp3, None)
    tl.store(out_ptr2 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_zhang402/5a/c5akumonbaqfidkhctxbp6f6n2mbnjmtputwfwqqnt6eb4c6ytqe.py
# Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
# x_1 => var_mean
triton_red_fused__native_batch_norm_legit_functional_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 448
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 7
    x1 = (xindex // 7)
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (64*r2) + (8192*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (64*r2) + (8192*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (64*r2) + (8192*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_combine(
            tmp6_mean, tmp6_m2, tmp6_weight,
            tmp3, tmp4, tmp5
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x1 + (64*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (64*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (64*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/ig/cigv4r7nur6ms5xbubae3shuoyn7fmjwtrxdzsf2hkc5hkl62bw3.py
# Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
# x_1 => add_1, add_2, add_3, mul_1, mul_2, mul_3, mul_4, mul_5, rsqrt, squeeze_1, var_mean
triton_per_fused__native_batch_norm_legit_functional_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 8],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_8', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (64*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 802816.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000012456169853
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/zu/czuxfvbtnkz7d7p7w5cf25xqjvj7wcua36qi3jo7y4g4ctr4b3qr.py
# Source Nodes: [x_1, x_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_1 => add_1, add_4, mul, mul_6, rsqrt, sub, var_mean
# x_2 => relu
triton_poi_fused__native_batch_norm_legit_functional_relu_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[67108864], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 802816.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_zhang402/vj/cvjrzbzqexliqjprogkf3wk63zg5ew644wzmcaiuycy5qr5jfe6n.py
# Source Nodes: [identity], Original ATen: [aten.max_pool2d_with_indices]
# identity => getitem_2, getitem_3
triton_poi_fused_max_pool2d_with_indices_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 3584) % 56
    x1 = (xindex // 64) % 56
    x0 = xindex % 64
    x5 = (xindex // 3584)
    x6 = xindex
    tmp0 = (-1) + (2*x2)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 112, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x1)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-7232) + x0 + (128*x1) + (14336*x5)), tmp10, other=0.0)
    tmp12 = tl.full(tmp11.shape, float("-inf"), tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 2*x1
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-7168) + x0 + (128*x1) + (14336*x5)), tmp18, other=0.0)
    tmp20 = tl.full(tmp19.shape, float("-inf"), tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = triton_helpers.maximum(tmp21, tmp13)
    tmp23 = 1 + (2*x1)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-7104) + x0 + (128*x1) + (14336*x5)), tmp27, other=0.0)
    tmp29 = tl.full(tmp28.shape, float("-inf"), tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = triton_helpers.maximum(tmp30, tmp22)
    tmp32 = 2*x2
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-64) + x0 + (128*x1) + (14336*x5)), tmp36, other=0.0)
    tmp38 = tl.full(tmp37.shape, float("-inf"), tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = triton_helpers.maximum(tmp39, tmp31)
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (x0 + (128*x1) + (14336*x5)), tmp41, other=0.0)
    tmp43 = tl.full(tmp42.shape, float("-inf"), tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = triton_helpers.maximum(tmp44, tmp40)
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (64 + x0 + (128*x1) + (14336*x5)), tmp46, other=0.0)
    tmp48 = tl.full(tmp47.shape, float("-inf"), tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = triton_helpers.maximum(tmp49, tmp45)
    tmp51 = 1 + (2*x2)
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (7104 + x0 + (128*x1) + (14336*x5)), tmp55, other=0.0)
    tmp57 = tl.full(tmp56.shape, float("-inf"), tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = triton_helpers.maximum(tmp58, tmp50)
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (7168 + x0 + (128*x1) + (14336*x5)), tmp60, other=0.0)
    tmp62 = tl.full(tmp61.shape, float("-inf"), tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = triton_helpers.maximum(tmp63, tmp59)
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (7232 + x0 + (128*x1) + (14336*x5)), tmp65, other=0.0)
    tmp67 = tl.full(tmp66.shape, float("-inf"), tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = triton_helpers.maximum(tmp68, tmp64)
    tmp70 = tmp21 > tmp13
    tmp71 = (-112) + (2*x1) + (224*x2)
    tmp72 = (-113) + (2*x1) + (224*x2)
    tmp73 = tl.where(tmp70, tmp71, tmp72)
    tmp74 = tmp30 > tmp22
    tmp75 = (-111) + (2*x1) + (224*x2)
    tmp76 = tl.where(tmp74, tmp75, tmp73)
    tmp77 = tmp39 > tmp31
    tmp78 = (-1) + (2*x1) + (224*x2)
    tmp79 = tl.where(tmp77, tmp78, tmp76)
    tmp80 = tmp44 > tmp40
    tmp81 = (2*x1) + (224*x2)
    tmp82 = tl.where(tmp80, tmp81, tmp79)
    tmp83 = tmp49 > tmp45
    tmp84 = 1 + (2*x1) + (224*x2)
    tmp85 = tl.where(tmp83, tmp84, tmp82)
    tmp86 = tmp58 > tmp50
    tmp87 = 111 + (2*x1) + (224*x2)
    tmp88 = tl.where(tmp86, tmp87, tmp85)
    tmp89 = tmp63 > tmp59
    tmp90 = 112 + (2*x1) + (224*x2)
    tmp91 = tl.where(tmp89, tmp90, tmp88)
    tmp92 = tmp68 > tmp64
    tmp93 = 113 + (2*x1) + (224*x2)
    tmp94 = tl.where(tmp92, tmp93, tmp91)
    tl.store(out_ptr0 + (x6), tmp69, None)
    tl.store(out_ptr1 + (x6), tmp94, None)
''')


# kernel path: /tmp/torchinductor_zhang402/7i/c7ir5g3e46angct6mbhd524rgk5t6wrxo2kmijq36aa3iefyake4.py
# Source Nodes: [out_1], Original ATen: [aten._native_batch_norm_legit_functional]
# out_1 => var_mean_1
triton_red_fused__native_batch_norm_legit_functional_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 256],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 57344
    rnumel = 224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (14336*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr1 + (x3), tmp3, None)
    tl.store(out_ptr2 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_zhang402/7a/c7a6rnmqwozjxhg2clziabhgghqoarmnndvuldpzyxq5gau7x4u5.py
# Source Nodes: [out_1], Original ATen: [aten._native_batch_norm_legit_functional]
# out_1 => add_6, add_7, add_8, mul_10, mul_11, mul_12, mul_8, mul_9, rsqrt_1, squeeze_4, var_mean_1
triton_per_fused__native_batch_norm_legit_functional_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 8],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_12', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (64*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 200704.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000049824865598
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/ea/cea23y2f7bwb3vpvpvohbspnatamptqdfyxx2xpw3rp5prfpgrwq.py
# Source Nodes: [out_1, out_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# out_1 => add_6, add_9, mul_13, mul_7, rsqrt_1, sub_1, var_mean_1
# out_2 => relu_1
triton_poi_fused__native_batch_norm_legit_functional_relu_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 200704.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_zhang402/sw/cswkmavnkfqjeqqap27eibo6ht2uhj3pvawvcdokj35j62kxadq2.py
# Source Nodes: [out_7], Original ATen: [aten._native_batch_norm_legit_functional]
# out_7 => var_mean_3
triton_red_fused__native_batch_norm_legit_functional_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[131072, 512],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 114688
    rnumel = 448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (114688*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr1 + (x3), tmp3, None)
    tl.store(out_ptr2 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_zhang402/5p/c5pmolvplxqe2d7fvpcslf6r7yqr2qpczk2blf5nxorj7z4zc323.py
# Source Nodes: [out_7], Original ATen: [aten._native_batch_norm_legit_functional]
# out_7 => var_mean_3
triton_red_fused__native_batch_norm_legit_functional_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 4
    x1 = (xindex // 4)
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (256*r2) + (28672*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (256*r2) + (28672*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (256*r2) + (28672*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_combine(
            tmp6_mean, tmp6_m2, tmp6_weight,
            tmp3, tmp4, tmp5
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x1 + (256*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (256*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (256*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/m3/cm3k2adydazhzkqlghq6kpziio5o5c77gvu5t3aq775mtdr4aqsp.py
# Source Nodes: [out_7], Original ATen: [aten._native_batch_norm_legit_functional]
# out_7 => add_16, add_17, add_18, mul_22, mul_23, mul_24, mul_25, mul_26, rsqrt_3, squeeze_10, var_mean_3
triton_per_fused__native_batch_norm_legit_functional_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 4],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_16', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 200704.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000049824865598
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/jl/cjlahxkeejaci4qbvkcega7dhddjgadchnq3l3ux4azrhew6hdvo.py
# Source Nodes: [identity_1, identity_2, out_7, out_8], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# identity_1 => add_21, add_24, mul_28, mul_34, rsqrt_4, sub_4, var_mean_4
# identity_2 => relu_3
# out_7 => add_16, add_19, mul_21, mul_27, rsqrt_3, sub_3, var_mean_3
# out_8 => add_25
triton_poi_fused__native_batch_norm_legit_functional_add_relu_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[67108864], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_17', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), None)
    tmp15 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 200704.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp16 = tmp14 - tmp15
    tmp18 = tmp17 / tmp4
    tmp19 = tmp18 + tmp6
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp16 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tmp13 + tmp25
    tmp27 = triton_helpers.maximum(0, tmp26)
    tl.store(in_out_ptr0 + (x2), tmp27, None)
''')


# kernel path: /tmp/torchinductor_zhang402/zr/czruk6ctz6zhgmakvj6bax32iv5cowlis6kjuislnkfsfywwgixp.py
# Source Nodes: [identity_3, out_17, out_18], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# identity_3 => relu_6
# out_17 => add_37, add_40, mul_49, mul_55, rsqrt_7, sub_7, var_mean_7
# out_18 => add_41
triton_poi_fused__native_batch_norm_legit_functional_add_relu_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[67108864], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 200704.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = triton_helpers.maximum(0, tmp15)
    tl.store(out_ptr0 + (x2), tmp16, None)
''')


# kernel path: /tmp/torchinductor_zhang402/fv/cfvlj7c5laylajfh5mapjw3vbgfkm5kjln3bme43o7nrcximrdsm.py
# Source Nodes: [out_31], Original ATen: [aten._native_batch_norm_legit_functional]
# out_31 => var_mean_11
triton_red_fused__native_batch_norm_legit_functional_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[131072, 256],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_19', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 114688
    rnumel = 224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (28672*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr1 + (x3), tmp3, None)
    tl.store(out_ptr2 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_zhang402/qt/cqtkbr7ob6jhh3xttghyyzpkxmsqcdfgevbntihz4ahtx4kogp2e.py
# Source Nodes: [out_31], Original ATen: [aten._native_batch_norm_legit_functional]
# out_31 => var_mean_11
triton_red_fused__native_batch_norm_legit_functional_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 896
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 7
    x1 = (xindex // 7)
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (128*r2) + (16384*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (128*r2) + (16384*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (128*r2) + (16384*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_combine(
            tmp6_mean, tmp6_m2, tmp6_weight,
            tmp3, tmp4, tmp5
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x1 + (128*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (128*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (128*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/23/c23j4asc5aplc3ayjicvzywovamw3pjblcorqzmgvd74o7evlxzw.py
# Source Nodes: [out_31], Original ATen: [aten._native_batch_norm_legit_functional]
# out_31 => add_59, add_60, add_61, mul_78, mul_79, mul_80, mul_81, mul_82, rsqrt_11, squeeze_34, var_mean_11
triton_per_fused__native_batch_norm_legit_functional_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 8],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_21', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 200704.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000049824865598
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/su/csu4r72mnvsqcz2qjdd2fufbrbei4i3xxf3dz2haghwhf6nvc44r.py
# Source Nodes: [out_31, out_32], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# out_31 => add_59, add_62, mul_77, mul_83, rsqrt_11, sub_11, var_mean_11
# out_32 => relu_10
triton_poi_fused__native_batch_norm_legit_functional_relu_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25690112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 200704.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_zhang402/37/c37crlxxqwaw4jwvy2gb3ecvxcwgx6ycnxqcirjvplu2oavouvi4.py
# Source Nodes: [out_34], Original ATen: [aten._native_batch_norm_legit_functional]
# out_34 => var_mean_12
triton_red_fused__native_batch_norm_legit_functional_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_23', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 50176
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr1 + (x3), tmp3, xmask)
    tl.store(out_ptr2 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/yw/cywktstllo3conwdonvbac2ka6lnqods7bhyx7juk6fcwkemzway.py
# Source Nodes: [out_34], Original ATen: [aten._native_batch_norm_legit_functional]
# out_34 => var_mean_12
triton_red_fused__native_batch_norm_legit_functional_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_24', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 4
    x1 = (xindex // 4)
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (128*r2) + (12544*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (128*r2) + (12544*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (128*r2) + (12544*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_combine(
            tmp6_mean, tmp6_m2, tmp6_weight,
            tmp3, tmp4, tmp5
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x1 + (128*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (128*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (128*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/zg/czg5lz2pmzxwol7ul6qqxnr7d4wyd7floarn3aoshlpg6gaan3ts.py
# Source Nodes: [out_34], Original ATen: [aten._native_batch_norm_legit_functional]
# out_34 => add_64, add_65, add_66, mul_85, mul_86, mul_87, mul_88, mul_89, rsqrt_12, squeeze_37, var_mean_12
triton_per_fused__native_batch_norm_legit_functional_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 4],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_25', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 50176.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000199302441455
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/47/c47em4hazq3kcpcsvaq4hwmbdt27z7flfnxbyhwvanlhdgibcziq.py
# Source Nodes: [out_34, out_35], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# out_34 => add_64, add_67, mul_84, mul_90, rsqrt_12, sub_12, var_mean_12
# out_35 => relu_11
triton_poi_fused__native_batch_norm_legit_functional_relu_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_26', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 50176.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_zhang402/x5/cx5tf4a2qfw4tjsuqhkmt3meoq4tzg5j7birosl3uyuycu3pwswe.py
# Source Nodes: [out_37], Original ATen: [aten._native_batch_norm_legit_functional]
# out_37 => var_mean_13
triton_red_fused__native_batch_norm_legit_functional_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[131072, 256],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_27', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 114688
    rnumel = 224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (114688*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr1 + (x3), tmp3, None)
    tl.store(out_ptr2 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_zhang402/sp/cspqavvpmgblnmrhc3owscavc34tqg6tynqvsnxh2s6yieylof2s.py
# Source Nodes: [out_37], Original ATen: [aten._native_batch_norm_legit_functional]
# out_37 => var_mean_13
triton_red_fused__native_batch_norm_legit_functional_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_28', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2
    x1 = (xindex // 2)
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (512*r2) + (57344*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (512*r2) + (57344*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (512*r2) + (57344*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_combine(
            tmp6_mean, tmp6_m2, tmp6_weight,
            tmp3, tmp4, tmp5
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x1 + (512*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (512*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (512*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/3n/c3nth6vwghrnbcfrk7tisngvfzwf43bp2uu2gvzg7dg2pljqxyy5.py
# Source Nodes: [out_37], Original ATen: [aten._native_batch_norm_legit_functional]
# out_37 => add_69, add_70, add_71, mul_92, mul_93, mul_94, mul_95, mul_96, rsqrt_13, squeeze_40, var_mean_13
triton_per_fused__native_batch_norm_legit_functional_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 2],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_29', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 50176.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000199302441455
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/yd/cyd56fm4idgerfhp645mc77j63yuzjgpanzj6wwnezqxvv7adi6y.py
# Source Nodes: [identity_5, identity_6, out_37, out_38], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# identity_5 => add_74, add_77, mul_104, mul_98, rsqrt_14, sub_14, var_mean_14
# identity_6 => relu_12
# out_37 => add_69, add_72, mul_91, mul_97, rsqrt_13, sub_13, var_mean_13
# out_38 => add_78
triton_poi_fused__native_batch_norm_legit_functional_add_relu_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_30', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25690112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), None)
    tmp15 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 50176.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp16 = tmp14 - tmp15
    tmp18 = tmp17 / tmp4
    tmp19 = tmp18 + tmp6
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp16 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tmp13 + tmp25
    tmp27 = triton_helpers.maximum(0, tmp26)
    tl.store(in_out_ptr0 + (x2), tmp27, None)
''')


# kernel path: /tmp/torchinductor_zhang402/aa/caah2mwnt4w72gfo7xg2ptvntd55r6pgb7c7wwfxpeczz3d6krab.py
# Source Nodes: [identity_7, out_47, out_48], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# identity_7 => relu_15
# out_47 => add_90, add_93, mul_119, mul_125, rsqrt_17, sub_17, var_mean_17
# out_48 => add_94
triton_poi_fused__native_batch_norm_legit_functional_add_relu_31 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25690112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 50176.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = triton_helpers.maximum(0, tmp15)
    tl.store(out_ptr0 + (x2), tmp16, None)
''')


# kernel path: /tmp/torchinductor_zhang402/3f/c3fzltzhvnzenehlk7i5iajvzj5kgel35clo4dvsjhkbk2jmw5vl.py
# Source Nodes: [out_71], Original ATen: [aten._native_batch_norm_legit_functional]
# out_71 => var_mean_24
triton_red_fused__native_batch_norm_legit_functional_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[131072, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_32', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 100352
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (32768*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr1 + (x3), tmp3, None)
    tl.store(out_ptr2 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_zhang402/t7/ct7upzkeez4xnvirc3debc3i3t7qgucihctluvv3i3btwcwilxp4.py
# Source Nodes: [out_71], Original ATen: [aten._native_batch_norm_legit_functional]
# out_71 => var_mean_24
triton_red_fused__native_batch_norm_legit_functional_33 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_33', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 4
    x1 = (xindex // 4)
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (256*r2) + (25088*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (256*r2) + (25088*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (256*r2) + (25088*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_combine(
            tmp6_mean, tmp6_m2, tmp6_weight,
            tmp3, tmp4, tmp5
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x1 + (256*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (256*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (256*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/ev/cev277c6mugknpjk3usc4k3ylzs2j4qjsxdhb2llvgapzvrnwdfd.py
# Source Nodes: [out_71], Original ATen: [aten._native_batch_norm_legit_functional]
# out_71 => add_128, add_129, add_130, mul_169, mul_170, mul_171, mul_172, mul_173, rsqrt_24, squeeze_73, var_mean_24
triton_per_fused__native_batch_norm_legit_functional_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 4],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_34', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 50176.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000199302441455
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/ew/cewr6owq4movj5e7eobmn72va4bnwsl7gtxmgqjycibtngidthn5.py
# Source Nodes: [out_71, out_72], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# out_71 => add_128, add_131, mul_168, mul_174, rsqrt_24, sub_24, var_mean_24
# out_72 => relu_22
triton_poi_fused__native_batch_norm_legit_functional_relu_35 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 50176.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_zhang402/lc/clcix3qowtfcguwpkwr55w23sczupba6qmqu7dzsi75pojsgcfmf.py
# Source Nodes: [out_74], Original ATen: [aten._native_batch_norm_legit_functional]
# out_74 => var_mean_25
triton_red_fused__native_batch_norm_legit_functional_36 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_36', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr1 + (x3), tmp3, xmask)
    tl.store(out_ptr2 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/gx/cgx332dgzlnceiq3mxozxweie5xqb7jzoda6vdupsxbhshiwcpsf.py
# Source Nodes: [out_74], Original ATen: [aten._native_batch_norm_legit_functional]
# out_74 => add_133, add_134, add_135, mul_176, mul_177, mul_178, mul_179, mul_180, rsqrt_25, squeeze_76, var_mean_25
triton_red_fused__native_batch_norm_legit_functional_37 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_37', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (256*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x0 + (256*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_combine(
            tmp6_mean, tmp6_m2, tmp6_weight,
            tmp3, tmp4, tmp5
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp7, xmask)
    tmp16 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp9 = 12544.0
    tmp10 = tmp7 / tmp9
    tmp11 = 1e-05
    tmp12 = tmp10 + tmp11
    tmp13 = tl.math.rsqrt(tmp12)
    tmp14 = 0.1
    tmp15 = tmp6 * tmp14
    tmp17 = 0.9
    tmp18 = tmp16 * tmp17
    tmp19 = tmp15 + tmp18
    tmp20 = 1.0000797257434426
    tmp21 = tmp10 * tmp20
    tmp22 = tmp21 * tmp14
    tmp24 = tmp23 * tmp17
    tmp25 = tmp22 + tmp24
    tl.store(out_ptr2 + (x0), tmp13, xmask)
    tl.store(out_ptr4 + (x0), tmp19, xmask)
    tl.store(out_ptr6 + (x0), tmp25, xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/iy/ciy4ymxpqa3db7idgi4622zct4675m6kokyyrsqhczj4k76ypn5z.py
# Source Nodes: [out_74, out_75], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# out_74 => add_133, add_136, mul_175, mul_181, rsqrt_25, sub_25, var_mean_25
# out_75 => relu_23
triton_poi_fused__native_batch_norm_legit_functional_relu_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_38', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 12544.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_zhang402/tb/ctbobdk4fpil4tt64b6tpc47uorby2nkij4qucvpfsok75yjz3un.py
# Source Nodes: [out_77], Original ATen: [aten._native_batch_norm_legit_functional]
# out_77 => var_mean_26
triton_red_fused__native_batch_norm_legit_functional_39 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[131072, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_39', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 100352
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (131072*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr1 + (x3), tmp3, None)
    tl.store(out_ptr2 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_zhang402/kh/ckhz4ubl244prvwzejgiuyixm74t7fvqpctjkojwoodvrwuk72vb.py
# Source Nodes: [out_77], Original ATen: [aten._native_batch_norm_legit_functional]
# out_77 => add_138, add_139, add_140, mul_183, mul_184, mul_185, mul_186, mul_187, rsqrt_26, squeeze_79, var_mean_26
triton_red_fused__native_batch_norm_legit_functional_40 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_40', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (1024*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x0 + (1024*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_combine(
            tmp6_mean, tmp6_m2, tmp6_weight,
            tmp3, tmp4, tmp5
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp7, xmask)
    tmp16 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp9 = 12544.0
    tmp10 = tmp7 / tmp9
    tmp11 = 1e-05
    tmp12 = tmp10 + tmp11
    tmp13 = tl.math.rsqrt(tmp12)
    tmp14 = 0.1
    tmp15 = tmp6 * tmp14
    tmp17 = 0.9
    tmp18 = tmp16 * tmp17
    tmp19 = tmp15 + tmp18
    tmp20 = 1.0000797257434426
    tmp21 = tmp10 * tmp20
    tmp22 = tmp21 * tmp14
    tmp24 = tmp23 * tmp17
    tmp25 = tmp22 + tmp24
    tl.store(out_ptr2 + (x0), tmp13, xmask)
    tl.store(out_ptr4 + (x0), tmp19, xmask)
    tl.store(out_ptr6 + (x0), tmp25, xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/45/c45ocmhoqx5uhf4t7ld4naap5figbc3vbhwu4q7htlpnbgp36dv4.py
# Source Nodes: [identity_10, identity_11, out_77, out_78], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# identity_10 => add_143, add_146, mul_189, mul_195, rsqrt_27, sub_27, var_mean_27
# identity_11 => relu_24
# out_77 => add_138, add_141, mul_182, mul_188, rsqrt_26, sub_26, var_mean_26
# out_78 => add_147
triton_poi_fused__native_batch_norm_legit_functional_add_relu_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_41', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), None)
    tmp15 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 12544.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp16 = tmp14 - tmp15
    tmp18 = tmp17 / tmp4
    tmp19 = tmp18 + tmp6
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp16 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tmp13 + tmp25
    tmp27 = triton_helpers.maximum(0, tmp26)
    tl.store(in_out_ptr0 + (x2), tmp27, None)
''')


# kernel path: /tmp/torchinductor_zhang402/bb/cbb5oxpw23p4viewiyzd5lgwfwrxvhk6hv6hw3e4zxu425fxompo.py
# Source Nodes: [identity_12, out_87, out_88], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# identity_12 => relu_27
# out_87 => add_159, add_162, mul_210, mul_216, rsqrt_30, sub_30, var_mean_30
# out_88 => add_163
triton_poi_fused__native_batch_norm_legit_functional_add_relu_42 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_42', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 12544.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = triton_helpers.maximum(0, tmp15)
    tl.store(out_ptr0 + (x2), tmp16, None)
''')


# kernel path: /tmp/torchinductor_zhang402/qm/cqmklrk4spa5wld33xwv3aaggic4qqmtbdxrl6qminkbnxrxcd6o.py
# Source Nodes: [out_131], Original ATen: [aten._native_batch_norm_legit_functional]
# out_131 => var_mean_43
triton_red_fused__native_batch_norm_legit_functional_43 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_43', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 50176
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (65536*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr1 + (x3), tmp3, xmask)
    tl.store(out_ptr2 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/jn/cjn5eni2ddu7xd7l7thytzauhqban46ichnpgdtbmrvm4fq2vben.py
# Source Nodes: [out_131], Original ATen: [aten._native_batch_norm_legit_functional]
# out_131 => add_229, add_230, add_231, mul_302, mul_303, mul_304, mul_305, mul_306, rsqrt_43, squeeze_130, var_mean_43
triton_red_fused__native_batch_norm_legit_functional_44 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_44', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (512*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x0 + (512*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_combine(
            tmp6_mean, tmp6_m2, tmp6_weight,
            tmp3, tmp4, tmp5
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp7, xmask)
    tmp16 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp9 = 12544.0
    tmp10 = tmp7 / tmp9
    tmp11 = 1e-05
    tmp12 = tmp10 + tmp11
    tmp13 = tl.math.rsqrt(tmp12)
    tmp14 = 0.1
    tmp15 = tmp6 * tmp14
    tmp17 = 0.9
    tmp18 = tmp16 * tmp17
    tmp19 = tmp15 + tmp18
    tmp20 = 1.0000797257434426
    tmp21 = tmp10 * tmp20
    tmp22 = tmp21 * tmp14
    tmp24 = tmp23 * tmp17
    tmp25 = tmp22 + tmp24
    tl.store(out_ptr2 + (x0), tmp13, xmask)
    tl.store(out_ptr4 + (x0), tmp19, xmask)
    tl.store(out_ptr6 + (x0), tmp25, xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/6g/c6gmnl2t7yviabzmdezvuuex37ndejmy6lrodjvnzqygn5a4w7yl.py
# Source Nodes: [out_131, out_132], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# out_131 => add_229, add_232, mul_301, mul_307, rsqrt_43, sub_43, var_mean_43
# out_132 => relu_40
triton_poi_fused__native_batch_norm_legit_functional_relu_45 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_45', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 12544.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_zhang402/7v/c7v5z4mb6nigcpsig6b7pr3wbmkw7b6fzzmhxkry6g52bod5s5sy.py
# Source Nodes: [out_134], Original ATen: [aten._native_batch_norm_legit_functional]
# out_134 => var_mean_44
triton_red_fused__native_batch_norm_legit_functional_46 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_46', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12800
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 512)
    x0 = xindex % 512
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (126*x1)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (512*(r2 % 7)) + (3584*(((r2 + (126*x1)) // 7) % 448))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = 0.0
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = 1.0
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp13 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp14 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp15_mean_next, tmp15_m2_next, tmp15_weight_next = triton_helpers.welford_combine(
            tmp15_mean, tmp15_m2, tmp15_weight,
            tmp12, tmp13, tmp14
        )
        tmp15_mean = tl.where(rmask & xmask, tmp15_mean_next, tmp15_mean)
        tmp15_m2 = tl.where(rmask & xmask, tmp15_m2_next, tmp15_m2)
        tmp15_weight = tl.where(rmask & xmask, tmp15_weight_next, tmp15_weight)
    tmp15_tmp, tmp16_tmp, tmp17_tmp = triton_helpers.welford(
        tmp15_mean, tmp15_m2, tmp15_weight, 1
    )
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp15, xmask)
    tl.store(out_ptr1 + (x3), tmp16, xmask)
    tl.store(out_ptr2 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/4k/c4kabbiwoo665ya2ufg6h2cyrg526itmogxgt222ij2visgguey4.py
# Source Nodes: [out_134], Original ATen: [aten._native_batch_norm_legit_functional]
# out_134 => add_234, add_235, add_236, mul_309, mul_310, mul_311, mul_312, mul_313, rsqrt_44, squeeze_133, var_mean_44
triton_per_fused__native_batch_norm_legit_functional_47 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 32],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_47', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 25
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 3136.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0003189792663476
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/by/cbybmwzjm7s2ty63ye2jhi3a7bvs55qty3d6lbml4alufn3ed6sv.py
# Source Nodes: [out_134, out_135], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# out_134 => add_234, add_237, mul_308, mul_314, rsqrt_44, sub_44, var_mean_44
# out_135 => relu_41
triton_poi_fused__native_batch_norm_legit_functional_relu_48 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_48', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 3136.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_zhang402/nb/cnb5hahqt5gr4hthclo4elsbnm6fooogtdfibpfc2dclvzi533fn.py
# Source Nodes: [out_137], Original ATen: [aten._native_batch_norm_legit_functional]
# out_137 => var_mean_45
triton_red_fused__native_batch_norm_legit_functional_49 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_49', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 51200
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 2048)
    x0 = xindex % 2048
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
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
        tmp6 = 0.0
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = 1.0
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp13 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp14 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp15_mean_next, tmp15_m2_next, tmp15_weight_next = triton_helpers.welford_combine(
            tmp15_mean, tmp15_m2, tmp15_weight,
            tmp12, tmp13, tmp14
        )
        tmp15_mean = tl.where(rmask, tmp15_mean_next, tmp15_mean)
        tmp15_m2 = tl.where(rmask, tmp15_m2_next, tmp15_m2)
        tmp15_weight = tl.where(rmask, tmp15_weight_next, tmp15_weight)
    tmp15_tmp, tmp16_tmp, tmp17_tmp = triton_helpers.welford(
        tmp15_mean, tmp15_m2, tmp15_weight, 1
    )
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp15, None)
    tl.store(out_ptr1 + (x3), tmp16, None)
    tl.store(out_ptr2 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_zhang402/kg/ckgpqz3om37xfii3y3uzty5b25wwrq5wgef6p4gftzlypfmvyndp.py
# Source Nodes: [out_137], Original ATen: [aten._native_batch_norm_legit_functional]
# out_137 => add_239, add_240, add_241, mul_316, mul_317, mul_318, mul_319, mul_320, rsqrt_45, squeeze_136, var_mean_45
triton_per_fused__native_batch_norm_legit_functional_50 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 32],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_50', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 25
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (2048*r1)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (2048*r1)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (2048*r1)), rmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask, tmp3, 0)
    tmp8 = tl.where(rmask, tmp4, 0)
    tmp9 = tl.where(rmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 3136.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0003189792663476
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, None)
    tl.store(out_ptr4 + (x0), tmp26, None)
    tl.store(out_ptr6 + (x0), tmp32, None)
    tl.store(out_ptr0 + (x0), tmp13, None)
    tl.store(out_ptr1 + (x0), tmp14, None)
''')


# kernel path: /tmp/torchinductor_zhang402/4c/c4c44viprn3fvu2mnzps6rcktffxbxn2vvaxlrrvyregtjgkgkak.py
# Source Nodes: [identity_17, identity_18, out_137, out_138], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# identity_17 => add_244, add_247, mul_322, mul_328, rsqrt_46, sub_46, var_mean_46
# identity_18 => relu_42
# out_137 => add_239, add_242, mul_315, mul_321, rsqrt_45, sub_45, var_mean_45
# out_138 => add_248
triton_poi_fused__native_batch_norm_legit_functional_add_relu_51 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_51', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 2048
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), None)
    tmp15 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 3136.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp16 = tmp14 - tmp15
    tmp18 = tmp17 / tmp4
    tmp19 = tmp18 + tmp6
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp16 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tmp13 + tmp25
    tmp27 = triton_helpers.maximum(0, tmp26)
    tl.store(in_out_ptr0 + (x2), tmp27, None)
''')


# kernel path: /tmp/torchinductor_zhang402/hm/chmxud5fycdijw4ditg7ehsvxusniioacwcv4iydlflkilbtlgfu.py
# Source Nodes: [identity_19, out_147, out_148], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# identity_19 => relu_45
# out_147 => add_260, add_263, mul_343, mul_349, rsqrt_49, sub_49, var_mean_49
# out_148 => add_264
triton_poi_fused__native_batch_norm_legit_functional_add_relu_52 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_52', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 2048
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 3136.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = triton_helpers.maximum(0, tmp15)
    tl.store(out_ptr0 + (x2), tmp16, None)
''')


# kernel path: /tmp/torchinductor_zhang402/ly/clydlhoxqomx234ygbtahaag7u7cunkydoaf4zyjofncgtiwcoez.py
# Source Nodes: [out_157, out_158, x_7], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu, aten.threshold_backward]
# out_157 => add_276, add_279, mul_364, mul_370, rsqrt_52, sub_52, var_mean_52
# out_158 => add_280
# x_7 => relu_48
triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_53 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_53', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 2048
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 3136.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = triton_helpers.maximum(0, tmp15)
    tmp17 = 0.0
    tmp18 = tmp16 <= tmp17
    tl.store(out_ptr0 + (x2), tmp16, None)
    tl.store(out_ptr1 + (x2), tmp18, None)
''')


# kernel path: /tmp/torchinductor_zhang402/fx/cfxdf5nrgkwcdmaz522uecoz746qkgpcvwfyowp3yaavfwue4pzo.py
# Source Nodes: [x_8, x_9], Original ATen: [aten.mean, aten.view]
# x_8 => mean
# x_9 => view
triton_per_fused_mean_view_54 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[131072, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_view_54', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 2048
    x1 = (xindex // 2048)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (2048*r2) + (100352*x1)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 49.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_zhang402/eu/ceuc4dn7ejhsegchqb5efz5jpfwqmwa5q3izzndu2k3hncitdldn.py
# Source Nodes: [x_1], Original ATen: [aten.add]
# x_1 => add
triton_poi_fused_add_55 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_55', 'mutated_arg_names': ['in_ptr0', 'out_ptr1']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.full([1], 1, tl.int64)
    tmp3 = tmp1 + tmp2
    tl.store(out_ptr1 + (tl.full([XBLOCK], 0, tl.int32)), tmp3, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_4, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (64, ), (1, ))
    assert_size_stride(primals_7, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_8, (64, ), (1, ))
    assert_size_stride(primals_9, (64, ), (1, ))
    assert_size_stride(primals_10, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_11, (256, ), (1, ))
    assert_size_stride(primals_12, (256, ), (1, ))
    assert_size_stride(primals_13, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_14, (256, ), (1, ))
    assert_size_stride(primals_15, (256, ), (1, ))
    assert_size_stride(primals_16, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_17, (64, ), (1, ))
    assert_size_stride(primals_18, (64, ), (1, ))
    assert_size_stride(primals_19, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_20, (64, ), (1, ))
    assert_size_stride(primals_21, (64, ), (1, ))
    assert_size_stride(primals_22, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_23, (256, ), (1, ))
    assert_size_stride(primals_24, (256, ), (1, ))
    assert_size_stride(primals_25, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_26, (64, ), (1, ))
    assert_size_stride(primals_27, (64, ), (1, ))
    assert_size_stride(primals_28, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_29, (64, ), (1, ))
    assert_size_stride(primals_30, (64, ), (1, ))
    assert_size_stride(primals_31, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_32, (256, ), (1, ))
    assert_size_stride(primals_33, (256, ), (1, ))
    assert_size_stride(primals_34, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_35, (128, ), (1, ))
    assert_size_stride(primals_36, (128, ), (1, ))
    assert_size_stride(primals_37, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_38, (128, ), (1, ))
    assert_size_stride(primals_39, (128, ), (1, ))
    assert_size_stride(primals_40, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_41, (512, ), (1, ))
    assert_size_stride(primals_42, (512, ), (1, ))
    assert_size_stride(primals_43, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_44, (512, ), (1, ))
    assert_size_stride(primals_45, (512, ), (1, ))
    assert_size_stride(primals_46, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_47, (128, ), (1, ))
    assert_size_stride(primals_48, (128, ), (1, ))
    assert_size_stride(primals_49, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_50, (128, ), (1, ))
    assert_size_stride(primals_51, (128, ), (1, ))
    assert_size_stride(primals_52, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_53, (512, ), (1, ))
    assert_size_stride(primals_54, (512, ), (1, ))
    assert_size_stride(primals_55, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_56, (128, ), (1, ))
    assert_size_stride(primals_57, (128, ), (1, ))
    assert_size_stride(primals_58, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_59, (128, ), (1, ))
    assert_size_stride(primals_60, (128, ), (1, ))
    assert_size_stride(primals_61, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_62, (512, ), (1, ))
    assert_size_stride(primals_63, (512, ), (1, ))
    assert_size_stride(primals_64, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_65, (128, ), (1, ))
    assert_size_stride(primals_66, (128, ), (1, ))
    assert_size_stride(primals_67, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_68, (128, ), (1, ))
    assert_size_stride(primals_69, (128, ), (1, ))
    assert_size_stride(primals_70, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_71, (512, ), (1, ))
    assert_size_stride(primals_72, (512, ), (1, ))
    assert_size_stride(primals_73, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_74, (256, ), (1, ))
    assert_size_stride(primals_75, (256, ), (1, ))
    assert_size_stride(primals_76, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_77, (256, ), (1, ))
    assert_size_stride(primals_78, (256, ), (1, ))
    assert_size_stride(primals_79, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_80, (1024, ), (1, ))
    assert_size_stride(primals_81, (1024, ), (1, ))
    assert_size_stride(primals_82, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_83, (1024, ), (1, ))
    assert_size_stride(primals_84, (1024, ), (1, ))
    assert_size_stride(primals_85, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_86, (256, ), (1, ))
    assert_size_stride(primals_87, (256, ), (1, ))
    assert_size_stride(primals_88, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_89, (256, ), (1, ))
    assert_size_stride(primals_90, (256, ), (1, ))
    assert_size_stride(primals_91, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_92, (1024, ), (1, ))
    assert_size_stride(primals_93, (1024, ), (1, ))
    assert_size_stride(primals_94, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_95, (256, ), (1, ))
    assert_size_stride(primals_96, (256, ), (1, ))
    assert_size_stride(primals_97, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_98, (256, ), (1, ))
    assert_size_stride(primals_99, (256, ), (1, ))
    assert_size_stride(primals_100, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_101, (1024, ), (1, ))
    assert_size_stride(primals_102, (1024, ), (1, ))
    assert_size_stride(primals_103, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_104, (256, ), (1, ))
    assert_size_stride(primals_105, (256, ), (1, ))
    assert_size_stride(primals_106, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_107, (256, ), (1, ))
    assert_size_stride(primals_108, (256, ), (1, ))
    assert_size_stride(primals_109, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_110, (1024, ), (1, ))
    assert_size_stride(primals_111, (1024, ), (1, ))
    assert_size_stride(primals_112, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_113, (256, ), (1, ))
    assert_size_stride(primals_114, (256, ), (1, ))
    assert_size_stride(primals_115, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_116, (256, ), (1, ))
    assert_size_stride(primals_117, (256, ), (1, ))
    assert_size_stride(primals_118, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_119, (1024, ), (1, ))
    assert_size_stride(primals_120, (1024, ), (1, ))
    assert_size_stride(primals_121, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_122, (256, ), (1, ))
    assert_size_stride(primals_123, (256, ), (1, ))
    assert_size_stride(primals_124, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_125, (256, ), (1, ))
    assert_size_stride(primals_126, (256, ), (1, ))
    assert_size_stride(primals_127, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_128, (1024, ), (1, ))
    assert_size_stride(primals_129, (1024, ), (1, ))
    assert_size_stride(primals_130, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_131, (512, ), (1, ))
    assert_size_stride(primals_132, (512, ), (1, ))
    assert_size_stride(primals_133, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_134, (512, ), (1, ))
    assert_size_stride(primals_135, (512, ), (1, ))
    assert_size_stride(primals_136, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_137, (2048, ), (1, ))
    assert_size_stride(primals_138, (2048, ), (1, ))
    assert_size_stride(primals_139, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_140, (2048, ), (1, ))
    assert_size_stride(primals_141, (2048, ), (1, ))
    assert_size_stride(primals_142, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_143, (512, ), (1, ))
    assert_size_stride(primals_144, (512, ), (1, ))
    assert_size_stride(primals_145, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_146, (512, ), (1, ))
    assert_size_stride(primals_147, (512, ), (1, ))
    assert_size_stride(primals_148, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_149, (2048, ), (1, ))
    assert_size_stride(primals_150, (2048, ), (1, ))
    assert_size_stride(primals_151, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_152, (512, ), (1, ))
    assert_size_stride(primals_153, (512, ), (1, ))
    assert_size_stride(primals_154, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_155, (512, ), (1, ))
    assert_size_stride(primals_156, (512, ), (1, ))
    assert_size_stride(primals_157, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_158, (2048, ), (1, ))
    assert_size_stride(primals_159, (2048, ), (1, ))
    assert_size_stride(primals_160, (1000, 2048), (2048, 1))
    assert_size_stride(primals_161, (1000, ), (1, ))
    assert_size_stride(primals_162, (64, ), (1, ))
    assert_size_stride(primals_163, (64, ), (1, ))
    assert_size_stride(primals_164, (), ())
    assert_size_stride(primals_165, (64, ), (1, ))
    assert_size_stride(primals_166, (64, ), (1, ))
    assert_size_stride(primals_167, (), ())
    assert_size_stride(primals_168, (64, ), (1, ))
    assert_size_stride(primals_169, (64, ), (1, ))
    assert_size_stride(primals_170, (), ())
    assert_size_stride(primals_171, (256, ), (1, ))
    assert_size_stride(primals_172, (256, ), (1, ))
    assert_size_stride(primals_173, (), ())
    assert_size_stride(primals_174, (256, ), (1, ))
    assert_size_stride(primals_175, (256, ), (1, ))
    assert_size_stride(primals_176, (), ())
    assert_size_stride(primals_177, (64, ), (1, ))
    assert_size_stride(primals_178, (64, ), (1, ))
    assert_size_stride(primals_179, (), ())
    assert_size_stride(primals_180, (64, ), (1, ))
    assert_size_stride(primals_181, (64, ), (1, ))
    assert_size_stride(primals_182, (), ())
    assert_size_stride(primals_183, (256, ), (1, ))
    assert_size_stride(primals_184, (256, ), (1, ))
    assert_size_stride(primals_185, (), ())
    assert_size_stride(primals_186, (64, ), (1, ))
    assert_size_stride(primals_187, (64, ), (1, ))
    assert_size_stride(primals_188, (), ())
    assert_size_stride(primals_189, (64, ), (1, ))
    assert_size_stride(primals_190, (64, ), (1, ))
    assert_size_stride(primals_191, (), ())
    assert_size_stride(primals_192, (256, ), (1, ))
    assert_size_stride(primals_193, (256, ), (1, ))
    assert_size_stride(primals_194, (), ())
    assert_size_stride(primals_195, (128, ), (1, ))
    assert_size_stride(primals_196, (128, ), (1, ))
    assert_size_stride(primals_197, (), ())
    assert_size_stride(primals_198, (128, ), (1, ))
    assert_size_stride(primals_199, (128, ), (1, ))
    assert_size_stride(primals_200, (), ())
    assert_size_stride(primals_201, (512, ), (1, ))
    assert_size_stride(primals_202, (512, ), (1, ))
    assert_size_stride(primals_203, (), ())
    assert_size_stride(primals_204, (512, ), (1, ))
    assert_size_stride(primals_205, (512, ), (1, ))
    assert_size_stride(primals_206, (), ())
    assert_size_stride(primals_207, (128, ), (1, ))
    assert_size_stride(primals_208, (128, ), (1, ))
    assert_size_stride(primals_209, (), ())
    assert_size_stride(primals_210, (128, ), (1, ))
    assert_size_stride(primals_211, (128, ), (1, ))
    assert_size_stride(primals_212, (), ())
    assert_size_stride(primals_213, (512, ), (1, ))
    assert_size_stride(primals_214, (512, ), (1, ))
    assert_size_stride(primals_215, (), ())
    assert_size_stride(primals_216, (128, ), (1, ))
    assert_size_stride(primals_217, (128, ), (1, ))
    assert_size_stride(primals_218, (), ())
    assert_size_stride(primals_219, (128, ), (1, ))
    assert_size_stride(primals_220, (128, ), (1, ))
    assert_size_stride(primals_221, (), ())
    assert_size_stride(primals_222, (512, ), (1, ))
    assert_size_stride(primals_223, (512, ), (1, ))
    assert_size_stride(primals_224, (), ())
    assert_size_stride(primals_225, (128, ), (1, ))
    assert_size_stride(primals_226, (128, ), (1, ))
    assert_size_stride(primals_227, (), ())
    assert_size_stride(primals_228, (128, ), (1, ))
    assert_size_stride(primals_229, (128, ), (1, ))
    assert_size_stride(primals_230, (), ())
    assert_size_stride(primals_231, (512, ), (1, ))
    assert_size_stride(primals_232, (512, ), (1, ))
    assert_size_stride(primals_233, (), ())
    assert_size_stride(primals_234, (256, ), (1, ))
    assert_size_stride(primals_235, (256, ), (1, ))
    assert_size_stride(primals_236, (), ())
    assert_size_stride(primals_237, (256, ), (1, ))
    assert_size_stride(primals_238, (256, ), (1, ))
    assert_size_stride(primals_239, (), ())
    assert_size_stride(primals_240, (1024, ), (1, ))
    assert_size_stride(primals_241, (1024, ), (1, ))
    assert_size_stride(primals_242, (), ())
    assert_size_stride(primals_243, (1024, ), (1, ))
    assert_size_stride(primals_244, (1024, ), (1, ))
    assert_size_stride(primals_245, (), ())
    assert_size_stride(primals_246, (256, ), (1, ))
    assert_size_stride(primals_247, (256, ), (1, ))
    assert_size_stride(primals_248, (), ())
    assert_size_stride(primals_249, (256, ), (1, ))
    assert_size_stride(primals_250, (256, ), (1, ))
    assert_size_stride(primals_251, (), ())
    assert_size_stride(primals_252, (1024, ), (1, ))
    assert_size_stride(primals_253, (1024, ), (1, ))
    assert_size_stride(primals_254, (), ())
    assert_size_stride(primals_255, (256, ), (1, ))
    assert_size_stride(primals_256, (256, ), (1, ))
    assert_size_stride(primals_257, (), ())
    assert_size_stride(primals_258, (256, ), (1, ))
    assert_size_stride(primals_259, (256, ), (1, ))
    assert_size_stride(primals_260, (), ())
    assert_size_stride(primals_261, (1024, ), (1, ))
    assert_size_stride(primals_262, (1024, ), (1, ))
    assert_size_stride(primals_263, (), ())
    assert_size_stride(primals_264, (256, ), (1, ))
    assert_size_stride(primals_265, (256, ), (1, ))
    assert_size_stride(primals_266, (), ())
    assert_size_stride(primals_267, (256, ), (1, ))
    assert_size_stride(primals_268, (256, ), (1, ))
    assert_size_stride(primals_269, (), ())
    assert_size_stride(primals_270, (1024, ), (1, ))
    assert_size_stride(primals_271, (1024, ), (1, ))
    assert_size_stride(primals_272, (), ())
    assert_size_stride(primals_273, (256, ), (1, ))
    assert_size_stride(primals_274, (256, ), (1, ))
    assert_size_stride(primals_275, (), ())
    assert_size_stride(primals_276, (256, ), (1, ))
    assert_size_stride(primals_277, (256, ), (1, ))
    assert_size_stride(primals_278, (), ())
    assert_size_stride(primals_279, (1024, ), (1, ))
    assert_size_stride(primals_280, (1024, ), (1, ))
    assert_size_stride(primals_281, (), ())
    assert_size_stride(primals_282, (256, ), (1, ))
    assert_size_stride(primals_283, (256, ), (1, ))
    assert_size_stride(primals_284, (), ())
    assert_size_stride(primals_285, (256, ), (1, ))
    assert_size_stride(primals_286, (256, ), (1, ))
    assert_size_stride(primals_287, (), ())
    assert_size_stride(primals_288, (1024, ), (1, ))
    assert_size_stride(primals_289, (1024, ), (1, ))
    assert_size_stride(primals_290, (), ())
    assert_size_stride(primals_291, (512, ), (1, ))
    assert_size_stride(primals_292, (512, ), (1, ))
    assert_size_stride(primals_293, (), ())
    assert_size_stride(primals_294, (512, ), (1, ))
    assert_size_stride(primals_295, (512, ), (1, ))
    assert_size_stride(primals_296, (), ())
    assert_size_stride(primals_297, (2048, ), (1, ))
    assert_size_stride(primals_298, (2048, ), (1, ))
    assert_size_stride(primals_299, (), ())
    assert_size_stride(primals_300, (2048, ), (1, ))
    assert_size_stride(primals_301, (2048, ), (1, ))
    assert_size_stride(primals_302, (), ())
    assert_size_stride(primals_303, (512, ), (1, ))
    assert_size_stride(primals_304, (512, ), (1, ))
    assert_size_stride(primals_305, (), ())
    assert_size_stride(primals_306, (512, ), (1, ))
    assert_size_stride(primals_307, (512, ), (1, ))
    assert_size_stride(primals_308, (), ())
    assert_size_stride(primals_309, (2048, ), (1, ))
    assert_size_stride(primals_310, (2048, ), (1, ))
    assert_size_stride(primals_311, (), ())
    assert_size_stride(primals_312, (512, ), (1, ))
    assert_size_stride(primals_313, (512, ), (1, ))
    assert_size_stride(primals_314, (), ())
    assert_size_stride(primals_315, (512, ), (1, ))
    assert_size_stride(primals_316, (512, ), (1, ))
    assert_size_stride(primals_317, (), ())
    assert_size_stride(primals_318, (2048, ), (1, ))
    assert_size_stride(primals_319, (2048, ), (1, ))
    assert_size_stride(primals_320, (), ())
    assert_size_stride(primals_321, (64, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((64, 3, 7, 7), (147, 1, 21, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        stream0 = get_cuda_stream(0)
        triton_poi_fused_0.run(primals_1, buf0, 192, 49, grid=grid(192, 49), stream=stream0)
        del primals_1
        buf1 = empty_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_7, buf1, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_7
        buf2 = empty_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_19, buf2, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_19
        buf3 = empty_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_28, buf3, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_28
        buf4 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_37, buf4, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_37
        buf5 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_49, buf5, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_49
        buf6 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_58, buf6, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_58
        buf7 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_67, buf7, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_67
        buf8 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_76, buf8, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_76
        buf9 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_88, buf9, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_88
        buf10 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_97, buf10, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_97
        buf11 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_106, buf11, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_106
        buf12 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_115, buf12, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_115
        buf13 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_124, buf13, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_124
        buf14 = empty_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_133, buf14, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_133
        buf15 = empty_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_145, buf15, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_145
        buf16 = empty_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_154, buf16, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_154
        buf17 = empty_strided((64, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_321, buf17, 192, 50176, grid=grid(192, 50176), stream=stream0)
        del primals_321
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, buf0, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (64, 64, 112, 112), (802816, 1, 7168, 64))
        buf19 = empty_strided((1, 64, 1, 1, 896), (57344, 1, 57344, 57344, 64), device='cuda', dtype=torch.float32)
        buf20 = empty_strided((1, 64, 1, 1, 896), (57344, 1, 57344, 57344, 64), device='cuda', dtype=torch.float32)
        buf21 = empty_strided((1, 64, 1, 1, 896), (57344, 1, 57344, 57344, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_6.run(buf18, buf19, buf20, buf21, 57344, 896, grid=grid(57344), stream=stream0)
        buf22 = empty_strided((1, 64, 1, 1, 7), (448, 1, 448, 448, 64), device='cuda', dtype=torch.float32)
        buf23 = empty_strided((1, 64, 1, 1, 7), (448, 1, 448, 448, 64), device='cuda', dtype=torch.float32)
        buf24 = empty_strided((1, 64, 1, 1, 7), (448, 1, 448, 448, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_7.run(buf19, buf20, buf21, buf22, buf23, buf24, 448, 128, grid=grid(448), stream=stream0)
        buf25 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf26 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf28 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_8.run(buf22, buf23, buf24, primals_162, primals_163, buf25, buf26, buf28, primals_162, primals_163, 64, 7, grid=grid(64), stream=stream0)
        del primals_162
        del primals_163
        buf29 = empty_strided((64, 64, 112, 112), (802816, 1, 7168, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1, x_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_9.run(buf18, buf25, buf26, primals_2, primals_3, buf29, 51380224, grid=grid(51380224), stream=stream0)
        del primals_3
        buf30 = empty_strided((64, 64, 56, 56), (200704, 1, 3584, 64), device='cuda', dtype=torch.float32)
        buf31 = empty_strided((64, 64, 56, 56), (200704, 1, 3584, 64), device='cuda', dtype=torch.int64)
        # Source Nodes: [identity], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_10.run(buf29, buf30, buf31, 12845056, grid=grid(12845056), stream=stream0)
        # Source Nodes: [out], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf30, primals_4, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (64, 64, 56, 56), (200704, 1, 3584, 64))
        buf33 = buf21; del buf21  # reuse
        buf34 = buf20; del buf20  # reuse
        buf35 = buf19; del buf19  # reuse
        # Source Nodes: [out_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_11.run(buf32, buf33, buf34, buf35, 57344, 224, grid=grid(57344), stream=stream0)
        buf36 = buf24; del buf24  # reuse
        buf37 = buf23; del buf23  # reuse
        buf38 = buf22; del buf22  # reuse
        # Source Nodes: [out_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_7.run(buf33, buf34, buf35, buf36, buf37, buf38, 448, 128, grid=grid(448), stream=stream0)
        buf39 = buf26; del buf26  # reuse
        buf40 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf42 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_12.run(buf36, buf37, buf38, primals_165, primals_166, buf39, buf40, buf42, primals_165, primals_166, 64, 7, grid=grid(64), stream=stream0)
        del primals_165
        del primals_166
        buf43 = empty_strided((64, 64, 56, 56), (200704, 1, 3584, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_1, out_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_13.run(buf32, buf39, buf40, primals_5, primals_6, buf43, 12845056, grid=grid(12845056), stream=stream0)
        del primals_6
        # Source Nodes: [out_3], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, buf1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (64, 64, 56, 56), (200704, 1, 3584, 64))
        buf45 = buf35; del buf35  # reuse
        buf46 = buf34; del buf34  # reuse
        buf47 = buf33; del buf33  # reuse
        # Source Nodes: [out_4], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_11.run(buf44, buf45, buf46, buf47, 57344, 224, grid=grid(57344), stream=stream0)
        buf48 = buf38; del buf38  # reuse
        buf49 = buf37; del buf37  # reuse
        buf50 = buf36; del buf36  # reuse
        # Source Nodes: [out_4], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_7.run(buf45, buf46, buf47, buf48, buf49, buf50, 448, 128, grid=grid(448), stream=stream0)
        buf51 = buf40; del buf40  # reuse
        buf52 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf54 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_4], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_12.run(buf48, buf49, buf50, primals_168, primals_169, buf51, buf52, buf54, primals_168, primals_169, 64, 7, grid=grid(64), stream=stream0)
        del primals_168
        del primals_169
        buf55 = empty_strided((64, 64, 56, 56), (200704, 1, 3584, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_4, out_5], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_13.run(buf44, buf51, buf52, primals_8, primals_9, buf55, 12845056, grid=grid(12845056), stream=stream0)
        del primals_9
        # Source Nodes: [out_6], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, primals_10, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (64, 256, 56, 56), (802816, 1, 14336, 256))
        buf57 = empty_strided((1, 256, 1, 1, 448), (114688, 1, 114688, 114688, 256), device='cuda', dtype=torch.float32)
        buf58 = empty_strided((1, 256, 1, 1, 448), (114688, 1, 114688, 114688, 256), device='cuda', dtype=torch.float32)
        buf59 = empty_strided((1, 256, 1, 1, 448), (114688, 1, 114688, 114688, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_7], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_14.run(buf56, buf57, buf58, buf59, 114688, 448, grid=grid(114688), stream=stream0)
        buf60 = empty_strided((1, 256, 1, 1, 4), (1024, 1, 1024, 1024, 256), device='cuda', dtype=torch.float32)
        buf61 = empty_strided((1, 256, 1, 1, 4), (1024, 1, 1024, 1024, 256), device='cuda', dtype=torch.float32)
        buf62 = empty_strided((1, 256, 1, 1, 4), (1024, 1, 1024, 1024, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_7], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_15.run(buf57, buf58, buf59, buf60, buf61, buf62, 1024, 112, grid=grid(1024), stream=stream0)
        buf63 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf64 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf66 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_7], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_16.run(buf60, buf61, buf62, primals_171, primals_172, buf63, buf64, buf66, primals_171, primals_172, 256, 4, grid=grid(256), stream=stream0)
        del primals_171
        del primals_172
        # Source Nodes: [getattr_l__self___layer1___0___downsample_0], Original ATen: [aten.convolution]
        buf67 = extern_kernels.convolution(buf30, primals_13, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf67, (64, 256, 56, 56), (802816, 1, 14336, 256))
        buf68 = buf59; del buf59  # reuse
        buf69 = buf58; del buf58  # reuse
        buf70 = buf57; del buf57  # reuse
        # Source Nodes: [identity_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_14.run(buf67, buf68, buf69, buf70, 114688, 448, grid=grid(114688), stream=stream0)
        buf71 = buf62; del buf62  # reuse
        buf72 = buf61; del buf61  # reuse
        buf73 = buf60; del buf60  # reuse
        # Source Nodes: [identity_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_15.run(buf68, buf69, buf70, buf71, buf72, buf73, 1024, 112, grid=grid(1024), stream=stream0)
        buf74 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf75 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf77 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [identity_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_16.run(buf71, buf72, buf73, primals_174, primals_175, buf74, buf75, buf77, primals_174, primals_175, 256, 4, grid=grid(256), stream=stream0)
        del primals_174
        del primals_175
        buf78 = empty_strided((64, 256, 56, 56), (802816, 1, 14336, 256), device='cuda', dtype=torch.float32)
        buf79 = buf78; del buf78  # reuse
        # Source Nodes: [identity_1, identity_2, out_7, out_8], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_17.run(buf79, buf56, buf63, buf64, primals_11, primals_12, buf67, buf74, buf75, primals_14, primals_15, 51380224, grid=grid(51380224), stream=stream0)
        del primals_12
        del primals_15
        # Source Nodes: [out_10], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf79, primals_16, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (64, 64, 56, 56), (200704, 1, 3584, 64))
        buf81 = buf47; del buf47  # reuse
        buf82 = buf46; del buf46  # reuse
        buf83 = buf45; del buf45  # reuse
        # Source Nodes: [out_11], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_11.run(buf80, buf81, buf82, buf83, 57344, 224, grid=grid(57344), stream=stream0)
        buf84 = buf50; del buf50  # reuse
        buf85 = buf49; del buf49  # reuse
        buf86 = buf48; del buf48  # reuse
        # Source Nodes: [out_11], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_7.run(buf81, buf82, buf83, buf84, buf85, buf86, 448, 128, grid=grid(448), stream=stream0)
        buf87 = buf52; del buf52  # reuse
        buf88 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf90 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_11], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_12.run(buf84, buf85, buf86, primals_177, primals_178, buf87, buf88, buf90, primals_177, primals_178, 64, 7, grid=grid(64), stream=stream0)
        del primals_177
        del primals_178
        buf91 = empty_strided((64, 64, 56, 56), (200704, 1, 3584, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_11, out_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_13.run(buf80, buf87, buf88, primals_17, primals_18, buf91, 12845056, grid=grid(12845056), stream=stream0)
        del primals_18
        # Source Nodes: [out_13], Original ATen: [aten.convolution]
        buf92 = extern_kernels.convolution(buf91, buf2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (64, 64, 56, 56), (200704, 1, 3584, 64))
        buf93 = buf83; del buf83  # reuse
        buf94 = buf82; del buf82  # reuse
        buf95 = buf81; del buf81  # reuse
        # Source Nodes: [out_14], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_11.run(buf92, buf93, buf94, buf95, 57344, 224, grid=grid(57344), stream=stream0)
        buf96 = buf86; del buf86  # reuse
        buf97 = buf85; del buf85  # reuse
        buf98 = buf84; del buf84  # reuse
        # Source Nodes: [out_14], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_7.run(buf93, buf94, buf95, buf96, buf97, buf98, 448, 128, grid=grid(448), stream=stream0)
        buf99 = buf88; del buf88  # reuse
        buf100 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf102 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_14], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_12.run(buf96, buf97, buf98, primals_180, primals_181, buf99, buf100, buf102, primals_180, primals_181, 64, 7, grid=grid(64), stream=stream0)
        del primals_180
        del primals_181
        buf103 = empty_strided((64, 64, 56, 56), (200704, 1, 3584, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_14, out_15], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_13.run(buf92, buf99, buf100, primals_20, primals_21, buf103, 12845056, grid=grid(12845056), stream=stream0)
        del primals_21
        # Source Nodes: [out_16], Original ATen: [aten.convolution]
        buf104 = extern_kernels.convolution(buf103, primals_22, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf104, (64, 256, 56, 56), (802816, 1, 14336, 256))
        buf105 = buf70; del buf70  # reuse
        buf106 = buf69; del buf69  # reuse
        buf107 = buf68; del buf68  # reuse
        # Source Nodes: [out_17], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_14.run(buf104, buf105, buf106, buf107, 114688, 448, grid=grid(114688), stream=stream0)
        buf108 = buf73; del buf73  # reuse
        buf109 = buf72; del buf72  # reuse
        buf110 = buf71; del buf71  # reuse
        # Source Nodes: [out_17], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_15.run(buf105, buf106, buf107, buf108, buf109, buf110, 1024, 112, grid=grid(1024), stream=stream0)
        buf111 = buf75; del buf75  # reuse
        buf112 = buf64; del buf64  # reuse
        buf114 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_17], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_16.run(buf108, buf109, buf110, primals_183, primals_184, buf111, buf112, buf114, primals_183, primals_184, 256, 4, grid=grid(256), stream=stream0)
        del primals_183
        del primals_184
        buf115 = empty_strided((64, 256, 56, 56), (802816, 1, 14336, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [identity_3, out_17, out_18], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_18.run(buf104, buf111, buf112, primals_23, primals_24, buf79, buf115, 51380224, grid=grid(51380224), stream=stream0)
        del primals_24
        # Source Nodes: [out_20], Original ATen: [aten.convolution]
        buf116 = extern_kernels.convolution(buf115, primals_25, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (64, 64, 56, 56), (200704, 1, 3584, 64))
        buf117 = buf95; del buf95  # reuse
        buf118 = buf94; del buf94  # reuse
        buf119 = buf93; del buf93  # reuse
        # Source Nodes: [out_21], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_11.run(buf116, buf117, buf118, buf119, 57344, 224, grid=grid(57344), stream=stream0)
        buf120 = buf98; del buf98  # reuse
        buf121 = buf97; del buf97  # reuse
        buf122 = buf96; del buf96  # reuse
        # Source Nodes: [out_21], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_7.run(buf117, buf118, buf119, buf120, buf121, buf122, 448, 128, grid=grid(448), stream=stream0)
        buf123 = buf100; del buf100  # reuse
        buf124 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf126 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_21], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_12.run(buf120, buf121, buf122, primals_186, primals_187, buf123, buf124, buf126, primals_186, primals_187, 64, 7, grid=grid(64), stream=stream0)
        del primals_186
        del primals_187
        buf127 = empty_strided((64, 64, 56, 56), (200704, 1, 3584, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_21, out_22], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_13.run(buf116, buf123, buf124, primals_26, primals_27, buf127, 12845056, grid=grid(12845056), stream=stream0)
        del primals_27
        # Source Nodes: [out_23], Original ATen: [aten.convolution]
        buf128 = extern_kernels.convolution(buf127, buf3, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (64, 64, 56, 56), (200704, 1, 3584, 64))
        buf129 = buf119; del buf119  # reuse
        buf130 = buf118; del buf118  # reuse
        buf131 = buf117; del buf117  # reuse
        # Source Nodes: [out_24], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_11.run(buf128, buf129, buf130, buf131, 57344, 224, grid=grid(57344), stream=stream0)
        buf132 = buf122; del buf122  # reuse
        buf133 = buf121; del buf121  # reuse
        buf134 = buf120; del buf120  # reuse
        # Source Nodes: [out_24], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_7.run(buf129, buf130, buf131, buf132, buf133, buf134, 448, 128, grid=grid(448), stream=stream0)
        del buf129
        del buf130
        del buf131
        buf135 = buf124; del buf124  # reuse
        buf136 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf138 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_24], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_12.run(buf132, buf133, buf134, primals_189, primals_190, buf135, buf136, buf138, primals_189, primals_190, 64, 7, grid=grid(64), stream=stream0)
        del buf132
        del buf133
        del buf134
        del primals_189
        del primals_190
        buf139 = empty_strided((64, 64, 56, 56), (200704, 1, 3584, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_24, out_25], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_13.run(buf128, buf135, buf136, primals_29, primals_30, buf139, 12845056, grid=grid(12845056), stream=stream0)
        del buf136
        del primals_30
        # Source Nodes: [out_26], Original ATen: [aten.convolution]
        buf140 = extern_kernels.convolution(buf139, primals_31, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf140, (64, 256, 56, 56), (802816, 1, 14336, 256))
        buf141 = buf107; del buf107  # reuse
        buf142 = buf106; del buf106  # reuse
        buf143 = buf105; del buf105  # reuse
        # Source Nodes: [out_27], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_14.run(buf140, buf141, buf142, buf143, 114688, 448, grid=grid(114688), stream=stream0)
        buf144 = buf110; del buf110  # reuse
        buf145 = buf109; del buf109  # reuse
        buf146 = buf108; del buf108  # reuse
        # Source Nodes: [out_27], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_15.run(buf141, buf142, buf143, buf144, buf145, buf146, 1024, 112, grid=grid(1024), stream=stream0)
        buf147 = buf112; del buf112  # reuse
        buf148 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf150 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_27], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_16.run(buf144, buf145, buf146, primals_192, primals_193, buf147, buf148, buf150, primals_192, primals_193, 256, 4, grid=grid(256), stream=stream0)
        del primals_192
        del primals_193
        buf151 = empty_strided((64, 256, 56, 56), (802816, 1, 14336, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [identity_4, out_27, out_28], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_18.run(buf140, buf147, buf148, primals_32, primals_33, buf115, buf151, 51380224, grid=grid(51380224), stream=stream0)
        del primals_33
        # Source Nodes: [out_30], Original ATen: [aten.convolution]
        buf152 = extern_kernels.convolution(buf151, primals_34, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf152, (64, 128, 56, 56), (401408, 1, 7168, 128))
        buf153 = reinterpret_tensor(buf143, (1, 128, 1, 1, 896), (114688, 1, 114688, 114688, 128), 0); del buf143  # reuse
        buf154 = reinterpret_tensor(buf142, (1, 128, 1, 1, 896), (114688, 1, 114688, 114688, 128), 0); del buf142  # reuse
        buf155 = reinterpret_tensor(buf141, (1, 128, 1, 1, 896), (114688, 1, 114688, 114688, 128), 0); del buf141  # reuse
        # Source Nodes: [out_31], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf152, buf153, buf154, buf155, 114688, 224, grid=grid(114688), stream=stream0)
        buf156 = empty_strided((1, 128, 1, 1, 7), (896, 1, 896, 896, 128), device='cuda', dtype=torch.float32)
        buf157 = empty_strided((1, 128, 1, 1, 7), (896, 1, 896, 896, 128), device='cuda', dtype=torch.float32)
        buf158 = empty_strided((1, 128, 1, 1, 7), (896, 1, 896, 896, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_31], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf153, buf154, buf155, buf156, buf157, buf158, 896, 128, grid=grid(896), stream=stream0)
        buf159 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf160 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf162 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_31], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_21.run(buf156, buf157, buf158, primals_195, primals_196, buf159, buf160, buf162, primals_195, primals_196, 128, 7, grid=grid(128), stream=stream0)
        del buf156
        del buf157
        del buf158
        del primals_195
        del primals_196
        buf163 = empty_strided((64, 128, 56, 56), (401408, 1, 7168, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_31, out_32], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_22.run(buf152, buf159, buf160, primals_35, primals_36, buf163, 25690112, grid=grid(25690112), stream=stream0)
        del primals_36
        # Source Nodes: [out_33], Original ATen: [aten.convolution]
        buf164 = extern_kernels.convolution(buf163, buf4, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf164, (64, 128, 28, 28), (100352, 1, 3584, 128))
        buf165 = empty_strided((1, 128, 1, 1, 392), (50176, 1, 50176, 50176, 128), device='cuda', dtype=torch.float32)
        buf166 = empty_strided((1, 128, 1, 1, 392), (50176, 1, 50176, 50176, 128), device='cuda', dtype=torch.float32)
        buf167 = empty_strided((1, 128, 1, 1, 392), (50176, 1, 50176, 50176, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_34], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_23.run(buf164, buf165, buf166, buf167, 50176, 128, grid=grid(50176), stream=stream0)
        buf168 = empty_strided((1, 128, 1, 1, 4), (512, 1, 512, 512, 128), device='cuda', dtype=torch.float32)
        buf169 = empty_strided((1, 128, 1, 1, 4), (512, 1, 512, 512, 128), device='cuda', dtype=torch.float32)
        buf170 = empty_strided((1, 128, 1, 1, 4), (512, 1, 512, 512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_34], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_24.run(buf165, buf166, buf167, buf168, buf169, buf170, 512, 98, grid=grid(512), stream=stream0)
        buf171 = buf160; del buf160  # reuse
        buf172 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf174 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_34], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_25.run(buf168, buf169, buf170, primals_198, primals_199, buf171, buf172, buf174, primals_198, primals_199, 128, 4, grid=grid(128), stream=stream0)
        del primals_198
        del primals_199
        buf175 = empty_strided((64, 128, 28, 28), (100352, 1, 3584, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_34, out_35], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_26.run(buf164, buf171, buf172, primals_38, primals_39, buf175, 6422528, grid=grid(6422528), stream=stream0)
        del primals_39
        # Source Nodes: [out_36], Original ATen: [aten.convolution]
        buf176 = extern_kernels.convolution(buf175, primals_40, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf176, (64, 512, 28, 28), (401408, 1, 14336, 512))
        buf177 = reinterpret_tensor(buf155, (1, 512, 1, 1, 224), (114688, 1, 114688, 114688, 512), 0); del buf155  # reuse
        buf178 = reinterpret_tensor(buf154, (1, 512, 1, 1, 224), (114688, 1, 114688, 114688, 512), 0); del buf154  # reuse
        buf179 = reinterpret_tensor(buf153, (1, 512, 1, 1, 224), (114688, 1, 114688, 114688, 512), 0); del buf153  # reuse
        # Source Nodes: [out_37], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_27.run(buf176, buf177, buf178, buf179, 114688, 224, grid=grid(114688), stream=stream0)
        buf180 = reinterpret_tensor(buf146, (1, 512, 1, 1, 2), (1024, 1, 1024, 1024, 512), 0); del buf146  # reuse
        buf181 = reinterpret_tensor(buf145, (1, 512, 1, 1, 2), (1024, 1, 1024, 1024, 512), 0); del buf145  # reuse
        buf182 = reinterpret_tensor(buf144, (1, 512, 1, 1, 2), (1024, 1, 1024, 1024, 512), 0); del buf144  # reuse
        # Source Nodes: [out_37], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_28.run(buf177, buf178, buf179, buf180, buf181, buf182, 1024, 112, grid=grid(1024), stream=stream0)
        buf183 = reinterpret_tensor(buf170, (1, 512, 1, 1), (512, 1, 512, 512), 0); del buf170  # reuse
        buf184 = reinterpret_tensor(buf169, (1, 512, 1, 1), (512, 1, 512, 512), 0); del buf169  # reuse
        buf186 = reinterpret_tensor(buf168, (512, ), (1, ), 0); del buf168  # reuse
        # Source Nodes: [out_37], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_29.run(buf180, buf181, buf182, primals_201, primals_202, buf183, buf184, buf186, primals_201, primals_202, 512, 2, grid=grid(512), stream=stream0)
        del primals_201
        del primals_202
        # Source Nodes: [getattr_l__self___layer2___0___downsample_0], Original ATen: [aten.convolution]
        buf187 = extern_kernels.convolution(buf151, primals_43, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf187, (64, 512, 28, 28), (401408, 1, 14336, 512))
        buf188 = buf179; del buf179  # reuse
        buf189 = buf178; del buf178  # reuse
        buf190 = buf177; del buf177  # reuse
        # Source Nodes: [identity_5], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_27.run(buf187, buf188, buf189, buf190, 114688, 224, grid=grid(114688), stream=stream0)
        buf191 = buf182; del buf182  # reuse
        buf192 = buf181; del buf181  # reuse
        buf193 = buf180; del buf180  # reuse
        # Source Nodes: [identity_5], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_28.run(buf188, buf189, buf190, buf191, buf192, buf193, 1024, 112, grid=grid(1024), stream=stream0)
        buf194 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf195 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf197 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [identity_5], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_29.run(buf191, buf192, buf193, primals_204, primals_205, buf194, buf195, buf197, primals_204, primals_205, 512, 2, grid=grid(512), stream=stream0)
        del primals_204
        del primals_205
        buf198 = empty_strided((64, 512, 28, 28), (401408, 1, 14336, 512), device='cuda', dtype=torch.float32)
        buf199 = buf198; del buf198  # reuse
        # Source Nodes: [identity_5, identity_6, out_37, out_38], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_30.run(buf199, buf176, buf183, buf184, primals_41, primals_42, buf187, buf194, buf195, primals_44, primals_45, 25690112, grid=grid(25690112), stream=stream0)
        del primals_42
        del primals_45
        # Source Nodes: [out_40], Original ATen: [aten.convolution]
        buf200 = extern_kernels.convolution(buf199, primals_46, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf200, (64, 128, 28, 28), (100352, 1, 3584, 128))
        buf201 = buf167; del buf167  # reuse
        buf202 = buf166; del buf166  # reuse
        buf203 = buf165; del buf165  # reuse
        # Source Nodes: [out_41], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_23.run(buf200, buf201, buf202, buf203, 50176, 128, grid=grid(50176), stream=stream0)
        buf204 = reinterpret_tensor(buf195, (1, 128, 1, 1, 4), (512, 1, 512, 512, 128), 0); del buf195  # reuse
        buf205 = reinterpret_tensor(buf184, (1, 128, 1, 1, 4), (512, 1, 512, 512, 128), 0); del buf184  # reuse
        buf206 = empty_strided((1, 128, 1, 1, 4), (512, 1, 512, 512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_41], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_24.run(buf201, buf202, buf203, buf204, buf205, buf206, 512, 98, grid=grid(512), stream=stream0)
        buf207 = buf172; del buf172  # reuse
        buf208 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf210 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_41], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_25.run(buf204, buf205, buf206, primals_207, primals_208, buf207, buf208, buf210, primals_207, primals_208, 128, 4, grid=grid(128), stream=stream0)
        del primals_207
        del primals_208
        buf211 = empty_strided((64, 128, 28, 28), (100352, 1, 3584, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_41, out_42], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_26.run(buf200, buf207, buf208, primals_47, primals_48, buf211, 6422528, grid=grid(6422528), stream=stream0)
        del primals_48
        # Source Nodes: [out_43], Original ATen: [aten.convolution]
        buf212 = extern_kernels.convolution(buf211, buf5, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf212, (64, 128, 28, 28), (100352, 1, 3584, 128))
        buf213 = buf203; del buf203  # reuse
        buf214 = buf202; del buf202  # reuse
        buf215 = buf201; del buf201  # reuse
        # Source Nodes: [out_44], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_23.run(buf212, buf213, buf214, buf215, 50176, 128, grid=grid(50176), stream=stream0)
        buf216 = buf206; del buf206  # reuse
        buf217 = buf205; del buf205  # reuse
        buf218 = buf204; del buf204  # reuse
        # Source Nodes: [out_44], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_24.run(buf213, buf214, buf215, buf216, buf217, buf218, 512, 98, grid=grid(512), stream=stream0)
        buf219 = buf208; del buf208  # reuse
        buf220 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf222 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_44], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_25.run(buf216, buf217, buf218, primals_210, primals_211, buf219, buf220, buf222, primals_210, primals_211, 128, 4, grid=grid(128), stream=stream0)
        del primals_210
        del primals_211
        buf223 = empty_strided((64, 128, 28, 28), (100352, 1, 3584, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_44, out_45], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_26.run(buf212, buf219, buf220, primals_50, primals_51, buf223, 6422528, grid=grid(6422528), stream=stream0)
        del primals_51
        # Source Nodes: [out_46], Original ATen: [aten.convolution]
        buf224 = extern_kernels.convolution(buf223, primals_52, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf224, (64, 512, 28, 28), (401408, 1, 14336, 512))
        buf225 = buf190; del buf190  # reuse
        buf226 = buf189; del buf189  # reuse
        buf227 = buf188; del buf188  # reuse
        # Source Nodes: [out_47], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_27.run(buf224, buf225, buf226, buf227, 114688, 224, grid=grid(114688), stream=stream0)
        buf228 = buf193; del buf193  # reuse
        buf229 = buf192; del buf192  # reuse
        buf230 = buf191; del buf191  # reuse
        # Source Nodes: [out_47], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_28.run(buf225, buf226, buf227, buf228, buf229, buf230, 1024, 112, grid=grid(1024), stream=stream0)
        buf231 = reinterpret_tensor(buf218, (1, 512, 1, 1), (512, 1, 512, 512), 0); del buf218  # reuse
        buf232 = reinterpret_tensor(buf217, (1, 512, 1, 1), (512, 1, 512, 512), 0); del buf217  # reuse
        buf234 = reinterpret_tensor(buf216, (512, ), (1, ), 0); del buf216  # reuse
        # Source Nodes: [out_47], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_29.run(buf228, buf229, buf230, primals_213, primals_214, buf231, buf232, buf234, primals_213, primals_214, 512, 2, grid=grid(512), stream=stream0)
        del primals_213
        del primals_214
        buf235 = empty_strided((64, 512, 28, 28), (401408, 1, 14336, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [identity_7, out_47, out_48], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_31.run(buf224, buf231, buf232, primals_53, primals_54, buf199, buf235, 25690112, grid=grid(25690112), stream=stream0)
        del primals_54
        # Source Nodes: [out_50], Original ATen: [aten.convolution]
        buf236 = extern_kernels.convolution(buf235, primals_55, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf236, (64, 128, 28, 28), (100352, 1, 3584, 128))
        buf237 = buf215; del buf215  # reuse
        buf238 = buf214; del buf214  # reuse
        buf239 = buf213; del buf213  # reuse
        # Source Nodes: [out_51], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_23.run(buf236, buf237, buf238, buf239, 50176, 128, grid=grid(50176), stream=stream0)
        buf240 = reinterpret_tensor(buf232, (1, 128, 1, 1, 4), (512, 1, 512, 512, 128), 0); del buf232  # reuse
        buf241 = empty_strided((1, 128, 1, 1, 4), (512, 1, 512, 512, 128), device='cuda', dtype=torch.float32)
        buf242 = empty_strided((1, 128, 1, 1, 4), (512, 1, 512, 512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_51], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_24.run(buf237, buf238, buf239, buf240, buf241, buf242, 512, 98, grid=grid(512), stream=stream0)
        buf243 = buf220; del buf220  # reuse
        buf244 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf246 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_51], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_25.run(buf240, buf241, buf242, primals_216, primals_217, buf243, buf244, buf246, primals_216, primals_217, 128, 4, grid=grid(128), stream=stream0)
        del primals_216
        del primals_217
        buf247 = empty_strided((64, 128, 28, 28), (100352, 1, 3584, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_51, out_52], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_26.run(buf236, buf243, buf244, primals_56, primals_57, buf247, 6422528, grid=grid(6422528), stream=stream0)
        del primals_57
        # Source Nodes: [out_53], Original ATen: [aten.convolution]
        buf248 = extern_kernels.convolution(buf247, buf6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf248, (64, 128, 28, 28), (100352, 1, 3584, 128))
        buf249 = buf239; del buf239  # reuse
        buf250 = buf238; del buf238  # reuse
        buf251 = buf237; del buf237  # reuse
        # Source Nodes: [out_54], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_23.run(buf248, buf249, buf250, buf251, 50176, 128, grid=grid(50176), stream=stream0)
        buf252 = buf242; del buf242  # reuse
        buf253 = buf241; del buf241  # reuse
        buf254 = buf240; del buf240  # reuse
        # Source Nodes: [out_54], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_24.run(buf249, buf250, buf251, buf252, buf253, buf254, 512, 98, grid=grid(512), stream=stream0)
        buf255 = buf244; del buf244  # reuse
        buf256 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf258 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_54], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_25.run(buf252, buf253, buf254, primals_219, primals_220, buf255, buf256, buf258, primals_219, primals_220, 128, 4, grid=grid(128), stream=stream0)
        del primals_219
        del primals_220
        buf259 = empty_strided((64, 128, 28, 28), (100352, 1, 3584, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_54, out_55], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_26.run(buf248, buf255, buf256, primals_59, primals_60, buf259, 6422528, grid=grid(6422528), stream=stream0)
        del primals_60
        # Source Nodes: [out_56], Original ATen: [aten.convolution]
        buf260 = extern_kernels.convolution(buf259, primals_61, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf260, (64, 512, 28, 28), (401408, 1, 14336, 512))
        buf261 = buf227; del buf227  # reuse
        buf262 = buf226; del buf226  # reuse
        buf263 = buf225; del buf225  # reuse
        # Source Nodes: [out_57], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_27.run(buf260, buf261, buf262, buf263, 114688, 224, grid=grid(114688), stream=stream0)
        buf264 = buf230; del buf230  # reuse
        buf265 = buf229; del buf229  # reuse
        buf266 = buf228; del buf228  # reuse
        # Source Nodes: [out_57], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_28.run(buf261, buf262, buf263, buf264, buf265, buf266, 1024, 112, grid=grid(1024), stream=stream0)
        buf267 = reinterpret_tensor(buf254, (1, 512, 1, 1), (512, 1, 512, 512), 0); del buf254  # reuse
        buf268 = reinterpret_tensor(buf253, (1, 512, 1, 1), (512, 1, 512, 512), 0); del buf253  # reuse
        buf270 = reinterpret_tensor(buf252, (512, ), (1, ), 0); del buf252  # reuse
        # Source Nodes: [out_57], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_29.run(buf264, buf265, buf266, primals_222, primals_223, buf267, buf268, buf270, primals_222, primals_223, 512, 2, grid=grid(512), stream=stream0)
        del primals_222
        del primals_223
        buf271 = empty_strided((64, 512, 28, 28), (401408, 1, 14336, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [identity_8, out_57, out_58], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_31.run(buf260, buf267, buf268, primals_62, primals_63, buf235, buf271, 25690112, grid=grid(25690112), stream=stream0)
        del primals_63
        # Source Nodes: [out_60], Original ATen: [aten.convolution]
        buf272 = extern_kernels.convolution(buf271, primals_64, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf272, (64, 128, 28, 28), (100352, 1, 3584, 128))
        buf273 = buf251; del buf251  # reuse
        buf274 = buf250; del buf250  # reuse
        buf275 = buf249; del buf249  # reuse
        # Source Nodes: [out_61], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_23.run(buf272, buf273, buf274, buf275, 50176, 128, grid=grid(50176), stream=stream0)
        buf276 = reinterpret_tensor(buf268, (1, 128, 1, 1, 4), (512, 1, 512, 512, 128), 0); del buf268  # reuse
        buf277 = empty_strided((1, 128, 1, 1, 4), (512, 1, 512, 512, 128), device='cuda', dtype=torch.float32)
        buf278 = empty_strided((1, 128, 1, 1, 4), (512, 1, 512, 512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_61], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_24.run(buf273, buf274, buf275, buf276, buf277, buf278, 512, 98, grid=grid(512), stream=stream0)
        buf279 = buf256; del buf256  # reuse
        buf280 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf282 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_61], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_25.run(buf276, buf277, buf278, primals_225, primals_226, buf279, buf280, buf282, primals_225, primals_226, 128, 4, grid=grid(128), stream=stream0)
        del primals_225
        del primals_226
        buf283 = empty_strided((64, 128, 28, 28), (100352, 1, 3584, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_61, out_62], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_26.run(buf272, buf279, buf280, primals_65, primals_66, buf283, 6422528, grid=grid(6422528), stream=stream0)
        del primals_66
        # Source Nodes: [out_63], Original ATen: [aten.convolution]
        buf284 = extern_kernels.convolution(buf283, buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf284, (64, 128, 28, 28), (100352, 1, 3584, 128))
        buf285 = buf275; del buf275  # reuse
        buf286 = buf274; del buf274  # reuse
        buf287 = buf273; del buf273  # reuse
        # Source Nodes: [out_64], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_23.run(buf284, buf285, buf286, buf287, 50176, 128, grid=grid(50176), stream=stream0)
        buf288 = buf278; del buf278  # reuse
        buf289 = buf277; del buf277  # reuse
        buf290 = buf276; del buf276  # reuse
        # Source Nodes: [out_64], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_24.run(buf285, buf286, buf287, buf288, buf289, buf290, 512, 98, grid=grid(512), stream=stream0)
        buf291 = buf280; del buf280  # reuse
        buf292 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf294 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_64], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_25.run(buf288, buf289, buf290, primals_228, primals_229, buf291, buf292, buf294, primals_228, primals_229, 128, 4, grid=grid(128), stream=stream0)
        del primals_228
        del primals_229
        buf295 = empty_strided((64, 128, 28, 28), (100352, 1, 3584, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_64, out_65], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_26.run(buf284, buf291, buf292, primals_68, primals_69, buf295, 6422528, grid=grid(6422528), stream=stream0)
        del buf292
        del primals_69
        # Source Nodes: [out_66], Original ATen: [aten.convolution]
        buf296 = extern_kernels.convolution(buf295, primals_70, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf296, (64, 512, 28, 28), (401408, 1, 14336, 512))
        buf297 = buf263; del buf263  # reuse
        buf298 = buf262; del buf262  # reuse
        buf299 = buf261; del buf261  # reuse
        # Source Nodes: [out_67], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_27.run(buf296, buf297, buf298, buf299, 114688, 224, grid=grid(114688), stream=stream0)
        buf300 = buf266; del buf266  # reuse
        buf301 = buf265; del buf265  # reuse
        buf302 = buf264; del buf264  # reuse
        # Source Nodes: [out_67], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_28.run(buf297, buf298, buf299, buf300, buf301, buf302, 1024, 112, grid=grid(1024), stream=stream0)
        del buf297
        del buf298
        del buf299
        buf303 = reinterpret_tensor(buf290, (1, 512, 1, 1), (512, 1, 512, 512), 0); del buf290  # reuse
        buf304 = reinterpret_tensor(buf289, (1, 512, 1, 1), (512, 1, 512, 512), 0); del buf289  # reuse
        buf306 = reinterpret_tensor(buf288, (512, ), (1, ), 0); del buf288  # reuse
        # Source Nodes: [out_67], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_29.run(buf300, buf301, buf302, primals_231, primals_232, buf303, buf304, buf306, primals_231, primals_232, 512, 2, grid=grid(512), stream=stream0)
        del primals_231
        del primals_232
        buf307 = empty_strided((64, 512, 28, 28), (401408, 1, 14336, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [identity_9, out_67, out_68], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_31.run(buf296, buf303, buf304, primals_71, primals_72, buf271, buf307, 25690112, grid=grid(25690112), stream=stream0)
        del primals_72
        # Source Nodes: [out_70], Original ATen: [aten.convolution]
        buf308 = extern_kernels.convolution(buf307, primals_73, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf308, (64, 256, 28, 28), (200704, 1, 7168, 256))
        buf309 = empty_strided((1, 256, 1, 1, 392), (100352, 1, 100352, 100352, 256), device='cuda', dtype=torch.float32)
        buf310 = empty_strided((1, 256, 1, 1, 392), (100352, 1, 100352, 100352, 256), device='cuda', dtype=torch.float32)
        buf311 = empty_strided((1, 256, 1, 1, 392), (100352, 1, 100352, 100352, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_71], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_32.run(buf308, buf309, buf310, buf311, 100352, 128, grid=grid(100352), stream=stream0)
        buf312 = reinterpret_tensor(buf302, (1, 256, 1, 1, 4), (1024, 1, 1024, 1024, 256), 0); del buf302  # reuse
        buf313 = reinterpret_tensor(buf301, (1, 256, 1, 1, 4), (1024, 1, 1024, 1024, 256), 0); del buf301  # reuse
        buf314 = reinterpret_tensor(buf300, (1, 256, 1, 1, 4), (1024, 1, 1024, 1024, 256), 0); del buf300  # reuse
        # Source Nodes: [out_71], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_33.run(buf309, buf310, buf311, buf312, buf313, buf314, 1024, 98, grid=grid(1024), stream=stream0)
        buf315 = buf148; del buf148  # reuse
        buf316 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf318 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_71], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_34.run(buf312, buf313, buf314, primals_234, primals_235, buf315, buf316, buf318, primals_234, primals_235, 256, 4, grid=grid(256), stream=stream0)
        del primals_234
        del primals_235
        buf319 = empty_strided((64, 256, 28, 28), (200704, 1, 7168, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_71, out_72], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_35.run(buf308, buf315, buf316, primals_74, primals_75, buf319, 12845056, grid=grid(12845056), stream=stream0)
        del primals_75
        # Source Nodes: [out_73], Original ATen: [aten.convolution]
        buf320 = extern_kernels.convolution(buf319, buf8, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf320, (64, 256, 14, 14), (50176, 1, 3584, 256))
        buf321 = empty_strided((1, 256, 1, 1, 98), (25088, 1, 25088, 25088, 256), device='cuda', dtype=torch.float32)
        buf322 = empty_strided((1, 256, 1, 1, 98), (25088, 1, 25088, 25088, 256), device='cuda', dtype=torch.float32)
        buf323 = empty_strided((1, 256, 1, 1, 98), (25088, 1, 25088, 25088, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_74], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_36.run(buf320, buf321, buf322, buf323, 25088, 128, grid=grid(25088), stream=stream0)
        buf324 = buf316; del buf316  # reuse
        buf325 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf327 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_74], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_37.run(buf321, buf322, buf323, primals_237, primals_238, buf324, buf325, buf327, primals_237, primals_238, 256, 98, grid=grid(256), stream=stream0)
        del primals_237
        del primals_238
        buf328 = empty_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_74, out_75], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_38.run(buf320, buf324, buf325, primals_77, primals_78, buf328, 3211264, grid=grid(3211264), stream=stream0)
        del primals_78
        # Source Nodes: [out_76], Original ATen: [aten.convolution]
        buf329 = extern_kernels.convolution(buf328, primals_79, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf329, (64, 1024, 14, 14), (200704, 1, 14336, 1024))
        buf330 = reinterpret_tensor(buf311, (1, 1024, 1, 1, 98), (100352, 1, 100352, 100352, 1024), 0); del buf311  # reuse
        buf331 = reinterpret_tensor(buf310, (1, 1024, 1, 1, 98), (100352, 1, 100352, 100352, 1024), 0); del buf310  # reuse
        buf332 = reinterpret_tensor(buf309, (1, 1024, 1, 1, 98), (100352, 1, 100352, 100352, 1024), 0); del buf309  # reuse
        # Source Nodes: [out_77], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_39.run(buf329, buf330, buf331, buf332, 100352, 128, grid=grid(100352), stream=stream0)
        buf333 = reinterpret_tensor(buf314, (1, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf314  # reuse
        buf334 = reinterpret_tensor(buf313, (1, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf313  # reuse
        buf336 = reinterpret_tensor(buf312, (1024, ), (1, ), 0); del buf312  # reuse
        # Source Nodes: [out_77], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_40.run(buf330, buf331, buf332, primals_240, primals_241, buf333, buf334, buf336, primals_240, primals_241, 1024, 98, grid=grid(1024), stream=stream0)
        del primals_240
        del primals_241
        # Source Nodes: [getattr_l__self___layer3___0___downsample_0], Original ATen: [aten.convolution]
        buf337 = extern_kernels.convolution(buf307, primals_82, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf337, (64, 1024, 14, 14), (200704, 1, 14336, 1024))
        buf338 = buf332; del buf332  # reuse
        buf339 = buf331; del buf331  # reuse
        buf340 = buf330; del buf330  # reuse
        # Source Nodes: [identity_10], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_39.run(buf337, buf338, buf339, buf340, 100352, 128, grid=grid(100352), stream=stream0)
        buf341 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf342 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf344 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [identity_10], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_40.run(buf338, buf339, buf340, primals_243, primals_244, buf341, buf342, buf344, primals_243, primals_244, 1024, 98, grid=grid(1024), stream=stream0)
        del primals_243
        del primals_244
        buf345 = empty_strided((64, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda', dtype=torch.float32)
        buf346 = buf345; del buf345  # reuse
        # Source Nodes: [identity_10, identity_11, out_77, out_78], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_41.run(buf346, buf329, buf333, buf334, primals_80, primals_81, buf337, buf341, buf342, primals_83, primals_84, 12845056, grid=grid(12845056), stream=stream0)
        del primals_81
        del primals_84
        # Source Nodes: [out_80], Original ATen: [aten.convolution]
        buf347 = extern_kernels.convolution(buf346, primals_85, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf347, (64, 256, 14, 14), (50176, 1, 3584, 256))
        buf348 = buf323; del buf323  # reuse
        buf349 = buf322; del buf322  # reuse
        buf350 = buf321; del buf321  # reuse
        # Source Nodes: [out_81], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_36.run(buf347, buf348, buf349, buf350, 25088, 128, grid=grid(25088), stream=stream0)
        buf351 = buf325; del buf325  # reuse
        buf352 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf354 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_81], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_37.run(buf348, buf349, buf350, primals_246, primals_247, buf351, buf352, buf354, primals_246, primals_247, 256, 98, grid=grid(256), stream=stream0)
        del primals_246
        del primals_247
        buf355 = empty_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_81, out_82], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_38.run(buf347, buf351, buf352, primals_86, primals_87, buf355, 3211264, grid=grid(3211264), stream=stream0)
        del primals_87
        # Source Nodes: [out_83], Original ATen: [aten.convolution]
        buf356 = extern_kernels.convolution(buf355, buf9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf356, (64, 256, 14, 14), (50176, 1, 3584, 256))
        buf357 = buf350; del buf350  # reuse
        buf358 = buf349; del buf349  # reuse
        buf359 = buf348; del buf348  # reuse
        # Source Nodes: [out_84], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_36.run(buf356, buf357, buf358, buf359, 25088, 128, grid=grid(25088), stream=stream0)
        buf360 = buf352; del buf352  # reuse
        buf361 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf363 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_84], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_37.run(buf357, buf358, buf359, primals_249, primals_250, buf360, buf361, buf363, primals_249, primals_250, 256, 98, grid=grid(256), stream=stream0)
        del primals_249
        del primals_250
        buf364 = empty_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_84, out_85], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_38.run(buf356, buf360, buf361, primals_89, primals_90, buf364, 3211264, grid=grid(3211264), stream=stream0)
        del primals_90
        # Source Nodes: [out_86], Original ATen: [aten.convolution]
        buf365 = extern_kernels.convolution(buf364, primals_91, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf365, (64, 1024, 14, 14), (200704, 1, 14336, 1024))
        buf366 = buf340; del buf340  # reuse
        buf367 = buf339; del buf339  # reuse
        buf368 = buf338; del buf338  # reuse
        # Source Nodes: [out_87], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_39.run(buf365, buf366, buf367, buf368, 100352, 128, grid=grid(100352), stream=stream0)
        buf369 = buf342; del buf342  # reuse
        buf370 = buf334; del buf334  # reuse
        buf372 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_87], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_40.run(buf366, buf367, buf368, primals_252, primals_253, buf369, buf370, buf372, primals_252, primals_253, 1024, 98, grid=grid(1024), stream=stream0)
        del primals_252
        del primals_253
        buf373 = empty_strided((64, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [identity_12, out_87, out_88], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_42.run(buf365, buf369, buf370, primals_92, primals_93, buf346, buf373, 12845056, grid=grid(12845056), stream=stream0)
        del primals_93
        # Source Nodes: [out_90], Original ATen: [aten.convolution]
        buf374 = extern_kernels.convolution(buf373, primals_94, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf374, (64, 256, 14, 14), (50176, 1, 3584, 256))
        buf375 = buf359; del buf359  # reuse
        buf376 = buf358; del buf358  # reuse
        buf377 = buf357; del buf357  # reuse
        # Source Nodes: [out_91], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_36.run(buf374, buf375, buf376, buf377, 25088, 128, grid=grid(25088), stream=stream0)
        buf378 = buf361; del buf361  # reuse
        buf379 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf381 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_91], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_37.run(buf375, buf376, buf377, primals_255, primals_256, buf378, buf379, buf381, primals_255, primals_256, 256, 98, grid=grid(256), stream=stream0)
        del primals_255
        del primals_256
        buf382 = empty_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_91, out_92], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_38.run(buf374, buf378, buf379, primals_95, primals_96, buf382, 3211264, grid=grid(3211264), stream=stream0)
        del primals_96
        # Source Nodes: [out_93], Original ATen: [aten.convolution]
        buf383 = extern_kernels.convolution(buf382, buf10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf383, (64, 256, 14, 14), (50176, 1, 3584, 256))
        buf384 = buf377; del buf377  # reuse
        buf385 = buf376; del buf376  # reuse
        buf386 = buf375; del buf375  # reuse
        # Source Nodes: [out_94], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_36.run(buf383, buf384, buf385, buf386, 25088, 128, grid=grid(25088), stream=stream0)
        buf387 = buf379; del buf379  # reuse
        buf388 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf390 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_94], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_37.run(buf384, buf385, buf386, primals_258, primals_259, buf387, buf388, buf390, primals_258, primals_259, 256, 98, grid=grid(256), stream=stream0)
        del primals_258
        del primals_259
        buf391 = empty_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_94, out_95], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_38.run(buf383, buf387, buf388, primals_98, primals_99, buf391, 3211264, grid=grid(3211264), stream=stream0)
        del primals_99
        # Source Nodes: [out_96], Original ATen: [aten.convolution]
        buf392 = extern_kernels.convolution(buf391, primals_100, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf392, (64, 1024, 14, 14), (200704, 1, 14336, 1024))
        buf393 = buf368; del buf368  # reuse
        buf394 = buf367; del buf367  # reuse
        buf395 = buf366; del buf366  # reuse
        # Source Nodes: [out_97], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_39.run(buf392, buf393, buf394, buf395, 100352, 128, grid=grid(100352), stream=stream0)
        buf396 = buf370; del buf370  # reuse
        buf397 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf399 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_97], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_40.run(buf393, buf394, buf395, primals_261, primals_262, buf396, buf397, buf399, primals_261, primals_262, 1024, 98, grid=grid(1024), stream=stream0)
        del primals_261
        del primals_262
        buf400 = empty_strided((64, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [identity_13, out_97, out_98], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_42.run(buf392, buf396, buf397, primals_101, primals_102, buf373, buf400, 12845056, grid=grid(12845056), stream=stream0)
        del primals_102
        # Source Nodes: [out_100], Original ATen: [aten.convolution]
        buf401 = extern_kernels.convolution(buf400, primals_103, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf401, (64, 256, 14, 14), (50176, 1, 3584, 256))
        buf402 = buf386; del buf386  # reuse
        buf403 = buf385; del buf385  # reuse
        buf404 = buf384; del buf384  # reuse
        # Source Nodes: [out_101], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_36.run(buf401, buf402, buf403, buf404, 25088, 128, grid=grid(25088), stream=stream0)
        buf405 = buf388; del buf388  # reuse
        buf406 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf408 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_101], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_37.run(buf402, buf403, buf404, primals_264, primals_265, buf405, buf406, buf408, primals_264, primals_265, 256, 98, grid=grid(256), stream=stream0)
        del primals_264
        del primals_265
        buf409 = empty_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_101, out_102], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_38.run(buf401, buf405, buf406, primals_104, primals_105, buf409, 3211264, grid=grid(3211264), stream=stream0)
        del primals_105
        # Source Nodes: [out_103], Original ATen: [aten.convolution]
        buf410 = extern_kernels.convolution(buf409, buf11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf410, (64, 256, 14, 14), (50176, 1, 3584, 256))
        buf411 = buf404; del buf404  # reuse
        buf412 = buf403; del buf403  # reuse
        buf413 = buf402; del buf402  # reuse
        # Source Nodes: [out_104], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_36.run(buf410, buf411, buf412, buf413, 25088, 128, grid=grid(25088), stream=stream0)
        buf414 = buf406; del buf406  # reuse
        buf415 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf417 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_104], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_37.run(buf411, buf412, buf413, primals_267, primals_268, buf414, buf415, buf417, primals_267, primals_268, 256, 98, grid=grid(256), stream=stream0)
        del primals_267
        del primals_268
        buf418 = empty_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_104, out_105], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_38.run(buf410, buf414, buf415, primals_107, primals_108, buf418, 3211264, grid=grid(3211264), stream=stream0)
        del primals_108
        # Source Nodes: [out_106], Original ATen: [aten.convolution]
        buf419 = extern_kernels.convolution(buf418, primals_109, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf419, (64, 1024, 14, 14), (200704, 1, 14336, 1024))
        buf420 = buf395; del buf395  # reuse
        buf421 = buf394; del buf394  # reuse
        buf422 = buf393; del buf393  # reuse
        # Source Nodes: [out_107], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_39.run(buf419, buf420, buf421, buf422, 100352, 128, grid=grid(100352), stream=stream0)
        buf423 = buf397; del buf397  # reuse
        buf424 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf426 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_107], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_40.run(buf420, buf421, buf422, primals_270, primals_271, buf423, buf424, buf426, primals_270, primals_271, 1024, 98, grid=grid(1024), stream=stream0)
        del primals_270
        del primals_271
        buf427 = empty_strided((64, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [identity_14, out_107, out_108], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_42.run(buf419, buf423, buf424, primals_110, primals_111, buf400, buf427, 12845056, grid=grid(12845056), stream=stream0)
        del primals_111
        # Source Nodes: [out_110], Original ATen: [aten.convolution]
        buf428 = extern_kernels.convolution(buf427, primals_112, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf428, (64, 256, 14, 14), (50176, 1, 3584, 256))
        buf429 = buf413; del buf413  # reuse
        buf430 = buf412; del buf412  # reuse
        buf431 = buf411; del buf411  # reuse
        # Source Nodes: [out_111], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_36.run(buf428, buf429, buf430, buf431, 25088, 128, grid=grid(25088), stream=stream0)
        buf432 = buf415; del buf415  # reuse
        buf433 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf435 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_111], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_37.run(buf429, buf430, buf431, primals_273, primals_274, buf432, buf433, buf435, primals_273, primals_274, 256, 98, grid=grid(256), stream=stream0)
        del primals_273
        del primals_274
        buf436 = empty_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_111, out_112], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_38.run(buf428, buf432, buf433, primals_113, primals_114, buf436, 3211264, grid=grid(3211264), stream=stream0)
        del primals_114
        # Source Nodes: [out_113], Original ATen: [aten.convolution]
        buf437 = extern_kernels.convolution(buf436, buf12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf437, (64, 256, 14, 14), (50176, 1, 3584, 256))
        buf438 = buf431; del buf431  # reuse
        buf439 = buf430; del buf430  # reuse
        buf440 = buf429; del buf429  # reuse
        # Source Nodes: [out_114], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_36.run(buf437, buf438, buf439, buf440, 25088, 128, grid=grid(25088), stream=stream0)
        buf441 = buf433; del buf433  # reuse
        buf442 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf444 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_114], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_37.run(buf438, buf439, buf440, primals_276, primals_277, buf441, buf442, buf444, primals_276, primals_277, 256, 98, grid=grid(256), stream=stream0)
        del primals_276
        del primals_277
        buf445 = empty_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_114, out_115], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_38.run(buf437, buf441, buf442, primals_116, primals_117, buf445, 3211264, grid=grid(3211264), stream=stream0)
        del primals_117
        # Source Nodes: [out_116], Original ATen: [aten.convolution]
        buf446 = extern_kernels.convolution(buf445, primals_118, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf446, (64, 1024, 14, 14), (200704, 1, 14336, 1024))
        buf447 = buf422; del buf422  # reuse
        buf448 = buf421; del buf421  # reuse
        buf449 = buf420; del buf420  # reuse
        # Source Nodes: [out_117], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_39.run(buf446, buf447, buf448, buf449, 100352, 128, grid=grid(100352), stream=stream0)
        buf450 = buf424; del buf424  # reuse
        buf451 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf453 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_117], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_40.run(buf447, buf448, buf449, primals_279, primals_280, buf450, buf451, buf453, primals_279, primals_280, 1024, 98, grid=grid(1024), stream=stream0)
        del primals_279
        del primals_280
        buf454 = empty_strided((64, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [identity_15, out_117, out_118], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_42.run(buf446, buf450, buf451, primals_119, primals_120, buf427, buf454, 12845056, grid=grid(12845056), stream=stream0)
        del primals_120
        # Source Nodes: [out_120], Original ATen: [aten.convolution]
        buf455 = extern_kernels.convolution(buf454, primals_121, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf455, (64, 256, 14, 14), (50176, 1, 3584, 256))
        buf456 = buf440; del buf440  # reuse
        buf457 = buf439; del buf439  # reuse
        buf458 = buf438; del buf438  # reuse
        # Source Nodes: [out_121], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_36.run(buf455, buf456, buf457, buf458, 25088, 128, grid=grid(25088), stream=stream0)
        buf459 = buf442; del buf442  # reuse
        buf460 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf462 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_121], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_37.run(buf456, buf457, buf458, primals_282, primals_283, buf459, buf460, buf462, primals_282, primals_283, 256, 98, grid=grid(256), stream=stream0)
        del primals_282
        del primals_283
        buf463 = empty_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_121, out_122], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_38.run(buf455, buf459, buf460, primals_122, primals_123, buf463, 3211264, grid=grid(3211264), stream=stream0)
        del primals_123
        # Source Nodes: [out_123], Original ATen: [aten.convolution]
        buf464 = extern_kernels.convolution(buf463, buf13, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf464, (64, 256, 14, 14), (50176, 1, 3584, 256))
        buf465 = buf458; del buf458  # reuse
        buf466 = buf457; del buf457  # reuse
        buf467 = buf456; del buf456  # reuse
        # Source Nodes: [out_124], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_36.run(buf464, buf465, buf466, buf467, 25088, 128, grid=grid(25088), stream=stream0)
        buf468 = buf460; del buf460  # reuse
        buf469 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf471 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_124], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_37.run(buf465, buf466, buf467, primals_285, primals_286, buf468, buf469, buf471, primals_285, primals_286, 256, 98, grid=grid(256), stream=stream0)
        del buf465
        del buf466
        del buf467
        del primals_285
        del primals_286
        buf472 = empty_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_124, out_125], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_38.run(buf464, buf468, buf469, primals_125, primals_126, buf472, 3211264, grid=grid(3211264), stream=stream0)
        del buf469
        del primals_126
        # Source Nodes: [out_126], Original ATen: [aten.convolution]
        buf473 = extern_kernels.convolution(buf472, primals_127, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf473, (64, 1024, 14, 14), (200704, 1, 14336, 1024))
        buf474 = buf449; del buf449  # reuse
        buf475 = buf448; del buf448  # reuse
        buf476 = buf447; del buf447  # reuse
        # Source Nodes: [out_127], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_39.run(buf473, buf474, buf475, buf476, 100352, 128, grid=grid(100352), stream=stream0)
        buf477 = buf451; del buf451  # reuse
        buf478 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf480 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_127], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_40.run(buf474, buf475, buf476, primals_288, primals_289, buf477, buf478, buf480, primals_288, primals_289, 1024, 98, grid=grid(1024), stream=stream0)
        del buf474
        del buf475
        del buf476
        del primals_288
        del primals_289
        buf481 = empty_strided((64, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [identity_16, out_127, out_128], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_42.run(buf473, buf477, buf478, primals_128, primals_129, buf454, buf481, 12845056, grid=grid(12845056), stream=stream0)
        del buf478
        del primals_129
        # Source Nodes: [out_130], Original ATen: [aten.convolution]
        buf482 = extern_kernels.convolution(buf481, primals_130, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf482, (64, 512, 14, 14), (100352, 1, 7168, 512))
        buf483 = reinterpret_tensor(buf287, (1, 512, 1, 1, 98), (50176, 1, 50176, 50176, 512), 0); del buf287  # reuse
        buf484 = reinterpret_tensor(buf286, (1, 512, 1, 1, 98), (50176, 1, 50176, 50176, 512), 0); del buf286  # reuse
        buf485 = reinterpret_tensor(buf285, (1, 512, 1, 1, 98), (50176, 1, 50176, 50176, 512), 0); del buf285  # reuse
        # Source Nodes: [out_131], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_43.run(buf482, buf483, buf484, buf485, 50176, 128, grid=grid(50176), stream=stream0)
        buf486 = buf304; del buf304  # reuse
        buf487 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf489 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_131], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_44.run(buf483, buf484, buf485, primals_291, primals_292, buf486, buf487, buf489, primals_291, primals_292, 512, 98, grid=grid(512), stream=stream0)
        del buf483
        del buf484
        del buf485
        del primals_291
        del primals_292
        buf490 = empty_strided((64, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_131, out_132], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_45.run(buf482, buf486, buf487, primals_131, primals_132, buf490, 6422528, grid=grid(6422528), stream=stream0)
        del primals_132
        # Source Nodes: [out_133], Original ATen: [aten.convolution]
        buf491 = extern_kernels.convolution(buf490, buf14, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf491, (64, 512, 7, 7), (25088, 1, 3584, 512))
        buf492 = empty_strided((1, 512, 1, 1, 25), (12800, 1, 12800, 12800, 512), device='cuda', dtype=torch.float32)
        buf493 = empty_strided((1, 512, 1, 1, 25), (12800, 1, 12800, 12800, 512), device='cuda', dtype=torch.float32)
        buf494 = empty_strided((1, 512, 1, 1, 25), (12800, 1, 12800, 12800, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_134], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_46.run(buf491, buf492, buf493, buf494, 12800, 126, grid=grid(12800), stream=stream0)
        buf495 = buf487; del buf487  # reuse
        buf496 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf498 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_134], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_47.run(buf492, buf493, buf494, primals_294, primals_295, buf495, buf496, buf498, primals_294, primals_295, 512, 25, grid=grid(512), stream=stream0)
        del primals_294
        del primals_295
        buf499 = empty_strided((64, 512, 7, 7), (25088, 1, 3584, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_134, out_135], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_48.run(buf491, buf495, buf496, primals_134, primals_135, buf499, 1605632, grid=grid(1605632), stream=stream0)
        del primals_135
        # Source Nodes: [out_136], Original ATen: [aten.convolution]
        buf500 = extern_kernels.convolution(buf499, primals_136, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf500, (64, 2048, 7, 7), (100352, 1, 14336, 2048))
        buf501 = empty_strided((1, 2048, 1, 1, 25), (51200, 1, 51200, 51200, 2048), device='cuda', dtype=torch.float32)
        buf502 = empty_strided((1, 2048, 1, 1, 25), (51200, 1, 51200, 51200, 2048), device='cuda', dtype=torch.float32)
        buf503 = empty_strided((1, 2048, 1, 1, 25), (51200, 1, 51200, 51200, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_137], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_49.run(buf500, buf501, buf502, buf503, 51200, 126, grid=grid(51200), stream=stream0)
        buf504 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf505 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf507 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_137], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_50.run(buf501, buf502, buf503, primals_297, primals_298, buf504, buf505, buf507, primals_297, primals_298, 2048, 25, grid=grid(2048), stream=stream0)
        del primals_297
        del primals_298
        # Source Nodes: [getattr_l__self___layer4___0___downsample_0], Original ATen: [aten.convolution]
        buf508 = extern_kernels.convolution(buf481, primals_139, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf508, (64, 2048, 7, 7), (100352, 1, 14336, 2048))
        buf509 = buf503; del buf503  # reuse
        buf510 = buf502; del buf502  # reuse
        buf511 = buf501; del buf501  # reuse
        # Source Nodes: [identity_17], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_49.run(buf508, buf509, buf510, buf511, 51200, 126, grid=grid(51200), stream=stream0)
        buf512 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf513 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf515 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [identity_17], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_50.run(buf509, buf510, buf511, primals_300, primals_301, buf512, buf513, buf515, primals_300, primals_301, 2048, 25, grid=grid(2048), stream=stream0)
        del primals_300
        del primals_301
        buf516 = empty_strided((64, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda', dtype=torch.float32)
        buf517 = buf516; del buf516  # reuse
        # Source Nodes: [identity_17, identity_18, out_137, out_138], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_51.run(buf517, buf500, buf504, buf505, primals_137, primals_138, buf508, buf512, buf513, primals_140, primals_141, 6422528, grid=grid(6422528), stream=stream0)
        del primals_138
        del primals_141
        # Source Nodes: [out_140], Original ATen: [aten.convolution]
        buf518 = extern_kernels.convolution(buf517, primals_142, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf518, (64, 512, 7, 7), (25088, 1, 3584, 512))
        buf519 = buf494; del buf494  # reuse
        buf520 = buf493; del buf493  # reuse
        buf521 = buf492; del buf492  # reuse
        # Source Nodes: [out_141], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_46.run(buf518, buf519, buf520, buf521, 12800, 126, grid=grid(12800), stream=stream0)
        buf522 = buf496; del buf496  # reuse
        buf523 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf525 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_141], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_47.run(buf519, buf520, buf521, primals_303, primals_304, buf522, buf523, buf525, primals_303, primals_304, 512, 25, grid=grid(512), stream=stream0)
        del primals_303
        del primals_304
        buf526 = empty_strided((64, 512, 7, 7), (25088, 1, 3584, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_141, out_142], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_48.run(buf518, buf522, buf523, primals_143, primals_144, buf526, 1605632, grid=grid(1605632), stream=stream0)
        del primals_144
        # Source Nodes: [out_143], Original ATen: [aten.convolution]
        buf527 = extern_kernels.convolution(buf526, buf15, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf527, (64, 512, 7, 7), (25088, 1, 3584, 512))
        buf528 = buf521; del buf521  # reuse
        buf529 = buf520; del buf520  # reuse
        buf530 = buf519; del buf519  # reuse
        # Source Nodes: [out_144], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_46.run(buf527, buf528, buf529, buf530, 12800, 126, grid=grid(12800), stream=stream0)
        buf531 = buf523; del buf523  # reuse
        buf532 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf534 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_144], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_47.run(buf528, buf529, buf530, primals_306, primals_307, buf531, buf532, buf534, primals_306, primals_307, 512, 25, grid=grid(512), stream=stream0)
        del primals_306
        del primals_307
        buf535 = empty_strided((64, 512, 7, 7), (25088, 1, 3584, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_144, out_145], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_48.run(buf527, buf531, buf532, primals_146, primals_147, buf535, 1605632, grid=grid(1605632), stream=stream0)
        del primals_147
        # Source Nodes: [out_146], Original ATen: [aten.convolution]
        buf536 = extern_kernels.convolution(buf535, primals_148, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf536, (64, 2048, 7, 7), (100352, 1, 14336, 2048))
        buf537 = buf511; del buf511  # reuse
        buf538 = buf510; del buf510  # reuse
        buf539 = buf509; del buf509  # reuse
        # Source Nodes: [out_147], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_49.run(buf536, buf537, buf538, buf539, 51200, 126, grid=grid(51200), stream=stream0)
        buf540 = buf513; del buf513  # reuse
        buf541 = buf505; del buf505  # reuse
        buf543 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_147], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_50.run(buf537, buf538, buf539, primals_309, primals_310, buf540, buf541, buf543, primals_309, primals_310, 2048, 25, grid=grid(2048), stream=stream0)
        del primals_309
        del primals_310
        buf544 = empty_strided((64, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [identity_19, out_147, out_148], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_52.run(buf536, buf540, buf541, primals_149, primals_150, buf517, buf544, 6422528, grid=grid(6422528), stream=stream0)
        del primals_150
        # Source Nodes: [out_150], Original ATen: [aten.convolution]
        buf545 = extern_kernels.convolution(buf544, primals_151, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf545, (64, 512, 7, 7), (25088, 1, 3584, 512))
        buf546 = buf530; del buf530  # reuse
        buf547 = buf529; del buf529  # reuse
        buf548 = buf528; del buf528  # reuse
        # Source Nodes: [out_151], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_46.run(buf545, buf546, buf547, buf548, 12800, 126, grid=grid(12800), stream=stream0)
        buf549 = buf532; del buf532  # reuse
        buf550 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf552 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_151], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_47.run(buf546, buf547, buf548, primals_312, primals_313, buf549, buf550, buf552, primals_312, primals_313, 512, 25, grid=grid(512), stream=stream0)
        del primals_312
        del primals_313
        buf553 = empty_strided((64, 512, 7, 7), (25088, 1, 3584, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_151, out_152], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_48.run(buf545, buf549, buf550, primals_152, primals_153, buf553, 1605632, grid=grid(1605632), stream=stream0)
        del primals_153
        # Source Nodes: [out_153], Original ATen: [aten.convolution]
        buf554 = extern_kernels.convolution(buf553, buf16, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf554, (64, 512, 7, 7), (25088, 1, 3584, 512))
        buf555 = buf548; del buf548  # reuse
        buf556 = buf547; del buf547  # reuse
        buf557 = buf546; del buf546  # reuse
        # Source Nodes: [out_154], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_46.run(buf554, buf555, buf556, buf557, 12800, 126, grid=grid(12800), stream=stream0)
        buf558 = buf550; del buf550  # reuse
        buf559 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf561 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_154], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_47.run(buf555, buf556, buf557, primals_315, primals_316, buf558, buf559, buf561, primals_315, primals_316, 512, 25, grid=grid(512), stream=stream0)
        del buf555
        del buf556
        del buf557
        del primals_315
        del primals_316
        buf562 = empty_strided((64, 512, 7, 7), (25088, 1, 3584, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_154, out_155], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_48.run(buf554, buf558, buf559, primals_155, primals_156, buf562, 1605632, grid=grid(1605632), stream=stream0)
        del buf559
        del primals_156
        # Source Nodes: [out_156], Original ATen: [aten.convolution]
        buf563 = extern_kernels.convolution(buf562, primals_157, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf563, (64, 2048, 7, 7), (100352, 1, 14336, 2048))
        buf564 = buf539; del buf539  # reuse
        buf565 = buf538; del buf538  # reuse
        buf566 = buf537; del buf537  # reuse
        # Source Nodes: [out_157], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_49.run(buf563, buf564, buf565, buf566, 51200, 126, grid=grid(51200), stream=stream0)
        buf567 = buf541; del buf541  # reuse
        buf568 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf570 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_157], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_50.run(buf564, buf565, buf566, primals_318, primals_319, buf567, buf568, buf570, primals_318, primals_319, 2048, 25, grid=grid(2048), stream=stream0)
        del buf564
        del buf565
        del buf566
        del primals_318
        del primals_319
        buf571 = empty_strided((64, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda', dtype=torch.float32)
        buf575 = empty_strided((64, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_157, out_158, x_7], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_53.run(buf563, buf567, buf568, primals_158, primals_159, buf544, buf571, buf575, 6422528, grid=grid(6422528), stream=stream0)
        del buf568
        del primals_159
        buf572 = empty_strided((64, 2048, 1, 1), (2048, 1, 131072, 131072), device='cuda', dtype=torch.float32)
        buf573 = reinterpret_tensor(buf572, (64, 2048), (2048, 1), 0); del buf572  # reuse
        # Source Nodes: [x_8, x_9], Original ATen: [aten.mean, aten.view]
        triton_per_fused_mean_view_54.run(buf573, buf571, 131072, 49, grid=grid(131072), stream=stream0)
        del buf571
        buf574 = empty((64, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_10], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_161, buf573, reinterpret_tensor(primals_160, (2048, 1000), (1, 2048), 0), alpha=1, beta=1, out=buf574)
        del primals_161
        # Source Nodes: [x_1], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_164, primals_164, 1, grid=grid(1), stream=stream0)
        del primals_164
        # Source Nodes: [out_1], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_167, primals_167, 1, grid=grid(1), stream=stream0)
        del primals_167
        # Source Nodes: [out_4], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_170, primals_170, 1, grid=grid(1), stream=stream0)
        del primals_170
        # Source Nodes: [out_7], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_173, primals_173, 1, grid=grid(1), stream=stream0)
        del primals_173
        # Source Nodes: [identity_1], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_176, primals_176, 1, grid=grid(1), stream=stream0)
        del primals_176
        # Source Nodes: [out_11], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_179, primals_179, 1, grid=grid(1), stream=stream0)
        del primals_179
        # Source Nodes: [out_14], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_182, primals_182, 1, grid=grid(1), stream=stream0)
        del primals_182
        # Source Nodes: [out_17], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_185, primals_185, 1, grid=grid(1), stream=stream0)
        del primals_185
        # Source Nodes: [out_21], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_188, primals_188, 1, grid=grid(1), stream=stream0)
        del primals_188
        # Source Nodes: [out_24], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_191, primals_191, 1, grid=grid(1), stream=stream0)
        del primals_191
        # Source Nodes: [out_27], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_194, primals_194, 1, grid=grid(1), stream=stream0)
        del primals_194
        # Source Nodes: [out_31], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_197, primals_197, 1, grid=grid(1), stream=stream0)
        del primals_197
        # Source Nodes: [out_34], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_200, primals_200, 1, grid=grid(1), stream=stream0)
        del primals_200
        # Source Nodes: [out_37], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_203, primals_203, 1, grid=grid(1), stream=stream0)
        del primals_203
        # Source Nodes: [identity_5], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_206, primals_206, 1, grid=grid(1), stream=stream0)
        del primals_206
        # Source Nodes: [out_41], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_209, primals_209, 1, grid=grid(1), stream=stream0)
        del primals_209
        # Source Nodes: [out_44], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_212, primals_212, 1, grid=grid(1), stream=stream0)
        del primals_212
        # Source Nodes: [out_47], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_215, primals_215, 1, grid=grid(1), stream=stream0)
        del primals_215
        # Source Nodes: [out_51], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_218, primals_218, 1, grid=grid(1), stream=stream0)
        del primals_218
        # Source Nodes: [out_54], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_221, primals_221, 1, grid=grid(1), stream=stream0)
        del primals_221
        # Source Nodes: [out_57], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_224, primals_224, 1, grid=grid(1), stream=stream0)
        del primals_224
        # Source Nodes: [out_61], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_227, primals_227, 1, grid=grid(1), stream=stream0)
        del primals_227
        # Source Nodes: [out_64], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_230, primals_230, 1, grid=grid(1), stream=stream0)
        del primals_230
        # Source Nodes: [out_67], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_233, primals_233, 1, grid=grid(1), stream=stream0)
        del primals_233
        # Source Nodes: [out_71], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_236, primals_236, 1, grid=grid(1), stream=stream0)
        del primals_236
        # Source Nodes: [out_74], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_239, primals_239, 1, grid=grid(1), stream=stream0)
        del primals_239
        # Source Nodes: [out_77], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_242, primals_242, 1, grid=grid(1), stream=stream0)
        del primals_242
        # Source Nodes: [identity_10], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_245, primals_245, 1, grid=grid(1), stream=stream0)
        del primals_245
        # Source Nodes: [out_81], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_248, primals_248, 1, grid=grid(1), stream=stream0)
        del primals_248
        # Source Nodes: [out_84], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_251, primals_251, 1, grid=grid(1), stream=stream0)
        del primals_251
        # Source Nodes: [out_87], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_254, primals_254, 1, grid=grid(1), stream=stream0)
        del primals_254
        # Source Nodes: [out_91], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_257, primals_257, 1, grid=grid(1), stream=stream0)
        del primals_257
        # Source Nodes: [out_94], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_260, primals_260, 1, grid=grid(1), stream=stream0)
        del primals_260
        # Source Nodes: [out_97], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_263, primals_263, 1, grid=grid(1), stream=stream0)
        del primals_263
        # Source Nodes: [out_101], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_266, primals_266, 1, grid=grid(1), stream=stream0)
        del primals_266
        # Source Nodes: [out_104], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_269, primals_269, 1, grid=grid(1), stream=stream0)
        del primals_269
        # Source Nodes: [out_107], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_272, primals_272, 1, grid=grid(1), stream=stream0)
        del primals_272
        # Source Nodes: [out_111], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_275, primals_275, 1, grid=grid(1), stream=stream0)
        del primals_275
        # Source Nodes: [out_114], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_278, primals_278, 1, grid=grid(1), stream=stream0)
        del primals_278
        # Source Nodes: [out_117], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_281, primals_281, 1, grid=grid(1), stream=stream0)
        del primals_281
        # Source Nodes: [out_121], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_284, primals_284, 1, grid=grid(1), stream=stream0)
        del primals_284
        # Source Nodes: [out_124], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_287, primals_287, 1, grid=grid(1), stream=stream0)
        del primals_287
        # Source Nodes: [out_127], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_290, primals_290, 1, grid=grid(1), stream=stream0)
        del primals_290
        # Source Nodes: [out_131], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_293, primals_293, 1, grid=grid(1), stream=stream0)
        del primals_293
        # Source Nodes: [out_134], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_296, primals_296, 1, grid=grid(1), stream=stream0)
        del primals_296
        # Source Nodes: [out_137], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_299, primals_299, 1, grid=grid(1), stream=stream0)
        del primals_299
        # Source Nodes: [identity_17], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_302, primals_302, 1, grid=grid(1), stream=stream0)
        del primals_302
        # Source Nodes: [out_141], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_305, primals_305, 1, grid=grid(1), stream=stream0)
        del primals_305
        # Source Nodes: [out_144], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_308, primals_308, 1, grid=grid(1), stream=stream0)
        del primals_308
        # Source Nodes: [out_147], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_311, primals_311, 1, grid=grid(1), stream=stream0)
        del primals_311
        # Source Nodes: [out_151], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_314, primals_314, 1, grid=grid(1), stream=stream0)
        del primals_314
        # Source Nodes: [out_154], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_317, primals_317, 1, grid=grid(1), stream=stream0)
        del primals_317
        # Source Nodes: [out_157], Original ATen: [aten.add]
        triton_poi_fused_add_55.run(primals_320, primals_320, 1, grid=grid(1), stream=stream0)
        del primals_320
        return (buf574, buf0, primals_2, primals_4, primals_5, buf1, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, buf2, primals_20, primals_22, primals_23, primals_25, primals_26, buf3, primals_29, primals_31, primals_32, primals_34, primals_35, buf4, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, buf5, primals_50, primals_52, primals_53, primals_55, primals_56, buf6, primals_59, primals_61, primals_62, primals_64, primals_65, buf7, primals_68, primals_70, primals_71, primals_73, primals_74, buf8, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, buf9, primals_89, primals_91, primals_92, primals_94, primals_95, buf10, primals_98, primals_100, primals_101, primals_103, primals_104, buf11, primals_107, primals_109, primals_110, primals_112, primals_113, buf12, primals_116, primals_118, primals_119, primals_121, primals_122, buf13, primals_125, primals_127, primals_128, primals_130, primals_131, buf14, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, buf15, primals_146, primals_148, primals_149, primals_151, primals_152, buf16, primals_155, primals_157, primals_158, buf17, buf18, buf28, buf29, buf30, buf31, buf32, buf42, buf43, buf44, buf54, buf55, buf56, buf66, buf67, buf77, buf79, buf80, buf90, buf91, buf92, buf102, buf103, buf104, buf114, buf115, buf116, buf126, buf127, buf128, buf138, buf139, buf140, buf150, buf151, buf152, buf162, buf163, buf164, buf174, buf175, buf176, buf186, buf187, buf197, buf199, buf200, buf210, buf211, buf212, buf222, buf223, buf224, buf234, buf235, buf236, buf246, buf247, buf248, buf258, buf259, buf260, buf270, buf271, buf272, buf282, buf283, buf284, buf294, buf295, buf296, buf306, buf307, buf308, buf318, buf319, buf320, buf327, buf328, buf329, buf336, buf337, buf344, buf346, buf347, buf354, buf355, buf356, buf363, buf364, buf365, buf372, buf373, buf374, buf381, buf382, buf383, buf390, buf391, buf392, buf399, buf400, buf401, buf408, buf409, buf410, buf417, buf418, buf419, buf426, buf427, buf428, buf435, buf436, buf437, buf444, buf445, buf446, buf453, buf454, buf455, buf462, buf463, buf464, buf471, buf472, buf473, buf480, buf481, buf482, buf489, buf490, buf491, buf498, buf499, buf500, buf507, buf508, buf515, buf517, buf518, buf525, buf526, buf527, buf534, buf535, buf536, buf543, buf544, buf545, buf552, buf553, buf554, buf561, buf562, buf563, buf570, buf573, reinterpret_tensor(primals_160, (1000, 2048), (2048, 1), 0), buf575, reinterpret_tensor(buf567, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf558, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf549, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf540, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf531, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf522, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf512, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf504, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf495, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf486, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf477, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf468, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf459, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf450, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf441, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf432, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf423, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf414, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf405, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf396, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf387, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf378, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf369, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf360, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf351, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf341, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf333, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf324, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf315, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf303, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf291, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf279, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf267, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf255, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf243, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf231, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf219, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf207, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf194, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf183, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf171, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf159, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf147, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf135, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf123, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf111, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf99, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf87, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf74, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf63, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf51, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf39, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf25, (1, 64, 1, 1), (64, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_165 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_168 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_171 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_174 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_177 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_180 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_183 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_186 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_189 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_192 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_195 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_198 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_201 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_204 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_207 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_210 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_213 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_216 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_219 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_222 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_225 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_228 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_231 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_234 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_237 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_240 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_243 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_246 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_249 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_252 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_255 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_258 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_261 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_264 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_267 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_270 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_273 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_276 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_279 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_282 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_285 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_288 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_291 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_294 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_297 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_300 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_303 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_306 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_309 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_312 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_315 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_318 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_321 = rand_strided((64, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
