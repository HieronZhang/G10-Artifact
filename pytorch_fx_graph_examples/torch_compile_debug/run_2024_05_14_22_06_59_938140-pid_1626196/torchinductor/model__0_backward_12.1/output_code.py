
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


# kernel path: /tmp/torchinductor_zhang402/4w/c4wnckev4dajebc5dcoknbsnaw7x4fhe6qegkx2bx2ll5xbqqprf.py
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
    size_hints=[1024, 32],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_0', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1000
    rnumel = 32
    RBLOCK: tl.constexpr = 32
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


# kernel path: /tmp/torchinductor_zhang402/6y/c6y43etxwg5yxhp4cvug5asyb47wcu77tsbyzzkegjhcmjnnvtue.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_1', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 192
    x1 = (xindex // 192)
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (192*r2) + (24576*x1)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (1856 + x0 + (2048*(r2 // 64)) + (4096*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (1856 + x0 + (2048*(r2 // 64)) + (4096*x1)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp14 = tl.load(in_ptr3 + (x0 + (192*r2) + (24576*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = 2.0
        tmp5 = tmp3 * tmp4
        tmp6 = tmp1 * tmp5
        tmp7 = 64.0
        tmp8 = tmp6 / tmp7
        tmp9 = 0.0
        tmp10 = tl.where(tmp0, tmp9, tmp8)
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
        tmp16 = tmp14 - tmp15
        tmp17 = tmp10 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask & xmask, tmp20, _tmp19)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, xmask)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp19, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/dp/cdplt73xrnrvhqd4wu6mlsphnio577fd2oi2dt7z327ius6tul3f.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[256, 16],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_2', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (192*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/ax/caxal4ebusy5zwei7tganznqzgazazq5ugqrats3afu2y6amy4uf.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[256, 16],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_3', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (192*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/cx/ccxfi6ofemfianwmiinghayc26dpz2qn3rcz6vsidjsdr5cqtfpq.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_4', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 192
    x2 = (xindex // 12288)
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (1856 + x0 + (2048*x2)), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (1856 + x0 + (2048*x2)), None, eviction_policy='evict_last').to(tl.int1)
    tmp11 = tl.load(in_ptr3 + (x3), None)
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp1 * tmp5
    tmp7 = 64.0
    tmp8 = tmp6 / tmp7
    tmp9 = 0.0
    tmp10 = tl.where(tmp0, tmp9, tmp8)
    tmp13 = tmp11 - tmp12
    tmp15 = 0.00048828125
    tmp16 = tmp14 * tmp15
    tmp18 = tmp17 * tmp17
    tmp19 = tmp16 * tmp18
    tmp20 = tmp13 * tmp19
    tmp21 = tmp10 - tmp20
    tmp23 = tmp22 * tmp15
    tmp24 = tmp21 - tmp23
    tmp26 = tmp17 * tmp25
    tmp27 = tmp24 * tmp26
    tl.store(out_ptr0 + (x3), tmp27, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/f4/cf4w7cemy5aytxir3h5dsylunba3kvxp4g4mzyj4pzew252jhwxp.py
# Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]

triton_poi_fused_avg_pool2d_backward_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[65536, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_backward_5', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 65536
    xnumel = 64
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 8
    x3 = (xindex // 8)
    y0 = yindex % 2048
    y1 = (yindex // 2048)
    x5 = xindex
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (2048*(tl.minimum(tl.maximum(0, (-1) + x2), (-1) + (tl.minimum(8, 2 + x2))))) + (16384*(tl.minimum(tl.maximum(0, (-1) + x3), (-1) + (tl.minimum(8, 2 + x3))))) + (131072*y1)), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (y0 + (2048*(tl.minimum(1 + (tl.maximum(0, (-1) + x2)), (-1) + (tl.minimum(8, 2 + x2))))) + (16384*(tl.minimum(tl.maximum(0, (-1) + x3), (-1) + (tl.minimum(8, 2 + x3))))) + (131072*y1)), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + (y0 + (2048*(tl.minimum(2 + (tl.maximum(0, (-1) + x2)), (-1) + (tl.minimum(8, 2 + x2))))) + (16384*(tl.minimum(tl.maximum(0, (-1) + x3), (-1) + (tl.minimum(8, 2 + x3))))) + (131072*y1)), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (y0 + (2048*(tl.minimum(tl.maximum(0, (-1) + x2), (-1) + (tl.minimum(8, 2 + x2))))) + (16384*(tl.minimum(1 + (tl.maximum(0, (-1) + x3)), (-1) + (tl.minimum(8, 2 + x3))))) + (131072*y1)), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr0 + (y0 + (2048*(tl.minimum(1 + (tl.maximum(0, (-1) + x2)), (-1) + (tl.minimum(8, 2 + x2))))) + (16384*(tl.minimum(1 + (tl.maximum(0, (-1) + x3)), (-1) + (tl.minimum(8, 2 + x3))))) + (131072*y1)), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr0 + (y0 + (2048*(tl.minimum(2 + (tl.maximum(0, (-1) + x2)), (-1) + (tl.minimum(8, 2 + x2))))) + (16384*(tl.minimum(1 + (tl.maximum(0, (-1) + x3)), (-1) + (tl.minimum(8, 2 + x3))))) + (131072*y1)), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr0 + (y0 + (2048*(tl.minimum(tl.maximum(0, (-1) + x2), (-1) + (tl.minimum(8, 2 + x2))))) + (16384*(tl.minimum(2 + (tl.maximum(0, (-1) + x3)), (-1) + (tl.minimum(8, 2 + x3))))) + (131072*y1)), xmask, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr0 + (y0 + (2048*(tl.minimum(1 + (tl.maximum(0, (-1) + x2)), (-1) + (tl.minimum(8, 2 + x2))))) + (16384*(tl.minimum(2 + (tl.maximum(0, (-1) + x3)), (-1) + (tl.minimum(8, 2 + x3))))) + (131072*y1)), xmask, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr0 + (y0 + (2048*(tl.minimum(2 + (tl.maximum(0, (-1) + x2)), (-1) + (tl.minimum(8, 2 + x2))))) + (16384*(tl.minimum(2 + (tl.maximum(0, (-1) + x3)), (-1) + (tl.minimum(8, 2 + x3))))) + (131072*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 / 9
    tmp2 = tl.maximum(0, (-1) + x3)
    tmp3 = tl.minimum(8, 2 + x3)
    tmp4 = tmp2 < tmp3
    tmp5 = tl.maximum(0, (-1) + x2)
    tmp6 = tl.minimum(8, 2 + x2)
    tmp7 = tmp5 < tmp6
    tmp8 = tmp4 & tmp7
    tmp9 = 0.0
    tmp10 = tl.where(tmp8, tmp1, tmp9)
    tmp12 = tmp11 / 9
    tmp13 = 1 + (tl.maximum(0, (-1) + x2))
    tmp14 = tmp13 < tmp6
    tmp15 = tmp4 & tmp14
    tmp16 = tmp10 + tmp12
    tmp17 = tl.where(tmp15, tmp16, tmp10)
    tmp19 = tmp18 / 9
    tmp20 = 2 + (tl.maximum(0, (-1) + x2))
    tmp21 = tmp20 < tmp6
    tmp22 = tmp4 & tmp21
    tmp23 = tmp17 + tmp19
    tmp24 = tl.where(tmp22, tmp23, tmp17)
    tmp26 = tmp25 / 9
    tmp27 = 1 + (tl.maximum(0, (-1) + x3))
    tmp28 = tmp27 < tmp3
    tmp29 = tmp28 & tmp7
    tmp30 = tmp24 + tmp26
    tmp31 = tl.where(tmp29, tmp30, tmp24)
    tmp33 = tmp32 / 9
    tmp34 = tmp28 & tmp14
    tmp35 = tmp31 + tmp33
    tmp36 = tl.where(tmp34, tmp35, tmp31)
    tmp38 = tmp37 / 9
    tmp39 = tmp28 & tmp21
    tmp40 = tmp36 + tmp38
    tmp41 = tl.where(tmp39, tmp40, tmp36)
    tmp43 = tmp42 / 9
    tmp44 = 2 + (tl.maximum(0, (-1) + x3))
    tmp45 = tmp44 < tmp3
    tmp46 = tmp45 & tmp7
    tmp47 = tmp41 + tmp43
    tmp48 = tl.where(tmp46, tmp47, tmp41)
    tmp50 = tmp49 / 9
    tmp51 = tmp45 & tmp14
    tmp52 = tmp48 + tmp50
    tmp53 = tl.where(tmp51, tmp52, tmp48)
    tmp55 = tmp54 / 9
    tmp56 = tmp45 & tmp21
    tmp57 = tmp53 + tmp55
    tmp58 = tl.where(tmp56, tmp57, tmp53)
    tl.store(out_ptr0 + (x5 + (64*y4)), tmp58, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/22/c22svus3s6maqob43hlyx7wqcyfiwwjj4hg6xqlpjwjxosgekh5t.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_6', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 384
    x1 = (xindex // 384)
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp15 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (384*r2) + (49152*x1)), rmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (1472 + x0 + (2048*(r2 // 64)) + (4096*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (1472 + x0 + (2048*(r2 // 64)) + (4096*x1)), rmask, eviction_policy='evict_first').to(tl.int1)
        tmp14 = tl.load(in_ptr3 + (x0 + (384*r2) + (49152*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = 2.0
        tmp5 = tmp3 * tmp4
        tmp6 = tmp1 * tmp5
        tmp7 = 64.0
        tmp8 = tmp6 / tmp7
        tmp9 = 0.0
        tmp10 = tl.where(tmp0, tmp9, tmp8)
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
        tmp16 = tmp14 - tmp15
        tmp17 = tmp10 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask, tmp20, _tmp19)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, None)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/qh/cqhx75jy4zguqrhaanvx4p3xm2gpk2rfuerkgf6u7fcl234luvfi.py
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
    size_hints=[512, 16],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_7', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (384*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/df/cdfgqbyztplkymiwxn6s3inq2h46tor6xh4svqnrgzkcbjxyo5xw.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[512, 16],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_8', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (384*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/qm/cqmc6opn7ilanvtc4pjdkctiqzlbhjfqmdnzegg54hqmexcbfhyh.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_9', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 384
    x2 = (xindex // 24576)
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (1472 + x0 + (2048*x2)), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (1472 + x0 + (2048*x2)), None, eviction_policy='evict_last').to(tl.int1)
    tmp11 = tl.load(in_ptr3 + (x3), None)
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp1 * tmp5
    tmp7 = 64.0
    tmp8 = tmp6 / tmp7
    tmp9 = 0.0
    tmp10 = tl.where(tmp0, tmp9, tmp8)
    tmp13 = tmp11 - tmp12
    tmp15 = 0.00048828125
    tmp16 = tmp14 * tmp15
    tmp18 = tmp17 * tmp17
    tmp19 = tmp16 * tmp18
    tmp20 = tmp13 * tmp19
    tmp21 = tmp10 - tmp20
    tmp23 = tmp22 * tmp15
    tmp24 = tmp21 - tmp23
    tmp26 = tmp17 * tmp25
    tmp27 = tmp24 * tmp26
    tl.store(out_ptr0 + (x3), tmp27, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/5u/c5u47t2riw4tyhtwepqj5iue3trt62lnrifzxdxtjsplqeaxtxqh.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_10', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 384
    x1 = (xindex // 384)
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp15 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (384*r2) + (49152*x1)), rmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (1088 + x0 + (2048*(r2 // 64)) + (4096*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (1088 + x0 + (2048*(r2 // 64)) + (4096*x1)), rmask, eviction_policy='evict_first').to(tl.int1)
        tmp14 = tl.load(in_ptr3 + (x0 + (384*r2) + (49152*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = 2.0
        tmp5 = tmp3 * tmp4
        tmp6 = tmp1 * tmp5
        tmp7 = 64.0
        tmp8 = tmp6 / tmp7
        tmp9 = 0.0
        tmp10 = tl.where(tmp0, tmp9, tmp8)
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
        tmp16 = tmp14 - tmp15
        tmp17 = tmp10 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask, tmp20, _tmp19)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, None)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/j2/cj2mrwmh6dcrkv7ze2vsybtmmbeqiclj5jd3txdz2i4n2btauudr.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_11', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 384
    x2 = (xindex // 24576)
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (1088 + x0 + (2048*x2)), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (1088 + x0 + (2048*x2)), None, eviction_policy='evict_last').to(tl.int1)
    tmp11 = tl.load(in_ptr3 + (x3), None)
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp1 * tmp5
    tmp7 = 64.0
    tmp8 = tmp6 / tmp7
    tmp9 = 0.0
    tmp10 = tl.where(tmp0, tmp9, tmp8)
    tmp13 = tmp11 - tmp12
    tmp15 = 0.00048828125
    tmp16 = tmp14 * tmp15
    tmp18 = tmp17 * tmp17
    tmp19 = tmp16 * tmp18
    tmp20 = tmp13 * tmp19
    tmp21 = tmp10 - tmp20
    tmp23 = tmp22 * tmp15
    tmp24 = tmp21 - tmp23
    tmp26 = tmp17 * tmp25
    tmp27 = tmp24 * tmp26
    tl.store(out_ptr0 + (x3), tmp27, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/2i/c2ibnav3wawbtnntprzl3zuhyjjtaovgqrowtvomqv4kpby3fyl2.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_12', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 384
    x1 = (xindex // 384)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp11 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (384*r2) + (49152*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (x0 + (384*r2) + (49152*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (384*r2) + (49152*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + (x0 + (384*r2) + (49152*x1)), rmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_zhang402/ok/cokvdgebaee5oo6di3dm6rlalsurjxnd2h4zjqxf3xxa3svh2gkt.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_13', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 384
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp4 = tl.load(in_ptr1 + (x2), None)
    tmp7 = tl.load(in_ptr2 + (x2), None)
    tmp8 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 0.00048828125
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(in_out_ptr0 + (x2), tmp23, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/if/cifpnaboh4gvpo3gy5yhmvyc7nuql2ymthxk5jle362e7uibrr2r.py
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
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_14', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7168
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 448
    x1 = (xindex // 448)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (448*r2) + (57344*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (x0 + (448*r2) + (57344*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (x0 + (448*r2) + (57344*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_zhang402/hn/chns3teqa437vzawm6xi6udxqpamhw3cyzi6lcliisafq2ffvwui.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[512, 16],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_15', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 448
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (448*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/5l/c5likh6kyupi5sporjf4ugx5x7hwb6f575c4rk5ladl7dugwwueu.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[512, 16],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_16', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 448
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (448*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/mp/cmp6fvypxc6x27umpjpwtmcz56hrosxxeczcnh44uukipppej47e.py
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
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_17', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 917504
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 448
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
    tmp9 = 0.00048828125
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


# kernel path: /tmp/torchinductor_zhang402/zb/czbaia5nnkh6elt2wozeqllzs3gb3zz6lnxkz52ffk22kmibavla.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_18', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 384
    x1 = (xindex // 384)
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp15 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (384*r2) + (49152*x1)), rmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (704 + x0 + (2048*(r2 // 64)) + (4096*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (704 + x0 + (2048*(r2 // 64)) + (4096*x1)), rmask, eviction_policy='evict_first').to(tl.int1)
        tmp14 = tl.load(in_ptr3 + (x0 + (384*r2) + (49152*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = 2.0
        tmp5 = tmp3 * tmp4
        tmp6 = tmp1 * tmp5
        tmp7 = 64.0
        tmp8 = tmp6 / tmp7
        tmp9 = 0.0
        tmp10 = tl.where(tmp0, tmp9, tmp8)
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
        tmp16 = tmp14 - tmp15
        tmp17 = tmp10 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask, tmp20, _tmp19)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, None)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/4v/c4vhs22i2htvvkwjobjteplhkcpj3lxj3o6ekje24k2mk5f3ykox.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_19', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 384
    x2 = (xindex // 24576)
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (704 + x0 + (2048*x2)), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (704 + x0 + (2048*x2)), None, eviction_policy='evict_last').to(tl.int1)
    tmp11 = tl.load(in_ptr3 + (x3), None)
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp1 * tmp5
    tmp7 = 64.0
    tmp8 = tmp6 / tmp7
    tmp9 = 0.0
    tmp10 = tl.where(tmp0, tmp9, tmp8)
    tmp13 = tmp11 - tmp12
    tmp15 = 0.00048828125
    tmp16 = tmp14 * tmp15
    tmp18 = tmp17 * tmp17
    tmp19 = tmp16 * tmp18
    tmp20 = tmp13 * tmp19
    tmp21 = tmp10 - tmp20
    tmp23 = tmp22 * tmp15
    tmp24 = tmp21 - tmp23
    tmp26 = tmp17 * tmp25
    tmp27 = tmp24 * tmp26
    tl.store(out_ptr0 + (x3), tmp27, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/rk/crkhgfkz6v37jlrip7lxejn5cwyqu7vnov74ehwfjrkz6eqerg5i.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_20', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 384
    x1 = (xindex // 384)
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp15 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (384*r2) + (49152*x1)), rmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (320 + x0 + (2048*(r2 // 64)) + (4096*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (320 + x0 + (2048*(r2 // 64)) + (4096*x1)), rmask, eviction_policy='evict_first').to(tl.int1)
        tmp14 = tl.load(in_ptr3 + (x0 + (384*r2) + (49152*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = 2.0
        tmp5 = tmp3 * tmp4
        tmp6 = tmp1 * tmp5
        tmp7 = 64.0
        tmp8 = tmp6 / tmp7
        tmp9 = 0.0
        tmp10 = tl.where(tmp0, tmp9, tmp8)
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
        tmp16 = tmp14 - tmp15
        tmp17 = tmp10 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask, tmp20, _tmp19)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, None)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/wr/cwr4z23oub6cef3feefawhn77iuiumastcuthnf7laeyik76csoq.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_21', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 384
    x2 = (xindex // 24576)
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (320 + x0 + (2048*x2)), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (320 + x0 + (2048*x2)), None, eviction_policy='evict_last').to(tl.int1)
    tmp11 = tl.load(in_ptr3 + (x3), None)
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp1 * tmp5
    tmp7 = 64.0
    tmp8 = tmp6 / tmp7
    tmp9 = 0.0
    tmp10 = tl.where(tmp0, tmp9, tmp8)
    tmp13 = tmp11 - tmp12
    tmp15 = 0.00048828125
    tmp16 = tmp14 * tmp15
    tmp18 = tmp17 * tmp17
    tmp19 = tmp16 * tmp18
    tmp20 = tmp13 * tmp19
    tmp21 = tmp10 - tmp20
    tmp23 = tmp22 * tmp15
    tmp24 = tmp21 - tmp23
    tmp26 = tmp17 * tmp25
    tmp27 = tmp24 * tmp26
    tl.store(out_ptr0 + (x3), tmp27, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/l2/cl23gj6wlqdcnintuvq5fgdxtn5bxqnu6lthy6s7mwafkyuxhpwu.py
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
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_22', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5120
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 320
    x1 = (xindex // 320)
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (320*r2) + (40960*x1)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (x0 + (2048*(r2 // 64)) + (4096*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x0 + (2048*(r2 // 64)) + (4096*x1)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp14 = tl.load(in_ptr3 + (x0 + (320*r2) + (40960*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = 2.0
        tmp5 = tmp3 * tmp4
        tmp6 = tmp1 * tmp5
        tmp7 = 64.0
        tmp8 = tmp6 / tmp7
        tmp9 = 0.0
        tmp10 = tl.where(tmp0, tmp9, tmp8)
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
        tmp16 = tmp14 - tmp15
        tmp17 = tmp10 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask & xmask, tmp20, _tmp19)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, xmask)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp19, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/y6/cy6mqmbso6e24sre746e5bgjday2ziq7iclqer3ta5vhwgtbeeiv.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[512, 16],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_23', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 320
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (320*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/g5/cg5hcmtdzznjlpiaalw4gh7zd4uzu2nyoddwldibgkd6xh4m4l25.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[512, 16],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_24', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 320
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (320*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/gj/cgjmdqkzezqz6kmp72xl65jgwtioiwfdhmwcj2xwiqeyma5w3cnd.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_25', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 655360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 320
    x2 = (xindex // 20480)
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x0 + (2048*x2)), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x0 + (2048*x2)), None, eviction_policy='evict_last').to(tl.int1)
    tmp11 = tl.load(in_ptr3 + (x3), None)
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp1 * tmp5
    tmp7 = 64.0
    tmp8 = tmp6 / tmp7
    tmp9 = 0.0
    tmp10 = tl.where(tmp0, tmp9, tmp8)
    tmp13 = tmp11 - tmp12
    tmp15 = 0.00048828125
    tmp16 = tmp14 * tmp15
    tmp18 = tmp17 * tmp17
    tmp19 = tmp16 * tmp18
    tmp20 = tmp13 * tmp19
    tmp21 = tmp10 - tmp20
    tmp23 = tmp22 * tmp15
    tmp24 = tmp21 - tmp23
    tmp26 = tmp17 * tmp25
    tmp27 = tmp24 * tmp26
    tl.store(out_ptr0 + (x3), tmp27, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/rg/crg6edg7yimipnoipefgfmsonqmkcazeu5czxa5hrwo5xbkuz4wu.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_26', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 192
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2 + (192*y3)), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (118784 + y0 + (64*x2) + (131072*y1)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (1856 + x2 + (2048*y3)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (1856 + x2 + (2048*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (1856 + x2 + (2048*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp0, tmp8, tmp7)
    tl.store(out_ptr0 + (x2 + (192*y3)), tmp9, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/tz/ctzgmgix36ctnqauksxythqrcbat6wthmne5xhdu3pxk3mqdd35a.py
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
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_27', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 192
    x1 = (xindex // 192)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (192*r2) + (24576*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (192*r2) + (24576*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp0 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp9, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/mw/cmw44xjrtgqllnan2p36mhwaousazdx7uybmb6m5n46ajosyampj.py
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
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_28', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 192
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x2), None)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.00048828125
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/hk/chknnsp5esuhcvb3nbcvh3q63zh7sdyuvgl2lb45ddiewtzqkw4l.py
# Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]

triton_poi_fused_avg_pool2d_backward_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[65536, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_backward_29', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 40960
    xnumel = 64
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 8
    x3 = (xindex // 8)
    y0 = yindex % 1280
    y1 = (yindex // 1280)
    x5 = xindex
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (1280*(tl.minimum(tl.maximum(0, (-1) + x2), (-1) + (tl.minimum(8, 2 + x2))))) + (10240*(tl.minimum(tl.maximum(0, (-1) + x3), (-1) + (tl.minimum(8, 2 + x3))))) + (81920*y1)), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (y0 + (1280*(tl.minimum(1 + (tl.maximum(0, (-1) + x2)), (-1) + (tl.minimum(8, 2 + x2))))) + (10240*(tl.minimum(tl.maximum(0, (-1) + x3), (-1) + (tl.minimum(8, 2 + x3))))) + (81920*y1)), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + (y0 + (1280*(tl.minimum(2 + (tl.maximum(0, (-1) + x2)), (-1) + (tl.minimum(8, 2 + x2))))) + (10240*(tl.minimum(tl.maximum(0, (-1) + x3), (-1) + (tl.minimum(8, 2 + x3))))) + (81920*y1)), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (y0 + (1280*(tl.minimum(tl.maximum(0, (-1) + x2), (-1) + (tl.minimum(8, 2 + x2))))) + (10240*(tl.minimum(1 + (tl.maximum(0, (-1) + x3)), (-1) + (tl.minimum(8, 2 + x3))))) + (81920*y1)), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr0 + (y0 + (1280*(tl.minimum(1 + (tl.maximum(0, (-1) + x2)), (-1) + (tl.minimum(8, 2 + x2))))) + (10240*(tl.minimum(1 + (tl.maximum(0, (-1) + x3)), (-1) + (tl.minimum(8, 2 + x3))))) + (81920*y1)), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr0 + (y0 + (1280*(tl.minimum(2 + (tl.maximum(0, (-1) + x2)), (-1) + (tl.minimum(8, 2 + x2))))) + (10240*(tl.minimum(1 + (tl.maximum(0, (-1) + x3)), (-1) + (tl.minimum(8, 2 + x3))))) + (81920*y1)), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr0 + (y0 + (1280*(tl.minimum(tl.maximum(0, (-1) + x2), (-1) + (tl.minimum(8, 2 + x2))))) + (10240*(tl.minimum(2 + (tl.maximum(0, (-1) + x3)), (-1) + (tl.minimum(8, 2 + x3))))) + (81920*y1)), xmask, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr0 + (y0 + (1280*(tl.minimum(1 + (tl.maximum(0, (-1) + x2)), (-1) + (tl.minimum(8, 2 + x2))))) + (10240*(tl.minimum(2 + (tl.maximum(0, (-1) + x3)), (-1) + (tl.minimum(8, 2 + x3))))) + (81920*y1)), xmask, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr0 + (y0 + (1280*(tl.minimum(2 + (tl.maximum(0, (-1) + x2)), (-1) + (tl.minimum(8, 2 + x2))))) + (10240*(tl.minimum(2 + (tl.maximum(0, (-1) + x3)), (-1) + (tl.minimum(8, 2 + x3))))) + (81920*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 / 9
    tmp2 = tl.maximum(0, (-1) + x3)
    tmp3 = tl.minimum(8, 2 + x3)
    tmp4 = tmp2 < tmp3
    tmp5 = tl.maximum(0, (-1) + x2)
    tmp6 = tl.minimum(8, 2 + x2)
    tmp7 = tmp5 < tmp6
    tmp8 = tmp4 & tmp7
    tmp9 = 0.0
    tmp10 = tl.where(tmp8, tmp1, tmp9)
    tmp12 = tmp11 / 9
    tmp13 = 1 + (tl.maximum(0, (-1) + x2))
    tmp14 = tmp13 < tmp6
    tmp15 = tmp4 & tmp14
    tmp16 = tmp10 + tmp12
    tmp17 = tl.where(tmp15, tmp16, tmp10)
    tmp19 = tmp18 / 9
    tmp20 = 2 + (tl.maximum(0, (-1) + x2))
    tmp21 = tmp20 < tmp6
    tmp22 = tmp4 & tmp21
    tmp23 = tmp17 + tmp19
    tmp24 = tl.where(tmp22, tmp23, tmp17)
    tmp26 = tmp25 / 9
    tmp27 = 1 + (tl.maximum(0, (-1) + x3))
    tmp28 = tmp27 < tmp3
    tmp29 = tmp28 & tmp7
    tmp30 = tmp24 + tmp26
    tmp31 = tl.where(tmp29, tmp30, tmp24)
    tmp33 = tmp32 / 9
    tmp34 = tmp28 & tmp14
    tmp35 = tmp31 + tmp33
    tmp36 = tl.where(tmp34, tmp35, tmp31)
    tmp38 = tmp37 / 9
    tmp39 = tmp28 & tmp21
    tmp40 = tmp36 + tmp38
    tmp41 = tl.where(tmp39, tmp40, tmp36)
    tmp43 = tmp42 / 9
    tmp44 = 2 + (tl.maximum(0, (-1) + x3))
    tmp45 = tmp44 < tmp3
    tmp46 = tmp45 & tmp7
    tmp47 = tmp41 + tmp43
    tmp48 = tl.where(tmp46, tmp47, tmp41)
    tmp50 = tmp49 / 9
    tmp51 = tmp45 & tmp14
    tmp52 = tmp48 + tmp50
    tmp53 = tl.where(tmp51, tmp52, tmp48)
    tmp55 = tmp54 / 9
    tmp56 = tmp45 & tmp21
    tmp57 = tmp53 + tmp55
    tmp58 = tl.where(tmp56, tmp57, tmp53)
    tl.store(out_ptr0 + (x5 + (64*y4)), tmp58, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/u3/cu3i5jgcrqzdoqexdvtvc2lzjzyxpolzjufu45yi5d5tkibhoyjd.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[2048, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_30', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 384
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2 + (384*y3)), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (94208 + y0 + (64*x2) + (131072*y1)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (1472 + x2 + (2048*y3)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (1472 + x2 + (2048*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (1472 + x2 + (2048*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp0, tmp8, tmp7)
    tl.store(out_ptr0 + (x2 + (384*y3)), tmp9, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/wg/cwgppnskhvlcior34e6j663yd7ueyifr63224fdplu3qrhua3yh6.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_31 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_31', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 384
    x1 = (xindex // 384)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (384*r2) + (49152*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (384*r2) + (49152*x1)), rmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_zhang402/ti/ctizr7jdnzhjhduhunsxpnzhyc4f45csbdpkzbvqbsvl2k6doazc.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_32', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 384
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x2), None)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.00048828125
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/m5/cm5c3gayjgbpbufylplddxfknzp7mmvc2gvlmuzmu4n2vzaojd52.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_33 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[2048, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_33', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 384
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2 + (384*y3)), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (69632 + y0 + (64*x2) + (131072*y1)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (1088 + x2 + (2048*y3)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (1088 + x2 + (2048*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (1088 + x2 + (2048*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp0, tmp8, tmp7)
    tl.store(out_ptr0 + (x2 + (384*y3)), tmp9, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/3d/c3d5fmb2fwpz5n4uuml2ty3h4vt7zsyv7n4rdeccqavs74l4lan2.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[2048, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_34', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 384
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2 + (384*y3)), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (45056 + y0 + (64*x2) + (131072*y1)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (704 + x2 + (2048*y3)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (704 + x2 + (2048*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (704 + x2 + (2048*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp0, tmp8, tmp7)
    tl.store(out_ptr0 + (x2 + (384*y3)), tmp9, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/vw/cvwitsafh7g7dfo3zq3r6dcqs7pgo6qwfg6eqaqpvtm5kn4vuocq.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_35 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[2048, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_35', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 384
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2 + (384*y3)), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (20480 + y0 + (64*x2) + (131072*y1)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (320 + x2 + (2048*y3)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (320 + x2 + (2048*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (320 + x2 + (2048*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp0, tmp8, tmp7)
    tl.store(out_ptr0 + (x2 + (384*y3)), tmp9, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/xk/cxkfk2qaiegjtjpfkj7luj265gvhuflxjw7y2yvapme7v53ldric.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_36 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[2048, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_36', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 320
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2 + (320*y3)), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (64*x2) + (131072*y1)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2 + (2048*y3)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2 + (2048*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (x2 + (2048*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp0, tmp8, tmp7)
    tl.store(out_ptr0 + (x2 + (320*y3)), tmp9, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/sr/csr2pf4rcx75dlpu7pewkrjyplqsqotl2mdit7ho2aqb5exudy5c.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_37 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_37', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5120
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 320
    x1 = (xindex // 320)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (320*r2) + (40960*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (320*r2) + (40960*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp0 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp9, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/ng/cngzydjjubnskpaf7quwsuksehqzwsoubd6uavomtv5nxaxg6agi.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_38', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 655360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 320
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x2), None)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.00048828125
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/t5/ct5cdcbzvvvvrcuxdd3wrxhnquhyqk2mvki47rdh65ncynmc73hi.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_39 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[2048, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_39', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 1280
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    y3 = yindex
    tmp0 = tl.load(in_out_ptr0 + (y0 + (64*x2) + (81920*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x2 + (1280*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (1280*y3)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (1280*y3)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (y0 + (64*x2) + (81920*y1)), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/ik/cikrjsexu2wyhfanak7m5uu4nftcbp6b74de4u3ug2otsvn6kjbo.py
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
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_40', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 192
    x1 = (xindex // 192)
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (192*r2) + (24576*x1)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (20480 + (64*x0) + (81920*(r2 // 64)) + (163840*x1) + (r2 % 64)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr2 + (x0 + (192*r2) + (24576*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = 0.0
        tmp3 = tl.where(tmp0, tmp2, tmp1)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp3 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, xmask)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/xz/cxzxyh6ktgimzyfy7mm47hdg6ogcy245sl5auonukadknuyr5lte.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_41', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 192
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2 + (192*y3)), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (20480 + y0 + (64*x2) + (81920*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (192*y3)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp6 = tmp4 - tmp5
    tmp8 = 0.00048828125
    tmp9 = tmp7 * tmp8
    tmp11 = tmp10 * tmp10
    tmp12 = tmp9 * tmp11
    tmp13 = tmp6 * tmp12
    tmp14 = tmp3 - tmp13
    tmp16 = tmp15 * tmp8
    tmp17 = tmp14 - tmp16
    tmp19 = tmp10 * tmp18
    tmp20 = tmp17 * tmp19
    tl.store(out_ptr0 + (x2 + (192*y3)), tmp20, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/u5/cu5ogwlvden6azgyaiox54btr6imvrcxo5ratnyjii4zzdy4wunk.py
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_42', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 14016
    rnumel = 127
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 192)
    x0 = xindex % 192
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (127*x1)
        tmp1 = tl.full([1, 1], 9248, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (192*((r2 + (127*x1)) % 9248))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + (x0 + (192*((r2 + (127*x1)) % 9248))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
        tmp13 = tl.load(in_ptr2 + (x0 + (192*((r2 + (127*x1)) % 9248))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_zhang402/hv/chvf3l4b4ogeigrfksylxrwgvkjvmfzld5tiiuv3upabysg6hpxk.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_43 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_43', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 73
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
        tmp0 = tl.load(in_ptr0 + (x0 + (192*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/5f/c5frgyn2fjrlvdwjce7gwz2bzmfpgjvsenrwu6zpxyq7cqddhzv7.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_44', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 73
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
        tmp0 = tl.load(in_ptr0 + (x0 + (192*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/j7/cj7jxroxb66cwbhjkcbcgbbdrfbndewh3zrprfprwoxvjmibll2w.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_45 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_45', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1775616
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 192
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
    tmp9 = 0.00010813148788927336
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


# kernel path: /tmp/torchinductor_zhang402/tx/ctxzhhag3knck5mkgc7m65ykmhmqyd6q2moji65ew6mxkcf4y2ey.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_46 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_46', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5120
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 320
    x1 = (xindex // 320)
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (320*r2) + (40960*x1)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((64*x0) + (81920*(r2 // 64)) + (163840*x1) + (r2 % 64)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr2 + (x0 + (320*r2) + (40960*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = 0.0
        tmp3 = tl.where(tmp0, tmp2, tmp1)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp3 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, xmask)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/7r/c7rdsyvnefy223xw2m7a6j2v2cbqx6xs35ltoephvxyaga4wqsdc.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_47 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[2048, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_47', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 320
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2 + (320*y3)), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (64*x2) + (81920*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (320*y3)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp6 = tmp4 - tmp5
    tmp8 = 0.00048828125
    tmp9 = tmp7 * tmp8
    tmp11 = tmp10 * tmp10
    tmp12 = tmp9 * tmp11
    tmp13 = tmp6 * tmp12
    tmp14 = tmp3 - tmp13
    tmp16 = tmp15 * tmp8
    tmp17 = tmp14 - tmp16
    tmp19 = tmp10 * tmp18
    tmp20 = tmp17 * tmp19
    tl.store(out_ptr0 + (x2 + (320*y3)), tmp20, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/7a/c7azriqetos2hwcltdaupyvk27e46lxyevrmnt652gilfyl7yi3d.py
# Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_div_native_batch_norm_backward_threshold_backward_48 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[1024, 32],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_native_batch_norm_backward_threshold_backward_48', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 32
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr2 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = 1.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp12 = tmp10 - tmp11
    tmp13 = tmp5 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tmp19 = tmp17 * tmp18
    tl.store(out_ptr2 + (x0), tmp19, xmask)
    tl.store(out_ptr0 + (x0), tmp9, xmask)
    tl.store(out_ptr1 + (x0), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/bo/cbonnsrqls4cyyqtcsujvkq463i25juhfo4xssebyl2mben5zwnr.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_49 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_49', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.int1)
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp6 = tl.load(in_ptr1 + (x2), None)
    tmp7 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp2 = 1.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp8 = tmp6 - tmp7
    tmp10 = 0.03125
    tmp11 = tmp9 * tmp10
    tmp13 = tmp12 * tmp12
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 * tmp14
    tmp16 = tmp5 - tmp15
    tmp18 = tmp17 * tmp10
    tmp19 = tmp16 - tmp18
    tmp21 = tmp12 * tmp20
    tmp22 = tmp19 * tmp21
    tl.store(in_out_ptr0 + (x2), tmp22, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/zy/czy3w2i3biy6ajhdej3futynwztctpabfao466niatyvmhvxvfml.py
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
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_50', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 896
    rnumel = 115
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 128)
    x0 = xindex % 128
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (115*x1)
        tmp1 = tl.full([1, 1], 800, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (128*(r2 % 5)) + (640*(((r2 + (115*x1)) // 5) % 160))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + (x0 + (128*(r2 % 5)) + (640*(((r2 + (115*x1)) // 5) % 160))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
        tmp13 = tl.load(in_ptr2 + (x0 + (128*(r2 % 5)) + (640*(((r2 + (115*x1)) // 5) % 160))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_zhang402/vd/cvdirelws4q4vuckmvgwdtxaicyqjajmpuqn6abjwje24aalqlah.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_51 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[128, 8],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_51', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/af/cafqf2ooaqvpfgwwqw3aizosfcuuelb47qippjd4kn4stezzn55j.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_52 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[128, 8],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_52', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/gr/cgr75fa53d2w6tzziz3wqzxxlzci7rtf6jkhtyneaaurp6q2hm4q.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_53 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_53', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102400
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
    tmp9 = 0.00125
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


# kernel path: /tmp/torchinductor_zhang402/ot/cot4nbpggnvhfihzudmwdxoewfqscmac73crtu6nghkyfbbe5q6l.py
# Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]

triton_poi_fused_avg_pool2d_backward_54 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_backward_54', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7102464
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 17
    x1 = (xindex // 17) % 17
    x2 = (xindex // 289) % 768
    x3 = (xindex // 221952)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x2 + (768*(tl.minimum(tl.maximum(0, (((-2) + x0) // 3)), (-1) + (tl.minimum(5, 1 + (x0 // 3)))))) + (768*(tl.where((tl.minimum(tl.maximum(0, (((-2) + x0) // 3)), (-1) + (tl.minimum(5, 1 + (x0 // 3))))) >= 0, 0, 5))) + (3840*(tl.minimum(tl.maximum(0, (((-2) + x1) // 3)), (-1) + (tl.minimum(5, 1 + (x1 // 3)))))) + (3840*(tl.where((tl.minimum(tl.maximum(0, (((-2) + x1) // 3)), (-1) + (tl.minimum(5, 1 + (x1 // 3))))) >= 0, 0, 5))) + (19200*x3)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (x2 + (768*(tl.minimum(1 + (tl.maximum(0, (((-2) + x0) // 3))), (-1) + (tl.minimum(5, 1 + (x0 // 3)))))) + (768*(tl.where((tl.minimum(1 + (tl.maximum(0, (((-2) + x0) // 3))), (-1) + (tl.minimum(5, 1 + (x0 // 3))))) >= 0, 0, 5))) + (3840*(tl.minimum(tl.maximum(0, (((-2) + x1) // 3)), (-1) + (tl.minimum(5, 1 + (x1 // 3)))))) + (3840*(tl.where((tl.minimum(tl.maximum(0, (((-2) + x1) // 3)), (-1) + (tl.minimum(5, 1 + (x1 // 3))))) >= 0, 0, 5))) + (19200*x3)), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + (x2 + (768*(tl.minimum(tl.maximum(0, (((-2) + x0) // 3)), (-1) + (tl.minimum(5, 1 + (x0 // 3)))))) + (768*(tl.where((tl.minimum(tl.maximum(0, (((-2) + x0) // 3)), (-1) + (tl.minimum(5, 1 + (x0 // 3))))) >= 0, 0, 5))) + (3840*(tl.minimum(1 + (tl.maximum(0, (((-2) + x1) // 3))), (-1) + (tl.minimum(5, 1 + (x1 // 3)))))) + (3840*(tl.where((tl.minimum(1 + (tl.maximum(0, (((-2) + x1) // 3))), (-1) + (tl.minimum(5, 1 + (x1 // 3))))) >= 0, 0, 5))) + (19200*x3)), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (x2 + (768*(tl.minimum(1 + (tl.maximum(0, (((-2) + x0) // 3))), (-1) + (tl.minimum(5, 1 + (x0 // 3)))))) + (768*(tl.where((tl.minimum(1 + (tl.maximum(0, (((-2) + x0) // 3))), (-1) + (tl.minimum(5, 1 + (x0 // 3))))) >= 0, 0, 5))) + (3840*(tl.minimum(1 + (tl.maximum(0, (((-2) + x1) // 3))), (-1) + (tl.minimum(5, 1 + (x1 // 3)))))) + (3840*(tl.where((tl.minimum(1 + (tl.maximum(0, (((-2) + x1) // 3))), (-1) + (tl.minimum(5, 1 + (x1 // 3))))) >= 0, 0, 5))) + (19200*x3)), None, eviction_policy='evict_last')
    tmp1 = tmp0 / 25
    tmp2 = tl.maximum(0, (((-2) + x1) // 3))
    tmp3 = tl.minimum(5, 1 + (x1 // 3))
    tmp4 = tmp2 < tmp3
    tmp5 = tl.maximum(0, (((-2) + x0) // 3))
    tmp6 = tl.minimum(5, 1 + (x0 // 3))
    tmp7 = tmp5 < tmp6
    tmp8 = tmp4 & tmp7
    tmp9 = 0.0
    tmp10 = tl.where(tmp8, tmp1, tmp9)
    tmp12 = tmp11 / 25
    tmp13 = 1 + (tl.maximum(0, (((-2) + x0) // 3)))
    tmp14 = tmp13 < tmp6
    tmp15 = tmp4 & tmp14
    tmp16 = tmp10 + tmp12
    tmp17 = tl.where(tmp15, tmp16, tmp10)
    tmp19 = tmp18 / 25
    tmp20 = 1 + (tl.maximum(0, (((-2) + x1) // 3)))
    tmp21 = tmp20 < tmp3
    tmp22 = tmp21 & tmp7
    tmp23 = tmp17 + tmp19
    tmp24 = tl.where(tmp22, tmp23, tmp17)
    tmp26 = tmp25 / 25
    tmp27 = tmp21 & tmp14
    tmp28 = tmp24 + tmp26
    tmp29 = tl.where(tmp27, tmp28, tmp24)
    tl.store(out_ptr0 + (x6), tmp29, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/jm/cjm6vtso2fqacdd5j2pj53emwy5d2r4566ofdp6qlhchmfudgdee.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_55 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[16384, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_55', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9248
    xnumel = 192
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 289
    y1 = (yindex // 289)
    tmp0 = tl.load(in_ptr0 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (576 + x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (576 + x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (576 + x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (166464 + y0 + (289*x2) + (221952*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp0, tmp8, tmp7)
    tl.store(out_ptr0 + (x2 + (192*y3)), tmp9, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/g4/cg4jrpon7nvi3ssqeyx5obk2u4co3cni6ln56zwh6iq56ufjpcod.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_56 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_56', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 14016
    rnumel = 127
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 192)
    x0 = xindex % 192
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (127*x1)
        tmp1 = tl.full([1, 1], 9248, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (192*((r2 + (127*x1)) % 9248))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
        tmp9 = tl.load(in_ptr1 + (x0 + (192*((r2 + (127*x1)) % 9248))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr2 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp9 - tmp10
        tmp12 = tmp3 * tmp11
        tmp13 = tl.full(tmp12.shape, 0, tmp12.dtype)
        tmp14 = tl.where(tmp2, tmp12, tmp13)
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask & xmask, tmp17, _tmp16)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp16, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/fa/cfacl3dypdpwy7hb5omtyi2txwxkyat7vza3cihi3vgh4eyeos7l.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_57 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_57', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1775616
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 192
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x2), None)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.00010813148788927336
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/vy/cvy6zfd3s32wwspxhtwe4lkeisoxknbxddz4qgcyodmy7g364v4p.py
# Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]

triton_poi_fused_avg_pool2d_backward_58 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[32768, 512], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_backward_58', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24576
    xnumel = 289
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 17
    x3 = (xindex // 17)
    y0 = yindex % 768
    y1 = (yindex // 768)
    x5 = xindex
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (768*(tl.minimum(tl.maximum(0, (-1) + x2), (-1) + (tl.minimum(17, 2 + x2))))) + (13056*(tl.minimum(tl.maximum(0, (-1) + x3), (-1) + (tl.minimum(17, 2 + x3))))) + (221952*y1)), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (y0 + (768*(tl.minimum(1 + (tl.maximum(0, (-1) + x2)), (-1) + (tl.minimum(17, 2 + x2))))) + (13056*(tl.minimum(tl.maximum(0, (-1) + x3), (-1) + (tl.minimum(17, 2 + x3))))) + (221952*y1)), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + (y0 + (768*(tl.minimum(2 + (tl.maximum(0, (-1) + x2)), (-1) + (tl.minimum(17, 2 + x2))))) + (13056*(tl.minimum(tl.maximum(0, (-1) + x3), (-1) + (tl.minimum(17, 2 + x3))))) + (221952*y1)), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (y0 + (768*(tl.minimum(tl.maximum(0, (-1) + x2), (-1) + (tl.minimum(17, 2 + x2))))) + (13056*(tl.minimum(1 + (tl.maximum(0, (-1) + x3)), (-1) + (tl.minimum(17, 2 + x3))))) + (221952*y1)), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr0 + (y0 + (768*(tl.minimum(1 + (tl.maximum(0, (-1) + x2)), (-1) + (tl.minimum(17, 2 + x2))))) + (13056*(tl.minimum(1 + (tl.maximum(0, (-1) + x3)), (-1) + (tl.minimum(17, 2 + x3))))) + (221952*y1)), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr0 + (y0 + (768*(tl.minimum(2 + (tl.maximum(0, (-1) + x2)), (-1) + (tl.minimum(17, 2 + x2))))) + (13056*(tl.minimum(1 + (tl.maximum(0, (-1) + x3)), (-1) + (tl.minimum(17, 2 + x3))))) + (221952*y1)), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr0 + (y0 + (768*(tl.minimum(tl.maximum(0, (-1) + x2), (-1) + (tl.minimum(17, 2 + x2))))) + (13056*(tl.minimum(2 + (tl.maximum(0, (-1) + x3)), (-1) + (tl.minimum(17, 2 + x3))))) + (221952*y1)), xmask, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr0 + (y0 + (768*(tl.minimum(1 + (tl.maximum(0, (-1) + x2)), (-1) + (tl.minimum(17, 2 + x2))))) + (13056*(tl.minimum(2 + (tl.maximum(0, (-1) + x3)), (-1) + (tl.minimum(17, 2 + x3))))) + (221952*y1)), xmask, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr0 + (y0 + (768*(tl.minimum(2 + (tl.maximum(0, (-1) + x2)), (-1) + (tl.minimum(17, 2 + x2))))) + (13056*(tl.minimum(2 + (tl.maximum(0, (-1) + x3)), (-1) + (tl.minimum(17, 2 + x3))))) + (221952*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 / 9
    tmp2 = tl.maximum(0, (-1) + x3)
    tmp3 = tl.minimum(17, 2 + x3)
    tmp4 = tmp2 < tmp3
    tmp5 = tl.maximum(0, (-1) + x2)
    tmp6 = tl.minimum(17, 2 + x2)
    tmp7 = tmp5 < tmp6
    tmp8 = tmp4 & tmp7
    tmp9 = 0.0
    tmp10 = tl.where(tmp8, tmp1, tmp9)
    tmp12 = tmp11 / 9
    tmp13 = 1 + (tl.maximum(0, (-1) + x2))
    tmp14 = tmp13 < tmp6
    tmp15 = tmp4 & tmp14
    tmp16 = tmp10 + tmp12
    tmp17 = tl.where(tmp15, tmp16, tmp10)
    tmp19 = tmp18 / 9
    tmp20 = 2 + (tl.maximum(0, (-1) + x2))
    tmp21 = tmp20 < tmp6
    tmp22 = tmp4 & tmp21
    tmp23 = tmp17 + tmp19
    tmp24 = tl.where(tmp22, tmp23, tmp17)
    tmp26 = tmp25 / 9
    tmp27 = 1 + (tl.maximum(0, (-1) + x3))
    tmp28 = tmp27 < tmp3
    tmp29 = tmp28 & tmp7
    tmp30 = tmp24 + tmp26
    tmp31 = tl.where(tmp29, tmp30, tmp24)
    tmp33 = tmp32 / 9
    tmp34 = tmp28 & tmp14
    tmp35 = tmp31 + tmp33
    tmp36 = tl.where(tmp34, tmp35, tmp31)
    tmp38 = tmp37 / 9
    tmp39 = tmp28 & tmp21
    tmp40 = tmp36 + tmp38
    tmp41 = tl.where(tmp39, tmp40, tmp36)
    tmp43 = tmp42 / 9
    tmp44 = 2 + (tl.maximum(0, (-1) + x3))
    tmp45 = tmp44 < tmp3
    tmp46 = tmp45 & tmp7
    tmp47 = tmp41 + tmp43
    tmp48 = tl.where(tmp46, tmp47, tmp41)
    tmp50 = tmp49 / 9
    tmp51 = tmp45 & tmp14
    tmp52 = tmp48 + tmp50
    tmp53 = tl.where(tmp51, tmp52, tmp48)
    tmp55 = tmp54 / 9
    tmp56 = tmp45 & tmp21
    tmp57 = tmp53 + tmp55
    tmp58 = tl.where(tmp56, tmp57, tmp53)
    tl.store(out_ptr0 + (x5 + (289*y4)), tmp58, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/n3/cn3wzkoo5spuskvnenzymfbkommaeigz556j4g5pbk4iqeqietlq.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_59 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[16384, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_59', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9248
    xnumel = 192
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 289
    y1 = (yindex // 289)
    tmp0 = tl.load(in_ptr0 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (384 + x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (384 + x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (384 + x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (110976 + y0 + (289*x2) + (221952*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp0, tmp8, tmp7)
    tl.store(out_ptr0 + (x2 + (192*y3)), tmp9, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/om/com4ls7jnkxolout24wh6gcc6p6e6c7px3mq4zoxw5cbwuxjuo42.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_60 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[16384, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_60', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9248
    xnumel = 192
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 289
    y1 = (yindex // 289)
    tmp0 = tl.load(in_ptr0 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (192 + x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (192 + x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (192 + x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (55488 + y0 + (289*x2) + (221952*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp0, tmp8, tmp7)
    tl.store(out_ptr0 + (x2 + (192*y3)), tmp9, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/qf/cqfu47jsc6rscbjlcwqh3fnixrucgs3nh3ovsfgori2dzzhe5kyv.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_61 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[16384, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_61', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9248
    xnumel = 192
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 289
    y1 = (yindex // 289)
    tmp0 = tl.load(in_ptr0 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (y0 + (289*x2) + (221952*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp0, tmp8, tmp7)
    tl.store(out_ptr0 + (x2 + (192*y3)), tmp9, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/xl/cxlns5n77q7gyopqwcnmr64dohufxyrs6gb6wovrjtzajfcq2cfc.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_62 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[16384, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_62', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9248
    xnumel = 192
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 289
    y1 = (yindex // 289)
    tmp0 = tl.load(in_ptr0 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (166464 + y0 + (289*x2) + (221952*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (576 + x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (576 + x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (576 + x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp0, tmp8, tmp7)
    tl.store(out_ptr0 + (x2 + (192*y3)), tmp9, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/zk/czky2xiztf6w4eartx6dvvvsi2mjprmz5vm6cus5csqhtshubw3o.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_63 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[16384, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_63', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9248
    xnumel = 192
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 289
    y1 = (yindex // 289)
    tmp0 = tl.load(in_ptr0 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (110976 + y0 + (289*x2) + (221952*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (384 + x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (384 + x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (384 + x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp0, tmp8, tmp7)
    tl.store(out_ptr0 + (x2 + (192*y3)), tmp9, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/5f/c5fwygtlx5z55uvpreazxabqrs72lu2y5haqgwv3qealox3oweit.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_64 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_64', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 11680
    rnumel = 127
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 160)
    x0 = xindex % 160
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (127*x1)
        tmp1 = tl.full([1, 1], 9248, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (160*((r2 + (127*x1)) % 9248))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + (x0 + (160*((r2 + (127*x1)) % 9248))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
        tmp13 = tl.load(in_ptr2 + (x0 + (160*((r2 + (127*x1)) % 9248))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_zhang402/j6/cj6r3tvvdfnnuo46g7oogaspl4o5euvh6jjauvqchy2vln5demfu.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_65 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_65', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 160
    rnumel = 73
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
        tmp0 = tl.load(in_ptr0 + (x0 + (160*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/av/cavbst5gr5j25cm4pfj3fpsnawb5qodpatb34h4mcf7wb4keneyw.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_66 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_66', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 160
    rnumel = 73
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
        tmp0 = tl.load(in_ptr0 + (x0 + (160*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/fo/cfov6agsmkmebt34kyqnh6jrkhzv52iqjq3dxuhuq4poi4yb4dhf.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_67 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_67', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1479680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 160
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp5 = tl.load(in_ptr1 + (x2), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.00010813148788927336
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/lg/clgqodiso7rsacjmbcd5cypp6sttkki3mseea5q5ty2obwv5nfwt.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_68 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[16384, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_68', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9248
    xnumel = 192
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 289
    y1 = (yindex // 289)
    tmp0 = tl.load(in_ptr0 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (55488 + y0 + (289*x2) + (221952*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (192 + x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (192 + x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (192 + x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp0, tmp8, tmp7)
    tl.store(out_ptr0 + (x2 + (192*y3)), tmp9, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/7g/c7gdjag7eixn37h23vaqquygqi3zqdilsrqhncoqi77mvzvq2hkv.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_69 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[16384, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_69', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9248
    xnumel = 192
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 289
    y1 = (yindex // 289)
    tmp0 = tl.load(in_ptr0 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (289*x2) + (221952*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp0, tmp8, tmp7)
    tl.store(out_ptr0 + (x2 + (192*y3)), tmp9, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/74/c74ywz3qirxdditcwgokwpsvtmll2fufv3c7l5epjehffhjnqasx.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_70 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_70', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9344
    rnumel = 127
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 128)
    x0 = xindex % 128
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (127*x1)
        tmp1 = tl.full([1, 1], 9248, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (128*((r2 + (127*x1)) % 9248))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + (x0 + (128*((r2 + (127*x1)) % 9248))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
        tmp13 = tl.load(in_ptr2 + (x0 + (128*((r2 + (127*x1)) % 9248))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_zhang402/2m/c2mk3enxtis4vfrsp5ntlu5obvw4ygidiarttcbdvagejreq3yv4.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_71 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[128, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_71', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 73
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


# kernel path: /tmp/torchinductor_zhang402/br/cbrimtqnoc52c2zzydmcldyjaurcelqv2xhutrqwn2lfpnxd6tac.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_72 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[128, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_72', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 73
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


# kernel path: /tmp/torchinductor_zhang402/4x/c4x6pclfqjulftyynnpyge7b25gstg7cug7e2vq7xbgd35vdvixw.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_73 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_73', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1183744
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
    tmp9 = 0.00010813148788927336
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


# kernel path: /tmp/torchinductor_zhang402/pi/cpicnn5laqvzwkgebuzijoatzj7ooacwydyyeemhyte4xuiyyaue.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_74 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[16384, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_74', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9248
    xnumel = 768
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 289
    y1 = (yindex // 289)
    y3 = yindex
    tmp0 = tl.load(in_out_ptr0 + (y0 + (289*x2) + (221952*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (y0 + (289*x2) + (221952*y1)), tmp6, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/cj/ccjmaoesh4qwcicnckrpu6abqzeis7ns3dc7pifcludjtjoa5fw2.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_75 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_75', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7008
    rnumel = 127
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 73
    x1 = (xindex // 73)
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (127*x0)
        tmp1 = tl.full([1, 1], 9248, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (96*((r2 + (127*x0)) % 9248))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + (110976 + (289*x1) + (221952*(((r2 + (127*x0)) // 289) % 32)) + ((r2 + (127*x0)) % 289)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = 0.0
        tmp6 = tl.where(tmp3, tmp5, tmp4)
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/uq/cuqiggjtaz3jjt2poon45gb3ybcs4mfpxpq625vuaqz5vld3nyw4.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_76 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[128, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_76', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 73
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (73*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/ld/cldjcwao5x67vicsmsqeirifd6g643332cyslopfs26g4zbm5d25.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_77 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_77', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7008
    rnumel = 127
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 96)
    x0 = xindex % 96
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (127*x1)
        tmp1 = tl.full([1, 1], 9248, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (96*((r2 + (127*x1)) % 9248))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + (110976 + (289*x0) + (221952*(((r2 + (127*x1)) // 289) % 32)) + ((r2 + (127*x1)) % 289)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = 0.0
        tmp6 = tl.where(tmp3, tmp5, tmp4)
        tmp7 = tl.load(in_ptr2 + (x0 + (96*((r2 + (127*x1)) % 9248))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp6 * tmp9
        tmp11 = tl.full(tmp10.shape, 0, tmp10.dtype)
        tmp12 = tl.where(tmp2, tmp10, tmp11)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/7x/c7xugo4phs2uhgueawfatgvshsl6okkx4dxecp76b2lpnnj4rwey.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_78 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[128, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_78', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 73
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
        tmp0 = tl.load(in_ptr0 + (x0 + (96*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/k3/ck3c6guxyr4p2vj5sucpcx6ciindepwg62trz2g6vras3lfwouxg.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_79 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[16384, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_79', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9248
    xnumel = 96
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 289
    y1 = (yindex // 289)
    tmp0 = tl.load(in_ptr0 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (110976 + y0 + (289*x2) + (221952*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp6 = tmp4 - tmp5
    tmp8 = 0.00010813148788927336
    tmp9 = tmp7 * tmp8
    tmp11 = tmp10 * tmp10
    tmp12 = tmp9 * tmp11
    tmp13 = tmp6 * tmp12
    tmp14 = tmp3 - tmp13
    tmp16 = tmp15 * tmp8
    tmp17 = tmp14 - tmp16
    tmp19 = tmp10 * tmp18
    tmp20 = tmp17 * tmp19
    tl.store(out_ptr0 + (x2 + (96*y3)), tmp20, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/dp/cdpag6sjz43nndai3imwwa5pv3glq4unbxytgipiowg74pex6jbo.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_80 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_80', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 29472
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 96)
    x0 = xindex % 96
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x1)
        tmp1 = tl.full([1, 1], 39200, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (96*((r2 + (128*x1)) % 39200))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + (x0 + (96*((r2 + (128*x1)) % 39200))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
        tmp13 = tl.load(in_ptr2 + (x0 + (96*((r2 + (128*x1)) % 39200))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_zhang402/aj/cajq4ucq3cjyshlzxpq2isoryy3htzt5gxesrpr5gc5kh4nutwud.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_81 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_81', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 307
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
        tmp0 = tl.load(in_ptr0 + (x0 + (96*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/ai/caifdbsqw7koahqrylpq2vr7vcem5h3p6xhmuz27krypxvdttzzv.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_82 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_82', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 307
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
        tmp0 = tl.load(in_ptr0 + (x0 + (96*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/on/conlvlkvwg5csjivgopoutxtcnvmjmdxcfiq3r4vhdjcyh2vq6qr.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_83 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_83', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3763200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 96
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp5 = tl.load(in_ptr1 + (x2), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 2.5510204081632654e-05
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/oe/coemerc4bhinwxi67sjj63dgtvzmiazichgs7ih4qilddjwvn5ly.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_84 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_84', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 19648
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 64)
    x0 = xindex % 64
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x1)
        tmp1 = tl.full([1, 1], 39200, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (64*((r2 + (128*x1)) % 39200))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + (x0 + (64*((r2 + (128*x1)) % 39200))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
        tmp13 = tl.load(in_ptr2 + (x0 + (64*((r2 + (128*x1)) % 39200))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_zhang402/tr/ctrrgwnky6ofynajlzkjpfewkfdbciohs2unjm5hk7o32snqt6ml.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_85 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[64, 512],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_85', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 307
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


# kernel path: /tmp/torchinductor_zhang402/34/c34cxxv77dsh36d3yi7uvypzhnvvouizopmfbeqtxpfuswqv7pkl.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_86 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[64, 512],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_86', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 307
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


# kernel path: /tmp/torchinductor_zhang402/2z/c2zuinne3zldauuwtqaslpfvyj7lf7kiiepo3e2dfdyw634zzui5.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_87 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_87', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2508800
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
    tmp9 = 2.5510204081632654e-05
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


# kernel path: /tmp/torchinductor_zhang402/ms/cmslk53mnxsjzx2fhgdfvjgpeebwsan67d3ep3nrlhklkotpqlmf.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_88 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_88', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 28032
    rnumel = 127
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 73
    x1 = (xindex // 73)
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (127*x0)
        tmp1 = tl.full([1, 1], 9248, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (384*((r2 + (127*x0)) % 9248))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + ((289*x1) + (221952*(((r2 + (127*x0)) // 289) % 32)) + ((r2 + (127*x0)) % 289)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = 0.0
        tmp6 = tl.where(tmp3, tmp5, tmp4)
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/bn/cbnqzluv5wt552bx25vxgloqqiyreeuf27ngr6vm3paejhlpwqt7.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_89 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_89', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 73
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (73*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/hn/chn5f75czxsomyxwxipgiy72weviupyvd2xlo5qmbdmcjnm4ipnw.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_90 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_90', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 28032
    rnumel = 127
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 384)
    x0 = xindex % 384
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (127*x1)
        tmp1 = tl.full([1, 1], 9248, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (384*((r2 + (127*x1)) % 9248))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + ((289*x0) + (221952*(((r2 + (127*x1)) // 289) % 32)) + ((r2 + (127*x1)) % 289)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = 0.0
        tmp6 = tl.where(tmp3, tmp5, tmp4)
        tmp7 = tl.load(in_ptr2 + (x0 + (384*((r2 + (127*x1)) % 9248))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp6 * tmp9
        tmp11 = tl.full(tmp10.shape, 0, tmp10.dtype)
        tmp12 = tl.where(tmp2, tmp10, tmp11)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/q4/cq4qnasit2w5xexwy4tgjwp2xwswtaxtprhub3oel2tvbzczjkra.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_91 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_91', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 73
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
        tmp0 = tl.load(in_ptr0 + (x0 + (384*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/z6/cz6h2qtk2r3hbdtpzcswq7w7p7tyqmbd372nd3nsiv4aa2iwbgmi.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_92 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[16384, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_92', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9248
    xnumel = 384
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 289
    y1 = (yindex // 289)
    tmp0 = tl.load(in_ptr0 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (289*x2) + (221952*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp6 = tmp4 - tmp5
    tmp8 = 0.00010813148788927336
    tmp9 = tmp7 * tmp8
    tmp11 = tmp10 * tmp10
    tmp12 = tmp9 * tmp11
    tmp13 = tmp6 * tmp12
    tmp14 = tmp3 - tmp13
    tmp16 = tmp15 * tmp8
    tmp17 = tmp14 - tmp16
    tmp19 = tmp10 * tmp18
    tmp20 = tmp17 * tmp19
    tl.store(out_ptr0 + (x2 + (384*y3)), tmp20, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/oc/cocyrlqrxmz5kxisr63rpb5ucjnliarj5rb3cx37d7qxdlm5r72f.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_93 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_93', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 19648
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 64)
    x0 = xindex % 64
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x1)
        tmp1 = tl.full([1, 1], 39200, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (64*((r2 + (128*x1)) % 39200))), rmask & tmp2 & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + (224 + x0 + (288*((r2 + (128*x1)) % 39200))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr2 + (224 + x0 + (288*((r2 + (128*x1)) % 39200))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tmp4 + tmp5
        tmp7 = tl.load(in_ptr3 + (224 + x0 + (288*((r2 + (128*x1)) % 39200))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tmp6 + tmp7
        tmp9 = 0.0
        tmp10 = tl.where(tmp3, tmp9, tmp8)
        tmp11 = tl.full(tmp10.shape, 0, tmp10.dtype)
        tmp12 = tl.where(tmp2, tmp10, tmp11)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
        tmp16 = tl.load(in_ptr4 + (x0 + (64*((r2 + (128*x1)) % 39200))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tl.load(in_ptr5 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp18 = tmp16 - tmp17
        tmp19 = tmp10 * tmp18
        tmp20 = tl.full(tmp19.shape, 0, tmp19.dtype)
        tmp21 = tl.where(tmp2, tmp19, tmp20)
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
        tmp24 = _tmp23 + tmp22
        _tmp23 = tl.where(rmask & xmask, tmp24, _tmp23)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, xmask)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp23, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/6a/c6a4iojadykp6nb473uhxfz4ag5xlywy564uxpmhb3jjkmy4wh4u.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_94 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_94', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2508800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (224 + x0 + (288*x1)), None)
    tmp2 = tl.load(in_ptr2 + (224 + x0 + (288*x1)), None)
    tmp4 = tl.load(in_ptr3 + (224 + x0 + (288*x1)), None)
    tmp8 = tl.load(in_ptr4 + (x2), None)
    tmp9 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp0, tmp6, tmp5)
    tmp10 = tmp8 - tmp9
    tmp12 = 2.5510204081632654e-05
    tmp13 = tmp11 * tmp12
    tmp15 = tmp14 * tmp14
    tmp16 = tmp13 * tmp15
    tmp17 = tmp10 * tmp16
    tmp18 = tmp7 - tmp17
    tmp20 = tmp19 * tmp12
    tmp21 = tmp18 - tmp20
    tmp23 = tmp14 * tmp22
    tmp24 = tmp21 * tmp23
    tl.store(in_out_ptr0 + (x2), tmp24, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/gg/cgg3k7g2ua7o6nulo3ib56ffjsu4ulu2u64tk23ngei3wnwwqw2e.py
# Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]

triton_poi_fused_avg_pool2d_backward_95 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[16384, 2048], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_backward_95', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9216
    xnumel = 1225
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 35
    x3 = (xindex // 35)
    y0 = yindex % 288
    y1 = (yindex // 288)
    x5 = xindex
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (288*(tl.minimum(tl.maximum(0, (-1) + x2), (-1) + (tl.minimum(35, 2 + x2))))) + (10080*(tl.minimum(tl.maximum(0, (-1) + x3), (-1) + (tl.minimum(35, 2 + x3))))) + (352800*y1)), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (y0 + (288*(tl.minimum(1 + (tl.maximum(0, (-1) + x2)), (-1) + (tl.minimum(35, 2 + x2))))) + (10080*(tl.minimum(tl.maximum(0, (-1) + x3), (-1) + (tl.minimum(35, 2 + x3))))) + (352800*y1)), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + (y0 + (288*(tl.minimum(2 + (tl.maximum(0, (-1) + x2)), (-1) + (tl.minimum(35, 2 + x2))))) + (10080*(tl.minimum(tl.maximum(0, (-1) + x3), (-1) + (tl.minimum(35, 2 + x3))))) + (352800*y1)), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (y0 + (288*(tl.minimum(tl.maximum(0, (-1) + x2), (-1) + (tl.minimum(35, 2 + x2))))) + (10080*(tl.minimum(1 + (tl.maximum(0, (-1) + x3)), (-1) + (tl.minimum(35, 2 + x3))))) + (352800*y1)), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr0 + (y0 + (288*(tl.minimum(1 + (tl.maximum(0, (-1) + x2)), (-1) + (tl.minimum(35, 2 + x2))))) + (10080*(tl.minimum(1 + (tl.maximum(0, (-1) + x3)), (-1) + (tl.minimum(35, 2 + x3))))) + (352800*y1)), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr0 + (y0 + (288*(tl.minimum(2 + (tl.maximum(0, (-1) + x2)), (-1) + (tl.minimum(35, 2 + x2))))) + (10080*(tl.minimum(1 + (tl.maximum(0, (-1) + x3)), (-1) + (tl.minimum(35, 2 + x3))))) + (352800*y1)), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr0 + (y0 + (288*(tl.minimum(tl.maximum(0, (-1) + x2), (-1) + (tl.minimum(35, 2 + x2))))) + (10080*(tl.minimum(2 + (tl.maximum(0, (-1) + x3)), (-1) + (tl.minimum(35, 2 + x3))))) + (352800*y1)), xmask, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr0 + (y0 + (288*(tl.minimum(1 + (tl.maximum(0, (-1) + x2)), (-1) + (tl.minimum(35, 2 + x2))))) + (10080*(tl.minimum(2 + (tl.maximum(0, (-1) + x3)), (-1) + (tl.minimum(35, 2 + x3))))) + (352800*y1)), xmask, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr0 + (y0 + (288*(tl.minimum(2 + (tl.maximum(0, (-1) + x2)), (-1) + (tl.minimum(35, 2 + x2))))) + (10080*(tl.minimum(2 + (tl.maximum(0, (-1) + x3)), (-1) + (tl.minimum(35, 2 + x3))))) + (352800*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 / 9
    tmp2 = tl.maximum(0, (-1) + x3)
    tmp3 = tl.minimum(35, 2 + x3)
    tmp4 = tmp2 < tmp3
    tmp5 = tl.maximum(0, (-1) + x2)
    tmp6 = tl.minimum(35, 2 + x2)
    tmp7 = tmp5 < tmp6
    tmp8 = tmp4 & tmp7
    tmp9 = 0.0
    tmp10 = tl.where(tmp8, tmp1, tmp9)
    tmp12 = tmp11 / 9
    tmp13 = 1 + (tl.maximum(0, (-1) + x2))
    tmp14 = tmp13 < tmp6
    tmp15 = tmp4 & tmp14
    tmp16 = tmp10 + tmp12
    tmp17 = tl.where(tmp15, tmp16, tmp10)
    tmp19 = tmp18 / 9
    tmp20 = 2 + (tl.maximum(0, (-1) + x2))
    tmp21 = tmp20 < tmp6
    tmp22 = tmp4 & tmp21
    tmp23 = tmp17 + tmp19
    tmp24 = tl.where(tmp22, tmp23, tmp17)
    tmp26 = tmp25 / 9
    tmp27 = 1 + (tl.maximum(0, (-1) + x3))
    tmp28 = tmp27 < tmp3
    tmp29 = tmp28 & tmp7
    tmp30 = tmp24 + tmp26
    tmp31 = tl.where(tmp29, tmp30, tmp24)
    tmp33 = tmp32 / 9
    tmp34 = tmp28 & tmp14
    tmp35 = tmp31 + tmp33
    tmp36 = tl.where(tmp34, tmp35, tmp31)
    tmp38 = tmp37 / 9
    tmp39 = tmp28 & tmp21
    tmp40 = tmp36 + tmp38
    tmp41 = tl.where(tmp39, tmp40, tmp36)
    tmp43 = tmp42 / 9
    tmp44 = 2 + (tl.maximum(0, (-1) + x3))
    tmp45 = tmp44 < tmp3
    tmp46 = tmp45 & tmp7
    tmp47 = tmp41 + tmp43
    tmp48 = tl.where(tmp46, tmp47, tmp41)
    tmp50 = tmp49 / 9
    tmp51 = tmp45 & tmp14
    tmp52 = tmp48 + tmp50
    tmp53 = tl.where(tmp51, tmp52, tmp48)
    tmp55 = tmp54 / 9
    tmp56 = tmp45 & tmp21
    tmp57 = tmp53 + tmp55
    tmp58 = tl.where(tmp56, tmp57, tmp53)
    tl.store(out_ptr0 + (x5 + (1225*y4)), tmp58, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/ac/cacmgzedtnsfmbxksfijc66pjl26iaqlc7ozm7zrdhfjdvobsslu.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_96 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_96', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 29472
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 96)
    x0 = xindex % 96
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x1)
        tmp1 = tl.full([1, 1], 39200, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (96*((r2 + (128*x1)) % 39200))), rmask & tmp2 & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + (128 + x0 + (288*((r2 + (128*x1)) % 39200))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr2 + (128 + x0 + (288*((r2 + (128*x1)) % 39200))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tmp4 + tmp5
        tmp7 = tl.load(in_ptr3 + (128 + x0 + (288*((r2 + (128*x1)) % 39200))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tmp6 + tmp7
        tmp9 = 0.0
        tmp10 = tl.where(tmp3, tmp9, tmp8)
        tmp11 = tl.full(tmp10.shape, 0, tmp10.dtype)
        tmp12 = tl.where(tmp2, tmp10, tmp11)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
        tmp16 = tl.load(in_ptr4 + (x0 + (96*((r2 + (128*x1)) % 39200))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tl.load(in_ptr5 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp18 = tmp16 - tmp17
        tmp19 = tmp10 * tmp18
        tmp20 = tl.full(tmp19.shape, 0, tmp19.dtype)
        tmp21 = tl.where(tmp2, tmp19, tmp20)
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
        tmp24 = _tmp23 + tmp22
        _tmp23 = tl.where(rmask & xmask, tmp24, _tmp23)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, xmask)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp23, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/hr/chrm3xbzvgtm5od2r2oochs6he7sm23sistrpssbrsb7gy7q7bsj.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_97 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_97', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3763200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 96
    x1 = (xindex // 96)
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (128 + x0 + (288*x1)), xmask)
    tmp2 = tl.load(in_ptr2 + (128 + x0 + (288*x1)), xmask)
    tmp4 = tl.load(in_ptr3 + (128 + x0 + (288*x1)), xmask)
    tmp8 = tl.load(in_ptr4 + (x2), xmask)
    tmp9 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp0, tmp6, tmp5)
    tmp10 = tmp8 - tmp9
    tmp12 = 2.5510204081632654e-05
    tmp13 = tmp11 * tmp12
    tmp15 = tmp14 * tmp14
    tmp16 = tmp13 * tmp15
    tmp17 = tmp10 * tmp16
    tmp18 = tmp7 - tmp17
    tmp20 = tmp19 * tmp12
    tmp21 = tmp18 - tmp20
    tmp23 = tmp14 * tmp22
    tmp24 = tmp21 * tmp23
    tl.store(in_out_ptr0 + (x2), tmp24, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/2u/c2uwhzskgyx4yretwvgj2wk2f7uqzrbto5ubmyl3vlmqt7i3cjhu.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_98 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_98', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 19648
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 64)
    x0 = xindex % 64
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x1)
        tmp1 = tl.full([1, 1], 39200, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (64*((r2 + (128*x1)) % 39200))), rmask & tmp2 & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + (64 + x0 + (288*((r2 + (128*x1)) % 39200))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr2 + (64 + x0 + (288*((r2 + (128*x1)) % 39200))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tmp4 + tmp5
        tmp7 = tl.load(in_ptr3 + (64 + x0 + (288*((r2 + (128*x1)) % 39200))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tmp6 + tmp7
        tmp9 = 0.0
        tmp10 = tl.where(tmp3, tmp9, tmp8)
        tmp11 = tl.full(tmp10.shape, 0, tmp10.dtype)
        tmp12 = tl.where(tmp2, tmp10, tmp11)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
        tmp16 = tl.load(in_ptr4 + (x0 + (64*((r2 + (128*x1)) % 39200))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tl.load(in_ptr5 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp18 = tmp16 - tmp17
        tmp19 = tmp10 * tmp18
        tmp20 = tl.full(tmp19.shape, 0, tmp19.dtype)
        tmp21 = tl.where(tmp2, tmp19, tmp20)
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
        tmp24 = _tmp23 + tmp22
        _tmp23 = tl.where(rmask & xmask, tmp24, _tmp23)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, xmask)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp23, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/w4/cw4az4eydwws5yr2peanx6ech2nuowvz4cnxtozzpwvlx6qbpncu.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_99 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_99', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2508800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (64 + x0 + (288*x1)), None)
    tmp2 = tl.load(in_ptr2 + (64 + x0 + (288*x1)), None)
    tmp4 = tl.load(in_ptr3 + (64 + x0 + (288*x1)), None)
    tmp8 = tl.load(in_ptr4 + (x2), None)
    tmp9 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp0, tmp6, tmp5)
    tmp10 = tmp8 - tmp9
    tmp12 = 2.5510204081632654e-05
    tmp13 = tmp11 * tmp12
    tmp15 = tmp14 * tmp14
    tmp16 = tmp13 * tmp15
    tmp17 = tmp10 * tmp16
    tmp18 = tmp7 - tmp17
    tmp20 = tmp19 * tmp12
    tmp21 = tmp18 - tmp20
    tmp23 = tmp14 * tmp22
    tmp24 = tmp21 * tmp23
    tl.store(in_out_ptr0 + (x2), tmp24, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/af/cafsbnhimzjasqiy7mwojhebnxu35cw652vlse6rfe6b5wt4k4jt.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_100 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_100', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 14736
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 48)
    x0 = xindex % 48
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x1)
        tmp1 = tl.full([1, 1], 39200, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (48*((r2 + (128*x1)) % 39200))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + (x0 + (48*((r2 + (128*x1)) % 39200))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
        tmp13 = tl.load(in_ptr2 + (x0 + (48*((r2 + (128*x1)) % 39200))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_zhang402/yr/cyrmuyawz7n46m5f6dcm26tbeiwnedguo6m2s3dmy26scey3kwrv.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_101 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[64, 512],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_101', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 48
    rnumel = 307
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
        tmp0 = tl.load(in_ptr0 + (x0 + (48*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/57/c57e3w35jtcb6pc6turlw5mdb7mhipxt6fpdmy6lbkuy7uj6roqs.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_102 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[64, 512],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_102', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 48
    rnumel = 307
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
        tmp0 = tl.load(in_ptr0 + (x0 + (48*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/ub/cubf3ehpgpifmcqazc5x6abialcnb3hvryknayze7kwr6z7deijx.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_103 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_103', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1881600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 48
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp5 = tl.load(in_ptr1 + (x2), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 2.5510204081632654e-05
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/ja/cjavdvbeyqlrhx72y2og4n4ebugnza6zir3lu6uupgujpop6mmhf.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_104 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_104', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 19648
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 64)
    x0 = xindex % 64
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x1)
        tmp1 = tl.full([1, 1], 39200, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (64*((r2 + (128*x1)) % 39200))), rmask & tmp2 & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + (x0 + (288*((r2 + (128*x1)) % 39200))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (288*((r2 + (128*x1)) % 39200))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tmp4 + tmp5
        tmp7 = tl.load(in_ptr3 + (x0 + (288*((r2 + (128*x1)) % 39200))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tmp6 + tmp7
        tmp9 = 0.0
        tmp10 = tl.where(tmp3, tmp9, tmp8)
        tmp11 = tl.full(tmp10.shape, 0, tmp10.dtype)
        tmp12 = tl.where(tmp2, tmp10, tmp11)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
        tmp16 = tl.load(in_ptr4 + (x0 + (64*((r2 + (128*x1)) % 39200))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tl.load(in_ptr5 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp18 = tmp16 - tmp17
        tmp19 = tmp10 * tmp18
        tmp20 = tl.full(tmp19.shape, 0, tmp19.dtype)
        tmp21 = tl.where(tmp2, tmp19, tmp20)
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
        tmp24 = _tmp23 + tmp22
        _tmp23 = tl.where(rmask & xmask, tmp24, _tmp23)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, xmask)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp23, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/gx/cgx3joh4xdzze3i7v5fzwyu376ucqo5hnq5nn7vuev4w6o6rkw3o.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_105 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_105', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2508800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x0 + (288*x1)), None)
    tmp2 = tl.load(in_ptr2 + (x0 + (288*x1)), None)
    tmp4 = tl.load(in_ptr3 + (x0 + (288*x1)), None)
    tmp8 = tl.load(in_ptr4 + (x2), None)
    tmp9 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp0, tmp6, tmp5)
    tmp10 = tmp8 - tmp9
    tmp12 = 2.5510204081632654e-05
    tmp13 = tmp11 * tmp12
    tmp15 = tmp14 * tmp14
    tmp16 = tmp13 * tmp15
    tmp17 = tmp10 * tmp16
    tmp18 = tmp7 - tmp17
    tmp20 = tmp19 * tmp12
    tmp21 = tmp18 - tmp20
    tmp23 = tmp14 * tmp22
    tmp24 = tmp21 * tmp23
    tl.store(in_out_ptr0 + (x2), tmp24, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/il/cilb75mifdl32zqp5cvqxtvg3un72l7dhvqxycxlmmcz5eqwgn6q.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_106 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[65536, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_106', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 39200
    xnumel = 64
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1225
    y1 = (yindex // 1225)
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (274400 + y0 + (1225*x2) + (352800*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (224 + x2 + (288*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (224 + x2 + (288*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (224 + x2 + (288*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp0, tmp8, tmp7)
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp9, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/6m/c6m4y2aetoqh2a7xsrbqjvyewflraa444bhfnej5qobnsiohrhjj.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_107 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_107', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 19648
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 64)
    x0 = xindex % 64
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x1)
        tmp1 = tl.full([1, 1], 39200, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (64*((r2 + (128*x1)) % 39200))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
        tmp9 = tl.load(in_ptr1 + (x0 + (64*((r2 + (128*x1)) % 39200))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr2 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp9 - tmp10
        tmp12 = tmp3 * tmp11
        tmp13 = tl.full(tmp12.shape, 0, tmp12.dtype)
        tmp14 = tl.where(tmp2, tmp12, tmp13)
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask & xmask, tmp17, _tmp16)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp16, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/5s/c5sl3pjuannbvgcqzgjqmpslgat55tdt5pdrc5tmtevycs72rbra.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_108 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_108', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2508800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x2), None)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 2.5510204081632654e-05
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/xf/cxfml6q6lwyltda3mlkndmbyiecuxnvbbnlfaohju6wpbfgs6sh2.py
# Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]

triton_poi_fused_avg_pool2d_backward_109 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[8192, 2048], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_backward_109', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 1225
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 35
    x3 = (xindex // 35)
    y0 = yindex % 256
    y1 = (yindex // 256)
    x5 = xindex
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (256*(tl.minimum(tl.maximum(0, (-1) + x2), (-1) + (tl.minimum(35, 2 + x2))))) + (8960*(tl.minimum(tl.maximum(0, (-1) + x3), (-1) + (tl.minimum(35, 2 + x3))))) + (313600*y1)), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (y0 + (256*(tl.minimum(1 + (tl.maximum(0, (-1) + x2)), (-1) + (tl.minimum(35, 2 + x2))))) + (8960*(tl.minimum(tl.maximum(0, (-1) + x3), (-1) + (tl.minimum(35, 2 + x3))))) + (313600*y1)), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + (y0 + (256*(tl.minimum(2 + (tl.maximum(0, (-1) + x2)), (-1) + (tl.minimum(35, 2 + x2))))) + (8960*(tl.minimum(tl.maximum(0, (-1) + x3), (-1) + (tl.minimum(35, 2 + x3))))) + (313600*y1)), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (y0 + (256*(tl.minimum(tl.maximum(0, (-1) + x2), (-1) + (tl.minimum(35, 2 + x2))))) + (8960*(tl.minimum(1 + (tl.maximum(0, (-1) + x3)), (-1) + (tl.minimum(35, 2 + x3))))) + (313600*y1)), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr0 + (y0 + (256*(tl.minimum(1 + (tl.maximum(0, (-1) + x2)), (-1) + (tl.minimum(35, 2 + x2))))) + (8960*(tl.minimum(1 + (tl.maximum(0, (-1) + x3)), (-1) + (tl.minimum(35, 2 + x3))))) + (313600*y1)), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr0 + (y0 + (256*(tl.minimum(2 + (tl.maximum(0, (-1) + x2)), (-1) + (tl.minimum(35, 2 + x2))))) + (8960*(tl.minimum(1 + (tl.maximum(0, (-1) + x3)), (-1) + (tl.minimum(35, 2 + x3))))) + (313600*y1)), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr0 + (y0 + (256*(tl.minimum(tl.maximum(0, (-1) + x2), (-1) + (tl.minimum(35, 2 + x2))))) + (8960*(tl.minimum(2 + (tl.maximum(0, (-1) + x3)), (-1) + (tl.minimum(35, 2 + x3))))) + (313600*y1)), xmask, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr0 + (y0 + (256*(tl.minimum(1 + (tl.maximum(0, (-1) + x2)), (-1) + (tl.minimum(35, 2 + x2))))) + (8960*(tl.minimum(2 + (tl.maximum(0, (-1) + x3)), (-1) + (tl.minimum(35, 2 + x3))))) + (313600*y1)), xmask, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr0 + (y0 + (256*(tl.minimum(2 + (tl.maximum(0, (-1) + x2)), (-1) + (tl.minimum(35, 2 + x2))))) + (8960*(tl.minimum(2 + (tl.maximum(0, (-1) + x3)), (-1) + (tl.minimum(35, 2 + x3))))) + (313600*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 / 9
    tmp2 = tl.maximum(0, (-1) + x3)
    tmp3 = tl.minimum(35, 2 + x3)
    tmp4 = tmp2 < tmp3
    tmp5 = tl.maximum(0, (-1) + x2)
    tmp6 = tl.minimum(35, 2 + x2)
    tmp7 = tmp5 < tmp6
    tmp8 = tmp4 & tmp7
    tmp9 = 0.0
    tmp10 = tl.where(tmp8, tmp1, tmp9)
    tmp12 = tmp11 / 9
    tmp13 = 1 + (tl.maximum(0, (-1) + x2))
    tmp14 = tmp13 < tmp6
    tmp15 = tmp4 & tmp14
    tmp16 = tmp10 + tmp12
    tmp17 = tl.where(tmp15, tmp16, tmp10)
    tmp19 = tmp18 / 9
    tmp20 = 2 + (tl.maximum(0, (-1) + x2))
    tmp21 = tmp20 < tmp6
    tmp22 = tmp4 & tmp21
    tmp23 = tmp17 + tmp19
    tmp24 = tl.where(tmp22, tmp23, tmp17)
    tmp26 = tmp25 / 9
    tmp27 = 1 + (tl.maximum(0, (-1) + x3))
    tmp28 = tmp27 < tmp3
    tmp29 = tmp28 & tmp7
    tmp30 = tmp24 + tmp26
    tmp31 = tl.where(tmp29, tmp30, tmp24)
    tmp33 = tmp32 / 9
    tmp34 = tmp28 & tmp14
    tmp35 = tmp31 + tmp33
    tmp36 = tl.where(tmp34, tmp35, tmp31)
    tmp38 = tmp37 / 9
    tmp39 = tmp28 & tmp21
    tmp40 = tmp36 + tmp38
    tmp41 = tl.where(tmp39, tmp40, tmp36)
    tmp43 = tmp42 / 9
    tmp44 = 2 + (tl.maximum(0, (-1) + x3))
    tmp45 = tmp44 < tmp3
    tmp46 = tmp45 & tmp7
    tmp47 = tmp41 + tmp43
    tmp48 = tl.where(tmp46, tmp47, tmp41)
    tmp50 = tmp49 / 9
    tmp51 = tmp45 & tmp14
    tmp52 = tmp48 + tmp50
    tmp53 = tl.where(tmp51, tmp52, tmp48)
    tmp55 = tmp54 / 9
    tmp56 = tmp45 & tmp21
    tmp57 = tmp53 + tmp55
    tmp58 = tl.where(tmp56, tmp57, tmp53)
    tl.store(out_ptr0 + (x5 + (1225*y4)), tmp58, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/xw/cxwrdyws5jvjeoftmpm56dzvcs347yqryzyqtjsfkalmuqf2qmup.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_110 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[65536, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_110', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 39200
    xnumel = 96
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1225
    y1 = (yindex // 1225)
    tmp0 = tl.load(in_ptr0 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (156800 + y0 + (1225*x2) + (352800*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (128 + x2 + (288*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (128 + x2 + (288*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (128 + x2 + (288*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp0, tmp8, tmp7)
    tl.store(out_ptr0 + (x2 + (96*y3)), tmp9, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/sp/cspomogvs7xl2cho53nkt2skxh56sngsyyd2lfhxvjbkmb4tnt74.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_111 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_111', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 29472
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 96)
    x0 = xindex % 96
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x1)
        tmp1 = tl.full([1, 1], 39200, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (96*((r2 + (128*x1)) % 39200))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
        tmp9 = tl.load(in_ptr1 + (x0 + (96*((r2 + (128*x1)) % 39200))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr2 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp9 - tmp10
        tmp12 = tmp3 * tmp11
        tmp13 = tl.full(tmp12.shape, 0, tmp12.dtype)
        tmp14 = tl.where(tmp2, tmp12, tmp13)
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask & xmask, tmp17, _tmp16)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp16, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/v5/cv5bwngwgytxtwbx7yrrbvnq5hqpctzhn7zrr4fro77w3ehz5fvn.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_112 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_112', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3763200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 96
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x2), xmask)
    tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 2.5510204081632654e-05
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/uu/cuuru7xpdaqj3sodeg7abkchuf4srelfip5rp7m3q6c2em62nayj.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_113 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[65536, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_113', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 39200
    xnumel = 64
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1225
    y1 = (yindex // 1225)
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (78400 + y0 + (1225*x2) + (352800*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (64 + x2 + (288*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (64 + x2 + (288*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (64 + x2 + (288*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp0, tmp8, tmp7)
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp9, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/4j/c4jxty4n6i52jmftd7kffhcyvnweotsp2lb3xkjhllem4lafvxgo.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_114 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[65536, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_114', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 39200
    xnumel = 64
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1225
    y1 = (yindex // 1225)
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (1225*x2) + (352800*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2 + (288*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2 + (288*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (x2 + (288*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp0, tmp8, tmp7)
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp9, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/pa/cpae6wm62resfki7ukwnqtjzldeaaxm52abhvs5rx4r6i2dpawro.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_115 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[65536, 32], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_115', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 39200
    xnumel = 32
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1225
    y1 = (yindex // 1225)
    tmp0 = tl.load(in_ptr0 + (x2 + (32*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (274400 + y0 + (1225*x2) + (313600*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (224 + x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (224 + x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (224 + x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp0, tmp8, tmp7)
    tl.store(out_ptr0 + (x2 + (32*y3)), tmp9, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/cs/ccsyen7atwtkqt5lb5uixjiepwy6l7glkpy5ngnjmfqo5gdomdve.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_116 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_116', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9824
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 32)
    x0 = xindex % 32
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x1)
        tmp1 = tl.full([1, 1], 39200, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (32*((r2 + (128*x1)) % 39200))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
        tmp9 = tl.load(in_ptr1 + (x0 + (32*((r2 + (128*x1)) % 39200))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr2 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp9 - tmp10
        tmp12 = tmp3 * tmp11
        tmp13 = tl.full(tmp12.shape, 0, tmp12.dtype)
        tmp14 = tl.where(tmp2, tmp12, tmp13)
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask & xmask, tmp17, _tmp16)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp16, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/fd/cfd2y3dvhrvv55l2pd6wpfbowjcpylf63rys6uxdy4tyeyqcme2v.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_117 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[32, 512],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_117', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 307
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
        tmp0 = tl.load(in_ptr0 + (x0 + (32*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/nc/cncbdjwdllbasz2de5elkfsyaiab545qit7ppn4ua7sioqmsf7nm.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_118 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[32, 512],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_118', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 307
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
        tmp0 = tl.load(in_ptr0 + (x0 + (32*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/7n/c7ne6mqftdfie54kv75ks6ssyblr6txoi5h4mmygwrfjyy7dse4g.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_119 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_119', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1254400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x2), xmask)
    tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 2.5510204081632654e-05
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/em/cemn6tg74vmjivlgfatwc6bmgbbvpx74q3idzoxtbb6hbjptyqvc.py
# Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]

triton_poi_fused_avg_pool2d_backward_120 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[8192, 2048], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_backward_120', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 1225
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 35
    x3 = (xindex // 35)
    y0 = yindex % 192
    y1 = (yindex // 192)
    x5 = xindex
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (192*(tl.minimum(tl.maximum(0, (-1) + x2), (-1) + (tl.minimum(35, 2 + x2))))) + (6720*(tl.minimum(tl.maximum(0, (-1) + x3), (-1) + (tl.minimum(35, 2 + x3))))) + (235200*y1)), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (y0 + (192*(tl.minimum(1 + (tl.maximum(0, (-1) + x2)), (-1) + (tl.minimum(35, 2 + x2))))) + (6720*(tl.minimum(tl.maximum(0, (-1) + x3), (-1) + (tl.minimum(35, 2 + x3))))) + (235200*y1)), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + (y0 + (192*(tl.minimum(2 + (tl.maximum(0, (-1) + x2)), (-1) + (tl.minimum(35, 2 + x2))))) + (6720*(tl.minimum(tl.maximum(0, (-1) + x3), (-1) + (tl.minimum(35, 2 + x3))))) + (235200*y1)), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (y0 + (192*(tl.minimum(tl.maximum(0, (-1) + x2), (-1) + (tl.minimum(35, 2 + x2))))) + (6720*(tl.minimum(1 + (tl.maximum(0, (-1) + x3)), (-1) + (tl.minimum(35, 2 + x3))))) + (235200*y1)), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr0 + (y0 + (192*(tl.minimum(1 + (tl.maximum(0, (-1) + x2)), (-1) + (tl.minimum(35, 2 + x2))))) + (6720*(tl.minimum(1 + (tl.maximum(0, (-1) + x3)), (-1) + (tl.minimum(35, 2 + x3))))) + (235200*y1)), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr0 + (y0 + (192*(tl.minimum(2 + (tl.maximum(0, (-1) + x2)), (-1) + (tl.minimum(35, 2 + x2))))) + (6720*(tl.minimum(1 + (tl.maximum(0, (-1) + x3)), (-1) + (tl.minimum(35, 2 + x3))))) + (235200*y1)), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr0 + (y0 + (192*(tl.minimum(tl.maximum(0, (-1) + x2), (-1) + (tl.minimum(35, 2 + x2))))) + (6720*(tl.minimum(2 + (tl.maximum(0, (-1) + x3)), (-1) + (tl.minimum(35, 2 + x3))))) + (235200*y1)), xmask, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr0 + (y0 + (192*(tl.minimum(1 + (tl.maximum(0, (-1) + x2)), (-1) + (tl.minimum(35, 2 + x2))))) + (6720*(tl.minimum(2 + (tl.maximum(0, (-1) + x3)), (-1) + (tl.minimum(35, 2 + x3))))) + (235200*y1)), xmask, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr0 + (y0 + (192*(tl.minimum(2 + (tl.maximum(0, (-1) + x2)), (-1) + (tl.minimum(35, 2 + x2))))) + (6720*(tl.minimum(2 + (tl.maximum(0, (-1) + x3)), (-1) + (tl.minimum(35, 2 + x3))))) + (235200*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 / 9
    tmp2 = tl.maximum(0, (-1) + x3)
    tmp3 = tl.minimum(35, 2 + x3)
    tmp4 = tmp2 < tmp3
    tmp5 = tl.maximum(0, (-1) + x2)
    tmp6 = tl.minimum(35, 2 + x2)
    tmp7 = tmp5 < tmp6
    tmp8 = tmp4 & tmp7
    tmp9 = 0.0
    tmp10 = tl.where(tmp8, tmp1, tmp9)
    tmp12 = tmp11 / 9
    tmp13 = 1 + (tl.maximum(0, (-1) + x2))
    tmp14 = tmp13 < tmp6
    tmp15 = tmp4 & tmp14
    tmp16 = tmp10 + tmp12
    tmp17 = tl.where(tmp15, tmp16, tmp10)
    tmp19 = tmp18 / 9
    tmp20 = 2 + (tl.maximum(0, (-1) + x2))
    tmp21 = tmp20 < tmp6
    tmp22 = tmp4 & tmp21
    tmp23 = tmp17 + tmp19
    tmp24 = tl.where(tmp22, tmp23, tmp17)
    tmp26 = tmp25 / 9
    tmp27 = 1 + (tl.maximum(0, (-1) + x3))
    tmp28 = tmp27 < tmp3
    tmp29 = tmp28 & tmp7
    tmp30 = tmp24 + tmp26
    tmp31 = tl.where(tmp29, tmp30, tmp24)
    tmp33 = tmp32 / 9
    tmp34 = tmp28 & tmp14
    tmp35 = tmp31 + tmp33
    tmp36 = tl.where(tmp34, tmp35, tmp31)
    tmp38 = tmp37 / 9
    tmp39 = tmp28 & tmp21
    tmp40 = tmp36 + tmp38
    tmp41 = tl.where(tmp39, tmp40, tmp36)
    tmp43 = tmp42 / 9
    tmp44 = 2 + (tl.maximum(0, (-1) + x3))
    tmp45 = tmp44 < tmp3
    tmp46 = tmp45 & tmp7
    tmp47 = tmp41 + tmp43
    tmp48 = tl.where(tmp46, tmp47, tmp41)
    tmp50 = tmp49 / 9
    tmp51 = tmp45 & tmp14
    tmp52 = tmp48 + tmp50
    tmp53 = tl.where(tmp51, tmp52, tmp48)
    tmp55 = tmp54 / 9
    tmp56 = tmp45 & tmp21
    tmp57 = tmp53 + tmp55
    tmp58 = tl.where(tmp56, tmp57, tmp53)
    tl.store(out_ptr0 + (x5 + (1225*y4)), tmp58, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/hg/chgft6mofutpau367shexrjpi542e76meoyc6y7wikwtpnz7dv3g.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_121 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[65536, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_121', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 39200
    xnumel = 96
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1225
    y1 = (yindex // 1225)
    tmp0 = tl.load(in_ptr0 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (156800 + y0 + (1225*x2) + (313600*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (128 + x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (128 + x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (128 + x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp0, tmp8, tmp7)
    tl.store(out_ptr0 + (x2 + (96*y3)), tmp9, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/pz/cpzmhhwevnkt4vobp3u7pni2emalmrfym5nvurt3gg75totansep.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_122 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[65536, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_122', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 39200
    xnumel = 64
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1225
    y1 = (yindex // 1225)
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (78400 + y0 + (1225*x2) + (313600*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (64 + x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (64 + x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (64 + x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp0, tmp8, tmp7)
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp9, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/vo/cvogaifhu7uiuqinskowpsumgoflxwje5p6vjvyuyvu46xt6cgyj.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_123 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[65536, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_123', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 39200
    xnumel = 64
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1225
    y1 = (yindex // 1225)
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (1225*x2) + (313600*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp0, tmp8, tmp7)
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp9, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/a4/ca4voavlqb3eehfmh73bvvfxtgrcwaon4bwkomm3ix6nlkjugl7k.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_124 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[65536, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_124', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 39200
    xnumel = 192
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 1225
    y1 = (yindex // 1225)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (1225*x2) + (235200*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (192*y3)), tmp6, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/34/c34tv22lekohgkzfy6bzhj5lpz3xe7iknipr7wuwmpgajevpz4md.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_125 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_125', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 82560
    rnumel = 376
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 192)
    x0 = xindex % 192
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (376*x1)
        tmp1 = tl.full([1, 1], 161312, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (192*((r2 + (376*x1)) % 161312))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + (x0 + (192*((r2 + (376*x1)) % 161312))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
        tmp13 = tl.load(in_ptr2 + (x0 + (192*((r2 + (376*x1)) % 161312))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_zhang402/xp/cxpttnim5pejabweucjfn7qlrmcsyvbsxgupt5n5vv45nvnqd4cr.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_126 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_126', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 430
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
        tmp0 = tl.load(in_ptr0 + (x0 + (192*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/t6/ct6o2julkcniasge7qi6ohxfcmpty5ss72b2il32wscyuon4fhbg.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_127 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_127', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 430
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
        tmp0 = tl.load(in_ptr0 + (x0 + (192*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/ko/ckogrid5fzjnojrwji3ywt4d4mpindjvapjligxmokxseujnl76q.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_128 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_128', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 30971904
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 192
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
    tmp9 = 6.199166831977782e-06
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


# kernel path: /tmp/torchinductor_zhang402/p5/cp57kptshtastmji3bhbuy2tllycwpn6jpk65qgq2xu6a2elpkzx.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_129 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_129', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 106640
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 80)
    x0 = xindex % 80
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x1)
        tmp1 = tl.full([1, 1], 170528, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (80*((r2 + (128*x1)) % 170528))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + (x0 + (80*((r2 + (128*x1)) % 170528))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
        tmp13 = tl.load(in_ptr2 + (x0 + (80*((r2 + (128*x1)) % 170528))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_zhang402/vg/cvgaj5myjxzyvreq53s3rhqscne7ngce4rlp7xjg5tntmnhauqpd.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_130 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[128, 2048],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_130', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 80
    rnumel = 1333
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
        tmp0 = tl.load(in_ptr0 + (x0 + (80*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/kk/ckkkamgtqmslckr4fjjd5zcispan5frhug3joimdp4ol5f5ldaav.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_131 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[128, 2048],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_131', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 80
    rnumel = 1333
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
        tmp0 = tl.load(in_ptr0 + (x0 + (80*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/zo/czogrxqvkhr7v4op36ivolwadxit4lrorv5ang4zsf3g6kshvhmw.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_132 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_132', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 13642240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 80
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp5 = tl.load(in_ptr1 + (x2), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 5.864139613435917e-06
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/oi/coialgv3k2ulwrgnknwjg2jdmzzla6x2tjdhdgdyi7cecbbi7nuk.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_133 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_133', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 56448
    rnumel = 784
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (50176*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (x0 + (64*r2) + (50176*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (x0 + (64*r2) + (50176*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_zhang402/ji/cjixucpfc7ptfra6afeuuzs3azxo5y23qyflzisdjnwjleuvfmw5.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_134 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_134', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 882
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


# kernel path: /tmp/torchinductor_zhang402/yx/cyxd2sfc4yzspbopf4mruipy4cw6meqwcndcdq4uphcfmgq5icpf.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_135 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_135', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 882
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


# kernel path: /tmp/torchinductor_zhang402/fl/cfletxehhvdzfg6hbvx3geaq7c6x3wdyq5hsdilx5lmyo5s43yyz.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_136 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_136', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 44255232
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
    tmp9 = 1.446156693970105e-06
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


# kernel path: /tmp/torchinductor_zhang402/ah/cahueuxmyxbrac7jxfunwmx6nzidpsglpewxcbuisncyqbdinarc.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_137 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[32768, 1024],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_137', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 28224
    rnumel = 784
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (32*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (x0 + (32*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (x0 + (32*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_zhang402/26/c26fh7arg5aj3556y67ubrokrp6qt57m7xjetgcu43k6gnv3l42o.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_138 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[32, 1024],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_138', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 882
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
        tmp0 = tl.load(in_ptr0 + (x0 + (32*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/oq/coqbq7dcohyyx6awzg6lqmxnp7sziqheuyqjgtp3kdldqhaedgo7.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_139 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[32, 1024],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_139', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 882
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
        tmp0 = tl.load(in_ptr0 + (x0 + (32*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/2h/c2hs2hq2y6nx62o3nfoqkprw6ubbo45vu45ieobxmynhseq76ejt.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_140 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_140', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 22127616
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp5 = tl.load(in_ptr1 + (x2), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 1.446156693970105e-06
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/b5/cb5plti5k3np5p7gbtdlnysovdgihepf6bxnk6nkxcefsczzxs5c.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_141 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[32768, 1024],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_141', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 27616
    rnumel = 824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 32)
    x0 = xindex % 32
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (824*x1)
        tmp1 = tl.full([1, 1], 710432, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (32*((r2 + (824*x1)) % 710432))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + (x0 + (32*((r2 + (824*x1)) % 710432))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
        tmp13 = tl.load(in_ptr2 + (x0 + (32*((r2 + (824*x1)) % 710432))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_zhang402/t5/ct5zktoyribcygmimbboh2rmty4i2spvfionqj4jerqjcby4uwsh.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_142 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[32, 1024],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_142', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 863
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
        tmp0 = tl.load(in_ptr0 + (x0 + (32*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/uv/cuvyrt7i5xqyr7wy3nxq4aulzlmfxmchjplvfzrarhzs6qaoqxbg.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_143 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[32, 1024],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_143', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 863
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
        tmp0 = tl.load(in_ptr0 + (x0 + (32*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/bd/cbdfk3narpskzc3ld6x254otzofa2rkivuulqufwppathv3hbreo.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_144 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_144', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 22733824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp5 = tl.load(in_ptr1 + (x2), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 1.4075942525111482e-06
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_160, primals_161, primals_163, primals_164, primals_166, primals_167, primals_169, primals_170, primals_172, primals_173, primals_175, primals_176, primals_178, primals_179, primals_181, primals_182, primals_184, primals_185, primals_187, primals_188, primals_190, primals_191, primals_193, primals_194, primals_196, primals_197, primals_199, primals_200, primals_202, primals_203, primals_205, primals_206, primals_208, primals_209, primals_211, primals_212, primals_214, primals_215, primals_219, primals_220, primals_222, primals_223, primals_225, primals_226, primals_228, primals_229, primals_231, primals_232, primals_234, primals_235, primals_237, primals_238, primals_240, primals_241, primals_243, primals_244, primals_246, primals_247, primals_249, primals_250, primals_252, primals_253, primals_255, primals_256, primals_258, primals_259, primals_261, primals_262, primals_264, primals_265, primals_267, primals_268, primals_270, primals_271, primals_273, primals_274, primals_276, primals_277, primals_279, primals_280, primals_282, primals_283, primals_285, primals_286, primals_288, primals_289, cat, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, getitem_6, getitem_7, convolution_3, squeeze_10, relu_3, convolution_4, squeeze_13, relu_4, getitem_12, getitem_13, convolution_5, squeeze_16, convolution_6, squeeze_19, relu_6, convolution_7, squeeze_22, convolution_8, squeeze_25, relu_8, convolution_9, squeeze_28, relu_9, convolution_10, squeeze_31, avg_pool2d, convolution_11, squeeze_34, cat_1, convolution_12, squeeze_37, convolution_13, squeeze_40, relu_13, convolution_14, squeeze_43, convolution_15, squeeze_46, relu_15, convolution_16, squeeze_49, relu_16, convolution_17, squeeze_52, avg_pool2d_1, convolution_18, squeeze_55, cat_2, convolution_19, squeeze_58, convolution_20, squeeze_61, relu_20, convolution_21, squeeze_64, convolution_22, squeeze_67, relu_22, convolution_23, squeeze_70, relu_23, convolution_24, squeeze_73, avg_pool2d_2, convolution_25, squeeze_76, cat_3, convolution_26, squeeze_79, convolution_27, squeeze_82, relu_27, convolution_28, squeeze_85, relu_28, convolution_29, squeeze_88, getitem_65, cat_4, convolution_30, squeeze_91, convolution_31, squeeze_94, relu_31, convolution_32, squeeze_97, relu_32, convolution_33, squeeze_100, convolution_34, squeeze_103, relu_34, convolution_35, squeeze_106, relu_35, convolution_36, squeeze_109, relu_36, convolution_37, squeeze_112, relu_37, convolution_38, squeeze_115, avg_pool2d_3, convolution_39, squeeze_118, cat_5, convolution_40, squeeze_121, convolution_41, squeeze_124, relu_41, convolution_42, squeeze_127, relu_42, convolution_43, squeeze_130, convolution_44, squeeze_133, relu_44, convolution_45, squeeze_136, relu_45, convolution_46, squeeze_139, relu_46, convolution_47, squeeze_142, relu_47, convolution_48, squeeze_145, avg_pool2d_4, convolution_49, squeeze_148, cat_6, convolution_50, squeeze_151, convolution_51, squeeze_154, relu_51, convolution_52, squeeze_157, relu_52, convolution_53, squeeze_160, convolution_54, squeeze_163, relu_54, convolution_55, squeeze_166, relu_55, convolution_56, squeeze_169, relu_56, convolution_57, squeeze_172, relu_57, convolution_58, squeeze_175, avg_pool2d_5, convolution_59, squeeze_178, cat_7, convolution_60, squeeze_181, convolution_61, squeeze_184, relu_61, convolution_62, squeeze_187, relu_62, convolution_63, squeeze_190, convolution_64, squeeze_193, relu_64, convolution_65, squeeze_196, relu_65, convolution_66, squeeze_199, relu_66, convolution_67, squeeze_202, relu_67, convolution_68, squeeze_205, avg_pool2d_6, convolution_69, squeeze_208, cat_8, avg_pool2d_7, convolution_70, squeeze_211, relu_70, convolution_71, squeeze_214, view, convolution_72, squeeze_217, relu_72, convolution_73, squeeze_220, convolution_74, squeeze_223, relu_74, convolution_75, squeeze_226, relu_75, convolution_76, squeeze_229, relu_76, convolution_77, squeeze_232, getitem_163, cat_9, convolution_78, squeeze_235, convolution_79, squeeze_238, relu_79, convolution_80, squeeze_241, convolution_81, squeeze_244, convolution_82, squeeze_247, relu_82, convolution_83, squeeze_250, relu_83, convolution_84, squeeze_253, convolution_85, squeeze_256, avg_pool2d_8, convolution_86, squeeze_259, cat_12, convolution_87, squeeze_262, convolution_88, squeeze_265, relu_88, convolution_89, squeeze_268, convolution_90, squeeze_271, convolution_91, squeeze_274, relu_91, convolution_92, squeeze_277, relu_92, convolution_93, squeeze_280, convolution_94, squeeze_283, avg_pool2d_9, convolution_95, squeeze_286, gt, view_1, permute_2, le, unsqueeze_389, le_1, unsqueeze_401, le_2, unsqueeze_413, unsqueeze_425, unsqueeze_437, le_5, unsqueeze_449, le_6, unsqueeze_461, unsqueeze_473, le_8, unsqueeze_485, le_9, unsqueeze_497, le_10, unsqueeze_509, le_11, unsqueeze_521, unsqueeze_533, unsqueeze_545, le_14, unsqueeze_557, le_15, unsqueeze_569, unsqueeze_581, le_17, unsqueeze_593, le_18, unsqueeze_605, unsqueeze_617, unsqueeze_629, unsqueeze_641, le_22, unsqueeze_653, unsqueeze_665, permute_6, le_24, unsqueeze_677, unsqueeze_689, le_26, unsqueeze_701, le_27, unsqueeze_713, unsqueeze_725, unsqueeze_737, unsqueeze_749, unsqueeze_761, le_32, unsqueeze_773, unsqueeze_785, unsqueeze_797, le_35, unsqueeze_809, le_36, unsqueeze_821, le_37, unsqueeze_833, unsqueeze_845, unsqueeze_857, unsqueeze_869, unsqueeze_881, le_42, unsqueeze_893, unsqueeze_905, unsqueeze_917, le_45, unsqueeze_929, le_46, unsqueeze_941, le_47, unsqueeze_953, unsqueeze_965, unsqueeze_977, unsqueeze_989, unsqueeze_1001, le_52, unsqueeze_1013, unsqueeze_1025, unsqueeze_1037, le_55, unsqueeze_1049, le_56, unsqueeze_1061, le_57, unsqueeze_1073, unsqueeze_1085, unsqueeze_1097, unsqueeze_1109, unsqueeze_1121, le_62, unsqueeze_1133, unsqueeze_1145, unsqueeze_1157, le_65, unsqueeze_1169, le_66, unsqueeze_1181, unsqueeze_1193, unsqueeze_1205, le_69, unsqueeze_1217, le_70, unsqueeze_1229, le_71, unsqueeze_1241, unsqueeze_1253, unsqueeze_1265, le_74, unsqueeze_1277, unsqueeze_1289, le_76, unsqueeze_1301, le_77, unsqueeze_1313, le_78, unsqueeze_1325, unsqueeze_1337, unsqueeze_1349, le_81, unsqueeze_1361, unsqueeze_1373, le_83, unsqueeze_1385, le_84, unsqueeze_1397, le_85, unsqueeze_1409, unsqueeze_1421, unsqueeze_1433, le_88, unsqueeze_1445, unsqueeze_1457, le_90, unsqueeze_1469, unsqueeze_1481, unsqueeze_1493, unsqueeze_1505, unsqueeze_1517, unsqueeze_1529, tangents_1, tangents_2 = args
    args.clear()
    assert_size_stride(primals_1, (32, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_2, (32, ), (1, ))
    assert_size_stride(primals_4, (32, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(primals_5, (32, ), (1, ))
    assert_size_stride(primals_7, (64, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(primals_8, (64, ), (1, ))
    assert_size_stride(primals_10, (80, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_11, (80, ), (1, ))
    assert_size_stride(primals_13, (192, 80, 3, 3), (720, 1, 240, 80))
    assert_size_stride(primals_14, (192, ), (1, ))
    assert_size_stride(primals_16, (64, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_17, (64, ), (1, ))
    assert_size_stride(primals_19, (48, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_20, (48, ), (1, ))
    assert_size_stride(primals_22, (64, 48, 5, 5), (1200, 1, 240, 48))
    assert_size_stride(primals_23, (64, ), (1, ))
    assert_size_stride(primals_25, (64, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_26, (64, ), (1, ))
    assert_size_stride(primals_28, (96, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_29, (96, ), (1, ))
    assert_size_stride(primals_31, (96, 96, 3, 3), (864, 1, 288, 96))
    assert_size_stride(primals_32, (96, ), (1, ))
    assert_size_stride(primals_34, (32, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_35, (32, ), (1, ))
    assert_size_stride(primals_37, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_38, (64, ), (1, ))
    assert_size_stride(primals_40, (48, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_41, (48, ), (1, ))
    assert_size_stride(primals_43, (64, 48, 5, 5), (1200, 1, 240, 48))
    assert_size_stride(primals_44, (64, ), (1, ))
    assert_size_stride(primals_46, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_47, (64, ), (1, ))
    assert_size_stride(primals_49, (96, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_50, (96, ), (1, ))
    assert_size_stride(primals_52, (96, 96, 3, 3), (864, 1, 288, 96))
    assert_size_stride(primals_53, (96, ), (1, ))
    assert_size_stride(primals_55, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_56, (64, ), (1, ))
    assert_size_stride(primals_58, (64, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(primals_59, (64, ), (1, ))
    assert_size_stride(primals_61, (48, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(primals_62, (48, ), (1, ))
    assert_size_stride(primals_64, (64, 48, 5, 5), (1200, 1, 240, 48))
    assert_size_stride(primals_65, (64, ), (1, ))
    assert_size_stride(primals_67, (64, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(primals_68, (64, ), (1, ))
    assert_size_stride(primals_70, (96, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_71, (96, ), (1, ))
    assert_size_stride(primals_73, (96, 96, 3, 3), (864, 1, 288, 96))
    assert_size_stride(primals_74, (96, ), (1, ))
    assert_size_stride(primals_76, (64, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(primals_77, (64, ), (1, ))
    assert_size_stride(primals_79, (384, 288, 3, 3), (2592, 1, 864, 288))
    assert_size_stride(primals_80, (384, ), (1, ))
    assert_size_stride(primals_82, (64, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(primals_83, (64, ), (1, ))
    assert_size_stride(primals_85, (96, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_86, (96, ), (1, ))
    assert_size_stride(primals_88, (96, 96, 3, 3), (864, 1, 288, 96))
    assert_size_stride(primals_89, (96, ), (1, ))
    assert_size_stride(primals_91, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_92, (192, ), (1, ))
    assert_size_stride(primals_94, (128, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_95, (128, ), (1, ))
    assert_size_stride(primals_97, (128, 128, 1, 7), (896, 1, 896, 128))
    assert_size_stride(primals_98, (128, ), (1, ))
    assert_size_stride(primals_100, (192, 128, 7, 1), (896, 1, 128, 128))
    assert_size_stride(primals_101, (192, ), (1, ))
    assert_size_stride(primals_103, (128, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_104, (128, ), (1, ))
    assert_size_stride(primals_106, (128, 128, 7, 1), (896, 1, 128, 128))
    assert_size_stride(primals_107, (128, ), (1, ))
    assert_size_stride(primals_109, (128, 128, 1, 7), (896, 1, 896, 128))
    assert_size_stride(primals_110, (128, ), (1, ))
    assert_size_stride(primals_112, (128, 128, 7, 1), (896, 1, 128, 128))
    assert_size_stride(primals_113, (128, ), (1, ))
    assert_size_stride(primals_115, (192, 128, 1, 7), (896, 1, 896, 128))
    assert_size_stride(primals_116, (192, ), (1, ))
    assert_size_stride(primals_118, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_119, (192, ), (1, ))
    assert_size_stride(primals_121, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_122, (192, ), (1, ))
    assert_size_stride(primals_124, (160, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_125, (160, ), (1, ))
    assert_size_stride(primals_127, (160, 160, 1, 7), (1120, 1, 1120, 160))
    assert_size_stride(primals_128, (160, ), (1, ))
    assert_size_stride(primals_130, (192, 160, 7, 1), (1120, 1, 160, 160))
    assert_size_stride(primals_131, (192, ), (1, ))
    assert_size_stride(primals_133, (160, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_134, (160, ), (1, ))
    assert_size_stride(primals_136, (160, 160, 7, 1), (1120, 1, 160, 160))
    assert_size_stride(primals_137, (160, ), (1, ))
    assert_size_stride(primals_139, (160, 160, 1, 7), (1120, 1, 1120, 160))
    assert_size_stride(primals_140, (160, ), (1, ))
    assert_size_stride(primals_142, (160, 160, 7, 1), (1120, 1, 160, 160))
    assert_size_stride(primals_143, (160, ), (1, ))
    assert_size_stride(primals_145, (192, 160, 1, 7), (1120, 1, 1120, 160))
    assert_size_stride(primals_146, (192, ), (1, ))
    assert_size_stride(primals_148, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_149, (192, ), (1, ))
    assert_size_stride(primals_151, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_152, (192, ), (1, ))
    assert_size_stride(primals_154, (160, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_155, (160, ), (1, ))
    assert_size_stride(primals_157, (160, 160, 1, 7), (1120, 1, 1120, 160))
    assert_size_stride(primals_158, (160, ), (1, ))
    assert_size_stride(primals_160, (192, 160, 7, 1), (1120, 1, 160, 160))
    assert_size_stride(primals_161, (192, ), (1, ))
    assert_size_stride(primals_163, (160, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_164, (160, ), (1, ))
    assert_size_stride(primals_166, (160, 160, 7, 1), (1120, 1, 160, 160))
    assert_size_stride(primals_167, (160, ), (1, ))
    assert_size_stride(primals_169, (160, 160, 1, 7), (1120, 1, 1120, 160))
    assert_size_stride(primals_170, (160, ), (1, ))
    assert_size_stride(primals_172, (160, 160, 7, 1), (1120, 1, 160, 160))
    assert_size_stride(primals_173, (160, ), (1, ))
    assert_size_stride(primals_175, (192, 160, 1, 7), (1120, 1, 1120, 160))
    assert_size_stride(primals_176, (192, ), (1, ))
    assert_size_stride(primals_178, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_179, (192, ), (1, ))
    assert_size_stride(primals_181, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_182, (192, ), (1, ))
    assert_size_stride(primals_184, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_185, (192, ), (1, ))
    assert_size_stride(primals_187, (192, 192, 1, 7), (1344, 1, 1344, 192))
    assert_size_stride(primals_188, (192, ), (1, ))
    assert_size_stride(primals_190, (192, 192, 7, 1), (1344, 1, 192, 192))
    assert_size_stride(primals_191, (192, ), (1, ))
    assert_size_stride(primals_193, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_194, (192, ), (1, ))
    assert_size_stride(primals_196, (192, 192, 7, 1), (1344, 1, 192, 192))
    assert_size_stride(primals_197, (192, ), (1, ))
    assert_size_stride(primals_199, (192, 192, 1, 7), (1344, 1, 1344, 192))
    assert_size_stride(primals_200, (192, ), (1, ))
    assert_size_stride(primals_202, (192, 192, 7, 1), (1344, 1, 192, 192))
    assert_size_stride(primals_203, (192, ), (1, ))
    assert_size_stride(primals_205, (192, 192, 1, 7), (1344, 1, 1344, 192))
    assert_size_stride(primals_206, (192, ), (1, ))
    assert_size_stride(primals_208, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_209, (192, ), (1, ))
    assert_size_stride(primals_211, (128, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_212, (128, ), (1, ))
    assert_size_stride(primals_214, (768, 128, 5, 5), (3200, 1, 640, 128))
    assert_size_stride(primals_215, (768, ), (1, ))
    assert_size_stride(primals_219, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_220, (192, ), (1, ))
    assert_size_stride(primals_222, (320, 192, 3, 3), (1728, 1, 576, 192))
    assert_size_stride(primals_223, (320, ), (1, ))
    assert_size_stride(primals_225, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_226, (192, ), (1, ))
    assert_size_stride(primals_228, (192, 192, 1, 7), (1344, 1, 1344, 192))
    assert_size_stride(primals_229, (192, ), (1, ))
    assert_size_stride(primals_231, (192, 192, 7, 1), (1344, 1, 192, 192))
    assert_size_stride(primals_232, (192, ), (1, ))
    assert_size_stride(primals_234, (192, 192, 3, 3), (1728, 1, 576, 192))
    assert_size_stride(primals_235, (192, ), (1, ))
    assert_size_stride(primals_237, (320, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(primals_238, (320, ), (1, ))
    assert_size_stride(primals_240, (384, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(primals_241, (384, ), (1, ))
    assert_size_stride(primals_243, (384, 384, 1, 3), (1152, 1, 1152, 384))
    assert_size_stride(primals_244, (384, ), (1, ))
    assert_size_stride(primals_246, (384, 384, 3, 1), (1152, 1, 384, 384))
    assert_size_stride(primals_247, (384, ), (1, ))
    assert_size_stride(primals_249, (448, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(primals_250, (448, ), (1, ))
    assert_size_stride(primals_252, (384, 448, 3, 3), (4032, 1, 1344, 448))
    assert_size_stride(primals_253, (384, ), (1, ))
    assert_size_stride(primals_255, (384, 384, 1, 3), (1152, 1, 1152, 384))
    assert_size_stride(primals_256, (384, ), (1, ))
    assert_size_stride(primals_258, (384, 384, 3, 1), (1152, 1, 384, 384))
    assert_size_stride(primals_259, (384, ), (1, ))
    assert_size_stride(primals_261, (192, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(primals_262, (192, ), (1, ))
    assert_size_stride(primals_264, (320, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_265, (320, ), (1, ))
    assert_size_stride(primals_267, (384, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_268, (384, ), (1, ))
    assert_size_stride(primals_270, (384, 384, 1, 3), (1152, 1, 1152, 384))
    assert_size_stride(primals_271, (384, ), (1, ))
    assert_size_stride(primals_273, (384, 384, 3, 1), (1152, 1, 384, 384))
    assert_size_stride(primals_274, (384, ), (1, ))
    assert_size_stride(primals_276, (448, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_277, (448, ), (1, ))
    assert_size_stride(primals_279, (384, 448, 3, 3), (4032, 1, 1344, 448))
    assert_size_stride(primals_280, (384, ), (1, ))
    assert_size_stride(primals_282, (384, 384, 1, 3), (1152, 1, 1152, 384))
    assert_size_stride(primals_283, (384, ), (1, ))
    assert_size_stride(primals_285, (384, 384, 3, 1), (1152, 1, 384, 384))
    assert_size_stride(primals_286, (384, ), (1, ))
    assert_size_stride(primals_288, (192, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_289, (192, ), (1, ))
    assert_size_stride(cat, (32, 3, 299, 299), (268203, 1, 897, 3))
    assert_size_stride(convolution, (32, 32, 149, 149), (710432, 1, 4768, 32))
    assert_size_stride(squeeze_1, (32, ), (1, ))
    assert_size_stride(relu, (32, 32, 149, 149), (710432, 1, 4768, 32))
    assert_size_stride(convolution_1, (32, 32, 147, 147), (691488, 1, 4704, 32))
    assert_size_stride(squeeze_4, (32, ), (1, ))
    assert_size_stride(relu_1, (32, 32, 147, 147), (691488, 1, 4704, 32))
    assert_size_stride(convolution_2, (32, 64, 147, 147), (1382976, 1, 9408, 64))
    assert_size_stride(squeeze_7, (64, ), (1, ))
    assert_size_stride(relu_2, (32, 64, 147, 147), (1382976, 1, 9408, 64))
    assert_size_stride(getitem_6, (32, 64, 73, 73), (341056, 1, 4672, 64))
    assert_size_stride(getitem_7, (32, 64, 73, 73), (341056, 1, 4672, 64))
    assert_size_stride(convolution_3, (32, 80, 73, 73), (426320, 1, 5840, 80))
    assert_size_stride(squeeze_10, (80, ), (1, ))
    assert_size_stride(relu_3, (32, 80, 73, 73), (426320, 1, 5840, 80))
    assert_size_stride(convolution_4, (32, 192, 71, 71), (967872, 1, 13632, 192))
    assert_size_stride(squeeze_13, (192, ), (1, ))
    assert_size_stride(relu_4, (32, 192, 71, 71), (967872, 1, 13632, 192))
    assert_size_stride(getitem_12, (32, 192, 35, 35), (235200, 1, 6720, 192))
    assert_size_stride(getitem_13, (32, 192, 35, 35), (235200, 1, 6720, 192))
    assert_size_stride(convolution_5, (32, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(squeeze_16, (64, ), (1, ))
    assert_size_stride(convolution_6, (32, 48, 35, 35), (58800, 1, 1680, 48))
    assert_size_stride(squeeze_19, (48, ), (1, ))
    assert_size_stride(relu_6, (32, 48, 35, 35), (58800, 1, 1680, 48))
    assert_size_stride(convolution_7, (32, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(squeeze_22, (64, ), (1, ))
    assert_size_stride(convolution_8, (32, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(squeeze_25, (64, ), (1, ))
    assert_size_stride(relu_8, (32, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(convolution_9, (32, 96, 35, 35), (117600, 1, 3360, 96))
    assert_size_stride(squeeze_28, (96, ), (1, ))
    assert_size_stride(relu_9, (32, 96, 35, 35), (117600, 1, 3360, 96))
    assert_size_stride(convolution_10, (32, 96, 35, 35), (117600, 1, 3360, 96))
    assert_size_stride(squeeze_31, (96, ), (1, ))
    assert_size_stride(avg_pool2d, (32, 192, 35, 35), (235200, 1, 6720, 192))
    assert_size_stride(convolution_11, (32, 32, 35, 35), (39200, 1, 1120, 32))
    assert_size_stride(squeeze_34, (32, ), (1, ))
    assert_size_stride(cat_1, (32, 256, 35, 35), (313600, 1, 8960, 256))
    assert_size_stride(convolution_12, (32, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(squeeze_37, (64, ), (1, ))
    assert_size_stride(convolution_13, (32, 48, 35, 35), (58800, 1, 1680, 48))
    assert_size_stride(squeeze_40, (48, ), (1, ))
    assert_size_stride(relu_13, (32, 48, 35, 35), (58800, 1, 1680, 48))
    assert_size_stride(convolution_14, (32, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(squeeze_43, (64, ), (1, ))
    assert_size_stride(convolution_15, (32, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(squeeze_46, (64, ), (1, ))
    assert_size_stride(relu_15, (32, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(convolution_16, (32, 96, 35, 35), (117600, 1, 3360, 96))
    assert_size_stride(squeeze_49, (96, ), (1, ))
    assert_size_stride(relu_16, (32, 96, 35, 35), (117600, 1, 3360, 96))
    assert_size_stride(convolution_17, (32, 96, 35, 35), (117600, 1, 3360, 96))
    assert_size_stride(squeeze_52, (96, ), (1, ))
    assert_size_stride(avg_pool2d_1, (32, 256, 35, 35), (313600, 1, 8960, 256))
    assert_size_stride(convolution_18, (32, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(squeeze_55, (64, ), (1, ))
    assert_size_stride(cat_2, (32, 288, 35, 35), (352800, 1, 10080, 288))
    assert_size_stride(convolution_19, (32, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(squeeze_58, (64, ), (1, ))
    assert_size_stride(convolution_20, (32, 48, 35, 35), (58800, 1, 1680, 48))
    assert_size_stride(squeeze_61, (48, ), (1, ))
    assert_size_stride(relu_20, (32, 48, 35, 35), (58800, 1, 1680, 48))
    assert_size_stride(convolution_21, (32, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(squeeze_64, (64, ), (1, ))
    assert_size_stride(convolution_22, (32, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(squeeze_67, (64, ), (1, ))
    assert_size_stride(relu_22, (32, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(convolution_23, (32, 96, 35, 35), (117600, 1, 3360, 96))
    assert_size_stride(squeeze_70, (96, ), (1, ))
    assert_size_stride(relu_23, (32, 96, 35, 35), (117600, 1, 3360, 96))
    assert_size_stride(convolution_24, (32, 96, 35, 35), (117600, 1, 3360, 96))
    assert_size_stride(squeeze_73, (96, ), (1, ))
    assert_size_stride(avg_pool2d_2, (32, 288, 35, 35), (352800, 1, 10080, 288))
    assert_size_stride(convolution_25, (32, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(squeeze_76, (64, ), (1, ))
    assert_size_stride(cat_3, (32, 288, 35, 35), (352800, 1, 10080, 288))
    assert_size_stride(convolution_26, (32, 384, 17, 17), (110976, 1, 6528, 384))
    assert_size_stride(squeeze_79, (384, ), (1, ))
    assert_size_stride(convolution_27, (32, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(squeeze_82, (64, ), (1, ))
    assert_size_stride(relu_27, (32, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(convolution_28, (32, 96, 35, 35), (117600, 1, 3360, 96))
    assert_size_stride(squeeze_85, (96, ), (1, ))
    assert_size_stride(relu_28, (32, 96, 35, 35), (117600, 1, 3360, 96))
    assert_size_stride(convolution_29, (32, 96, 17, 17), (27744, 1, 1632, 96))
    assert_size_stride(squeeze_88, (96, ), (1, ))
    assert_size_stride(getitem_65, (32, 288, 17, 17), (83232, 1, 4896, 288))
    assert_size_stride(cat_4, (32, 768, 17, 17), (221952, 1, 13056, 768))
    assert_size_stride(convolution_30, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_91, (192, ), (1, ))
    assert_size_stride(convolution_31, (32, 128, 17, 17), (36992, 1, 2176, 128))
    assert_size_stride(squeeze_94, (128, ), (1, ))
    assert_size_stride(relu_31, (32, 128, 17, 17), (36992, 1, 2176, 128))
    assert_size_stride(convolution_32, (32, 128, 17, 17), (36992, 1, 2176, 128))
    assert_size_stride(squeeze_97, (128, ), (1, ))
    assert_size_stride(relu_32, (32, 128, 17, 17), (36992, 1, 2176, 128))
    assert_size_stride(convolution_33, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_100, (192, ), (1, ))
    assert_size_stride(convolution_34, (32, 128, 17, 17), (36992, 1, 2176, 128))
    assert_size_stride(squeeze_103, (128, ), (1, ))
    assert_size_stride(relu_34, (32, 128, 17, 17), (36992, 1, 2176, 128))
    assert_size_stride(convolution_35, (32, 128, 17, 17), (36992, 1, 2176, 128))
    assert_size_stride(squeeze_106, (128, ), (1, ))
    assert_size_stride(relu_35, (32, 128, 17, 17), (36992, 1, 2176, 128))
    assert_size_stride(convolution_36, (32, 128, 17, 17), (36992, 1, 2176, 128))
    assert_size_stride(squeeze_109, (128, ), (1, ))
    assert_size_stride(relu_36, (32, 128, 17, 17), (36992, 1, 2176, 128))
    assert_size_stride(convolution_37, (32, 128, 17, 17), (36992, 1, 2176, 128))
    assert_size_stride(squeeze_112, (128, ), (1, ))
    assert_size_stride(relu_37, (32, 128, 17, 17), (36992, 1, 2176, 128))
    assert_size_stride(convolution_38, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_115, (192, ), (1, ))
    assert_size_stride(avg_pool2d_3, (32, 768, 17, 17), (221952, 1, 13056, 768))
    assert_size_stride(convolution_39, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_118, (192, ), (1, ))
    assert_size_stride(cat_5, (32, 768, 17, 17), (221952, 1, 13056, 768))
    assert_size_stride(convolution_40, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_121, (192, ), (1, ))
    assert_size_stride(convolution_41, (32, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(squeeze_124, (160, ), (1, ))
    assert_size_stride(relu_41, (32, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(convolution_42, (32, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(squeeze_127, (160, ), (1, ))
    assert_size_stride(relu_42, (32, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(convolution_43, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_130, (192, ), (1, ))
    assert_size_stride(convolution_44, (32, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(squeeze_133, (160, ), (1, ))
    assert_size_stride(relu_44, (32, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(convolution_45, (32, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(squeeze_136, (160, ), (1, ))
    assert_size_stride(relu_45, (32, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(convolution_46, (32, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(squeeze_139, (160, ), (1, ))
    assert_size_stride(relu_46, (32, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(convolution_47, (32, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(squeeze_142, (160, ), (1, ))
    assert_size_stride(relu_47, (32, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(convolution_48, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_145, (192, ), (1, ))
    assert_size_stride(avg_pool2d_4, (32, 768, 17, 17), (221952, 1, 13056, 768))
    assert_size_stride(convolution_49, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_148, (192, ), (1, ))
    assert_size_stride(cat_6, (32, 768, 17, 17), (221952, 1, 13056, 768))
    assert_size_stride(convolution_50, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_151, (192, ), (1, ))
    assert_size_stride(convolution_51, (32, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(squeeze_154, (160, ), (1, ))
    assert_size_stride(relu_51, (32, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(convolution_52, (32, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(squeeze_157, (160, ), (1, ))
    assert_size_stride(relu_52, (32, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(convolution_53, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_160, (192, ), (1, ))
    assert_size_stride(convolution_54, (32, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(squeeze_163, (160, ), (1, ))
    assert_size_stride(relu_54, (32, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(convolution_55, (32, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(squeeze_166, (160, ), (1, ))
    assert_size_stride(relu_55, (32, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(convolution_56, (32, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(squeeze_169, (160, ), (1, ))
    assert_size_stride(relu_56, (32, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(convolution_57, (32, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(squeeze_172, (160, ), (1, ))
    assert_size_stride(relu_57, (32, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(convolution_58, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_175, (192, ), (1, ))
    assert_size_stride(avg_pool2d_5, (32, 768, 17, 17), (221952, 1, 13056, 768))
    assert_size_stride(convolution_59, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_178, (192, ), (1, ))
    assert_size_stride(cat_7, (32, 768, 17, 17), (221952, 1, 13056, 768))
    assert_size_stride(convolution_60, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_181, (192, ), (1, ))
    assert_size_stride(convolution_61, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_184, (192, ), (1, ))
    assert_size_stride(relu_61, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(convolution_62, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_187, (192, ), (1, ))
    assert_size_stride(relu_62, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(convolution_63, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_190, (192, ), (1, ))
    assert_size_stride(convolution_64, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_193, (192, ), (1, ))
    assert_size_stride(relu_64, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(convolution_65, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_196, (192, ), (1, ))
    assert_size_stride(relu_65, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(convolution_66, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_199, (192, ), (1, ))
    assert_size_stride(relu_66, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(convolution_67, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_202, (192, ), (1, ))
    assert_size_stride(relu_67, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(convolution_68, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_205, (192, ), (1, ))
    assert_size_stride(avg_pool2d_6, (32, 768, 17, 17), (221952, 1, 13056, 768))
    assert_size_stride(convolution_69, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_208, (192, ), (1, ))
    assert_size_stride(cat_8, (32, 768, 17, 17), (221952, 1, 13056, 768))
    assert_size_stride(avg_pool2d_7, (32, 768, 5, 5), (19200, 1, 3840, 768))
    assert_size_stride(convolution_70, (32, 128, 5, 5), (3200, 1, 640, 128))
    assert_size_stride(squeeze_211, (128, ), (1, ))
    assert_size_stride(relu_70, (32, 128, 5, 5), (3200, 1, 640, 128))
    assert_size_stride(convolution_71, (32, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(squeeze_214, (768, ), (1, ))
    assert_size_stride(view, (32, 768), (768, 1))
    assert_size_stride(convolution_72, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_217, (192, ), (1, ))
    assert_size_stride(relu_72, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(convolution_73, (32, 320, 8, 8), (20480, 1, 2560, 320))
    assert_size_stride(squeeze_220, (320, ), (1, ))
    assert_size_stride(convolution_74, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_223, (192, ), (1, ))
    assert_size_stride(relu_74, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(convolution_75, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_226, (192, ), (1, ))
    assert_size_stride(relu_75, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(convolution_76, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_229, (192, ), (1, ))
    assert_size_stride(relu_76, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(convolution_77, (32, 192, 8, 8), (12288, 1, 1536, 192))
    assert_size_stride(squeeze_232, (192, ), (1, ))
    assert_size_stride(getitem_163, (32, 768, 8, 8), (49152, 1, 6144, 768))
    assert_size_stride(cat_9, (32, 1280, 8, 8), (81920, 1, 10240, 1280))
    assert_size_stride(convolution_78, (32, 320, 8, 8), (20480, 1, 2560, 320))
    assert_size_stride(squeeze_235, (320, ), (1, ))
    assert_size_stride(convolution_79, (32, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(squeeze_238, (384, ), (1, ))
    assert_size_stride(relu_79, (32, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(convolution_80, (32, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(squeeze_241, (384, ), (1, ))
    assert_size_stride(convolution_81, (32, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(squeeze_244, (384, ), (1, ))
    assert_size_stride(convolution_82, (32, 448, 8, 8), (28672, 1, 3584, 448))
    assert_size_stride(squeeze_247, (448, ), (1, ))
    assert_size_stride(relu_82, (32, 448, 8, 8), (28672, 1, 3584, 448))
    assert_size_stride(convolution_83, (32, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(squeeze_250, (384, ), (1, ))
    assert_size_stride(relu_83, (32, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(convolution_84, (32, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(squeeze_253, (384, ), (1, ))
    assert_size_stride(convolution_85, (32, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(squeeze_256, (384, ), (1, ))
    assert_size_stride(avg_pool2d_8, (32, 1280, 8, 8), (81920, 1, 10240, 1280))
    assert_size_stride(convolution_86, (32, 192, 8, 8), (12288, 1, 1536, 192))
    assert_size_stride(squeeze_259, (192, ), (1, ))
    assert_size_stride(cat_12, (32, 2048, 8, 8), (131072, 1, 16384, 2048))
    assert_size_stride(convolution_87, (32, 320, 8, 8), (20480, 1, 2560, 320))
    assert_size_stride(squeeze_262, (320, ), (1, ))
    assert_size_stride(convolution_88, (32, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(squeeze_265, (384, ), (1, ))
    assert_size_stride(relu_88, (32, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(convolution_89, (32, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(squeeze_268, (384, ), (1, ))
    assert_size_stride(convolution_90, (32, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(squeeze_271, (384, ), (1, ))
    assert_size_stride(convolution_91, (32, 448, 8, 8), (28672, 1, 3584, 448))
    assert_size_stride(squeeze_274, (448, ), (1, ))
    assert_size_stride(relu_91, (32, 448, 8, 8), (28672, 1, 3584, 448))
    assert_size_stride(convolution_92, (32, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(squeeze_277, (384, ), (1, ))
    assert_size_stride(relu_92, (32, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(convolution_93, (32, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(squeeze_280, (384, ), (1, ))
    assert_size_stride(convolution_94, (32, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(squeeze_283, (384, ), (1, ))
    assert_size_stride(avg_pool2d_9, (32, 2048, 8, 8), (131072, 1, 16384, 2048))
    assert_size_stride(convolution_95, (32, 192, 8, 8), (12288, 1, 1536, 192))
    assert_size_stride(squeeze_286, (192, ), (1, ))
    assert_size_stride(gt, (32, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(view_1, (32, 2048), (2048, 1))
    assert_size_stride(permute_2, (1000, 2048), (2048, 1))
    assert_size_stride(le, (32, 192, 8, 8), (12288, 1, 1536, 192))
    assert_size_stride(unsqueeze_389, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(le_1, (32, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(unsqueeze_401, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(le_2, (32, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(unsqueeze_413, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_425, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_437, (1, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(le_5, (32, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(unsqueeze_449, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(le_6, (32, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(unsqueeze_461, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_473, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(le_8, (32, 320, 8, 8), (20480, 1, 2560, 320))
    assert_size_stride(unsqueeze_485, (1, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(le_9, (32, 192, 8, 8), (12288, 1, 1536, 192))
    assert_size_stride(unsqueeze_497, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(le_10, (32, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(unsqueeze_509, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(le_11, (32, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(unsqueeze_521, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_533, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_545, (1, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(le_14, (32, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(unsqueeze_557, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(le_15, (32, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(unsqueeze_569, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_581, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(le_17, (32, 320, 8, 8), (20480, 1, 2560, 320))
    assert_size_stride(unsqueeze_593, (1, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(le_18, (32, 192, 8, 8), (12288, 1, 1536, 192))
    assert_size_stride(unsqueeze_605, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_617, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_629, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_641, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(le_22, (32, 320, 8, 8), (20480, 1, 2560, 320))
    assert_size_stride(unsqueeze_653, (1, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(unsqueeze_665, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(permute_6, (1000, 768), (768, 1))
    assert_size_stride(le_24, (32, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(unsqueeze_677, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_689, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(le_26, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(unsqueeze_701, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(le_27, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(unsqueeze_713, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_725, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_737, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_749, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_761, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(le_32, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(unsqueeze_773, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_785, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_797, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(le_35, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(unsqueeze_809, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(le_36, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(unsqueeze_821, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(le_37, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(unsqueeze_833, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_845, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_857, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_869, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_881, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(le_42, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(unsqueeze_893, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_905, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_917, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(le_45, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(unsqueeze_929, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(le_46, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(unsqueeze_941, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(le_47, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(unsqueeze_953, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_965, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_977, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_989, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_1001, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(le_52, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(unsqueeze_1013, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_1025, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_1037, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(le_55, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(unsqueeze_1049, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(le_56, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(unsqueeze_1061, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(le_57, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(unsqueeze_1073, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_1085, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1097, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1109, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1121, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(le_62, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(unsqueeze_1133, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_1145, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1157, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(le_65, (32, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(unsqueeze_1169, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(le_66, (32, 96, 17, 17), (27744, 1, 1632, 96))
    assert_size_stride(unsqueeze_1181, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(unsqueeze_1193, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(unsqueeze_1205, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_69, (32, 384, 17, 17), (110976, 1, 6528, 384))
    assert_size_stride(unsqueeze_1217, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(le_70, (32, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(unsqueeze_1229, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_71, (32, 96, 35, 35), (117600, 1, 3360, 96))
    assert_size_stride(unsqueeze_1241, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(unsqueeze_1253, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(unsqueeze_1265, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_74, (32, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(unsqueeze_1277, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_1289, (1, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(le_76, (32, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(unsqueeze_1301, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_77, (32, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(unsqueeze_1313, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_78, (32, 96, 35, 35), (117600, 1, 3360, 96))
    assert_size_stride(unsqueeze_1325, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(unsqueeze_1337, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(unsqueeze_1349, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_81, (32, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(unsqueeze_1361, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_1373, (1, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(le_83, (32, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(unsqueeze_1385, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_84, (32, 32, 35, 35), (39200, 1, 1120, 32))
    assert_size_stride(unsqueeze_1397, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(le_85, (32, 96, 35, 35), (117600, 1, 3360, 96))
    assert_size_stride(unsqueeze_1409, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(unsqueeze_1421, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(unsqueeze_1433, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_88, (32, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(unsqueeze_1445, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_1457, (1, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(le_90, (32, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(unsqueeze_1469, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_1481, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_1493, (1, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(unsqueeze_1505, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_1517, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(unsqueeze_1529, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(tangents_1, (32, 1000), (1000, 1))
    assert_size_stride(tangents_2, (32, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32, 2048), (2048, 1), torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_2, out=buf0)
        del permute_2
        buf1 = empty_strided_cuda((1000, 2048), (2048, 1), torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 32), (1, 1000), 0), view_1, out=buf1)
        del view_1
        buf2 = empty_strided_cuda((1, 1000), (1000, 1), torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_0.run(tangents_1, buf2, 1000, 32, grid=grid(1000), stream=stream0)
        del tangents_1
        buf3 = empty_strided_cuda((192, 16), (1, 192), torch.float32)
        buf5 = empty_strided_cuda((192, 16), (1, 192), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_1.run(le, buf0, gt, convolution_95, unsqueeze_389, buf3, buf5, 3072, 128, grid=grid(3072), stream=stream0)
        buf4 = empty_strided_cuda((192, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_2.run(buf3, buf4, 192, 16, grid=grid(192), stream=stream0)
        buf6 = empty_strided_cuda((192, ), (1, ), torch.float32)
        buf8 = empty_strided_cuda((192, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_3.run(buf5, squeeze_286, buf6, buf8, 192, 16, grid=grid(192), stream=stream0)
        buf7 = empty_strided_cuda((32, 192, 8, 8), (12288, 1, 1536, 192), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_4.run(le, buf0, gt, convolution_95, unsqueeze_389, buf6, squeeze_286, buf4, primals_289, buf7, 393216, grid=grid(393216), stream=stream0)
        del convolution_95
        del le
        del primals_289
        del squeeze_286
        del unsqueeze_389
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf9 = aten.convolution_backward.default(buf7, avg_pool2d_9, primals_288, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del avg_pool2d_9
        del primals_288
        buf10 = buf9[0]
        buf11 = buf9[1]
        del buf9
        buf12 = empty_strided_cuda((32, 2048, 8, 8), (131072, 64, 8, 1), torch.float32)
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]
        triton_poi_fused_avg_pool2d_backward_5.run(buf10, buf12, 65536, 64, grid=grid(65536, 64), stream=stream0)
        del buf10
        buf13 = empty_strided_cuda((384, 16), (1, 384), torch.float32)
        buf15 = empty_strided_cuda((384, 16), (1, 384), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_6.run(le_1, buf0, gt, convolution_94, unsqueeze_401, buf13, buf15, 6144, 128, grid=grid(6144), stream=stream0)
        buf14 = empty_strided_cuda((384, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf13, buf14, 384, 16, grid=grid(384), stream=stream0)
        buf16 = empty_strided_cuda((384, ), (1, ), torch.float32)
        buf18 = empty_strided_cuda((384, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_8.run(buf15, squeeze_283, buf16, buf18, 384, 16, grid=grid(384), stream=stream0)
        buf17 = empty_strided_cuda((32, 384, 8, 8), (24576, 1, 3072, 384), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_9.run(le_1, buf0, gt, convolution_94, unsqueeze_401, buf16, squeeze_283, buf14, primals_286, buf17, 786432, grid=grid(786432), stream=stream0)
        del convolution_94
        del le_1
        del primals_286
        del squeeze_283
        del unsqueeze_401
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf19 = aten.convolution_backward.default(buf17, relu_92, primals_285, [0], [1, 1], [1, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_285
        buf20 = buf19[0]
        buf21 = buf19[1]
        del buf19
        buf22 = buf15; del buf15  # reuse
        buf24 = buf13; del buf13  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_10.run(le_2, buf0, gt, convolution_93, unsqueeze_413, buf22, buf24, 6144, 128, grid=grid(6144), stream=stream0)
        buf23 = buf16; del buf16  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf22, buf23, 384, 16, grid=grid(384), stream=stream0)
        buf25 = empty_strided_cuda((384, ), (1, ), torch.float32)
        buf27 = empty_strided_cuda((384, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_8.run(buf24, squeeze_280, buf25, buf27, 384, 16, grid=grid(384), stream=stream0)
        buf26 = buf17; del buf17  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_11.run(le_2, buf0, gt, convolution_93, unsqueeze_413, buf25, squeeze_280, buf23, primals_283, buf26, 786432, grid=grid(786432), stream=stream0)
        del convolution_93
        del le_2
        del primals_283
        del squeeze_280
        del unsqueeze_413
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf28 = aten.convolution_backward.default(buf26, relu_92, primals_282, [0], [1, 1], [0, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf26
        del primals_282
        buf29 = buf28[0]
        buf30 = buf28[1]
        del buf28
        buf31 = buf24; del buf24  # reuse
        buf33 = buf22; del buf22  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_12.run(relu_92, buf20, buf29, convolution_92, unsqueeze_425, buf31, buf33, 6144, 128, grid=grid(6144), stream=stream0)
        buf32 = buf25; del buf25  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf31, buf32, 384, 16, grid=grid(384), stream=stream0)
        buf34 = empty_strided_cuda((384, ), (1, ), torch.float32)
        buf36 = empty_strided_cuda((384, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_8.run(buf33, squeeze_277, buf34, buf36, 384, 16, grid=grid(384), stream=stream0)
        buf35 = buf20; del buf20  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_13.run(buf35, relu_92, buf29, convolution_92, unsqueeze_425, buf34, squeeze_277, buf32, primals_280, 786432, grid=grid(786432), stream=stream0)
        del buf29
        del convolution_92
        del primals_280
        del relu_92
        del squeeze_277
        del unsqueeze_425
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf37 = aten.convolution_backward.default(buf35, relu_91, primals_279, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_279
        buf38 = buf37[0]
        buf39 = buf37[1]
        del buf37
        buf40 = empty_strided_cuda((448, 16), (1, 448), torch.float32)
        buf42 = empty_strided_cuda((448, 16), (1, 448), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(relu_91, buf38, convolution_91, unsqueeze_437, buf40, buf42, 7168, 128, grid=grid(7168), stream=stream0)
        buf41 = empty_strided_cuda((448, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_15.run(buf40, buf41, 448, 16, grid=grid(448), stream=stream0)
        buf43 = empty_strided_cuda((448, ), (1, ), torch.float32)
        buf44 = empty_strided_cuda((448, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_16.run(buf42, squeeze_274, buf43, buf44, 448, 16, grid=grid(448), stream=stream0)
        buf45 = buf38; del buf38  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_17.run(buf45, relu_91, convolution_91, unsqueeze_437, buf43, squeeze_274, buf41, primals_277, 917504, grid=grid(917504), stream=stream0)
        del convolution_91
        del primals_277
        del relu_91
        del squeeze_274
        del unsqueeze_437
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf46 = aten.convolution_backward.default(buf45, cat_12, primals_276, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf45
        del primals_276
        buf47 = buf46[0]
        buf48 = buf46[1]
        del buf46
        buf49 = buf33; del buf33  # reuse
        buf51 = buf31; del buf31  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_18.run(le_5, buf0, gt, convolution_90, unsqueeze_449, buf49, buf51, 6144, 128, grid=grid(6144), stream=stream0)
        buf50 = buf34; del buf34  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf49, buf50, 384, 16, grid=grid(384), stream=stream0)
        buf52 = empty_strided_cuda((384, ), (1, ), torch.float32)
        buf54 = empty_strided_cuda((384, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_8.run(buf51, squeeze_271, buf52, buf54, 384, 16, grid=grid(384), stream=stream0)
        buf53 = buf35; del buf35  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_19.run(le_5, buf0, gt, convolution_90, unsqueeze_449, buf52, squeeze_271, buf50, primals_274, buf53, 786432, grid=grid(786432), stream=stream0)
        del convolution_90
        del le_5
        del primals_274
        del squeeze_271
        del unsqueeze_449
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf55 = aten.convolution_backward.default(buf53, relu_88, primals_273, [0], [1, 1], [1, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_273
        buf56 = buf55[0]
        buf57 = buf55[1]
        del buf55
        buf58 = buf51; del buf51  # reuse
        buf60 = buf49; del buf49  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_20.run(le_6, buf0, gt, convolution_89, unsqueeze_461, buf58, buf60, 6144, 128, grid=grid(6144), stream=stream0)
        buf59 = buf52; del buf52  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf58, buf59, 384, 16, grid=grid(384), stream=stream0)
        buf61 = empty_strided_cuda((384, ), (1, ), torch.float32)
        buf63 = empty_strided_cuda((384, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_8.run(buf60, squeeze_268, buf61, buf63, 384, 16, grid=grid(384), stream=stream0)
        buf62 = buf53; del buf53  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_21.run(le_6, buf0, gt, convolution_89, unsqueeze_461, buf61, squeeze_268, buf59, primals_271, buf62, 786432, grid=grid(786432), stream=stream0)
        del convolution_89
        del le_6
        del primals_271
        del squeeze_268
        del unsqueeze_461
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf64 = aten.convolution_backward.default(buf62, relu_88, primals_270, [0], [1, 1], [0, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf62
        del primals_270
        buf65 = buf64[0]
        buf66 = buf64[1]
        del buf64
        buf67 = buf60; del buf60  # reuse
        buf69 = buf58; del buf58  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_12.run(relu_88, buf56, buf65, convolution_88, unsqueeze_473, buf67, buf69, 6144, 128, grid=grid(6144), stream=stream0)
        buf68 = buf61; del buf61  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf67, buf68, 384, 16, grid=grid(384), stream=stream0)
        buf70 = empty_strided_cuda((384, ), (1, ), torch.float32)
        buf72 = empty_strided_cuda((384, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_8.run(buf69, squeeze_265, buf70, buf72, 384, 16, grid=grid(384), stream=stream0)
        buf71 = buf56; del buf56  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_13.run(buf71, relu_88, buf65, convolution_88, unsqueeze_473, buf70, squeeze_265, buf68, primals_268, 786432, grid=grid(786432), stream=stream0)
        del buf65
        del convolution_88
        del primals_268
        del relu_88
        del squeeze_265
        del unsqueeze_473
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf73 = aten.convolution_backward.default(buf71, cat_12, primals_267, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_267
        buf74 = buf73[0]
        buf75 = buf73[1]
        del buf73
        buf76 = empty_strided_cuda((320, 16), (1, 320), torch.float32)
        buf78 = empty_strided_cuda((320, 16), (1, 320), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_22.run(le_8, buf0, gt, convolution_87, unsqueeze_485, buf76, buf78, 5120, 128, grid=grid(5120), stream=stream0)
        buf77 = empty_strided_cuda((320, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_23.run(buf76, buf77, 320, 16, grid=grid(320), stream=stream0)
        buf79 = empty_strided_cuda((320, ), (1, ), torch.float32)
        buf81 = empty_strided_cuda((320, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_24.run(buf78, squeeze_262, buf79, buf81, 320, 16, grid=grid(320), stream=stream0)
        buf80 = empty_strided_cuda((32, 320, 8, 8), (20480, 1, 2560, 320), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_25.run(le_8, buf0, gt, convolution_87, unsqueeze_485, buf79, squeeze_262, buf77, primals_265, buf80, 655360, grid=grid(655360), stream=stream0)
        del buf0
        del convolution_87
        del gt
        del le_8
        del primals_265
        del squeeze_262
        del unsqueeze_485
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf82 = aten.convolution_backward.default(buf80, cat_12, primals_264, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_12
        del primals_264
        buf83 = buf82[0]
        buf84 = buf82[1]
        del buf82
        buf85 = buf7; del buf7  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_26.run(le_9, buf12, buf47, buf74, buf83, buf85, 2048, 192, grid=grid(2048, 192), stream=stream0)
        del le_9
        buf86 = buf5; del buf5  # reuse
        buf88 = buf3; del buf3  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_27.run(buf85, convolution_86, unsqueeze_497, buf86, buf88, 3072, 128, grid=grid(3072), stream=stream0)
        buf87 = buf6; del buf6  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_2.run(buf86, buf87, 192, 16, grid=grid(192), stream=stream0)
        buf89 = empty_strided_cuda((192, ), (1, ), torch.float32)
        buf90 = empty_strided_cuda((192, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_3.run(buf88, squeeze_259, buf89, buf90, 192, 16, grid=grid(192), stream=stream0)
        buf91 = buf85; del buf85  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_28.run(buf91, convolution_86, unsqueeze_497, buf89, squeeze_259, buf87, primals_262, 393216, grid=grid(393216), stream=stream0)
        del convolution_86
        del primals_262
        del squeeze_259
        del unsqueeze_497
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf92 = aten.convolution_backward.default(buf91, avg_pool2d_8, primals_261, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del avg_pool2d_8
        del primals_261
        buf93 = buf92[0]
        buf94 = buf92[1]
        del buf92
        buf95 = empty_strided_cuda((32, 1280, 8, 8), (81920, 64, 8, 1), torch.float32)
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]
        triton_poi_fused_avg_pool2d_backward_29.run(buf93, buf95, 40960, 64, grid=grid(40960, 64), stream=stream0)
        del buf93
        buf96 = buf71; del buf71  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_30.run(le_10, buf12, buf47, buf74, buf83, buf96, 2048, 384, grid=grid(2048, 384), stream=stream0)
        del le_10
        buf97 = buf69; del buf69  # reuse
        buf99 = buf67; del buf67  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_31.run(buf96, convolution_85, unsqueeze_509, buf97, buf99, 6144, 128, grid=grid(6144), stream=stream0)
        buf98 = buf70; del buf70  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf97, buf98, 384, 16, grid=grid(384), stream=stream0)
        buf100 = empty_strided_cuda((384, ), (1, ), torch.float32)
        buf101 = empty_strided_cuda((384, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_8.run(buf99, squeeze_256, buf100, buf101, 384, 16, grid=grid(384), stream=stream0)
        buf102 = buf96; del buf96  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_32.run(buf102, convolution_85, unsqueeze_509, buf100, squeeze_256, buf98, primals_259, 786432, grid=grid(786432), stream=stream0)
        del convolution_85
        del primals_259
        del squeeze_256
        del unsqueeze_509
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf103 = aten.convolution_backward.default(buf102, relu_83, primals_258, [0], [1, 1], [1, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_258
        buf104 = buf103[0]
        buf105 = buf103[1]
        del buf103
        buf106 = buf102; del buf102  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_33.run(le_11, buf12, buf47, buf74, buf83, buf106, 2048, 384, grid=grid(2048, 384), stream=stream0)
        del le_11
        buf107 = buf99; del buf99  # reuse
        buf109 = buf97; del buf97  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_31.run(buf106, convolution_84, unsqueeze_521, buf107, buf109, 6144, 128, grid=grid(6144), stream=stream0)
        buf108 = buf100; del buf100  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf107, buf108, 384, 16, grid=grid(384), stream=stream0)
        buf110 = empty_strided_cuda((384, ), (1, ), torch.float32)
        buf111 = empty_strided_cuda((384, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_8.run(buf109, squeeze_253, buf110, buf111, 384, 16, grid=grid(384), stream=stream0)
        buf112 = buf106; del buf106  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_32.run(buf112, convolution_84, unsqueeze_521, buf110, squeeze_253, buf108, primals_256, 786432, grid=grid(786432), stream=stream0)
        del convolution_84
        del primals_256
        del squeeze_253
        del unsqueeze_521
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf113 = aten.convolution_backward.default(buf112, relu_83, primals_255, [0], [1, 1], [0, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf112
        del primals_255
        buf114 = buf113[0]
        buf115 = buf113[1]
        del buf113
        buf116 = buf109; del buf109  # reuse
        buf118 = buf107; del buf107  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_12.run(relu_83, buf104, buf114, convolution_83, unsqueeze_533, buf116, buf118, 6144, 128, grid=grid(6144), stream=stream0)
        buf117 = buf110; del buf110  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf116, buf117, 384, 16, grid=grid(384), stream=stream0)
        buf119 = empty_strided_cuda((384, ), (1, ), torch.float32)
        buf121 = empty_strided_cuda((384, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_8.run(buf118, squeeze_250, buf119, buf121, 384, 16, grid=grid(384), stream=stream0)
        buf120 = buf104; del buf104  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_13.run(buf120, relu_83, buf114, convolution_83, unsqueeze_533, buf119, squeeze_250, buf117, primals_253, 786432, grid=grid(786432), stream=stream0)
        del buf114
        del convolution_83
        del primals_253
        del relu_83
        del squeeze_250
        del unsqueeze_533
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf122 = aten.convolution_backward.default(buf120, relu_82, primals_252, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_252
        buf123 = buf122[0]
        buf124 = buf122[1]
        del buf122
        buf125 = buf42; del buf42  # reuse
        buf127 = buf40; del buf40  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(relu_82, buf123, convolution_82, unsqueeze_545, buf125, buf127, 7168, 128, grid=grid(7168), stream=stream0)
        buf126 = buf43; del buf43  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_15.run(buf125, buf126, 448, 16, grid=grid(448), stream=stream0)
        del buf125
        buf128 = empty_strided_cuda((448, ), (1, ), torch.float32)
        buf129 = empty_strided_cuda((448, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_16.run(buf127, squeeze_247, buf128, buf129, 448, 16, grid=grid(448), stream=stream0)
        del buf127
        buf130 = buf123; del buf123  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_17.run(buf130, relu_82, convolution_82, unsqueeze_545, buf128, squeeze_247, buf126, primals_250, 917504, grid=grid(917504), stream=stream0)
        del buf128
        del convolution_82
        del primals_250
        del relu_82
        del squeeze_247
        del unsqueeze_545
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf131 = aten.convolution_backward.default(buf130, cat_9, primals_249, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf130
        del primals_249
        buf132 = buf131[0]
        buf133 = buf131[1]
        del buf131
        buf134 = buf120; del buf120  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_34.run(le_14, buf12, buf47, buf74, buf83, buf134, 2048, 384, grid=grid(2048, 384), stream=stream0)
        del le_14
        buf135 = buf118; del buf118  # reuse
        buf137 = buf116; del buf116  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_31.run(buf134, convolution_81, unsqueeze_557, buf135, buf137, 6144, 128, grid=grid(6144), stream=stream0)
        buf136 = buf119; del buf119  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf135, buf136, 384, 16, grid=grid(384), stream=stream0)
        buf138 = empty_strided_cuda((384, ), (1, ), torch.float32)
        buf139 = empty_strided_cuda((384, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_8.run(buf137, squeeze_244, buf138, buf139, 384, 16, grid=grid(384), stream=stream0)
        buf140 = buf134; del buf134  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_32.run(buf140, convolution_81, unsqueeze_557, buf138, squeeze_244, buf136, primals_247, 786432, grid=grid(786432), stream=stream0)
        del convolution_81
        del primals_247
        del squeeze_244
        del unsqueeze_557
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf141 = aten.convolution_backward.default(buf140, relu_79, primals_246, [0], [1, 1], [1, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_246
        buf142 = buf141[0]
        buf143 = buf141[1]
        del buf141
        buf144 = buf140; del buf140  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_35.run(le_15, buf12, buf47, buf74, buf83, buf144, 2048, 384, grid=grid(2048, 384), stream=stream0)
        del le_15
        buf145 = buf137; del buf137  # reuse
        buf147 = buf135; del buf135  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_31.run(buf144, convolution_80, unsqueeze_569, buf145, buf147, 6144, 128, grid=grid(6144), stream=stream0)
        buf146 = buf138; del buf138  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf145, buf146, 384, 16, grid=grid(384), stream=stream0)
        buf148 = empty_strided_cuda((384, ), (1, ), torch.float32)
        buf149 = empty_strided_cuda((384, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_8.run(buf147, squeeze_241, buf148, buf149, 384, 16, grid=grid(384), stream=stream0)
        buf150 = buf144; del buf144  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_32.run(buf150, convolution_80, unsqueeze_569, buf148, squeeze_241, buf146, primals_244, 786432, grid=grid(786432), stream=stream0)
        del convolution_80
        del primals_244
        del squeeze_241
        del unsqueeze_569
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf151 = aten.convolution_backward.default(buf150, relu_79, primals_243, [0], [1, 1], [0, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf150
        del primals_243
        buf152 = buf151[0]
        buf153 = buf151[1]
        del buf151
        buf154 = buf147; del buf147  # reuse
        buf156 = buf145; del buf145  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_12.run(relu_79, buf142, buf152, convolution_79, unsqueeze_581, buf154, buf156, 6144, 128, grid=grid(6144), stream=stream0)
        buf155 = buf148; del buf148  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf154, buf155, 384, 16, grid=grid(384), stream=stream0)
        del buf154
        buf157 = empty_strided_cuda((384, ), (1, ), torch.float32)
        buf159 = empty_strided_cuda((384, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_8.run(buf156, squeeze_238, buf157, buf159, 384, 16, grid=grid(384), stream=stream0)
        del buf156
        buf158 = buf142; del buf142  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_13.run(buf158, relu_79, buf152, convolution_79, unsqueeze_581, buf157, squeeze_238, buf155, primals_241, 786432, grid=grid(786432), stream=stream0)
        del buf152
        del convolution_79
        del primals_241
        del relu_79
        del squeeze_238
        del unsqueeze_581
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf160 = aten.convolution_backward.default(buf158, cat_9, primals_240, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf158
        del primals_240
        buf161 = buf160[0]
        buf162 = buf160[1]
        del buf160
        buf163 = buf80; del buf80  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_36.run(le_17, buf12, buf47, buf74, buf83, buf163, 2048, 320, grid=grid(2048, 320), stream=stream0)
        del buf12
        del buf47
        del buf74
        del buf83
        del le_17
        buf164 = buf78; del buf78  # reuse
        buf166 = buf76; del buf76  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_37.run(buf163, convolution_78, unsqueeze_593, buf164, buf166, 5120, 128, grid=grid(5120), stream=stream0)
        buf165 = buf79; del buf79  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_23.run(buf164, buf165, 320, 16, grid=grid(320), stream=stream0)
        buf167 = empty_strided_cuda((320, ), (1, ), torch.float32)
        buf168 = empty_strided_cuda((320, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_24.run(buf166, squeeze_235, buf167, buf168, 320, 16, grid=grid(320), stream=stream0)
        buf169 = buf163; del buf163  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_38.run(buf169, convolution_78, unsqueeze_593, buf167, squeeze_235, buf165, primals_238, 655360, grid=grid(655360), stream=stream0)
        del convolution_78
        del primals_238
        del squeeze_235
        del unsqueeze_593
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf170 = aten.convolution_backward.default(buf169, cat_9, primals_237, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_9
        del primals_237
        buf171 = buf170[0]
        buf172 = buf170[1]
        del buf170
        buf173 = buf95; del buf95  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_39.run(buf173, buf132, buf161, buf171, 2048, 1280, grid=grid(2048, 1280), stream=stream0)
        del buf132
        del buf161
        del buf171
        # Source Nodes: [], Original ATen: [aten.max_pool2d_with_indices_backward]
        buf174 = aten.max_pool2d_with_indices_backward.default(reinterpret_tensor(buf173, (32, 768, 8, 8), (81920, 64, 8, 1), 32768), cat_8, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_163)
        del getitem_163
        buf175 = buf174
        del buf174
        buf176 = buf88; del buf88  # reuse
        buf178 = buf86; del buf86  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_40.run(le_18, buf173, convolution_77, unsqueeze_605, buf176, buf178, 3072, 128, grid=grid(3072), stream=stream0)
        buf177 = buf89; del buf89  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_2.run(buf176, buf177, 192, 16, grid=grid(192), stream=stream0)
        del buf176
        buf179 = empty_strided_cuda((192, ), (1, ), torch.float32)
        buf180 = empty_strided_cuda((192, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_3.run(buf178, squeeze_232, buf179, buf180, 192, 16, grid=grid(192), stream=stream0)
        del buf178
        buf181 = buf91; del buf91  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_41.run(le_18, buf173, convolution_77, unsqueeze_605, buf179, squeeze_232, buf177, primals_235, buf181, 2048, 192, grid=grid(2048, 192), stream=stream0)
        del convolution_77
        del le_18
        del primals_235
        del squeeze_232
        del unsqueeze_605
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf182 = aten.convolution_backward.default(buf181, relu_76, primals_234, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf181
        del primals_234
        buf183 = buf182[0]
        buf184 = buf182[1]
        del buf182
        buf185 = empty_strided_cuda((192, 73), (1, 192), torch.float32)
        buf187 = empty_strided_cuda((192, 73), (1, 192), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_42.run(relu_76, buf183, convolution_76, unsqueeze_617, buf185, buf187, 14016, 127, grid=grid(14016), stream=stream0)
        buf186 = buf179; del buf179  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_43.run(buf185, buf186, 192, 73, grid=grid(192), stream=stream0)
        buf188 = empty_strided_cuda((192, ), (1, ), torch.float32)
        buf189 = empty_strided_cuda((192, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_44.run(buf187, squeeze_229, buf188, buf189, 192, 73, grid=grid(192), stream=stream0)
        buf190 = buf183; del buf183  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_45.run(buf190, relu_76, convolution_76, unsqueeze_617, buf188, squeeze_229, buf186, primals_232, 1775616, grid=grid(1775616), stream=stream0)
        del convolution_76
        del primals_232
        del relu_76
        del squeeze_229
        del unsqueeze_617
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf191 = aten.convolution_backward.default(buf190, relu_75, primals_231, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf190
        del primals_231
        buf192 = buf191[0]
        buf193 = buf191[1]
        del buf191
        buf194 = buf187; del buf187  # reuse
        buf196 = buf185; del buf185  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_42.run(relu_75, buf192, convolution_75, unsqueeze_629, buf194, buf196, 14016, 127, grid=grid(14016), stream=stream0)
        buf195 = buf188; del buf188  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_43.run(buf194, buf195, 192, 73, grid=grid(192), stream=stream0)
        buf197 = empty_strided_cuda((192, ), (1, ), torch.float32)
        buf198 = empty_strided_cuda((192, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_44.run(buf196, squeeze_226, buf197, buf198, 192, 73, grid=grid(192), stream=stream0)
        buf199 = buf192; del buf192  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_45.run(buf199, relu_75, convolution_75, unsqueeze_629, buf197, squeeze_226, buf195, primals_229, 1775616, grid=grid(1775616), stream=stream0)
        del convolution_75
        del primals_229
        del relu_75
        del squeeze_226
        del unsqueeze_629
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf200 = aten.convolution_backward.default(buf199, relu_74, primals_228, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf199
        del primals_228
        buf201 = buf200[0]
        buf202 = buf200[1]
        del buf200
        buf203 = buf196; del buf196  # reuse
        buf205 = buf194; del buf194  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_42.run(relu_74, buf201, convolution_74, unsqueeze_641, buf203, buf205, 14016, 127, grid=grid(14016), stream=stream0)
        buf204 = buf197; del buf197  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_43.run(buf203, buf204, 192, 73, grid=grid(192), stream=stream0)
        buf206 = empty_strided_cuda((192, ), (1, ), torch.float32)
        buf207 = empty_strided_cuda((192, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_44.run(buf205, squeeze_223, buf206, buf207, 192, 73, grid=grid(192), stream=stream0)
        buf208 = buf201; del buf201  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_45.run(buf208, relu_74, convolution_74, unsqueeze_641, buf206, squeeze_223, buf204, primals_226, 1775616, grid=grid(1775616), stream=stream0)
        del convolution_74
        del primals_226
        del relu_74
        del squeeze_223
        del unsqueeze_641
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf209 = aten.convolution_backward.default(buf208, cat_8, primals_225, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf208
        del primals_225
        buf210 = buf209[0]
        buf211 = buf209[1]
        del buf209
        buf212 = buf166; del buf166  # reuse
        buf214 = buf164; del buf164  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_46.run(le_22, buf173, convolution_73, unsqueeze_653, buf212, buf214, 5120, 128, grid=grid(5120), stream=stream0)
        buf213 = buf167; del buf167  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_23.run(buf212, buf213, 320, 16, grid=grid(320), stream=stream0)
        del buf212
        buf215 = empty_strided_cuda((320, ), (1, ), torch.float32)
        buf216 = empty_strided_cuda((320, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_24.run(buf214, squeeze_220, buf215, buf216, 320, 16, grid=grid(320), stream=stream0)
        del buf214
        buf217 = buf169; del buf169  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_47.run(le_22, buf173, convolution_73, unsqueeze_653, buf215, squeeze_220, buf213, primals_223, buf217, 2048, 320, grid=grid(2048, 320), stream=stream0)
        del buf173
        del buf215
        del convolution_73
        del le_22
        del primals_223
        del squeeze_220
        del unsqueeze_653
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf218 = aten.convolution_backward.default(buf217, relu_72, primals_222, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf217
        del primals_222
        buf219 = buf218[0]
        buf220 = buf218[1]
        del buf218
        buf221 = buf205; del buf205  # reuse
        buf223 = buf203; del buf203  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_42.run(relu_72, buf219, convolution_72, unsqueeze_665, buf221, buf223, 14016, 127, grid=grid(14016), stream=stream0)
        buf222 = buf206; del buf206  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_43.run(buf221, buf222, 192, 73, grid=grid(192), stream=stream0)
        buf224 = empty_strided_cuda((192, ), (1, ), torch.float32)
        buf225 = empty_strided_cuda((192, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_44.run(buf223, squeeze_217, buf224, buf225, 192, 73, grid=grid(192), stream=stream0)
        buf226 = buf219; del buf219  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_45.run(buf226, relu_72, convolution_72, unsqueeze_665, buf224, squeeze_217, buf222, primals_220, 1775616, grid=grid(1775616), stream=stream0)
        del convolution_72
        del primals_220
        del relu_72
        del squeeze_217
        del unsqueeze_665
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf227 = aten.convolution_backward.default(buf226, cat_8, primals_219, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_8
        del primals_219
        buf228 = buf227[0]
        buf229 = buf227[1]
        del buf227
        buf230 = empty_strided_cuda((32, 768), (768, 1), torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_2, permute_6, out=buf230)
        del permute_6
        buf231 = empty_strided_cuda((1000, 768), (768, 1), torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_2, (1000, 32), (1, 1000), 0), view, out=buf231)
        del view
        buf232 = empty_strided_cuda((1, 1000), (1000, 1), torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_0.run(tangents_2, buf232, 1000, 32, grid=grid(1000), stream=stream0)
        del tangents_2
        buf233 = empty_strided_cuda((768, ), (1, ), torch.float32)
        buf234 = empty_strided_cuda((768, ), (1, ), torch.float32)
        buf235 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_48.run(le_24, buf230, convolution_71, unsqueeze_677, squeeze_214, buf233, buf234, buf235, 768, 32, grid=grid(768), stream=stream0)
        buf236 = reinterpret_tensor(buf230, (32, 768, 1, 1), (768, 1, 1, 1), 0); del buf230  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_49.run(buf236, le_24, convolution_71, unsqueeze_677, buf234, squeeze_214, buf233, primals_215, 24576, grid=grid(24576), stream=stream0)
        del buf234
        del convolution_71
        del le_24
        del primals_215
        del squeeze_214
        del unsqueeze_677
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        buf237 = aten.convolution_backward.default(buf236, relu_70, primals_214, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf236
        del primals_214
        buf238 = buf237[0]
        buf239 = buf237[1]
        del buf237
        buf240 = empty_strided_cuda((128, 7), (1, 128), torch.float32)
        buf242 = empty_strided_cuda((128, 7), (1, 128), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_50.run(relu_70, buf238, convolution_70, unsqueeze_689, buf240, buf242, 896, 115, grid=grid(896), stream=stream0)
        buf241 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_51.run(buf240, buf241, 128, 7, grid=grid(128), stream=stream0)
        del buf240
        buf243 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf244 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_52.run(buf242, squeeze_211, buf243, buf244, 128, 7, grid=grid(128), stream=stream0)
        del buf242
        buf245 = buf238; del buf238  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_53.run(buf245, relu_70, convolution_70, unsqueeze_689, buf243, squeeze_211, buf241, primals_212, 102400, grid=grid(102400), stream=stream0)
        del convolution_70
        del primals_212
        del relu_70
        del squeeze_211
        del unsqueeze_689
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf246 = aten.convolution_backward.default(buf245, avg_pool2d_7, primals_211, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del avg_pool2d_7
        del buf245
        del primals_211
        buf247 = buf246[0]
        buf248 = buf246[1]
        del buf246
        buf249 = empty_strided_cuda((32, 768, 17, 17), (221952, 289, 17, 1), torch.float32)
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]
        triton_poi_fused_avg_pool2d_backward_54.run(buf247, buf249, 7102464, grid=grid(7102464), stream=stream0)
        del buf247
        buf250 = buf226; del buf226  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_55.run(le_26, buf175, buf210, buf228, buf249, buf250, 9248, 192, grid=grid(9248, 192), stream=stream0)
        del le_26
        buf251 = buf223; del buf223  # reuse
        buf253 = buf221; del buf221  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_56.run(buf250, convolution_69, unsqueeze_701, buf251, buf253, 14016, 127, grid=grid(14016), stream=stream0)
        buf252 = buf224; del buf224  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_43.run(buf251, buf252, 192, 73, grid=grid(192), stream=stream0)
        buf254 = empty_strided_cuda((192, ), (1, ), torch.float32)
        buf255 = empty_strided_cuda((192, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_44.run(buf253, squeeze_208, buf254, buf255, 192, 73, grid=grid(192), stream=stream0)
        buf256 = buf250; del buf250  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_57.run(buf256, convolution_69, unsqueeze_701, buf254, squeeze_208, buf252, primals_209, 1775616, grid=grid(1775616), stream=stream0)
        del convolution_69
        del primals_209
        del squeeze_208
        del unsqueeze_701
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf257 = aten.convolution_backward.default(buf256, avg_pool2d_6, primals_208, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del avg_pool2d_6
        del primals_208
        buf258 = buf257[0]
        buf259 = buf257[1]
        del buf257
        buf260 = empty_strided_cuda((32, 768, 17, 17), (221952, 289, 17, 1), torch.float32)
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]
        triton_poi_fused_avg_pool2d_backward_58.run(buf258, buf260, 24576, 289, grid=grid(24576, 289), stream=stream0)
        del buf258
        buf261 = buf256; del buf256  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_59.run(le_27, buf175, buf210, buf228, buf249, buf261, 9248, 192, grid=grid(9248, 192), stream=stream0)
        del le_27
        buf262 = buf253; del buf253  # reuse
        buf264 = buf251; del buf251  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_56.run(buf261, convolution_68, unsqueeze_713, buf262, buf264, 14016, 127, grid=grid(14016), stream=stream0)
        buf263 = buf254; del buf254  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_43.run(buf262, buf263, 192, 73, grid=grid(192), stream=stream0)
        buf265 = empty_strided_cuda((192, ), (1, ), torch.float32)
        buf266 = empty_strided_cuda((192, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_44.run(buf264, squeeze_205, buf265, buf266, 192, 73, grid=grid(192), stream=stream0)
        buf267 = buf261; del buf261  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_57.run(buf267, convolution_68, unsqueeze_713, buf265, squeeze_205, buf263, primals_206, 1775616, grid=grid(1775616), stream=stream0)
        del convolution_68
        del primals_206
        del squeeze_205
        del unsqueeze_713
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf268 = aten.convolution_backward.default(buf267, relu_67, primals_205, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf267
        del primals_205
        buf269 = buf268[0]
        buf270 = buf268[1]
        del buf268
        buf271 = buf264; del buf264  # reuse
        buf273 = buf262; del buf262  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_42.run(relu_67, buf269, convolution_67, unsqueeze_725, buf271, buf273, 14016, 127, grid=grid(14016), stream=stream0)
        buf272 = buf265; del buf265  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_43.run(buf271, buf272, 192, 73, grid=grid(192), stream=stream0)
        buf274 = empty_strided_cuda((192, ), (1, ), torch.float32)
        buf275 = empty_strided_cuda((192, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_44.run(buf273, squeeze_202, buf274, buf275, 192, 73, grid=grid(192), stream=stream0)
        buf276 = buf269; del buf269  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_45.run(buf276, relu_67, convolution_67, unsqueeze_725, buf274, squeeze_202, buf272, primals_203, 1775616, grid=grid(1775616), stream=stream0)
        del convolution_67
        del primals_203
        del relu_67
        del squeeze_202
        del unsqueeze_725
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf277 = aten.convolution_backward.default(buf276, relu_66, primals_202, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf276
        del primals_202
        buf278 = buf277[0]
        buf279 = buf277[1]
        del buf277
        buf280 = buf273; del buf273  # reuse
        buf282 = buf271; del buf271  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_42.run(relu_66, buf278, convolution_66, unsqueeze_737, buf280, buf282, 14016, 127, grid=grid(14016), stream=stream0)
        buf281 = buf274; del buf274  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_43.run(buf280, buf281, 192, 73, grid=grid(192), stream=stream0)
        buf283 = empty_strided_cuda((192, ), (1, ), torch.float32)
        buf284 = empty_strided_cuda((192, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_44.run(buf282, squeeze_199, buf283, buf284, 192, 73, grid=grid(192), stream=stream0)
        buf285 = buf278; del buf278  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_45.run(buf285, relu_66, convolution_66, unsqueeze_737, buf283, squeeze_199, buf281, primals_200, 1775616, grid=grid(1775616), stream=stream0)
        del convolution_66
        del primals_200
        del relu_66
        del squeeze_199
        del unsqueeze_737
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf286 = aten.convolution_backward.default(buf285, relu_65, primals_199, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf285
        del primals_199
        buf287 = buf286[0]
        buf288 = buf286[1]
        del buf286
        buf289 = buf282; del buf282  # reuse
        buf291 = buf280; del buf280  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_42.run(relu_65, buf287, convolution_65, unsqueeze_749, buf289, buf291, 14016, 127, grid=grid(14016), stream=stream0)
        buf290 = buf283; del buf283  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_43.run(buf289, buf290, 192, 73, grid=grid(192), stream=stream0)
        buf292 = empty_strided_cuda((192, ), (1, ), torch.float32)
        buf293 = empty_strided_cuda((192, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_44.run(buf291, squeeze_196, buf292, buf293, 192, 73, grid=grid(192), stream=stream0)
        buf294 = buf287; del buf287  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_45.run(buf294, relu_65, convolution_65, unsqueeze_749, buf292, squeeze_196, buf290, primals_197, 1775616, grid=grid(1775616), stream=stream0)
        del convolution_65
        del primals_197
        del relu_65
        del squeeze_196
        del unsqueeze_749
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf295 = aten.convolution_backward.default(buf294, relu_64, primals_196, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf294
        del primals_196
        buf296 = buf295[0]
        buf297 = buf295[1]
        del buf295
        buf298 = buf291; del buf291  # reuse
        buf300 = buf289; del buf289  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_42.run(relu_64, buf296, convolution_64, unsqueeze_761, buf298, buf300, 14016, 127, grid=grid(14016), stream=stream0)
        buf299 = buf292; del buf292  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_43.run(buf298, buf299, 192, 73, grid=grid(192), stream=stream0)
        buf301 = empty_strided_cuda((192, ), (1, ), torch.float32)
        buf302 = empty_strided_cuda((192, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_44.run(buf300, squeeze_193, buf301, buf302, 192, 73, grid=grid(192), stream=stream0)
        buf303 = buf296; del buf296  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_45.run(buf303, relu_64, convolution_64, unsqueeze_761, buf301, squeeze_193, buf299, primals_194, 1775616, grid=grid(1775616), stream=stream0)
        del convolution_64
        del primals_194
        del relu_64
        del squeeze_193
        del unsqueeze_761
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf304 = aten.convolution_backward.default(buf303, cat_7, primals_193, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_193
        buf305 = buf304[0]
        buf306 = buf304[1]
        del buf304
        buf307 = buf303; del buf303  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_60.run(le_32, buf175, buf210, buf228, buf249, buf307, 9248, 192, grid=grid(9248, 192), stream=stream0)
        del le_32
        buf308 = buf300; del buf300  # reuse
        buf310 = buf298; del buf298  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_56.run(buf307, convolution_63, unsqueeze_773, buf308, buf310, 14016, 127, grid=grid(14016), stream=stream0)
        buf309 = buf301; del buf301  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_43.run(buf308, buf309, 192, 73, grid=grid(192), stream=stream0)
        buf311 = empty_strided_cuda((192, ), (1, ), torch.float32)
        buf312 = empty_strided_cuda((192, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_44.run(buf310, squeeze_190, buf311, buf312, 192, 73, grid=grid(192), stream=stream0)
        buf313 = buf307; del buf307  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_57.run(buf313, convolution_63, unsqueeze_773, buf311, squeeze_190, buf309, primals_191, 1775616, grid=grid(1775616), stream=stream0)
        del convolution_63
        del primals_191
        del squeeze_190
        del unsqueeze_773
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf314 = aten.convolution_backward.default(buf313, relu_62, primals_190, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf313
        del primals_190
        buf315 = buf314[0]
        buf316 = buf314[1]
        del buf314
        buf317 = buf310; del buf310  # reuse
        buf319 = buf308; del buf308  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_42.run(relu_62, buf315, convolution_62, unsqueeze_785, buf317, buf319, 14016, 127, grid=grid(14016), stream=stream0)
        buf318 = buf311; del buf311  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_43.run(buf317, buf318, 192, 73, grid=grid(192), stream=stream0)
        buf320 = empty_strided_cuda((192, ), (1, ), torch.float32)
        buf321 = empty_strided_cuda((192, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_44.run(buf319, squeeze_187, buf320, buf321, 192, 73, grid=grid(192), stream=stream0)
        buf322 = buf315; del buf315  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_45.run(buf322, relu_62, convolution_62, unsqueeze_785, buf320, squeeze_187, buf318, primals_188, 1775616, grid=grid(1775616), stream=stream0)
        del convolution_62
        del primals_188
        del relu_62
        del squeeze_187
        del unsqueeze_785
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf323 = aten.convolution_backward.default(buf322, relu_61, primals_187, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf322
        del primals_187
        buf324 = buf323[0]
        buf325 = buf323[1]
        del buf323
        buf326 = buf319; del buf319  # reuse
        buf328 = buf317; del buf317  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_42.run(relu_61, buf324, convolution_61, unsqueeze_797, buf326, buf328, 14016, 127, grid=grid(14016), stream=stream0)
        buf327 = buf320; del buf320  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_43.run(buf326, buf327, 192, 73, grid=grid(192), stream=stream0)
        buf329 = empty_strided_cuda((192, ), (1, ), torch.float32)
        buf330 = empty_strided_cuda((192, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_44.run(buf328, squeeze_184, buf329, buf330, 192, 73, grid=grid(192), stream=stream0)
        buf331 = buf324; del buf324  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_45.run(buf331, relu_61, convolution_61, unsqueeze_797, buf329, squeeze_184, buf327, primals_185, 1775616, grid=grid(1775616), stream=stream0)
        del convolution_61
        del primals_185
        del relu_61
        del squeeze_184
        del unsqueeze_797
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf332 = aten.convolution_backward.default(buf331, cat_7, primals_184, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_184
        buf333 = buf332[0]
        buf334 = buf332[1]
        del buf332
        buf335 = buf331; del buf331  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_61.run(le_35, buf175, buf210, buf228, buf249, buf335, 9248, 192, grid=grid(9248, 192), stream=stream0)
        del buf175
        del buf210
        del buf228
        del le_35
        buf336 = buf328; del buf328  # reuse
        buf338 = buf326; del buf326  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_56.run(buf335, convolution_60, unsqueeze_809, buf336, buf338, 14016, 127, grid=grid(14016), stream=stream0)
        buf337 = buf329; del buf329  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_43.run(buf336, buf337, 192, 73, grid=grid(192), stream=stream0)
        buf339 = empty_strided_cuda((192, ), (1, ), torch.float32)
        buf340 = empty_strided_cuda((192, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_44.run(buf338, squeeze_181, buf339, buf340, 192, 73, grid=grid(192), stream=stream0)
        buf341 = buf335; del buf335  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_57.run(buf341, convolution_60, unsqueeze_809, buf339, squeeze_181, buf337, primals_182, 1775616, grid=grid(1775616), stream=stream0)
        del convolution_60
        del primals_182
        del squeeze_181
        del unsqueeze_809
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf342 = aten.convolution_backward.default(buf341, cat_7, primals_181, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_7
        del primals_181
        buf343 = buf342[0]
        buf344 = buf342[1]
        del buf342
        buf345 = buf341; del buf341  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_62.run(le_36, buf260, buf305, buf333, buf343, buf345, 9248, 192, grid=grid(9248, 192), stream=stream0)
        del le_36
        buf346 = buf338; del buf338  # reuse
        buf348 = buf336; del buf336  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_56.run(buf345, convolution_59, unsqueeze_821, buf346, buf348, 14016, 127, grid=grid(14016), stream=stream0)
        buf347 = buf339; del buf339  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_43.run(buf346, buf347, 192, 73, grid=grid(192), stream=stream0)
        buf349 = empty_strided_cuda((192, ), (1, ), torch.float32)
        buf350 = empty_strided_cuda((192, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_44.run(buf348, squeeze_178, buf349, buf350, 192, 73, grid=grid(192), stream=stream0)
        buf351 = buf345; del buf345  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_57.run(buf351, convolution_59, unsqueeze_821, buf349, squeeze_178, buf347, primals_179, 1775616, grid=grid(1775616), stream=stream0)
        del convolution_59
        del primals_179
        del squeeze_178
        del unsqueeze_821
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf352 = aten.convolution_backward.default(buf351, avg_pool2d_5, primals_178, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del avg_pool2d_5
        del primals_178
        buf353 = buf352[0]
        buf354 = buf352[1]
        del buf352
        buf355 = buf249; del buf249  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]
        triton_poi_fused_avg_pool2d_backward_58.run(buf353, buf355, 24576, 289, grid=grid(24576, 289), stream=stream0)
        del buf353
        buf356 = buf351; del buf351  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_63.run(le_37, buf260, buf305, buf333, buf343, buf356, 9248, 192, grid=grid(9248, 192), stream=stream0)
        del le_37
        buf357 = buf348; del buf348  # reuse
        buf359 = buf346; del buf346  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_56.run(buf356, convolution_58, unsqueeze_833, buf357, buf359, 14016, 127, grid=grid(14016), stream=stream0)
        buf358 = buf349; del buf349  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_43.run(buf357, buf358, 192, 73, grid=grid(192), stream=stream0)
        buf360 = empty_strided_cuda((192, ), (1, ), torch.float32)
        buf361 = empty_strided_cuda((192, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_44.run(buf359, squeeze_175, buf360, buf361, 192, 73, grid=grid(192), stream=stream0)
        buf362 = buf356; del buf356  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_57.run(buf362, convolution_58, unsqueeze_833, buf360, squeeze_175, buf358, primals_176, 1775616, grid=grid(1775616), stream=stream0)
        del convolution_58
        del primals_176
        del squeeze_175
        del unsqueeze_833
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf363 = aten.convolution_backward.default(buf362, relu_57, primals_175, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_175
        buf364 = buf363[0]
        buf365 = buf363[1]
        del buf363
        buf366 = empty_strided_cuda((160, 73), (1, 160), torch.float32)
        buf368 = empty_strided_cuda((160, 73), (1, 160), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_64.run(relu_57, buf364, convolution_57, unsqueeze_845, buf366, buf368, 11680, 127, grid=grid(11680), stream=stream0)
        buf367 = empty_strided_cuda((160, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_65.run(buf366, buf367, 160, 73, grid=grid(160), stream=stream0)
        buf369 = empty_strided_cuda((160, ), (1, ), torch.float32)
        buf370 = empty_strided_cuda((160, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_66.run(buf368, squeeze_172, buf369, buf370, 160, 73, grid=grid(160), stream=stream0)
        buf371 = buf364; del buf364  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_67.run(buf371, relu_57, convolution_57, unsqueeze_845, buf369, squeeze_172, buf367, primals_173, 1479680, grid=grid(1479680), stream=stream0)
        del convolution_57
        del primals_173
        del relu_57
        del squeeze_172
        del unsqueeze_845
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf372 = aten.convolution_backward.default(buf371, relu_56, primals_172, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf371
        del primals_172
        buf373 = buf372[0]
        buf374 = buf372[1]
        del buf372
        buf375 = buf368; del buf368  # reuse
        buf377 = buf366; del buf366  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_64.run(relu_56, buf373, convolution_56, unsqueeze_857, buf375, buf377, 11680, 127, grid=grid(11680), stream=stream0)
        buf376 = buf369; del buf369  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_65.run(buf375, buf376, 160, 73, grid=grid(160), stream=stream0)
        buf378 = empty_strided_cuda((160, ), (1, ), torch.float32)
        buf379 = empty_strided_cuda((160, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_66.run(buf377, squeeze_169, buf378, buf379, 160, 73, grid=grid(160), stream=stream0)
        buf380 = buf373; del buf373  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_67.run(buf380, relu_56, convolution_56, unsqueeze_857, buf378, squeeze_169, buf376, primals_170, 1479680, grid=grid(1479680), stream=stream0)
        del convolution_56
        del primals_170
        del relu_56
        del squeeze_169
        del unsqueeze_857
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf381 = aten.convolution_backward.default(buf380, relu_55, primals_169, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf380
        del primals_169
        buf382 = buf381[0]
        buf383 = buf381[1]
        del buf381
        buf384 = buf377; del buf377  # reuse
        buf386 = buf375; del buf375  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_64.run(relu_55, buf382, convolution_55, unsqueeze_869, buf384, buf386, 11680, 127, grid=grid(11680), stream=stream0)
        buf385 = buf378; del buf378  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_65.run(buf384, buf385, 160, 73, grid=grid(160), stream=stream0)
        buf387 = empty_strided_cuda((160, ), (1, ), torch.float32)
        buf388 = empty_strided_cuda((160, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_66.run(buf386, squeeze_166, buf387, buf388, 160, 73, grid=grid(160), stream=stream0)
        buf389 = buf382; del buf382  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_67.run(buf389, relu_55, convolution_55, unsqueeze_869, buf387, squeeze_166, buf385, primals_167, 1479680, grid=grid(1479680), stream=stream0)
        del convolution_55
        del primals_167
        del relu_55
        del squeeze_166
        del unsqueeze_869
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf390 = aten.convolution_backward.default(buf389, relu_54, primals_166, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf389
        del primals_166
        buf391 = buf390[0]
        buf392 = buf390[1]
        del buf390
        buf393 = buf386; del buf386  # reuse
        buf395 = buf384; del buf384  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_64.run(relu_54, buf391, convolution_54, unsqueeze_881, buf393, buf395, 11680, 127, grid=grid(11680), stream=stream0)
        buf394 = buf387; del buf387  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_65.run(buf393, buf394, 160, 73, grid=grid(160), stream=stream0)
        buf396 = empty_strided_cuda((160, ), (1, ), torch.float32)
        buf397 = empty_strided_cuda((160, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_66.run(buf395, squeeze_163, buf396, buf397, 160, 73, grid=grid(160), stream=stream0)
        buf398 = buf391; del buf391  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_67.run(buf398, relu_54, convolution_54, unsqueeze_881, buf396, squeeze_163, buf394, primals_164, 1479680, grid=grid(1479680), stream=stream0)
        del convolution_54
        del primals_164
        del relu_54
        del squeeze_163
        del unsqueeze_881
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf399 = aten.convolution_backward.default(buf398, cat_6, primals_163, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf398
        del primals_163
        buf400 = buf399[0]
        buf401 = buf399[1]
        del buf399
        buf402 = buf362; del buf362  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_68.run(le_42, buf260, buf305, buf333, buf343, buf402, 9248, 192, grid=grid(9248, 192), stream=stream0)
        del le_42
        buf403 = buf359; del buf359  # reuse
        buf405 = buf357; del buf357  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_56.run(buf402, convolution_53, unsqueeze_893, buf403, buf405, 14016, 127, grid=grid(14016), stream=stream0)
        buf404 = buf360; del buf360  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_43.run(buf403, buf404, 192, 73, grid=grid(192), stream=stream0)
        buf406 = empty_strided_cuda((192, ), (1, ), torch.float32)
        buf407 = empty_strided_cuda((192, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_44.run(buf405, squeeze_160, buf406, buf407, 192, 73, grid=grid(192), stream=stream0)
        buf408 = buf402; del buf402  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_57.run(buf408, convolution_53, unsqueeze_893, buf406, squeeze_160, buf404, primals_161, 1775616, grid=grid(1775616), stream=stream0)
        del convolution_53
        del primals_161
        del squeeze_160
        del unsqueeze_893
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf409 = aten.convolution_backward.default(buf408, relu_52, primals_160, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_160
        buf410 = buf409[0]
        buf411 = buf409[1]
        del buf409
        buf412 = buf395; del buf395  # reuse
        buf414 = buf393; del buf393  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_64.run(relu_52, buf410, convolution_52, unsqueeze_905, buf412, buf414, 11680, 127, grid=grid(11680), stream=stream0)
        buf413 = buf396; del buf396  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_65.run(buf412, buf413, 160, 73, grid=grid(160), stream=stream0)
        buf415 = empty_strided_cuda((160, ), (1, ), torch.float32)
        buf416 = empty_strided_cuda((160, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_66.run(buf414, squeeze_157, buf415, buf416, 160, 73, grid=grid(160), stream=stream0)
        buf417 = buf410; del buf410  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_67.run(buf417, relu_52, convolution_52, unsqueeze_905, buf415, squeeze_157, buf413, primals_158, 1479680, grid=grid(1479680), stream=stream0)
        del convolution_52
        del primals_158
        del relu_52
        del squeeze_157
        del unsqueeze_905
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf418 = aten.convolution_backward.default(buf417, relu_51, primals_157, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf417
        del primals_157
        buf419 = buf418[0]
        buf420 = buf418[1]
        del buf418
        buf421 = buf414; del buf414  # reuse
        buf423 = buf412; del buf412  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_64.run(relu_51, buf419, convolution_51, unsqueeze_917, buf421, buf423, 11680, 127, grid=grid(11680), stream=stream0)
        buf422 = buf415; del buf415  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_65.run(buf421, buf422, 160, 73, grid=grid(160), stream=stream0)
        buf424 = empty_strided_cuda((160, ), (1, ), torch.float32)
        buf425 = empty_strided_cuda((160, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_66.run(buf423, squeeze_154, buf424, buf425, 160, 73, grid=grid(160), stream=stream0)
        buf426 = buf419; del buf419  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_67.run(buf426, relu_51, convolution_51, unsqueeze_917, buf424, squeeze_154, buf422, primals_155, 1479680, grid=grid(1479680), stream=stream0)
        del convolution_51
        del primals_155
        del relu_51
        del squeeze_154
        del unsqueeze_917
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf427 = aten.convolution_backward.default(buf426, cat_6, primals_154, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf426
        del primals_154
        buf428 = buf427[0]
        buf429 = buf427[1]
        del buf427
        buf430 = buf408; del buf408  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_69.run(le_45, buf260, buf305, buf333, buf343, buf430, 9248, 192, grid=grid(9248, 192), stream=stream0)
        del buf260
        del buf305
        del buf333
        del le_45
        buf431 = buf405; del buf405  # reuse
        buf433 = buf403; del buf403  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_56.run(buf430, convolution_50, unsqueeze_929, buf431, buf433, 14016, 127, grid=grid(14016), stream=stream0)
        buf432 = buf406; del buf406  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_43.run(buf431, buf432, 192, 73, grid=grid(192), stream=stream0)
        buf434 = empty_strided_cuda((192, ), (1, ), torch.float32)
        buf435 = empty_strided_cuda((192, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_44.run(buf433, squeeze_151, buf434, buf435, 192, 73, grid=grid(192), stream=stream0)
        buf436 = buf430; del buf430  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_57.run(buf436, convolution_50, unsqueeze_929, buf434, squeeze_151, buf432, primals_152, 1775616, grid=grid(1775616), stream=stream0)
        del convolution_50
        del primals_152
        del squeeze_151
        del unsqueeze_929
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf437 = aten.convolution_backward.default(buf436, cat_6, primals_151, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_6
        del primals_151
        buf438 = buf437[0]
        buf439 = buf437[1]
        del buf437
        buf440 = buf436; del buf436  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_62.run(le_46, buf355, buf400, buf428, buf438, buf440, 9248, 192, grid=grid(9248, 192), stream=stream0)
        del le_46
        buf441 = buf433; del buf433  # reuse
        buf443 = buf431; del buf431  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_56.run(buf440, convolution_49, unsqueeze_941, buf441, buf443, 14016, 127, grid=grid(14016), stream=stream0)
        buf442 = buf434; del buf434  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_43.run(buf441, buf442, 192, 73, grid=grid(192), stream=stream0)
        buf444 = empty_strided_cuda((192, ), (1, ), torch.float32)
        buf445 = empty_strided_cuda((192, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_44.run(buf443, squeeze_148, buf444, buf445, 192, 73, grid=grid(192), stream=stream0)
        buf446 = buf440; del buf440  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_57.run(buf446, convolution_49, unsqueeze_941, buf444, squeeze_148, buf442, primals_149, 1775616, grid=grid(1775616), stream=stream0)
        del convolution_49
        del primals_149
        del squeeze_148
        del unsqueeze_941
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf447 = aten.convolution_backward.default(buf446, avg_pool2d_4, primals_148, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del avg_pool2d_4
        del primals_148
        buf448 = buf447[0]
        buf449 = buf447[1]
        del buf447
        buf450 = reinterpret_tensor(buf343, (32, 768, 17, 17), (221952, 289, 17, 1), 0); del buf343  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]
        triton_poi_fused_avg_pool2d_backward_58.run(buf448, buf450, 24576, 289, grid=grid(24576, 289), stream=stream0)
        del buf448
        buf451 = buf446; del buf446  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_63.run(le_47, buf355, buf400, buf428, buf438, buf451, 9248, 192, grid=grid(9248, 192), stream=stream0)
        del le_47
        buf452 = buf443; del buf443  # reuse
        buf454 = buf441; del buf441  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_56.run(buf451, convolution_48, unsqueeze_953, buf452, buf454, 14016, 127, grid=grid(14016), stream=stream0)
        buf453 = buf444; del buf444  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_43.run(buf452, buf453, 192, 73, grid=grid(192), stream=stream0)
        buf455 = empty_strided_cuda((192, ), (1, ), torch.float32)
        buf456 = empty_strided_cuda((192, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_44.run(buf454, squeeze_145, buf455, buf456, 192, 73, grid=grid(192), stream=stream0)
        buf457 = buf451; del buf451  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_57.run(buf457, convolution_48, unsqueeze_953, buf455, squeeze_145, buf453, primals_146, 1775616, grid=grid(1775616), stream=stream0)
        del convolution_48
        del primals_146
        del squeeze_145
        del unsqueeze_953
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf458 = aten.convolution_backward.default(buf457, relu_47, primals_145, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_145
        buf459 = buf458[0]
        buf460 = buf458[1]
        del buf458
        buf461 = buf423; del buf423  # reuse
        buf463 = buf421; del buf421  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_64.run(relu_47, buf459, convolution_47, unsqueeze_965, buf461, buf463, 11680, 127, grid=grid(11680), stream=stream0)
        buf462 = buf424; del buf424  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_65.run(buf461, buf462, 160, 73, grid=grid(160), stream=stream0)
        buf464 = empty_strided_cuda((160, ), (1, ), torch.float32)
        buf465 = empty_strided_cuda((160, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_66.run(buf463, squeeze_142, buf464, buf465, 160, 73, grid=grid(160), stream=stream0)
        buf466 = buf459; del buf459  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_67.run(buf466, relu_47, convolution_47, unsqueeze_965, buf464, squeeze_142, buf462, primals_143, 1479680, grid=grid(1479680), stream=stream0)
        del convolution_47
        del primals_143
        del relu_47
        del squeeze_142
        del unsqueeze_965
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf467 = aten.convolution_backward.default(buf466, relu_46, primals_142, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf466
        del primals_142
        buf468 = buf467[0]
        buf469 = buf467[1]
        del buf467
        buf470 = buf463; del buf463  # reuse
        buf472 = buf461; del buf461  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_64.run(relu_46, buf468, convolution_46, unsqueeze_977, buf470, buf472, 11680, 127, grid=grid(11680), stream=stream0)
        buf471 = buf464; del buf464  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_65.run(buf470, buf471, 160, 73, grid=grid(160), stream=stream0)
        buf473 = empty_strided_cuda((160, ), (1, ), torch.float32)
        buf474 = empty_strided_cuda((160, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_66.run(buf472, squeeze_139, buf473, buf474, 160, 73, grid=grid(160), stream=stream0)
        buf475 = buf468; del buf468  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_67.run(buf475, relu_46, convolution_46, unsqueeze_977, buf473, squeeze_139, buf471, primals_140, 1479680, grid=grid(1479680), stream=stream0)
        del convolution_46
        del primals_140
        del relu_46
        del squeeze_139
        del unsqueeze_977
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf476 = aten.convolution_backward.default(buf475, relu_45, primals_139, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf475
        del primals_139
        buf477 = buf476[0]
        buf478 = buf476[1]
        del buf476
        buf479 = buf472; del buf472  # reuse
        buf481 = buf470; del buf470  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_64.run(relu_45, buf477, convolution_45, unsqueeze_989, buf479, buf481, 11680, 127, grid=grid(11680), stream=stream0)
        buf480 = buf473; del buf473  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_65.run(buf479, buf480, 160, 73, grid=grid(160), stream=stream0)
        buf482 = empty_strided_cuda((160, ), (1, ), torch.float32)
        buf483 = empty_strided_cuda((160, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_66.run(buf481, squeeze_136, buf482, buf483, 160, 73, grid=grid(160), stream=stream0)
        buf484 = buf477; del buf477  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_67.run(buf484, relu_45, convolution_45, unsqueeze_989, buf482, squeeze_136, buf480, primals_137, 1479680, grid=grid(1479680), stream=stream0)
        del convolution_45
        del primals_137
        del relu_45
        del squeeze_136
        del unsqueeze_989
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf485 = aten.convolution_backward.default(buf484, relu_44, primals_136, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf484
        del primals_136
        buf486 = buf485[0]
        buf487 = buf485[1]
        del buf485
        buf488 = buf481; del buf481  # reuse
        buf490 = buf479; del buf479  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_64.run(relu_44, buf486, convolution_44, unsqueeze_1001, buf488, buf490, 11680, 127, grid=grid(11680), stream=stream0)
        buf489 = buf482; del buf482  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_65.run(buf488, buf489, 160, 73, grid=grid(160), stream=stream0)
        buf491 = empty_strided_cuda((160, ), (1, ), torch.float32)
        buf492 = empty_strided_cuda((160, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_66.run(buf490, squeeze_133, buf491, buf492, 160, 73, grid=grid(160), stream=stream0)
        buf493 = buf486; del buf486  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_67.run(buf493, relu_44, convolution_44, unsqueeze_1001, buf491, squeeze_133, buf489, primals_134, 1479680, grid=grid(1479680), stream=stream0)
        del convolution_44
        del primals_134
        del relu_44
        del squeeze_133
        del unsqueeze_1001
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf494 = aten.convolution_backward.default(buf493, cat_5, primals_133, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf493
        del primals_133
        buf495 = buf494[0]
        buf496 = buf494[1]
        del buf494
        buf497 = buf457; del buf457  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_68.run(le_52, buf355, buf400, buf428, buf438, buf497, 9248, 192, grid=grid(9248, 192), stream=stream0)
        del le_52
        buf498 = buf454; del buf454  # reuse
        buf500 = buf452; del buf452  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_56.run(buf497, convolution_43, unsqueeze_1013, buf498, buf500, 14016, 127, grid=grid(14016), stream=stream0)
        buf499 = buf455; del buf455  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_43.run(buf498, buf499, 192, 73, grid=grid(192), stream=stream0)
        buf501 = empty_strided_cuda((192, ), (1, ), torch.float32)
        buf502 = empty_strided_cuda((192, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_44.run(buf500, squeeze_130, buf501, buf502, 192, 73, grid=grid(192), stream=stream0)
        buf503 = buf497; del buf497  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_57.run(buf503, convolution_43, unsqueeze_1013, buf501, squeeze_130, buf499, primals_131, 1775616, grid=grid(1775616), stream=stream0)
        del convolution_43
        del primals_131
        del squeeze_130
        del unsqueeze_1013
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf504 = aten.convolution_backward.default(buf503, relu_42, primals_130, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_130
        buf505 = buf504[0]
        buf506 = buf504[1]
        del buf504
        buf507 = buf490; del buf490  # reuse
        buf509 = buf488; del buf488  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_64.run(relu_42, buf505, convolution_42, unsqueeze_1025, buf507, buf509, 11680, 127, grid=grid(11680), stream=stream0)
        buf508 = buf491; del buf491  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_65.run(buf507, buf508, 160, 73, grid=grid(160), stream=stream0)
        buf510 = empty_strided_cuda((160, ), (1, ), torch.float32)
        buf511 = empty_strided_cuda((160, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_66.run(buf509, squeeze_127, buf510, buf511, 160, 73, grid=grid(160), stream=stream0)
        buf512 = buf505; del buf505  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_67.run(buf512, relu_42, convolution_42, unsqueeze_1025, buf510, squeeze_127, buf508, primals_128, 1479680, grid=grid(1479680), stream=stream0)
        del convolution_42
        del primals_128
        del relu_42
        del squeeze_127
        del unsqueeze_1025
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf513 = aten.convolution_backward.default(buf512, relu_41, primals_127, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf512
        del primals_127
        buf514 = buf513[0]
        buf515 = buf513[1]
        del buf513
        buf516 = buf509; del buf509  # reuse
        buf518 = buf507; del buf507  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_64.run(relu_41, buf514, convolution_41, unsqueeze_1037, buf516, buf518, 11680, 127, grid=grid(11680), stream=stream0)
        buf517 = buf510; del buf510  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_65.run(buf516, buf517, 160, 73, grid=grid(160), stream=stream0)
        del buf516
        buf519 = empty_strided_cuda((160, ), (1, ), torch.float32)
        buf520 = empty_strided_cuda((160, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_66.run(buf518, squeeze_124, buf519, buf520, 160, 73, grid=grid(160), stream=stream0)
        del buf518
        buf521 = buf514; del buf514  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_67.run(buf521, relu_41, convolution_41, unsqueeze_1037, buf519, squeeze_124, buf517, primals_125, 1479680, grid=grid(1479680), stream=stream0)
        del buf519
        del convolution_41
        del primals_125
        del relu_41
        del squeeze_124
        del unsqueeze_1037
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf522 = aten.convolution_backward.default(buf521, cat_5, primals_124, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf521
        del primals_124
        buf523 = buf522[0]
        buf524 = buf522[1]
        del buf522
        buf525 = buf503; del buf503  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_69.run(le_55, buf355, buf400, buf428, buf438, buf525, 9248, 192, grid=grid(9248, 192), stream=stream0)
        del buf355
        del buf400
        del buf428
        del le_55
        buf526 = buf500; del buf500  # reuse
        buf528 = buf498; del buf498  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_56.run(buf525, convolution_40, unsqueeze_1049, buf526, buf528, 14016, 127, grid=grid(14016), stream=stream0)
        buf527 = buf501; del buf501  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_43.run(buf526, buf527, 192, 73, grid=grid(192), stream=stream0)
        buf529 = empty_strided_cuda((192, ), (1, ), torch.float32)
        buf530 = empty_strided_cuda((192, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_44.run(buf528, squeeze_121, buf529, buf530, 192, 73, grid=grid(192), stream=stream0)
        buf531 = buf525; del buf525  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_57.run(buf531, convolution_40, unsqueeze_1049, buf529, squeeze_121, buf527, primals_122, 1775616, grid=grid(1775616), stream=stream0)
        del convolution_40
        del primals_122
        del squeeze_121
        del unsqueeze_1049
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf532 = aten.convolution_backward.default(buf531, cat_5, primals_121, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_5
        del primals_121
        buf533 = buf532[0]
        buf534 = buf532[1]
        del buf532
        buf535 = buf531; del buf531  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_62.run(le_56, buf450, buf495, buf523, buf533, buf535, 9248, 192, grid=grid(9248, 192), stream=stream0)
        del le_56
        buf536 = buf528; del buf528  # reuse
        buf538 = buf526; del buf526  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_56.run(buf535, convolution_39, unsqueeze_1061, buf536, buf538, 14016, 127, grid=grid(14016), stream=stream0)
        buf537 = buf529; del buf529  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_43.run(buf536, buf537, 192, 73, grid=grid(192), stream=stream0)
        buf539 = empty_strided_cuda((192, ), (1, ), torch.float32)
        buf540 = empty_strided_cuda((192, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_44.run(buf538, squeeze_118, buf539, buf540, 192, 73, grid=grid(192), stream=stream0)
        buf541 = buf535; del buf535  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_57.run(buf541, convolution_39, unsqueeze_1061, buf539, squeeze_118, buf537, primals_119, 1775616, grid=grid(1775616), stream=stream0)
        del convolution_39
        del primals_119
        del squeeze_118
        del unsqueeze_1061
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf542 = aten.convolution_backward.default(buf541, avg_pool2d_3, primals_118, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del avg_pool2d_3
        del primals_118
        buf543 = buf542[0]
        buf544 = buf542[1]
        del buf542
        buf545 = reinterpret_tensor(buf438, (32, 768, 17, 17), (221952, 289, 17, 1), 0); del buf438  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]
        triton_poi_fused_avg_pool2d_backward_58.run(buf543, buf545, 24576, 289, grid=grid(24576, 289), stream=stream0)
        del buf543
        buf546 = buf541; del buf541  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_63.run(le_57, buf450, buf495, buf523, buf533, buf546, 9248, 192, grid=grid(9248, 192), stream=stream0)
        del le_57
        buf547 = buf538; del buf538  # reuse
        buf549 = buf536; del buf536  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_56.run(buf546, convolution_38, unsqueeze_1073, buf547, buf549, 14016, 127, grid=grid(14016), stream=stream0)
        buf548 = buf539; del buf539  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_43.run(buf547, buf548, 192, 73, grid=grid(192), stream=stream0)
        buf550 = empty_strided_cuda((192, ), (1, ), torch.float32)
        buf551 = empty_strided_cuda((192, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_44.run(buf549, squeeze_115, buf550, buf551, 192, 73, grid=grid(192), stream=stream0)
        buf552 = buf546; del buf546  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_57.run(buf552, convolution_38, unsqueeze_1073, buf550, squeeze_115, buf548, primals_116, 1775616, grid=grid(1775616), stream=stream0)
        del convolution_38
        del primals_116
        del squeeze_115
        del unsqueeze_1073
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf553 = aten.convolution_backward.default(buf552, relu_37, primals_115, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_115
        buf554 = buf553[0]
        buf555 = buf553[1]
        del buf553
        buf556 = empty_strided_cuda((128, 73), (1, 128), torch.float32)
        buf558 = empty_strided_cuda((128, 73), (1, 128), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_70.run(relu_37, buf554, convolution_37, unsqueeze_1085, buf556, buf558, 9344, 127, grid=grid(9344), stream=stream0)
        buf557 = buf243; del buf243  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_71.run(buf556, buf557, 128, 73, grid=grid(128), stream=stream0)
        buf559 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf560 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_72.run(buf558, squeeze_112, buf559, buf560, 128, 73, grid=grid(128), stream=stream0)
        buf561 = buf554; del buf554  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_73.run(buf561, relu_37, convolution_37, unsqueeze_1085, buf559, squeeze_112, buf557, primals_113, 1183744, grid=grid(1183744), stream=stream0)
        del convolution_37
        del primals_113
        del relu_37
        del squeeze_112
        del unsqueeze_1085
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf562 = aten.convolution_backward.default(buf561, relu_36, primals_112, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf561
        del primals_112
        buf563 = buf562[0]
        buf564 = buf562[1]
        del buf562
        buf565 = buf558; del buf558  # reuse
        buf567 = buf556; del buf556  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_70.run(relu_36, buf563, convolution_36, unsqueeze_1097, buf565, buf567, 9344, 127, grid=grid(9344), stream=stream0)
        buf566 = buf559; del buf559  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_71.run(buf565, buf566, 128, 73, grid=grid(128), stream=stream0)
        buf568 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf569 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_72.run(buf567, squeeze_109, buf568, buf569, 128, 73, grid=grid(128), stream=stream0)
        buf570 = buf563; del buf563  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_73.run(buf570, relu_36, convolution_36, unsqueeze_1097, buf568, squeeze_109, buf566, primals_110, 1183744, grid=grid(1183744), stream=stream0)
        del convolution_36
        del primals_110
        del relu_36
        del squeeze_109
        del unsqueeze_1097
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf571 = aten.convolution_backward.default(buf570, relu_35, primals_109, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf570
        del primals_109
        buf572 = buf571[0]
        buf573 = buf571[1]
        del buf571
        buf574 = buf567; del buf567  # reuse
        buf576 = buf565; del buf565  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_70.run(relu_35, buf572, convolution_35, unsqueeze_1109, buf574, buf576, 9344, 127, grid=grid(9344), stream=stream0)
        buf575 = buf568; del buf568  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_71.run(buf574, buf575, 128, 73, grid=grid(128), stream=stream0)
        buf577 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf578 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_72.run(buf576, squeeze_106, buf577, buf578, 128, 73, grid=grid(128), stream=stream0)
        buf579 = buf572; del buf572  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_73.run(buf579, relu_35, convolution_35, unsqueeze_1109, buf577, squeeze_106, buf575, primals_107, 1183744, grid=grid(1183744), stream=stream0)
        del convolution_35
        del primals_107
        del relu_35
        del squeeze_106
        del unsqueeze_1109
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf580 = aten.convolution_backward.default(buf579, relu_34, primals_106, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf579
        del primals_106
        buf581 = buf580[0]
        buf582 = buf580[1]
        del buf580
        buf583 = buf576; del buf576  # reuse
        buf585 = buf574; del buf574  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_70.run(relu_34, buf581, convolution_34, unsqueeze_1121, buf583, buf585, 9344, 127, grid=grid(9344), stream=stream0)
        buf584 = buf577; del buf577  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_71.run(buf583, buf584, 128, 73, grid=grid(128), stream=stream0)
        buf586 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf587 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_72.run(buf585, squeeze_103, buf586, buf587, 128, 73, grid=grid(128), stream=stream0)
        buf588 = buf581; del buf581  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_73.run(buf588, relu_34, convolution_34, unsqueeze_1121, buf586, squeeze_103, buf584, primals_104, 1183744, grid=grid(1183744), stream=stream0)
        del convolution_34
        del primals_104
        del relu_34
        del squeeze_103
        del unsqueeze_1121
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf589 = aten.convolution_backward.default(buf588, cat_4, primals_103, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf588
        del primals_103
        buf590 = buf589[0]
        buf591 = buf589[1]
        del buf589
        buf592 = buf552; del buf552  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_68.run(le_62, buf450, buf495, buf523, buf533, buf592, 9248, 192, grid=grid(9248, 192), stream=stream0)
        del le_62
        buf593 = buf549; del buf549  # reuse
        buf595 = buf547; del buf547  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_56.run(buf592, convolution_33, unsqueeze_1133, buf593, buf595, 14016, 127, grid=grid(14016), stream=stream0)
        buf594 = buf550; del buf550  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_43.run(buf593, buf594, 192, 73, grid=grid(192), stream=stream0)
        buf596 = empty_strided_cuda((192, ), (1, ), torch.float32)
        buf597 = empty_strided_cuda((192, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_44.run(buf595, squeeze_100, buf596, buf597, 192, 73, grid=grid(192), stream=stream0)
        buf598 = buf592; del buf592  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_57.run(buf598, convolution_33, unsqueeze_1133, buf596, squeeze_100, buf594, primals_101, 1775616, grid=grid(1775616), stream=stream0)
        del convolution_33
        del primals_101
        del squeeze_100
        del unsqueeze_1133
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf599 = aten.convolution_backward.default(buf598, relu_32, primals_100, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_100
        buf600 = buf599[0]
        buf601 = buf599[1]
        del buf599
        buf602 = buf585; del buf585  # reuse
        buf604 = buf583; del buf583  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_70.run(relu_32, buf600, convolution_32, unsqueeze_1145, buf602, buf604, 9344, 127, grid=grid(9344), stream=stream0)
        buf603 = buf586; del buf586  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_71.run(buf602, buf603, 128, 73, grid=grid(128), stream=stream0)
        buf605 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf606 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_72.run(buf604, squeeze_97, buf605, buf606, 128, 73, grid=grid(128), stream=stream0)
        buf607 = buf600; del buf600  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_73.run(buf607, relu_32, convolution_32, unsqueeze_1145, buf605, squeeze_97, buf603, primals_98, 1183744, grid=grid(1183744), stream=stream0)
        del convolution_32
        del primals_98
        del relu_32
        del squeeze_97
        del unsqueeze_1145
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf608 = aten.convolution_backward.default(buf607, relu_31, primals_97, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf607
        del primals_97
        buf609 = buf608[0]
        buf610 = buf608[1]
        del buf608
        buf611 = buf604; del buf604  # reuse
        buf613 = buf602; del buf602  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_70.run(relu_31, buf609, convolution_31, unsqueeze_1157, buf611, buf613, 9344, 127, grid=grid(9344), stream=stream0)
        buf612 = buf605; del buf605  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_71.run(buf611, buf612, 128, 73, grid=grid(128), stream=stream0)
        del buf611
        buf614 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf615 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_72.run(buf613, squeeze_94, buf614, buf615, 128, 73, grid=grid(128), stream=stream0)
        del buf613
        buf616 = buf609; del buf609  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_73.run(buf616, relu_31, convolution_31, unsqueeze_1157, buf614, squeeze_94, buf612, primals_95, 1183744, grid=grid(1183744), stream=stream0)
        del buf614
        del convolution_31
        del primals_95
        del relu_31
        del squeeze_94
        del unsqueeze_1157
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf617 = aten.convolution_backward.default(buf616, cat_4, primals_94, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf616
        del primals_94
        buf618 = buf617[0]
        buf619 = buf617[1]
        del buf617
        buf620 = buf598; del buf598  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_69.run(le_65, buf450, buf495, buf523, buf533, buf620, 9248, 192, grid=grid(9248, 192), stream=stream0)
        del buf450
        del buf495
        del buf523
        del buf533
        del le_65
        buf621 = buf595; del buf595  # reuse
        buf623 = buf593; del buf593  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_56.run(buf620, convolution_30, unsqueeze_1169, buf621, buf623, 14016, 127, grid=grid(14016), stream=stream0)
        buf622 = buf596; del buf596  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_43.run(buf621, buf622, 192, 73, grid=grid(192), stream=stream0)
        del buf621
        buf624 = empty_strided_cuda((192, ), (1, ), torch.float32)
        buf625 = empty_strided_cuda((192, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_44.run(buf623, squeeze_91, buf624, buf625, 192, 73, grid=grid(192), stream=stream0)
        del buf623
        buf626 = buf620; del buf620  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_57.run(buf626, convolution_30, unsqueeze_1169, buf624, squeeze_91, buf622, primals_92, 1775616, grid=grid(1775616), stream=stream0)
        del convolution_30
        del primals_92
        del squeeze_91
        del unsqueeze_1169
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf627 = aten.convolution_backward.default(buf626, cat_4, primals_91, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf626
        del cat_4
        del primals_91
        buf628 = buf627[0]
        buf629 = buf627[1]
        del buf627
        buf630 = buf545; del buf545  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_74.run(buf630, buf590, buf618, buf628, 9248, 768, grid=grid(9248, 768), stream=stream0)
        del buf590
        del buf618
        del buf628
        # Source Nodes: [], Original ATen: [aten.max_pool2d_with_indices_backward]
        buf631 = aten.max_pool2d_with_indices_backward.default(reinterpret_tensor(buf630, (32, 288, 17, 17), (221952, 289, 17, 1), 138720), cat_3, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_65)
        del getitem_65
        buf632 = buf631
        del buf631
        buf633 = empty_strided_cuda((96, 73), (73, 1), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_75.run(le_66, buf630, buf633, 7008, 127, grid=grid(7008), stream=stream0)
        buf634 = empty_strided_cuda((96, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_76.run(buf633, buf634, 96, 73, grid=grid(96), stream=stream0)
        buf635 = reinterpret_tensor(buf633, (96, 73), (1, 96), 0); del buf633  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_77.run(le_66, buf630, convolution_29, unsqueeze_1181, buf635, 7008, 127, grid=grid(7008), stream=stream0)
        buf636 = empty_strided_cuda((96, ), (1, ), torch.float32)
        buf637 = empty_strided_cuda((96, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_78.run(buf635, squeeze_88, buf636, buf637, 96, 73, grid=grid(96), stream=stream0)
        del buf635
        buf638 = empty_strided_cuda((32, 96, 17, 17), (27744, 1, 1632, 96), torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_79.run(le_66, buf630, convolution_29, unsqueeze_1181, buf636, squeeze_88, buf634, primals_89, buf638, 9248, 96, grid=grid(9248, 96), stream=stream0)
        del convolution_29
        del le_66
        del primals_89
        del squeeze_88
        del unsqueeze_1181
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf639 = aten.convolution_backward.default(buf638, relu_28, primals_88, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf638
        del primals_88
        buf640 = buf639[0]
        buf641 = buf639[1]
        del buf639
        buf642 = empty_strided_cuda((96, 307), (1, 96), torch.float32)
        buf644 = empty_strided_cuda((96, 307), (1, 96), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_80.run(relu_28, buf640, convolution_28, unsqueeze_1193, buf642, buf644, 29472, 128, grid=grid(29472), stream=stream0)
        buf643 = buf636; del buf636  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_81.run(buf642, buf643, 96, 307, grid=grid(96), stream=stream0)
        buf645 = empty_strided_cuda((96, ), (1, ), torch.float32)
        buf646 = empty_strided_cuda((96, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_82.run(buf644, squeeze_85, buf645, buf646, 96, 307, grid=grid(96), stream=stream0)
        buf647 = buf640; del buf640  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_83.run(buf647, relu_28, convolution_28, unsqueeze_1193, buf645, squeeze_85, buf643, primals_86, 3763200, grid=grid(3763200), stream=stream0)
        del convolution_28
        del primals_86
        del relu_28
        del squeeze_85
        del unsqueeze_1193
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf648 = aten.convolution_backward.default(buf647, relu_27, primals_85, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_85
        buf649 = buf648[0]
        buf650 = buf648[1]
        del buf648
        buf651 = empty_strided_cuda((64, 307), (1, 64), torch.float32)
        buf653 = empty_strided_cuda((64, 307), (1, 64), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_84.run(relu_27, buf649, convolution_27, unsqueeze_1205, buf651, buf653, 19648, 128, grid=grid(19648), stream=stream0)
        buf652 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_85.run(buf651, buf652, 64, 307, grid=grid(64), stream=stream0)
        buf654 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf655 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_86.run(buf653, squeeze_82, buf654, buf655, 64, 307, grid=grid(64), stream=stream0)
        buf656 = buf649; del buf649  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_87.run(buf656, relu_27, convolution_27, unsqueeze_1205, buf654, squeeze_82, buf652, primals_83, 2508800, grid=grid(2508800), stream=stream0)
        del convolution_27
        del primals_83
        del relu_27
        del squeeze_82
        del unsqueeze_1205
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf657 = aten.convolution_backward.default(buf656, cat_3, primals_82, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_82
        buf658 = buf657[0]
        buf659 = buf657[1]
        del buf657
        buf660 = empty_strided_cuda((384, 73), (73, 1), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_88.run(le_69, buf630, buf660, 28032, 127, grid=grid(28032), stream=stream0)
        buf661 = buf157; del buf157  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_89.run(buf660, buf661, 384, 73, grid=grid(384), stream=stream0)
        buf662 = reinterpret_tensor(buf660, (384, 73), (1, 384), 0); del buf660  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_90.run(le_69, buf630, convolution_26, unsqueeze_1217, buf662, 28032, 127, grid=grid(28032), stream=stream0)
        buf663 = empty_strided_cuda((384, ), (1, ), torch.float32)
        buf664 = empty_strided_cuda((384, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_91.run(buf662, squeeze_79, buf663, buf664, 384, 73, grid=grid(384), stream=stream0)
        del buf662
        buf665 = empty_strided_cuda((32, 384, 17, 17), (110976, 1, 6528, 384), torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_92.run(le_69, buf630, convolution_26, unsqueeze_1217, buf663, squeeze_79, buf661, primals_80, buf665, 9248, 384, grid=grid(9248, 384), stream=stream0)
        del buf630
        del buf663
        del convolution_26
        del le_69
        del primals_80
        del squeeze_79
        del unsqueeze_1217
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf666 = aten.convolution_backward.default(buf665, cat_3, primals_79, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf665
        del cat_3
        del primals_79
        buf667 = buf666[0]
        buf668 = buf666[1]
        del buf666
        buf669 = buf653; del buf653  # reuse
        buf671 = buf651; del buf651  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_93.run(le_70, buf632, buf658, buf667, convolution_25, unsqueeze_1229, buf669, buf671, 19648, 128, grid=grid(19648), stream=stream0)
        buf670 = buf654; del buf654  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_85.run(buf669, buf670, 64, 307, grid=grid(64), stream=stream0)
        buf672 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf674 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_86.run(buf671, squeeze_76, buf672, buf674, 64, 307, grid=grid(64), stream=stream0)
        buf673 = buf656; del buf656  # reuse
        buf675 = buf673; del buf673  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_94.run(buf675, le_70, buf632, buf658, buf667, convolution_25, unsqueeze_1229, buf672, squeeze_76, buf670, primals_77, 2508800, grid=grid(2508800), stream=stream0)
        del convolution_25
        del le_70
        del primals_77
        del squeeze_76
        del unsqueeze_1229
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf676 = aten.convolution_backward.default(buf675, avg_pool2d_2, primals_76, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del avg_pool2d_2
        del buf675
        del primals_76
        buf677 = buf676[0]
        buf678 = buf676[1]
        del buf676
        buf679 = empty_strided_cuda((32, 288, 35, 35), (352800, 1225, 35, 1), torch.float32)
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]
        triton_poi_fused_avg_pool2d_backward_95.run(buf677, buf679, 9216, 1225, grid=grid(9216, 1225), stream=stream0)
        del buf677
        buf680 = buf644; del buf644  # reuse
        buf682 = buf642; del buf642  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_96.run(le_71, buf632, buf658, buf667, convolution_24, unsqueeze_1241, buf680, buf682, 29472, 128, grid=grid(29472), stream=stream0)
        buf681 = buf645; del buf645  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_81.run(buf680, buf681, 96, 307, grid=grid(96), stream=stream0)
        buf683 = empty_strided_cuda((96, ), (1, ), torch.float32)
        buf685 = empty_strided_cuda((96, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_82.run(buf682, squeeze_73, buf683, buf685, 96, 307, grid=grid(96), stream=stream0)
        buf684 = buf647; del buf647  # reuse
        buf686 = buf684; del buf684  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_97.run(buf686, le_71, buf632, buf658, buf667, convolution_24, unsqueeze_1241, buf683, squeeze_73, buf681, primals_74, 3763200, grid=grid(3763200), stream=stream0)
        del convolution_24
        del le_71
        del primals_74
        del squeeze_73
        del unsqueeze_1241
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf687 = aten.convolution_backward.default(buf686, relu_23, primals_73, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf686
        del primals_73
        buf688 = buf687[0]
        buf689 = buf687[1]
        del buf687
        buf690 = buf682; del buf682  # reuse
        buf692 = buf680; del buf680  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_80.run(relu_23, buf688, convolution_23, unsqueeze_1253, buf690, buf692, 29472, 128, grid=grid(29472), stream=stream0)
        buf691 = buf683; del buf683  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_81.run(buf690, buf691, 96, 307, grid=grid(96), stream=stream0)
        buf693 = empty_strided_cuda((96, ), (1, ), torch.float32)
        buf694 = empty_strided_cuda((96, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_82.run(buf692, squeeze_70, buf693, buf694, 96, 307, grid=grid(96), stream=stream0)
        buf695 = buf688; del buf688  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_83.run(buf695, relu_23, convolution_23, unsqueeze_1253, buf693, squeeze_70, buf691, primals_71, 3763200, grid=grid(3763200), stream=stream0)
        del convolution_23
        del primals_71
        del relu_23
        del squeeze_70
        del unsqueeze_1253
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf696 = aten.convolution_backward.default(buf695, relu_22, primals_70, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_70
        buf697 = buf696[0]
        buf698 = buf696[1]
        del buf696
        buf699 = buf671; del buf671  # reuse
        buf701 = buf669; del buf669  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_84.run(relu_22, buf697, convolution_22, unsqueeze_1265, buf699, buf701, 19648, 128, grid=grid(19648), stream=stream0)
        buf700 = buf672; del buf672  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_85.run(buf699, buf700, 64, 307, grid=grid(64), stream=stream0)
        buf702 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf703 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_86.run(buf701, squeeze_67, buf702, buf703, 64, 307, grid=grid(64), stream=stream0)
        buf704 = buf697; del buf697  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_87.run(buf704, relu_22, convolution_22, unsqueeze_1265, buf702, squeeze_67, buf700, primals_68, 2508800, grid=grid(2508800), stream=stream0)
        del convolution_22
        del primals_68
        del relu_22
        del squeeze_67
        del unsqueeze_1265
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf705 = aten.convolution_backward.default(buf704, cat_2, primals_67, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_67
        buf706 = buf705[0]
        buf707 = buf705[1]
        del buf705
        buf708 = buf701; del buf701  # reuse
        buf710 = buf699; del buf699  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_98.run(le_74, buf632, buf658, buf667, convolution_21, unsqueeze_1277, buf708, buf710, 19648, 128, grid=grid(19648), stream=stream0)
        buf709 = buf702; del buf702  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_85.run(buf708, buf709, 64, 307, grid=grid(64), stream=stream0)
        buf711 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf713 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_86.run(buf710, squeeze_64, buf711, buf713, 64, 307, grid=grid(64), stream=stream0)
        buf712 = buf704; del buf704  # reuse
        buf714 = buf712; del buf712  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_99.run(buf714, le_74, buf632, buf658, buf667, convolution_21, unsqueeze_1277, buf711, squeeze_64, buf709, primals_65, 2508800, grid=grid(2508800), stream=stream0)
        del convolution_21
        del le_74
        del primals_65
        del squeeze_64
        del unsqueeze_1277
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf715 = aten.convolution_backward.default(buf714, relu_20, primals_64, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_64
        buf716 = buf715[0]
        buf717 = buf715[1]
        del buf715
        buf718 = empty_strided_cuda((48, 307), (1, 48), torch.float32)
        buf720 = empty_strided_cuda((48, 307), (1, 48), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_100.run(relu_20, buf716, convolution_20, unsqueeze_1289, buf718, buf720, 14736, 128, grid=grid(14736), stream=stream0)
        buf719 = empty_strided_cuda((48, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_101.run(buf718, buf719, 48, 307, grid=grid(48), stream=stream0)
        buf721 = empty_strided_cuda((48, ), (1, ), torch.float32)
        buf722 = empty_strided_cuda((48, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_102.run(buf720, squeeze_61, buf721, buf722, 48, 307, grid=grid(48), stream=stream0)
        buf723 = buf716; del buf716  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_103.run(buf723, relu_20, convolution_20, unsqueeze_1289, buf721, squeeze_61, buf719, primals_62, 1881600, grid=grid(1881600), stream=stream0)
        del convolution_20
        del primals_62
        del relu_20
        del squeeze_61
        del unsqueeze_1289
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf724 = aten.convolution_backward.default(buf723, cat_2, primals_61, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf723
        del primals_61
        buf725 = buf724[0]
        buf726 = buf724[1]
        del buf724
        buf727 = buf710; del buf710  # reuse
        buf729 = buf708; del buf708  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_104.run(le_76, buf632, buf658, buf667, convolution_19, unsqueeze_1301, buf727, buf729, 19648, 128, grid=grid(19648), stream=stream0)
        buf728 = buf711; del buf711  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_85.run(buf727, buf728, 64, 307, grid=grid(64), stream=stream0)
        buf730 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf732 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_86.run(buf729, squeeze_58, buf730, buf732, 64, 307, grid=grid(64), stream=stream0)
        buf731 = buf714; del buf714  # reuse
        buf733 = buf731; del buf731  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_105.run(buf733, le_76, buf632, buf658, buf667, convolution_19, unsqueeze_1301, buf730, squeeze_58, buf728, primals_59, 2508800, grid=grid(2508800), stream=stream0)
        del buf632
        del buf658
        del buf667
        del convolution_19
        del le_76
        del primals_59
        del squeeze_58
        del unsqueeze_1301
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf734 = aten.convolution_backward.default(buf733, cat_2, primals_58, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_2
        del primals_58
        buf735 = buf734[0]
        buf736 = buf734[1]
        del buf734
        buf737 = buf733; del buf733  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_106.run(le_77, buf679, buf706, buf725, buf735, buf737, 39200, 64, grid=grid(39200, 64), stream=stream0)
        del le_77
        buf738 = buf729; del buf729  # reuse
        buf740 = buf727; del buf727  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_107.run(buf737, convolution_18, unsqueeze_1313, buf738, buf740, 19648, 128, grid=grid(19648), stream=stream0)
        buf739 = buf730; del buf730  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_85.run(buf738, buf739, 64, 307, grid=grid(64), stream=stream0)
        buf741 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf742 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_86.run(buf740, squeeze_55, buf741, buf742, 64, 307, grid=grid(64), stream=stream0)
        buf743 = buf737; del buf737  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_108.run(buf743, convolution_18, unsqueeze_1313, buf741, squeeze_55, buf739, primals_56, 2508800, grid=grid(2508800), stream=stream0)
        del convolution_18
        del primals_56
        del squeeze_55
        del unsqueeze_1313
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf744 = aten.convolution_backward.default(buf743, avg_pool2d_1, primals_55, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del avg_pool2d_1
        del buf743
        del primals_55
        buf745 = buf744[0]
        buf746 = buf744[1]
        del buf744
        buf747 = empty_strided_cuda((32, 256, 35, 35), (313600, 1225, 35, 1), torch.float32)
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]
        triton_poi_fused_avg_pool2d_backward_109.run(buf745, buf747, 8192, 1225, grid=grid(8192, 1225), stream=stream0)
        del buf745
        buf748 = buf695; del buf695  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_110.run(le_78, buf679, buf706, buf725, buf735, buf748, 39200, 96, grid=grid(39200, 96), stream=stream0)
        del le_78
        buf749 = buf692; del buf692  # reuse
        buf751 = buf690; del buf690  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_111.run(buf748, convolution_17, unsqueeze_1325, buf749, buf751, 29472, 128, grid=grid(29472), stream=stream0)
        buf750 = buf693; del buf693  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_81.run(buf749, buf750, 96, 307, grid=grid(96), stream=stream0)
        buf752 = empty_strided_cuda((96, ), (1, ), torch.float32)
        buf753 = empty_strided_cuda((96, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_82.run(buf751, squeeze_52, buf752, buf753, 96, 307, grid=grid(96), stream=stream0)
        buf754 = buf748; del buf748  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_112.run(buf754, convolution_17, unsqueeze_1325, buf752, squeeze_52, buf750, primals_53, 3763200, grid=grid(3763200), stream=stream0)
        del convolution_17
        del primals_53
        del squeeze_52
        del unsqueeze_1325
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf755 = aten.convolution_backward.default(buf754, relu_16, primals_52, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf754
        del primals_52
        buf756 = buf755[0]
        buf757 = buf755[1]
        del buf755
        buf758 = buf751; del buf751  # reuse
        buf760 = buf749; del buf749  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_80.run(relu_16, buf756, convolution_16, unsqueeze_1337, buf758, buf760, 29472, 128, grid=grid(29472), stream=stream0)
        buf759 = buf752; del buf752  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_81.run(buf758, buf759, 96, 307, grid=grid(96), stream=stream0)
        buf761 = empty_strided_cuda((96, ), (1, ), torch.float32)
        buf762 = empty_strided_cuda((96, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_82.run(buf760, squeeze_49, buf761, buf762, 96, 307, grid=grid(96), stream=stream0)
        buf763 = buf756; del buf756  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_83.run(buf763, relu_16, convolution_16, unsqueeze_1337, buf761, squeeze_49, buf759, primals_50, 3763200, grid=grid(3763200), stream=stream0)
        del convolution_16
        del primals_50
        del relu_16
        del squeeze_49
        del unsqueeze_1337
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf764 = aten.convolution_backward.default(buf763, relu_15, primals_49, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_49
        buf765 = buf764[0]
        buf766 = buf764[1]
        del buf764
        buf767 = buf740; del buf740  # reuse
        buf769 = buf738; del buf738  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_84.run(relu_15, buf765, convolution_15, unsqueeze_1349, buf767, buf769, 19648, 128, grid=grid(19648), stream=stream0)
        buf768 = buf741; del buf741  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_85.run(buf767, buf768, 64, 307, grid=grid(64), stream=stream0)
        buf770 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf771 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_86.run(buf769, squeeze_46, buf770, buf771, 64, 307, grid=grid(64), stream=stream0)
        buf772 = buf765; del buf765  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_87.run(buf772, relu_15, convolution_15, unsqueeze_1349, buf770, squeeze_46, buf768, primals_47, 2508800, grid=grid(2508800), stream=stream0)
        del convolution_15
        del primals_47
        del relu_15
        del squeeze_46
        del unsqueeze_1349
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf773 = aten.convolution_backward.default(buf772, cat_1, primals_46, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_46
        buf774 = buf773[0]
        buf775 = buf773[1]
        del buf773
        buf776 = buf772; del buf772  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_113.run(le_81, buf679, buf706, buf725, buf735, buf776, 39200, 64, grid=grid(39200, 64), stream=stream0)
        del le_81
        buf777 = buf769; del buf769  # reuse
        buf779 = buf767; del buf767  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_107.run(buf776, convolution_14, unsqueeze_1361, buf777, buf779, 19648, 128, grid=grid(19648), stream=stream0)
        buf778 = buf770; del buf770  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_85.run(buf777, buf778, 64, 307, grid=grid(64), stream=stream0)
        buf780 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf781 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_86.run(buf779, squeeze_43, buf780, buf781, 64, 307, grid=grid(64), stream=stream0)
        buf782 = buf776; del buf776  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_108.run(buf782, convolution_14, unsqueeze_1361, buf780, squeeze_43, buf778, primals_44, 2508800, grid=grid(2508800), stream=stream0)
        del convolution_14
        del primals_44
        del squeeze_43
        del unsqueeze_1361
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf783 = aten.convolution_backward.default(buf782, relu_13, primals_43, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_43
        buf784 = buf783[0]
        buf785 = buf783[1]
        del buf783
        buf786 = buf720; del buf720  # reuse
        buf788 = buf718; del buf718  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_100.run(relu_13, buf784, convolution_13, unsqueeze_1373, buf786, buf788, 14736, 128, grid=grid(14736), stream=stream0)
        buf787 = buf721; del buf721  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_101.run(buf786, buf787, 48, 307, grid=grid(48), stream=stream0)
        buf789 = empty_strided_cuda((48, ), (1, ), torch.float32)
        buf790 = empty_strided_cuda((48, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_102.run(buf788, squeeze_40, buf789, buf790, 48, 307, grid=grid(48), stream=stream0)
        buf791 = buf784; del buf784  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_103.run(buf791, relu_13, convolution_13, unsqueeze_1373, buf789, squeeze_40, buf787, primals_41, 1881600, grid=grid(1881600), stream=stream0)
        del convolution_13
        del primals_41
        del relu_13
        del squeeze_40
        del unsqueeze_1373
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf792 = aten.convolution_backward.default(buf791, cat_1, primals_40, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf791
        del primals_40
        buf793 = buf792[0]
        buf794 = buf792[1]
        del buf792
        buf795 = buf782; del buf782  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_114.run(le_83, buf679, buf706, buf725, buf735, buf795, 39200, 64, grid=grid(39200, 64), stream=stream0)
        del buf679
        del buf706
        del buf725
        del buf735
        del le_83
        buf796 = buf779; del buf779  # reuse
        buf798 = buf777; del buf777  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_107.run(buf795, convolution_12, unsqueeze_1385, buf796, buf798, 19648, 128, grid=grid(19648), stream=stream0)
        buf797 = buf780; del buf780  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_85.run(buf796, buf797, 64, 307, grid=grid(64), stream=stream0)
        buf799 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf800 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_86.run(buf798, squeeze_37, buf799, buf800, 64, 307, grid=grid(64), stream=stream0)
        buf801 = buf795; del buf795  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_108.run(buf801, convolution_12, unsqueeze_1385, buf799, squeeze_37, buf797, primals_38, 2508800, grid=grid(2508800), stream=stream0)
        del convolution_12
        del primals_38
        del squeeze_37
        del unsqueeze_1385
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf802 = aten.convolution_backward.default(buf801, cat_1, primals_37, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf801
        del cat_1
        del primals_37
        buf803 = buf802[0]
        buf804 = buf802[1]
        del buf802
        buf805 = empty_strided_cuda((32, 32, 35, 35), (39200, 1, 1120, 32), torch.float32)
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_115.run(le_84, buf747, buf774, buf793, buf803, buf805, 39200, 32, grid=grid(39200, 32), stream=stream0)
        del le_84
        buf806 = empty_strided_cuda((32, 307), (1, 32), torch.float32)
        buf808 = empty_strided_cuda((32, 307), (1, 32), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_116.run(buf805, convolution_11, unsqueeze_1397, buf806, buf808, 9824, 128, grid=grid(9824), stream=stream0)
        buf807 = empty_strided_cuda((32, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_117.run(buf806, buf807, 32, 307, grid=grid(32), stream=stream0)
        del buf806
        buf809 = empty_strided_cuda((32, ), (1, ), torch.float32)
        buf810 = empty_strided_cuda((32, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_118.run(buf808, squeeze_34, buf809, buf810, 32, 307, grid=grid(32), stream=stream0)
        del buf808
        buf811 = buf805; del buf805  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_119.run(buf811, convolution_11, unsqueeze_1397, buf809, squeeze_34, buf807, primals_35, 1254400, grid=grid(1254400), stream=stream0)
        del convolution_11
        del primals_35
        del squeeze_34
        del unsqueeze_1397
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf812 = aten.convolution_backward.default(buf811, avg_pool2d, primals_34, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del avg_pool2d
        del buf811
        del primals_34
        buf813 = buf812[0]
        buf814 = buf812[1]
        del buf812
        buf815 = empty_strided_cuda((32, 192, 35, 35), (235200, 1225, 35, 1), torch.float32)
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]
        triton_poi_fused_avg_pool2d_backward_120.run(buf813, buf815, 6144, 1225, grid=grid(6144, 1225), stream=stream0)
        del buf813
        buf816 = buf763; del buf763  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_121.run(le_85, buf747, buf774, buf793, buf803, buf816, 39200, 96, grid=grid(39200, 96), stream=stream0)
        del le_85
        buf817 = buf760; del buf760  # reuse
        buf819 = buf758; del buf758  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_111.run(buf816, convolution_10, unsqueeze_1409, buf817, buf819, 29472, 128, grid=grid(29472), stream=stream0)
        buf818 = buf761; del buf761  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_81.run(buf817, buf818, 96, 307, grid=grid(96), stream=stream0)
        buf820 = empty_strided_cuda((96, ), (1, ), torch.float32)
        buf821 = empty_strided_cuda((96, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_82.run(buf819, squeeze_31, buf820, buf821, 96, 307, grid=grid(96), stream=stream0)
        buf822 = buf816; del buf816  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_112.run(buf822, convolution_10, unsqueeze_1409, buf820, squeeze_31, buf818, primals_32, 3763200, grid=grid(3763200), stream=stream0)
        del convolution_10
        del primals_32
        del squeeze_31
        del unsqueeze_1409
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf823 = aten.convolution_backward.default(buf822, relu_9, primals_31, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf822
        del primals_31
        buf824 = buf823[0]
        buf825 = buf823[1]
        del buf823
        buf826 = buf819; del buf819  # reuse
        buf828 = buf817; del buf817  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_80.run(relu_9, buf824, convolution_9, unsqueeze_1421, buf826, buf828, 29472, 128, grid=grid(29472), stream=stream0)
        buf827 = buf820; del buf820  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_81.run(buf826, buf827, 96, 307, grid=grid(96), stream=stream0)
        del buf826
        buf829 = empty_strided_cuda((96, ), (1, ), torch.float32)
        buf830 = empty_strided_cuda((96, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_82.run(buf828, squeeze_28, buf829, buf830, 96, 307, grid=grid(96), stream=stream0)
        del buf828
        buf831 = buf824; del buf824  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_83.run(buf831, relu_9, convolution_9, unsqueeze_1421, buf829, squeeze_28, buf827, primals_29, 3763200, grid=grid(3763200), stream=stream0)
        del buf829
        del convolution_9
        del primals_29
        del relu_9
        del squeeze_28
        del unsqueeze_1421
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf832 = aten.convolution_backward.default(buf831, relu_8, primals_28, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf831
        del primals_28
        buf833 = buf832[0]
        buf834 = buf832[1]
        del buf832
        buf835 = buf798; del buf798  # reuse
        buf837 = buf796; del buf796  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_84.run(relu_8, buf833, convolution_8, unsqueeze_1433, buf835, buf837, 19648, 128, grid=grid(19648), stream=stream0)
        buf836 = buf799; del buf799  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_85.run(buf835, buf836, 64, 307, grid=grid(64), stream=stream0)
        buf838 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf839 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_86.run(buf837, squeeze_25, buf838, buf839, 64, 307, grid=grid(64), stream=stream0)
        buf840 = buf833; del buf833  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_87.run(buf840, relu_8, convolution_8, unsqueeze_1433, buf838, squeeze_25, buf836, primals_26, 2508800, grid=grid(2508800), stream=stream0)
        del convolution_8
        del primals_26
        del relu_8
        del squeeze_25
        del unsqueeze_1433
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf841 = aten.convolution_backward.default(buf840, getitem_12, primals_25, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_25
        buf842 = buf841[0]
        buf843 = buf841[1]
        del buf841
        buf844 = buf840; del buf840  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_122.run(le_88, buf747, buf774, buf793, buf803, buf844, 39200, 64, grid=grid(39200, 64), stream=stream0)
        del le_88
        buf845 = buf837; del buf837  # reuse
        buf847 = buf835; del buf835  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_107.run(buf844, convolution_7, unsqueeze_1445, buf845, buf847, 19648, 128, grid=grid(19648), stream=stream0)
        buf846 = buf838; del buf838  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_85.run(buf845, buf846, 64, 307, grid=grid(64), stream=stream0)
        buf848 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf849 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_86.run(buf847, squeeze_22, buf848, buf849, 64, 307, grid=grid(64), stream=stream0)
        buf850 = buf844; del buf844  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_108.run(buf850, convolution_7, unsqueeze_1445, buf848, squeeze_22, buf846, primals_23, 2508800, grid=grid(2508800), stream=stream0)
        del convolution_7
        del primals_23
        del squeeze_22
        del unsqueeze_1445
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf851 = aten.convolution_backward.default(buf850, relu_6, primals_22, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_22
        buf852 = buf851[0]
        buf853 = buf851[1]
        del buf851
        buf854 = buf788; del buf788  # reuse
        buf856 = buf786; del buf786  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_100.run(relu_6, buf852, convolution_6, unsqueeze_1457, buf854, buf856, 14736, 128, grid=grid(14736), stream=stream0)
        buf855 = buf789; del buf789  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_101.run(buf854, buf855, 48, 307, grid=grid(48), stream=stream0)
        del buf854
        buf857 = empty_strided_cuda((48, ), (1, ), torch.float32)
        buf858 = empty_strided_cuda((48, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_102.run(buf856, squeeze_19, buf857, buf858, 48, 307, grid=grid(48), stream=stream0)
        del buf856
        buf859 = buf852; del buf852  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_103.run(buf859, relu_6, convolution_6, unsqueeze_1457, buf857, squeeze_19, buf855, primals_20, 1881600, grid=grid(1881600), stream=stream0)
        del buf857
        del convolution_6
        del primals_20
        del relu_6
        del squeeze_19
        del unsqueeze_1457
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf860 = aten.convolution_backward.default(buf859, getitem_12, primals_19, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf859
        del primals_19
        buf861 = buf860[0]
        buf862 = buf860[1]
        del buf860
        buf863 = buf850; del buf850  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_123.run(le_90, buf747, buf774, buf793, buf803, buf863, 39200, 64, grid=grid(39200, 64), stream=stream0)
        del buf747
        del buf774
        del buf793
        del buf803
        del le_90
        buf864 = buf847; del buf847  # reuse
        buf866 = buf845; del buf845  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_107.run(buf863, convolution_5, unsqueeze_1469, buf864, buf866, 19648, 128, grid=grid(19648), stream=stream0)
        buf865 = buf848; del buf848  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_85.run(buf864, buf865, 64, 307, grid=grid(64), stream=stream0)
        del buf864
        buf867 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf868 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_86.run(buf866, squeeze_16, buf867, buf868, 64, 307, grid=grid(64), stream=stream0)
        del buf866
        buf869 = buf863; del buf863  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_108.run(buf869, convolution_5, unsqueeze_1469, buf867, squeeze_16, buf865, primals_17, 2508800, grid=grid(2508800), stream=stream0)
        del convolution_5
        del primals_17
        del squeeze_16
        del unsqueeze_1469
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf870 = aten.convolution_backward.default(buf869, getitem_12, primals_16, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf869
        del getitem_12
        del primals_16
        buf871 = buf870[0]
        buf872 = buf870[1]
        del buf870
        buf873 = buf842; del buf842  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_124.run(buf873, buf815, buf861, buf871, 39200, 192, grid=grid(39200, 192), stream=stream0)
        del buf815
        del buf861
        del buf871
        # Source Nodes: [], Original ATen: [aten.add, aten.max_pool2d_with_indices_backward]
        buf874 = aten.max_pool2d_with_indices_backward.default(buf873, relu_4, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_13)
        del buf873
        del getitem_13
        buf875 = buf874
        del buf874
        buf876 = empty_strided_cuda((192, 430), (1, 192), torch.float32)
        buf878 = empty_strided_cuda((192, 430), (1, 192), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_125.run(relu_4, buf875, convolution_4, unsqueeze_1481, buf876, buf878, 82560, 376, grid=grid(82560), stream=stream0)
        buf877 = buf624; del buf624  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_126.run(buf876, buf877, 192, 430, grid=grid(192), stream=stream0)
        del buf876
        buf879 = empty_strided_cuda((192, ), (1, ), torch.float32)
        buf880 = empty_strided_cuda((192, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_127.run(buf878, squeeze_13, buf879, buf880, 192, 430, grid=grid(192), stream=stream0)
        del buf878
        buf881 = buf875; del buf875  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_128.run(buf881, relu_4, convolution_4, unsqueeze_1481, buf879, squeeze_13, buf877, primals_14, 30971904, grid=grid(30971904), stream=stream0)
        del buf879
        del convolution_4
        del primals_14
        del relu_4
        del squeeze_13
        del unsqueeze_1481
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf882 = aten.convolution_backward.default(buf881, relu_3, primals_13, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf881
        del primals_13
        buf883 = buf882[0]
        buf884 = buf882[1]
        del buf882
        buf885 = empty_strided_cuda((80, 1333), (1, 80), torch.float32)
        buf887 = empty_strided_cuda((80, 1333), (1, 80), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_129.run(relu_3, buf883, convolution_3, unsqueeze_1493, buf885, buf887, 106640, 128, grid=grid(106640), stream=stream0)
        buf886 = empty_strided_cuda((80, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_130.run(buf885, buf886, 80, 1333, grid=grid(80), stream=stream0)
        del buf885
        buf888 = empty_strided_cuda((80, ), (1, ), torch.float32)
        buf889 = empty_strided_cuda((80, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_131.run(buf887, squeeze_10, buf888, buf889, 80, 1333, grid=grid(80), stream=stream0)
        del buf887
        buf890 = buf883; del buf883  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_132.run(buf890, relu_3, convolution_3, unsqueeze_1493, buf888, squeeze_10, buf886, primals_11, 13642240, grid=grid(13642240), stream=stream0)
        del buf888
        del convolution_3
        del primals_11
        del relu_3
        del squeeze_10
        del unsqueeze_1493
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf891 = aten.convolution_backward.default(buf890, getitem_6, primals_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf890
        del getitem_6
        del primals_10
        buf892 = buf891[0]
        buf893 = buf891[1]
        del buf891
        # Source Nodes: [], Original ATen: [aten.max_pool2d_with_indices_backward]
        buf894 = aten.max_pool2d_with_indices_backward.default(buf892, relu_2, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_7)
        del buf892
        del getitem_7
        buf895 = buf894
        del buf894
        buf896 = empty_strided_cuda((64, 882), (1, 64), torch.float32)
        buf898 = empty_strided_cuda((64, 882), (1, 64), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_133.run(relu_2, buf895, convolution_2, unsqueeze_1505, buf896, buf898, 56448, 784, grid=grid(56448), stream=stream0)
        buf897 = buf867; del buf867  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_134.run(buf896, buf897, 64, 882, grid=grid(64), stream=stream0)
        del buf896
        buf899 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf900 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_135.run(buf898, squeeze_7, buf899, buf900, 64, 882, grid=grid(64), stream=stream0)
        del buf898
        buf901 = buf895; del buf895  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_136.run(buf901, relu_2, convolution_2, unsqueeze_1505, buf899, squeeze_7, buf897, primals_8, 44255232, grid=grid(44255232), stream=stream0)
        del buf899
        del convolution_2
        del primals_8
        del relu_2
        del squeeze_7
        del unsqueeze_1505
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf902 = aten.convolution_backward.default(buf901, relu_1, primals_7, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf901
        del primals_7
        buf903 = buf902[0]
        buf904 = buf902[1]
        del buf902
        buf905 = empty_strided_cuda((32, 882), (1, 32), torch.float32)
        buf907 = empty_strided_cuda((32, 882), (1, 32), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_137.run(relu_1, buf903, convolution_1, unsqueeze_1517, buf905, buf907, 28224, 784, grid=grid(28224), stream=stream0)
        buf906 = buf809; del buf809  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_138.run(buf905, buf906, 32, 882, grid=grid(32), stream=stream0)
        del buf905
        buf908 = empty_strided_cuda((32, ), (1, ), torch.float32)
        buf909 = empty_strided_cuda((32, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_139.run(buf907, squeeze_4, buf908, buf909, 32, 882, grid=grid(32), stream=stream0)
        del buf907
        buf910 = buf903; del buf903  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_140.run(buf910, relu_1, convolution_1, unsqueeze_1517, buf908, squeeze_4, buf906, primals_5, 22127616, grid=grid(22127616), stream=stream0)
        del convolution_1
        del primals_5
        del relu_1
        del squeeze_4
        del unsqueeze_1517
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf911 = aten.convolution_backward.default(buf910, relu, primals_4, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf910
        del primals_4
        buf912 = buf911[0]
        buf913 = buf911[1]
        del buf911
        buf914 = empty_strided_cuda((32, 863), (1, 32), torch.float32)
        buf916 = empty_strided_cuda((32, 863), (1, 32), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_141.run(relu, buf912, convolution, unsqueeze_1529, buf914, buf916, 27616, 824, grid=grid(27616), stream=stream0)
        buf915 = buf908; del buf908  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_142.run(buf914, buf915, 32, 863, grid=grid(32), stream=stream0)
        del buf914
        buf917 = empty_strided_cuda((32, ), (1, ), torch.float32)
        buf918 = empty_strided_cuda((32, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_143.run(buf916, squeeze_1, buf917, buf918, 32, 863, grid=grid(32), stream=stream0)
        del buf916
        buf919 = buf912; del buf912  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_144.run(buf919, relu, convolution, unsqueeze_1529, buf917, squeeze_1, buf915, primals_2, 22733824, grid=grid(22733824), stream=stream0)
        del buf917
        del convolution
        del primals_2
        del relu
        del squeeze_1
        del unsqueeze_1529
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf920 = aten.convolution_backward.default(buf919, cat, primals_1, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf919
        del cat
        del primals_1
        buf921 = buf920[1]
        del buf920
    return (buf921, buf918, buf915, buf913, buf909, buf906, buf904, buf900, buf897, buf893, buf889, buf886, buf884, buf880, buf877, buf872, buf868, buf865, buf862, buf858, buf855, buf853, buf849, buf846, buf843, buf839, buf836, buf834, buf830, buf827, buf825, buf821, buf818, buf814, buf810, buf807, buf804, buf800, buf797, buf794, buf790, buf787, buf785, buf781, buf778, buf775, buf771, buf768, buf766, buf762, buf759, buf757, buf753, buf750, buf746, buf742, buf739, buf736, buf732, buf728, buf726, buf722, buf719, buf717, buf713, buf709, buf707, buf703, buf700, buf698, buf694, buf691, buf689, buf685, buf681, buf678, buf674, buf670, buf668, buf664, buf661, buf659, buf655, buf652, buf650, buf646, buf643, buf641, buf637, buf634, buf629, buf625, buf622, buf619, buf615, buf612, buf610, buf606, buf603, buf601, buf597, buf594, buf591, buf587, buf584, buf582, buf578, buf575, buf573, buf569, buf566, buf564, buf560, buf557, buf555, buf551, buf548, buf544, buf540, buf537, buf534, buf530, buf527, buf524, buf520, buf517, buf515, buf511, buf508, buf506, buf502, buf499, buf496, buf492, buf489, buf487, buf483, buf480, buf478, buf474, buf471, buf469, buf465, buf462, buf460, buf456, buf453, buf449, buf445, buf442, buf439, buf435, buf432, buf429, buf425, buf422, buf420, buf416, buf413, buf411, buf407, buf404, buf401, buf397, buf394, buf392, buf388, buf385, buf383, buf379, buf376, buf374, buf370, buf367, buf365, buf361, buf358, buf354, buf350, buf347, buf344, buf340, buf337, buf334, buf330, buf327, buf325, buf321, buf318, buf316, buf312, buf309, buf306, buf302, buf299, buf297, buf293, buf290, buf288, buf284, buf281, buf279, buf275, buf272, buf270, buf266, buf263, buf259, buf255, buf252, buf248, buf244, buf241, buf239, buf235, buf233, reinterpret_tensor(buf231, (1000, 768), (768, 1), 0), reinterpret_tensor(buf232, (1000, ), (1, ), 0), buf229, buf225, buf222, buf220, buf216, buf213, buf211, buf207, buf204, buf202, buf198, buf195, buf193, buf189, buf186, buf184, buf180, buf177, buf172, buf168, buf165, buf162, buf159, buf155, buf153, buf149, buf146, buf143, buf139, buf136, buf133, buf129, buf126, buf124, buf121, buf117, buf115, buf111, buf108, buf105, buf101, buf98, buf94, buf90, buf87, buf84, buf81, buf77, buf75, buf72, buf68, buf66, buf63, buf59, buf57, buf54, buf50, buf48, buf44, buf41, buf39, buf36, buf32, buf30, buf27, buf23, buf21, buf18, buf14, buf11, buf8, buf4, reinterpret_tensor(buf1, (1000, 2048), (2048, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, 32, 3, 3), (288, 1, 96, 32), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, 32, 3, 3), (288, 1, 96, 32), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((80, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((192, 80, 3, 3), (720, 1, 240, 80), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((64, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((48, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((64, 48, 5, 5), (1200, 1, 240, 48), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((64, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((96, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((96, 96, 3, 3), (864, 1, 288, 96), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((32, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((48, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((64, 48, 5, 5), (1200, 1, 240, 48), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((96, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((96, 96, 3, 3), (864, 1, 288, 96), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((64, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((48, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((64, 48, 5, 5), (1200, 1, 240, 48), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((64, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((96, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((96, 96, 3, 3), (864, 1, 288, 96), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((64, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((384, 288, 3, 3), (2592, 1, 864, 288), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((64, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((96, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((96, 96, 3, 3), (864, 1, 288, 96), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((128, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((128, 128, 1, 7), (896, 1, 896, 128), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((192, 128, 7, 1), (896, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((128, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((128, 128, 7, 1), (896, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((128, 128, 1, 7), (896, 1, 896, 128), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((128, 128, 7, 1), (896, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((192, 128, 1, 7), (896, 1, 896, 128), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((160, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((160, 160, 1, 7), (1120, 1, 1120, 160), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((192, 160, 7, 1), (1120, 1, 160, 160), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((160, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((160, 160, 7, 1), (1120, 1, 160, 160), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((160, 160, 1, 7), (1120, 1, 1120, 160), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((160, 160, 7, 1), (1120, 1, 160, 160), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((192, 160, 1, 7), (1120, 1, 1120, 160), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((160, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((160, 160, 1, 7), (1120, 1, 1120, 160), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((192, 160, 7, 1), (1120, 1, 160, 160), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((160, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((160, 160, 7, 1), (1120, 1, 160, 160), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((160, 160, 1, 7), (1120, 1, 1120, 160), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((160, 160, 7, 1), (1120, 1, 160, 160), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((192, 160, 1, 7), (1120, 1, 1120, 160), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((192, 192, 1, 7), (1344, 1, 1344, 192), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((192, 192, 7, 1), (1344, 1, 192, 192), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((192, 192, 7, 1), (1344, 1, 192, 192), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((192, 192, 1, 7), (1344, 1, 1344, 192), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((192, 192, 7, 1), (1344, 1, 192, 192), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((192, 192, 1, 7), (1344, 1, 1344, 192), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((128, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((768, 128, 5, 5), (3200, 1, 640, 128), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((320, 192, 3, 3), (1728, 1, 576, 192), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((192, 192, 1, 7), (1344, 1, 1344, 192), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((192, 192, 7, 1), (1344, 1, 192, 192), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((192, 192, 3, 3), (1728, 1, 576, 192), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((320, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((384, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((384, 384, 1, 3), (1152, 1, 1152, 384), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((384, 384, 3, 1), (1152, 1, 384, 384), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((448, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((384, 448, 3, 3), (4032, 1, 1344, 448), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((384, 384, 1, 3), (1152, 1, 1152, 384), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((384, 384, 3, 1), (1152, 1, 384, 384), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((192, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((320, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((384, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((384, 384, 1, 3), (1152, 1, 1152, 384), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((384, 384, 3, 1), (1152, 1, 384, 384), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((448, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((384, 448, 3, 3), (4032, 1, 1344, 448), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((384, 384, 1, 3), (1152, 1, 1152, 384), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((384, 384, 3, 1), (1152, 1, 384, 384), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((192, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat = rand_strided((32, 3, 299, 299), (268203, 1, 897, 3), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((32, 32, 149, 149), (710432, 1, 4768, 32), device='cuda:0', dtype=torch.float32)
    squeeze_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu = rand_strided((32, 32, 149, 149), (710432, 1, 4768, 32), device='cuda:0', dtype=torch.float32)
    convolution_1 = rand_strided((32, 32, 147, 147), (691488, 1, 4704, 32), device='cuda:0', dtype=torch.float32)
    squeeze_4 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_1 = rand_strided((32, 32, 147, 147), (691488, 1, 4704, 32), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((32, 64, 147, 147), (1382976, 1, 9408, 64), device='cuda:0', dtype=torch.float32)
    squeeze_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_2 = rand_strided((32, 64, 147, 147), (1382976, 1, 9408, 64), device='cuda:0', dtype=torch.float32)
    getitem_6 = rand_strided((32, 64, 73, 73), (341056, 1, 4672, 64), device='cuda:0', dtype=torch.float32)
    getitem_7 = rand_strided((32, 64, 73, 73), (341056, 1, 4672, 64), device='cuda:0', dtype=torch.int64)
    convolution_3 = rand_strided((32, 80, 73, 73), (426320, 1, 5840, 80), device='cuda:0', dtype=torch.float32)
    squeeze_10 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_3 = rand_strided((32, 80, 73, 73), (426320, 1, 5840, 80), device='cuda:0', dtype=torch.float32)
    convolution_4 = rand_strided((32, 192, 71, 71), (967872, 1, 13632, 192), device='cuda:0', dtype=torch.float32)
    squeeze_13 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_4 = rand_strided((32, 192, 71, 71), (967872, 1, 13632, 192), device='cuda:0', dtype=torch.float32)
    getitem_12 = rand_strided((32, 192, 35, 35), (235200, 1, 6720, 192), device='cuda:0', dtype=torch.float32)
    getitem_13 = rand_strided((32, 192, 35, 35), (235200, 1, 6720, 192), device='cuda:0', dtype=torch.int64)
    convolution_5 = rand_strided((32, 64, 35, 35), (78400, 1, 2240, 64), device='cuda:0', dtype=torch.float32)
    squeeze_16 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_6 = rand_strided((32, 48, 35, 35), (58800, 1, 1680, 48), device='cuda:0', dtype=torch.float32)
    squeeze_19 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_6 = rand_strided((32, 48, 35, 35), (58800, 1, 1680, 48), device='cuda:0', dtype=torch.float32)
    convolution_7 = rand_strided((32, 64, 35, 35), (78400, 1, 2240, 64), device='cuda:0', dtype=torch.float32)
    squeeze_22 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_8 = rand_strided((32, 64, 35, 35), (78400, 1, 2240, 64), device='cuda:0', dtype=torch.float32)
    squeeze_25 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_8 = rand_strided((32, 64, 35, 35), (78400, 1, 2240, 64), device='cuda:0', dtype=torch.float32)
    convolution_9 = rand_strided((32, 96, 35, 35), (117600, 1, 3360, 96), device='cuda:0', dtype=torch.float32)
    squeeze_28 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_9 = rand_strided((32, 96, 35, 35), (117600, 1, 3360, 96), device='cuda:0', dtype=torch.float32)
    convolution_10 = rand_strided((32, 96, 35, 35), (117600, 1, 3360, 96), device='cuda:0', dtype=torch.float32)
    squeeze_31 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    avg_pool2d = rand_strided((32, 192, 35, 35), (235200, 1, 6720, 192), device='cuda:0', dtype=torch.float32)
    convolution_11 = rand_strided((32, 32, 35, 35), (39200, 1, 1120, 32), device='cuda:0', dtype=torch.float32)
    squeeze_34 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_1 = rand_strided((32, 256, 35, 35), (313600, 1, 8960, 256), device='cuda:0', dtype=torch.float32)
    convolution_12 = rand_strided((32, 64, 35, 35), (78400, 1, 2240, 64), device='cuda:0', dtype=torch.float32)
    squeeze_37 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_13 = rand_strided((32, 48, 35, 35), (58800, 1, 1680, 48), device='cuda:0', dtype=torch.float32)
    squeeze_40 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_13 = rand_strided((32, 48, 35, 35), (58800, 1, 1680, 48), device='cuda:0', dtype=torch.float32)
    convolution_14 = rand_strided((32, 64, 35, 35), (78400, 1, 2240, 64), device='cuda:0', dtype=torch.float32)
    squeeze_43 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_15 = rand_strided((32, 64, 35, 35), (78400, 1, 2240, 64), device='cuda:0', dtype=torch.float32)
    squeeze_46 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_15 = rand_strided((32, 64, 35, 35), (78400, 1, 2240, 64), device='cuda:0', dtype=torch.float32)
    convolution_16 = rand_strided((32, 96, 35, 35), (117600, 1, 3360, 96), device='cuda:0', dtype=torch.float32)
    squeeze_49 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_16 = rand_strided((32, 96, 35, 35), (117600, 1, 3360, 96), device='cuda:0', dtype=torch.float32)
    convolution_17 = rand_strided((32, 96, 35, 35), (117600, 1, 3360, 96), device='cuda:0', dtype=torch.float32)
    squeeze_52 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    avg_pool2d_1 = rand_strided((32, 256, 35, 35), (313600, 1, 8960, 256), device='cuda:0', dtype=torch.float32)
    convolution_18 = rand_strided((32, 64, 35, 35), (78400, 1, 2240, 64), device='cuda:0', dtype=torch.float32)
    squeeze_55 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_2 = rand_strided((32, 288, 35, 35), (352800, 1, 10080, 288), device='cuda:0', dtype=torch.float32)
    convolution_19 = rand_strided((32, 64, 35, 35), (78400, 1, 2240, 64), device='cuda:0', dtype=torch.float32)
    squeeze_58 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_20 = rand_strided((32, 48, 35, 35), (58800, 1, 1680, 48), device='cuda:0', dtype=torch.float32)
    squeeze_61 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_20 = rand_strided((32, 48, 35, 35), (58800, 1, 1680, 48), device='cuda:0', dtype=torch.float32)
    convolution_21 = rand_strided((32, 64, 35, 35), (78400, 1, 2240, 64), device='cuda:0', dtype=torch.float32)
    squeeze_64 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_22 = rand_strided((32, 64, 35, 35), (78400, 1, 2240, 64), device='cuda:0', dtype=torch.float32)
    squeeze_67 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_22 = rand_strided((32, 64, 35, 35), (78400, 1, 2240, 64), device='cuda:0', dtype=torch.float32)
    convolution_23 = rand_strided((32, 96, 35, 35), (117600, 1, 3360, 96), device='cuda:0', dtype=torch.float32)
    squeeze_70 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_23 = rand_strided((32, 96, 35, 35), (117600, 1, 3360, 96), device='cuda:0', dtype=torch.float32)
    convolution_24 = rand_strided((32, 96, 35, 35), (117600, 1, 3360, 96), device='cuda:0', dtype=torch.float32)
    squeeze_73 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    avg_pool2d_2 = rand_strided((32, 288, 35, 35), (352800, 1, 10080, 288), device='cuda:0', dtype=torch.float32)
    convolution_25 = rand_strided((32, 64, 35, 35), (78400, 1, 2240, 64), device='cuda:0', dtype=torch.float32)
    squeeze_76 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_3 = rand_strided((32, 288, 35, 35), (352800, 1, 10080, 288), device='cuda:0', dtype=torch.float32)
    convolution_26 = rand_strided((32, 384, 17, 17), (110976, 1, 6528, 384), device='cuda:0', dtype=torch.float32)
    squeeze_79 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_27 = rand_strided((32, 64, 35, 35), (78400, 1, 2240, 64), device='cuda:0', dtype=torch.float32)
    squeeze_82 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_27 = rand_strided((32, 64, 35, 35), (78400, 1, 2240, 64), device='cuda:0', dtype=torch.float32)
    convolution_28 = rand_strided((32, 96, 35, 35), (117600, 1, 3360, 96), device='cuda:0', dtype=torch.float32)
    squeeze_85 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_28 = rand_strided((32, 96, 35, 35), (117600, 1, 3360, 96), device='cuda:0', dtype=torch.float32)
    convolution_29 = rand_strided((32, 96, 17, 17), (27744, 1, 1632, 96), device='cuda:0', dtype=torch.float32)
    squeeze_88 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_65 = rand_strided((32, 288, 17, 17), (83232, 1, 4896, 288), device='cuda:0', dtype=torch.int64)
    cat_4 = rand_strided((32, 768, 17, 17), (221952, 1, 13056, 768), device='cuda:0', dtype=torch.float32)
    convolution_30 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_91 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_31 = rand_strided((32, 128, 17, 17), (36992, 1, 2176, 128), device='cuda:0', dtype=torch.float32)
    squeeze_94 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_31 = rand_strided((32, 128, 17, 17), (36992, 1, 2176, 128), device='cuda:0', dtype=torch.float32)
    convolution_32 = rand_strided((32, 128, 17, 17), (36992, 1, 2176, 128), device='cuda:0', dtype=torch.float32)
    squeeze_97 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_32 = rand_strided((32, 128, 17, 17), (36992, 1, 2176, 128), device='cuda:0', dtype=torch.float32)
    convolution_33 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_100 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_34 = rand_strided((32, 128, 17, 17), (36992, 1, 2176, 128), device='cuda:0', dtype=torch.float32)
    squeeze_103 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_34 = rand_strided((32, 128, 17, 17), (36992, 1, 2176, 128), device='cuda:0', dtype=torch.float32)
    convolution_35 = rand_strided((32, 128, 17, 17), (36992, 1, 2176, 128), device='cuda:0', dtype=torch.float32)
    squeeze_106 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_35 = rand_strided((32, 128, 17, 17), (36992, 1, 2176, 128), device='cuda:0', dtype=torch.float32)
    convolution_36 = rand_strided((32, 128, 17, 17), (36992, 1, 2176, 128), device='cuda:0', dtype=torch.float32)
    squeeze_109 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_36 = rand_strided((32, 128, 17, 17), (36992, 1, 2176, 128), device='cuda:0', dtype=torch.float32)
    convolution_37 = rand_strided((32, 128, 17, 17), (36992, 1, 2176, 128), device='cuda:0', dtype=torch.float32)
    squeeze_112 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_37 = rand_strided((32, 128, 17, 17), (36992, 1, 2176, 128), device='cuda:0', dtype=torch.float32)
    convolution_38 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_115 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    avg_pool2d_3 = rand_strided((32, 768, 17, 17), (221952, 1, 13056, 768), device='cuda:0', dtype=torch.float32)
    convolution_39 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_118 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_5 = rand_strided((32, 768, 17, 17), (221952, 1, 13056, 768), device='cuda:0', dtype=torch.float32)
    convolution_40 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_121 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_41 = rand_strided((32, 160, 17, 17), (46240, 1, 2720, 160), device='cuda:0', dtype=torch.float32)
    squeeze_124 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_41 = rand_strided((32, 160, 17, 17), (46240, 1, 2720, 160), device='cuda:0', dtype=torch.float32)
    convolution_42 = rand_strided((32, 160, 17, 17), (46240, 1, 2720, 160), device='cuda:0', dtype=torch.float32)
    squeeze_127 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_42 = rand_strided((32, 160, 17, 17), (46240, 1, 2720, 160), device='cuda:0', dtype=torch.float32)
    convolution_43 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_130 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_44 = rand_strided((32, 160, 17, 17), (46240, 1, 2720, 160), device='cuda:0', dtype=torch.float32)
    squeeze_133 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_44 = rand_strided((32, 160, 17, 17), (46240, 1, 2720, 160), device='cuda:0', dtype=torch.float32)
    convolution_45 = rand_strided((32, 160, 17, 17), (46240, 1, 2720, 160), device='cuda:0', dtype=torch.float32)
    squeeze_136 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_45 = rand_strided((32, 160, 17, 17), (46240, 1, 2720, 160), device='cuda:0', dtype=torch.float32)
    convolution_46 = rand_strided((32, 160, 17, 17), (46240, 1, 2720, 160), device='cuda:0', dtype=torch.float32)
    squeeze_139 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_46 = rand_strided((32, 160, 17, 17), (46240, 1, 2720, 160), device='cuda:0', dtype=torch.float32)
    convolution_47 = rand_strided((32, 160, 17, 17), (46240, 1, 2720, 160), device='cuda:0', dtype=torch.float32)
    squeeze_142 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_47 = rand_strided((32, 160, 17, 17), (46240, 1, 2720, 160), device='cuda:0', dtype=torch.float32)
    convolution_48 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_145 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    avg_pool2d_4 = rand_strided((32, 768, 17, 17), (221952, 1, 13056, 768), device='cuda:0', dtype=torch.float32)
    convolution_49 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_148 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_6 = rand_strided((32, 768, 17, 17), (221952, 1, 13056, 768), device='cuda:0', dtype=torch.float32)
    convolution_50 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_151 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_51 = rand_strided((32, 160, 17, 17), (46240, 1, 2720, 160), device='cuda:0', dtype=torch.float32)
    squeeze_154 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_51 = rand_strided((32, 160, 17, 17), (46240, 1, 2720, 160), device='cuda:0', dtype=torch.float32)
    convolution_52 = rand_strided((32, 160, 17, 17), (46240, 1, 2720, 160), device='cuda:0', dtype=torch.float32)
    squeeze_157 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_52 = rand_strided((32, 160, 17, 17), (46240, 1, 2720, 160), device='cuda:0', dtype=torch.float32)
    convolution_53 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_160 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_54 = rand_strided((32, 160, 17, 17), (46240, 1, 2720, 160), device='cuda:0', dtype=torch.float32)
    squeeze_163 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_54 = rand_strided((32, 160, 17, 17), (46240, 1, 2720, 160), device='cuda:0', dtype=torch.float32)
    convolution_55 = rand_strided((32, 160, 17, 17), (46240, 1, 2720, 160), device='cuda:0', dtype=torch.float32)
    squeeze_166 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_55 = rand_strided((32, 160, 17, 17), (46240, 1, 2720, 160), device='cuda:0', dtype=torch.float32)
    convolution_56 = rand_strided((32, 160, 17, 17), (46240, 1, 2720, 160), device='cuda:0', dtype=torch.float32)
    squeeze_169 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_56 = rand_strided((32, 160, 17, 17), (46240, 1, 2720, 160), device='cuda:0', dtype=torch.float32)
    convolution_57 = rand_strided((32, 160, 17, 17), (46240, 1, 2720, 160), device='cuda:0', dtype=torch.float32)
    squeeze_172 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_57 = rand_strided((32, 160, 17, 17), (46240, 1, 2720, 160), device='cuda:0', dtype=torch.float32)
    convolution_58 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_175 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    avg_pool2d_5 = rand_strided((32, 768, 17, 17), (221952, 1, 13056, 768), device='cuda:0', dtype=torch.float32)
    convolution_59 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_178 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_7 = rand_strided((32, 768, 17, 17), (221952, 1, 13056, 768), device='cuda:0', dtype=torch.float32)
    convolution_60 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_181 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_61 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_184 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_61 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    convolution_62 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_187 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_62 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    convolution_63 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_190 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_64 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_193 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_64 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    convolution_65 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_196 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_65 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    convolution_66 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_199 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_66 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    convolution_67 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_202 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_67 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    convolution_68 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_205 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    avg_pool2d_6 = rand_strided((32, 768, 17, 17), (221952, 1, 13056, 768), device='cuda:0', dtype=torch.float32)
    convolution_69 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_208 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_8 = rand_strided((32, 768, 17, 17), (221952, 1, 13056, 768), device='cuda:0', dtype=torch.float32)
    avg_pool2d_7 = rand_strided((32, 768, 5, 5), (19200, 1, 3840, 768), device='cuda:0', dtype=torch.float32)
    convolution_70 = rand_strided((32, 128, 5, 5), (3200, 1, 640, 128), device='cuda:0', dtype=torch.float32)
    squeeze_211 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_70 = rand_strided((32, 128, 5, 5), (3200, 1, 640, 128), device='cuda:0', dtype=torch.float32)
    convolution_71 = rand_strided((32, 768, 1, 1), (768, 1, 768, 768), device='cuda:0', dtype=torch.float32)
    squeeze_214 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    view = rand_strided((32, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convolution_72 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_217 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_72 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    convolution_73 = rand_strided((32, 320, 8, 8), (20480, 1, 2560, 320), device='cuda:0', dtype=torch.float32)
    squeeze_220 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_74 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_223 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_74 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    convolution_75 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_226 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_75 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    convolution_76 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_229 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_76 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    convolution_77 = rand_strided((32, 192, 8, 8), (12288, 1, 1536, 192), device='cuda:0', dtype=torch.float32)
    squeeze_232 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_163 = rand_strided((32, 768, 8, 8), (49152, 1, 6144, 768), device='cuda:0', dtype=torch.int64)
    cat_9 = rand_strided((32, 1280, 8, 8), (81920, 1, 10240, 1280), device='cuda:0', dtype=torch.float32)
    convolution_78 = rand_strided((32, 320, 8, 8), (20480, 1, 2560, 320), device='cuda:0', dtype=torch.float32)
    squeeze_235 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_79 = rand_strided((32, 384, 8, 8), (24576, 1, 3072, 384), device='cuda:0', dtype=torch.float32)
    squeeze_238 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_79 = rand_strided((32, 384, 8, 8), (24576, 1, 3072, 384), device='cuda:0', dtype=torch.float32)
    convolution_80 = rand_strided((32, 384, 8, 8), (24576, 1, 3072, 384), device='cuda:0', dtype=torch.float32)
    squeeze_241 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_81 = rand_strided((32, 384, 8, 8), (24576, 1, 3072, 384), device='cuda:0', dtype=torch.float32)
    squeeze_244 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_82 = rand_strided((32, 448, 8, 8), (28672, 1, 3584, 448), device='cuda:0', dtype=torch.float32)
    squeeze_247 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_82 = rand_strided((32, 448, 8, 8), (28672, 1, 3584, 448), device='cuda:0', dtype=torch.float32)
    convolution_83 = rand_strided((32, 384, 8, 8), (24576, 1, 3072, 384), device='cuda:0', dtype=torch.float32)
    squeeze_250 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_83 = rand_strided((32, 384, 8, 8), (24576, 1, 3072, 384), device='cuda:0', dtype=torch.float32)
    convolution_84 = rand_strided((32, 384, 8, 8), (24576, 1, 3072, 384), device='cuda:0', dtype=torch.float32)
    squeeze_253 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_85 = rand_strided((32, 384, 8, 8), (24576, 1, 3072, 384), device='cuda:0', dtype=torch.float32)
    squeeze_256 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    avg_pool2d_8 = rand_strided((32, 1280, 8, 8), (81920, 1, 10240, 1280), device='cuda:0', dtype=torch.float32)
    convolution_86 = rand_strided((32, 192, 8, 8), (12288, 1, 1536, 192), device='cuda:0', dtype=torch.float32)
    squeeze_259 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_12 = rand_strided((32, 2048, 8, 8), (131072, 1, 16384, 2048), device='cuda:0', dtype=torch.float32)
    convolution_87 = rand_strided((32, 320, 8, 8), (20480, 1, 2560, 320), device='cuda:0', dtype=torch.float32)
    squeeze_262 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_88 = rand_strided((32, 384, 8, 8), (24576, 1, 3072, 384), device='cuda:0', dtype=torch.float32)
    squeeze_265 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_88 = rand_strided((32, 384, 8, 8), (24576, 1, 3072, 384), device='cuda:0', dtype=torch.float32)
    convolution_89 = rand_strided((32, 384, 8, 8), (24576, 1, 3072, 384), device='cuda:0', dtype=torch.float32)
    squeeze_268 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_90 = rand_strided((32, 384, 8, 8), (24576, 1, 3072, 384), device='cuda:0', dtype=torch.float32)
    squeeze_271 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_91 = rand_strided((32, 448, 8, 8), (28672, 1, 3584, 448), device='cuda:0', dtype=torch.float32)
    squeeze_274 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_91 = rand_strided((32, 448, 8, 8), (28672, 1, 3584, 448), device='cuda:0', dtype=torch.float32)
    convolution_92 = rand_strided((32, 384, 8, 8), (24576, 1, 3072, 384), device='cuda:0', dtype=torch.float32)
    squeeze_277 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_92 = rand_strided((32, 384, 8, 8), (24576, 1, 3072, 384), device='cuda:0', dtype=torch.float32)
    convolution_93 = rand_strided((32, 384, 8, 8), (24576, 1, 3072, 384), device='cuda:0', dtype=torch.float32)
    squeeze_280 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_94 = rand_strided((32, 384, 8, 8), (24576, 1, 3072, 384), device='cuda:0', dtype=torch.float32)
    squeeze_283 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    avg_pool2d_9 = rand_strided((32, 2048, 8, 8), (131072, 1, 16384, 2048), device='cuda:0', dtype=torch.float32)
    convolution_95 = rand_strided((32, 192, 8, 8), (12288, 1, 1536, 192), device='cuda:0', dtype=torch.float32)
    squeeze_286 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    gt = rand_strided((32, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.bool)
    view_1 = rand_strided((32, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_2 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    le = rand_strided((32, 192, 8, 8), (12288, 1, 1536, 192), device='cuda:0', dtype=torch.bool)
    unsqueeze_389 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_1 = rand_strided((32, 384, 8, 8), (24576, 1, 3072, 384), device='cuda:0', dtype=torch.bool)
    unsqueeze_401 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_2 = rand_strided((32, 384, 8, 8), (24576, 1, 3072, 384), device='cuda:0', dtype=torch.bool)
    unsqueeze_413 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_425 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_437 = rand_strided((1, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_5 = rand_strided((32, 384, 8, 8), (24576, 1, 3072, 384), device='cuda:0', dtype=torch.bool)
    unsqueeze_449 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_6 = rand_strided((32, 384, 8, 8), (24576, 1, 3072, 384), device='cuda:0', dtype=torch.bool)
    unsqueeze_461 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_473 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_8 = rand_strided((32, 320, 8, 8), (20480, 1, 2560, 320), device='cuda:0', dtype=torch.bool)
    unsqueeze_485 = rand_strided((1, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_9 = rand_strided((32, 192, 8, 8), (12288, 1, 1536, 192), device='cuda:0', dtype=torch.bool)
    unsqueeze_497 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_10 = rand_strided((32, 384, 8, 8), (24576, 1, 3072, 384), device='cuda:0', dtype=torch.bool)
    unsqueeze_509 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_11 = rand_strided((32, 384, 8, 8), (24576, 1, 3072, 384), device='cuda:0', dtype=torch.bool)
    unsqueeze_521 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_533 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_545 = rand_strided((1, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_14 = rand_strided((32, 384, 8, 8), (24576, 1, 3072, 384), device='cuda:0', dtype=torch.bool)
    unsqueeze_557 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_15 = rand_strided((32, 384, 8, 8), (24576, 1, 3072, 384), device='cuda:0', dtype=torch.bool)
    unsqueeze_569 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_581 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_17 = rand_strided((32, 320, 8, 8), (20480, 1, 2560, 320), device='cuda:0', dtype=torch.bool)
    unsqueeze_593 = rand_strided((1, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_18 = rand_strided((32, 192, 8, 8), (12288, 1, 1536, 192), device='cuda:0', dtype=torch.bool)
    unsqueeze_605 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_617 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_629 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_641 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_22 = rand_strided((32, 320, 8, 8), (20480, 1, 2560, 320), device='cuda:0', dtype=torch.bool)
    unsqueeze_653 = rand_strided((1, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_665 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_6 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    le_24 = rand_strided((32, 768, 1, 1), (768, 1, 768, 768), device='cuda:0', dtype=torch.bool)
    unsqueeze_677 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_689 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_26 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.bool)
    unsqueeze_701 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_27 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.bool)
    unsqueeze_713 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_725 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_737 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_749 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_761 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_32 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.bool)
    unsqueeze_773 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_785 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_797 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_35 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.bool)
    unsqueeze_809 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_36 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.bool)
    unsqueeze_821 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_37 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.bool)
    unsqueeze_833 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_845 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_857 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_869 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_881 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_42 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.bool)
    unsqueeze_893 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_905 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_917 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_45 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.bool)
    unsqueeze_929 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_46 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.bool)
    unsqueeze_941 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_47 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.bool)
    unsqueeze_953 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_965 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_977 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_989 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1001 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_52 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.bool)
    unsqueeze_1013 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1025 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1037 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_55 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.bool)
    unsqueeze_1049 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_56 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.bool)
    unsqueeze_1061 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_57 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.bool)
    unsqueeze_1073 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1085 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1097 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1109 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1121 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_62 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.bool)
    unsqueeze_1133 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1145 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1157 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_65 = rand_strided((32, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.bool)
    unsqueeze_1169 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_66 = rand_strided((32, 96, 17, 17), (27744, 1, 1632, 96), device='cuda:0', dtype=torch.bool)
    unsqueeze_1181 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1193 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1205 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_69 = rand_strided((32, 384, 17, 17), (110976, 1, 6528, 384), device='cuda:0', dtype=torch.bool)
    unsqueeze_1217 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_70 = rand_strided((32, 64, 35, 35), (78400, 1, 2240, 64), device='cuda:0', dtype=torch.bool)
    unsqueeze_1229 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_71 = rand_strided((32, 96, 35, 35), (117600, 1, 3360, 96), device='cuda:0', dtype=torch.bool)
    unsqueeze_1241 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1253 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1265 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_74 = rand_strided((32, 64, 35, 35), (78400, 1, 2240, 64), device='cuda:0', dtype=torch.bool)
    unsqueeze_1277 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1289 = rand_strided((1, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_76 = rand_strided((32, 64, 35, 35), (78400, 1, 2240, 64), device='cuda:0', dtype=torch.bool)
    unsqueeze_1301 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_77 = rand_strided((32, 64, 35, 35), (78400, 1, 2240, 64), device='cuda:0', dtype=torch.bool)
    unsqueeze_1313 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_78 = rand_strided((32, 96, 35, 35), (117600, 1, 3360, 96), device='cuda:0', dtype=torch.bool)
    unsqueeze_1325 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1337 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1349 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_81 = rand_strided((32, 64, 35, 35), (78400, 1, 2240, 64), device='cuda:0', dtype=torch.bool)
    unsqueeze_1361 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1373 = rand_strided((1, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_83 = rand_strided((32, 64, 35, 35), (78400, 1, 2240, 64), device='cuda:0', dtype=torch.bool)
    unsqueeze_1385 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_84 = rand_strided((32, 32, 35, 35), (39200, 1, 1120, 32), device='cuda:0', dtype=torch.bool)
    unsqueeze_1397 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_85 = rand_strided((32, 96, 35, 35), (117600, 1, 3360, 96), device='cuda:0', dtype=torch.bool)
    unsqueeze_1409 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1421 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1433 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_88 = rand_strided((32, 64, 35, 35), (78400, 1, 2240, 64), device='cuda:0', dtype=torch.bool)
    unsqueeze_1445 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1457 = rand_strided((1, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_90 = rand_strided((32, 64, 35, 35), (78400, 1, 2240, 64), device='cuda:0', dtype=torch.bool)
    unsqueeze_1469 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1481 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1493 = rand_strided((1, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1505 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1517 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1529 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((32, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((32, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_160, primals_161, primals_163, primals_164, primals_166, primals_167, primals_169, primals_170, primals_172, primals_173, primals_175, primals_176, primals_178, primals_179, primals_181, primals_182, primals_184, primals_185, primals_187, primals_188, primals_190, primals_191, primals_193, primals_194, primals_196, primals_197, primals_199, primals_200, primals_202, primals_203, primals_205, primals_206, primals_208, primals_209, primals_211, primals_212, primals_214, primals_215, primals_219, primals_220, primals_222, primals_223, primals_225, primals_226, primals_228, primals_229, primals_231, primals_232, primals_234, primals_235, primals_237, primals_238, primals_240, primals_241, primals_243, primals_244, primals_246, primals_247, primals_249, primals_250, primals_252, primals_253, primals_255, primals_256, primals_258, primals_259, primals_261, primals_262, primals_264, primals_265, primals_267, primals_268, primals_270, primals_271, primals_273, primals_274, primals_276, primals_277, primals_279, primals_280, primals_282, primals_283, primals_285, primals_286, primals_288, primals_289, cat, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, getitem_6, getitem_7, convolution_3, squeeze_10, relu_3, convolution_4, squeeze_13, relu_4, getitem_12, getitem_13, convolution_5, squeeze_16, convolution_6, squeeze_19, relu_6, convolution_7, squeeze_22, convolution_8, squeeze_25, relu_8, convolution_9, squeeze_28, relu_9, convolution_10, squeeze_31, avg_pool2d, convolution_11, squeeze_34, cat_1, convolution_12, squeeze_37, convolution_13, squeeze_40, relu_13, convolution_14, squeeze_43, convolution_15, squeeze_46, relu_15, convolution_16, squeeze_49, relu_16, convolution_17, squeeze_52, avg_pool2d_1, convolution_18, squeeze_55, cat_2, convolution_19, squeeze_58, convolution_20, squeeze_61, relu_20, convolution_21, squeeze_64, convolution_22, squeeze_67, relu_22, convolution_23, squeeze_70, relu_23, convolution_24, squeeze_73, avg_pool2d_2, convolution_25, squeeze_76, cat_3, convolution_26, squeeze_79, convolution_27, squeeze_82, relu_27, convolution_28, squeeze_85, relu_28, convolution_29, squeeze_88, getitem_65, cat_4, convolution_30, squeeze_91, convolution_31, squeeze_94, relu_31, convolution_32, squeeze_97, relu_32, convolution_33, squeeze_100, convolution_34, squeeze_103, relu_34, convolution_35, squeeze_106, relu_35, convolution_36, squeeze_109, relu_36, convolution_37, squeeze_112, relu_37, convolution_38, squeeze_115, avg_pool2d_3, convolution_39, squeeze_118, cat_5, convolution_40, squeeze_121, convolution_41, squeeze_124, relu_41, convolution_42, squeeze_127, relu_42, convolution_43, squeeze_130, convolution_44, squeeze_133, relu_44, convolution_45, squeeze_136, relu_45, convolution_46, squeeze_139, relu_46, convolution_47, squeeze_142, relu_47, convolution_48, squeeze_145, avg_pool2d_4, convolution_49, squeeze_148, cat_6, convolution_50, squeeze_151, convolution_51, squeeze_154, relu_51, convolution_52, squeeze_157, relu_52, convolution_53, squeeze_160, convolution_54, squeeze_163, relu_54, convolution_55, squeeze_166, relu_55, convolution_56, squeeze_169, relu_56, convolution_57, squeeze_172, relu_57, convolution_58, squeeze_175, avg_pool2d_5, convolution_59, squeeze_178, cat_7, convolution_60, squeeze_181, convolution_61, squeeze_184, relu_61, convolution_62, squeeze_187, relu_62, convolution_63, squeeze_190, convolution_64, squeeze_193, relu_64, convolution_65, squeeze_196, relu_65, convolution_66, squeeze_199, relu_66, convolution_67, squeeze_202, relu_67, convolution_68, squeeze_205, avg_pool2d_6, convolution_69, squeeze_208, cat_8, avg_pool2d_7, convolution_70, squeeze_211, relu_70, convolution_71, squeeze_214, view, convolution_72, squeeze_217, relu_72, convolution_73, squeeze_220, convolution_74, squeeze_223, relu_74, convolution_75, squeeze_226, relu_75, convolution_76, squeeze_229, relu_76, convolution_77, squeeze_232, getitem_163, cat_9, convolution_78, squeeze_235, convolution_79, squeeze_238, relu_79, convolution_80, squeeze_241, convolution_81, squeeze_244, convolution_82, squeeze_247, relu_82, convolution_83, squeeze_250, relu_83, convolution_84, squeeze_253, convolution_85, squeeze_256, avg_pool2d_8, convolution_86, squeeze_259, cat_12, convolution_87, squeeze_262, convolution_88, squeeze_265, relu_88, convolution_89, squeeze_268, convolution_90, squeeze_271, convolution_91, squeeze_274, relu_91, convolution_92, squeeze_277, relu_92, convolution_93, squeeze_280, convolution_94, squeeze_283, avg_pool2d_9, convolution_95, squeeze_286, gt, view_1, permute_2, le, unsqueeze_389, le_1, unsqueeze_401, le_2, unsqueeze_413, unsqueeze_425, unsqueeze_437, le_5, unsqueeze_449, le_6, unsqueeze_461, unsqueeze_473, le_8, unsqueeze_485, le_9, unsqueeze_497, le_10, unsqueeze_509, le_11, unsqueeze_521, unsqueeze_533, unsqueeze_545, le_14, unsqueeze_557, le_15, unsqueeze_569, unsqueeze_581, le_17, unsqueeze_593, le_18, unsqueeze_605, unsqueeze_617, unsqueeze_629, unsqueeze_641, le_22, unsqueeze_653, unsqueeze_665, permute_6, le_24, unsqueeze_677, unsqueeze_689, le_26, unsqueeze_701, le_27, unsqueeze_713, unsqueeze_725, unsqueeze_737, unsqueeze_749, unsqueeze_761, le_32, unsqueeze_773, unsqueeze_785, unsqueeze_797, le_35, unsqueeze_809, le_36, unsqueeze_821, le_37, unsqueeze_833, unsqueeze_845, unsqueeze_857, unsqueeze_869, unsqueeze_881, le_42, unsqueeze_893, unsqueeze_905, unsqueeze_917, le_45, unsqueeze_929, le_46, unsqueeze_941, le_47, unsqueeze_953, unsqueeze_965, unsqueeze_977, unsqueeze_989, unsqueeze_1001, le_52, unsqueeze_1013, unsqueeze_1025, unsqueeze_1037, le_55, unsqueeze_1049, le_56, unsqueeze_1061, le_57, unsqueeze_1073, unsqueeze_1085, unsqueeze_1097, unsqueeze_1109, unsqueeze_1121, le_62, unsqueeze_1133, unsqueeze_1145, unsqueeze_1157, le_65, unsqueeze_1169, le_66, unsqueeze_1181, unsqueeze_1193, unsqueeze_1205, le_69, unsqueeze_1217, le_70, unsqueeze_1229, le_71, unsqueeze_1241, unsqueeze_1253, unsqueeze_1265, le_74, unsqueeze_1277, unsqueeze_1289, le_76, unsqueeze_1301, le_77, unsqueeze_1313, le_78, unsqueeze_1325, unsqueeze_1337, unsqueeze_1349, le_81, unsqueeze_1361, unsqueeze_1373, le_83, unsqueeze_1385, le_84, unsqueeze_1397, le_85, unsqueeze_1409, unsqueeze_1421, unsqueeze_1433, le_88, unsqueeze_1445, unsqueeze_1457, le_90, unsqueeze_1469, unsqueeze_1481, unsqueeze_1493, unsqueeze_1505, unsqueeze_1517, unsqueeze_1529, tangents_1, tangents_2])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
