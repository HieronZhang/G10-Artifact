
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


# kernel path: /tmp/torchinductor_zhang402/p2/cp2mmojoteratl7c26dhokvkeraksmwjfklbzwbekeeiukx7op5j.py
# Source Nodes: [], Original ATen: [aten.tanh_backward]

triton_poi_fused_tanh_backward_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_tanh_backward_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = tmp1 * tmp1
    tmp3 = 1.0
    tmp4 = tmp3 - tmp2
    tmp5 = tmp0 * tmp4
    tl.store(out_ptr0 + (x0), tmp5, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_zhang402/ft/cftm4agbfyzieaqchijqmpzr2thjrzayggt2aazl4oqjusjqvprn.py
# Source Nodes: [], Original ATen: [aten.sum, aten.view]

triton_poi_fused_sum_view_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sum_view_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (768 + x0), xmask)
    tmp3 = tl.load(in_ptr0 + (1536 + x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/dg/cdgemnozuj6rp4o6lkwv4z2a23iqsq6kugan6ewfyzmkcmsyekvy.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward, aten.select_backward]

triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_select_backward_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_select_backward_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 15
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 5
    x1 = (xindex // 5)
    tmp0 = tl.load(in_ptr0 + (r2 + (768*x3)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr1 + (r2 + (768*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.load(in_ptr3 + (r2 + (768*x3)), rmask & xmask, other=0.0)
    tmp20 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr5 + (r2 + (768*x3)), rmask & xmask).to(tl.int1)
    tmp1 = x0
    tmp2 = tl.full([1], 0, tl.int32)
    tmp3 = tmp1 == tmp2
    tmp5 = 0.0
    tmp6 = tl.where(tmp3, tmp4, tmp5)
    tmp7 = tmp0 + tmp6
    tmp9 = tmp7 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask & xmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp15 = tmp9 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp21 = 768.0
    tmp22 = tmp9 * tmp21
    tmp23 = tmp22 - tmp13
    tmp24 = tmp14 * tmp19
    tmp25 = tmp23 - tmp24
    tmp26 = tmp20 * tmp25
    tmp28 = tmp27.to(tl.float32)
    tmp29 = 1.1111111111111112
    tmp30 = tmp28 * tmp29
    tmp31 = tmp26 * tmp30
    tl.store(out_ptr2 + (r2 + (768*x3)), tmp26, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (768*x3)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/4o/c4oaiy36phzva654imamcypzqv6mtgfubfogjjnn6pfeivth3mel.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.select_backward]

triton_per_fused_add_native_layer_norm_backward_select_backward_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 16],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_select_backward_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 15
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex
    r1 = rindex % 5
    r2 = (rindex // 5)
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r3)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr1 + (x0 + (768*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr2 + (x0 + (768*r3)), rmask & xmask, other=0.0)
    tmp1 = r1
    tmp2 = tl.full([1, 1], 0, tl.int32)
    tmp3 = tmp1 == tmp2
    tmp5 = 0.0
    tmp6 = tl.where(tmp3, tmp4, tmp5)
    tmp7 = tmp0 + tmp6
    tmp9 = tmp7 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp12 = tl.where(rmask & xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tmp14 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/sm/csmeag6npkos2ubwu5td6pahpdde7pcky46fpzyidp4r2tpr7ipn.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 16],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 15
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/ip/cipezeiycmmmyccbiorrlzsfupq2jrfpennvplqt7kbacbc4hd53.py
# Source Nodes: [intermediate_output_11], Original ATen: [aten.gelu, aten.gelu_backward]
# intermediate_output_11 => add_96, erf_11, mul_155
triton_poi_fused_gelu_gelu_backward_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_5', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 46080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = 0.7071067811865476
    tmp3 = tmp1 * tmp2
    tmp4 = tl.math.erf(tmp3)
    tmp5 = 1.0
    tmp6 = tmp4 + tmp5
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp1 * tmp1
    tmp10 = -0.5
    tmp11 = tmp9 * tmp10
    tmp12 = tl.exp(tmp11)
    tmp13 = 0.3989422804014327
    tmp14 = tmp12 * tmp13
    tmp15 = tmp1 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tmp0 * tmp16
    tl.store(in_out_ptr0 + (x0), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/qv/cqvovehzgu57fvkomy6ahr3pjogb5nthvqvbfqrpfkg4cvygigno.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 16],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 15
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (3072*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/4f/c4fqenubi3eewqueftte5vgsin4vrb3yipe6qtxo4pnsaon27fld.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 15
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask).to(tl.int1)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp10 = tmp4 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = 768.0
    tmp17 = tmp4 * tmp16
    tmp18 = tmp17 - tmp8
    tmp19 = tmp9 * tmp14
    tmp20 = tmp18 - tmp19
    tmp21 = tmp15 * tmp20
    tmp23 = tmp22.to(tl.float32)
    tmp24 = 1.1111111111111112
    tmp25 = tmp23 * tmp24
    tmp26 = tmp21 * tmp25
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp26, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/fw/cfw6x2pcizbqfazzakslel3o6oiatholvjnybeklsyjptbbl7vqb.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 16],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 15
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None]
    tmp9 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp8, xmask)
    tl.store(out_ptr1 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/ea/cealegtxfccirca5a7ltc7qhabffbko3jhic7gtqahoqtq6f66hh.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 11520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 5
    x2 = (xindex // 320) % 12
    x3 = (xindex // 3840)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (768*x1) + (3840*x3)), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/xf/cxfxv2pz6h3tyaj4jllaeunlsrpisdjxvbkqnnfbtr66rqjx5g7o.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.native_dropout_backward]

triton_poi_fused__softmax_backward_data_native_dropout_backward_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_backward_data_native_dropout_backward_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 180
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (5*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (5*x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp6 = tl.load(in_ptr2 + (5*x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (1 + (5*x0)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (1 + (5*x0)), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp13 = tl.load(in_ptr2 + (1 + (5*x0)), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (2 + (5*x0)), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr1 + (2 + (5*x0)), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp21 = tl.load(in_ptr2 + (2 + (5*x0)), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr0 + (3 + (5*x0)), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr1 + (3 + (5*x0)), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp29 = tl.load(in_ptr2 + (3 + (5*x0)), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr0 + (4 + (5*x0)), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr1 + (4 + (5*x0)), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp37 = tl.load(in_ptr2 + (4 + (5*x0)), xmask, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.1111111111111112
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = tmp5 * tmp6
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp10 * tmp3
    tmp12 = tmp8 * tmp11
    tmp14 = tmp12 * tmp13
    tmp15 = tmp7 + tmp14
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp18 * tmp3
    tmp20 = tmp16 * tmp19
    tmp22 = tmp20 * tmp21
    tmp23 = tmp15 + tmp22
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp26 * tmp3
    tmp28 = tmp24 * tmp27
    tmp30 = tmp28 * tmp29
    tmp31 = tmp23 + tmp30
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp34 * tmp3
    tmp36 = tmp32 * tmp35
    tmp38 = tmp36 * tmp37
    tmp39 = tmp31 + tmp38
    tl.store(out_ptr0 + (x0), tmp39, xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/yt/cyt3sbzm7seqa7ha46bffuuv7zqwin2qo4v3efwas5ds3duvmeej.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div, aten.native_dropout_backward]

triton_poi_fused__softmax_backward_data_div_native_dropout_backward_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_backward_data_div_native_dropout_backward_11', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 900
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 5)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x2), xmask).to(tl.int1)
    tmp6 = tl.load(in_ptr1 + (x2), xmask)
    tmp8 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.1111111111111112
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = tmp5 * tmp6
    tmp9 = tmp6 * tmp8
    tmp10 = tmp7 - tmp9
    tmp11 = 8.0
    tmp12 = tmp10 / tmp11
    tl.store(in_out_ptr0 + (x2), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/rz/crzogyddi2yd6q4oabuxnnqdmz747grelc7xs3x4u3tr3p2qjs4a.py
# Source Nodes: [], Original ATen: [aten.view]

triton_poi_fused_view_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 11520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 768
    x1 = (xindex // 768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*(x1 % 5)) + (320*(x0 // 64)) + (3840*(x1 // 5)) + (x0 % 64)), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/cw/ccwshghe2j7sfk5vkb6zabx7vmihbwgppjvvi5ketjhuft5wd6tu.py
# Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]

triton_poi_fused__unsafe_view_clone_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 15
    xnumel = 768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + ((5*x1) + (3840*(y0 // 5)) + (y0 % 5)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x1 + (768*y0)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_zhang402/ey/ceybc4cdspobdp2hkvuxkqcsrpz3e7rtxt3qocr2jbsgosyerkir.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 15
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp19 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr7 + (r1 + (768*x0)), rmask & xmask).to(tl.int1)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = tmp8 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp20 = 768.0
    tmp21 = tmp8 * tmp20
    tmp22 = tmp21 - tmp12
    tmp23 = tmp13 * tmp18
    tmp24 = tmp22 - tmp23
    tmp25 = tmp19 * tmp24
    tmp27 = tmp26.to(tl.float32)
    tmp28 = 1.1111111111111112
    tmp29 = tmp27 * tmp28
    tmp30 = tmp25 * tmp29
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp25, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp30, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/yq/cyq7zjn64fet327tpacjrqv2zxxusroyy2xwkhe7vstlo24efwdx.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 16],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 15
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr4 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp13 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp12, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/6s/c6skrkk66bl2ah5ysypzy4hd4dtt5colsje7ixzlybj2hgx7j6v6.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/my/cmy3zaerbkjtzf4ch2ey7dejqlx2i3gyaczcxyvgyjvbxfu3upt3.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 23440896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/ok/coksqbre7tdschzezsgyzuzkuq6m7e2v36qrk23aurlkf7tfskd6.py
# Source Nodes: [], Original ATen: [aten.add, aten.embedding_dense_backward, aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*i1', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i64', 9: '*i64', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(14,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_18', 'mutated_arg_names': ['in_out_ptr0', 'out_ptr3', 'out_ptr4']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 15
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x2 = xindex % 5
    tmp0 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask).to(tl.int1)
    tmp12 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp7.to(tl.float32)
    tmp9 = 1.1111111111111112
    tmp10 = tmp8 * tmp9
    tmp11 = tmp6 * tmp10
    tmp13 = tmp11 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp19 = tmp13 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp22 = tl.where(rmask & xmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp25 = 768.0
    tmp26 = tmp13 * tmp25
    tmp27 = tmp26 - tmp17
    tmp28 = tmp18 * tmp23
    tmp29 = tmp27 - tmp28
    tmp30 = tmp24 * tmp29
    tmp32 = tmp31 + 2
    tmp33 = tmp31 < 0
    tmp34 = tl.where(tmp33, tmp32, tmp31)
    tmp35 = tl.full([1], -1, tl.int64)
    tmp36 = tmp31 == tmp35
    tmp37 = 0.0
    tmp38 = tl.where(tmp36, tmp37, tmp30)
    tmp40 = tmp39 + 30522
    tmp41 = tmp39 < 0
    tmp42 = tl.where(tmp41, tmp40, tmp39)
    tmp43 = tl.full([1], 0, tl.int64)
    tmp44 = tmp39 == tmp43
    tmp45 = tl.where(tmp44, tmp37, tmp30)
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp11, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp30, rmask & xmask)
    tl.atomic_add(out_ptr3 + (tl.broadcast_to(r1 + (768*tmp34), [RBLOCK])), tmp38, rmask & xmask)
    tl.atomic_add(out_ptr4 + (tl.broadcast_to(r1 + (768*tmp42), [RBLOCK])), tmp45, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/ox/coxhbgeooi6ygijly3cwk57gds7drrm3iccfsi2xfafp2ygi7i5u.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 16],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_19', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 15
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/k6/ck6kst3kvx2dqwknmyds5xcqal5mt2bppjhuv4tyo5v7xmxqqhtk.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_zhang402/vu/cvuyvcflqm4p6447uhzfi7j3ksbflvtj5k37zmmtcny3p2kijsz7.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward, aten.sum]

triton_poi_fused_embedding_dense_backward_sum_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_sum_21', 'mutated_arg_names': ['out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 768)
    x2 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x2), xmask)
    tmp7 = tl.load(in_ptr1 + (3840 + x2), xmask)
    tmp9 = tl.load(in_ptr1 + (7680 + x2), xmask)
    tmp1 = tmp0 + 512
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tmp4 = tl.full([1], -1, tl.int64)
    tmp5 = tmp0 == tmp4
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp11 = 0.0
    tmp12 = tl.where(tmp5, tmp11, tmp10)
    tl.atomic_add(out_ptr0 + (x0 + (768*tmp3)), tmp12, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_4, primals_14, primals_20, primals_30, primals_36, primals_46, primals_52, primals_62, primals_68, primals_78, primals_84, primals_94, primals_100, primals_110, primals_116, primals_126, primals_132, primals_142, primals_148, primals_158, primals_164, primals_174, primals_180, primals_190, primals_196, primals_202, expand, slice_6, mul_1, gt, view, gt_1, view_16, gt_2, mul_9, view_18, addmm_4, view_20, gt_3, mul_16, view_22, gt_4, view_38, gt_5, mul_22, view_40, addmm_10, view_42, gt_6, mul_29, view_44, gt_7, view_60, gt_8, mul_35, view_62, addmm_16, view_64, gt_9, mul_42, view_66, gt_10, view_82, gt_11, mul_48, view_84, addmm_22, view_86, gt_12, mul_55, view_88, gt_13, view_104, gt_14, mul_61, view_106, addmm_28, view_108, gt_15, mul_68, view_110, gt_16, view_126, gt_17, mul_74, view_128, addmm_34, view_130, gt_18, mul_81, view_132, gt_19, view_148, gt_20, mul_87, view_150, addmm_40, view_152, gt_21, mul_94, view_154, gt_22, view_170, gt_23, mul_100, view_172, addmm_46, view_174, gt_24, mul_107, view_176, gt_25, view_192, gt_26, mul_113, view_194, addmm_52, view_196, gt_27, mul_120, view_198, gt_28, view_214, gt_29, mul_126, view_216, addmm_58, view_218, gt_30, mul_133, view_220, gt_31, view_236, gt_32, mul_139, view_238, addmm_64, view_240, gt_33, mul_146, view_242, gt_34, view_258, gt_35, mul_152, view_260, addmm_70, view_262, gt_36, mul_159, select, tanh, permute_133, div_24, permute_137, permute_141, div_25, permute_145, permute_150, permute_151, alias_14, permute_152, permute_153, permute_157, permute_162, permute_166, div_27, permute_170, permute_174, div_28, permute_178, permute_183, permute_184, alias_15, permute_185, permute_186, permute_190, permute_195, permute_199, div_30, permute_203, permute_207, div_31, permute_211, permute_216, permute_217, alias_16, permute_218, permute_219, permute_223, permute_228, permute_232, div_33, permute_236, permute_240, div_34, permute_244, permute_249, permute_250, alias_17, permute_251, permute_252, permute_256, permute_261, permute_265, div_36, permute_269, permute_273, div_37, permute_277, permute_282, permute_283, alias_18, permute_284, permute_285, permute_289, permute_294, permute_298, div_39, permute_302, permute_306, div_40, permute_310, permute_315, permute_316, alias_19, permute_317, permute_318, permute_322, permute_327, permute_331, div_42, permute_335, permute_339, div_43, permute_343, permute_348, permute_349, alias_20, permute_350, permute_351, permute_355, permute_360, permute_364, div_45, permute_368, permute_372, div_46, permute_376, permute_381, permute_382, alias_21, permute_383, permute_384, permute_388, permute_393, permute_397, div_48, permute_401, permute_405, div_49, permute_409, permute_414, permute_415, alias_22, permute_416, permute_417, permute_421, permute_426, permute_430, div_51, permute_434, permute_438, div_52, permute_442, permute_447, permute_448, alias_23, permute_449, permute_450, permute_454, permute_459, permute_463, div_54, permute_467, permute_471, div_55, permute_475, permute_480, permute_481, alias_24, permute_482, permute_483, permute_487, permute_492, permute_496, div_57, permute_500, permute_504, div_58, permute_508, permute_513, permute_514, alias_25, permute_515, permute_516, permute_520, permute_525, permute_529, div_60, tangents_1, tangents_2 = args
    args.clear()
    assert_size_stride(primals_4, (768, ), (1, ))
    assert_size_stride(primals_14, (768, ), (1, ))
    assert_size_stride(primals_20, (768, ), (1, ))
    assert_size_stride(primals_30, (768, ), (1, ))
    assert_size_stride(primals_36, (768, ), (1, ))
    assert_size_stride(primals_46, (768, ), (1, ))
    assert_size_stride(primals_52, (768, ), (1, ))
    assert_size_stride(primals_62, (768, ), (1, ))
    assert_size_stride(primals_68, (768, ), (1, ))
    assert_size_stride(primals_78, (768, ), (1, ))
    assert_size_stride(primals_84, (768, ), (1, ))
    assert_size_stride(primals_94, (768, ), (1, ))
    assert_size_stride(primals_100, (768, ), (1, ))
    assert_size_stride(primals_110, (768, ), (1, ))
    assert_size_stride(primals_116, (768, ), (1, ))
    assert_size_stride(primals_126, (768, ), (1, ))
    assert_size_stride(primals_132, (768, ), (1, ))
    assert_size_stride(primals_142, (768, ), (1, ))
    assert_size_stride(primals_148, (768, ), (1, ))
    assert_size_stride(primals_158, (768, ), (1, ))
    assert_size_stride(primals_164, (768, ), (1, ))
    assert_size_stride(primals_174, (768, ), (1, ))
    assert_size_stride(primals_180, (768, ), (1, ))
    assert_size_stride(primals_190, (768, ), (1, ))
    assert_size_stride(primals_196, (768, ), (1, ))
    assert_size_stride(primals_202, (3, 5), (5, 1))
    assert_size_stride(expand, (3, 5), (0, 1))
    assert_size_stride(slice_6, (1, 5), (512, 1))
    assert_size_stride(mul_1, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(gt, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(view, (15, 768), (768, 1))
    assert_size_stride(gt_1, (3, 12, 5, 5), (300, 25, 5, 1))
    assert_size_stride(view_16, (15, 768), (768, 1))
    assert_size_stride(gt_2, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(mul_9, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(view_18, (15, 768), (768, 1))
    assert_size_stride(addmm_4, (15, 3072), (3072, 1))
    assert_size_stride(view_20, (15, 3072), (3072, 1))
    assert_size_stride(gt_3, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(mul_16, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(view_22, (15, 768), (768, 1))
    assert_size_stride(gt_4, (3, 12, 5, 5), (300, 25, 5, 1))
    assert_size_stride(view_38, (15, 768), (768, 1))
    assert_size_stride(gt_5, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(mul_22, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(view_40, (15, 768), (768, 1))
    assert_size_stride(addmm_10, (15, 3072), (3072, 1))
    assert_size_stride(view_42, (15, 3072), (3072, 1))
    assert_size_stride(gt_6, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(mul_29, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(view_44, (15, 768), (768, 1))
    assert_size_stride(gt_7, (3, 12, 5, 5), (300, 25, 5, 1))
    assert_size_stride(view_60, (15, 768), (768, 1))
    assert_size_stride(gt_8, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(mul_35, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(view_62, (15, 768), (768, 1))
    assert_size_stride(addmm_16, (15, 3072), (3072, 1))
    assert_size_stride(view_64, (15, 3072), (3072, 1))
    assert_size_stride(gt_9, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(mul_42, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(view_66, (15, 768), (768, 1))
    assert_size_stride(gt_10, (3, 12, 5, 5), (300, 25, 5, 1))
    assert_size_stride(view_82, (15, 768), (768, 1))
    assert_size_stride(gt_11, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(mul_48, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(view_84, (15, 768), (768, 1))
    assert_size_stride(addmm_22, (15, 3072), (3072, 1))
    assert_size_stride(view_86, (15, 3072), (3072, 1))
    assert_size_stride(gt_12, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(mul_55, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(view_88, (15, 768), (768, 1))
    assert_size_stride(gt_13, (3, 12, 5, 5), (300, 25, 5, 1))
    assert_size_stride(view_104, (15, 768), (768, 1))
    assert_size_stride(gt_14, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(mul_61, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(view_106, (15, 768), (768, 1))
    assert_size_stride(addmm_28, (15, 3072), (3072, 1))
    assert_size_stride(view_108, (15, 3072), (3072, 1))
    assert_size_stride(gt_15, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(mul_68, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(view_110, (15, 768), (768, 1))
    assert_size_stride(gt_16, (3, 12, 5, 5), (300, 25, 5, 1))
    assert_size_stride(view_126, (15, 768), (768, 1))
    assert_size_stride(gt_17, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(mul_74, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(view_128, (15, 768), (768, 1))
    assert_size_stride(addmm_34, (15, 3072), (3072, 1))
    assert_size_stride(view_130, (15, 3072), (3072, 1))
    assert_size_stride(gt_18, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(mul_81, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(view_132, (15, 768), (768, 1))
    assert_size_stride(gt_19, (3, 12, 5, 5), (300, 25, 5, 1))
    assert_size_stride(view_148, (15, 768), (768, 1))
    assert_size_stride(gt_20, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(mul_87, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(view_150, (15, 768), (768, 1))
    assert_size_stride(addmm_40, (15, 3072), (3072, 1))
    assert_size_stride(view_152, (15, 3072), (3072, 1))
    assert_size_stride(gt_21, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(mul_94, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(view_154, (15, 768), (768, 1))
    assert_size_stride(gt_22, (3, 12, 5, 5), (300, 25, 5, 1))
    assert_size_stride(view_170, (15, 768), (768, 1))
    assert_size_stride(gt_23, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(mul_100, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(view_172, (15, 768), (768, 1))
    assert_size_stride(addmm_46, (15, 3072), (3072, 1))
    assert_size_stride(view_174, (15, 3072), (3072, 1))
    assert_size_stride(gt_24, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(mul_107, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(view_176, (15, 768), (768, 1))
    assert_size_stride(gt_25, (3, 12, 5, 5), (300, 25, 5, 1))
    assert_size_stride(view_192, (15, 768), (768, 1))
    assert_size_stride(gt_26, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(mul_113, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(view_194, (15, 768), (768, 1))
    assert_size_stride(addmm_52, (15, 3072), (3072, 1))
    assert_size_stride(view_196, (15, 3072), (3072, 1))
    assert_size_stride(gt_27, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(mul_120, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(view_198, (15, 768), (768, 1))
    assert_size_stride(gt_28, (3, 12, 5, 5), (300, 25, 5, 1))
    assert_size_stride(view_214, (15, 768), (768, 1))
    assert_size_stride(gt_29, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(mul_126, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(view_216, (15, 768), (768, 1))
    assert_size_stride(addmm_58, (15, 3072), (3072, 1))
    assert_size_stride(view_218, (15, 3072), (3072, 1))
    assert_size_stride(gt_30, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(mul_133, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(view_220, (15, 768), (768, 1))
    assert_size_stride(gt_31, (3, 12, 5, 5), (300, 25, 5, 1))
    assert_size_stride(view_236, (15, 768), (768, 1))
    assert_size_stride(gt_32, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(mul_139, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(view_238, (15, 768), (768, 1))
    assert_size_stride(addmm_64, (15, 3072), (3072, 1))
    assert_size_stride(view_240, (15, 3072), (3072, 1))
    assert_size_stride(gt_33, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(mul_146, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(view_242, (15, 768), (768, 1))
    assert_size_stride(gt_34, (3, 12, 5, 5), (300, 25, 5, 1))
    assert_size_stride(view_258, (15, 768), (768, 1))
    assert_size_stride(gt_35, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(mul_152, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(view_260, (15, 768), (768, 1))
    assert_size_stride(addmm_70, (15, 3072), (3072, 1))
    assert_size_stride(view_262, (15, 3072), (3072, 1))
    assert_size_stride(gt_36, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(mul_159, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(select, (3, 768), (3840, 1))
    assert_size_stride(tanh, (3, 768), (768, 1))
    assert_size_stride(permute_133, (768, 768), (768, 1))
    assert_size_stride(div_24, (3, 5, 1), (5, 1, 1))
    assert_size_stride(permute_137, (768, 3072), (3072, 1))
    assert_size_stride(permute_141, (3072, 768), (768, 1))
    assert_size_stride(div_25, (3, 5, 1), (5, 1, 1))
    assert_size_stride(permute_145, (768, 768), (768, 1))
    assert_size_stride(permute_150, (36, 5, 5), (25, 1, 5))
    assert_size_stride(permute_151, (36, 64, 5), (320, 1, 64))
    assert_size_stride(alias_14, (3, 12, 5, 5), (300, 25, 5, 1))
    assert_size_stride(permute_152, (36, 64, 5), (320, 1, 64))
    assert_size_stride(permute_153, (36, 5, 64), (320, 1, 5))
    assert_size_stride(permute_157, (768, 768), (768, 1))
    assert_size_stride(permute_162, (768, 768), (768, 1))
    assert_size_stride(permute_166, (768, 768), (768, 1))
    assert_size_stride(div_27, (3, 5, 1), (5, 1, 1))
    assert_size_stride(permute_170, (768, 3072), (3072, 1))
    assert_size_stride(permute_174, (3072, 768), (768, 1))
    assert_size_stride(div_28, (3, 5, 1), (5, 1, 1))
    assert_size_stride(permute_178, (768, 768), (768, 1))
    assert_size_stride(permute_183, (36, 5, 5), (25, 1, 5))
    assert_size_stride(permute_184, (36, 64, 5), (320, 1, 64))
    assert_size_stride(alias_15, (3, 12, 5, 5), (300, 25, 5, 1))
    assert_size_stride(permute_185, (36, 64, 5), (320, 1, 64))
    assert_size_stride(permute_186, (36, 5, 64), (320, 1, 5))
    assert_size_stride(permute_190, (768, 768), (768, 1))
    assert_size_stride(permute_195, (768, 768), (768, 1))
    assert_size_stride(permute_199, (768, 768), (768, 1))
    assert_size_stride(div_30, (3, 5, 1), (5, 1, 1))
    assert_size_stride(permute_203, (768, 3072), (3072, 1))
    assert_size_stride(permute_207, (3072, 768), (768, 1))
    assert_size_stride(div_31, (3, 5, 1), (5, 1, 1))
    assert_size_stride(permute_211, (768, 768), (768, 1))
    assert_size_stride(permute_216, (36, 5, 5), (25, 1, 5))
    assert_size_stride(permute_217, (36, 64, 5), (320, 1, 64))
    assert_size_stride(alias_16, (3, 12, 5, 5), (300, 25, 5, 1))
    assert_size_stride(permute_218, (36, 64, 5), (320, 1, 64))
    assert_size_stride(permute_219, (36, 5, 64), (320, 1, 5))
    assert_size_stride(permute_223, (768, 768), (768, 1))
    assert_size_stride(permute_228, (768, 768), (768, 1))
    assert_size_stride(permute_232, (768, 768), (768, 1))
    assert_size_stride(div_33, (3, 5, 1), (5, 1, 1))
    assert_size_stride(permute_236, (768, 3072), (3072, 1))
    assert_size_stride(permute_240, (3072, 768), (768, 1))
    assert_size_stride(div_34, (3, 5, 1), (5, 1, 1))
    assert_size_stride(permute_244, (768, 768), (768, 1))
    assert_size_stride(permute_249, (36, 5, 5), (25, 1, 5))
    assert_size_stride(permute_250, (36, 64, 5), (320, 1, 64))
    assert_size_stride(alias_17, (3, 12, 5, 5), (300, 25, 5, 1))
    assert_size_stride(permute_251, (36, 64, 5), (320, 1, 64))
    assert_size_stride(permute_252, (36, 5, 64), (320, 1, 5))
    assert_size_stride(permute_256, (768, 768), (768, 1))
    assert_size_stride(permute_261, (768, 768), (768, 1))
    assert_size_stride(permute_265, (768, 768), (768, 1))
    assert_size_stride(div_36, (3, 5, 1), (5, 1, 1))
    assert_size_stride(permute_269, (768, 3072), (3072, 1))
    assert_size_stride(permute_273, (3072, 768), (768, 1))
    assert_size_stride(div_37, (3, 5, 1), (5, 1, 1))
    assert_size_stride(permute_277, (768, 768), (768, 1))
    assert_size_stride(permute_282, (36, 5, 5), (25, 1, 5))
    assert_size_stride(permute_283, (36, 64, 5), (320, 1, 64))
    assert_size_stride(alias_18, (3, 12, 5, 5), (300, 25, 5, 1))
    assert_size_stride(permute_284, (36, 64, 5), (320, 1, 64))
    assert_size_stride(permute_285, (36, 5, 64), (320, 1, 5))
    assert_size_stride(permute_289, (768, 768), (768, 1))
    assert_size_stride(permute_294, (768, 768), (768, 1))
    assert_size_stride(permute_298, (768, 768), (768, 1))
    assert_size_stride(div_39, (3, 5, 1), (5, 1, 1))
    assert_size_stride(permute_302, (768, 3072), (3072, 1))
    assert_size_stride(permute_306, (3072, 768), (768, 1))
    assert_size_stride(div_40, (3, 5, 1), (5, 1, 1))
    assert_size_stride(permute_310, (768, 768), (768, 1))
    assert_size_stride(permute_315, (36, 5, 5), (25, 1, 5))
    assert_size_stride(permute_316, (36, 64, 5), (320, 1, 64))
    assert_size_stride(alias_19, (3, 12, 5, 5), (300, 25, 5, 1))
    assert_size_stride(permute_317, (36, 64, 5), (320, 1, 64))
    assert_size_stride(permute_318, (36, 5, 64), (320, 1, 5))
    assert_size_stride(permute_322, (768, 768), (768, 1))
    assert_size_stride(permute_327, (768, 768), (768, 1))
    assert_size_stride(permute_331, (768, 768), (768, 1))
    assert_size_stride(div_42, (3, 5, 1), (5, 1, 1))
    assert_size_stride(permute_335, (768, 3072), (3072, 1))
    assert_size_stride(permute_339, (3072, 768), (768, 1))
    assert_size_stride(div_43, (3, 5, 1), (5, 1, 1))
    assert_size_stride(permute_343, (768, 768), (768, 1))
    assert_size_stride(permute_348, (36, 5, 5), (25, 1, 5))
    assert_size_stride(permute_349, (36, 64, 5), (320, 1, 64))
    assert_size_stride(alias_20, (3, 12, 5, 5), (300, 25, 5, 1))
    assert_size_stride(permute_350, (36, 64, 5), (320, 1, 64))
    assert_size_stride(permute_351, (36, 5, 64), (320, 1, 5))
    assert_size_stride(permute_355, (768, 768), (768, 1))
    assert_size_stride(permute_360, (768, 768), (768, 1))
    assert_size_stride(permute_364, (768, 768), (768, 1))
    assert_size_stride(div_45, (3, 5, 1), (5, 1, 1))
    assert_size_stride(permute_368, (768, 3072), (3072, 1))
    assert_size_stride(permute_372, (3072, 768), (768, 1))
    assert_size_stride(div_46, (3, 5, 1), (5, 1, 1))
    assert_size_stride(permute_376, (768, 768), (768, 1))
    assert_size_stride(permute_381, (36, 5, 5), (25, 1, 5))
    assert_size_stride(permute_382, (36, 64, 5), (320, 1, 64))
    assert_size_stride(alias_21, (3, 12, 5, 5), (300, 25, 5, 1))
    assert_size_stride(permute_383, (36, 64, 5), (320, 1, 64))
    assert_size_stride(permute_384, (36, 5, 64), (320, 1, 5))
    assert_size_stride(permute_388, (768, 768), (768, 1))
    assert_size_stride(permute_393, (768, 768), (768, 1))
    assert_size_stride(permute_397, (768, 768), (768, 1))
    assert_size_stride(div_48, (3, 5, 1), (5, 1, 1))
    assert_size_stride(permute_401, (768, 3072), (3072, 1))
    assert_size_stride(permute_405, (3072, 768), (768, 1))
    assert_size_stride(div_49, (3, 5, 1), (5, 1, 1))
    assert_size_stride(permute_409, (768, 768), (768, 1))
    assert_size_stride(permute_414, (36, 5, 5), (25, 1, 5))
    assert_size_stride(permute_415, (36, 64, 5), (320, 1, 64))
    assert_size_stride(alias_22, (3, 12, 5, 5), (300, 25, 5, 1))
    assert_size_stride(permute_416, (36, 64, 5), (320, 1, 64))
    assert_size_stride(permute_417, (36, 5, 64), (320, 1, 5))
    assert_size_stride(permute_421, (768, 768), (768, 1))
    assert_size_stride(permute_426, (768, 768), (768, 1))
    assert_size_stride(permute_430, (768, 768), (768, 1))
    assert_size_stride(div_51, (3, 5, 1), (5, 1, 1))
    assert_size_stride(permute_434, (768, 3072), (3072, 1))
    assert_size_stride(permute_438, (3072, 768), (768, 1))
    assert_size_stride(div_52, (3, 5, 1), (5, 1, 1))
    assert_size_stride(permute_442, (768, 768), (768, 1))
    assert_size_stride(permute_447, (36, 5, 5), (25, 1, 5))
    assert_size_stride(permute_448, (36, 64, 5), (320, 1, 64))
    assert_size_stride(alias_23, (3, 12, 5, 5), (300, 25, 5, 1))
    assert_size_stride(permute_449, (36, 64, 5), (320, 1, 64))
    assert_size_stride(permute_450, (36, 5, 64), (320, 1, 5))
    assert_size_stride(permute_454, (768, 768), (768, 1))
    assert_size_stride(permute_459, (768, 768), (768, 1))
    assert_size_stride(permute_463, (768, 768), (768, 1))
    assert_size_stride(div_54, (3, 5, 1), (5, 1, 1))
    assert_size_stride(permute_467, (768, 3072), (3072, 1))
    assert_size_stride(permute_471, (3072, 768), (768, 1))
    assert_size_stride(div_55, (3, 5, 1), (5, 1, 1))
    assert_size_stride(permute_475, (768, 768), (768, 1))
    assert_size_stride(permute_480, (36, 5, 5), (25, 1, 5))
    assert_size_stride(permute_481, (36, 64, 5), (320, 1, 64))
    assert_size_stride(alias_24, (3, 12, 5, 5), (300, 25, 5, 1))
    assert_size_stride(permute_482, (36, 64, 5), (320, 1, 64))
    assert_size_stride(permute_483, (36, 5, 64), (320, 1, 5))
    assert_size_stride(permute_487, (768, 768), (768, 1))
    assert_size_stride(permute_492, (768, 768), (768, 1))
    assert_size_stride(permute_496, (768, 768), (768, 1))
    assert_size_stride(div_57, (3, 5, 1), (5, 1, 1))
    assert_size_stride(permute_500, (768, 3072), (3072, 1))
    assert_size_stride(permute_504, (3072, 768), (768, 1))
    assert_size_stride(div_58, (3, 5, 1), (5, 1, 1))
    assert_size_stride(permute_508, (768, 768), (768, 1))
    assert_size_stride(permute_513, (36, 5, 5), (25, 1, 5))
    assert_size_stride(permute_514, (36, 64, 5), (320, 1, 64))
    assert_size_stride(alias_25, (3, 12, 5, 5), (300, 25, 5, 1))
    assert_size_stride(permute_515, (36, 64, 5), (320, 1, 64))
    assert_size_stride(permute_516, (36, 5, 64), (320, 1, 5))
    assert_size_stride(permute_520, (768, 768), (768, 1))
    assert_size_stride(permute_525, (768, 768), (768, 1))
    assert_size_stride(permute_529, (768, 768), (768, 1))
    assert_size_stride(div_60, (3, 5, 1), (5, 1, 1))
    assert_size_stride(tangents_1, (3, 5, 768), (3840, 768, 1))
    assert_size_stride(tangents_2, (3, 768), (768, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((3, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.tanh_backward]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_tanh_backward_0.run(tangents_2, tanh, buf0, 2304, grid=grid(2304), stream=stream0)
        del tangents_2
        del tanh
        buf1 = empty((3, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf0, permute_133, out=buf1)
        del permute_133
        buf2 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf0, (768, 3), (1, 768), 0), select, out=buf2)
        del select
        buf3 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum, aten.view]
        triton_poi_fused_sum_view_1.run(buf0, buf3, 768, grid=grid(768), stream=stream0)
        del buf0
        buf6 = empty((3, 5, 768), device='cuda', dtype=torch.float32)
        buf9 = empty((3, 5, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward, aten.select_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_select_backward_2.run(tangents_1, buf1, primals_196, mul_159, div_24, gt_36, buf6, buf9, 15, 768, grid=grid(15), stream=stream0)
        del div_24
        del gt_36
        del primals_196
        buf7 = empty((768, ), device='cuda', dtype=torch.float32)
        buf8 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.select_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_3.run(tangents_1, buf1, mul_159, buf7, buf8, 768, 15, grid=grid(768), stream=stream0)
        del buf1
        del mul_159
        del tangents_1
        buf10 = empty((15, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf9, (15, 768), (768, 1), 0), permute_137, out=buf10)
        del permute_137
        buf11 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf9, (768, 15), (1, 768), 0), view_262, out=buf11)
        del view_262
        buf12 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf9, buf12, 768, 15, grid=grid(768), stream=stream0)
        buf13 = reinterpret_tensor(buf10, (3, 5, 3072), (15360, 3072, 1), 0); del buf10  # reuse
        # Source Nodes: [intermediate_output_11], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_5.run(buf13, addmm_70, 46080, grid=grid(46080), stream=stream0)
        del addmm_70
        buf14 = reinterpret_tensor(buf9, (15, 768), (768, 1), 0); del buf9  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (15, 3072), (3072, 1), 0), permute_141, out=buf14)
        del permute_141
        buf15 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (3072, 15), (1, 3072), 0), view_260, out=buf15)
        del view_260
        buf16 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf13, buf16, 3072, 15, grid=grid(3072), stream=stream0)
        buf19 = empty((3, 5, 768), device='cuda', dtype=torch.float32)
        buf22 = empty((3, 5, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_7.run(buf6, buf14, primals_190, mul_152, div_25, gt_35, buf19, buf22, 15, 768, grid=grid(15), stream=stream0)
        del div_25
        del gt_35
        del primals_190
        buf20 = empty((768, ), device='cuda', dtype=torch.float32)
        buf21 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_8.run(buf6, buf14, mul_152, buf20, buf21, 768, 15, grid=grid(768), stream=stream0)
        del mul_152
        buf23 = reinterpret_tensor(buf6, (15, 768), (768, 1), 0); del buf6  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf22, (15, 768), (768, 1), 0), permute_145, out=buf23)
        del permute_145
        buf24 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf22, (768, 15), (1, 768), 0), view_258, out=buf24)
        del view_258
        buf25 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf22, buf25, 768, 15, grid=grid(768), stream=stream0)
        buf26 = reinterpret_tensor(buf22, (3, 12, 5, 64), (3840, 320, 64, 1), 0); del buf22  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf23, buf26, 11520, grid=grid(11520), stream=stream0)
        buf27 = reinterpret_tensor(buf23, (36, 5, 64), (320, 64, 1), 0); del buf23  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_150, reinterpret_tensor(buf26, (36, 5, 64), (320, 64, 1), 0), out=buf27)
        del permute_150
        buf28 = empty((36, 5, 5), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf26, (36, 5, 64), (320, 64, 1), 0), permute_151, out=buf28)
        del permute_151
        buf29 = empty_strided((3, 12, 5, 1), (60, 5, 1, 180), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.native_dropout_backward]
        triton_poi_fused__softmax_backward_data_native_dropout_backward_10.run(buf28, gt_34, alias_14, buf29, 180, grid=grid(180), stream=stream0)
        buf30 = reinterpret_tensor(buf28, (3, 12, 5, 5), (300, 25, 5, 1), 0); del buf28  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div, aten.native_dropout_backward]
        triton_poi_fused__softmax_backward_data_div_native_dropout_backward_11.run(buf30, gt_34, alias_14, buf29, 900, grid=grid(900), stream=stream0)
        del alias_14
        del gt_34
        buf31 = reinterpret_tensor(buf26, (36, 64, 5), (320, 5, 1), 0); del buf26  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_152, reinterpret_tensor(buf30, (36, 5, 5), (25, 5, 1), 0), out=buf31)
        del permute_152
        buf32 = reinterpret_tensor(buf14, (36, 5, 64), (320, 64, 1), 0); del buf14  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf30, (36, 5, 5), (25, 5, 1), 0), permute_153, out=buf32)
        del permute_153
        buf33 = empty((15, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf27, buf33, 11520, grid=grid(11520), stream=stream0)
        buf34 = reinterpret_tensor(buf27, (15, 768), (768, 1), 0); del buf27  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf33, permute_157, out=buf34)
        del permute_157
        buf35 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf33, (768, 15), (1, 768), 0), view_242, out=buf35)
        buf36 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf33, buf36, 768, 15, grid=grid(768), stream=stream0)
        buf37 = buf33; del buf33  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_13.run(buf31, buf37, 15, 768, grid=grid(15, 768), stream=stream0)
        buf38 = reinterpret_tensor(buf31, (15, 768), (768, 1), 0); del buf31  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf37, permute_162, out=buf38)
        del permute_162
        buf39 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf37, (768, 15), (1, 768), 0), view_242, out=buf39)
        buf40 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf37, buf40, 768, 15, grid=grid(768), stream=stream0)
        buf41 = buf37; del buf37  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf32, buf41, 11520, grid=grid(11520), stream=stream0)
        buf42 = reinterpret_tensor(buf32, (15, 768), (768, 1), 0); del buf32  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf41, permute_166, out=buf42)
        del permute_166
        buf43 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf41, (768, 15), (1, 768), 0), view_242, out=buf43)
        del view_242
        buf44 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf41, buf44, 768, 15, grid=grid(768), stream=stream0)
        buf48 = reinterpret_tensor(buf41, (3, 5, 768), (3840, 768, 1), 0); del buf41  # reuse
        buf51 = empty((3, 5, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14.run(buf19, buf34, buf38, buf42, primals_180, mul_146, div_27, gt_33, buf48, buf51, 15, 768, grid=grid(15), stream=stream0)
        del div_27
        del gt_33
        del primals_180
        buf49 = empty((768, ), device='cuda', dtype=torch.float32)
        buf50 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf19, buf34, buf38, buf42, mul_146, buf49, buf50, 768, 15, grid=grid(768), stream=stream0)
        del mul_146
        buf52 = reinterpret_tensor(buf13, (15, 3072), (3072, 1), 0); del buf13  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf51, (15, 768), (768, 1), 0), permute_170, out=buf52)
        del permute_170
        buf53 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf51, (768, 15), (1, 768), 0), view_240, out=buf53)
        del view_240
        buf54 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf51, buf54, 768, 15, grid=grid(768), stream=stream0)
        buf55 = reinterpret_tensor(buf52, (3, 5, 3072), (15360, 3072, 1), 0); del buf52  # reuse
        # Source Nodes: [intermediate_output_10], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_5.run(buf55, addmm_64, 46080, grid=grid(46080), stream=stream0)
        del addmm_64
        buf56 = reinterpret_tensor(buf51, (15, 768), (768, 1), 0); del buf51  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf55, (15, 3072), (3072, 1), 0), permute_174, out=buf56)
        del permute_174
        buf57 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf55, (3072, 15), (1, 3072), 0), view_238, out=buf57)
        del view_238
        buf58 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf55, buf58, 3072, 15, grid=grid(3072), stream=stream0)
        buf61 = reinterpret_tensor(buf42, (3, 5, 768), (3840, 768, 1), 0); del buf42  # reuse
        buf64 = reinterpret_tensor(buf38, (3, 5, 768), (3840, 768, 1), 0); del buf38  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_7.run(buf48, buf56, primals_174, mul_139, div_28, gt_32, buf61, buf64, 15, 768, grid=grid(15), stream=stream0)
        del div_28
        del gt_32
        del primals_174
        buf62 = empty((768, ), device='cuda', dtype=torch.float32)
        buf63 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_8.run(buf48, buf56, mul_139, buf62, buf63, 768, 15, grid=grid(768), stream=stream0)
        del mul_139
        buf65 = buf56; del buf56  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf64, (15, 768), (768, 1), 0), permute_178, out=buf65)
        del permute_178
        buf66 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf64, (768, 15), (1, 768), 0), view_236, out=buf66)
        del view_236
        buf67 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf64, buf67, 768, 15, grid=grid(768), stream=stream0)
        buf68 = reinterpret_tensor(buf64, (3, 12, 5, 64), (3840, 320, 64, 1), 0); del buf64  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf65, buf68, 11520, grid=grid(11520), stream=stream0)
        buf69 = reinterpret_tensor(buf65, (36, 5, 64), (320, 64, 1), 0); del buf65  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_183, reinterpret_tensor(buf68, (36, 5, 64), (320, 64, 1), 0), out=buf69)
        del permute_183
        buf70 = reinterpret_tensor(buf30, (36, 5, 5), (25, 5, 1), 0); del buf30  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf68, (36, 5, 64), (320, 64, 1), 0), permute_184, out=buf70)
        del permute_184
        buf71 = buf29; del buf29  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.native_dropout_backward]
        triton_poi_fused__softmax_backward_data_native_dropout_backward_10.run(buf70, gt_31, alias_15, buf71, 180, grid=grid(180), stream=stream0)
        buf72 = reinterpret_tensor(buf70, (3, 12, 5, 5), (300, 25, 5, 1), 0); del buf70  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div, aten.native_dropout_backward]
        triton_poi_fused__softmax_backward_data_div_native_dropout_backward_11.run(buf72, gt_31, alias_15, buf71, 900, grid=grid(900), stream=stream0)
        del alias_15
        del gt_31
        buf73 = reinterpret_tensor(buf68, (36, 64, 5), (320, 5, 1), 0); del buf68  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_185, reinterpret_tensor(buf72, (36, 5, 5), (25, 5, 1), 0), out=buf73)
        del permute_185
        buf74 = reinterpret_tensor(buf48, (36, 5, 64), (320, 64, 1), 0); del buf48  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf72, (36, 5, 5), (25, 5, 1), 0), permute_186, out=buf74)
        del permute_186
        buf75 = buf34; del buf34  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf69, buf75, 11520, grid=grid(11520), stream=stream0)
        buf76 = reinterpret_tensor(buf69, (15, 768), (768, 1), 0); del buf69  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf75, permute_190, out=buf76)
        del permute_190
        buf77 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf75, (768, 15), (1, 768), 0), view_220, out=buf77)
        buf78 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf75, buf78, 768, 15, grid=grid(768), stream=stream0)
        buf79 = buf75; del buf75  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_13.run(buf73, buf79, 15, 768, grid=grid(15, 768), stream=stream0)
        buf80 = reinterpret_tensor(buf73, (15, 768), (768, 1), 0); del buf73  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf79, permute_195, out=buf80)
        del permute_195
        buf81 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf79, (768, 15), (1, 768), 0), view_220, out=buf81)
        buf82 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf79, buf82, 768, 15, grid=grid(768), stream=stream0)
        buf83 = buf79; del buf79  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf74, buf83, 11520, grid=grid(11520), stream=stream0)
        buf84 = reinterpret_tensor(buf74, (15, 768), (768, 1), 0); del buf74  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf83, permute_199, out=buf84)
        del permute_199
        buf85 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf83, (768, 15), (1, 768), 0), view_220, out=buf85)
        del view_220
        buf86 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf83, buf86, 768, 15, grid=grid(768), stream=stream0)
        buf90 = reinterpret_tensor(buf83, (3, 5, 768), (3840, 768, 1), 0); del buf83  # reuse
        buf93 = buf19; del buf19  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14.run(buf61, buf76, buf80, buf84, primals_164, mul_133, div_30, gt_30, buf90, buf93, 15, 768, grid=grid(15), stream=stream0)
        del div_30
        del gt_30
        del primals_164
        buf91 = empty((768, ), device='cuda', dtype=torch.float32)
        buf92 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf61, buf76, buf80, buf84, mul_133, buf91, buf92, 768, 15, grid=grid(768), stream=stream0)
        del mul_133
        buf94 = reinterpret_tensor(buf55, (15, 3072), (3072, 1), 0); del buf55  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf93, (15, 768), (768, 1), 0), permute_203, out=buf94)
        del permute_203
        buf95 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf93, (768, 15), (1, 768), 0), view_218, out=buf95)
        del view_218
        buf96 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf93, buf96, 768, 15, grid=grid(768), stream=stream0)
        buf97 = reinterpret_tensor(buf94, (3, 5, 3072), (15360, 3072, 1), 0); del buf94  # reuse
        # Source Nodes: [intermediate_output_9], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_5.run(buf97, addmm_58, 46080, grid=grid(46080), stream=stream0)
        del addmm_58
        buf98 = reinterpret_tensor(buf93, (15, 768), (768, 1), 0); del buf93  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf97, (15, 3072), (3072, 1), 0), permute_207, out=buf98)
        del permute_207
        buf99 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf97, (3072, 15), (1, 3072), 0), view_216, out=buf99)
        del view_216
        buf100 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf97, buf100, 3072, 15, grid=grid(3072), stream=stream0)
        buf103 = reinterpret_tensor(buf84, (3, 5, 768), (3840, 768, 1), 0); del buf84  # reuse
        buf106 = reinterpret_tensor(buf80, (3, 5, 768), (3840, 768, 1), 0); del buf80  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_7.run(buf90, buf98, primals_158, mul_126, div_31, gt_29, buf103, buf106, 15, 768, grid=grid(15), stream=stream0)
        del div_31
        del gt_29
        del primals_158
        buf104 = empty((768, ), device='cuda', dtype=torch.float32)
        buf105 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_8.run(buf90, buf98, mul_126, buf104, buf105, 768, 15, grid=grid(768), stream=stream0)
        del mul_126
        buf107 = buf98; del buf98  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf106, (15, 768), (768, 1), 0), permute_211, out=buf107)
        del permute_211
        buf108 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf106, (768, 15), (1, 768), 0), view_214, out=buf108)
        del view_214
        buf109 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf106, buf109, 768, 15, grid=grid(768), stream=stream0)
        buf110 = reinterpret_tensor(buf106, (3, 12, 5, 64), (3840, 320, 64, 1), 0); del buf106  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf107, buf110, 11520, grid=grid(11520), stream=stream0)
        buf111 = reinterpret_tensor(buf107, (36, 5, 64), (320, 64, 1), 0); del buf107  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_216, reinterpret_tensor(buf110, (36, 5, 64), (320, 64, 1), 0), out=buf111)
        del permute_216
        buf112 = reinterpret_tensor(buf72, (36, 5, 5), (25, 5, 1), 0); del buf72  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf110, (36, 5, 64), (320, 64, 1), 0), permute_217, out=buf112)
        del permute_217
        buf113 = buf71; del buf71  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.native_dropout_backward]
        triton_poi_fused__softmax_backward_data_native_dropout_backward_10.run(buf112, gt_28, alias_16, buf113, 180, grid=grid(180), stream=stream0)
        buf114 = reinterpret_tensor(buf112, (3, 12, 5, 5), (300, 25, 5, 1), 0); del buf112  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div, aten.native_dropout_backward]
        triton_poi_fused__softmax_backward_data_div_native_dropout_backward_11.run(buf114, gt_28, alias_16, buf113, 900, grid=grid(900), stream=stream0)
        del alias_16
        del gt_28
        buf115 = reinterpret_tensor(buf110, (36, 64, 5), (320, 5, 1), 0); del buf110  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_218, reinterpret_tensor(buf114, (36, 5, 5), (25, 5, 1), 0), out=buf115)
        del permute_218
        buf116 = reinterpret_tensor(buf90, (36, 5, 64), (320, 64, 1), 0); del buf90  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf114, (36, 5, 5), (25, 5, 1), 0), permute_219, out=buf116)
        del permute_219
        buf117 = buf76; del buf76  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf111, buf117, 11520, grid=grid(11520), stream=stream0)
        buf118 = reinterpret_tensor(buf111, (15, 768), (768, 1), 0); del buf111  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf117, permute_223, out=buf118)
        del permute_223
        buf119 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf117, (768, 15), (1, 768), 0), view_198, out=buf119)
        buf120 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf117, buf120, 768, 15, grid=grid(768), stream=stream0)
        buf121 = buf117; del buf117  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_13.run(buf115, buf121, 15, 768, grid=grid(15, 768), stream=stream0)
        buf122 = reinterpret_tensor(buf115, (15, 768), (768, 1), 0); del buf115  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf121, permute_228, out=buf122)
        del permute_228
        buf123 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf121, (768, 15), (1, 768), 0), view_198, out=buf123)
        buf124 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf121, buf124, 768, 15, grid=grid(768), stream=stream0)
        buf125 = buf121; del buf121  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf116, buf125, 11520, grid=grid(11520), stream=stream0)
        buf126 = reinterpret_tensor(buf116, (15, 768), (768, 1), 0); del buf116  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf125, permute_232, out=buf126)
        del permute_232
        buf127 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf125, (768, 15), (1, 768), 0), view_198, out=buf127)
        del view_198
        buf128 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf125, buf128, 768, 15, grid=grid(768), stream=stream0)
        buf132 = reinterpret_tensor(buf125, (3, 5, 768), (3840, 768, 1), 0); del buf125  # reuse
        buf135 = buf61; del buf61  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14.run(buf103, buf118, buf122, buf126, primals_148, mul_120, div_33, gt_27, buf132, buf135, 15, 768, grid=grid(15), stream=stream0)
        del div_33
        del gt_27
        del primals_148
        buf133 = empty((768, ), device='cuda', dtype=torch.float32)
        buf134 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf103, buf118, buf122, buf126, mul_120, buf133, buf134, 768, 15, grid=grid(768), stream=stream0)
        del mul_120
        buf136 = reinterpret_tensor(buf97, (15, 3072), (3072, 1), 0); del buf97  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf135, (15, 768), (768, 1), 0), permute_236, out=buf136)
        del permute_236
        buf137 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf135, (768, 15), (1, 768), 0), view_196, out=buf137)
        del view_196
        buf138 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf135, buf138, 768, 15, grid=grid(768), stream=stream0)
        buf139 = reinterpret_tensor(buf136, (3, 5, 3072), (15360, 3072, 1), 0); del buf136  # reuse
        # Source Nodes: [intermediate_output_8], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_5.run(buf139, addmm_52, 46080, grid=grid(46080), stream=stream0)
        del addmm_52
        buf140 = reinterpret_tensor(buf135, (15, 768), (768, 1), 0); del buf135  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf139, (15, 3072), (3072, 1), 0), permute_240, out=buf140)
        del permute_240
        buf141 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf139, (3072, 15), (1, 3072), 0), view_194, out=buf141)
        del view_194
        buf142 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf139, buf142, 3072, 15, grid=grid(3072), stream=stream0)
        buf145 = reinterpret_tensor(buf126, (3, 5, 768), (3840, 768, 1), 0); del buf126  # reuse
        buf148 = reinterpret_tensor(buf122, (3, 5, 768), (3840, 768, 1), 0); del buf122  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_7.run(buf132, buf140, primals_142, mul_113, div_34, gt_26, buf145, buf148, 15, 768, grid=grid(15), stream=stream0)
        del div_34
        del gt_26
        del primals_142
        buf146 = empty((768, ), device='cuda', dtype=torch.float32)
        buf147 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_8.run(buf132, buf140, mul_113, buf146, buf147, 768, 15, grid=grid(768), stream=stream0)
        del mul_113
        buf149 = buf140; del buf140  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf148, (15, 768), (768, 1), 0), permute_244, out=buf149)
        del permute_244
        buf150 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf148, (768, 15), (1, 768), 0), view_192, out=buf150)
        del view_192
        buf151 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf148, buf151, 768, 15, grid=grid(768), stream=stream0)
        buf152 = reinterpret_tensor(buf148, (3, 12, 5, 64), (3840, 320, 64, 1), 0); del buf148  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf149, buf152, 11520, grid=grid(11520), stream=stream0)
        buf153 = reinterpret_tensor(buf149, (36, 5, 64), (320, 64, 1), 0); del buf149  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_249, reinterpret_tensor(buf152, (36, 5, 64), (320, 64, 1), 0), out=buf153)
        del permute_249
        buf154 = reinterpret_tensor(buf114, (36, 5, 5), (25, 5, 1), 0); del buf114  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf152, (36, 5, 64), (320, 64, 1), 0), permute_250, out=buf154)
        del permute_250
        buf155 = buf113; del buf113  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.native_dropout_backward]
        triton_poi_fused__softmax_backward_data_native_dropout_backward_10.run(buf154, gt_25, alias_17, buf155, 180, grid=grid(180), stream=stream0)
        buf156 = reinterpret_tensor(buf154, (3, 12, 5, 5), (300, 25, 5, 1), 0); del buf154  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div, aten.native_dropout_backward]
        triton_poi_fused__softmax_backward_data_div_native_dropout_backward_11.run(buf156, gt_25, alias_17, buf155, 900, grid=grid(900), stream=stream0)
        del alias_17
        del gt_25
        buf157 = reinterpret_tensor(buf152, (36, 64, 5), (320, 5, 1), 0); del buf152  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_251, reinterpret_tensor(buf156, (36, 5, 5), (25, 5, 1), 0), out=buf157)
        del permute_251
        buf158 = reinterpret_tensor(buf132, (36, 5, 64), (320, 64, 1), 0); del buf132  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf156, (36, 5, 5), (25, 5, 1), 0), permute_252, out=buf158)
        del permute_252
        buf159 = buf118; del buf118  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf153, buf159, 11520, grid=grid(11520), stream=stream0)
        buf160 = reinterpret_tensor(buf153, (15, 768), (768, 1), 0); del buf153  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf159, permute_256, out=buf160)
        del permute_256
        buf161 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf159, (768, 15), (1, 768), 0), view_176, out=buf161)
        buf162 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf159, buf162, 768, 15, grid=grid(768), stream=stream0)
        buf163 = buf159; del buf159  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_13.run(buf157, buf163, 15, 768, grid=grid(15, 768), stream=stream0)
        buf164 = reinterpret_tensor(buf157, (15, 768), (768, 1), 0); del buf157  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf163, permute_261, out=buf164)
        del permute_261
        buf165 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf163, (768, 15), (1, 768), 0), view_176, out=buf165)
        buf166 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf163, buf166, 768, 15, grid=grid(768), stream=stream0)
        buf167 = buf163; del buf163  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf158, buf167, 11520, grid=grid(11520), stream=stream0)
        buf168 = reinterpret_tensor(buf158, (15, 768), (768, 1), 0); del buf158  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf167, permute_265, out=buf168)
        del permute_265
        buf169 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf167, (768, 15), (1, 768), 0), view_176, out=buf169)
        del view_176
        buf170 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf167, buf170, 768, 15, grid=grid(768), stream=stream0)
        buf174 = reinterpret_tensor(buf167, (3, 5, 768), (3840, 768, 1), 0); del buf167  # reuse
        buf177 = buf103; del buf103  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14.run(buf145, buf160, buf164, buf168, primals_132, mul_107, div_36, gt_24, buf174, buf177, 15, 768, grid=grid(15), stream=stream0)
        del div_36
        del gt_24
        del primals_132
        buf175 = empty((768, ), device='cuda', dtype=torch.float32)
        buf176 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf145, buf160, buf164, buf168, mul_107, buf175, buf176, 768, 15, grid=grid(768), stream=stream0)
        del mul_107
        buf178 = reinterpret_tensor(buf139, (15, 3072), (3072, 1), 0); del buf139  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf177, (15, 768), (768, 1), 0), permute_269, out=buf178)
        del permute_269
        buf179 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf177, (768, 15), (1, 768), 0), view_174, out=buf179)
        del view_174
        buf180 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf177, buf180, 768, 15, grid=grid(768), stream=stream0)
        buf181 = reinterpret_tensor(buf178, (3, 5, 3072), (15360, 3072, 1), 0); del buf178  # reuse
        # Source Nodes: [intermediate_output_7], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_5.run(buf181, addmm_46, 46080, grid=grid(46080), stream=stream0)
        del addmm_46
        buf182 = reinterpret_tensor(buf177, (15, 768), (768, 1), 0); del buf177  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf181, (15, 3072), (3072, 1), 0), permute_273, out=buf182)
        del permute_273
        buf183 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf181, (3072, 15), (1, 3072), 0), view_172, out=buf183)
        del view_172
        buf184 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf181, buf184, 3072, 15, grid=grid(3072), stream=stream0)
        buf187 = reinterpret_tensor(buf168, (3, 5, 768), (3840, 768, 1), 0); del buf168  # reuse
        buf190 = reinterpret_tensor(buf164, (3, 5, 768), (3840, 768, 1), 0); del buf164  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_7.run(buf174, buf182, primals_126, mul_100, div_37, gt_23, buf187, buf190, 15, 768, grid=grid(15), stream=stream0)
        del div_37
        del gt_23
        del primals_126
        buf188 = empty((768, ), device='cuda', dtype=torch.float32)
        buf189 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_8.run(buf174, buf182, mul_100, buf188, buf189, 768, 15, grid=grid(768), stream=stream0)
        del mul_100
        buf191 = buf182; del buf182  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf190, (15, 768), (768, 1), 0), permute_277, out=buf191)
        del permute_277
        buf192 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf190, (768, 15), (1, 768), 0), view_170, out=buf192)
        del view_170
        buf193 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf190, buf193, 768, 15, grid=grid(768), stream=stream0)
        buf194 = reinterpret_tensor(buf190, (3, 12, 5, 64), (3840, 320, 64, 1), 0); del buf190  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf191, buf194, 11520, grid=grid(11520), stream=stream0)
        buf195 = reinterpret_tensor(buf191, (36, 5, 64), (320, 64, 1), 0); del buf191  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_282, reinterpret_tensor(buf194, (36, 5, 64), (320, 64, 1), 0), out=buf195)
        del permute_282
        buf196 = reinterpret_tensor(buf156, (36, 5, 5), (25, 5, 1), 0); del buf156  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf194, (36, 5, 64), (320, 64, 1), 0), permute_283, out=buf196)
        del permute_283
        buf197 = buf155; del buf155  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.native_dropout_backward]
        triton_poi_fused__softmax_backward_data_native_dropout_backward_10.run(buf196, gt_22, alias_18, buf197, 180, grid=grid(180), stream=stream0)
        buf198 = reinterpret_tensor(buf196, (3, 12, 5, 5), (300, 25, 5, 1), 0); del buf196  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div, aten.native_dropout_backward]
        triton_poi_fused__softmax_backward_data_div_native_dropout_backward_11.run(buf198, gt_22, alias_18, buf197, 900, grid=grid(900), stream=stream0)
        del alias_18
        del gt_22
        buf199 = reinterpret_tensor(buf194, (36, 64, 5), (320, 5, 1), 0); del buf194  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_284, reinterpret_tensor(buf198, (36, 5, 5), (25, 5, 1), 0), out=buf199)
        del permute_284
        buf200 = reinterpret_tensor(buf174, (36, 5, 64), (320, 64, 1), 0); del buf174  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf198, (36, 5, 5), (25, 5, 1), 0), permute_285, out=buf200)
        del permute_285
        buf201 = buf160; del buf160  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf195, buf201, 11520, grid=grid(11520), stream=stream0)
        buf202 = reinterpret_tensor(buf195, (15, 768), (768, 1), 0); del buf195  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf201, permute_289, out=buf202)
        del permute_289
        buf203 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf201, (768, 15), (1, 768), 0), view_154, out=buf203)
        buf204 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf201, buf204, 768, 15, grid=grid(768), stream=stream0)
        buf205 = buf201; del buf201  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_13.run(buf199, buf205, 15, 768, grid=grid(15, 768), stream=stream0)
        buf206 = reinterpret_tensor(buf199, (15, 768), (768, 1), 0); del buf199  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf205, permute_294, out=buf206)
        del permute_294
        buf207 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf205, (768, 15), (1, 768), 0), view_154, out=buf207)
        buf208 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf205, buf208, 768, 15, grid=grid(768), stream=stream0)
        buf209 = buf205; del buf205  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf200, buf209, 11520, grid=grid(11520), stream=stream0)
        buf210 = reinterpret_tensor(buf200, (15, 768), (768, 1), 0); del buf200  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf209, permute_298, out=buf210)
        del permute_298
        buf211 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf209, (768, 15), (1, 768), 0), view_154, out=buf211)
        del view_154
        buf212 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf209, buf212, 768, 15, grid=grid(768), stream=stream0)
        buf216 = reinterpret_tensor(buf209, (3, 5, 768), (3840, 768, 1), 0); del buf209  # reuse
        buf219 = buf145; del buf145  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14.run(buf187, buf202, buf206, buf210, primals_116, mul_94, div_39, gt_21, buf216, buf219, 15, 768, grid=grid(15), stream=stream0)
        del div_39
        del gt_21
        del primals_116
        buf217 = empty((768, ), device='cuda', dtype=torch.float32)
        buf218 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf187, buf202, buf206, buf210, mul_94, buf217, buf218, 768, 15, grid=grid(768), stream=stream0)
        del mul_94
        buf220 = reinterpret_tensor(buf181, (15, 3072), (3072, 1), 0); del buf181  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf219, (15, 768), (768, 1), 0), permute_302, out=buf220)
        del permute_302
        buf221 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf219, (768, 15), (1, 768), 0), view_152, out=buf221)
        del view_152
        buf222 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf219, buf222, 768, 15, grid=grid(768), stream=stream0)
        buf223 = reinterpret_tensor(buf220, (3, 5, 3072), (15360, 3072, 1), 0); del buf220  # reuse
        # Source Nodes: [intermediate_output_6], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_5.run(buf223, addmm_40, 46080, grid=grid(46080), stream=stream0)
        del addmm_40
        buf224 = reinterpret_tensor(buf219, (15, 768), (768, 1), 0); del buf219  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf223, (15, 3072), (3072, 1), 0), permute_306, out=buf224)
        del permute_306
        buf225 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf223, (3072, 15), (1, 3072), 0), view_150, out=buf225)
        del view_150
        buf226 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf223, buf226, 3072, 15, grid=grid(3072), stream=stream0)
        buf229 = reinterpret_tensor(buf210, (3, 5, 768), (3840, 768, 1), 0); del buf210  # reuse
        buf232 = reinterpret_tensor(buf206, (3, 5, 768), (3840, 768, 1), 0); del buf206  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_7.run(buf216, buf224, primals_110, mul_87, div_40, gt_20, buf229, buf232, 15, 768, grid=grid(15), stream=stream0)
        del div_40
        del gt_20
        del primals_110
        buf230 = empty((768, ), device='cuda', dtype=torch.float32)
        buf231 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_8.run(buf216, buf224, mul_87, buf230, buf231, 768, 15, grid=grid(768), stream=stream0)
        del mul_87
        buf233 = buf224; del buf224  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf232, (15, 768), (768, 1), 0), permute_310, out=buf233)
        del permute_310
        buf234 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf232, (768, 15), (1, 768), 0), view_148, out=buf234)
        del view_148
        buf235 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf232, buf235, 768, 15, grid=grid(768), stream=stream0)
        buf236 = reinterpret_tensor(buf232, (3, 12, 5, 64), (3840, 320, 64, 1), 0); del buf232  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf233, buf236, 11520, grid=grid(11520), stream=stream0)
        buf237 = reinterpret_tensor(buf233, (36, 5, 64), (320, 64, 1), 0); del buf233  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_315, reinterpret_tensor(buf236, (36, 5, 64), (320, 64, 1), 0), out=buf237)
        del permute_315
        buf238 = reinterpret_tensor(buf198, (36, 5, 5), (25, 5, 1), 0); del buf198  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf236, (36, 5, 64), (320, 64, 1), 0), permute_316, out=buf238)
        del permute_316
        buf239 = buf197; del buf197  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.native_dropout_backward]
        triton_poi_fused__softmax_backward_data_native_dropout_backward_10.run(buf238, gt_19, alias_19, buf239, 180, grid=grid(180), stream=stream0)
        buf240 = reinterpret_tensor(buf238, (3, 12, 5, 5), (300, 25, 5, 1), 0); del buf238  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div, aten.native_dropout_backward]
        triton_poi_fused__softmax_backward_data_div_native_dropout_backward_11.run(buf240, gt_19, alias_19, buf239, 900, grid=grid(900), stream=stream0)
        del alias_19
        del gt_19
        buf241 = reinterpret_tensor(buf236, (36, 64, 5), (320, 5, 1), 0); del buf236  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_317, reinterpret_tensor(buf240, (36, 5, 5), (25, 5, 1), 0), out=buf241)
        del permute_317
        buf242 = reinterpret_tensor(buf216, (36, 5, 64), (320, 64, 1), 0); del buf216  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf240, (36, 5, 5), (25, 5, 1), 0), permute_318, out=buf242)
        del permute_318
        buf243 = buf202; del buf202  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf237, buf243, 11520, grid=grid(11520), stream=stream0)
        buf244 = reinterpret_tensor(buf237, (15, 768), (768, 1), 0); del buf237  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf243, permute_322, out=buf244)
        del permute_322
        buf245 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf243, (768, 15), (1, 768), 0), view_132, out=buf245)
        buf246 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf243, buf246, 768, 15, grid=grid(768), stream=stream0)
        buf247 = buf243; del buf243  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_13.run(buf241, buf247, 15, 768, grid=grid(15, 768), stream=stream0)
        buf248 = reinterpret_tensor(buf241, (15, 768), (768, 1), 0); del buf241  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf247, permute_327, out=buf248)
        del permute_327
        buf249 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf247, (768, 15), (1, 768), 0), view_132, out=buf249)
        buf250 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf247, buf250, 768, 15, grid=grid(768), stream=stream0)
        buf251 = buf247; del buf247  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf242, buf251, 11520, grid=grid(11520), stream=stream0)
        buf252 = reinterpret_tensor(buf242, (15, 768), (768, 1), 0); del buf242  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf251, permute_331, out=buf252)
        del permute_331
        buf253 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf251, (768, 15), (1, 768), 0), view_132, out=buf253)
        del view_132
        buf254 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf251, buf254, 768, 15, grid=grid(768), stream=stream0)
        buf258 = reinterpret_tensor(buf251, (3, 5, 768), (3840, 768, 1), 0); del buf251  # reuse
        buf261 = buf187; del buf187  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14.run(buf229, buf244, buf248, buf252, primals_100, mul_81, div_42, gt_18, buf258, buf261, 15, 768, grid=grid(15), stream=stream0)
        del div_42
        del gt_18
        del primals_100
        buf259 = empty((768, ), device='cuda', dtype=torch.float32)
        buf260 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf229, buf244, buf248, buf252, mul_81, buf259, buf260, 768, 15, grid=grid(768), stream=stream0)
        del mul_81
        buf262 = reinterpret_tensor(buf223, (15, 3072), (3072, 1), 0); del buf223  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf261, (15, 768), (768, 1), 0), permute_335, out=buf262)
        del permute_335
        buf263 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf261, (768, 15), (1, 768), 0), view_130, out=buf263)
        del view_130
        buf264 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf261, buf264, 768, 15, grid=grid(768), stream=stream0)
        buf265 = reinterpret_tensor(buf262, (3, 5, 3072), (15360, 3072, 1), 0); del buf262  # reuse
        # Source Nodes: [intermediate_output_5], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_5.run(buf265, addmm_34, 46080, grid=grid(46080), stream=stream0)
        del addmm_34
        buf266 = reinterpret_tensor(buf261, (15, 768), (768, 1), 0); del buf261  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf265, (15, 3072), (3072, 1), 0), permute_339, out=buf266)
        del permute_339
        buf267 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf265, (3072, 15), (1, 3072), 0), view_128, out=buf267)
        del view_128
        buf268 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf265, buf268, 3072, 15, grid=grid(3072), stream=stream0)
        buf271 = reinterpret_tensor(buf252, (3, 5, 768), (3840, 768, 1), 0); del buf252  # reuse
        buf274 = reinterpret_tensor(buf248, (3, 5, 768), (3840, 768, 1), 0); del buf248  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_7.run(buf258, buf266, primals_94, mul_74, div_43, gt_17, buf271, buf274, 15, 768, grid=grid(15), stream=stream0)
        del div_43
        del gt_17
        del primals_94
        buf272 = empty((768, ), device='cuda', dtype=torch.float32)
        buf273 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_8.run(buf258, buf266, mul_74, buf272, buf273, 768, 15, grid=grid(768), stream=stream0)
        del mul_74
        buf275 = buf266; del buf266  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf274, (15, 768), (768, 1), 0), permute_343, out=buf275)
        del permute_343
        buf276 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf274, (768, 15), (1, 768), 0), view_126, out=buf276)
        del view_126
        buf277 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf274, buf277, 768, 15, grid=grid(768), stream=stream0)
        buf278 = reinterpret_tensor(buf274, (3, 12, 5, 64), (3840, 320, 64, 1), 0); del buf274  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf275, buf278, 11520, grid=grid(11520), stream=stream0)
        buf279 = reinterpret_tensor(buf275, (36, 5, 64), (320, 64, 1), 0); del buf275  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_348, reinterpret_tensor(buf278, (36, 5, 64), (320, 64, 1), 0), out=buf279)
        del permute_348
        buf280 = reinterpret_tensor(buf240, (36, 5, 5), (25, 5, 1), 0); del buf240  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf278, (36, 5, 64), (320, 64, 1), 0), permute_349, out=buf280)
        del permute_349
        buf281 = buf239; del buf239  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.native_dropout_backward]
        triton_poi_fused__softmax_backward_data_native_dropout_backward_10.run(buf280, gt_16, alias_20, buf281, 180, grid=grid(180), stream=stream0)
        buf282 = reinterpret_tensor(buf280, (3, 12, 5, 5), (300, 25, 5, 1), 0); del buf280  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div, aten.native_dropout_backward]
        triton_poi_fused__softmax_backward_data_div_native_dropout_backward_11.run(buf282, gt_16, alias_20, buf281, 900, grid=grid(900), stream=stream0)
        del alias_20
        del gt_16
        buf283 = reinterpret_tensor(buf278, (36, 64, 5), (320, 5, 1), 0); del buf278  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_350, reinterpret_tensor(buf282, (36, 5, 5), (25, 5, 1), 0), out=buf283)
        del permute_350
        buf284 = reinterpret_tensor(buf258, (36, 5, 64), (320, 64, 1), 0); del buf258  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf282, (36, 5, 5), (25, 5, 1), 0), permute_351, out=buf284)
        del permute_351
        buf285 = buf244; del buf244  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf279, buf285, 11520, grid=grid(11520), stream=stream0)
        buf286 = reinterpret_tensor(buf279, (15, 768), (768, 1), 0); del buf279  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf285, permute_355, out=buf286)
        del permute_355
        buf287 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf285, (768, 15), (1, 768), 0), view_110, out=buf287)
        buf288 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf285, buf288, 768, 15, grid=grid(768), stream=stream0)
        buf289 = buf285; del buf285  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_13.run(buf283, buf289, 15, 768, grid=grid(15, 768), stream=stream0)
        buf290 = reinterpret_tensor(buf283, (15, 768), (768, 1), 0); del buf283  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf289, permute_360, out=buf290)
        del permute_360
        buf291 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf289, (768, 15), (1, 768), 0), view_110, out=buf291)
        buf292 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf289, buf292, 768, 15, grid=grid(768), stream=stream0)
        buf293 = buf289; del buf289  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf284, buf293, 11520, grid=grid(11520), stream=stream0)
        buf294 = reinterpret_tensor(buf284, (15, 768), (768, 1), 0); del buf284  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf293, permute_364, out=buf294)
        del permute_364
        buf295 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf293, (768, 15), (1, 768), 0), view_110, out=buf295)
        del view_110
        buf296 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf293, buf296, 768, 15, grid=grid(768), stream=stream0)
        buf300 = reinterpret_tensor(buf293, (3, 5, 768), (3840, 768, 1), 0); del buf293  # reuse
        buf303 = buf229; del buf229  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14.run(buf271, buf286, buf290, buf294, primals_84, mul_68, div_45, gt_15, buf300, buf303, 15, 768, grid=grid(15), stream=stream0)
        del div_45
        del gt_15
        del primals_84
        buf301 = empty((768, ), device='cuda', dtype=torch.float32)
        buf302 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf271, buf286, buf290, buf294, mul_68, buf301, buf302, 768, 15, grid=grid(768), stream=stream0)
        del mul_68
        buf304 = reinterpret_tensor(buf265, (15, 3072), (3072, 1), 0); del buf265  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf303, (15, 768), (768, 1), 0), permute_368, out=buf304)
        del permute_368
        buf305 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf303, (768, 15), (1, 768), 0), view_108, out=buf305)
        del view_108
        buf306 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf303, buf306, 768, 15, grid=grid(768), stream=stream0)
        buf307 = reinterpret_tensor(buf304, (3, 5, 3072), (15360, 3072, 1), 0); del buf304  # reuse
        # Source Nodes: [intermediate_output_4], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_5.run(buf307, addmm_28, 46080, grid=grid(46080), stream=stream0)
        del addmm_28
        buf308 = reinterpret_tensor(buf303, (15, 768), (768, 1), 0); del buf303  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf307, (15, 3072), (3072, 1), 0), permute_372, out=buf308)
        del permute_372
        buf309 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf307, (3072, 15), (1, 3072), 0), view_106, out=buf309)
        del view_106
        buf310 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf307, buf310, 3072, 15, grid=grid(3072), stream=stream0)
        buf313 = reinterpret_tensor(buf294, (3, 5, 768), (3840, 768, 1), 0); del buf294  # reuse
        buf316 = reinterpret_tensor(buf290, (3, 5, 768), (3840, 768, 1), 0); del buf290  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_7.run(buf300, buf308, primals_78, mul_61, div_46, gt_14, buf313, buf316, 15, 768, grid=grid(15), stream=stream0)
        del div_46
        del gt_14
        del primals_78
        buf314 = empty((768, ), device='cuda', dtype=torch.float32)
        buf315 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_8.run(buf300, buf308, mul_61, buf314, buf315, 768, 15, grid=grid(768), stream=stream0)
        del mul_61
        buf317 = buf308; del buf308  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf316, (15, 768), (768, 1), 0), permute_376, out=buf317)
        del permute_376
        buf318 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf316, (768, 15), (1, 768), 0), view_104, out=buf318)
        del view_104
        buf319 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf316, buf319, 768, 15, grid=grid(768), stream=stream0)
        buf320 = reinterpret_tensor(buf316, (3, 12, 5, 64), (3840, 320, 64, 1), 0); del buf316  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf317, buf320, 11520, grid=grid(11520), stream=stream0)
        buf321 = reinterpret_tensor(buf317, (36, 5, 64), (320, 64, 1), 0); del buf317  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_381, reinterpret_tensor(buf320, (36, 5, 64), (320, 64, 1), 0), out=buf321)
        del permute_381
        buf322 = reinterpret_tensor(buf282, (36, 5, 5), (25, 5, 1), 0); del buf282  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf320, (36, 5, 64), (320, 64, 1), 0), permute_382, out=buf322)
        del permute_382
        buf323 = buf281; del buf281  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.native_dropout_backward]
        triton_poi_fused__softmax_backward_data_native_dropout_backward_10.run(buf322, gt_13, alias_21, buf323, 180, grid=grid(180), stream=stream0)
        buf324 = reinterpret_tensor(buf322, (3, 12, 5, 5), (300, 25, 5, 1), 0); del buf322  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div, aten.native_dropout_backward]
        triton_poi_fused__softmax_backward_data_div_native_dropout_backward_11.run(buf324, gt_13, alias_21, buf323, 900, grid=grid(900), stream=stream0)
        del alias_21
        del gt_13
        buf325 = reinterpret_tensor(buf320, (36, 64, 5), (320, 5, 1), 0); del buf320  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_383, reinterpret_tensor(buf324, (36, 5, 5), (25, 5, 1), 0), out=buf325)
        del permute_383
        buf326 = reinterpret_tensor(buf300, (36, 5, 64), (320, 64, 1), 0); del buf300  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf324, (36, 5, 5), (25, 5, 1), 0), permute_384, out=buf326)
        del permute_384
        buf327 = buf286; del buf286  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf321, buf327, 11520, grid=grid(11520), stream=stream0)
        buf328 = reinterpret_tensor(buf321, (15, 768), (768, 1), 0); del buf321  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf327, permute_388, out=buf328)
        del permute_388
        buf329 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf327, (768, 15), (1, 768), 0), view_88, out=buf329)
        buf330 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf327, buf330, 768, 15, grid=grid(768), stream=stream0)
        buf331 = buf327; del buf327  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_13.run(buf325, buf331, 15, 768, grid=grid(15, 768), stream=stream0)
        buf332 = reinterpret_tensor(buf325, (15, 768), (768, 1), 0); del buf325  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf331, permute_393, out=buf332)
        del permute_393
        buf333 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf331, (768, 15), (1, 768), 0), view_88, out=buf333)
        buf334 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf331, buf334, 768, 15, grid=grid(768), stream=stream0)
        buf335 = buf331; del buf331  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf326, buf335, 11520, grid=grid(11520), stream=stream0)
        buf336 = reinterpret_tensor(buf326, (15, 768), (768, 1), 0); del buf326  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf335, permute_397, out=buf336)
        del permute_397
        buf337 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf335, (768, 15), (1, 768), 0), view_88, out=buf337)
        del view_88
        buf338 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf335, buf338, 768, 15, grid=grid(768), stream=stream0)
        buf342 = reinterpret_tensor(buf335, (3, 5, 768), (3840, 768, 1), 0); del buf335  # reuse
        buf345 = buf271; del buf271  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14.run(buf313, buf328, buf332, buf336, primals_68, mul_55, div_48, gt_12, buf342, buf345, 15, 768, grid=grid(15), stream=stream0)
        del div_48
        del gt_12
        del primals_68
        buf343 = empty((768, ), device='cuda', dtype=torch.float32)
        buf344 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf313, buf328, buf332, buf336, mul_55, buf343, buf344, 768, 15, grid=grid(768), stream=stream0)
        del mul_55
        buf346 = reinterpret_tensor(buf307, (15, 3072), (3072, 1), 0); del buf307  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf345, (15, 768), (768, 1), 0), permute_401, out=buf346)
        del permute_401
        buf347 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf345, (768, 15), (1, 768), 0), view_86, out=buf347)
        del view_86
        buf348 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf345, buf348, 768, 15, grid=grid(768), stream=stream0)
        buf349 = reinterpret_tensor(buf346, (3, 5, 3072), (15360, 3072, 1), 0); del buf346  # reuse
        # Source Nodes: [intermediate_output_3], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_5.run(buf349, addmm_22, 46080, grid=grid(46080), stream=stream0)
        del addmm_22
        buf350 = reinterpret_tensor(buf345, (15, 768), (768, 1), 0); del buf345  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf349, (15, 3072), (3072, 1), 0), permute_405, out=buf350)
        del permute_405
        buf351 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf349, (3072, 15), (1, 3072), 0), view_84, out=buf351)
        del view_84
        buf352 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf349, buf352, 3072, 15, grid=grid(3072), stream=stream0)
        buf355 = reinterpret_tensor(buf336, (3, 5, 768), (3840, 768, 1), 0); del buf336  # reuse
        buf358 = reinterpret_tensor(buf332, (3, 5, 768), (3840, 768, 1), 0); del buf332  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_7.run(buf342, buf350, primals_62, mul_48, div_49, gt_11, buf355, buf358, 15, 768, grid=grid(15), stream=stream0)
        del div_49
        del gt_11
        del primals_62
        buf356 = empty((768, ), device='cuda', dtype=torch.float32)
        buf357 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_8.run(buf342, buf350, mul_48, buf356, buf357, 768, 15, grid=grid(768), stream=stream0)
        del mul_48
        buf359 = buf350; del buf350  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf358, (15, 768), (768, 1), 0), permute_409, out=buf359)
        del permute_409
        buf360 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf358, (768, 15), (1, 768), 0), view_82, out=buf360)
        del view_82
        buf361 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf358, buf361, 768, 15, grid=grid(768), stream=stream0)
        buf362 = reinterpret_tensor(buf358, (3, 12, 5, 64), (3840, 320, 64, 1), 0); del buf358  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf359, buf362, 11520, grid=grid(11520), stream=stream0)
        buf363 = reinterpret_tensor(buf359, (36, 5, 64), (320, 64, 1), 0); del buf359  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_414, reinterpret_tensor(buf362, (36, 5, 64), (320, 64, 1), 0), out=buf363)
        del permute_414
        buf364 = reinterpret_tensor(buf324, (36, 5, 5), (25, 5, 1), 0); del buf324  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf362, (36, 5, 64), (320, 64, 1), 0), permute_415, out=buf364)
        del permute_415
        buf365 = buf323; del buf323  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.native_dropout_backward]
        triton_poi_fused__softmax_backward_data_native_dropout_backward_10.run(buf364, gt_10, alias_22, buf365, 180, grid=grid(180), stream=stream0)
        buf366 = reinterpret_tensor(buf364, (3, 12, 5, 5), (300, 25, 5, 1), 0); del buf364  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div, aten.native_dropout_backward]
        triton_poi_fused__softmax_backward_data_div_native_dropout_backward_11.run(buf366, gt_10, alias_22, buf365, 900, grid=grid(900), stream=stream0)
        del alias_22
        del gt_10
        buf367 = reinterpret_tensor(buf362, (36, 64, 5), (320, 5, 1), 0); del buf362  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_416, reinterpret_tensor(buf366, (36, 5, 5), (25, 5, 1), 0), out=buf367)
        del permute_416
        buf368 = reinterpret_tensor(buf342, (36, 5, 64), (320, 64, 1), 0); del buf342  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf366, (36, 5, 5), (25, 5, 1), 0), permute_417, out=buf368)
        del permute_417
        buf369 = buf328; del buf328  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf363, buf369, 11520, grid=grid(11520), stream=stream0)
        buf370 = reinterpret_tensor(buf363, (15, 768), (768, 1), 0); del buf363  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf369, permute_421, out=buf370)
        del permute_421
        buf371 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf369, (768, 15), (1, 768), 0), view_66, out=buf371)
        buf372 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf369, buf372, 768, 15, grid=grid(768), stream=stream0)
        buf373 = buf369; del buf369  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_13.run(buf367, buf373, 15, 768, grid=grid(15, 768), stream=stream0)
        buf374 = reinterpret_tensor(buf367, (15, 768), (768, 1), 0); del buf367  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf373, permute_426, out=buf374)
        del permute_426
        buf375 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf373, (768, 15), (1, 768), 0), view_66, out=buf375)
        buf376 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf373, buf376, 768, 15, grid=grid(768), stream=stream0)
        buf377 = buf373; del buf373  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf368, buf377, 11520, grid=grid(11520), stream=stream0)
        buf378 = reinterpret_tensor(buf368, (15, 768), (768, 1), 0); del buf368  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf377, permute_430, out=buf378)
        del permute_430
        buf379 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf377, (768, 15), (1, 768), 0), view_66, out=buf379)
        del view_66
        buf380 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf377, buf380, 768, 15, grid=grid(768), stream=stream0)
        buf384 = reinterpret_tensor(buf377, (3, 5, 768), (3840, 768, 1), 0); del buf377  # reuse
        buf387 = buf313; del buf313  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14.run(buf355, buf370, buf374, buf378, primals_52, mul_42, div_51, gt_9, buf384, buf387, 15, 768, grid=grid(15), stream=stream0)
        del div_51
        del gt_9
        del primals_52
        buf385 = empty((768, ), device='cuda', dtype=torch.float32)
        buf386 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf355, buf370, buf374, buf378, mul_42, buf385, buf386, 768, 15, grid=grid(768), stream=stream0)
        del mul_42
        buf388 = reinterpret_tensor(buf349, (15, 3072), (3072, 1), 0); del buf349  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf387, (15, 768), (768, 1), 0), permute_434, out=buf388)
        del permute_434
        buf389 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf387, (768, 15), (1, 768), 0), view_64, out=buf389)
        del view_64
        buf390 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf387, buf390, 768, 15, grid=grid(768), stream=stream0)
        buf391 = reinterpret_tensor(buf388, (3, 5, 3072), (15360, 3072, 1), 0); del buf388  # reuse
        # Source Nodes: [intermediate_output_2], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_5.run(buf391, addmm_16, 46080, grid=grid(46080), stream=stream0)
        del addmm_16
        buf392 = reinterpret_tensor(buf387, (15, 768), (768, 1), 0); del buf387  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf391, (15, 3072), (3072, 1), 0), permute_438, out=buf392)
        del permute_438
        buf393 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf391, (3072, 15), (1, 3072), 0), view_62, out=buf393)
        del view_62
        buf394 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf391, buf394, 3072, 15, grid=grid(3072), stream=stream0)
        buf397 = reinterpret_tensor(buf378, (3, 5, 768), (3840, 768, 1), 0); del buf378  # reuse
        buf400 = reinterpret_tensor(buf374, (3, 5, 768), (3840, 768, 1), 0); del buf374  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_7.run(buf384, buf392, primals_46, mul_35, div_52, gt_8, buf397, buf400, 15, 768, grid=grid(15), stream=stream0)
        del div_52
        del gt_8
        del primals_46
        buf398 = empty((768, ), device='cuda', dtype=torch.float32)
        buf399 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_8.run(buf384, buf392, mul_35, buf398, buf399, 768, 15, grid=grid(768), stream=stream0)
        del mul_35
        buf401 = buf392; del buf392  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf400, (15, 768), (768, 1), 0), permute_442, out=buf401)
        del permute_442
        buf402 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf400, (768, 15), (1, 768), 0), view_60, out=buf402)
        del view_60
        buf403 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf400, buf403, 768, 15, grid=grid(768), stream=stream0)
        buf404 = reinterpret_tensor(buf400, (3, 12, 5, 64), (3840, 320, 64, 1), 0); del buf400  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf401, buf404, 11520, grid=grid(11520), stream=stream0)
        buf405 = reinterpret_tensor(buf401, (36, 5, 64), (320, 64, 1), 0); del buf401  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_447, reinterpret_tensor(buf404, (36, 5, 64), (320, 64, 1), 0), out=buf405)
        del permute_447
        buf406 = reinterpret_tensor(buf366, (36, 5, 5), (25, 5, 1), 0); del buf366  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf404, (36, 5, 64), (320, 64, 1), 0), permute_448, out=buf406)
        del permute_448
        buf407 = buf365; del buf365  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.native_dropout_backward]
        triton_poi_fused__softmax_backward_data_native_dropout_backward_10.run(buf406, gt_7, alias_23, buf407, 180, grid=grid(180), stream=stream0)
        buf408 = reinterpret_tensor(buf406, (3, 12, 5, 5), (300, 25, 5, 1), 0); del buf406  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div, aten.native_dropout_backward]
        triton_poi_fused__softmax_backward_data_div_native_dropout_backward_11.run(buf408, gt_7, alias_23, buf407, 900, grid=grid(900), stream=stream0)
        del alias_23
        del gt_7
        buf409 = reinterpret_tensor(buf404, (36, 64, 5), (320, 5, 1), 0); del buf404  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_449, reinterpret_tensor(buf408, (36, 5, 5), (25, 5, 1), 0), out=buf409)
        del permute_449
        buf410 = reinterpret_tensor(buf384, (36, 5, 64), (320, 64, 1), 0); del buf384  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf408, (36, 5, 5), (25, 5, 1), 0), permute_450, out=buf410)
        del permute_450
        buf411 = buf370; del buf370  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf405, buf411, 11520, grid=grid(11520), stream=stream0)
        buf412 = reinterpret_tensor(buf405, (15, 768), (768, 1), 0); del buf405  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf411, permute_454, out=buf412)
        del permute_454
        buf413 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf411, (768, 15), (1, 768), 0), view_44, out=buf413)
        buf414 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf411, buf414, 768, 15, grid=grid(768), stream=stream0)
        buf415 = buf411; del buf411  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_13.run(buf409, buf415, 15, 768, grid=grid(15, 768), stream=stream0)
        buf416 = reinterpret_tensor(buf409, (15, 768), (768, 1), 0); del buf409  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf415, permute_459, out=buf416)
        del permute_459
        buf417 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf415, (768, 15), (1, 768), 0), view_44, out=buf417)
        buf418 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf415, buf418, 768, 15, grid=grid(768), stream=stream0)
        buf419 = buf415; del buf415  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf410, buf419, 11520, grid=grid(11520), stream=stream0)
        buf420 = reinterpret_tensor(buf410, (15, 768), (768, 1), 0); del buf410  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf419, permute_463, out=buf420)
        del permute_463
        buf421 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf419, (768, 15), (1, 768), 0), view_44, out=buf421)
        del view_44
        buf422 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf419, buf422, 768, 15, grid=grid(768), stream=stream0)
        buf426 = reinterpret_tensor(buf419, (3, 5, 768), (3840, 768, 1), 0); del buf419  # reuse
        buf429 = buf355; del buf355  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14.run(buf397, buf412, buf416, buf420, primals_36, mul_29, div_54, gt_6, buf426, buf429, 15, 768, grid=grid(15), stream=stream0)
        del div_54
        del gt_6
        del primals_36
        buf427 = empty((768, ), device='cuda', dtype=torch.float32)
        buf428 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf397, buf412, buf416, buf420, mul_29, buf427, buf428, 768, 15, grid=grid(768), stream=stream0)
        del mul_29
        buf430 = reinterpret_tensor(buf391, (15, 3072), (3072, 1), 0); del buf391  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf429, (15, 768), (768, 1), 0), permute_467, out=buf430)
        del permute_467
        buf431 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf429, (768, 15), (1, 768), 0), view_42, out=buf431)
        del view_42
        buf432 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf429, buf432, 768, 15, grid=grid(768), stream=stream0)
        buf433 = reinterpret_tensor(buf430, (3, 5, 3072), (15360, 3072, 1), 0); del buf430  # reuse
        # Source Nodes: [intermediate_output_1], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_5.run(buf433, addmm_10, 46080, grid=grid(46080), stream=stream0)
        del addmm_10
        buf434 = reinterpret_tensor(buf429, (15, 768), (768, 1), 0); del buf429  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf433, (15, 3072), (3072, 1), 0), permute_471, out=buf434)
        del permute_471
        buf435 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf433, (3072, 15), (1, 3072), 0), view_40, out=buf435)
        del view_40
        buf436 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf433, buf436, 3072, 15, grid=grid(3072), stream=stream0)
        buf439 = reinterpret_tensor(buf420, (3, 5, 768), (3840, 768, 1), 0); del buf420  # reuse
        buf442 = reinterpret_tensor(buf416, (3, 5, 768), (3840, 768, 1), 0); del buf416  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_7.run(buf426, buf434, primals_30, mul_22, div_55, gt_5, buf439, buf442, 15, 768, grid=grid(15), stream=stream0)
        del div_55
        del gt_5
        del primals_30
        buf440 = empty((768, ), device='cuda', dtype=torch.float32)
        buf441 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_8.run(buf426, buf434, mul_22, buf440, buf441, 768, 15, grid=grid(768), stream=stream0)
        del mul_22
        buf443 = buf434; del buf434  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf442, (15, 768), (768, 1), 0), permute_475, out=buf443)
        del permute_475
        buf444 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf442, (768, 15), (1, 768), 0), view_38, out=buf444)
        del view_38
        buf445 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf442, buf445, 768, 15, grid=grid(768), stream=stream0)
        buf446 = reinterpret_tensor(buf442, (3, 12, 5, 64), (3840, 320, 64, 1), 0); del buf442  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf443, buf446, 11520, grid=grid(11520), stream=stream0)
        buf447 = reinterpret_tensor(buf443, (36, 5, 64), (320, 64, 1), 0); del buf443  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_480, reinterpret_tensor(buf446, (36, 5, 64), (320, 64, 1), 0), out=buf447)
        del permute_480
        buf448 = reinterpret_tensor(buf408, (36, 5, 5), (25, 5, 1), 0); del buf408  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf446, (36, 5, 64), (320, 64, 1), 0), permute_481, out=buf448)
        del permute_481
        buf449 = buf407; del buf407  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.native_dropout_backward]
        triton_poi_fused__softmax_backward_data_native_dropout_backward_10.run(buf448, gt_4, alias_24, buf449, 180, grid=grid(180), stream=stream0)
        buf450 = reinterpret_tensor(buf448, (3, 12, 5, 5), (300, 25, 5, 1), 0); del buf448  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div, aten.native_dropout_backward]
        triton_poi_fused__softmax_backward_data_div_native_dropout_backward_11.run(buf450, gt_4, alias_24, buf449, 900, grid=grid(900), stream=stream0)
        del alias_24
        del gt_4
        buf451 = reinterpret_tensor(buf446, (36, 64, 5), (320, 5, 1), 0); del buf446  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_482, reinterpret_tensor(buf450, (36, 5, 5), (25, 5, 1), 0), out=buf451)
        del permute_482
        buf452 = reinterpret_tensor(buf426, (36, 5, 64), (320, 64, 1), 0); del buf426  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf450, (36, 5, 5), (25, 5, 1), 0), permute_483, out=buf452)
        del permute_483
        buf453 = buf412; del buf412  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf447, buf453, 11520, grid=grid(11520), stream=stream0)
        buf454 = reinterpret_tensor(buf447, (15, 768), (768, 1), 0); del buf447  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf453, permute_487, out=buf454)
        del permute_487
        buf455 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf453, (768, 15), (1, 768), 0), view_22, out=buf455)
        buf456 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf453, buf456, 768, 15, grid=grid(768), stream=stream0)
        buf457 = buf453; del buf453  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_13.run(buf451, buf457, 15, 768, grid=grid(15, 768), stream=stream0)
        buf458 = reinterpret_tensor(buf451, (15, 768), (768, 1), 0); del buf451  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf457, permute_492, out=buf458)
        del permute_492
        buf459 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf457, (768, 15), (1, 768), 0), view_22, out=buf459)
        buf460 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf457, buf460, 768, 15, grid=grid(768), stream=stream0)
        buf461 = buf457; del buf457  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf452, buf461, 11520, grid=grid(11520), stream=stream0)
        buf462 = reinterpret_tensor(buf452, (15, 768), (768, 1), 0); del buf452  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf461, permute_496, out=buf462)
        del permute_496
        buf463 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf461, (768, 15), (1, 768), 0), view_22, out=buf463)
        del view_22
        buf464 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf461, buf464, 768, 15, grid=grid(768), stream=stream0)
        buf468 = reinterpret_tensor(buf461, (3, 5, 768), (3840, 768, 1), 0); del buf461  # reuse
        buf471 = buf397; del buf397  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14.run(buf439, buf454, buf458, buf462, primals_20, mul_16, div_57, gt_3, buf468, buf471, 15, 768, grid=grid(15), stream=stream0)
        del div_57
        del gt_3
        del primals_20
        buf469 = empty((768, ), device='cuda', dtype=torch.float32)
        buf470 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf439, buf454, buf458, buf462, mul_16, buf469, buf470, 768, 15, grid=grid(768), stream=stream0)
        del buf439
        del mul_16
        buf472 = reinterpret_tensor(buf433, (15, 3072), (3072, 1), 0); del buf433  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf471, (15, 768), (768, 1), 0), permute_500, out=buf472)
        del permute_500
        buf473 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf471, (768, 15), (1, 768), 0), view_20, out=buf473)
        del view_20
        buf474 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf471, buf474, 768, 15, grid=grid(768), stream=stream0)
        buf475 = reinterpret_tensor(buf472, (3, 5, 3072), (15360, 3072, 1), 0); del buf472  # reuse
        # Source Nodes: [intermediate_output], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_5.run(buf475, addmm_4, 46080, grid=grid(46080), stream=stream0)
        del addmm_4
        buf476 = reinterpret_tensor(buf471, (15, 768), (768, 1), 0); del buf471  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf475, (15, 3072), (3072, 1), 0), permute_504, out=buf476)
        del permute_504
        buf477 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf475, (3072, 15), (1, 3072), 0), view_18, out=buf477)
        del view_18
        buf478 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf475, buf478, 3072, 15, grid=grid(3072), stream=stream0)
        del buf475
        buf481 = reinterpret_tensor(buf462, (3, 5, 768), (3840, 768, 1), 0); del buf462  # reuse
        buf484 = reinterpret_tensor(buf458, (3, 5, 768), (3840, 768, 1), 0); del buf458  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_7.run(buf468, buf476, primals_14, mul_9, div_58, gt_2, buf481, buf484, 15, 768, grid=grid(15), stream=stream0)
        del div_58
        del gt_2
        del primals_14
        buf482 = empty((768, ), device='cuda', dtype=torch.float32)
        buf483 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_8.run(buf468, buf476, mul_9, buf482, buf483, 768, 15, grid=grid(768), stream=stream0)
        del mul_9
        buf485 = buf476; del buf476  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf484, (15, 768), (768, 1), 0), permute_508, out=buf485)
        del permute_508
        buf486 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf484, (768, 15), (1, 768), 0), view_16, out=buf486)
        del view_16
        buf487 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf484, buf487, 768, 15, grid=grid(768), stream=stream0)
        buf488 = reinterpret_tensor(buf484, (3, 12, 5, 64), (3840, 320, 64, 1), 0); del buf484  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf485, buf488, 11520, grid=grid(11520), stream=stream0)
        buf489 = reinterpret_tensor(buf485, (36, 5, 64), (320, 64, 1), 0); del buf485  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_513, reinterpret_tensor(buf488, (36, 5, 64), (320, 64, 1), 0), out=buf489)
        del permute_513
        buf490 = reinterpret_tensor(buf450, (36, 5, 5), (25, 5, 1), 0); del buf450  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf488, (36, 5, 64), (320, 64, 1), 0), permute_514, out=buf490)
        del permute_514
        buf491 = buf449; del buf449  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.native_dropout_backward]
        triton_poi_fused__softmax_backward_data_native_dropout_backward_10.run(buf490, gt_1, alias_25, buf491, 180, grid=grid(180), stream=stream0)
        buf492 = reinterpret_tensor(buf490, (3, 12, 5, 5), (300, 25, 5, 1), 0); del buf490  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div, aten.native_dropout_backward]
        triton_poi_fused__softmax_backward_data_div_native_dropout_backward_11.run(buf492, gt_1, alias_25, buf491, 900, grid=grid(900), stream=stream0)
        del alias_25
        del buf491
        del gt_1
        buf493 = reinterpret_tensor(buf488, (36, 64, 5), (320, 5, 1), 0); del buf488  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_515, reinterpret_tensor(buf492, (36, 5, 5), (25, 5, 1), 0), out=buf493)
        del permute_515
        buf494 = reinterpret_tensor(buf468, (36, 5, 64), (320, 64, 1), 0); del buf468  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf492, (36, 5, 5), (25, 5, 1), 0), permute_516, out=buf494)
        del buf492
        del permute_516
        buf495 = buf454; del buf454  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf489, buf495, 11520, grid=grid(11520), stream=stream0)
        buf496 = reinterpret_tensor(buf489, (15, 768), (768, 1), 0); del buf489  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf495, permute_520, out=buf496)
        del permute_520
        buf497 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf495, (768, 15), (1, 768), 0), view, out=buf497)
        buf498 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf495, buf498, 768, 15, grid=grid(768), stream=stream0)
        buf499 = buf495; del buf495  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_13.run(buf493, buf499, 15, 768, grid=grid(15, 768), stream=stream0)
        buf500 = reinterpret_tensor(buf493, (15, 768), (768, 1), 0); del buf493  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf499, permute_525, out=buf500)
        del permute_525
        buf501 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf499, (768, 15), (1, 768), 0), view, out=buf501)
        buf502 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf499, buf502, 768, 15, grid=grid(768), stream=stream0)
        buf503 = buf499; del buf499  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf494, buf503, 11520, grid=grid(11520), stream=stream0)
        buf504 = reinterpret_tensor(buf494, (15, 768), (768, 1), 0); del buf494  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf503, permute_529, out=buf504)
        del permute_529
        buf505 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf503, (768, 15), (1, 768), 0), view, out=buf505)
        del view
        buf506 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf503, buf506, 768, 15, grid=grid(768), stream=stream0)
        buf515 = empty((2, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_16.run(buf515, 1536, grid=grid(1536), stream=stream0)
        buf517 = empty((30522, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_17.run(buf517, 23440896, grid=grid(23440896), stream=stream0)
        buf507 = buf481; del buf481  # reuse
        buf510 = reinterpret_tensor(buf503, (3, 5, 768), (3840, 768, 1), 0); del buf503  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.embedding_dense_backward, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_18.run(buf507, buf496, buf500, buf504, gt, primals_4, mul_1, div_60, expand, primals_202, buf510, buf515, buf517, 15, 768, grid=grid(15), stream=stream0)
        del buf496
        del buf500
        del buf504
        del div_60
        del expand
        del gt
        del primals_202
        del primals_4
        buf511 = empty((768, ), device='cuda', dtype=torch.float32)
        buf512 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_19.run(buf507, mul_1, buf511, buf512, 768, 15, grid=grid(768), stream=stream0)
        del buf507
        del mul_1
        buf513 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_20.run(buf513, 393216, grid=grid(393216), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward, aten.sum]
        triton_poi_fused_embedding_dense_backward_sum_21.run(slice_6, buf510, buf513, 3840, grid=grid(3840), stream=stream0)
        del buf510
        del slice_6
        return (buf517, buf515, buf513, buf511, buf512, reinterpret_tensor(buf505, (768, 768), (768, 1), 0), reinterpret_tensor(buf506, (768, ), (1, ), 0), reinterpret_tensor(buf501, (768, 768), (768, 1), 0), reinterpret_tensor(buf502, (768, ), (1, ), 0), reinterpret_tensor(buf497, (768, 768), (768, 1), 0), reinterpret_tensor(buf498, (768, ), (1, ), 0), reinterpret_tensor(buf486, (768, 768), (768, 1), 0), reinterpret_tensor(buf487, (768, ), (1, ), 0), buf482, buf483, reinterpret_tensor(buf477, (3072, 768), (768, 1), 0), reinterpret_tensor(buf478, (3072, ), (1, ), 0), reinterpret_tensor(buf473, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf474, (768, ), (1, ), 0), buf469, buf470, reinterpret_tensor(buf463, (768, 768), (768, 1), 0), reinterpret_tensor(buf464, (768, ), (1, ), 0), reinterpret_tensor(buf459, (768, 768), (768, 1), 0), reinterpret_tensor(buf460, (768, ), (1, ), 0), reinterpret_tensor(buf455, (768, 768), (768, 1), 0), reinterpret_tensor(buf456, (768, ), (1, ), 0), reinterpret_tensor(buf444, (768, 768), (768, 1), 0), reinterpret_tensor(buf445, (768, ), (1, ), 0), buf440, buf441, reinterpret_tensor(buf435, (3072, 768), (768, 1), 0), reinterpret_tensor(buf436, (3072, ), (1, ), 0), reinterpret_tensor(buf431, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf432, (768, ), (1, ), 0), buf427, buf428, reinterpret_tensor(buf421, (768, 768), (768, 1), 0), reinterpret_tensor(buf422, (768, ), (1, ), 0), reinterpret_tensor(buf417, (768, 768), (768, 1), 0), reinterpret_tensor(buf418, (768, ), (1, ), 0), reinterpret_tensor(buf413, (768, 768), (768, 1), 0), reinterpret_tensor(buf414, (768, ), (1, ), 0), reinterpret_tensor(buf402, (768, 768), (768, 1), 0), reinterpret_tensor(buf403, (768, ), (1, ), 0), buf398, buf399, reinterpret_tensor(buf393, (3072, 768), (768, 1), 0), reinterpret_tensor(buf394, (3072, ), (1, ), 0), reinterpret_tensor(buf389, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf390, (768, ), (1, ), 0), buf385, buf386, reinterpret_tensor(buf379, (768, 768), (768, 1), 0), reinterpret_tensor(buf380, (768, ), (1, ), 0), reinterpret_tensor(buf375, (768, 768), (768, 1), 0), reinterpret_tensor(buf376, (768, ), (1, ), 0), reinterpret_tensor(buf371, (768, 768), (768, 1), 0), reinterpret_tensor(buf372, (768, ), (1, ), 0), reinterpret_tensor(buf360, (768, 768), (768, 1), 0), reinterpret_tensor(buf361, (768, ), (1, ), 0), buf356, buf357, reinterpret_tensor(buf351, (3072, 768), (768, 1), 0), reinterpret_tensor(buf352, (3072, ), (1, ), 0), reinterpret_tensor(buf347, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf348, (768, ), (1, ), 0), buf343, buf344, reinterpret_tensor(buf337, (768, 768), (768, 1), 0), reinterpret_tensor(buf338, (768, ), (1, ), 0), reinterpret_tensor(buf333, (768, 768), (768, 1), 0), reinterpret_tensor(buf334, (768, ), (1, ), 0), reinterpret_tensor(buf329, (768, 768), (768, 1), 0), reinterpret_tensor(buf330, (768, ), (1, ), 0), reinterpret_tensor(buf318, (768, 768), (768, 1), 0), reinterpret_tensor(buf319, (768, ), (1, ), 0), buf314, buf315, reinterpret_tensor(buf309, (3072, 768), (768, 1), 0), reinterpret_tensor(buf310, (3072, ), (1, ), 0), reinterpret_tensor(buf305, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf306, (768, ), (1, ), 0), buf301, buf302, reinterpret_tensor(buf295, (768, 768), (768, 1), 0), reinterpret_tensor(buf296, (768, ), (1, ), 0), reinterpret_tensor(buf291, (768, 768), (768, 1), 0), reinterpret_tensor(buf292, (768, ), (1, ), 0), reinterpret_tensor(buf287, (768, 768), (768, 1), 0), reinterpret_tensor(buf288, (768, ), (1, ), 0), reinterpret_tensor(buf276, (768, 768), (768, 1), 0), reinterpret_tensor(buf277, (768, ), (1, ), 0), buf272, buf273, reinterpret_tensor(buf267, (3072, 768), (768, 1), 0), reinterpret_tensor(buf268, (3072, ), (1, ), 0), reinterpret_tensor(buf263, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf264, (768, ), (1, ), 0), buf259, buf260, reinterpret_tensor(buf253, (768, 768), (768, 1), 0), reinterpret_tensor(buf254, (768, ), (1, ), 0), reinterpret_tensor(buf249, (768, 768), (768, 1), 0), reinterpret_tensor(buf250, (768, ), (1, ), 0), reinterpret_tensor(buf245, (768, 768), (768, 1), 0), reinterpret_tensor(buf246, (768, ), (1, ), 0), reinterpret_tensor(buf234, (768, 768), (768, 1), 0), reinterpret_tensor(buf235, (768, ), (1, ), 0), buf230, buf231, reinterpret_tensor(buf225, (3072, 768), (768, 1), 0), reinterpret_tensor(buf226, (3072, ), (1, ), 0), reinterpret_tensor(buf221, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf222, (768, ), (1, ), 0), buf217, buf218, reinterpret_tensor(buf211, (768, 768), (768, 1), 0), reinterpret_tensor(buf212, (768, ), (1, ), 0), reinterpret_tensor(buf207, (768, 768), (768, 1), 0), reinterpret_tensor(buf208, (768, ), (1, ), 0), reinterpret_tensor(buf203, (768, 768), (768, 1), 0), reinterpret_tensor(buf204, (768, ), (1, ), 0), reinterpret_tensor(buf192, (768, 768), (768, 1), 0), reinterpret_tensor(buf193, (768, ), (1, ), 0), buf188, buf189, reinterpret_tensor(buf183, (3072, 768), (768, 1), 0), reinterpret_tensor(buf184, (3072, ), (1, ), 0), reinterpret_tensor(buf179, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf180, (768, ), (1, ), 0), buf175, buf176, reinterpret_tensor(buf169, (768, 768), (768, 1), 0), reinterpret_tensor(buf170, (768, ), (1, ), 0), reinterpret_tensor(buf165, (768, 768), (768, 1), 0), reinterpret_tensor(buf166, (768, ), (1, ), 0), reinterpret_tensor(buf161, (768, 768), (768, 1), 0), reinterpret_tensor(buf162, (768, ), (1, ), 0), reinterpret_tensor(buf150, (768, 768), (768, 1), 0), reinterpret_tensor(buf151, (768, ), (1, ), 0), buf146, buf147, reinterpret_tensor(buf141, (3072, 768), (768, 1), 0), reinterpret_tensor(buf142, (3072, ), (1, ), 0), reinterpret_tensor(buf137, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf138, (768, ), (1, ), 0), buf133, buf134, reinterpret_tensor(buf127, (768, 768), (768, 1), 0), reinterpret_tensor(buf128, (768, ), (1, ), 0), reinterpret_tensor(buf123, (768, 768), (768, 1), 0), reinterpret_tensor(buf124, (768, ), (1, ), 0), reinterpret_tensor(buf119, (768, 768), (768, 1), 0), reinterpret_tensor(buf120, (768, ), (1, ), 0), reinterpret_tensor(buf108, (768, 768), (768, 1), 0), reinterpret_tensor(buf109, (768, ), (1, ), 0), buf104, buf105, reinterpret_tensor(buf99, (3072, 768), (768, 1), 0), reinterpret_tensor(buf100, (3072, ), (1, ), 0), reinterpret_tensor(buf95, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf96, (768, ), (1, ), 0), buf91, buf92, reinterpret_tensor(buf85, (768, 768), (768, 1), 0), reinterpret_tensor(buf86, (768, ), (1, ), 0), reinterpret_tensor(buf81, (768, 768), (768, 1), 0), reinterpret_tensor(buf82, (768, ), (1, ), 0), reinterpret_tensor(buf77, (768, 768), (768, 1), 0), reinterpret_tensor(buf78, (768, ), (1, ), 0), reinterpret_tensor(buf66, (768, 768), (768, 1), 0), reinterpret_tensor(buf67, (768, ), (1, ), 0), buf62, buf63, reinterpret_tensor(buf57, (3072, 768), (768, 1), 0), reinterpret_tensor(buf58, (3072, ), (1, ), 0), reinterpret_tensor(buf53, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf54, (768, ), (1, ), 0), buf49, buf50, reinterpret_tensor(buf43, (768, 768), (768, 1), 0), reinterpret_tensor(buf44, (768, ), (1, ), 0), reinterpret_tensor(buf39, (768, 768), (768, 1), 0), reinterpret_tensor(buf40, (768, ), (1, ), 0), reinterpret_tensor(buf35, (768, 768), (768, 1), 0), reinterpret_tensor(buf36, (768, ), (1, ), 0), reinterpret_tensor(buf24, (768, 768), (768, 1), 0), reinterpret_tensor(buf25, (768, ), (1, ), 0), buf20, buf21, reinterpret_tensor(buf15, (3072, 768), (768, 1), 0), reinterpret_tensor(buf16, (3072, ), (1, ), 0), reinterpret_tensor(buf11, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf12, (768, ), (1, ), 0), buf7, buf8, reinterpret_tensor(buf2, (768, 768), (768, 1), 0), buf3, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_4 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((3, 5), (5, 1), device='cuda:0', dtype=torch.int64)
    expand = rand_strided((3, 5), (0, 1), device='cuda:0', dtype=torch.int64)
    slice_6 = rand_strided((1, 5), (512, 1), device='cuda:0', dtype=torch.int64)
    mul_1 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.float32)
    gt = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.bool)
    view = rand_strided((15, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    gt_1 = rand_strided((3, 12, 5, 5), (300, 25, 5, 1), device='cuda:0', dtype=torch.bool)
    view_16 = rand_strided((15, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    gt_2 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_9 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.float32)
    view_18 = rand_strided((15, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_4 = rand_strided((15, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_20 = rand_strided((15, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    gt_3 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_16 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.float32)
    view_22 = rand_strided((15, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    gt_4 = rand_strided((3, 12, 5, 5), (300, 25, 5, 1), device='cuda:0', dtype=torch.bool)
    view_38 = rand_strided((15, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    gt_5 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_22 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.float32)
    view_40 = rand_strided((15, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_10 = rand_strided((15, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_42 = rand_strided((15, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    gt_6 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_29 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.float32)
    view_44 = rand_strided((15, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    gt_7 = rand_strided((3, 12, 5, 5), (300, 25, 5, 1), device='cuda:0', dtype=torch.bool)
    view_60 = rand_strided((15, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    gt_8 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_35 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.float32)
    view_62 = rand_strided((15, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_16 = rand_strided((15, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_64 = rand_strided((15, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    gt_9 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_42 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.float32)
    view_66 = rand_strided((15, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    gt_10 = rand_strided((3, 12, 5, 5), (300, 25, 5, 1), device='cuda:0', dtype=torch.bool)
    view_82 = rand_strided((15, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    gt_11 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_48 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.float32)
    view_84 = rand_strided((15, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_22 = rand_strided((15, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_86 = rand_strided((15, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    gt_12 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_55 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.float32)
    view_88 = rand_strided((15, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    gt_13 = rand_strided((3, 12, 5, 5), (300, 25, 5, 1), device='cuda:0', dtype=torch.bool)
    view_104 = rand_strided((15, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    gt_14 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_61 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.float32)
    view_106 = rand_strided((15, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_28 = rand_strided((15, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_108 = rand_strided((15, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    gt_15 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_68 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.float32)
    view_110 = rand_strided((15, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    gt_16 = rand_strided((3, 12, 5, 5), (300, 25, 5, 1), device='cuda:0', dtype=torch.bool)
    view_126 = rand_strided((15, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    gt_17 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_74 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.float32)
    view_128 = rand_strided((15, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_34 = rand_strided((15, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_130 = rand_strided((15, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    gt_18 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_81 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.float32)
    view_132 = rand_strided((15, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    gt_19 = rand_strided((3, 12, 5, 5), (300, 25, 5, 1), device='cuda:0', dtype=torch.bool)
    view_148 = rand_strided((15, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    gt_20 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_87 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.float32)
    view_150 = rand_strided((15, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_40 = rand_strided((15, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_152 = rand_strided((15, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    gt_21 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_94 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.float32)
    view_154 = rand_strided((15, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    gt_22 = rand_strided((3, 12, 5, 5), (300, 25, 5, 1), device='cuda:0', dtype=torch.bool)
    view_170 = rand_strided((15, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    gt_23 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_100 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.float32)
    view_172 = rand_strided((15, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_46 = rand_strided((15, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_174 = rand_strided((15, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    gt_24 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_107 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.float32)
    view_176 = rand_strided((15, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    gt_25 = rand_strided((3, 12, 5, 5), (300, 25, 5, 1), device='cuda:0', dtype=torch.bool)
    view_192 = rand_strided((15, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    gt_26 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_113 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.float32)
    view_194 = rand_strided((15, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_52 = rand_strided((15, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_196 = rand_strided((15, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    gt_27 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_120 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.float32)
    view_198 = rand_strided((15, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    gt_28 = rand_strided((3, 12, 5, 5), (300, 25, 5, 1), device='cuda:0', dtype=torch.bool)
    view_214 = rand_strided((15, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    gt_29 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_126 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.float32)
    view_216 = rand_strided((15, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_58 = rand_strided((15, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_218 = rand_strided((15, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    gt_30 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_133 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.float32)
    view_220 = rand_strided((15, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    gt_31 = rand_strided((3, 12, 5, 5), (300, 25, 5, 1), device='cuda:0', dtype=torch.bool)
    view_236 = rand_strided((15, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    gt_32 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_139 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.float32)
    view_238 = rand_strided((15, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_64 = rand_strided((15, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_240 = rand_strided((15, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    gt_33 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_146 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.float32)
    view_242 = rand_strided((15, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    gt_34 = rand_strided((3, 12, 5, 5), (300, 25, 5, 1), device='cuda:0', dtype=torch.bool)
    view_258 = rand_strided((15, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    gt_35 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_152 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.float32)
    view_260 = rand_strided((15, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_70 = rand_strided((15, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_262 = rand_strided((15, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    gt_36 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_159 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.float32)
    select = rand_strided((3, 768), (3840, 1), device='cuda:0', dtype=torch.float32)
    tanh = rand_strided((3, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_133 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_24 = rand_strided((3, 5, 1), (5, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_137 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_141 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_25 = rand_strided((3, 5, 1), (5, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_145 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_150 = rand_strided((36, 5, 5), (25, 1, 5), device='cuda:0', dtype=torch.float32)
    permute_151 = rand_strided((36, 64, 5), (320, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_14 = rand_strided((3, 12, 5, 5), (300, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    permute_152 = rand_strided((36, 64, 5), (320, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_153 = rand_strided((36, 5, 64), (320, 1, 5), device='cuda:0', dtype=torch.float32)
    permute_157 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_162 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_166 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_27 = rand_strided((3, 5, 1), (5, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_170 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_174 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_28 = rand_strided((3, 5, 1), (5, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_178 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_183 = rand_strided((36, 5, 5), (25, 1, 5), device='cuda:0', dtype=torch.float32)
    permute_184 = rand_strided((36, 64, 5), (320, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_15 = rand_strided((3, 12, 5, 5), (300, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    permute_185 = rand_strided((36, 64, 5), (320, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_186 = rand_strided((36, 5, 64), (320, 1, 5), device='cuda:0', dtype=torch.float32)
    permute_190 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_195 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_199 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_30 = rand_strided((3, 5, 1), (5, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_203 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_207 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_31 = rand_strided((3, 5, 1), (5, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_211 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_216 = rand_strided((36, 5, 5), (25, 1, 5), device='cuda:0', dtype=torch.float32)
    permute_217 = rand_strided((36, 64, 5), (320, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_16 = rand_strided((3, 12, 5, 5), (300, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    permute_218 = rand_strided((36, 64, 5), (320, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_219 = rand_strided((36, 5, 64), (320, 1, 5), device='cuda:0', dtype=torch.float32)
    permute_223 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_228 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_232 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_33 = rand_strided((3, 5, 1), (5, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_236 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_240 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_34 = rand_strided((3, 5, 1), (5, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_244 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_249 = rand_strided((36, 5, 5), (25, 1, 5), device='cuda:0', dtype=torch.float32)
    permute_250 = rand_strided((36, 64, 5), (320, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_17 = rand_strided((3, 12, 5, 5), (300, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    permute_251 = rand_strided((36, 64, 5), (320, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_252 = rand_strided((36, 5, 64), (320, 1, 5), device='cuda:0', dtype=torch.float32)
    permute_256 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_261 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_265 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_36 = rand_strided((3, 5, 1), (5, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_269 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_273 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_37 = rand_strided((3, 5, 1), (5, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_277 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_282 = rand_strided((36, 5, 5), (25, 1, 5), device='cuda:0', dtype=torch.float32)
    permute_283 = rand_strided((36, 64, 5), (320, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_18 = rand_strided((3, 12, 5, 5), (300, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    permute_284 = rand_strided((36, 64, 5), (320, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_285 = rand_strided((36, 5, 64), (320, 1, 5), device='cuda:0', dtype=torch.float32)
    permute_289 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_294 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_298 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_39 = rand_strided((3, 5, 1), (5, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_302 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_306 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_40 = rand_strided((3, 5, 1), (5, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_310 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_315 = rand_strided((36, 5, 5), (25, 1, 5), device='cuda:0', dtype=torch.float32)
    permute_316 = rand_strided((36, 64, 5), (320, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_19 = rand_strided((3, 12, 5, 5), (300, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    permute_317 = rand_strided((36, 64, 5), (320, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_318 = rand_strided((36, 5, 64), (320, 1, 5), device='cuda:0', dtype=torch.float32)
    permute_322 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_327 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_331 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_42 = rand_strided((3, 5, 1), (5, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_335 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_339 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_43 = rand_strided((3, 5, 1), (5, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_343 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_348 = rand_strided((36, 5, 5), (25, 1, 5), device='cuda:0', dtype=torch.float32)
    permute_349 = rand_strided((36, 64, 5), (320, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_20 = rand_strided((3, 12, 5, 5), (300, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    permute_350 = rand_strided((36, 64, 5), (320, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_351 = rand_strided((36, 5, 64), (320, 1, 5), device='cuda:0', dtype=torch.float32)
    permute_355 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_360 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_364 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_45 = rand_strided((3, 5, 1), (5, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_368 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_372 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_46 = rand_strided((3, 5, 1), (5, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_376 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_381 = rand_strided((36, 5, 5), (25, 1, 5), device='cuda:0', dtype=torch.float32)
    permute_382 = rand_strided((36, 64, 5), (320, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_21 = rand_strided((3, 12, 5, 5), (300, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    permute_383 = rand_strided((36, 64, 5), (320, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_384 = rand_strided((36, 5, 64), (320, 1, 5), device='cuda:0', dtype=torch.float32)
    permute_388 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_393 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_397 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_48 = rand_strided((3, 5, 1), (5, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_401 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_405 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_49 = rand_strided((3, 5, 1), (5, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_409 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_414 = rand_strided((36, 5, 5), (25, 1, 5), device='cuda:0', dtype=torch.float32)
    permute_415 = rand_strided((36, 64, 5), (320, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_22 = rand_strided((3, 12, 5, 5), (300, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    permute_416 = rand_strided((36, 64, 5), (320, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_417 = rand_strided((36, 5, 64), (320, 1, 5), device='cuda:0', dtype=torch.float32)
    permute_421 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_426 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_430 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_51 = rand_strided((3, 5, 1), (5, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_434 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_438 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_52 = rand_strided((3, 5, 1), (5, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_442 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_447 = rand_strided((36, 5, 5), (25, 1, 5), device='cuda:0', dtype=torch.float32)
    permute_448 = rand_strided((36, 64, 5), (320, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_23 = rand_strided((3, 12, 5, 5), (300, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    permute_449 = rand_strided((36, 64, 5), (320, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_450 = rand_strided((36, 5, 64), (320, 1, 5), device='cuda:0', dtype=torch.float32)
    permute_454 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_459 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_463 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_54 = rand_strided((3, 5, 1), (5, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_467 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_471 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_55 = rand_strided((3, 5, 1), (5, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_475 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_480 = rand_strided((36, 5, 5), (25, 1, 5), device='cuda:0', dtype=torch.float32)
    permute_481 = rand_strided((36, 64, 5), (320, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_24 = rand_strided((3, 12, 5, 5), (300, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    permute_482 = rand_strided((36, 64, 5), (320, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_483 = rand_strided((36, 5, 64), (320, 1, 5), device='cuda:0', dtype=torch.float32)
    permute_487 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_492 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_496 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_57 = rand_strided((3, 5, 1), (5, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_500 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_504 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_58 = rand_strided((3, 5, 1), (5, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_508 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_513 = rand_strided((36, 5, 5), (25, 1, 5), device='cuda:0', dtype=torch.float32)
    permute_514 = rand_strided((36, 64, 5), (320, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_25 = rand_strided((3, 12, 5, 5), (300, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    permute_515 = rand_strided((36, 64, 5), (320, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_516 = rand_strided((36, 5, 64), (320, 1, 5), device='cuda:0', dtype=torch.float32)
    permute_520 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_525 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_529 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_60 = rand_strided((3, 5, 1), (5, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((3, 5, 768), (3840, 768, 1), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((3, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_4, primals_14, primals_20, primals_30, primals_36, primals_46, primals_52, primals_62, primals_68, primals_78, primals_84, primals_94, primals_100, primals_110, primals_116, primals_126, primals_132, primals_142, primals_148, primals_158, primals_164, primals_174, primals_180, primals_190, primals_196, primals_202, expand, slice_6, mul_1, gt, view, gt_1, view_16, gt_2, mul_9, view_18, addmm_4, view_20, gt_3, mul_16, view_22, gt_4, view_38, gt_5, mul_22, view_40, addmm_10, view_42, gt_6, mul_29, view_44, gt_7, view_60, gt_8, mul_35, view_62, addmm_16, view_64, gt_9, mul_42, view_66, gt_10, view_82, gt_11, mul_48, view_84, addmm_22, view_86, gt_12, mul_55, view_88, gt_13, view_104, gt_14, mul_61, view_106, addmm_28, view_108, gt_15, mul_68, view_110, gt_16, view_126, gt_17, mul_74, view_128, addmm_34, view_130, gt_18, mul_81, view_132, gt_19, view_148, gt_20, mul_87, view_150, addmm_40, view_152, gt_21, mul_94, view_154, gt_22, view_170, gt_23, mul_100, view_172, addmm_46, view_174, gt_24, mul_107, view_176, gt_25, view_192, gt_26, mul_113, view_194, addmm_52, view_196, gt_27, mul_120, view_198, gt_28, view_214, gt_29, mul_126, view_216, addmm_58, view_218, gt_30, mul_133, view_220, gt_31, view_236, gt_32, mul_139, view_238, addmm_64, view_240, gt_33, mul_146, view_242, gt_34, view_258, gt_35, mul_152, view_260, addmm_70, view_262, gt_36, mul_159, select, tanh, permute_133, div_24, permute_137, permute_141, div_25, permute_145, permute_150, permute_151, alias_14, permute_152, permute_153, permute_157, permute_162, permute_166, div_27, permute_170, permute_174, div_28, permute_178, permute_183, permute_184, alias_15, permute_185, permute_186, permute_190, permute_195, permute_199, div_30, permute_203, permute_207, div_31, permute_211, permute_216, permute_217, alias_16, permute_218, permute_219, permute_223, permute_228, permute_232, div_33, permute_236, permute_240, div_34, permute_244, permute_249, permute_250, alias_17, permute_251, permute_252, permute_256, permute_261, permute_265, div_36, permute_269, permute_273, div_37, permute_277, permute_282, permute_283, alias_18, permute_284, permute_285, permute_289, permute_294, permute_298, div_39, permute_302, permute_306, div_40, permute_310, permute_315, permute_316, alias_19, permute_317, permute_318, permute_322, permute_327, permute_331, div_42, permute_335, permute_339, div_43, permute_343, permute_348, permute_349, alias_20, permute_350, permute_351, permute_355, permute_360, permute_364, div_45, permute_368, permute_372, div_46, permute_376, permute_381, permute_382, alias_21, permute_383, permute_384, permute_388, permute_393, permute_397, div_48, permute_401, permute_405, div_49, permute_409, permute_414, permute_415, alias_22, permute_416, permute_417, permute_421, permute_426, permute_430, div_51, permute_434, permute_438, div_52, permute_442, permute_447, permute_448, alias_23, permute_449, permute_450, permute_454, permute_459, permute_463, div_54, permute_467, permute_471, div_55, permute_475, permute_480, permute_481, alias_24, permute_482, permute_483, permute_487, permute_492, permute_496, div_57, permute_500, permute_504, div_58, permute_508, permute_513, permute_514, alias_25, permute_515, permute_516, permute_520, permute_525, permute_529, div_60, tangents_1, tangents_2])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
