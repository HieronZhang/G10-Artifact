
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


# kernel path: /tmp/torchinductor_zhang402/3n/c3nfnptcxjbzncgjukw2kwmy6zi3aowvhgra3i5snmpzejnjifpj.py
# Source Nodes: [embedding_output, embeddings, embeddings_1, embeddings_2, inputs_embeds, mixed_query_layer, position_embeddings, token_type_embeddings], Original ATen: [aten.add, aten.embedding, aten.native_dropout, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# embedding_output => gt, mul_3, mul_4
# embeddings => add
# embeddings_1 => add_1
# embeddings_2 => add_2, add_3, mul_1, mul_2, rsqrt, sub_1, var_mean
# inputs_embeds => embedding
# mixed_query_layer => view
# position_embeddings => embedding_2
# token_type_embeddings => embedding_1
triton_per_fused_add_embedding_native_dropout_native_layer_norm_native_layer_norm_backward_view_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*i64', 5: '*fp32', 6: '*i64', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*i1', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32', 15: 'i32', 16: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(16,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_native_dropout_native_layer_norm_native_layer_norm_backward_view_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr4, out_ptr5, out_ptr6, out_ptr7, load_seed_offset, xnumel, rnumel):
    xnumel = 15
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x3 = xindex
    r2 = rindex
    x0 = xindex % 5
    tmp0 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.load(in_ptr8 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0 + 30522
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tl.device_assert(((0 <= tmp3) & (tmp3 < 30522)) | ~xmask, "index out of bounds: 0 <= tmp3 < 30522")
    tmp4 = tl.load(in_ptr1 + (r2 + (768*tmp3)), rmask & xmask, other=0.0)
    tmp6 = tmp5 + 2
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tl.device_assert(((0 <= tmp8) & (tmp8 < 2)) | ~xmask, "index out of bounds: 0 <= tmp8 < 2")
    tmp9 = tl.load(in_ptr3 + (r2 + (768*tmp8)), rmask & xmask, other=0.0)
    tmp10 = tmp4 + tmp9
    tmp12 = tmp11 + 512
    tmp13 = tmp11 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp11)
    tl.device_assert(((0 <= tmp14) & (tmp14 < 512)) | ~xmask, "index out of bounds: 0 <= tmp14 < 512")
    tmp15 = tl.load(in_ptr5 + (r2 + (768*tmp14)), rmask & xmask, other=0.0)
    tmp16 = tmp10 + tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp22 = tl.where(rmask & xmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp24 = tl.full([1], 768, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tmp17 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp33 = tl.load(in_ptr6 + load_seed_offset)
    tmp34 = r2 + (768*x3)
    tmp35 = tl.rand(tmp33, (tmp34).to(tl.uint32))
    tmp36 = 0.1
    tmp37 = tmp35 > tmp36
    tmp38 = tmp16 - tmp26
    tmp39 = 768.0
    tmp40 = tmp32 / tmp39
    tmp41 = 1e-12
    tmp42 = tmp40 + tmp41
    tmp43 = tl.math.rsqrt(tmp42)
    tmp44 = tmp38 * tmp43
    tmp45 = tmp37.to(tl.float32)
    tmp47 = tmp44 * tmp46
    tmp49 = tmp47 + tmp48
    tmp50 = tmp45 * tmp49
    tmp51 = 1.1111111111111112
    tmp52 = tmp50 * tmp51
    tmp53 = tmp43 / tmp39
    tl.store(out_ptr0 + (r2 + (768*x3)), tmp16, rmask & xmask)
    tl.store(out_ptr4 + (r2 + (768*x3)), tmp37, rmask & xmask)
    tl.store(out_ptr5 + (r2 + (768*x3)), tmp44, rmask & xmask)
    tl.store(out_ptr6 + (r2 + (768*x3)), tmp52, rmask & xmask)
    tl.store(out_ptr7 + (x3), tmp53, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_zhang402/dx/cdxca57nt7anmdpew4pxrjgtwhaedyh4njwqonil46jbjkltbe2f.py
# Source Nodes: [attention_scores], Original ATen: [aten.clone]
# attention_scores => clone
triton_poi_fused_clone_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/7k/c7kchi37kyfncjf6xfywbqmsr2lxgn34rduurxyl4j4gzm7tjrrp.py
# Source Nodes: [attention_scores], Original ATen: [aten.clone]
# attention_scores => clone_1
triton_poi_fused_clone_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 8], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2304
    xnumel = 5
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 768
    y1 = (yindex // 768)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (768*x2) + (3840*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (5*y3)), tmp2, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_zhang402/hi/chiwplgi35oc64sdsa4munfg2lyxf2oq2um22d34o2evuw2fmb5v.py
# Source Nodes: [attention_probs, attention_scores_1, attention_scores_2, extended_attention_mask_1, extended_attention_mask_3, sub], Original ATen: [aten._softmax, aten._to_copy, aten.add, aten.div, aten.mul, aten.rsub]
# attention_probs => amax, exp, sub_2, sum_1
# attention_scores_1 => div
# attention_scores_2 => add_4
# extended_attention_mask_1 => convert_element_type
# extended_attention_mask_3 => mul
# sub => sub
triton_poi_fused__softmax__to_copy_add_div_mul_rsub_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax__to_copy_add_div_mul_rsub_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 180
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 60)
    tmp0 = tl.load(in_ptr0 + (5*x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (5*x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (1 + (5*x2)), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr1 + (1 + (5*x1)), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + (2 + (5*x2)), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr1 + (2 + (5*x1)), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr0 + (3 + (5*x2)), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr1 + (3 + (5*x1)), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr0 + (4 + (5*x2)), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr1 + (4 + (5*x1)), xmask, eviction_policy='evict_last')
    tmp1 = 8.0
    tmp2 = tmp0 / tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 1.0
    tmp6 = tmp5 - tmp4
    tmp7 = -3.4028234663852886e+38
    tmp8 = tmp6 * tmp7
    tmp9 = tmp2 + tmp8
    tmp11 = tmp10 / tmp1
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp5 - tmp13
    tmp15 = tmp14 * tmp7
    tmp16 = tmp11 + tmp15
    tmp17 = triton_helpers.maximum(tmp9, tmp16)
    tmp19 = tmp18 / tmp1
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp5 - tmp21
    tmp23 = tmp22 * tmp7
    tmp24 = tmp19 + tmp23
    tmp25 = triton_helpers.maximum(tmp17, tmp24)
    tmp27 = tmp26 / tmp1
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp5 - tmp29
    tmp31 = tmp30 * tmp7
    tmp32 = tmp27 + tmp31
    tmp33 = triton_helpers.maximum(tmp25, tmp32)
    tmp35 = tmp34 / tmp1
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tmp5 - tmp37
    tmp39 = tmp38 * tmp7
    tmp40 = tmp35 + tmp39
    tmp41 = triton_helpers.maximum(tmp33, tmp40)
    tmp42 = tmp9 - tmp41
    tmp43 = tl.exp(tmp42)
    tmp44 = tmp16 - tmp41
    tmp45 = tl.exp(tmp44)
    tmp46 = tmp43 + tmp45
    tmp47 = tmp24 - tmp41
    tmp48 = tl.exp(tmp47)
    tmp49 = tmp46 + tmp48
    tmp50 = tmp32 - tmp41
    tmp51 = tl.exp(tmp50)
    tmp52 = tmp49 + tmp51
    tmp53 = tmp40 - tmp41
    tmp54 = tl.exp(tmp53)
    tmp55 = tmp52 + tmp54
    tl.store(out_ptr0 + (x2), tmp41, xmask)
    tl.store(out_ptr1 + (x2), tmp55, xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/2r/c2rljeenrbvwf2f7qbon4xron2iibe5d55pn5emdvi5cfmdxsjgq.py
# Source Nodes: [attention_probs, attention_probs_1, attention_scores_1, attention_scores_2, extended_attention_mask_1, extended_attention_mask_3, sub], Original ATen: [aten._softmax, aten._to_copy, aten.add, aten.div, aten.mul, aten.native_dropout, aten.rsub]
# attention_probs => div_1, exp, sub_2
# attention_probs_1 => gt_1, mul_5, mul_6
# attention_scores_1 => div
# attention_scores_2 => add_4
# extended_attention_mask_1 => convert_element_type
# extended_attention_mask_3 => mul
# sub => sub
triton_poi_fused__softmax__to_copy_add_div_mul_native_dropout_rsub_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax__to_copy_add_div_mul_native_dropout_rsub_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 900
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x1 = xindex % 5
    x3 = (xindex // 300)
    x4 = (xindex // 5)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp9 = tl.load(in_ptr2 + (x1 + (5*x3)), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr3 + (x4), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x4), xmask, eviction_policy='evict_last')
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = 8.0
    tmp8 = tmp6 / tmp7
    tmp10 = tmp9.to(tl.float32)
    tmp11 = 1.0
    tmp12 = tmp11 - tmp10
    tmp13 = -3.4028234663852886e+38
    tmp14 = tmp12 * tmp13
    tmp15 = tmp8 + tmp14
    tmp17 = tmp15 - tmp16
    tmp18 = tl.exp(tmp17)
    tmp20 = tmp18 / tmp19
    tmp21 = tmp5 * tmp20
    tmp22 = 1.1111111111111112
    tmp23 = tmp21 * tmp22
    tl.store(out_ptr1 + (x0), tmp4, xmask)
    tl.store(out_ptr2 + (x0), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/ng/cngxo6wq3tuytarvv2vk54q3abkgsis5ohtrf2dszxyoscqedl2l.py
# Source Nodes: [hidden_states], Original ATen: [aten.view]
# hidden_states => view_16
triton_poi_fused_view_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_5', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_zhang402/hk/chklcqc727a5j2ezze6otu6lpinlhk3p6c5c52wnd75thvkbqzai.py
# Source Nodes: [add_2, attention_output, embedding_output, embeddings_2, hidden_states_1, hidden_states_3], Original ATen: [aten.add, aten.native_dropout, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# add_2 => add_5
# attention_output => add_6, add_7, mul_10, mul_9, rsqrt_1, sub_3, var_mean_1
# embedding_output => mul_3, mul_4
# embeddings_2 => add_3, mul_2
# hidden_states_1 => gt_2, mul_7, mul_8
# hidden_states_3 => view_18
triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*i1', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*i1', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(15,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_6', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, out_ptr4, out_ptr5, out_ptr6, load_seed_offset, xnumel, rnumel):
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
    tmp6 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask).to(tl.int1)
    tmp14 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp15 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp45 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp47 = tl.load(in_ptr7 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = r1 + (768*x0)
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = 1.1111111111111112
    tmp11 = tmp9 * tmp10
    tmp13 = tmp12.to(tl.float32)
    tmp16 = tmp14 * tmp15
    tmp18 = tmp16 + tmp17
    tmp19 = tmp13 * tmp18
    tmp20 = tmp19 * tmp10
    tmp21 = tmp11 + tmp20
    tmp22 = tl.broadcast_to(tmp21, [RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp27 = tl.where(rmask & xmask, tmp25, 0)
    tmp28 = triton_helpers.promote_to_tensor(tl.sum(tmp27, 0))
    tmp29 = tl.full([1], 768, tl.int32)
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp28 / tmp30
    tmp32 = tmp22 - tmp31
    tmp33 = tmp32 * tmp32
    tmp34 = tl.broadcast_to(tmp33, [RBLOCK])
    tmp36 = tl.where(rmask & xmask, tmp34, 0)
    tmp37 = triton_helpers.promote_to_tensor(tl.sum(tmp36, 0))
    tmp38 = tmp21 - tmp31
    tmp39 = 768.0
    tmp40 = tmp37 / tmp39
    tmp41 = 1e-12
    tmp42 = tmp40 + tmp41
    tmp43 = tl.math.rsqrt(tmp42)
    tmp44 = tmp38 * tmp43
    tmp46 = tmp44 * tmp45
    tmp48 = tmp46 + tmp47
    tmp49 = tmp43 / tmp39
    tl.store(out_ptr1 + (r1 + (768*x0)), tmp4, rmask & xmask)
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp44, rmask & xmask)
    tl.store(out_ptr5 + (r1 + (768*x0)), tmp48, rmask & xmask)
    tl.store(out_ptr6 + (x0), tmp49, xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/ur/curk3dkxhlwyt27akb3df56i5xnrqpoyre5zfuou64734tvxus3n.py
# Source Nodes: [hidden_states_5, intermediate_output], Original ATen: [aten.gelu, aten.view]
# hidden_states_5 => view_20
# intermediate_output => add_8, erf, mul_11, mul_12, mul_13
triton_poi_fused_gelu_view_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 46080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = tl.math.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/xl/cxl5waohw4stvly535wcp5vlqqi4g6htvwlqs5t74sjxid4ygkke.py
# Source Nodes: [add_3, attention_output, hidden_states_6, hidden_states_8, mixed_query_layer_1], Original ATen: [aten.add, aten.native_dropout, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# add_3 => add_9
# attention_output => add_7, mul_10
# hidden_states_6 => gt_3, mul_14, mul_15
# hidden_states_8 => add_10, add_11, mul_16, mul_17, rsqrt_2, sub_4, var_mean_2
# mixed_query_layer_1 => view_22
triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i1', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(14,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_8', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, out_ptr4, out_ptr5, out_ptr6, load_seed_offset, xnumel, rnumel):
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
    tmp6 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp41 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = r1 + (768*x0)
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = 1.1111111111111112
    tmp11 = tmp9 * tmp10
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp11 + tmp16
    tmp18 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp20 = tl.where(rmask & xmask, tmp18, 0)
    tmp21 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tl.full([1], 768, tl.int32)
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp24 / tmp26
    tmp28 = tmp18 - tmp27
    tmp29 = tmp28 * tmp28
    tmp30 = tl.broadcast_to(tmp29, [RBLOCK])
    tmp32 = tl.where(rmask & xmask, tmp30, 0)
    tmp33 = triton_helpers.promote_to_tensor(tl.sum(tmp32, 0))
    tmp34 = tmp17 - tmp27
    tmp35 = 768.0
    tmp36 = tmp33 / tmp35
    tmp37 = 1e-12
    tmp38 = tmp36 + tmp37
    tmp39 = tl.math.rsqrt(tmp38)
    tmp40 = tmp34 * tmp39
    tmp42 = tmp40 * tmp41
    tmp44 = tmp42 + tmp43
    tmp45 = tmp39 / tmp35
    tl.store(out_ptr1 + (r1 + (768*x0)), tmp4, rmask & xmask)
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp17, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp40, rmask & xmask)
    tl.store(out_ptr5 + (r1 + (768*x0)), tmp44, rmask & xmask)
    tl.store(out_ptr6 + (x0), tmp45, xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/hc/chcd7ihj4zo3w7x7da52ufilg2i7l3dwys3o7tzzcfrnqlpt3cv7.py
# Source Nodes: [attention_probs_20, attention_probs_21, attention_scores_31, attention_scores_32, extended_attention_mask_1, extended_attention_mask_3, sub], Original ATen: [aten._softmax, aten._to_copy, aten.add, aten.detach, aten.div, aten.mul, aten.native_dropout, aten.rsub]
# attention_probs_20 => div_21, exp_10, sub_32
# attention_probs_21 => gt_31, mul_135, mul_136
# attention_scores_31 => div_20
# attention_scores_32 => add_84
# extended_attention_mask_1 => convert_element_type
# extended_attention_mask_3 => mul
# sub => sub
triton_poi_fused__softmax__to_copy_add_detach_div_mul_native_dropout_rsub_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax__to_copy_add_detach_div_mul_native_dropout_rsub_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, out_ptr3, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 900
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x1 = xindex % 5
    x3 = (xindex // 300)
    x4 = (xindex // 5)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp9 = tl.load(in_ptr2 + (x1 + (5*x3)), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr3 + (x4), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x4), xmask, eviction_policy='evict_last')
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = 8.0
    tmp8 = tmp6 / tmp7
    tmp10 = tmp9.to(tl.float32)
    tmp11 = 1.0
    tmp12 = tmp11 - tmp10
    tmp13 = -3.4028234663852886e+38
    tmp14 = tmp12 * tmp13
    tmp15 = tmp8 + tmp14
    tmp17 = tmp15 - tmp16
    tmp18 = tl.exp(tmp17)
    tmp20 = tmp18 / tmp19
    tmp21 = tmp5 * tmp20
    tmp22 = 1.1111111111111112
    tmp23 = tmp21 * tmp22
    tl.store(out_ptr1 + (x0), tmp4, xmask)
    tl.store(out_ptr2 + (x0), tmp23, xmask)
    tl.store(out_ptr3 + (x0), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/qp/cqpmgx7usmv47z52hsqod2vvjb6bhw2tbpeat5x476jvt4wect56.py
# Source Nodes: [attention_probs, attention_probs_10, attention_probs_12, attention_probs_14, attention_probs_16, attention_probs_18, attention_probs_2, attention_probs_22, attention_probs_23, attention_probs_4, attention_probs_6, attention_probs_8, attention_scores_1, attention_scores_10, attention_scores_11, attention_scores_13, attention_scores_14, attention_scores_16, attention_scores_17, attention_scores_19, attention_scores_2, attention_scores_20, attention_scores_22, attention_scores_23, attention_scores_25, attention_scores_26, attention_scores_28, attention_scores_29, attention_scores_34, attention_scores_35, attention_scores_4, attention_scores_5, attention_scores_7, attention_scores_8, extended_attention_mask_1, extended_attention_mask_3, sub], Original ATen: [aten._softmax, aten._to_copy, aten.add, aten.detach, aten.div, aten.mul, aten.native_dropout, aten.rsub]
# attention_probs => div_1, exp, sub_2
# attention_probs_10 => div_11, exp_5, sub_17
# attention_probs_12 => div_13, exp_6, sub_20
# attention_probs_14 => div_15, exp_7, sub_23
# attention_probs_16 => div_17, exp_8, sub_26
# attention_probs_18 => div_19, exp_9, sub_29
# attention_probs_2 => div_3, exp_1, sub_5
# attention_probs_22 => div_23, exp_11, sub_35
# attention_probs_23 => gt_34, mul_148, mul_149
# attention_probs_4 => div_5, exp_2, sub_8
# attention_probs_6 => div_7, exp_3, sub_11
# attention_probs_8 => div_9, exp_4, sub_14
# attention_scores_1 => div
# attention_scores_10 => div_6
# attention_scores_11 => add_28
# attention_scores_13 => div_8
# attention_scores_14 => add_36
# attention_scores_16 => div_10
# attention_scores_17 => add_44
# attention_scores_19 => div_12
# attention_scores_2 => add_4
# attention_scores_20 => add_52
# attention_scores_22 => div_14
# attention_scores_23 => add_60
# attention_scores_25 => div_16
# attention_scores_26 => add_68
# attention_scores_28 => div_18
# attention_scores_29 => add_76
# attention_scores_34 => div_22
# attention_scores_35 => add_92
# attention_scores_4 => div_2
# attention_scores_5 => add_12
# attention_scores_7 => div_4
# attention_scores_8 => add_20
# extended_attention_mask_1 => convert_element_type
# extended_attention_mask_3 => mul
# sub => sub
triton_poi_fused__softmax__to_copy_add_detach_div_mul_native_dropout_rsub_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*i64', 11: '*fp32', 12: '*i64', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: '*fp32', 35: '*i1', 36: '*fp32', 37: '*fp32', 38: 'i32', 39: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax__to_copy_add_detach_div_mul_native_dropout_rsub_10', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2', 'in_out_ptr3', 'in_out_ptr4', 'in_out_ptr5', 'in_out_ptr6', 'in_out_ptr7', 'in_out_ptr8', 'in_out_ptr9']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_out_ptr3, in_out_ptr4, in_out_ptr5, in_out_ptr6, in_out_ptr7, in_out_ptr8, in_out_ptr9, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, out_ptr1, out_ptr2, out_ptr3, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 900
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x1 = xindex % 5
    x3 = (xindex // 300)
    x4 = (xindex // 5)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp9 = tl.load(in_ptr2 + (x1 + (5*x3)), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr3 + (x4), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x4), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp27 = tl.load(in_ptr5 + (x4), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr6 + (x4), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_out_ptr1 + (x0), xmask)
    tmp35 = tl.load(in_ptr7 + (x4), xmask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr8 + (x4), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_out_ptr2 + (x0), xmask)
    tmp43 = tl.load(in_ptr9 + (x4), xmask, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr10 + (x4), xmask, eviction_policy='evict_last')
    tmp48 = tl.load(in_out_ptr3 + (x0), xmask)
    tmp51 = tl.load(in_ptr11 + (x4), xmask, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr12 + (x4), xmask, eviction_policy='evict_last')
    tmp56 = tl.load(in_out_ptr4 + (x0), xmask)
    tmp59 = tl.load(in_ptr13 + (x4), xmask, eviction_policy='evict_last')
    tmp62 = tl.load(in_ptr14 + (x4), xmask, eviction_policy='evict_last')
    tmp64 = tl.load(in_out_ptr5 + (x0), xmask)
    tmp67 = tl.load(in_ptr15 + (x4), xmask, eviction_policy='evict_last')
    tmp70 = tl.load(in_ptr16 + (x4), xmask, eviction_policy='evict_last')
    tmp72 = tl.load(in_out_ptr6 + (x0), xmask)
    tmp75 = tl.load(in_ptr17 + (x4), xmask, eviction_policy='evict_last')
    tmp78 = tl.load(in_ptr18 + (x4), xmask, eviction_policy='evict_last')
    tmp80 = tl.load(in_out_ptr7 + (x0), xmask)
    tmp83 = tl.load(in_ptr19 + (x4), xmask, eviction_policy='evict_last')
    tmp86 = tl.load(in_ptr20 + (x4), xmask, eviction_policy='evict_last')
    tmp88 = tl.load(in_out_ptr8 + (x0), xmask)
    tmp91 = tl.load(in_ptr21 + (x4), xmask, eviction_policy='evict_last')
    tmp94 = tl.load(in_ptr22 + (x4), xmask, eviction_policy='evict_last')
    tmp96 = tl.load(in_out_ptr9 + (x0), xmask)
    tmp99 = tl.load(in_ptr23 + (x4), xmask, eviction_policy='evict_last')
    tmp102 = tl.load(in_ptr24 + (x4), xmask, eviction_policy='evict_last')
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = 8.0
    tmp8 = tmp6 / tmp7
    tmp10 = tmp9.to(tl.float32)
    tmp11 = 1.0
    tmp12 = tmp11 - tmp10
    tmp13 = -3.4028234663852886e+38
    tmp14 = tmp12 * tmp13
    tmp15 = tmp8 + tmp14
    tmp17 = tmp15 - tmp16
    tmp18 = tl.exp(tmp17)
    tmp20 = tmp18 / tmp19
    tmp21 = tmp5 * tmp20
    tmp22 = 1.1111111111111112
    tmp23 = tmp21 * tmp22
    tmp25 = tmp24 / tmp7
    tmp26 = tmp25 + tmp14
    tmp28 = tmp26 - tmp27
    tmp29 = tl.exp(tmp28)
    tmp31 = tmp29 / tmp30
    tmp33 = tmp32 / tmp7
    tmp34 = tmp33 + tmp14
    tmp36 = tmp34 - tmp35
    tmp37 = tl.exp(tmp36)
    tmp39 = tmp37 / tmp38
    tmp41 = tmp40 / tmp7
    tmp42 = tmp41 + tmp14
    tmp44 = tmp42 - tmp43
    tmp45 = tl.exp(tmp44)
    tmp47 = tmp45 / tmp46
    tmp49 = tmp48 / tmp7
    tmp50 = tmp49 + tmp14
    tmp52 = tmp50 - tmp51
    tmp53 = tl.exp(tmp52)
    tmp55 = tmp53 / tmp54
    tmp57 = tmp56 / tmp7
    tmp58 = tmp57 + tmp14
    tmp60 = tmp58 - tmp59
    tmp61 = tl.exp(tmp60)
    tmp63 = tmp61 / tmp62
    tmp65 = tmp64 / tmp7
    tmp66 = tmp65 + tmp14
    tmp68 = tmp66 - tmp67
    tmp69 = tl.exp(tmp68)
    tmp71 = tmp69 / tmp70
    tmp73 = tmp72 / tmp7
    tmp74 = tmp73 + tmp14
    tmp76 = tmp74 - tmp75
    tmp77 = tl.exp(tmp76)
    tmp79 = tmp77 / tmp78
    tmp81 = tmp80 / tmp7
    tmp82 = tmp81 + tmp14
    tmp84 = tmp82 - tmp83
    tmp85 = tl.exp(tmp84)
    tmp87 = tmp85 / tmp86
    tmp89 = tmp88 / tmp7
    tmp90 = tmp89 + tmp14
    tmp92 = tmp90 - tmp91
    tmp93 = tl.exp(tmp92)
    tmp95 = tmp93 / tmp94
    tmp97 = tmp96 / tmp7
    tmp98 = tmp97 + tmp14
    tmp100 = tmp98 - tmp99
    tmp101 = tl.exp(tmp100)
    tmp103 = tmp101 / tmp102
    tl.store(out_ptr1 + (x0), tmp4, xmask)
    tl.store(out_ptr2 + (x0), tmp23, xmask)
    tl.store(out_ptr3 + (x0), tmp20, xmask)
    tl.store(in_out_ptr0 + (x0), tmp31, xmask)
    tl.store(in_out_ptr1 + (x0), tmp39, xmask)
    tl.store(in_out_ptr2 + (x0), tmp47, xmask)
    tl.store(in_out_ptr3 + (x0), tmp55, xmask)
    tl.store(in_out_ptr4 + (x0), tmp63, xmask)
    tl.store(in_out_ptr5 + (x0), tmp71, xmask)
    tl.store(in_out_ptr6 + (x0), tmp79, xmask)
    tl.store(in_out_ptr7 + (x0), tmp87, xmask)
    tl.store(in_out_ptr8 + (x0), tmp95, xmask)
    tl.store(in_out_ptr9 + (x0), tmp103, xmask)
''')


# kernel path: /tmp/torchinductor_zhang402/bd/cbdgyitwsxmjvxtxxh6obghwibshyqzr3bvgjyjw2qxbplj4gw3u.py
# Source Nodes: [pooled_output_2], Original ATen: [aten.tanh]
# pooled_output_2 => tanh
triton_poi_fused_tanh_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_tanh_11', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.math.tanh(tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203 = args
    args.clear()
    assert_size_stride(primals_1, (30522, 768), (768, 1))
    assert_size_stride(primals_2, (2, 768), (768, 1))
    assert_size_stride(primals_3, (512, 768), (768, 1))
    assert_size_stride(primals_4, (768, ), (1, ))
    assert_size_stride(primals_5, (768, ), (1, ))
    assert_size_stride(primals_6, (768, 768), (768, 1))
    assert_size_stride(primals_7, (768, ), (1, ))
    assert_size_stride(primals_8, (768, 768), (768, 1))
    assert_size_stride(primals_9, (768, ), (1, ))
    assert_size_stride(primals_10, (768, 768), (768, 1))
    assert_size_stride(primals_11, (768, ), (1, ))
    assert_size_stride(primals_12, (768, 768), (768, 1))
    assert_size_stride(primals_13, (768, ), (1, ))
    assert_size_stride(primals_14, (768, ), (1, ))
    assert_size_stride(primals_15, (768, ), (1, ))
    assert_size_stride(primals_16, (3072, 768), (768, 1))
    assert_size_stride(primals_17, (3072, ), (1, ))
    assert_size_stride(primals_18, (768, 3072), (3072, 1))
    assert_size_stride(primals_19, (768, ), (1, ))
    assert_size_stride(primals_20, (768, ), (1, ))
    assert_size_stride(primals_21, (768, ), (1, ))
    assert_size_stride(primals_22, (768, 768), (768, 1))
    assert_size_stride(primals_23, (768, ), (1, ))
    assert_size_stride(primals_24, (768, 768), (768, 1))
    assert_size_stride(primals_25, (768, ), (1, ))
    assert_size_stride(primals_26, (768, 768), (768, 1))
    assert_size_stride(primals_27, (768, ), (1, ))
    assert_size_stride(primals_28, (768, 768), (768, 1))
    assert_size_stride(primals_29, (768, ), (1, ))
    assert_size_stride(primals_30, (768, ), (1, ))
    assert_size_stride(primals_31, (768, ), (1, ))
    assert_size_stride(primals_32, (3072, 768), (768, 1))
    assert_size_stride(primals_33, (3072, ), (1, ))
    assert_size_stride(primals_34, (768, 3072), (3072, 1))
    assert_size_stride(primals_35, (768, ), (1, ))
    assert_size_stride(primals_36, (768, ), (1, ))
    assert_size_stride(primals_37, (768, ), (1, ))
    assert_size_stride(primals_38, (768, 768), (768, 1))
    assert_size_stride(primals_39, (768, ), (1, ))
    assert_size_stride(primals_40, (768, 768), (768, 1))
    assert_size_stride(primals_41, (768, ), (1, ))
    assert_size_stride(primals_42, (768, 768), (768, 1))
    assert_size_stride(primals_43, (768, ), (1, ))
    assert_size_stride(primals_44, (768, 768), (768, 1))
    assert_size_stride(primals_45, (768, ), (1, ))
    assert_size_stride(primals_46, (768, ), (1, ))
    assert_size_stride(primals_47, (768, ), (1, ))
    assert_size_stride(primals_48, (3072, 768), (768, 1))
    assert_size_stride(primals_49, (3072, ), (1, ))
    assert_size_stride(primals_50, (768, 3072), (3072, 1))
    assert_size_stride(primals_51, (768, ), (1, ))
    assert_size_stride(primals_52, (768, ), (1, ))
    assert_size_stride(primals_53, (768, ), (1, ))
    assert_size_stride(primals_54, (768, 768), (768, 1))
    assert_size_stride(primals_55, (768, ), (1, ))
    assert_size_stride(primals_56, (768, 768), (768, 1))
    assert_size_stride(primals_57, (768, ), (1, ))
    assert_size_stride(primals_58, (768, 768), (768, 1))
    assert_size_stride(primals_59, (768, ), (1, ))
    assert_size_stride(primals_60, (768, 768), (768, 1))
    assert_size_stride(primals_61, (768, ), (1, ))
    assert_size_stride(primals_62, (768, ), (1, ))
    assert_size_stride(primals_63, (768, ), (1, ))
    assert_size_stride(primals_64, (3072, 768), (768, 1))
    assert_size_stride(primals_65, (3072, ), (1, ))
    assert_size_stride(primals_66, (768, 3072), (3072, 1))
    assert_size_stride(primals_67, (768, ), (1, ))
    assert_size_stride(primals_68, (768, ), (1, ))
    assert_size_stride(primals_69, (768, ), (1, ))
    assert_size_stride(primals_70, (768, 768), (768, 1))
    assert_size_stride(primals_71, (768, ), (1, ))
    assert_size_stride(primals_72, (768, 768), (768, 1))
    assert_size_stride(primals_73, (768, ), (1, ))
    assert_size_stride(primals_74, (768, 768), (768, 1))
    assert_size_stride(primals_75, (768, ), (1, ))
    assert_size_stride(primals_76, (768, 768), (768, 1))
    assert_size_stride(primals_77, (768, ), (1, ))
    assert_size_stride(primals_78, (768, ), (1, ))
    assert_size_stride(primals_79, (768, ), (1, ))
    assert_size_stride(primals_80, (3072, 768), (768, 1))
    assert_size_stride(primals_81, (3072, ), (1, ))
    assert_size_stride(primals_82, (768, 3072), (3072, 1))
    assert_size_stride(primals_83, (768, ), (1, ))
    assert_size_stride(primals_84, (768, ), (1, ))
    assert_size_stride(primals_85, (768, ), (1, ))
    assert_size_stride(primals_86, (768, 768), (768, 1))
    assert_size_stride(primals_87, (768, ), (1, ))
    assert_size_stride(primals_88, (768, 768), (768, 1))
    assert_size_stride(primals_89, (768, ), (1, ))
    assert_size_stride(primals_90, (768, 768), (768, 1))
    assert_size_stride(primals_91, (768, ), (1, ))
    assert_size_stride(primals_92, (768, 768), (768, 1))
    assert_size_stride(primals_93, (768, ), (1, ))
    assert_size_stride(primals_94, (768, ), (1, ))
    assert_size_stride(primals_95, (768, ), (1, ))
    assert_size_stride(primals_96, (3072, 768), (768, 1))
    assert_size_stride(primals_97, (3072, ), (1, ))
    assert_size_stride(primals_98, (768, 3072), (3072, 1))
    assert_size_stride(primals_99, (768, ), (1, ))
    assert_size_stride(primals_100, (768, ), (1, ))
    assert_size_stride(primals_101, (768, ), (1, ))
    assert_size_stride(primals_102, (768, 768), (768, 1))
    assert_size_stride(primals_103, (768, ), (1, ))
    assert_size_stride(primals_104, (768, 768), (768, 1))
    assert_size_stride(primals_105, (768, ), (1, ))
    assert_size_stride(primals_106, (768, 768), (768, 1))
    assert_size_stride(primals_107, (768, ), (1, ))
    assert_size_stride(primals_108, (768, 768), (768, 1))
    assert_size_stride(primals_109, (768, ), (1, ))
    assert_size_stride(primals_110, (768, ), (1, ))
    assert_size_stride(primals_111, (768, ), (1, ))
    assert_size_stride(primals_112, (3072, 768), (768, 1))
    assert_size_stride(primals_113, (3072, ), (1, ))
    assert_size_stride(primals_114, (768, 3072), (3072, 1))
    assert_size_stride(primals_115, (768, ), (1, ))
    assert_size_stride(primals_116, (768, ), (1, ))
    assert_size_stride(primals_117, (768, ), (1, ))
    assert_size_stride(primals_118, (768, 768), (768, 1))
    assert_size_stride(primals_119, (768, ), (1, ))
    assert_size_stride(primals_120, (768, 768), (768, 1))
    assert_size_stride(primals_121, (768, ), (1, ))
    assert_size_stride(primals_122, (768, 768), (768, 1))
    assert_size_stride(primals_123, (768, ), (1, ))
    assert_size_stride(primals_124, (768, 768), (768, 1))
    assert_size_stride(primals_125, (768, ), (1, ))
    assert_size_stride(primals_126, (768, ), (1, ))
    assert_size_stride(primals_127, (768, ), (1, ))
    assert_size_stride(primals_128, (3072, 768), (768, 1))
    assert_size_stride(primals_129, (3072, ), (1, ))
    assert_size_stride(primals_130, (768, 3072), (3072, 1))
    assert_size_stride(primals_131, (768, ), (1, ))
    assert_size_stride(primals_132, (768, ), (1, ))
    assert_size_stride(primals_133, (768, ), (1, ))
    assert_size_stride(primals_134, (768, 768), (768, 1))
    assert_size_stride(primals_135, (768, ), (1, ))
    assert_size_stride(primals_136, (768, 768), (768, 1))
    assert_size_stride(primals_137, (768, ), (1, ))
    assert_size_stride(primals_138, (768, 768), (768, 1))
    assert_size_stride(primals_139, (768, ), (1, ))
    assert_size_stride(primals_140, (768, 768), (768, 1))
    assert_size_stride(primals_141, (768, ), (1, ))
    assert_size_stride(primals_142, (768, ), (1, ))
    assert_size_stride(primals_143, (768, ), (1, ))
    assert_size_stride(primals_144, (3072, 768), (768, 1))
    assert_size_stride(primals_145, (3072, ), (1, ))
    assert_size_stride(primals_146, (768, 3072), (3072, 1))
    assert_size_stride(primals_147, (768, ), (1, ))
    assert_size_stride(primals_148, (768, ), (1, ))
    assert_size_stride(primals_149, (768, ), (1, ))
    assert_size_stride(primals_150, (768, 768), (768, 1))
    assert_size_stride(primals_151, (768, ), (1, ))
    assert_size_stride(primals_152, (768, 768), (768, 1))
    assert_size_stride(primals_153, (768, ), (1, ))
    assert_size_stride(primals_154, (768, 768), (768, 1))
    assert_size_stride(primals_155, (768, ), (1, ))
    assert_size_stride(primals_156, (768, 768), (768, 1))
    assert_size_stride(primals_157, (768, ), (1, ))
    assert_size_stride(primals_158, (768, ), (1, ))
    assert_size_stride(primals_159, (768, ), (1, ))
    assert_size_stride(primals_160, (3072, 768), (768, 1))
    assert_size_stride(primals_161, (3072, ), (1, ))
    assert_size_stride(primals_162, (768, 3072), (3072, 1))
    assert_size_stride(primals_163, (768, ), (1, ))
    assert_size_stride(primals_164, (768, ), (1, ))
    assert_size_stride(primals_165, (768, ), (1, ))
    assert_size_stride(primals_166, (768, 768), (768, 1))
    assert_size_stride(primals_167, (768, ), (1, ))
    assert_size_stride(primals_168, (768, 768), (768, 1))
    assert_size_stride(primals_169, (768, ), (1, ))
    assert_size_stride(primals_170, (768, 768), (768, 1))
    assert_size_stride(primals_171, (768, ), (1, ))
    assert_size_stride(primals_172, (768, 768), (768, 1))
    assert_size_stride(primals_173, (768, ), (1, ))
    assert_size_stride(primals_174, (768, ), (1, ))
    assert_size_stride(primals_175, (768, ), (1, ))
    assert_size_stride(primals_176, (3072, 768), (768, 1))
    assert_size_stride(primals_177, (3072, ), (1, ))
    assert_size_stride(primals_178, (768, 3072), (3072, 1))
    assert_size_stride(primals_179, (768, ), (1, ))
    assert_size_stride(primals_180, (768, ), (1, ))
    assert_size_stride(primals_181, (768, ), (1, ))
    assert_size_stride(primals_182, (768, 768), (768, 1))
    assert_size_stride(primals_183, (768, ), (1, ))
    assert_size_stride(primals_184, (768, 768), (768, 1))
    assert_size_stride(primals_185, (768, ), (1, ))
    assert_size_stride(primals_186, (768, 768), (768, 1))
    assert_size_stride(primals_187, (768, ), (1, ))
    assert_size_stride(primals_188, (768, 768), (768, 1))
    assert_size_stride(primals_189, (768, ), (1, ))
    assert_size_stride(primals_190, (768, ), (1, ))
    assert_size_stride(primals_191, (768, ), (1, ))
    assert_size_stride(primals_192, (3072, 768), (768, 1))
    assert_size_stride(primals_193, (3072, ), (1, ))
    assert_size_stride(primals_194, (768, 3072), (3072, 1))
    assert_size_stride(primals_195, (768, ), (1, ))
    assert_size_stride(primals_196, (768, ), (1, ))
    assert_size_stride(primals_197, (768, ), (1, ))
    assert_size_stride(primals_198, (768, 768), (768, 1))
    assert_size_stride(primals_199, (768, ), (1, ))
    assert_size_stride(primals_200, (1, 512), (512, 1))
    assert_size_stride(primals_201, (1, 512), (512, 1))
    assert_size_stride(primals_202, (3, 5), (5, 1))
    assert_size_stride(primals_203, (3, 5), (5, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf5 = empty((37, ), device='cuda', dtype=torch.int64)
        # Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [37], out=buf5)
        buf0 = empty((3, 5, 768), device='cuda', dtype=torch.float32)
        buf7 = empty((3, 5, 768), device='cuda', dtype=torch.bool)
        buf4 = empty((3, 5, 768), device='cuda', dtype=torch.float32)
        buf8 = empty((15, 768), device='cuda', dtype=torch.float32)
        buf455 = empty((3, 5, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [embedding_output, embeddings, embeddings_1, embeddings_2, inputs_embeds, mixed_query_layer, position_embeddings, token_type_embeddings], Original ATen: [aten.add, aten.embedding, aten.native_dropout, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        stream0 = get_cuda_stream(0)
        triton_per_fused_add_embedding_native_dropout_native_layer_norm_native_layer_norm_backward_view_0.run(primals_202, primals_1, primals_200, primals_2, primals_201, primals_3, buf5, primals_4, primals_5, buf0, buf7, buf4, buf8, buf455, 0, 15, 768, grid=grid(15), stream=stream0)
        del primals_1
        del primals_2
        del primals_3
        buf9 = reinterpret_tensor(buf0, (15, 768), (768, 1), 0); del buf0  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf8, reinterpret_tensor(primals_6, (768, 768), (1, 768), 0), out=buf9)
        buf10 = empty((15, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf8, reinterpret_tensor(primals_8, (768, 768), (1, 768), 0), out=buf10)
        buf11 = empty((15, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf8, reinterpret_tensor(primals_10, (768, 768), (1, 768), 0), out=buf11)
        buf12 = empty((3, 12, 5, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf9, primals_7, buf12, 11520, grid=grid(11520), stream=stream0)
        del primals_7
        buf13 = reinterpret_tensor(buf9, (3, 12, 64, 5), (3840, 320, 5, 1), 0); del buf9  # reuse
        # Source Nodes: [attention_scores], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf10, primals_9, buf13, 2304, 5, grid=grid(2304, 5), stream=stream0)
        del primals_9
        buf14 = empty((36, 5, 5), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf12, (36, 5, 64), (320, 64, 1), 0), reinterpret_tensor(buf13, (36, 64, 5), (320, 5, 1), 0), out=buf14)
        buf15 = empty_strided((3, 12, 5, 1), (60, 5, 1, 180), device='cuda', dtype=torch.float32)
        buf16 = empty_strided((3, 12, 5, 1), (60, 5, 1, 180), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs, attention_scores_1, attention_scores_2, extended_attention_mask_1, extended_attention_mask_3, sub], Original ATen: [aten._softmax, aten._to_copy, aten.add, aten.div, aten.mul, aten.rsub]
        triton_poi_fused__softmax__to_copy_add_div_mul_rsub_3.run(buf14, primals_203, buf15, buf16, 180, grid=grid(180), stream=stream0)
        buf18 = empty((3, 12, 5, 5), device='cuda', dtype=torch.bool)
        buf19 = empty((3, 12, 5, 5), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs, attention_probs_1, attention_scores_1, attention_scores_2, extended_attention_mask_1, extended_attention_mask_3, sub], Original ATen: [aten._softmax, aten._to_copy, aten.add, aten.div, aten.mul, aten.native_dropout, aten.rsub]
        triton_poi_fused__softmax__to_copy_add_div_mul_native_dropout_rsub_4.run(buf5, buf14, primals_203, buf15, buf16, buf18, buf19, 1, 900, grid=grid(900), stream=stream0)
        buf20 = reinterpret_tensor(buf10, (3, 12, 5, 64), (3840, 320, 64, 1), 0); del buf10  # reuse
        # Source Nodes: [context_layer], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf11, primals_11, buf20, 11520, grid=grid(11520), stream=stream0)
        del primals_11
        buf21 = reinterpret_tensor(buf11, (36, 5, 64), (320, 64, 1), 0); del buf11  # reuse
        # Source Nodes: [context_layer], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf19, (36, 5, 5), (25, 5, 1), 0), reinterpret_tensor(buf20, (36, 5, 64), (320, 64, 1), 0), out=buf21)
        buf22 = empty((15, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf21, buf22, 11520, grid=grid(11520), stream=stream0)
        buf23 = reinterpret_tensor(buf21, (15, 768), (768, 1), 0); del buf21  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf22, reinterpret_tensor(primals_12, (768, 768), (1, 768), 0), out=buf23)
        buf25 = empty((3, 5, 768), device='cuda', dtype=torch.bool)
        buf26 = reinterpret_tensor(buf23, (3, 5, 768), (3840, 768, 1), 0); del buf23  # reuse
        buf30 = empty((3, 5, 768), device='cuda', dtype=torch.float32)
        buf31 = empty((15, 768), device='cuda', dtype=torch.float32)
        buf453 = empty((3, 5, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_2, attention_output, embedding_output, embeddings_2, hidden_states_1, hidden_states_3], Original ATen: [aten.add, aten.native_dropout, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_6.run(buf26, buf5, primals_13, buf7, buf4, primals_4, primals_5, primals_14, primals_15, buf25, buf30, buf31, buf453, 2, 15, 768, grid=grid(15), stream=stream0)
        del primals_13
        del primals_5
        buf32 = empty((15, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_17, buf31, reinterpret_tensor(primals_16, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf32)
        del primals_17
        buf33 = empty((15, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_5, intermediate_output], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_7.run(buf32, buf33, 46080, grid=grid(46080), stream=stream0)
        buf34 = reinterpret_tensor(buf26, (15, 768), (768, 1), 0); del buf26  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf33, reinterpret_tensor(primals_18, (3072, 768), (1, 3072), 0), out=buf34)
        buf36 = empty((3, 5, 768), device='cuda', dtype=torch.bool)
        buf37 = reinterpret_tensor(buf34, (3, 5, 768), (3840, 768, 1), 0); del buf34  # reuse
        buf41 = empty((3, 5, 768), device='cuda', dtype=torch.float32)
        buf42 = empty((15, 768), device='cuda', dtype=torch.float32)
        buf452 = empty((3, 5, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_3, attention_output, hidden_states_6, hidden_states_8, mixed_query_layer_1], Original ATen: [aten.add, aten.native_dropout, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_8.run(buf37, buf5, primals_19, buf30, primals_14, primals_15, primals_20, primals_21, buf36, buf41, buf42, buf452, 3, 15, 768, grid=grid(15), stream=stream0)
        del primals_15
        del primals_19
        buf43 = reinterpret_tensor(buf37, (15, 768), (768, 1), 0); del buf37  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf42, reinterpret_tensor(primals_22, (768, 768), (1, 768), 0), out=buf43)
        buf44 = empty((15, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf42, reinterpret_tensor(primals_24, (768, 768), (1, 768), 0), out=buf44)
        buf45 = empty((15, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf42, reinterpret_tensor(primals_26, (768, 768), (1, 768), 0), out=buf45)
        buf46 = empty((3, 12, 5, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf43, primals_23, buf46, 11520, grid=grid(11520), stream=stream0)
        del primals_23
        buf47 = reinterpret_tensor(buf43, (3, 12, 64, 5), (3840, 320, 5, 1), 0); del buf43  # reuse
        # Source Nodes: [attention_scores_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf44, primals_25, buf47, 2304, 5, grid=grid(2304, 5), stream=stream0)
        del primals_25
        buf48 = empty((36, 5, 5), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf46, (36, 5, 64), (320, 64, 1), 0), reinterpret_tensor(buf47, (36, 64, 5), (320, 5, 1), 0), out=buf48)
        buf49 = empty_strided((3, 12, 5, 1), (60, 5, 1, 180), device='cuda', dtype=torch.float32)
        buf50 = empty_strided((3, 12, 5, 1), (60, 5, 1, 180), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_2, attention_scores_4, attention_scores_5, extended_attention_mask_1, extended_attention_mask_3, sub], Original ATen: [aten._softmax, aten._to_copy, aten.add, aten.div, aten.mul, aten.rsub]
        triton_poi_fused__softmax__to_copy_add_div_mul_rsub_3.run(buf48, primals_203, buf49, buf50, 180, grid=grid(180), stream=stream0)
        buf52 = empty((3, 12, 5, 5), device='cuda', dtype=torch.bool)
        buf53 = empty((3, 12, 5, 5), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_2, attention_probs_3, attention_scores_4, attention_scores_5, extended_attention_mask_1, extended_attention_mask_3, sub], Original ATen: [aten._softmax, aten._to_copy, aten.add, aten.div, aten.mul, aten.native_dropout, aten.rsub]
        triton_poi_fused__softmax__to_copy_add_div_mul_native_dropout_rsub_4.run(buf5, buf48, primals_203, buf49, buf50, buf52, buf53, 4, 900, grid=grid(900), stream=stream0)
        buf54 = reinterpret_tensor(buf44, (3, 12, 5, 64), (3840, 320, 64, 1), 0); del buf44  # reuse
        # Source Nodes: [context_layer_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf45, primals_27, buf54, 11520, grid=grid(11520), stream=stream0)
        del primals_27
        buf55 = reinterpret_tensor(buf45, (36, 5, 64), (320, 64, 1), 0); del buf45  # reuse
        # Source Nodes: [context_layer_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf53, (36, 5, 5), (25, 5, 1), 0), reinterpret_tensor(buf54, (36, 5, 64), (320, 64, 1), 0), out=buf55)
        buf56 = empty((15, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_9], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf55, buf56, 11520, grid=grid(11520), stream=stream0)
        buf57 = reinterpret_tensor(buf55, (15, 768), (768, 1), 0); del buf55  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf56, reinterpret_tensor(primals_28, (768, 768), (1, 768), 0), out=buf57)
        buf59 = empty((3, 5, 768), device='cuda', dtype=torch.bool)
        buf60 = reinterpret_tensor(buf57, (3, 5, 768), (3840, 768, 1), 0); del buf57  # reuse
        buf64 = empty((3, 5, 768), device='cuda', dtype=torch.float32)
        buf65 = empty((15, 768), device='cuda', dtype=torch.float32)
        buf450 = empty((3, 5, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_5, attention_output_2, hidden_states_10, hidden_states_12, hidden_states_8], Original ATen: [aten.add, aten.native_dropout, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_8.run(buf60, buf5, primals_29, buf41, primals_20, primals_21, primals_30, primals_31, buf59, buf64, buf65, buf450, 5, 15, 768, grid=grid(15), stream=stream0)
        del primals_21
        del primals_29
        buf66 = empty((15, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_12], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_33, buf65, reinterpret_tensor(primals_32, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf66)
        del primals_33
        buf67 = empty((15, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_14, intermediate_output_1], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_7.run(buf66, buf67, 46080, grid=grid(46080), stream=stream0)
        buf68 = reinterpret_tensor(buf60, (15, 768), (768, 1), 0); del buf60  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf67, reinterpret_tensor(primals_34, (3072, 768), (1, 3072), 0), out=buf68)
        buf70 = empty((3, 5, 768), device='cuda', dtype=torch.bool)
        buf71 = reinterpret_tensor(buf68, (3, 5, 768), (3840, 768, 1), 0); del buf68  # reuse
        buf75 = empty((3, 5, 768), device='cuda', dtype=torch.float32)
        buf76 = empty((15, 768), device='cuda', dtype=torch.float32)
        buf449 = empty((3, 5, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_6, attention_output_2, hidden_states_15, hidden_states_17, mixed_query_layer_2], Original ATen: [aten.add, aten.native_dropout, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_8.run(buf71, buf5, primals_35, buf64, primals_30, primals_31, primals_36, primals_37, buf70, buf75, buf76, buf449, 6, 15, 768, grid=grid(15), stream=stream0)
        del primals_31
        del primals_35
        buf77 = reinterpret_tensor(buf71, (15, 768), (768, 1), 0); del buf71  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf76, reinterpret_tensor(primals_38, (768, 768), (1, 768), 0), out=buf77)
        buf78 = empty((15, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf76, reinterpret_tensor(primals_40, (768, 768), (1, 768), 0), out=buf78)
        buf79 = empty((15, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf76, reinterpret_tensor(primals_42, (768, 768), (1, 768), 0), out=buf79)
        buf80 = empty((3, 12, 5, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf77, primals_39, buf80, 11520, grid=grid(11520), stream=stream0)
        del primals_39
        buf81 = reinterpret_tensor(buf77, (3, 12, 64, 5), (3840, 320, 5, 1), 0); del buf77  # reuse
        # Source Nodes: [attention_scores_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf78, primals_41, buf81, 2304, 5, grid=grid(2304, 5), stream=stream0)
        del primals_41
        buf82 = empty((36, 5, 5), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf80, (36, 5, 64), (320, 64, 1), 0), reinterpret_tensor(buf81, (36, 64, 5), (320, 5, 1), 0), out=buf82)
        buf83 = empty_strided((3, 12, 5, 1), (60, 5, 1, 180), device='cuda', dtype=torch.float32)
        buf84 = empty_strided((3, 12, 5, 1), (60, 5, 1, 180), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_4, attention_scores_7, attention_scores_8, extended_attention_mask_1, extended_attention_mask_3, sub], Original ATen: [aten._softmax, aten._to_copy, aten.add, aten.div, aten.mul, aten.rsub]
        triton_poi_fused__softmax__to_copy_add_div_mul_rsub_3.run(buf82, primals_203, buf83, buf84, 180, grid=grid(180), stream=stream0)
        buf86 = empty((3, 12, 5, 5), device='cuda', dtype=torch.bool)
        buf87 = empty((3, 12, 5, 5), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_4, attention_probs_5, attention_scores_7, attention_scores_8, extended_attention_mask_1, extended_attention_mask_3, sub], Original ATen: [aten._softmax, aten._to_copy, aten.add, aten.div, aten.mul, aten.native_dropout, aten.rsub]
        triton_poi_fused__softmax__to_copy_add_div_mul_native_dropout_rsub_4.run(buf5, buf82, primals_203, buf83, buf84, buf86, buf87, 7, 900, grid=grid(900), stream=stream0)
        buf88 = reinterpret_tensor(buf78, (3, 12, 5, 64), (3840, 320, 64, 1), 0); del buf78  # reuse
        # Source Nodes: [context_layer_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf79, primals_43, buf88, 11520, grid=grid(11520), stream=stream0)
        del primals_43
        buf89 = reinterpret_tensor(buf79, (36, 5, 64), (320, 64, 1), 0); del buf79  # reuse
        # Source Nodes: [context_layer_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf87, (36, 5, 5), (25, 5, 1), 0), reinterpret_tensor(buf88, (36, 5, 64), (320, 64, 1), 0), out=buf89)
        buf90 = empty((15, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_18], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf89, buf90, 11520, grid=grid(11520), stream=stream0)
        buf91 = reinterpret_tensor(buf89, (15, 768), (768, 1), 0); del buf89  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf90, reinterpret_tensor(primals_44, (768, 768), (1, 768), 0), out=buf91)
        buf93 = empty((3, 5, 768), device='cuda', dtype=torch.bool)
        buf94 = reinterpret_tensor(buf91, (3, 5, 768), (3840, 768, 1), 0); del buf91  # reuse
        buf98 = empty((3, 5, 768), device='cuda', dtype=torch.float32)
        buf99 = empty((15, 768), device='cuda', dtype=torch.float32)
        buf447 = empty((3, 5, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_8, attention_output_4, hidden_states_17, hidden_states_19, hidden_states_21], Original ATen: [aten.add, aten.native_dropout, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_8.run(buf94, buf5, primals_45, buf75, primals_36, primals_37, primals_46, primals_47, buf93, buf98, buf99, buf447, 8, 15, 768, grid=grid(15), stream=stream0)
        del primals_37
        del primals_45
        buf100 = empty((15, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_21], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_49, buf99, reinterpret_tensor(primals_48, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf100)
        del primals_49
        buf101 = empty((15, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_23, intermediate_output_2], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_7.run(buf100, buf101, 46080, grid=grid(46080), stream=stream0)
        buf102 = reinterpret_tensor(buf94, (15, 768), (768, 1), 0); del buf94  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf101, reinterpret_tensor(primals_50, (3072, 768), (1, 3072), 0), out=buf102)
        buf104 = empty((3, 5, 768), device='cuda', dtype=torch.bool)
        buf105 = reinterpret_tensor(buf102, (3, 5, 768), (3840, 768, 1), 0); del buf102  # reuse
        buf109 = empty((3, 5, 768), device='cuda', dtype=torch.float32)
        buf110 = empty((15, 768), device='cuda', dtype=torch.float32)
        buf446 = empty((3, 5, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_9, attention_output_4, hidden_states_24, hidden_states_26, mixed_query_layer_3], Original ATen: [aten.add, aten.native_dropout, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_8.run(buf105, buf5, primals_51, buf98, primals_46, primals_47, primals_52, primals_53, buf104, buf109, buf110, buf446, 9, 15, 768, grid=grid(15), stream=stream0)
        del primals_47
        del primals_51
        buf111 = reinterpret_tensor(buf105, (15, 768), (768, 1), 0); del buf105  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf110, reinterpret_tensor(primals_54, (768, 768), (1, 768), 0), out=buf111)
        buf112 = empty((15, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf110, reinterpret_tensor(primals_56, (768, 768), (1, 768), 0), out=buf112)
        buf113 = empty((15, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf110, reinterpret_tensor(primals_58, (768, 768), (1, 768), 0), out=buf113)
        buf114 = empty((3, 12, 5, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf111, primals_55, buf114, 11520, grid=grid(11520), stream=stream0)
        del primals_55
        buf115 = reinterpret_tensor(buf111, (3, 12, 64, 5), (3840, 320, 5, 1), 0); del buf111  # reuse
        # Source Nodes: [attention_scores_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf112, primals_57, buf115, 2304, 5, grid=grid(2304, 5), stream=stream0)
        del primals_57
        buf116 = empty((36, 5, 5), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf114, (36, 5, 64), (320, 64, 1), 0), reinterpret_tensor(buf115, (36, 64, 5), (320, 5, 1), 0), out=buf116)
        buf117 = empty_strided((3, 12, 5, 1), (60, 5, 1, 180), device='cuda', dtype=torch.float32)
        buf118 = empty_strided((3, 12, 5, 1), (60, 5, 1, 180), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_6, attention_scores_10, attention_scores_11, extended_attention_mask_1, extended_attention_mask_3, sub], Original ATen: [aten._softmax, aten._to_copy, aten.add, aten.div, aten.mul, aten.rsub]
        triton_poi_fused__softmax__to_copy_add_div_mul_rsub_3.run(buf116, primals_203, buf117, buf118, 180, grid=grid(180), stream=stream0)
        buf120 = empty((3, 12, 5, 5), device='cuda', dtype=torch.bool)
        buf121 = empty((3, 12, 5, 5), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_6, attention_probs_7, attention_scores_10, attention_scores_11, extended_attention_mask_1, extended_attention_mask_3, sub], Original ATen: [aten._softmax, aten._to_copy, aten.add, aten.div, aten.mul, aten.native_dropout, aten.rsub]
        triton_poi_fused__softmax__to_copy_add_div_mul_native_dropout_rsub_4.run(buf5, buf116, primals_203, buf117, buf118, buf120, buf121, 10, 900, grid=grid(900), stream=stream0)
        buf122 = reinterpret_tensor(buf112, (3, 12, 5, 64), (3840, 320, 64, 1), 0); del buf112  # reuse
        # Source Nodes: [context_layer_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf113, primals_59, buf122, 11520, grid=grid(11520), stream=stream0)
        del primals_59
        buf123 = reinterpret_tensor(buf113, (36, 5, 64), (320, 64, 1), 0); del buf113  # reuse
        # Source Nodes: [context_layer_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf121, (36, 5, 5), (25, 5, 1), 0), reinterpret_tensor(buf122, (36, 5, 64), (320, 64, 1), 0), out=buf123)
        buf124 = empty((15, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_27], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf123, buf124, 11520, grid=grid(11520), stream=stream0)
        buf125 = reinterpret_tensor(buf123, (15, 768), (768, 1), 0); del buf123  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf124, reinterpret_tensor(primals_60, (768, 768), (1, 768), 0), out=buf125)
        buf127 = empty((3, 5, 768), device='cuda', dtype=torch.bool)
        buf128 = reinterpret_tensor(buf125, (3, 5, 768), (3840, 768, 1), 0); del buf125  # reuse
        buf132 = empty((3, 5, 768), device='cuda', dtype=torch.float32)
        buf133 = empty((15, 768), device='cuda', dtype=torch.float32)
        buf444 = empty((3, 5, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_11, attention_output_6, hidden_states_26, hidden_states_28, hidden_states_30], Original ATen: [aten.add, aten.native_dropout, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_8.run(buf128, buf5, primals_61, buf109, primals_52, primals_53, primals_62, primals_63, buf127, buf132, buf133, buf444, 11, 15, 768, grid=grid(15), stream=stream0)
        del primals_53
        del primals_61
        buf134 = empty((15, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_30], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_65, buf133, reinterpret_tensor(primals_64, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf134)
        del primals_65
        buf135 = empty((15, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_32, intermediate_output_3], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_7.run(buf134, buf135, 46080, grid=grid(46080), stream=stream0)
        buf136 = reinterpret_tensor(buf128, (15, 768), (768, 1), 0); del buf128  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf135, reinterpret_tensor(primals_66, (3072, 768), (1, 3072), 0), out=buf136)
        buf138 = empty((3, 5, 768), device='cuda', dtype=torch.bool)
        buf139 = reinterpret_tensor(buf136, (3, 5, 768), (3840, 768, 1), 0); del buf136  # reuse
        buf143 = empty((3, 5, 768), device='cuda', dtype=torch.float32)
        buf144 = empty((15, 768), device='cuda', dtype=torch.float32)
        buf443 = empty((3, 5, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_12, attention_output_6, hidden_states_33, hidden_states_35, mixed_query_layer_4], Original ATen: [aten.add, aten.native_dropout, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_8.run(buf139, buf5, primals_67, buf132, primals_62, primals_63, primals_68, primals_69, buf138, buf143, buf144, buf443, 12, 15, 768, grid=grid(15), stream=stream0)
        del primals_63
        del primals_67
        buf145 = reinterpret_tensor(buf139, (15, 768), (768, 1), 0); del buf139  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf144, reinterpret_tensor(primals_70, (768, 768), (1, 768), 0), out=buf145)
        buf146 = empty((15, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf144, reinterpret_tensor(primals_72, (768, 768), (1, 768), 0), out=buf146)
        buf147 = empty((15, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf144, reinterpret_tensor(primals_74, (768, 768), (1, 768), 0), out=buf147)
        buf148 = empty((3, 12, 5, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf145, primals_71, buf148, 11520, grid=grid(11520), stream=stream0)
        del primals_71
        buf149 = reinterpret_tensor(buf145, (3, 12, 64, 5), (3840, 320, 5, 1), 0); del buf145  # reuse
        # Source Nodes: [attention_scores_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf146, primals_73, buf149, 2304, 5, grid=grid(2304, 5), stream=stream0)
        del primals_73
        buf150 = empty((36, 5, 5), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf148, (36, 5, 64), (320, 64, 1), 0), reinterpret_tensor(buf149, (36, 64, 5), (320, 5, 1), 0), out=buf150)
        buf151 = empty_strided((3, 12, 5, 1), (60, 5, 1, 180), device='cuda', dtype=torch.float32)
        buf152 = empty_strided((3, 12, 5, 1), (60, 5, 1, 180), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_8, attention_scores_13, attention_scores_14, extended_attention_mask_1, extended_attention_mask_3, sub], Original ATen: [aten._softmax, aten._to_copy, aten.add, aten.div, aten.mul, aten.rsub]
        triton_poi_fused__softmax__to_copy_add_div_mul_rsub_3.run(buf150, primals_203, buf151, buf152, 180, grid=grid(180), stream=stream0)
        buf154 = empty((3, 12, 5, 5), device='cuda', dtype=torch.bool)
        buf155 = empty((3, 12, 5, 5), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_8, attention_probs_9, attention_scores_13, attention_scores_14, extended_attention_mask_1, extended_attention_mask_3, sub], Original ATen: [aten._softmax, aten._to_copy, aten.add, aten.div, aten.mul, aten.native_dropout, aten.rsub]
        triton_poi_fused__softmax__to_copy_add_div_mul_native_dropout_rsub_4.run(buf5, buf150, primals_203, buf151, buf152, buf154, buf155, 13, 900, grid=grid(900), stream=stream0)
        buf156 = reinterpret_tensor(buf146, (3, 12, 5, 64), (3840, 320, 64, 1), 0); del buf146  # reuse
        # Source Nodes: [context_layer_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf147, primals_75, buf156, 11520, grid=grid(11520), stream=stream0)
        del primals_75
        buf157 = reinterpret_tensor(buf147, (36, 5, 64), (320, 64, 1), 0); del buf147  # reuse
        # Source Nodes: [context_layer_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf155, (36, 5, 5), (25, 5, 1), 0), reinterpret_tensor(buf156, (36, 5, 64), (320, 64, 1), 0), out=buf157)
        buf158 = empty((15, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_36], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf157, buf158, 11520, grid=grid(11520), stream=stream0)
        buf159 = reinterpret_tensor(buf157, (15, 768), (768, 1), 0); del buf157  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf158, reinterpret_tensor(primals_76, (768, 768), (1, 768), 0), out=buf159)
        buf161 = empty((3, 5, 768), device='cuda', dtype=torch.bool)
        buf162 = reinterpret_tensor(buf159, (3, 5, 768), (3840, 768, 1), 0); del buf159  # reuse
        buf166 = empty((3, 5, 768), device='cuda', dtype=torch.float32)
        buf167 = empty((15, 768), device='cuda', dtype=torch.float32)
        buf441 = empty((3, 5, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_14, attention_output_8, hidden_states_35, hidden_states_37, hidden_states_39], Original ATen: [aten.add, aten.native_dropout, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_8.run(buf162, buf5, primals_77, buf143, primals_68, primals_69, primals_78, primals_79, buf161, buf166, buf167, buf441, 14, 15, 768, grid=grid(15), stream=stream0)
        del primals_69
        del primals_77
        buf168 = empty((15, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_39], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_81, buf167, reinterpret_tensor(primals_80, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf168)
        del primals_81
        buf169 = empty((15, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_41, intermediate_output_4], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_7.run(buf168, buf169, 46080, grid=grid(46080), stream=stream0)
        buf170 = reinterpret_tensor(buf162, (15, 768), (768, 1), 0); del buf162  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf169, reinterpret_tensor(primals_82, (3072, 768), (1, 3072), 0), out=buf170)
        buf172 = empty((3, 5, 768), device='cuda', dtype=torch.bool)
        buf173 = reinterpret_tensor(buf170, (3, 5, 768), (3840, 768, 1), 0); del buf170  # reuse
        buf177 = empty((3, 5, 768), device='cuda', dtype=torch.float32)
        buf178 = empty((15, 768), device='cuda', dtype=torch.float32)
        buf440 = empty((3, 5, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_15, attention_output_8, hidden_states_42, hidden_states_44, mixed_query_layer_5], Original ATen: [aten.add, aten.native_dropout, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_8.run(buf173, buf5, primals_83, buf166, primals_78, primals_79, primals_84, primals_85, buf172, buf177, buf178, buf440, 15, 15, 768, grid=grid(15), stream=stream0)
        del primals_79
        del primals_83
        buf179 = reinterpret_tensor(buf173, (15, 768), (768, 1), 0); del buf173  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf178, reinterpret_tensor(primals_86, (768, 768), (1, 768), 0), out=buf179)
        buf180 = empty((15, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf178, reinterpret_tensor(primals_88, (768, 768), (1, 768), 0), out=buf180)
        buf181 = empty((15, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf178, reinterpret_tensor(primals_90, (768, 768), (1, 768), 0), out=buf181)
        buf182 = empty((3, 12, 5, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores_15], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf179, primals_87, buf182, 11520, grid=grid(11520), stream=stream0)
        del primals_87
        buf183 = reinterpret_tensor(buf179, (3, 12, 64, 5), (3840, 320, 5, 1), 0); del buf179  # reuse
        # Source Nodes: [attention_scores_15], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf180, primals_89, buf183, 2304, 5, grid=grid(2304, 5), stream=stream0)
        del primals_89
        buf184 = empty((36, 5, 5), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf182, (36, 5, 64), (320, 64, 1), 0), reinterpret_tensor(buf183, (36, 64, 5), (320, 5, 1), 0), out=buf184)
        buf185 = empty_strided((3, 12, 5, 1), (60, 5, 1, 180), device='cuda', dtype=torch.float32)
        buf186 = empty_strided((3, 12, 5, 1), (60, 5, 1, 180), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_10, attention_scores_16, attention_scores_17, extended_attention_mask_1, extended_attention_mask_3, sub], Original ATen: [aten._softmax, aten._to_copy, aten.add, aten.div, aten.mul, aten.rsub]
        triton_poi_fused__softmax__to_copy_add_div_mul_rsub_3.run(buf184, primals_203, buf185, buf186, 180, grid=grid(180), stream=stream0)
        buf188 = empty((3, 12, 5, 5), device='cuda', dtype=torch.bool)
        buf189 = empty((3, 12, 5, 5), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_10, attention_probs_11, attention_scores_16, attention_scores_17, extended_attention_mask_1, extended_attention_mask_3, sub], Original ATen: [aten._softmax, aten._to_copy, aten.add, aten.div, aten.mul, aten.native_dropout, aten.rsub]
        triton_poi_fused__softmax__to_copy_add_div_mul_native_dropout_rsub_4.run(buf5, buf184, primals_203, buf185, buf186, buf188, buf189, 16, 900, grid=grid(900), stream=stream0)
        buf190 = reinterpret_tensor(buf180, (3, 12, 5, 64), (3840, 320, 64, 1), 0); del buf180  # reuse
        # Source Nodes: [context_layer_15], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf181, primals_91, buf190, 11520, grid=grid(11520), stream=stream0)
        del primals_91
        buf191 = reinterpret_tensor(buf181, (36, 5, 64), (320, 64, 1), 0); del buf181  # reuse
        # Source Nodes: [context_layer_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf189, (36, 5, 5), (25, 5, 1), 0), reinterpret_tensor(buf190, (36, 5, 64), (320, 64, 1), 0), out=buf191)
        buf192 = empty((15, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_45], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf191, buf192, 11520, grid=grid(11520), stream=stream0)
        buf193 = reinterpret_tensor(buf191, (15, 768), (768, 1), 0); del buf191  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf192, reinterpret_tensor(primals_92, (768, 768), (1, 768), 0), out=buf193)
        buf195 = empty((3, 5, 768), device='cuda', dtype=torch.bool)
        buf196 = reinterpret_tensor(buf193, (3, 5, 768), (3840, 768, 1), 0); del buf193  # reuse
        buf200 = empty((3, 5, 768), device='cuda', dtype=torch.float32)
        buf201 = empty((15, 768), device='cuda', dtype=torch.float32)
        buf438 = empty((3, 5, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_17, attention_output_10, hidden_states_44, hidden_states_46, hidden_states_48], Original ATen: [aten.add, aten.native_dropout, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_8.run(buf196, buf5, primals_93, buf177, primals_84, primals_85, primals_94, primals_95, buf195, buf200, buf201, buf438, 17, 15, 768, grid=grid(15), stream=stream0)
        del primals_85
        del primals_93
        buf202 = empty((15, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_48], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_97, buf201, reinterpret_tensor(primals_96, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf202)
        del primals_97
        buf203 = empty((15, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_50, intermediate_output_5], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_7.run(buf202, buf203, 46080, grid=grid(46080), stream=stream0)
        buf204 = reinterpret_tensor(buf196, (15, 768), (768, 1), 0); del buf196  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf203, reinterpret_tensor(primals_98, (3072, 768), (1, 3072), 0), out=buf204)
        buf206 = empty((3, 5, 768), device='cuda', dtype=torch.bool)
        buf207 = reinterpret_tensor(buf204, (3, 5, 768), (3840, 768, 1), 0); del buf204  # reuse
        buf211 = empty((3, 5, 768), device='cuda', dtype=torch.float32)
        buf212 = empty((15, 768), device='cuda', dtype=torch.float32)
        buf437 = empty((3, 5, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_18, attention_output_10, hidden_states_51, hidden_states_53, mixed_query_layer_6], Original ATen: [aten.add, aten.native_dropout, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_8.run(buf207, buf5, primals_99, buf200, primals_94, primals_95, primals_100, primals_101, buf206, buf211, buf212, buf437, 18, 15, 768, grid=grid(15), stream=stream0)
        del primals_95
        del primals_99
        buf213 = reinterpret_tensor(buf207, (15, 768), (768, 1), 0); del buf207  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf212, reinterpret_tensor(primals_102, (768, 768), (1, 768), 0), out=buf213)
        buf214 = empty((15, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf212, reinterpret_tensor(primals_104, (768, 768), (1, 768), 0), out=buf214)
        buf215 = empty((15, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf212, reinterpret_tensor(primals_106, (768, 768), (1, 768), 0), out=buf215)
        buf216 = empty((3, 12, 5, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf213, primals_103, buf216, 11520, grid=grid(11520), stream=stream0)
        del primals_103
        buf217 = reinterpret_tensor(buf213, (3, 12, 64, 5), (3840, 320, 5, 1), 0); del buf213  # reuse
        # Source Nodes: [attention_scores_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf214, primals_105, buf217, 2304, 5, grid=grid(2304, 5), stream=stream0)
        del primals_105
        buf218 = empty((36, 5, 5), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf216, (36, 5, 64), (320, 64, 1), 0), reinterpret_tensor(buf217, (36, 64, 5), (320, 5, 1), 0), out=buf218)
        buf219 = empty_strided((3, 12, 5, 1), (60, 5, 1, 180), device='cuda', dtype=torch.float32)
        buf220 = empty_strided((3, 12, 5, 1), (60, 5, 1, 180), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_12, attention_scores_19, attention_scores_20, extended_attention_mask_1, extended_attention_mask_3, sub], Original ATen: [aten._softmax, aten._to_copy, aten.add, aten.div, aten.mul, aten.rsub]
        triton_poi_fused__softmax__to_copy_add_div_mul_rsub_3.run(buf218, primals_203, buf219, buf220, 180, grid=grid(180), stream=stream0)
        buf222 = empty((3, 12, 5, 5), device='cuda', dtype=torch.bool)
        buf223 = empty((3, 12, 5, 5), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_12, attention_probs_13, attention_scores_19, attention_scores_20, extended_attention_mask_1, extended_attention_mask_3, sub], Original ATen: [aten._softmax, aten._to_copy, aten.add, aten.div, aten.mul, aten.native_dropout, aten.rsub]
        triton_poi_fused__softmax__to_copy_add_div_mul_native_dropout_rsub_4.run(buf5, buf218, primals_203, buf219, buf220, buf222, buf223, 19, 900, grid=grid(900), stream=stream0)
        buf224 = reinterpret_tensor(buf214, (3, 12, 5, 64), (3840, 320, 64, 1), 0); del buf214  # reuse
        # Source Nodes: [context_layer_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf215, primals_107, buf224, 11520, grid=grid(11520), stream=stream0)
        del primals_107
        buf225 = reinterpret_tensor(buf215, (36, 5, 64), (320, 64, 1), 0); del buf215  # reuse
        # Source Nodes: [context_layer_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf223, (36, 5, 5), (25, 5, 1), 0), reinterpret_tensor(buf224, (36, 5, 64), (320, 64, 1), 0), out=buf225)
        buf226 = empty((15, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_54], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf225, buf226, 11520, grid=grid(11520), stream=stream0)
        buf227 = reinterpret_tensor(buf225, (15, 768), (768, 1), 0); del buf225  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf226, reinterpret_tensor(primals_108, (768, 768), (1, 768), 0), out=buf227)
        buf229 = empty((3, 5, 768), device='cuda', dtype=torch.bool)
        buf230 = reinterpret_tensor(buf227, (3, 5, 768), (3840, 768, 1), 0); del buf227  # reuse
        buf234 = empty((3, 5, 768), device='cuda', dtype=torch.float32)
        buf235 = empty((15, 768), device='cuda', dtype=torch.float32)
        buf435 = empty((3, 5, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_20, attention_output_12, hidden_states_53, hidden_states_55, hidden_states_57], Original ATen: [aten.add, aten.native_dropout, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_8.run(buf230, buf5, primals_109, buf211, primals_100, primals_101, primals_110, primals_111, buf229, buf234, buf235, buf435, 20, 15, 768, grid=grid(15), stream=stream0)
        del primals_101
        del primals_109
        buf236 = empty((15, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_57], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_113, buf235, reinterpret_tensor(primals_112, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf236)
        del primals_113
        buf237 = empty((15, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_59, intermediate_output_6], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_7.run(buf236, buf237, 46080, grid=grid(46080), stream=stream0)
        buf238 = reinterpret_tensor(buf230, (15, 768), (768, 1), 0); del buf230  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf237, reinterpret_tensor(primals_114, (3072, 768), (1, 3072), 0), out=buf238)
        buf240 = empty((3, 5, 768), device='cuda', dtype=torch.bool)
        buf241 = reinterpret_tensor(buf238, (3, 5, 768), (3840, 768, 1), 0); del buf238  # reuse
        buf245 = empty((3, 5, 768), device='cuda', dtype=torch.float32)
        buf246 = empty((15, 768), device='cuda', dtype=torch.float32)
        buf434 = empty((3, 5, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_21, attention_output_12, hidden_states_60, hidden_states_62, mixed_query_layer_7], Original ATen: [aten.add, aten.native_dropout, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_8.run(buf241, buf5, primals_115, buf234, primals_110, primals_111, primals_116, primals_117, buf240, buf245, buf246, buf434, 21, 15, 768, grid=grid(15), stream=stream0)
        del primals_111
        del primals_115
        buf247 = reinterpret_tensor(buf241, (15, 768), (768, 1), 0); del buf241  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf246, reinterpret_tensor(primals_118, (768, 768), (1, 768), 0), out=buf247)
        buf248 = empty((15, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf246, reinterpret_tensor(primals_120, (768, 768), (1, 768), 0), out=buf248)
        buf249 = empty((15, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf246, reinterpret_tensor(primals_122, (768, 768), (1, 768), 0), out=buf249)
        buf250 = empty((3, 12, 5, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores_21], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf247, primals_119, buf250, 11520, grid=grid(11520), stream=stream0)
        del primals_119
        buf251 = reinterpret_tensor(buf247, (3, 12, 64, 5), (3840, 320, 5, 1), 0); del buf247  # reuse
        # Source Nodes: [attention_scores_21], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf248, primals_121, buf251, 2304, 5, grid=grid(2304, 5), stream=stream0)
        del primals_121
        buf252 = empty((36, 5, 5), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf250, (36, 5, 64), (320, 64, 1), 0), reinterpret_tensor(buf251, (36, 64, 5), (320, 5, 1), 0), out=buf252)
        buf253 = empty_strided((3, 12, 5, 1), (60, 5, 1, 180), device='cuda', dtype=torch.float32)
        buf254 = empty_strided((3, 12, 5, 1), (60, 5, 1, 180), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_14, attention_scores_22, attention_scores_23, extended_attention_mask_1, extended_attention_mask_3, sub], Original ATen: [aten._softmax, aten._to_copy, aten.add, aten.div, aten.mul, aten.rsub]
        triton_poi_fused__softmax__to_copy_add_div_mul_rsub_3.run(buf252, primals_203, buf253, buf254, 180, grid=grid(180), stream=stream0)
        buf256 = empty((3, 12, 5, 5), device='cuda', dtype=torch.bool)
        buf257 = empty((3, 12, 5, 5), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_14, attention_probs_15, attention_scores_22, attention_scores_23, extended_attention_mask_1, extended_attention_mask_3, sub], Original ATen: [aten._softmax, aten._to_copy, aten.add, aten.div, aten.mul, aten.native_dropout, aten.rsub]
        triton_poi_fused__softmax__to_copy_add_div_mul_native_dropout_rsub_4.run(buf5, buf252, primals_203, buf253, buf254, buf256, buf257, 22, 900, grid=grid(900), stream=stream0)
        buf258 = reinterpret_tensor(buf248, (3, 12, 5, 64), (3840, 320, 64, 1), 0); del buf248  # reuse
        # Source Nodes: [context_layer_21], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf249, primals_123, buf258, 11520, grid=grid(11520), stream=stream0)
        del primals_123
        buf259 = reinterpret_tensor(buf249, (36, 5, 64), (320, 64, 1), 0); del buf249  # reuse
        # Source Nodes: [context_layer_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf257, (36, 5, 5), (25, 5, 1), 0), reinterpret_tensor(buf258, (36, 5, 64), (320, 64, 1), 0), out=buf259)
        buf260 = empty((15, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_63], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf259, buf260, 11520, grid=grid(11520), stream=stream0)
        buf261 = reinterpret_tensor(buf259, (15, 768), (768, 1), 0); del buf259  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf260, reinterpret_tensor(primals_124, (768, 768), (1, 768), 0), out=buf261)
        buf263 = empty((3, 5, 768), device='cuda', dtype=torch.bool)
        buf264 = reinterpret_tensor(buf261, (3, 5, 768), (3840, 768, 1), 0); del buf261  # reuse
        buf268 = empty((3, 5, 768), device='cuda', dtype=torch.float32)
        buf269 = empty((15, 768), device='cuda', dtype=torch.float32)
        buf432 = empty((3, 5, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_23, attention_output_14, hidden_states_62, hidden_states_64, hidden_states_66], Original ATen: [aten.add, aten.native_dropout, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_8.run(buf264, buf5, primals_125, buf245, primals_116, primals_117, primals_126, primals_127, buf263, buf268, buf269, buf432, 23, 15, 768, grid=grid(15), stream=stream0)
        del primals_117
        del primals_125
        buf270 = empty((15, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_66], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_129, buf269, reinterpret_tensor(primals_128, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf270)
        del primals_129
        buf271 = empty((15, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_68, intermediate_output_7], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_7.run(buf270, buf271, 46080, grid=grid(46080), stream=stream0)
        buf272 = reinterpret_tensor(buf264, (15, 768), (768, 1), 0); del buf264  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf271, reinterpret_tensor(primals_130, (3072, 768), (1, 3072), 0), out=buf272)
        buf274 = empty((3, 5, 768), device='cuda', dtype=torch.bool)
        buf275 = reinterpret_tensor(buf272, (3, 5, 768), (3840, 768, 1), 0); del buf272  # reuse
        buf279 = empty((3, 5, 768), device='cuda', dtype=torch.float32)
        buf280 = empty((15, 768), device='cuda', dtype=torch.float32)
        buf431 = empty((3, 5, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_24, attention_output_14, hidden_states_69, hidden_states_71, mixed_query_layer_8], Original ATen: [aten.add, aten.native_dropout, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_8.run(buf275, buf5, primals_131, buf268, primals_126, primals_127, primals_132, primals_133, buf274, buf279, buf280, buf431, 24, 15, 768, grid=grid(15), stream=stream0)
        del primals_127
        del primals_131
        buf281 = reinterpret_tensor(buf275, (15, 768), (768, 1), 0); del buf275  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf280, reinterpret_tensor(primals_134, (768, 768), (1, 768), 0), out=buf281)
        buf282 = empty((15, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf280, reinterpret_tensor(primals_136, (768, 768), (1, 768), 0), out=buf282)
        buf283 = empty((15, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf280, reinterpret_tensor(primals_138, (768, 768), (1, 768), 0), out=buf283)
        buf284 = empty((3, 12, 5, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf281, primals_135, buf284, 11520, grid=grid(11520), stream=stream0)
        del primals_135
        buf285 = reinterpret_tensor(buf281, (3, 12, 64, 5), (3840, 320, 5, 1), 0); del buf281  # reuse
        # Source Nodes: [attention_scores_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf282, primals_137, buf285, 2304, 5, grid=grid(2304, 5), stream=stream0)
        del primals_137
        buf286 = empty((36, 5, 5), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf284, (36, 5, 64), (320, 64, 1), 0), reinterpret_tensor(buf285, (36, 64, 5), (320, 5, 1), 0), out=buf286)
        buf287 = empty_strided((3, 12, 5, 1), (60, 5, 1, 180), device='cuda', dtype=torch.float32)
        buf288 = empty_strided((3, 12, 5, 1), (60, 5, 1, 180), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_16, attention_scores_25, attention_scores_26, extended_attention_mask_1, extended_attention_mask_3, sub], Original ATen: [aten._softmax, aten._to_copy, aten.add, aten.div, aten.mul, aten.rsub]
        triton_poi_fused__softmax__to_copy_add_div_mul_rsub_3.run(buf286, primals_203, buf287, buf288, 180, grid=grid(180), stream=stream0)
        buf290 = empty((3, 12, 5, 5), device='cuda', dtype=torch.bool)
        buf291 = empty((3, 12, 5, 5), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_16, attention_probs_17, attention_scores_25, attention_scores_26, extended_attention_mask_1, extended_attention_mask_3, sub], Original ATen: [aten._softmax, aten._to_copy, aten.add, aten.div, aten.mul, aten.native_dropout, aten.rsub]
        triton_poi_fused__softmax__to_copy_add_div_mul_native_dropout_rsub_4.run(buf5, buf286, primals_203, buf287, buf288, buf290, buf291, 25, 900, grid=grid(900), stream=stream0)
        buf292 = reinterpret_tensor(buf282, (3, 12, 5, 64), (3840, 320, 64, 1), 0); del buf282  # reuse
        # Source Nodes: [context_layer_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf283, primals_139, buf292, 11520, grid=grid(11520), stream=stream0)
        del primals_139
        buf293 = reinterpret_tensor(buf283, (36, 5, 64), (320, 64, 1), 0); del buf283  # reuse
        # Source Nodes: [context_layer_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf291, (36, 5, 5), (25, 5, 1), 0), reinterpret_tensor(buf292, (36, 5, 64), (320, 64, 1), 0), out=buf293)
        buf294 = empty((15, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_72], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf293, buf294, 11520, grid=grid(11520), stream=stream0)
        buf295 = reinterpret_tensor(buf293, (15, 768), (768, 1), 0); del buf293  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf294, reinterpret_tensor(primals_140, (768, 768), (1, 768), 0), out=buf295)
        buf297 = empty((3, 5, 768), device='cuda', dtype=torch.bool)
        buf298 = reinterpret_tensor(buf295, (3, 5, 768), (3840, 768, 1), 0); del buf295  # reuse
        buf302 = empty((3, 5, 768), device='cuda', dtype=torch.float32)
        buf303 = empty((15, 768), device='cuda', dtype=torch.float32)
        buf429 = empty((3, 5, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_26, attention_output_16, hidden_states_71, hidden_states_73, hidden_states_75], Original ATen: [aten.add, aten.native_dropout, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_8.run(buf298, buf5, primals_141, buf279, primals_132, primals_133, primals_142, primals_143, buf297, buf302, buf303, buf429, 26, 15, 768, grid=grid(15), stream=stream0)
        del primals_133
        del primals_141
        buf304 = empty((15, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_75], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_145, buf303, reinterpret_tensor(primals_144, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf304)
        del primals_145
        buf305 = empty((15, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_77, intermediate_output_8], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_7.run(buf304, buf305, 46080, grid=grid(46080), stream=stream0)
        buf306 = reinterpret_tensor(buf298, (15, 768), (768, 1), 0); del buf298  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf305, reinterpret_tensor(primals_146, (3072, 768), (1, 3072), 0), out=buf306)
        buf308 = empty((3, 5, 768), device='cuda', dtype=torch.bool)
        buf309 = reinterpret_tensor(buf306, (3, 5, 768), (3840, 768, 1), 0); del buf306  # reuse
        buf313 = empty((3, 5, 768), device='cuda', dtype=torch.float32)
        buf314 = empty((15, 768), device='cuda', dtype=torch.float32)
        buf428 = empty((3, 5, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_27, attention_output_16, hidden_states_78, hidden_states_80, mixed_query_layer_9], Original ATen: [aten.add, aten.native_dropout, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_8.run(buf309, buf5, primals_147, buf302, primals_142, primals_143, primals_148, primals_149, buf308, buf313, buf314, buf428, 27, 15, 768, grid=grid(15), stream=stream0)
        del primals_143
        del primals_147
        buf315 = reinterpret_tensor(buf309, (15, 768), (768, 1), 0); del buf309  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf314, reinterpret_tensor(primals_150, (768, 768), (1, 768), 0), out=buf315)
        buf316 = empty((15, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf314, reinterpret_tensor(primals_152, (768, 768), (1, 768), 0), out=buf316)
        buf317 = empty((15, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf314, reinterpret_tensor(primals_154, (768, 768), (1, 768), 0), out=buf317)
        buf318 = empty((3, 12, 5, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores_27], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf315, primals_151, buf318, 11520, grid=grid(11520), stream=stream0)
        del primals_151
        buf319 = reinterpret_tensor(buf315, (3, 12, 64, 5), (3840, 320, 5, 1), 0); del buf315  # reuse
        # Source Nodes: [attention_scores_27], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf316, primals_153, buf319, 2304, 5, grid=grid(2304, 5), stream=stream0)
        del primals_153
        buf320 = empty((36, 5, 5), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores_27], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf318, (36, 5, 64), (320, 64, 1), 0), reinterpret_tensor(buf319, (36, 64, 5), (320, 5, 1), 0), out=buf320)
        buf321 = empty_strided((3, 12, 5, 1), (60, 5, 1, 180), device='cuda', dtype=torch.float32)
        buf322 = empty_strided((3, 12, 5, 1), (60, 5, 1, 180), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_18, attention_scores_28, attention_scores_29, extended_attention_mask_1, extended_attention_mask_3, sub], Original ATen: [aten._softmax, aten._to_copy, aten.add, aten.div, aten.mul, aten.rsub]
        triton_poi_fused__softmax__to_copy_add_div_mul_rsub_3.run(buf320, primals_203, buf321, buf322, 180, grid=grid(180), stream=stream0)
        buf324 = empty((3, 12, 5, 5), device='cuda', dtype=torch.bool)
        buf325 = empty((3, 12, 5, 5), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_18, attention_probs_19, attention_scores_28, attention_scores_29, extended_attention_mask_1, extended_attention_mask_3, sub], Original ATen: [aten._softmax, aten._to_copy, aten.add, aten.div, aten.mul, aten.native_dropout, aten.rsub]
        triton_poi_fused__softmax__to_copy_add_div_mul_native_dropout_rsub_4.run(buf5, buf320, primals_203, buf321, buf322, buf324, buf325, 28, 900, grid=grid(900), stream=stream0)
        buf326 = reinterpret_tensor(buf316, (3, 12, 5, 64), (3840, 320, 64, 1), 0); del buf316  # reuse
        # Source Nodes: [context_layer_27], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf317, primals_155, buf326, 11520, grid=grid(11520), stream=stream0)
        del primals_155
        buf327 = reinterpret_tensor(buf317, (36, 5, 64), (320, 64, 1), 0); del buf317  # reuse
        # Source Nodes: [context_layer_27], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf325, (36, 5, 5), (25, 5, 1), 0), reinterpret_tensor(buf326, (36, 5, 64), (320, 64, 1), 0), out=buf327)
        buf328 = empty((15, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_81], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf327, buf328, 11520, grid=grid(11520), stream=stream0)
        buf329 = reinterpret_tensor(buf327, (15, 768), (768, 1), 0); del buf327  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf328, reinterpret_tensor(primals_156, (768, 768), (1, 768), 0), out=buf329)
        buf331 = empty((3, 5, 768), device='cuda', dtype=torch.bool)
        buf332 = reinterpret_tensor(buf329, (3, 5, 768), (3840, 768, 1), 0); del buf329  # reuse
        buf336 = empty((3, 5, 768), device='cuda', dtype=torch.float32)
        buf337 = empty((15, 768), device='cuda', dtype=torch.float32)
        buf426 = empty((3, 5, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_29, attention_output_18, hidden_states_80, hidden_states_82, hidden_states_84], Original ATen: [aten.add, aten.native_dropout, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_8.run(buf332, buf5, primals_157, buf313, primals_148, primals_149, primals_158, primals_159, buf331, buf336, buf337, buf426, 29, 15, 768, grid=grid(15), stream=stream0)
        del primals_149
        del primals_157
        buf338 = empty((15, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_84], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_161, buf337, reinterpret_tensor(primals_160, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf338)
        del primals_161
        buf339 = empty((15, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_86, intermediate_output_9], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_7.run(buf338, buf339, 46080, grid=grid(46080), stream=stream0)
        buf340 = reinterpret_tensor(buf332, (15, 768), (768, 1), 0); del buf332  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf339, reinterpret_tensor(primals_162, (3072, 768), (1, 3072), 0), out=buf340)
        buf342 = empty((3, 5, 768), device='cuda', dtype=torch.bool)
        buf343 = reinterpret_tensor(buf340, (3, 5, 768), (3840, 768, 1), 0); del buf340  # reuse
        buf347 = empty((3, 5, 768), device='cuda', dtype=torch.float32)
        buf348 = empty((15, 768), device='cuda', dtype=torch.float32)
        buf425 = empty((3, 5, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_30, attention_output_18, hidden_states_87, hidden_states_89, mixed_query_layer_10], Original ATen: [aten.add, aten.native_dropout, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_8.run(buf343, buf5, primals_163, buf336, primals_158, primals_159, primals_164, primals_165, buf342, buf347, buf348, buf425, 30, 15, 768, grid=grid(15), stream=stream0)
        del primals_159
        del primals_163
        buf349 = reinterpret_tensor(buf343, (15, 768), (768, 1), 0); del buf343  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf348, reinterpret_tensor(primals_166, (768, 768), (1, 768), 0), out=buf349)
        buf350 = empty((15, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf348, reinterpret_tensor(primals_168, (768, 768), (1, 768), 0), out=buf350)
        buf351 = empty((15, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf348, reinterpret_tensor(primals_170, (768, 768), (1, 768), 0), out=buf351)
        buf352 = empty((3, 12, 5, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf349, primals_167, buf352, 11520, grid=grid(11520), stream=stream0)
        del primals_167
        buf353 = reinterpret_tensor(buf349, (3, 12, 64, 5), (3840, 320, 5, 1), 0); del buf349  # reuse
        # Source Nodes: [attention_scores_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf350, primals_169, buf353, 2304, 5, grid=grid(2304, 5), stream=stream0)
        del primals_169
        buf354 = empty((36, 5, 5), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf352, (36, 5, 64), (320, 64, 1), 0), reinterpret_tensor(buf353, (36, 64, 5), (320, 5, 1), 0), out=buf354)
        buf355 = empty_strided((3, 12, 5, 1), (60, 5, 1, 180), device='cuda', dtype=torch.float32)
        buf356 = empty_strided((3, 12, 5, 1), (60, 5, 1, 180), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_20, attention_scores_31, attention_scores_32, extended_attention_mask_1, extended_attention_mask_3, sub], Original ATen: [aten._softmax, aten._to_copy, aten.add, aten.div, aten.mul, aten.rsub]
        triton_poi_fused__softmax__to_copy_add_div_mul_rsub_3.run(buf354, primals_203, buf355, buf356, 180, grid=grid(180), stream=stream0)
        buf358 = empty((3, 12, 5, 5), device='cuda', dtype=torch.bool)
        buf359 = empty((3, 12, 5, 5), device='cuda', dtype=torch.float32)
        buf424 = empty((3, 12, 5, 5), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_20, attention_probs_21, attention_scores_31, attention_scores_32, extended_attention_mask_1, extended_attention_mask_3, sub], Original ATen: [aten._softmax, aten._to_copy, aten.add, aten.detach, aten.div, aten.mul, aten.native_dropout, aten.rsub]
        triton_poi_fused__softmax__to_copy_add_detach_div_mul_native_dropout_rsub_9.run(buf5, buf354, primals_203, buf355, buf356, buf358, buf359, buf424, 31, 900, grid=grid(900), stream=stream0)
        buf360 = reinterpret_tensor(buf350, (3, 12, 5, 64), (3840, 320, 64, 1), 0); del buf350  # reuse
        # Source Nodes: [context_layer_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf351, primals_171, buf360, 11520, grid=grid(11520), stream=stream0)
        del primals_171
        buf361 = reinterpret_tensor(buf351, (36, 5, 64), (320, 64, 1), 0); del buf351  # reuse
        # Source Nodes: [context_layer_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf359, (36, 5, 5), (25, 5, 1), 0), reinterpret_tensor(buf360, (36, 5, 64), (320, 64, 1), 0), out=buf361)
        buf362 = empty((15, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_90], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf361, buf362, 11520, grid=grid(11520), stream=stream0)
        buf363 = reinterpret_tensor(buf361, (15, 768), (768, 1), 0); del buf361  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf362, reinterpret_tensor(primals_172, (768, 768), (1, 768), 0), out=buf363)
        buf365 = empty((3, 5, 768), device='cuda', dtype=torch.bool)
        buf366 = reinterpret_tensor(buf363, (3, 5, 768), (3840, 768, 1), 0); del buf363  # reuse
        buf370 = empty((3, 5, 768), device='cuda', dtype=torch.float32)
        buf371 = empty((15, 768), device='cuda', dtype=torch.float32)
        buf423 = empty((3, 5, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_32, attention_output_20, hidden_states_89, hidden_states_91, hidden_states_93], Original ATen: [aten.add, aten.native_dropout, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_8.run(buf366, buf5, primals_173, buf347, primals_164, primals_165, primals_174, primals_175, buf365, buf370, buf371, buf423, 32, 15, 768, grid=grid(15), stream=stream0)
        del primals_165
        del primals_173
        buf372 = empty((15, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_93], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_177, buf371, reinterpret_tensor(primals_176, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf372)
        del primals_177
        buf373 = empty((15, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_95, intermediate_output_10], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_7.run(buf372, buf373, 46080, grid=grid(46080), stream=stream0)
        buf374 = reinterpret_tensor(buf366, (15, 768), (768, 1), 0); del buf366  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf373, reinterpret_tensor(primals_178, (3072, 768), (1, 3072), 0), out=buf374)
        buf376 = empty((3, 5, 768), device='cuda', dtype=torch.bool)
        buf377 = reinterpret_tensor(buf374, (3, 5, 768), (3840, 768, 1), 0); del buf374  # reuse
        buf381 = empty((3, 5, 768), device='cuda', dtype=torch.float32)
        buf382 = empty((15, 768), device='cuda', dtype=torch.float32)
        buf422 = empty((3, 5, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_33, attention_output_20, hidden_states_96, hidden_states_98, mixed_query_layer_11], Original ATen: [aten.add, aten.native_dropout, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_8.run(buf377, buf5, primals_179, buf370, primals_174, primals_175, primals_180, primals_181, buf376, buf381, buf382, buf422, 33, 15, 768, grid=grid(15), stream=stream0)
        del primals_175
        del primals_179
        buf383 = reinterpret_tensor(buf377, (15, 768), (768, 1), 0); del buf377  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf382, reinterpret_tensor(primals_182, (768, 768), (1, 768), 0), out=buf383)
        buf384 = empty((15, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf382, reinterpret_tensor(primals_184, (768, 768), (1, 768), 0), out=buf384)
        buf385 = empty((15, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf382, reinterpret_tensor(primals_186, (768, 768), (1, 768), 0), out=buf385)
        buf386 = empty((3, 12, 5, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores_33], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf383, primals_183, buf386, 11520, grid=grid(11520), stream=stream0)
        del primals_183
        buf387 = reinterpret_tensor(buf383, (3, 12, 64, 5), (3840, 320, 5, 1), 0); del buf383  # reuse
        # Source Nodes: [attention_scores_33], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf384, primals_185, buf387, 2304, 5, grid=grid(2304, 5), stream=stream0)
        del primals_185
        buf388 = buf354; del buf354  # reuse
        # Source Nodes: [attention_scores_33], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf386, (36, 5, 64), (320, 64, 1), 0), reinterpret_tensor(buf387, (36, 64, 5), (320, 5, 1), 0), out=buf388)
        buf389 = buf356; del buf356  # reuse
        buf390 = buf355; del buf355  # reuse
        # Source Nodes: [attention_probs_22, attention_scores_34, attention_scores_35, extended_attention_mask_1, extended_attention_mask_3, sub], Original ATen: [aten._softmax, aten._to_copy, aten.add, aten.div, aten.mul, aten.rsub]
        triton_poi_fused__softmax__to_copy_add_div_mul_rsub_3.run(buf388, primals_203, buf389, buf390, 180, grid=grid(180), stream=stream0)
        buf392 = empty((3, 12, 5, 5), device='cuda', dtype=torch.bool)
        buf393 = empty((3, 12, 5, 5), device='cuda', dtype=torch.float32)
        buf421 = empty((3, 12, 5, 5), device='cuda', dtype=torch.float32)
        buf427 = reinterpret_tensor(buf320, (3, 12, 5, 5), (300, 25, 5, 1), 0); del buf320  # reuse
        buf430 = reinterpret_tensor(buf286, (3, 12, 5, 5), (300, 25, 5, 1), 0); del buf286  # reuse
        buf433 = reinterpret_tensor(buf252, (3, 12, 5, 5), (300, 25, 5, 1), 0); del buf252  # reuse
        buf436 = reinterpret_tensor(buf218, (3, 12, 5, 5), (300, 25, 5, 1), 0); del buf218  # reuse
        buf439 = reinterpret_tensor(buf184, (3, 12, 5, 5), (300, 25, 5, 1), 0); del buf184  # reuse
        buf442 = reinterpret_tensor(buf150, (3, 12, 5, 5), (300, 25, 5, 1), 0); del buf150  # reuse
        buf445 = reinterpret_tensor(buf116, (3, 12, 5, 5), (300, 25, 5, 1), 0); del buf116  # reuse
        buf448 = reinterpret_tensor(buf82, (3, 12, 5, 5), (300, 25, 5, 1), 0); del buf82  # reuse
        buf451 = reinterpret_tensor(buf48, (3, 12, 5, 5), (300, 25, 5, 1), 0); del buf48  # reuse
        buf454 = reinterpret_tensor(buf14, (3, 12, 5, 5), (300, 25, 5, 1), 0); del buf14  # reuse
        # Source Nodes: [attention_probs, attention_probs_10, attention_probs_12, attention_probs_14, attention_probs_16, attention_probs_18, attention_probs_2, attention_probs_22, attention_probs_23, attention_probs_4, attention_probs_6, attention_probs_8, attention_scores_1, attention_scores_10, attention_scores_11, attention_scores_13, attention_scores_14, attention_scores_16, attention_scores_17, attention_scores_19, attention_scores_2, attention_scores_20, attention_scores_22, attention_scores_23, attention_scores_25, attention_scores_26, attention_scores_28, attention_scores_29, attention_scores_34, attention_scores_35, attention_scores_4, attention_scores_5, attention_scores_7, attention_scores_8, extended_attention_mask_1, extended_attention_mask_3, sub], Original ATen: [aten._softmax, aten._to_copy, aten.add, aten.detach, aten.div, aten.mul, aten.native_dropout, aten.rsub]
        triton_poi_fused__softmax__to_copy_add_detach_div_mul_native_dropout_rsub_10.run(buf427, buf430, buf433, buf436, buf439, buf442, buf445, buf448, buf451, buf454, buf5, buf388, primals_203, buf389, buf390, buf321, buf322, buf287, buf288, buf253, buf254, buf219, buf220, buf185, buf186, buf151, buf152, buf117, buf118, buf83, buf84, buf49, buf50, buf15, buf16, buf392, buf393, buf421, 34, 900, grid=grid(900), stream=stream0)
        del buf117
        del buf118
        del buf15
        del buf151
        del buf152
        del buf16
        del buf185
        del buf186
        del buf219
        del buf220
        del buf253
        del buf254
        del buf287
        del buf288
        del buf321
        del buf322
        del buf388
        del buf389
        del buf390
        del buf49
        del buf50
        del buf83
        del buf84
        del primals_203
        buf394 = reinterpret_tensor(buf384, (3, 12, 5, 64), (3840, 320, 64, 1), 0); del buf384  # reuse
        # Source Nodes: [context_layer_33], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf385, primals_187, buf394, 11520, grid=grid(11520), stream=stream0)
        del primals_187
        buf395 = reinterpret_tensor(buf385, (36, 5, 64), (320, 64, 1), 0); del buf385  # reuse
        # Source Nodes: [context_layer_33], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf393, (36, 5, 5), (25, 5, 1), 0), reinterpret_tensor(buf394, (36, 5, 64), (320, 64, 1), 0), out=buf395)
        buf396 = empty((15, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_99], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf395, buf396, 11520, grid=grid(11520), stream=stream0)
        buf397 = reinterpret_tensor(buf395, (15, 768), (768, 1), 0); del buf395  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf396, reinterpret_tensor(primals_188, (768, 768), (1, 768), 0), out=buf397)
        buf399 = empty((3, 5, 768), device='cuda', dtype=torch.bool)
        buf400 = reinterpret_tensor(buf397, (3, 5, 768), (3840, 768, 1), 0); del buf397  # reuse
        buf404 = empty((3, 5, 768), device='cuda', dtype=torch.float32)
        buf405 = empty((15, 768), device='cuda', dtype=torch.float32)
        buf420 = empty((3, 5, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_35, attention_output_22, hidden_states_100, hidden_states_102, hidden_states_98], Original ATen: [aten.add, aten.native_dropout, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_8.run(buf400, buf5, primals_189, buf381, primals_180, primals_181, primals_190, primals_191, buf399, buf404, buf405, buf420, 35, 15, 768, grid=grid(15), stream=stream0)
        del primals_181
        del primals_189
        buf406 = empty((15, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_102], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_193, buf405, reinterpret_tensor(primals_192, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf406)
        del primals_193
        buf407 = empty((15, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_104, intermediate_output_11], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_7.run(buf406, buf407, 46080, grid=grid(46080), stream=stream0)
        buf408 = reinterpret_tensor(buf400, (15, 768), (768, 1), 0); del buf400  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf407, reinterpret_tensor(primals_194, (3072, 768), (1, 3072), 0), out=buf408)
        buf410 = empty((3, 5, 768), device='cuda', dtype=torch.bool)
        buf411 = reinterpret_tensor(buf408, (3, 5, 768), (3840, 768, 1), 0); del buf408  # reuse
        buf415 = empty((3, 5, 768), device='cuda', dtype=torch.float32)
        buf416 = empty((3, 5, 768), device='cuda', dtype=torch.float32)
        buf419 = empty((3, 5, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_36, attention_output_22, hidden_states_105, sequence_output], Original ATen: [aten.add, aten.native_dropout, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_8.run(buf411, buf5, primals_195, buf404, primals_190, primals_191, primals_196, primals_197, buf410, buf415, buf416, buf419, 36, 15, 768, grid=grid(15), stream=stream0)
        del buf411
        del buf5
        del primals_191
        del primals_195
        del primals_197
        buf417 = empty((3, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf416, (3, 768), (3840, 1), 0), reinterpret_tensor(primals_198, (768, 768), (1, 768), 0), out=buf417)
        buf418 = buf417; del buf417  # reuse
        # Source Nodes: [pooled_output_2], Original ATen: [aten.tanh]
        triton_poi_fused_tanh_11.run(buf418, primals_199, 2304, grid=grid(2304), stream=stream0)
        del primals_199
        return (buf416, buf418, primals_4, primals_14, primals_20, primals_30, primals_36, primals_46, primals_52, primals_62, primals_68, primals_78, primals_84, primals_94, primals_100, primals_110, primals_116, primals_126, primals_132, primals_142, primals_148, primals_158, primals_164, primals_174, primals_180, primals_190, primals_196, primals_202, reinterpret_tensor(primals_200, (3, 5), (0, 1), 0), reinterpret_tensor(primals_201, (1, 5), (512, 1), 0), buf4, buf7, buf8, buf18, buf22, buf25, buf30, buf31, buf32, buf33, buf36, buf41, buf42, buf52, buf56, buf59, buf64, buf65, buf66, buf67, buf70, buf75, buf76, buf86, buf90, buf93, buf98, buf99, buf100, buf101, buf104, buf109, buf110, buf120, buf124, buf127, buf132, buf133, buf134, buf135, buf138, buf143, buf144, buf154, buf158, buf161, buf166, buf167, buf168, buf169, buf172, buf177, buf178, buf188, buf192, buf195, buf200, buf201, buf202, buf203, buf206, buf211, buf212, buf222, buf226, buf229, buf234, buf235, buf236, buf237, buf240, buf245, buf246, buf256, buf260, buf263, buf268, buf269, buf270, buf271, buf274, buf279, buf280, buf290, buf294, buf297, buf302, buf303, buf304, buf305, buf308, buf313, buf314, buf324, buf328, buf331, buf336, buf337, buf338, buf339, buf342, buf347, buf348, buf358, buf362, buf365, buf370, buf371, buf372, buf373, buf376, buf381, buf382, buf392, buf396, buf399, buf404, buf405, buf406, buf407, buf410, buf415, reinterpret_tensor(buf416, (3, 768), (3840, 1), 0), buf418, reinterpret_tensor(primals_198, (768, 768), (768, 1), 0), buf419, reinterpret_tensor(primals_194, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_192, (3072, 768), (768, 1), 0), buf420, reinterpret_tensor(primals_188, (768, 768), (768, 1), 0), reinterpret_tensor(buf393, (36, 5, 5), (25, 1, 5), 0), reinterpret_tensor(buf394, (36, 64, 5), (320, 1, 64), 0), buf421, reinterpret_tensor(buf386, (36, 64, 5), (320, 1, 64), 0), reinterpret_tensor(buf387, (36, 5, 64), (320, 1, 5), 0), reinterpret_tensor(primals_186, (768, 768), (768, 1), 0), reinterpret_tensor(primals_184, (768, 768), (768, 1), 0), reinterpret_tensor(primals_182, (768, 768), (768, 1), 0), buf422, reinterpret_tensor(primals_178, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_176, (3072, 768), (768, 1), 0), buf423, reinterpret_tensor(primals_172, (768, 768), (768, 1), 0), reinterpret_tensor(buf359, (36, 5, 5), (25, 1, 5), 0), reinterpret_tensor(buf360, (36, 64, 5), (320, 1, 64), 0), buf424, reinterpret_tensor(buf352, (36, 64, 5), (320, 1, 64), 0), reinterpret_tensor(buf353, (36, 5, 64), (320, 1, 5), 0), reinterpret_tensor(primals_170, (768, 768), (768, 1), 0), reinterpret_tensor(primals_168, (768, 768), (768, 1), 0), reinterpret_tensor(primals_166, (768, 768), (768, 1), 0), buf425, reinterpret_tensor(primals_162, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_160, (3072, 768), (768, 1), 0), buf426, reinterpret_tensor(primals_156, (768, 768), (768, 1), 0), reinterpret_tensor(buf325, (36, 5, 5), (25, 1, 5), 0), reinterpret_tensor(buf326, (36, 64, 5), (320, 1, 64), 0), buf427, reinterpret_tensor(buf318, (36, 64, 5), (320, 1, 64), 0), reinterpret_tensor(buf319, (36, 5, 64), (320, 1, 5), 0), reinterpret_tensor(primals_154, (768, 768), (768, 1), 0), reinterpret_tensor(primals_152, (768, 768), (768, 1), 0), reinterpret_tensor(primals_150, (768, 768), (768, 1), 0), buf428, reinterpret_tensor(primals_146, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_144, (3072, 768), (768, 1), 0), buf429, reinterpret_tensor(primals_140, (768, 768), (768, 1), 0), reinterpret_tensor(buf291, (36, 5, 5), (25, 1, 5), 0), reinterpret_tensor(buf292, (36, 64, 5), (320, 1, 64), 0), buf430, reinterpret_tensor(buf284, (36, 64, 5), (320, 1, 64), 0), reinterpret_tensor(buf285, (36, 5, 64), (320, 1, 5), 0), reinterpret_tensor(primals_138, (768, 768), (768, 1), 0), reinterpret_tensor(primals_136, (768, 768), (768, 1), 0), reinterpret_tensor(primals_134, (768, 768), (768, 1), 0), buf431, reinterpret_tensor(primals_130, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_128, (3072, 768), (768, 1), 0), buf432, reinterpret_tensor(primals_124, (768, 768), (768, 1), 0), reinterpret_tensor(buf257, (36, 5, 5), (25, 1, 5), 0), reinterpret_tensor(buf258, (36, 64, 5), (320, 1, 64), 0), buf433, reinterpret_tensor(buf250, (36, 64, 5), (320, 1, 64), 0), reinterpret_tensor(buf251, (36, 5, 64), (320, 1, 5), 0), reinterpret_tensor(primals_122, (768, 768), (768, 1), 0), reinterpret_tensor(primals_120, (768, 768), (768, 1), 0), reinterpret_tensor(primals_118, (768, 768), (768, 1), 0), buf434, reinterpret_tensor(primals_114, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_112, (3072, 768), (768, 1), 0), buf435, reinterpret_tensor(primals_108, (768, 768), (768, 1), 0), reinterpret_tensor(buf223, (36, 5, 5), (25, 1, 5), 0), reinterpret_tensor(buf224, (36, 64, 5), (320, 1, 64), 0), buf436, reinterpret_tensor(buf216, (36, 64, 5), (320, 1, 64), 0), reinterpret_tensor(buf217, (36, 5, 64), (320, 1, 5), 0), reinterpret_tensor(primals_106, (768, 768), (768, 1), 0), reinterpret_tensor(primals_104, (768, 768), (768, 1), 0), reinterpret_tensor(primals_102, (768, 768), (768, 1), 0), buf437, reinterpret_tensor(primals_98, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_96, (3072, 768), (768, 1), 0), buf438, reinterpret_tensor(primals_92, (768, 768), (768, 1), 0), reinterpret_tensor(buf189, (36, 5, 5), (25, 1, 5), 0), reinterpret_tensor(buf190, (36, 64, 5), (320, 1, 64), 0), buf439, reinterpret_tensor(buf182, (36, 64, 5), (320, 1, 64), 0), reinterpret_tensor(buf183, (36, 5, 64), (320, 1, 5), 0), reinterpret_tensor(primals_90, (768, 768), (768, 1), 0), reinterpret_tensor(primals_88, (768, 768), (768, 1), 0), reinterpret_tensor(primals_86, (768, 768), (768, 1), 0), buf440, reinterpret_tensor(primals_82, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_80, (3072, 768), (768, 1), 0), buf441, reinterpret_tensor(primals_76, (768, 768), (768, 1), 0), reinterpret_tensor(buf155, (36, 5, 5), (25, 1, 5), 0), reinterpret_tensor(buf156, (36, 64, 5), (320, 1, 64), 0), buf442, reinterpret_tensor(buf148, (36, 64, 5), (320, 1, 64), 0), reinterpret_tensor(buf149, (36, 5, 64), (320, 1, 5), 0), reinterpret_tensor(primals_74, (768, 768), (768, 1), 0), reinterpret_tensor(primals_72, (768, 768), (768, 1), 0), reinterpret_tensor(primals_70, (768, 768), (768, 1), 0), buf443, reinterpret_tensor(primals_66, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_64, (3072, 768), (768, 1), 0), buf444, reinterpret_tensor(primals_60, (768, 768), (768, 1), 0), reinterpret_tensor(buf121, (36, 5, 5), (25, 1, 5), 0), reinterpret_tensor(buf122, (36, 64, 5), (320, 1, 64), 0), buf445, reinterpret_tensor(buf114, (36, 64, 5), (320, 1, 64), 0), reinterpret_tensor(buf115, (36, 5, 64), (320, 1, 5), 0), reinterpret_tensor(primals_58, (768, 768), (768, 1), 0), reinterpret_tensor(primals_56, (768, 768), (768, 1), 0), reinterpret_tensor(primals_54, (768, 768), (768, 1), 0), buf446, reinterpret_tensor(primals_50, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_48, (3072, 768), (768, 1), 0), buf447, reinterpret_tensor(primals_44, (768, 768), (768, 1), 0), reinterpret_tensor(buf87, (36, 5, 5), (25, 1, 5), 0), reinterpret_tensor(buf88, (36, 64, 5), (320, 1, 64), 0), buf448, reinterpret_tensor(buf80, (36, 64, 5), (320, 1, 64), 0), reinterpret_tensor(buf81, (36, 5, 64), (320, 1, 5), 0), reinterpret_tensor(primals_42, (768, 768), (768, 1), 0), reinterpret_tensor(primals_40, (768, 768), (768, 1), 0), reinterpret_tensor(primals_38, (768, 768), (768, 1), 0), buf449, reinterpret_tensor(primals_34, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_32, (3072, 768), (768, 1), 0), buf450, reinterpret_tensor(primals_28, (768, 768), (768, 1), 0), reinterpret_tensor(buf53, (36, 5, 5), (25, 1, 5), 0), reinterpret_tensor(buf54, (36, 64, 5), (320, 1, 64), 0), buf451, reinterpret_tensor(buf46, (36, 64, 5), (320, 1, 64), 0), reinterpret_tensor(buf47, (36, 5, 64), (320, 1, 5), 0), reinterpret_tensor(primals_26, (768, 768), (768, 1), 0), reinterpret_tensor(primals_24, (768, 768), (768, 1), 0), reinterpret_tensor(primals_22, (768, 768), (768, 1), 0), buf452, reinterpret_tensor(primals_18, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_16, (3072, 768), (768, 1), 0), buf453, reinterpret_tensor(primals_12, (768, 768), (768, 1), 0), reinterpret_tensor(buf19, (36, 5, 5), (25, 1, 5), 0), reinterpret_tensor(buf20, (36, 64, 5), (320, 1, 64), 0), buf454, reinterpret_tensor(buf12, (36, 64, 5), (320, 1, 64), 0), reinterpret_tensor(buf13, (36, 5, 64), (320, 1, 5), 0), reinterpret_tensor(primals_10, (768, 768), (768, 1), 0), reinterpret_tensor(primals_8, (768, 768), (768, 1), 0), reinterpret_tensor(primals_6, (768, 768), (768, 1), 0), buf455, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((30522, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((2, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    primals_201 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    primals_202 = rand_strided((3, 5), (5, 1), device='cuda:0', dtype=torch.int64)
    primals_203 = rand_strided((3, 5), (5, 1), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
