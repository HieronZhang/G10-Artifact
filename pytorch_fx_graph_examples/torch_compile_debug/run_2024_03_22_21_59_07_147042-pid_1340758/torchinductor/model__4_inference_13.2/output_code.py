
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


# kernel path: /tmp/torchinductor_zhang402/uu/cuurptm2dbc5b5njj7nv6ccb5o2yfy7wdcpctnpzhizrrj47dqey.py
# Source Nodes: [], Original ATen: []

triton_for_fused_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.triton_heuristics import foreach
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
@foreach(num_warps=8, triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]}, inductor_meta={'kernel_name': 'triton_for_fused_0'})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5):
    xpid = tl.program_id(0)
    XBLOCK: tl.constexpr = 1024
    if xpid >= 0 and xpid < 1:
        xpid_offset = xpid - 0
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp0 = tl.load(in_ptr0 + (0))
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
        tmp2 = 1.0
        tmp3 = tmp1 + tmp2
        tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp3, None)
    elif xpid >= 1 and xpid < 2:
        xpid_offset = xpid - 1
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp4 = tl.load(in_ptr1 + (0))
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
        tmp6 = 1.0
        tmp7 = tmp5 + tmp6
        tl.store(out_ptr1 + (tl.full([XBLOCK], 0, tl.int32)), tmp7, None)
    elif xpid >= 2 and xpid < 3:
        xpid_offset = xpid - 2
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp8 = tl.load(in_ptr2 + (0))
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
        tmp10 = 1.0
        tmp11 = tmp9 + tmp10
        tl.store(out_ptr2 + (tl.full([XBLOCK], 0, tl.int32)), tmp11, None)
    elif xpid >= 3 and xpid < 4:
        xpid_offset = xpid - 3
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp12 = tl.load(in_ptr3 + (0))
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK])
        tmp14 = 1.0
        tmp15 = tmp13 + tmp14
        tl.store(out_ptr3 + (tl.full([XBLOCK], 0, tl.int32)), tmp15, None)
    elif xpid >= 4 and xpid < 5:
        xpid_offset = xpid - 4
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp16 = tl.load(in_ptr4 + (0))
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK])
        tmp18 = 1.0
        tmp19 = tmp17 + tmp18
        tl.store(out_ptr4 + (tl.full([XBLOCK], 0, tl.int32)), tmp19, None)
    elif xpid >= 5 and xpid < 6:
        xpid_offset = xpid - 5
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp20 = tl.load(in_ptr5 + (0))
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK])
        tmp22 = 1.0
        tmp23 = tmp21 + tmp22
        tl.store(out_ptr5 + (tl.full([XBLOCK], 0, tl.int32)), tmp23, None)
    else:
        pass
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_zhang402/gz/cgz5clzfg3r4cumt4f6odhj3qdilg6ngu6fets47oystj3hvxwr2.py
# Source Nodes: [], Original ATen: []

triton_for_fused_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.triton_heuristics import foreach
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
@foreach(num_warps=8, triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: '*fp32', 35: '*fp32', 36: '*fp32', 37: '*fp32', 38: '*fp32', 39: '*fp32', 40: '*fp32', 41: '*fp32', 42: '*fp32', 43: '*fp32', 44: '*fp32', 45: '*fp32', 46: '*fp32', 47: '*fp32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]}, inductor_meta={'kernel_name': 'triton_for_fused_1'})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, out_ptr0, out_ptr2, out_ptr3, out_ptr4, out_ptr6, out_ptr7, out_ptr8, out_ptr10, out_ptr11, out_ptr12, out_ptr14, out_ptr15, out_ptr16, out_ptr18, out_ptr19, out_ptr20, out_ptr22, out_ptr23):
    xpid = tl.program_id(0)
    XBLOCK: tl.constexpr = 1024
    if xpid >= 0 and xpid < 392:
        xpid_offset = xpid - 0
        xnumel = 401408
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), None)
        tmp1 = tl.load(in_ptr1 + (x0), None)
        tmp6 = tl.load(in_ptr2 + (x0), None)
        tmp13 = tl.load(in_ptr3 + (x0), None)
        tmp15 = tl.load(in_ptr4 + (0))
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
        tmp2 = tmp1 - tmp0
        tmp3 = 0.09999999999999998
        tmp4 = tmp2 * tmp3
        tmp5 = tmp0 + tmp4
        tmp7 = 0.999
        tmp8 = tmp6 * tmp7
        tmp9 = tmp1 * tmp1
        tmp10 = 0.0010000000000000009
        tmp11 = tmp9 * tmp10
        tmp12 = tmp8 + tmp11
        tmp14 = tl.sqrt(tmp12)
        tmp17 = tl.math.pow(tmp7, tmp16)
        tmp18 = 1.0
        tmp19 = tmp17 - tmp18
        tmp20 = -tmp19
        tmp21 = tl.sqrt(tmp20)
        tmp22 = tmp14 / tmp21
        tmp23 = 1e-08
        tmp24 = tmp22 + tmp23
        tmp25 = 0.9
        tmp26 = tl.math.pow(tmp25, tmp16)
        tmp27 = tmp26 - tmp18
        tmp28 = 0.001
        tmp29 = tmp27 / tmp28
        tmp30 = 1 / tmp29
        tmp31 = tmp24 / tmp30
        tmp32 = tmp5 / tmp31
        tmp33 = tmp13 + tmp32
        tl.store(out_ptr0 + (x0), tmp5, None)
        tl.store(out_ptr2 + (x0), tmp33, None)
        tl.store(out_ptr3 + (x0), tmp12, None)
    elif xpid >= 392 and xpid < 393:
        xpid_offset = xpid - 392
        xnumel = 512
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x1 = xindex
        tmp34 = tl.load(in_ptr5 + (x1), xmask)
        tmp35 = tl.load(in_ptr6 + (x1), xmask)
        tmp40 = tl.load(in_ptr7 + (x1), xmask)
        tmp47 = tl.load(in_ptr8 + (x1), xmask)
        tmp49 = tl.load(in_ptr9 + (0))
        tmp50 = tl.broadcast_to(tmp49, [XBLOCK])
        tmp36 = tmp35 - tmp34
        tmp37 = 0.09999999999999998
        tmp38 = tmp36 * tmp37
        tmp39 = tmp34 + tmp38
        tmp41 = 0.999
        tmp42 = tmp40 * tmp41
        tmp43 = tmp35 * tmp35
        tmp44 = 0.0010000000000000009
        tmp45 = tmp43 * tmp44
        tmp46 = tmp42 + tmp45
        tmp48 = tl.sqrt(tmp46)
        tmp51 = tl.math.pow(tmp41, tmp50)
        tmp52 = 1.0
        tmp53 = tmp51 - tmp52
        tmp54 = -tmp53
        tmp55 = tl.sqrt(tmp54)
        tmp56 = tmp48 / tmp55
        tmp57 = 1e-08
        tmp58 = tmp56 + tmp57
        tmp59 = 0.9
        tmp60 = tl.math.pow(tmp59, tmp50)
        tmp61 = tmp60 - tmp52
        tmp62 = 0.001
        tmp63 = tmp61 / tmp62
        tmp64 = 1 / tmp63
        tmp65 = tmp58 / tmp64
        tmp66 = tmp39 / tmp65
        tmp67 = tmp47 + tmp66
        tl.store(out_ptr4 + (x1), tmp39, xmask)
        tl.store(out_ptr6 + (x1), tmp67, xmask)
        tl.store(out_ptr7 + (x1), tmp46, xmask)
    elif xpid >= 393 and xpid < 649:
        xpid_offset = xpid - 393
        xnumel = 262144
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x2 = xindex
        tmp68 = tl.load(in_ptr10 + (x2), None)
        tmp69 = tl.load(in_ptr11 + (x2), None)
        tmp74 = tl.load(in_ptr12 + (x2), None)
        tmp81 = tl.load(in_ptr13 + (x2), None)
        tmp83 = tl.load(in_ptr14 + (0))
        tmp84 = tl.broadcast_to(tmp83, [XBLOCK])
        tmp70 = tmp69 - tmp68
        tmp71 = 0.09999999999999998
        tmp72 = tmp70 * tmp71
        tmp73 = tmp68 + tmp72
        tmp75 = 0.999
        tmp76 = tmp74 * tmp75
        tmp77 = tmp69 * tmp69
        tmp78 = 0.0010000000000000009
        tmp79 = tmp77 * tmp78
        tmp80 = tmp76 + tmp79
        tmp82 = tl.sqrt(tmp80)
        tmp85 = tl.math.pow(tmp75, tmp84)
        tmp86 = 1.0
        tmp87 = tmp85 - tmp86
        tmp88 = -tmp87
        tmp89 = tl.sqrt(tmp88)
        tmp90 = tmp82 / tmp89
        tmp91 = 1e-08
        tmp92 = tmp90 + tmp91
        tmp93 = 0.9
        tmp94 = tl.math.pow(tmp93, tmp84)
        tmp95 = tmp94 - tmp86
        tmp96 = 0.001
        tmp97 = tmp95 / tmp96
        tmp98 = 1 / tmp97
        tmp99 = tmp92 / tmp98
        tmp100 = tmp73 / tmp99
        tmp101 = tmp81 + tmp100
        tl.store(out_ptr8 + (x2), tmp73, None)
        tl.store(out_ptr10 + (x2), tmp101, None)
        tl.store(out_ptr11 + (x2), tmp80, None)
    elif xpid >= 649 and xpid < 650:
        xpid_offset = xpid - 649
        xnumel = 512
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x3 = xindex
        tmp102 = tl.load(in_ptr15 + (x3), xmask)
        tmp103 = tl.load(in_ptr16 + (x3), xmask)
        tmp108 = tl.load(in_ptr17 + (x3), xmask)
        tmp115 = tl.load(in_ptr18 + (x3), xmask)
        tmp117 = tl.load(in_ptr19 + (0))
        tmp118 = tl.broadcast_to(tmp117, [XBLOCK])
        tmp104 = tmp103 - tmp102
        tmp105 = 0.09999999999999998
        tmp106 = tmp104 * tmp105
        tmp107 = tmp102 + tmp106
        tmp109 = 0.999
        tmp110 = tmp108 * tmp109
        tmp111 = tmp103 * tmp103
        tmp112 = 0.0010000000000000009
        tmp113 = tmp111 * tmp112
        tmp114 = tmp110 + tmp113
        tmp116 = tl.sqrt(tmp114)
        tmp119 = tl.math.pow(tmp109, tmp118)
        tmp120 = 1.0
        tmp121 = tmp119 - tmp120
        tmp122 = -tmp121
        tmp123 = tl.sqrt(tmp122)
        tmp124 = tmp116 / tmp123
        tmp125 = 1e-08
        tmp126 = tmp124 + tmp125
        tmp127 = 0.9
        tmp128 = tl.math.pow(tmp127, tmp118)
        tmp129 = tmp128 - tmp120
        tmp130 = 0.001
        tmp131 = tmp129 / tmp130
        tmp132 = 1 / tmp131
        tmp133 = tmp126 / tmp132
        tmp134 = tmp107 / tmp133
        tmp135 = tmp115 + tmp134
        tl.store(out_ptr12 + (x3), tmp107, xmask)
        tl.store(out_ptr14 + (x3), tmp135, xmask)
        tl.store(out_ptr15 + (x3), tmp114, xmask)
    elif xpid >= 650 and xpid < 655:
        xpid_offset = xpid - 650
        xnumel = 5120
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x4 = xindex
        tmp136 = tl.load(in_ptr20 + (x4), xmask)
        tmp137 = tl.load(in_ptr21 + (x4), xmask)
        tmp142 = tl.load(in_ptr22 + (x4), xmask)
        tmp149 = tl.load(in_ptr23 + (x4), xmask)
        tmp151 = tl.load(in_ptr24 + (0))
        tmp152 = tl.broadcast_to(tmp151, [XBLOCK])
        tmp138 = tmp137 - tmp136
        tmp139 = 0.09999999999999998
        tmp140 = tmp138 * tmp139
        tmp141 = tmp136 + tmp140
        tmp143 = 0.999
        tmp144 = tmp142 * tmp143
        tmp145 = tmp137 * tmp137
        tmp146 = 0.0010000000000000009
        tmp147 = tmp145 * tmp146
        tmp148 = tmp144 + tmp147
        tmp150 = tl.sqrt(tmp148)
        tmp153 = tl.math.pow(tmp143, tmp152)
        tmp154 = 1.0
        tmp155 = tmp153 - tmp154
        tmp156 = -tmp155
        tmp157 = tl.sqrt(tmp156)
        tmp158 = tmp150 / tmp157
        tmp159 = 1e-08
        tmp160 = tmp158 + tmp159
        tmp161 = 0.9
        tmp162 = tl.math.pow(tmp161, tmp152)
        tmp163 = tmp162 - tmp154
        tmp164 = 0.001
        tmp165 = tmp163 / tmp164
        tmp166 = 1 / tmp165
        tmp167 = tmp160 / tmp166
        tmp168 = tmp141 / tmp167
        tmp169 = tmp149 + tmp168
        tl.store(out_ptr16 + (x4), tmp141, xmask)
        tl.store(out_ptr18 + (x4), tmp169, xmask)
        tl.store(out_ptr19 + (x4), tmp148, xmask)
    elif xpid >= 655 and xpid < 656:
        xpid_offset = xpid - 655
        xnumel = 10
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x5 = xindex
        tmp170 = tl.load(in_ptr25 + (x5), xmask)
        tmp171 = tl.load(in_ptr26 + (x5), xmask)
        tmp176 = tl.load(in_ptr27 + (x5), xmask)
        tmp183 = tl.load(in_ptr28 + (x5), xmask)
        tmp185 = tl.load(in_ptr29 + (0))
        tmp186 = tl.broadcast_to(tmp185, [XBLOCK])
        tmp172 = tmp171 - tmp170
        tmp173 = 0.09999999999999998
        tmp174 = tmp172 * tmp173
        tmp175 = tmp170 + tmp174
        tmp177 = 0.999
        tmp178 = tmp176 * tmp177
        tmp179 = tmp171 * tmp171
        tmp180 = 0.0010000000000000009
        tmp181 = tmp179 * tmp180
        tmp182 = tmp178 + tmp181
        tmp184 = tl.sqrt(tmp182)
        tmp187 = tl.math.pow(tmp177, tmp186)
        tmp188 = 1.0
        tmp189 = tmp187 - tmp188
        tmp190 = -tmp189
        tmp191 = tl.sqrt(tmp190)
        tmp192 = tmp184 / tmp191
        tmp193 = 1e-08
        tmp194 = tmp192 + tmp193
        tmp195 = 0.9
        tmp196 = tl.math.pow(tmp195, tmp186)
        tmp197 = tmp196 - tmp188
        tmp198 = 0.001
        tmp199 = tmp197 / tmp198
        tmp200 = 1 / tmp199
        tmp201 = tmp194 / tmp200
        tmp202 = tmp175 / tmp201
        tmp203 = tmp183 + tmp202
        tl.store(out_ptr20 + (x5), tmp175, xmask)
        tl.store(out_ptr22 + (x5), tmp203, xmask)
        tl.store(out_ptr23 + (x5), tmp182, xmask)
    else:
        pass
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1 = args
    args.clear()
    assert_size_stride(arg0_1, (512, 784), (784, 1))
    assert_size_stride(arg1_1, (512, ), (1, ))
    assert_size_stride(arg2_1, (512, 512), (512, 1))
    assert_size_stride(arg3_1, (512, ), (1, ))
    assert_size_stride(arg4_1, (10, 512), (512, 1))
    assert_size_stride(arg5_1, (10, ), (1, ))
    assert_size_stride(arg6_1, (512, 784), (784, 1))
    assert_size_stride(arg7_1, (512, ), (1, ))
    assert_size_stride(arg8_1, (512, 512), (512, 1))
    assert_size_stride(arg9_1, (512, ), (1, ))
    assert_size_stride(arg10_1, (10, 512), (512, 1))
    assert_size_stride(arg11_1, (10, ), (1, ))
    assert_size_stride(arg12_1, (512, 784), (784, 1))
    assert_size_stride(arg13_1, (512, ), (1, ))
    assert_size_stride(arg14_1, (512, 512), (512, 1))
    assert_size_stride(arg15_1, (512, ), (1, ))
    assert_size_stride(arg16_1, (10, 512), (512, 1))
    assert_size_stride(arg17_1, (10, ), (1, ))
    assert_size_stride(arg18_1, (), ())
    assert_size_stride(arg19_1, (), ())
    assert_size_stride(arg20_1, (), ())
    assert_size_stride(arg21_1, (), ())
    assert_size_stride(arg22_1, (), ())
    assert_size_stride(arg23_1, (), ())
    assert_size_stride(arg24_1, (512, 784), (784, 1))
    assert_size_stride(arg25_1, (512, ), (1, ))
    assert_size_stride(arg26_1, (512, 512), (512, 1))
    assert_size_stride(arg27_1, (512, ), (1, ))
    assert_size_stride(arg28_1, (10, 512), (512, 1))
    assert_size_stride(arg29_1, (10, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [], Original ATen: []
        stream0 = get_cuda_stream(0)
        triton_for_fused_0.run(arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, grid=((6, 1, 1)), stream=stream0)
        # Source Nodes: [], Original ATen: []
        triton_for_fused_1.run(arg6_1, arg24_1, arg12_1, arg0_1, arg18_1, arg7_1, arg25_1, arg13_1, arg1_1, arg19_1, arg8_1, arg26_1, arg14_1, arg2_1, arg20_1, arg9_1, arg27_1, arg15_1, arg3_1, arg21_1, arg10_1, arg28_1, arg16_1, arg4_1, arg22_1, arg11_1, arg29_1, arg17_1, arg5_1, arg23_1, arg6_1, arg0_1, arg12_1, arg7_1, arg1_1, arg13_1, arg8_1, arg2_1, arg14_1, arg9_1, arg3_1, arg15_1, arg10_1, arg4_1, arg16_1, arg11_1, arg5_1, arg17_1, grid=((656, 1, 1)), stream=stream0)
        del arg0_1
        del arg10_1
        del arg11_1
        del arg12_1
        del arg13_1
        del arg14_1
        del arg15_1
        del arg16_1
        del arg17_1
        del arg18_1
        del arg19_1
        del arg1_1
        del arg20_1
        del arg21_1
        del arg22_1
        del arg23_1
        del arg24_1
        del arg25_1
        del arg26_1
        del arg27_1
        del arg28_1
        del arg29_1
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        del arg6_1
        del arg7_1
        del arg8_1
        del arg9_1
        return ()


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((512, 784), (784, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((10, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((512, 784), (784, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((10, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((512, 784), (784, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((10, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((512, 784), (784, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((10, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
