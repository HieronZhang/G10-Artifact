
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


# kernel path: /tmp/torchinductor_zhang402/vg/cvgc3n74r6bkxxzxzbxk42tqeyplrfkwui67wkdx6jubluyc3luf.py
# Source Nodes: [], Original ATen: []

triton_for_fused_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.foreach(
    num_warps=8,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: '*fp32', 35: '*fp32', 36: '*fp32', 37: '*fp32', 38: '*fp32', 39: '*fp32', 40: '*fp32', 41: '*fp32', 42: '*fp32', 43: '*fp32', 44: '*fp32', 45: '*fp32', 46: '*fp32', 47: '*fp32', 48: '*fp32', 49: '*fp32', 50: '*fp32', 51: '*fp32', 52: '*fp32', 53: '*fp32', 54: '*fp32', 55: '*fp32', 56: '*fp32', 57: '*fp32', 58: '*fp32', 59: '*fp32', 60: '*fp32', 61: '*fp32', 62: '*fp32', 63: '*fp32', 64: '*fp32', 65: '*fp32', 66: '*fp32', 67: '*fp32', 68: '*fp32', 69: '*fp32', 70: '*fp32', 71: '*fp32', 72: '*fp32', 73: '*fp32', 74: '*fp32', 75: '*fp32', 76: '*fp32', 77: '*fp32', 78: '*fp32', 79: '*fp32', 80: '*fp32', 81: '*fp32', 82: '*fp32', 83: '*fp32', 84: '*fp32', 85: '*fp32', 86: '*fp32', 87: '*fp32', 88: '*fp32', 89: '*fp32', 90: '*fp32', 91: '*fp32', 92: '*fp32', 93: '*fp32', 94: '*fp32', 95: '*fp32', 96: '*fp32', 97: '*fp32', 98: '*fp32', 99: '*fp32', 100: '*fp32', 101: '*fp32', 102: '*fp32', 103: '*fp32', 104: '*fp32', 105: '*fp32', 106: '*fp32', 107: '*fp32', 108: '*fp32', 109: '*fp32', 110: '*fp32', 111: '*fp32', 112: '*fp32', 113: '*fp32', 114: '*fp32', 115: '*fp32', 116: '*fp32', 117: '*fp32', 118: '*fp32', 119: '*fp32', 120: '*fp32', 121: '*fp32', 122: '*fp32', 123: '*fp32', 124: '*fp32', 125: '*fp32', 126: '*fp32', 127: '*fp32', 128: '*fp32', 129: '*fp32', 130: '*fp32', 131: '*fp32', 132: '*fp32', 133: '*fp32', 134: '*fp32', 135: '*fp32', 136: '*fp32', 137: '*fp32', 138: '*fp32', 139: '*fp32', 140: '*fp32', 141: '*fp32', 142: '*fp32', 143: '*fp32', 144: '*fp32', 145: '*fp32', 146: '*fp32', 147: '*fp32', 148: '*fp32', 149: '*fp32', 150: '*fp32', 151: '*fp32', 152: '*fp32', 153: '*fp32', 154: '*fp32', 155: '*fp32', 156: '*fp32', 157: '*fp32', 158: '*fp32', 159: '*fp32', 160: '*fp32', 161: '*fp32', 162: '*fp32', 163: '*fp32', 164: '*fp32', 165: '*fp32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'kernel_name': 'triton_for_fused_0', 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, in_ptr47, in_ptr48, in_ptr49, in_ptr50, in_ptr51, in_ptr52, in_ptr53, in_ptr54, in_ptr55, in_ptr56, in_ptr57, in_ptr58, in_ptr59, in_ptr60, in_ptr61, in_ptr62, in_ptr63, in_ptr64, in_ptr65, in_ptr66, in_ptr67, in_ptr68, in_ptr69, in_ptr70, in_ptr71, in_ptr72, in_ptr73, in_ptr74, in_ptr75, in_ptr76, in_ptr77, in_ptr78, in_ptr79, in_ptr80, in_ptr81, in_ptr82, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, out_ptr13, out_ptr14, out_ptr15, out_ptr16, out_ptr17, out_ptr18, out_ptr19, out_ptr20, out_ptr21, out_ptr22, out_ptr23, out_ptr24, out_ptr25, out_ptr26, out_ptr27, out_ptr28, out_ptr29, out_ptr30, out_ptr31, out_ptr32, out_ptr33, out_ptr34, out_ptr35, out_ptr36, out_ptr37, out_ptr38, out_ptr39, out_ptr40, out_ptr41, out_ptr42, out_ptr43, out_ptr44, out_ptr45, out_ptr46, out_ptr47, out_ptr48, out_ptr49, out_ptr50, out_ptr51, out_ptr52, out_ptr53, out_ptr54, out_ptr55, out_ptr56, out_ptr57, out_ptr58, out_ptr59, out_ptr60, out_ptr61, out_ptr62, out_ptr63, out_ptr64, out_ptr65, out_ptr66, out_ptr67, out_ptr68, out_ptr69, out_ptr70, out_ptr71, out_ptr72, out_ptr73, out_ptr74, out_ptr75, out_ptr76, out_ptr77, out_ptr78, out_ptr79, out_ptr80, out_ptr81, out_ptr82):
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
    elif xpid >= 6 and xpid < 7:
        xpid_offset = xpid - 6
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp24 = tl.load(in_ptr6 + (0))
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK])
        tmp26 = 1.0
        tmp27 = tmp25 + tmp26
        tl.store(out_ptr6 + (tl.full([XBLOCK], 0, tl.int32)), tmp27, None)
    elif xpid >= 7 and xpid < 8:
        xpid_offset = xpid - 7
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp28 = tl.load(in_ptr7 + (0))
        tmp29 = tl.broadcast_to(tmp28, [XBLOCK])
        tmp30 = 1.0
        tmp31 = tmp29 + tmp30
        tl.store(out_ptr7 + (tl.full([XBLOCK], 0, tl.int32)), tmp31, None)
    elif xpid >= 8 and xpid < 9:
        xpid_offset = xpid - 8
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp32 = tl.load(in_ptr8 + (0))
        tmp33 = tl.broadcast_to(tmp32, [XBLOCK])
        tmp34 = 1.0
        tmp35 = tmp33 + tmp34
        tl.store(out_ptr8 + (tl.full([XBLOCK], 0, tl.int32)), tmp35, None)
    elif xpid >= 9 and xpid < 10:
        xpid_offset = xpid - 9
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp36 = tl.load(in_ptr9 + (0))
        tmp37 = tl.broadcast_to(tmp36, [XBLOCK])
        tmp38 = 1.0
        tmp39 = tmp37 + tmp38
        tl.store(out_ptr9 + (tl.full([XBLOCK], 0, tl.int32)), tmp39, None)
    elif xpid >= 10 and xpid < 11:
        xpid_offset = xpid - 10
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp40 = tl.load(in_ptr10 + (0))
        tmp41 = tl.broadcast_to(tmp40, [XBLOCK])
        tmp42 = 1.0
        tmp43 = tmp41 + tmp42
        tl.store(out_ptr10 + (tl.full([XBLOCK], 0, tl.int32)), tmp43, None)
    elif xpid >= 11 and xpid < 12:
        xpid_offset = xpid - 11
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp44 = tl.load(in_ptr11 + (0))
        tmp45 = tl.broadcast_to(tmp44, [XBLOCK])
        tmp46 = 1.0
        tmp47 = tmp45 + tmp46
        tl.store(out_ptr11 + (tl.full([XBLOCK], 0, tl.int32)), tmp47, None)
    elif xpid >= 12 and xpid < 13:
        xpid_offset = xpid - 12
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp48 = tl.load(in_ptr12 + (0))
        tmp49 = tl.broadcast_to(tmp48, [XBLOCK])
        tmp50 = 1.0
        tmp51 = tmp49 + tmp50
        tl.store(out_ptr12 + (tl.full([XBLOCK], 0, tl.int32)), tmp51, None)
    elif xpid >= 13 and xpid < 14:
        xpid_offset = xpid - 13
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp52 = tl.load(in_ptr13 + (0))
        tmp53 = tl.broadcast_to(tmp52, [XBLOCK])
        tmp54 = 1.0
        tmp55 = tmp53 + tmp54
        tl.store(out_ptr13 + (tl.full([XBLOCK], 0, tl.int32)), tmp55, None)
    elif xpid >= 14 and xpid < 15:
        xpid_offset = xpid - 14
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp56 = tl.load(in_ptr14 + (0))
        tmp57 = tl.broadcast_to(tmp56, [XBLOCK])
        tmp58 = 1.0
        tmp59 = tmp57 + tmp58
        tl.store(out_ptr14 + (tl.full([XBLOCK], 0, tl.int32)), tmp59, None)
    elif xpid >= 15 and xpid < 16:
        xpid_offset = xpid - 15
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp60 = tl.load(in_ptr15 + (0))
        tmp61 = tl.broadcast_to(tmp60, [XBLOCK])
        tmp62 = 1.0
        tmp63 = tmp61 + tmp62
        tl.store(out_ptr15 + (tl.full([XBLOCK], 0, tl.int32)), tmp63, None)
    elif xpid >= 16 and xpid < 17:
        xpid_offset = xpid - 16
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp64 = tl.load(in_ptr16 + (0))
        tmp65 = tl.broadcast_to(tmp64, [XBLOCK])
        tmp66 = 1.0
        tmp67 = tmp65 + tmp66
        tl.store(out_ptr16 + (tl.full([XBLOCK], 0, tl.int32)), tmp67, None)
    elif xpid >= 17 and xpid < 18:
        xpid_offset = xpid - 17
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp68 = tl.load(in_ptr17 + (0))
        tmp69 = tl.broadcast_to(tmp68, [XBLOCK])
        tmp70 = 1.0
        tmp71 = tmp69 + tmp70
        tl.store(out_ptr17 + (tl.full([XBLOCK], 0, tl.int32)), tmp71, None)
    elif xpid >= 18 and xpid < 19:
        xpid_offset = xpid - 18
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp72 = tl.load(in_ptr18 + (0))
        tmp73 = tl.broadcast_to(tmp72, [XBLOCK])
        tmp74 = 1.0
        tmp75 = tmp73 + tmp74
        tl.store(out_ptr18 + (tl.full([XBLOCK], 0, tl.int32)), tmp75, None)
    elif xpid >= 19 and xpid < 20:
        xpid_offset = xpid - 19
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp76 = tl.load(in_ptr19 + (0))
        tmp77 = tl.broadcast_to(tmp76, [XBLOCK])
        tmp78 = 1.0
        tmp79 = tmp77 + tmp78
        tl.store(out_ptr19 + (tl.full([XBLOCK], 0, tl.int32)), tmp79, None)
    elif xpid >= 20 and xpid < 21:
        xpid_offset = xpid - 20
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp80 = tl.load(in_ptr20 + (0))
        tmp81 = tl.broadcast_to(tmp80, [XBLOCK])
        tmp82 = 1.0
        tmp83 = tmp81 + tmp82
        tl.store(out_ptr20 + (tl.full([XBLOCK], 0, tl.int32)), tmp83, None)
    elif xpid >= 21 and xpid < 22:
        xpid_offset = xpid - 21
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp84 = tl.load(in_ptr21 + (0))
        tmp85 = tl.broadcast_to(tmp84, [XBLOCK])
        tmp86 = 1.0
        tmp87 = tmp85 + tmp86
        tl.store(out_ptr21 + (tl.full([XBLOCK], 0, tl.int32)), tmp87, None)
    elif xpid >= 22 and xpid < 23:
        xpid_offset = xpid - 22
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp88 = tl.load(in_ptr22 + (0))
        tmp89 = tl.broadcast_to(tmp88, [XBLOCK])
        tmp90 = 1.0
        tmp91 = tmp89 + tmp90
        tl.store(out_ptr22 + (tl.full([XBLOCK], 0, tl.int32)), tmp91, None)
    elif xpid >= 23 and xpid < 24:
        xpid_offset = xpid - 23
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp92 = tl.load(in_ptr23 + (0))
        tmp93 = tl.broadcast_to(tmp92, [XBLOCK])
        tmp94 = 1.0
        tmp95 = tmp93 + tmp94
        tl.store(out_ptr23 + (tl.full([XBLOCK], 0, tl.int32)), tmp95, None)
    elif xpid >= 24 and xpid < 25:
        xpid_offset = xpid - 24
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp96 = tl.load(in_ptr24 + (0))
        tmp97 = tl.broadcast_to(tmp96, [XBLOCK])
        tmp98 = 1.0
        tmp99 = tmp97 + tmp98
        tl.store(out_ptr24 + (tl.full([XBLOCK], 0, tl.int32)), tmp99, None)
    elif xpid >= 25 and xpid < 26:
        xpid_offset = xpid - 25
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp100 = tl.load(in_ptr25 + (0))
        tmp101 = tl.broadcast_to(tmp100, [XBLOCK])
        tmp102 = 1.0
        tmp103 = tmp101 + tmp102
        tl.store(out_ptr25 + (tl.full([XBLOCK], 0, tl.int32)), tmp103, None)
    elif xpid >= 26 and xpid < 27:
        xpid_offset = xpid - 26
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp104 = tl.load(in_ptr26 + (0))
        tmp105 = tl.broadcast_to(tmp104, [XBLOCK])
        tmp106 = 1.0
        tmp107 = tmp105 + tmp106
        tl.store(out_ptr26 + (tl.full([XBLOCK], 0, tl.int32)), tmp107, None)
    elif xpid >= 27 and xpid < 28:
        xpid_offset = xpid - 27
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp108 = tl.load(in_ptr27 + (0))
        tmp109 = tl.broadcast_to(tmp108, [XBLOCK])
        tmp110 = 1.0
        tmp111 = tmp109 + tmp110
        tl.store(out_ptr27 + (tl.full([XBLOCK], 0, tl.int32)), tmp111, None)
    elif xpid >= 28 and xpid < 29:
        xpid_offset = xpid - 28
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp112 = tl.load(in_ptr28 + (0))
        tmp113 = tl.broadcast_to(tmp112, [XBLOCK])
        tmp114 = 1.0
        tmp115 = tmp113 + tmp114
        tl.store(out_ptr28 + (tl.full([XBLOCK], 0, tl.int32)), tmp115, None)
    elif xpid >= 29 and xpid < 30:
        xpid_offset = xpid - 29
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp116 = tl.load(in_ptr29 + (0))
        tmp117 = tl.broadcast_to(tmp116, [XBLOCK])
        tmp118 = 1.0
        tmp119 = tmp117 + tmp118
        tl.store(out_ptr29 + (tl.full([XBLOCK], 0, tl.int32)), tmp119, None)
    elif xpid >= 30 and xpid < 31:
        xpid_offset = xpid - 30
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp120 = tl.load(in_ptr30 + (0))
        tmp121 = tl.broadcast_to(tmp120, [XBLOCK])
        tmp122 = 1.0
        tmp123 = tmp121 + tmp122
        tl.store(out_ptr30 + (tl.full([XBLOCK], 0, tl.int32)), tmp123, None)
    elif xpid >= 31 and xpid < 32:
        xpid_offset = xpid - 31
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp124 = tl.load(in_ptr31 + (0))
        tmp125 = tl.broadcast_to(tmp124, [XBLOCK])
        tmp126 = 1.0
        tmp127 = tmp125 + tmp126
        tl.store(out_ptr31 + (tl.full([XBLOCK], 0, tl.int32)), tmp127, None)
    elif xpid >= 32 and xpid < 33:
        xpid_offset = xpid - 32
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp128 = tl.load(in_ptr32 + (0))
        tmp129 = tl.broadcast_to(tmp128, [XBLOCK])
        tmp130 = 1.0
        tmp131 = tmp129 + tmp130
        tl.store(out_ptr32 + (tl.full([XBLOCK], 0, tl.int32)), tmp131, None)
    elif xpid >= 33 and xpid < 34:
        xpid_offset = xpid - 33
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp132 = tl.load(in_ptr33 + (0))
        tmp133 = tl.broadcast_to(tmp132, [XBLOCK])
        tmp134 = 1.0
        tmp135 = tmp133 + tmp134
        tl.store(out_ptr33 + (tl.full([XBLOCK], 0, tl.int32)), tmp135, None)
    elif xpid >= 34 and xpid < 35:
        xpid_offset = xpid - 34
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp136 = tl.load(in_ptr34 + (0))
        tmp137 = tl.broadcast_to(tmp136, [XBLOCK])
        tmp138 = 1.0
        tmp139 = tmp137 + tmp138
        tl.store(out_ptr34 + (tl.full([XBLOCK], 0, tl.int32)), tmp139, None)
    elif xpid >= 35 and xpid < 36:
        xpid_offset = xpid - 35
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp140 = tl.load(in_ptr35 + (0))
        tmp141 = tl.broadcast_to(tmp140, [XBLOCK])
        tmp142 = 1.0
        tmp143 = tmp141 + tmp142
        tl.store(out_ptr35 + (tl.full([XBLOCK], 0, tl.int32)), tmp143, None)
    elif xpid >= 36 and xpid < 37:
        xpid_offset = xpid - 36
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp144 = tl.load(in_ptr36 + (0))
        tmp145 = tl.broadcast_to(tmp144, [XBLOCK])
        tmp146 = 1.0
        tmp147 = tmp145 + tmp146
        tl.store(out_ptr36 + (tl.full([XBLOCK], 0, tl.int32)), tmp147, None)
    elif xpid >= 37 and xpid < 38:
        xpid_offset = xpid - 37
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp148 = tl.load(in_ptr37 + (0))
        tmp149 = tl.broadcast_to(tmp148, [XBLOCK])
        tmp150 = 1.0
        tmp151 = tmp149 + tmp150
        tl.store(out_ptr37 + (tl.full([XBLOCK], 0, tl.int32)), tmp151, None)
    elif xpid >= 38 and xpid < 39:
        xpid_offset = xpid - 38
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp152 = tl.load(in_ptr38 + (0))
        tmp153 = tl.broadcast_to(tmp152, [XBLOCK])
        tmp154 = 1.0
        tmp155 = tmp153 + tmp154
        tl.store(out_ptr38 + (tl.full([XBLOCK], 0, tl.int32)), tmp155, None)
    elif xpid >= 39 and xpid < 40:
        xpid_offset = xpid - 39
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp156 = tl.load(in_ptr39 + (0))
        tmp157 = tl.broadcast_to(tmp156, [XBLOCK])
        tmp158 = 1.0
        tmp159 = tmp157 + tmp158
        tl.store(out_ptr39 + (tl.full([XBLOCK], 0, tl.int32)), tmp159, None)
    elif xpid >= 40 and xpid < 41:
        xpid_offset = xpid - 40
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp160 = tl.load(in_ptr40 + (0))
        tmp161 = tl.broadcast_to(tmp160, [XBLOCK])
        tmp162 = 1.0
        tmp163 = tmp161 + tmp162
        tl.store(out_ptr40 + (tl.full([XBLOCK], 0, tl.int32)), tmp163, None)
    elif xpid >= 41 and xpid < 42:
        xpid_offset = xpid - 41
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp164 = tl.load(in_ptr41 + (0))
        tmp165 = tl.broadcast_to(tmp164, [XBLOCK])
        tmp166 = 1.0
        tmp167 = tmp165 + tmp166
        tl.store(out_ptr41 + (tl.full([XBLOCK], 0, tl.int32)), tmp167, None)
    elif xpid >= 42 and xpid < 43:
        xpid_offset = xpid - 42
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp168 = tl.load(in_ptr42 + (0))
        tmp169 = tl.broadcast_to(tmp168, [XBLOCK])
        tmp170 = 1.0
        tmp171 = tmp169 + tmp170
        tl.store(out_ptr42 + (tl.full([XBLOCK], 0, tl.int32)), tmp171, None)
    elif xpid >= 43 and xpid < 44:
        xpid_offset = xpid - 43
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp172 = tl.load(in_ptr43 + (0))
        tmp173 = tl.broadcast_to(tmp172, [XBLOCK])
        tmp174 = 1.0
        tmp175 = tmp173 + tmp174
        tl.store(out_ptr43 + (tl.full([XBLOCK], 0, tl.int32)), tmp175, None)
    elif xpid >= 44 and xpid < 45:
        xpid_offset = xpid - 44
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp176 = tl.load(in_ptr44 + (0))
        tmp177 = tl.broadcast_to(tmp176, [XBLOCK])
        tmp178 = 1.0
        tmp179 = tmp177 + tmp178
        tl.store(out_ptr44 + (tl.full([XBLOCK], 0, tl.int32)), tmp179, None)
    elif xpid >= 45 and xpid < 46:
        xpid_offset = xpid - 45
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp180 = tl.load(in_ptr45 + (0))
        tmp181 = tl.broadcast_to(tmp180, [XBLOCK])
        tmp182 = 1.0
        tmp183 = tmp181 + tmp182
        tl.store(out_ptr45 + (tl.full([XBLOCK], 0, tl.int32)), tmp183, None)
    elif xpid >= 46 and xpid < 47:
        xpid_offset = xpid - 46
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp184 = tl.load(in_ptr46 + (0))
        tmp185 = tl.broadcast_to(tmp184, [XBLOCK])
        tmp186 = 1.0
        tmp187 = tmp185 + tmp186
        tl.store(out_ptr46 + (tl.full([XBLOCK], 0, tl.int32)), tmp187, None)
    elif xpid >= 47 and xpid < 48:
        xpid_offset = xpid - 47
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp188 = tl.load(in_ptr47 + (0))
        tmp189 = tl.broadcast_to(tmp188, [XBLOCK])
        tmp190 = 1.0
        tmp191 = tmp189 + tmp190
        tl.store(out_ptr47 + (tl.full([XBLOCK], 0, tl.int32)), tmp191, None)
    elif xpid >= 48 and xpid < 49:
        xpid_offset = xpid - 48
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp192 = tl.load(in_ptr48 + (0))
        tmp193 = tl.broadcast_to(tmp192, [XBLOCK])
        tmp194 = 1.0
        tmp195 = tmp193 + tmp194
        tl.store(out_ptr48 + (tl.full([XBLOCK], 0, tl.int32)), tmp195, None)
    elif xpid >= 49 and xpid < 50:
        xpid_offset = xpid - 49
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp196 = tl.load(in_ptr49 + (0))
        tmp197 = tl.broadcast_to(tmp196, [XBLOCK])
        tmp198 = 1.0
        tmp199 = tmp197 + tmp198
        tl.store(out_ptr49 + (tl.full([XBLOCK], 0, tl.int32)), tmp199, None)
    elif xpid >= 50 and xpid < 51:
        xpid_offset = xpid - 50
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp200 = tl.load(in_ptr50 + (0))
        tmp201 = tl.broadcast_to(tmp200, [XBLOCK])
        tmp202 = 1.0
        tmp203 = tmp201 + tmp202
        tl.store(out_ptr50 + (tl.full([XBLOCK], 0, tl.int32)), tmp203, None)
    elif xpid >= 51 and xpid < 52:
        xpid_offset = xpid - 51
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp204 = tl.load(in_ptr51 + (0))
        tmp205 = tl.broadcast_to(tmp204, [XBLOCK])
        tmp206 = 1.0
        tmp207 = tmp205 + tmp206
        tl.store(out_ptr51 + (tl.full([XBLOCK], 0, tl.int32)), tmp207, None)
    elif xpid >= 52 and xpid < 53:
        xpid_offset = xpid - 52
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp208 = tl.load(in_ptr52 + (0))
        tmp209 = tl.broadcast_to(tmp208, [XBLOCK])
        tmp210 = 1.0
        tmp211 = tmp209 + tmp210
        tl.store(out_ptr52 + (tl.full([XBLOCK], 0, tl.int32)), tmp211, None)
    elif xpid >= 53 and xpid < 54:
        xpid_offset = xpid - 53
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp212 = tl.load(in_ptr53 + (0))
        tmp213 = tl.broadcast_to(tmp212, [XBLOCK])
        tmp214 = 1.0
        tmp215 = tmp213 + tmp214
        tl.store(out_ptr53 + (tl.full([XBLOCK], 0, tl.int32)), tmp215, None)
    elif xpid >= 54 and xpid < 55:
        xpid_offset = xpid - 54
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp216 = tl.load(in_ptr54 + (0))
        tmp217 = tl.broadcast_to(tmp216, [XBLOCK])
        tmp218 = 1.0
        tmp219 = tmp217 + tmp218
        tl.store(out_ptr54 + (tl.full([XBLOCK], 0, tl.int32)), tmp219, None)
    elif xpid >= 55 and xpid < 56:
        xpid_offset = xpid - 55
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp220 = tl.load(in_ptr55 + (0))
        tmp221 = tl.broadcast_to(tmp220, [XBLOCK])
        tmp222 = 1.0
        tmp223 = tmp221 + tmp222
        tl.store(out_ptr55 + (tl.full([XBLOCK], 0, tl.int32)), tmp223, None)
    elif xpid >= 56 and xpid < 57:
        xpid_offset = xpid - 56
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp224 = tl.load(in_ptr56 + (0))
        tmp225 = tl.broadcast_to(tmp224, [XBLOCK])
        tmp226 = 1.0
        tmp227 = tmp225 + tmp226
        tl.store(out_ptr56 + (tl.full([XBLOCK], 0, tl.int32)), tmp227, None)
    elif xpid >= 57 and xpid < 58:
        xpid_offset = xpid - 57
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp228 = tl.load(in_ptr57 + (0))
        tmp229 = tl.broadcast_to(tmp228, [XBLOCK])
        tmp230 = 1.0
        tmp231 = tmp229 + tmp230
        tl.store(out_ptr57 + (tl.full([XBLOCK], 0, tl.int32)), tmp231, None)
    elif xpid >= 58 and xpid < 59:
        xpid_offset = xpid - 58
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp232 = tl.load(in_ptr58 + (0))
        tmp233 = tl.broadcast_to(tmp232, [XBLOCK])
        tmp234 = 1.0
        tmp235 = tmp233 + tmp234
        tl.store(out_ptr58 + (tl.full([XBLOCK], 0, tl.int32)), tmp235, None)
    elif xpid >= 59 and xpid < 60:
        xpid_offset = xpid - 59
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp236 = tl.load(in_ptr59 + (0))
        tmp237 = tl.broadcast_to(tmp236, [XBLOCK])
        tmp238 = 1.0
        tmp239 = tmp237 + tmp238
        tl.store(out_ptr59 + (tl.full([XBLOCK], 0, tl.int32)), tmp239, None)
    elif xpid >= 60 and xpid < 61:
        xpid_offset = xpid - 60
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp240 = tl.load(in_ptr60 + (0))
        tmp241 = tl.broadcast_to(tmp240, [XBLOCK])
        tmp242 = 1.0
        tmp243 = tmp241 + tmp242
        tl.store(out_ptr60 + (tl.full([XBLOCK], 0, tl.int32)), tmp243, None)
    elif xpid >= 61 and xpid < 62:
        xpid_offset = xpid - 61
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp244 = tl.load(in_ptr61 + (0))
        tmp245 = tl.broadcast_to(tmp244, [XBLOCK])
        tmp246 = 1.0
        tmp247 = tmp245 + tmp246
        tl.store(out_ptr61 + (tl.full([XBLOCK], 0, tl.int32)), tmp247, None)
    elif xpid >= 62 and xpid < 63:
        xpid_offset = xpid - 62
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp248 = tl.load(in_ptr62 + (0))
        tmp249 = tl.broadcast_to(tmp248, [XBLOCK])
        tmp250 = 1.0
        tmp251 = tmp249 + tmp250
        tl.store(out_ptr62 + (tl.full([XBLOCK], 0, tl.int32)), tmp251, None)
    elif xpid >= 63 and xpid < 64:
        xpid_offset = xpid - 63
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp252 = tl.load(in_ptr63 + (0))
        tmp253 = tl.broadcast_to(tmp252, [XBLOCK])
        tmp254 = 1.0
        tmp255 = tmp253 + tmp254
        tl.store(out_ptr63 + (tl.full([XBLOCK], 0, tl.int32)), tmp255, None)
    elif xpid >= 64 and xpid < 65:
        xpid_offset = xpid - 64
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp256 = tl.load(in_ptr64 + (0))
        tmp257 = tl.broadcast_to(tmp256, [XBLOCK])
        tmp258 = 1.0
        tmp259 = tmp257 + tmp258
        tl.store(out_ptr64 + (tl.full([XBLOCK], 0, tl.int32)), tmp259, None)
    elif xpid >= 65 and xpid < 66:
        xpid_offset = xpid - 65
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp260 = tl.load(in_ptr65 + (0))
        tmp261 = tl.broadcast_to(tmp260, [XBLOCK])
        tmp262 = 1.0
        tmp263 = tmp261 + tmp262
        tl.store(out_ptr65 + (tl.full([XBLOCK], 0, tl.int32)), tmp263, None)
    elif xpid >= 66 and xpid < 67:
        xpid_offset = xpid - 66
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp264 = tl.load(in_ptr66 + (0))
        tmp265 = tl.broadcast_to(tmp264, [XBLOCK])
        tmp266 = 1.0
        tmp267 = tmp265 + tmp266
        tl.store(out_ptr66 + (tl.full([XBLOCK], 0, tl.int32)), tmp267, None)
    elif xpid >= 67 and xpid < 68:
        xpid_offset = xpid - 67
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp268 = tl.load(in_ptr67 + (0))
        tmp269 = tl.broadcast_to(tmp268, [XBLOCK])
        tmp270 = 1.0
        tmp271 = tmp269 + tmp270
        tl.store(out_ptr67 + (tl.full([XBLOCK], 0, tl.int32)), tmp271, None)
    elif xpid >= 68 and xpid < 69:
        xpid_offset = xpid - 68
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp272 = tl.load(in_ptr68 + (0))
        tmp273 = tl.broadcast_to(tmp272, [XBLOCK])
        tmp274 = 1.0
        tmp275 = tmp273 + tmp274
        tl.store(out_ptr68 + (tl.full([XBLOCK], 0, tl.int32)), tmp275, None)
    elif xpid >= 69 and xpid < 70:
        xpid_offset = xpid - 69
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp276 = tl.load(in_ptr69 + (0))
        tmp277 = tl.broadcast_to(tmp276, [XBLOCK])
        tmp278 = 1.0
        tmp279 = tmp277 + tmp278
        tl.store(out_ptr69 + (tl.full([XBLOCK], 0, tl.int32)), tmp279, None)
    elif xpid >= 70 and xpid < 71:
        xpid_offset = xpid - 70
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp280 = tl.load(in_ptr70 + (0))
        tmp281 = tl.broadcast_to(tmp280, [XBLOCK])
        tmp282 = 1.0
        tmp283 = tmp281 + tmp282
        tl.store(out_ptr70 + (tl.full([XBLOCK], 0, tl.int32)), tmp283, None)
    elif xpid >= 71 and xpid < 72:
        xpid_offset = xpid - 71
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp284 = tl.load(in_ptr71 + (0))
        tmp285 = tl.broadcast_to(tmp284, [XBLOCK])
        tmp286 = 1.0
        tmp287 = tmp285 + tmp286
        tl.store(out_ptr71 + (tl.full([XBLOCK], 0, tl.int32)), tmp287, None)
    elif xpid >= 72 and xpid < 73:
        xpid_offset = xpid - 72
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp288 = tl.load(in_ptr72 + (0))
        tmp289 = tl.broadcast_to(tmp288, [XBLOCK])
        tmp290 = 1.0
        tmp291 = tmp289 + tmp290
        tl.store(out_ptr72 + (tl.full([XBLOCK], 0, tl.int32)), tmp291, None)
    elif xpid >= 73 and xpid < 74:
        xpid_offset = xpid - 73
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp292 = tl.load(in_ptr73 + (0))
        tmp293 = tl.broadcast_to(tmp292, [XBLOCK])
        tmp294 = 1.0
        tmp295 = tmp293 + tmp294
        tl.store(out_ptr73 + (tl.full([XBLOCK], 0, tl.int32)), tmp295, None)
    elif xpid >= 74 and xpid < 75:
        xpid_offset = xpid - 74
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp296 = tl.load(in_ptr74 + (0))
        tmp297 = tl.broadcast_to(tmp296, [XBLOCK])
        tmp298 = 1.0
        tmp299 = tmp297 + tmp298
        tl.store(out_ptr74 + (tl.full([XBLOCK], 0, tl.int32)), tmp299, None)
    elif xpid >= 75 and xpid < 76:
        xpid_offset = xpid - 75
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp300 = tl.load(in_ptr75 + (0))
        tmp301 = tl.broadcast_to(tmp300, [XBLOCK])
        tmp302 = 1.0
        tmp303 = tmp301 + tmp302
        tl.store(out_ptr75 + (tl.full([XBLOCK], 0, tl.int32)), tmp303, None)
    elif xpid >= 76 and xpid < 77:
        xpid_offset = xpid - 76
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp304 = tl.load(in_ptr76 + (0))
        tmp305 = tl.broadcast_to(tmp304, [XBLOCK])
        tmp306 = 1.0
        tmp307 = tmp305 + tmp306
        tl.store(out_ptr76 + (tl.full([XBLOCK], 0, tl.int32)), tmp307, None)
    elif xpid >= 77 and xpid < 78:
        xpid_offset = xpid - 77
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp308 = tl.load(in_ptr77 + (0))
        tmp309 = tl.broadcast_to(tmp308, [XBLOCK])
        tmp310 = 1.0
        tmp311 = tmp309 + tmp310
        tl.store(out_ptr77 + (tl.full([XBLOCK], 0, tl.int32)), tmp311, None)
    elif xpid >= 78 and xpid < 79:
        xpid_offset = xpid - 78
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp312 = tl.load(in_ptr78 + (0))
        tmp313 = tl.broadcast_to(tmp312, [XBLOCK])
        tmp314 = 1.0
        tmp315 = tmp313 + tmp314
        tl.store(out_ptr78 + (tl.full([XBLOCK], 0, tl.int32)), tmp315, None)
    elif xpid >= 79 and xpid < 80:
        xpid_offset = xpid - 79
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp316 = tl.load(in_ptr79 + (0))
        tmp317 = tl.broadcast_to(tmp316, [XBLOCK])
        tmp318 = 1.0
        tmp319 = tmp317 + tmp318
        tl.store(out_ptr79 + (tl.full([XBLOCK], 0, tl.int32)), tmp319, None)
    elif xpid >= 80 and xpid < 81:
        xpid_offset = xpid - 80
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp320 = tl.load(in_ptr80 + (0))
        tmp321 = tl.broadcast_to(tmp320, [XBLOCK])
        tmp322 = 1.0
        tmp323 = tmp321 + tmp322
        tl.store(out_ptr80 + (tl.full([XBLOCK], 0, tl.int32)), tmp323, None)
    elif xpid >= 81 and xpid < 82:
        xpid_offset = xpid - 81
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp324 = tl.load(in_ptr81 + (0))
        tmp325 = tl.broadcast_to(tmp324, [XBLOCK])
        tmp326 = 1.0
        tmp327 = tmp325 + tmp326
        tl.store(out_ptr81 + (tl.full([XBLOCK], 0, tl.int32)), tmp327, None)
    elif xpid >= 82 and xpid < 83:
        xpid_offset = xpid - 82
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp328 = tl.load(in_ptr82 + (0))
        tmp329 = tl.broadcast_to(tmp328, [XBLOCK])
        tmp330 = 1.0
        tmp331 = tmp329 + tmp330
        tl.store(out_ptr82 + (tl.full([XBLOCK], 0, tl.int32)), tmp331, None)
    else:
        pass
''', device_str='cuda')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, split_scan_grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream


# kernel path: /tmp/torchinductor_zhang402/35/c35hwvsrryxr7z4qzifht7exgpbiuablnb74o25eotxmi7ed6t6s.py
# Source Nodes: [], Original ATen: []

triton_for_fused_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.foreach(
    num_warps=8,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: '*fp32', 35: '*fp32', 36: '*fp32', 37: '*fp32', 38: '*fp32', 39: '*fp32', 40: '*fp32', 41: '*fp32', 42: '*fp32', 43: '*fp32', 44: '*fp32', 45: '*fp32', 46: '*fp32', 47: '*fp32', 48: '*fp32', 49: '*fp32', 50: '*fp32', 51: '*fp32', 52: '*fp32', 53: '*fp32', 54: '*fp32', 55: '*fp32', 56: '*fp32', 57: '*fp32', 58: '*fp32', 59: '*fp32', 60: '*fp32', 61: '*fp32', 62: '*fp32', 63: '*fp32', 64: '*fp32', 65: '*fp32', 66: '*fp32', 67: '*fp32', 68: '*fp32', 69: '*fp32', 70: '*fp32', 71: '*fp32', 72: '*fp32', 73: '*fp32', 74: '*fp32', 75: '*fp32', 76: '*fp32', 77: '*fp32', 78: '*fp32', 79: '*fp32', 80: '*fp32', 81: '*fp32', 82: '*fp32', 83: '*fp32', 84: '*fp32', 85: '*fp32', 86: '*fp32', 87: '*fp32', 88: '*fp32', 89: '*fp32', 90: '*fp32', 91: '*fp32', 92: '*fp32', 93: '*fp32', 94: '*fp32', 95: '*fp32', 96: '*fp32', 97: '*fp32', 98: '*fp32', 99: '*fp32', 100: '*fp32', 101: '*fp32', 102: '*fp32', 103: '*fp32', 104: '*fp32', 105: '*fp32', 106: '*fp32', 107: '*fp32', 108: '*fp32', 109: '*fp32', 110: '*fp32', 111: '*fp32', 112: '*fp32', 113: '*fp32', 114: '*fp32', 115: '*fp32', 116: '*fp32', 117: '*fp32', 118: '*fp32', 119: '*fp32', 120: '*fp32', 121: '*fp32', 122: '*fp32', 123: '*fp32', 124: '*fp32', 125: '*fp32', 126: '*fp32', 127: '*fp32', 128: '*fp32', 129: '*fp32', 130: '*fp32', 131: '*fp32', 132: '*fp32', 133: '*fp32', 134: '*fp32', 135: '*fp32', 136: '*fp32', 137: '*fp32', 138: '*fp32', 139: '*fp32', 140: '*fp32', 141: '*fp32', 142: '*fp32', 143: '*fp32', 144: '*fp32', 145: '*fp32', 146: '*fp32', 147: '*fp32', 148: '*fp32', 149: '*fp32', 150: '*fp32', 151: '*fp32', 152: '*fp32', 153: '*fp32', 154: '*fp32', 155: '*fp32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'kernel_name': 'triton_for_fused_1', 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, in_ptr47, in_ptr48, in_ptr49, in_ptr50, in_ptr51, in_ptr52, in_ptr53, in_ptr54, in_ptr55, in_ptr56, in_ptr57, in_ptr58, in_ptr59, in_ptr60, in_ptr61, in_ptr62, in_ptr63, in_ptr64, in_ptr65, in_ptr66, in_ptr67, in_ptr68, in_ptr69, in_ptr70, in_ptr71, in_ptr72, in_ptr73, in_ptr74, in_ptr75, in_ptr76, in_ptr77, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, out_ptr13, out_ptr14, out_ptr15, out_ptr16, out_ptr17, out_ptr18, out_ptr19, out_ptr20, out_ptr21, out_ptr22, out_ptr23, out_ptr24, out_ptr25, out_ptr26, out_ptr27, out_ptr28, out_ptr29, out_ptr30, out_ptr31, out_ptr32, out_ptr33, out_ptr34, out_ptr35, out_ptr36, out_ptr37, out_ptr38, out_ptr39, out_ptr40, out_ptr41, out_ptr42, out_ptr43, out_ptr44, out_ptr45, out_ptr46, out_ptr47, out_ptr48, out_ptr49, out_ptr50, out_ptr51, out_ptr52, out_ptr53, out_ptr54, out_ptr55, out_ptr56, out_ptr57, out_ptr58, out_ptr59, out_ptr60, out_ptr61, out_ptr62, out_ptr63, out_ptr64, out_ptr65, out_ptr66, out_ptr67, out_ptr68, out_ptr69, out_ptr70, out_ptr71, out_ptr72, out_ptr73, out_ptr74, out_ptr75, out_ptr76, out_ptr77):
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
    elif xpid >= 6 and xpid < 7:
        xpid_offset = xpid - 6
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp24 = tl.load(in_ptr6 + (0))
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK])
        tmp26 = 1.0
        tmp27 = tmp25 + tmp26
        tl.store(out_ptr6 + (tl.full([XBLOCK], 0, tl.int32)), tmp27, None)
    elif xpid >= 7 and xpid < 8:
        xpid_offset = xpid - 7
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp28 = tl.load(in_ptr7 + (0))
        tmp29 = tl.broadcast_to(tmp28, [XBLOCK])
        tmp30 = 1.0
        tmp31 = tmp29 + tmp30
        tl.store(out_ptr7 + (tl.full([XBLOCK], 0, tl.int32)), tmp31, None)
    elif xpid >= 8 and xpid < 9:
        xpid_offset = xpid - 8
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp32 = tl.load(in_ptr8 + (0))
        tmp33 = tl.broadcast_to(tmp32, [XBLOCK])
        tmp34 = 1.0
        tmp35 = tmp33 + tmp34
        tl.store(out_ptr8 + (tl.full([XBLOCK], 0, tl.int32)), tmp35, None)
    elif xpid >= 9 and xpid < 10:
        xpid_offset = xpid - 9
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp36 = tl.load(in_ptr9 + (0))
        tmp37 = tl.broadcast_to(tmp36, [XBLOCK])
        tmp38 = 1.0
        tmp39 = tmp37 + tmp38
        tl.store(out_ptr9 + (tl.full([XBLOCK], 0, tl.int32)), tmp39, None)
    elif xpid >= 10 and xpid < 11:
        xpid_offset = xpid - 10
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp40 = tl.load(in_ptr10 + (0))
        tmp41 = tl.broadcast_to(tmp40, [XBLOCK])
        tmp42 = 1.0
        tmp43 = tmp41 + tmp42
        tl.store(out_ptr10 + (tl.full([XBLOCK], 0, tl.int32)), tmp43, None)
    elif xpid >= 11 and xpid < 12:
        xpid_offset = xpid - 11
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp44 = tl.load(in_ptr11 + (0))
        tmp45 = tl.broadcast_to(tmp44, [XBLOCK])
        tmp46 = 1.0
        tmp47 = tmp45 + tmp46
        tl.store(out_ptr11 + (tl.full([XBLOCK], 0, tl.int32)), tmp47, None)
    elif xpid >= 12 and xpid < 13:
        xpid_offset = xpid - 12
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp48 = tl.load(in_ptr12 + (0))
        tmp49 = tl.broadcast_to(tmp48, [XBLOCK])
        tmp50 = 1.0
        tmp51 = tmp49 + tmp50
        tl.store(out_ptr12 + (tl.full([XBLOCK], 0, tl.int32)), tmp51, None)
    elif xpid >= 13 and xpid < 14:
        xpid_offset = xpid - 13
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp52 = tl.load(in_ptr13 + (0))
        tmp53 = tl.broadcast_to(tmp52, [XBLOCK])
        tmp54 = 1.0
        tmp55 = tmp53 + tmp54
        tl.store(out_ptr13 + (tl.full([XBLOCK], 0, tl.int32)), tmp55, None)
    elif xpid >= 14 and xpid < 15:
        xpid_offset = xpid - 14
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp56 = tl.load(in_ptr14 + (0))
        tmp57 = tl.broadcast_to(tmp56, [XBLOCK])
        tmp58 = 1.0
        tmp59 = tmp57 + tmp58
        tl.store(out_ptr14 + (tl.full([XBLOCK], 0, tl.int32)), tmp59, None)
    elif xpid >= 15 and xpid < 16:
        xpid_offset = xpid - 15
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp60 = tl.load(in_ptr15 + (0))
        tmp61 = tl.broadcast_to(tmp60, [XBLOCK])
        tmp62 = 1.0
        tmp63 = tmp61 + tmp62
        tl.store(out_ptr15 + (tl.full([XBLOCK], 0, tl.int32)), tmp63, None)
    elif xpid >= 16 and xpid < 17:
        xpid_offset = xpid - 16
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp64 = tl.load(in_ptr16 + (0))
        tmp65 = tl.broadcast_to(tmp64, [XBLOCK])
        tmp66 = 1.0
        tmp67 = tmp65 + tmp66
        tl.store(out_ptr16 + (tl.full([XBLOCK], 0, tl.int32)), tmp67, None)
    elif xpid >= 17 and xpid < 18:
        xpid_offset = xpid - 17
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp68 = tl.load(in_ptr17 + (0))
        tmp69 = tl.broadcast_to(tmp68, [XBLOCK])
        tmp70 = 1.0
        tmp71 = tmp69 + tmp70
        tl.store(out_ptr17 + (tl.full([XBLOCK], 0, tl.int32)), tmp71, None)
    elif xpid >= 18 and xpid < 19:
        xpid_offset = xpid - 18
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp72 = tl.load(in_ptr18 + (0))
        tmp73 = tl.broadcast_to(tmp72, [XBLOCK])
        tmp74 = 1.0
        tmp75 = tmp73 + tmp74
        tl.store(out_ptr18 + (tl.full([XBLOCK], 0, tl.int32)), tmp75, None)
    elif xpid >= 19 and xpid < 20:
        xpid_offset = xpid - 19
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp76 = tl.load(in_ptr19 + (0))
        tmp77 = tl.broadcast_to(tmp76, [XBLOCK])
        tmp78 = 1.0
        tmp79 = tmp77 + tmp78
        tl.store(out_ptr19 + (tl.full([XBLOCK], 0, tl.int32)), tmp79, None)
    elif xpid >= 20 and xpid < 21:
        xpid_offset = xpid - 20
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp80 = tl.load(in_ptr20 + (0))
        tmp81 = tl.broadcast_to(tmp80, [XBLOCK])
        tmp82 = 1.0
        tmp83 = tmp81 + tmp82
        tl.store(out_ptr20 + (tl.full([XBLOCK], 0, tl.int32)), tmp83, None)
    elif xpid >= 21 and xpid < 22:
        xpid_offset = xpid - 21
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp84 = tl.load(in_ptr21 + (0))
        tmp85 = tl.broadcast_to(tmp84, [XBLOCK])
        tmp86 = 1.0
        tmp87 = tmp85 + tmp86
        tl.store(out_ptr21 + (tl.full([XBLOCK], 0, tl.int32)), tmp87, None)
    elif xpid >= 22 and xpid < 23:
        xpid_offset = xpid - 22
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp88 = tl.load(in_ptr22 + (0))
        tmp89 = tl.broadcast_to(tmp88, [XBLOCK])
        tmp90 = 1.0
        tmp91 = tmp89 + tmp90
        tl.store(out_ptr22 + (tl.full([XBLOCK], 0, tl.int32)), tmp91, None)
    elif xpid >= 23 and xpid < 24:
        xpid_offset = xpid - 23
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp92 = tl.load(in_ptr23 + (0))
        tmp93 = tl.broadcast_to(tmp92, [XBLOCK])
        tmp94 = 1.0
        tmp95 = tmp93 + tmp94
        tl.store(out_ptr23 + (tl.full([XBLOCK], 0, tl.int32)), tmp95, None)
    elif xpid >= 24 and xpid < 25:
        xpid_offset = xpid - 24
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp96 = tl.load(in_ptr24 + (0))
        tmp97 = tl.broadcast_to(tmp96, [XBLOCK])
        tmp98 = 1.0
        tmp99 = tmp97 + tmp98
        tl.store(out_ptr24 + (tl.full([XBLOCK], 0, tl.int32)), tmp99, None)
    elif xpid >= 25 and xpid < 26:
        xpid_offset = xpid - 25
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp100 = tl.load(in_ptr25 + (0))
        tmp101 = tl.broadcast_to(tmp100, [XBLOCK])
        tmp102 = 1.0
        tmp103 = tmp101 + tmp102
        tl.store(out_ptr25 + (tl.full([XBLOCK], 0, tl.int32)), tmp103, None)
    elif xpid >= 26 and xpid < 27:
        xpid_offset = xpid - 26
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp104 = tl.load(in_ptr26 + (0))
        tmp105 = tl.broadcast_to(tmp104, [XBLOCK])
        tmp106 = 1.0
        tmp107 = tmp105 + tmp106
        tl.store(out_ptr26 + (tl.full([XBLOCK], 0, tl.int32)), tmp107, None)
    elif xpid >= 27 and xpid < 28:
        xpid_offset = xpid - 27
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp108 = tl.load(in_ptr27 + (0))
        tmp109 = tl.broadcast_to(tmp108, [XBLOCK])
        tmp110 = 1.0
        tmp111 = tmp109 + tmp110
        tl.store(out_ptr27 + (tl.full([XBLOCK], 0, tl.int32)), tmp111, None)
    elif xpid >= 28 and xpid < 29:
        xpid_offset = xpid - 28
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp112 = tl.load(in_ptr28 + (0))
        tmp113 = tl.broadcast_to(tmp112, [XBLOCK])
        tmp114 = 1.0
        tmp115 = tmp113 + tmp114
        tl.store(out_ptr28 + (tl.full([XBLOCK], 0, tl.int32)), tmp115, None)
    elif xpid >= 29 and xpid < 30:
        xpid_offset = xpid - 29
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp116 = tl.load(in_ptr29 + (0))
        tmp117 = tl.broadcast_to(tmp116, [XBLOCK])
        tmp118 = 1.0
        tmp119 = tmp117 + tmp118
        tl.store(out_ptr29 + (tl.full([XBLOCK], 0, tl.int32)), tmp119, None)
    elif xpid >= 30 and xpid < 31:
        xpid_offset = xpid - 30
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp120 = tl.load(in_ptr30 + (0))
        tmp121 = tl.broadcast_to(tmp120, [XBLOCK])
        tmp122 = 1.0
        tmp123 = tmp121 + tmp122
        tl.store(out_ptr30 + (tl.full([XBLOCK], 0, tl.int32)), tmp123, None)
    elif xpid >= 31 and xpid < 32:
        xpid_offset = xpid - 31
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp124 = tl.load(in_ptr31 + (0))
        tmp125 = tl.broadcast_to(tmp124, [XBLOCK])
        tmp126 = 1.0
        tmp127 = tmp125 + tmp126
        tl.store(out_ptr31 + (tl.full([XBLOCK], 0, tl.int32)), tmp127, None)
    elif xpid >= 32 and xpid < 33:
        xpid_offset = xpid - 32
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp128 = tl.load(in_ptr32 + (0))
        tmp129 = tl.broadcast_to(tmp128, [XBLOCK])
        tmp130 = 1.0
        tmp131 = tmp129 + tmp130
        tl.store(out_ptr32 + (tl.full([XBLOCK], 0, tl.int32)), tmp131, None)
    elif xpid >= 33 and xpid < 34:
        xpid_offset = xpid - 33
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp132 = tl.load(in_ptr33 + (0))
        tmp133 = tl.broadcast_to(tmp132, [XBLOCK])
        tmp134 = 1.0
        tmp135 = tmp133 + tmp134
        tl.store(out_ptr33 + (tl.full([XBLOCK], 0, tl.int32)), tmp135, None)
    elif xpid >= 34 and xpid < 35:
        xpid_offset = xpid - 34
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp136 = tl.load(in_ptr34 + (0))
        tmp137 = tl.broadcast_to(tmp136, [XBLOCK])
        tmp138 = 1.0
        tmp139 = tmp137 + tmp138
        tl.store(out_ptr34 + (tl.full([XBLOCK], 0, tl.int32)), tmp139, None)
    elif xpid >= 35 and xpid < 36:
        xpid_offset = xpid - 35
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp140 = tl.load(in_ptr35 + (0))
        tmp141 = tl.broadcast_to(tmp140, [XBLOCK])
        tmp142 = 1.0
        tmp143 = tmp141 + tmp142
        tl.store(out_ptr35 + (tl.full([XBLOCK], 0, tl.int32)), tmp143, None)
    elif xpid >= 36 and xpid < 37:
        xpid_offset = xpid - 36
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp144 = tl.load(in_ptr36 + (0))
        tmp145 = tl.broadcast_to(tmp144, [XBLOCK])
        tmp146 = 1.0
        tmp147 = tmp145 + tmp146
        tl.store(out_ptr36 + (tl.full([XBLOCK], 0, tl.int32)), tmp147, None)
    elif xpid >= 37 and xpid < 38:
        xpid_offset = xpid - 37
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp148 = tl.load(in_ptr37 + (0))
        tmp149 = tl.broadcast_to(tmp148, [XBLOCK])
        tmp150 = 1.0
        tmp151 = tmp149 + tmp150
        tl.store(out_ptr37 + (tl.full([XBLOCK], 0, tl.int32)), tmp151, None)
    elif xpid >= 38 and xpid < 39:
        xpid_offset = xpid - 38
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp152 = tl.load(in_ptr38 + (0))
        tmp153 = tl.broadcast_to(tmp152, [XBLOCK])
        tmp154 = 1.0
        tmp155 = tmp153 + tmp154
        tl.store(out_ptr38 + (tl.full([XBLOCK], 0, tl.int32)), tmp155, None)
    elif xpid >= 39 and xpid < 40:
        xpid_offset = xpid - 39
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp156 = tl.load(in_ptr39 + (0))
        tmp157 = tl.broadcast_to(tmp156, [XBLOCK])
        tmp158 = 1.0
        tmp159 = tmp157 + tmp158
        tl.store(out_ptr39 + (tl.full([XBLOCK], 0, tl.int32)), tmp159, None)
    elif xpid >= 40 and xpid < 41:
        xpid_offset = xpid - 40
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp160 = tl.load(in_ptr40 + (0))
        tmp161 = tl.broadcast_to(tmp160, [XBLOCK])
        tmp162 = 1.0
        tmp163 = tmp161 + tmp162
        tl.store(out_ptr40 + (tl.full([XBLOCK], 0, tl.int32)), tmp163, None)
    elif xpid >= 41 and xpid < 42:
        xpid_offset = xpid - 41
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp164 = tl.load(in_ptr41 + (0))
        tmp165 = tl.broadcast_to(tmp164, [XBLOCK])
        tmp166 = 1.0
        tmp167 = tmp165 + tmp166
        tl.store(out_ptr41 + (tl.full([XBLOCK], 0, tl.int32)), tmp167, None)
    elif xpid >= 42 and xpid < 43:
        xpid_offset = xpid - 42
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp168 = tl.load(in_ptr42 + (0))
        tmp169 = tl.broadcast_to(tmp168, [XBLOCK])
        tmp170 = 1.0
        tmp171 = tmp169 + tmp170
        tl.store(out_ptr42 + (tl.full([XBLOCK], 0, tl.int32)), tmp171, None)
    elif xpid >= 43 and xpid < 44:
        xpid_offset = xpid - 43
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp172 = tl.load(in_ptr43 + (0))
        tmp173 = tl.broadcast_to(tmp172, [XBLOCK])
        tmp174 = 1.0
        tmp175 = tmp173 + tmp174
        tl.store(out_ptr43 + (tl.full([XBLOCK], 0, tl.int32)), tmp175, None)
    elif xpid >= 44 and xpid < 45:
        xpid_offset = xpid - 44
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp176 = tl.load(in_ptr44 + (0))
        tmp177 = tl.broadcast_to(tmp176, [XBLOCK])
        tmp178 = 1.0
        tmp179 = tmp177 + tmp178
        tl.store(out_ptr44 + (tl.full([XBLOCK], 0, tl.int32)), tmp179, None)
    elif xpid >= 45 and xpid < 46:
        xpid_offset = xpid - 45
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp180 = tl.load(in_ptr45 + (0))
        tmp181 = tl.broadcast_to(tmp180, [XBLOCK])
        tmp182 = 1.0
        tmp183 = tmp181 + tmp182
        tl.store(out_ptr45 + (tl.full([XBLOCK], 0, tl.int32)), tmp183, None)
    elif xpid >= 46 and xpid < 47:
        xpid_offset = xpid - 46
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp184 = tl.load(in_ptr46 + (0))
        tmp185 = tl.broadcast_to(tmp184, [XBLOCK])
        tmp186 = 1.0
        tmp187 = tmp185 + tmp186
        tl.store(out_ptr46 + (tl.full([XBLOCK], 0, tl.int32)), tmp187, None)
    elif xpid >= 47 and xpid < 48:
        xpid_offset = xpid - 47
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp188 = tl.load(in_ptr47 + (0))
        tmp189 = tl.broadcast_to(tmp188, [XBLOCK])
        tmp190 = 1.0
        tmp191 = tmp189 + tmp190
        tl.store(out_ptr47 + (tl.full([XBLOCK], 0, tl.int32)), tmp191, None)
    elif xpid >= 48 and xpid < 49:
        xpid_offset = xpid - 48
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp192 = tl.load(in_ptr48 + (0))
        tmp193 = tl.broadcast_to(tmp192, [XBLOCK])
        tmp194 = 1.0
        tmp195 = tmp193 + tmp194
        tl.store(out_ptr48 + (tl.full([XBLOCK], 0, tl.int32)), tmp195, None)
    elif xpid >= 49 and xpid < 50:
        xpid_offset = xpid - 49
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp196 = tl.load(in_ptr49 + (0))
        tmp197 = tl.broadcast_to(tmp196, [XBLOCK])
        tmp198 = 1.0
        tmp199 = tmp197 + tmp198
        tl.store(out_ptr49 + (tl.full([XBLOCK], 0, tl.int32)), tmp199, None)
    elif xpid >= 50 and xpid < 51:
        xpid_offset = xpid - 50
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp200 = tl.load(in_ptr50 + (0))
        tmp201 = tl.broadcast_to(tmp200, [XBLOCK])
        tmp202 = 1.0
        tmp203 = tmp201 + tmp202
        tl.store(out_ptr50 + (tl.full([XBLOCK], 0, tl.int32)), tmp203, None)
    elif xpid >= 51 and xpid < 52:
        xpid_offset = xpid - 51
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp204 = tl.load(in_ptr51 + (0))
        tmp205 = tl.broadcast_to(tmp204, [XBLOCK])
        tmp206 = 1.0
        tmp207 = tmp205 + tmp206
        tl.store(out_ptr51 + (tl.full([XBLOCK], 0, tl.int32)), tmp207, None)
    elif xpid >= 52 and xpid < 53:
        xpid_offset = xpid - 52
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp208 = tl.load(in_ptr52 + (0))
        tmp209 = tl.broadcast_to(tmp208, [XBLOCK])
        tmp210 = 1.0
        tmp211 = tmp209 + tmp210
        tl.store(out_ptr52 + (tl.full([XBLOCK], 0, tl.int32)), tmp211, None)
    elif xpid >= 53 and xpid < 54:
        xpid_offset = xpid - 53
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp212 = tl.load(in_ptr53 + (0))
        tmp213 = tl.broadcast_to(tmp212, [XBLOCK])
        tmp214 = 1.0
        tmp215 = tmp213 + tmp214
        tl.store(out_ptr53 + (tl.full([XBLOCK], 0, tl.int32)), tmp215, None)
    elif xpid >= 54 and xpid < 55:
        xpid_offset = xpid - 54
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp216 = tl.load(in_ptr54 + (0))
        tmp217 = tl.broadcast_to(tmp216, [XBLOCK])
        tmp218 = 1.0
        tmp219 = tmp217 + tmp218
        tl.store(out_ptr54 + (tl.full([XBLOCK], 0, tl.int32)), tmp219, None)
    elif xpid >= 55 and xpid < 56:
        xpid_offset = xpid - 55
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp220 = tl.load(in_ptr55 + (0))
        tmp221 = tl.broadcast_to(tmp220, [XBLOCK])
        tmp222 = 1.0
        tmp223 = tmp221 + tmp222
        tl.store(out_ptr55 + (tl.full([XBLOCK], 0, tl.int32)), tmp223, None)
    elif xpid >= 56 and xpid < 57:
        xpid_offset = xpid - 56
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp224 = tl.load(in_ptr56 + (0))
        tmp225 = tl.broadcast_to(tmp224, [XBLOCK])
        tmp226 = 1.0
        tmp227 = tmp225 + tmp226
        tl.store(out_ptr56 + (tl.full([XBLOCK], 0, tl.int32)), tmp227, None)
    elif xpid >= 57 and xpid < 58:
        xpid_offset = xpid - 57
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp228 = tl.load(in_ptr57 + (0))
        tmp229 = tl.broadcast_to(tmp228, [XBLOCK])
        tmp230 = 1.0
        tmp231 = tmp229 + tmp230
        tl.store(out_ptr57 + (tl.full([XBLOCK], 0, tl.int32)), tmp231, None)
    elif xpid >= 58 and xpid < 59:
        xpid_offset = xpid - 58
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp232 = tl.load(in_ptr58 + (0))
        tmp233 = tl.broadcast_to(tmp232, [XBLOCK])
        tmp234 = 1.0
        tmp235 = tmp233 + tmp234
        tl.store(out_ptr58 + (tl.full([XBLOCK], 0, tl.int32)), tmp235, None)
    elif xpid >= 59 and xpid < 60:
        xpid_offset = xpid - 59
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp236 = tl.load(in_ptr59 + (0))
        tmp237 = tl.broadcast_to(tmp236, [XBLOCK])
        tmp238 = 1.0
        tmp239 = tmp237 + tmp238
        tl.store(out_ptr59 + (tl.full([XBLOCK], 0, tl.int32)), tmp239, None)
    elif xpid >= 60 and xpid < 61:
        xpid_offset = xpid - 60
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp240 = tl.load(in_ptr60 + (0))
        tmp241 = tl.broadcast_to(tmp240, [XBLOCK])
        tmp242 = 1.0
        tmp243 = tmp241 + tmp242
        tl.store(out_ptr60 + (tl.full([XBLOCK], 0, tl.int32)), tmp243, None)
    elif xpid >= 61 and xpid < 62:
        xpid_offset = xpid - 61
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp244 = tl.load(in_ptr61 + (0))
        tmp245 = tl.broadcast_to(tmp244, [XBLOCK])
        tmp246 = 1.0
        tmp247 = tmp245 + tmp246
        tl.store(out_ptr61 + (tl.full([XBLOCK], 0, tl.int32)), tmp247, None)
    elif xpid >= 62 and xpid < 63:
        xpid_offset = xpid - 62
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp248 = tl.load(in_ptr62 + (0))
        tmp249 = tl.broadcast_to(tmp248, [XBLOCK])
        tmp250 = 1.0
        tmp251 = tmp249 + tmp250
        tl.store(out_ptr62 + (tl.full([XBLOCK], 0, tl.int32)), tmp251, None)
    elif xpid >= 63 and xpid < 64:
        xpid_offset = xpid - 63
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp252 = tl.load(in_ptr63 + (0))
        tmp253 = tl.broadcast_to(tmp252, [XBLOCK])
        tmp254 = 1.0
        tmp255 = tmp253 + tmp254
        tl.store(out_ptr63 + (tl.full([XBLOCK], 0, tl.int32)), tmp255, None)
    elif xpid >= 64 and xpid < 65:
        xpid_offset = xpid - 64
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp256 = tl.load(in_ptr64 + (0))
        tmp257 = tl.broadcast_to(tmp256, [XBLOCK])
        tmp258 = 1.0
        tmp259 = tmp257 + tmp258
        tl.store(out_ptr64 + (tl.full([XBLOCK], 0, tl.int32)), tmp259, None)
    elif xpid >= 65 and xpid < 66:
        xpid_offset = xpid - 65
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp260 = tl.load(in_ptr65 + (0))
        tmp261 = tl.broadcast_to(tmp260, [XBLOCK])
        tmp262 = 1.0
        tmp263 = tmp261 + tmp262
        tl.store(out_ptr65 + (tl.full([XBLOCK], 0, tl.int32)), tmp263, None)
    elif xpid >= 66 and xpid < 67:
        xpid_offset = xpid - 66
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp264 = tl.load(in_ptr66 + (0))
        tmp265 = tl.broadcast_to(tmp264, [XBLOCK])
        tmp266 = 1.0
        tmp267 = tmp265 + tmp266
        tl.store(out_ptr66 + (tl.full([XBLOCK], 0, tl.int32)), tmp267, None)
    elif xpid >= 67 and xpid < 68:
        xpid_offset = xpid - 67
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp268 = tl.load(in_ptr67 + (0))
        tmp269 = tl.broadcast_to(tmp268, [XBLOCK])
        tmp270 = 1.0
        tmp271 = tmp269 + tmp270
        tl.store(out_ptr67 + (tl.full([XBLOCK], 0, tl.int32)), tmp271, None)
    elif xpid >= 68 and xpid < 69:
        xpid_offset = xpid - 68
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp272 = tl.load(in_ptr68 + (0))
        tmp273 = tl.broadcast_to(tmp272, [XBLOCK])
        tmp274 = 1.0
        tmp275 = tmp273 + tmp274
        tl.store(out_ptr68 + (tl.full([XBLOCK], 0, tl.int32)), tmp275, None)
    elif xpid >= 69 and xpid < 70:
        xpid_offset = xpid - 69
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp276 = tl.load(in_ptr69 + (0))
        tmp277 = tl.broadcast_to(tmp276, [XBLOCK])
        tmp278 = 1.0
        tmp279 = tmp277 + tmp278
        tl.store(out_ptr69 + (tl.full([XBLOCK], 0, tl.int32)), tmp279, None)
    elif xpid >= 70 and xpid < 71:
        xpid_offset = xpid - 70
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp280 = tl.load(in_ptr70 + (0))
        tmp281 = tl.broadcast_to(tmp280, [XBLOCK])
        tmp282 = 1.0
        tmp283 = tmp281 + tmp282
        tl.store(out_ptr70 + (tl.full([XBLOCK], 0, tl.int32)), tmp283, None)
    elif xpid >= 71 and xpid < 72:
        xpid_offset = xpid - 71
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp284 = tl.load(in_ptr71 + (0))
        tmp285 = tl.broadcast_to(tmp284, [XBLOCK])
        tmp286 = 1.0
        tmp287 = tmp285 + tmp286
        tl.store(out_ptr71 + (tl.full([XBLOCK], 0, tl.int32)), tmp287, None)
    elif xpid >= 72 and xpid < 73:
        xpid_offset = xpid - 72
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp288 = tl.load(in_ptr72 + (0))
        tmp289 = tl.broadcast_to(tmp288, [XBLOCK])
        tmp290 = 1.0
        tmp291 = tmp289 + tmp290
        tl.store(out_ptr72 + (tl.full([XBLOCK], 0, tl.int32)), tmp291, None)
    elif xpid >= 73 and xpid < 74:
        xpid_offset = xpid - 73
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp292 = tl.load(in_ptr73 + (0))
        tmp293 = tl.broadcast_to(tmp292, [XBLOCK])
        tmp294 = 1.0
        tmp295 = tmp293 + tmp294
        tl.store(out_ptr73 + (tl.full([XBLOCK], 0, tl.int32)), tmp295, None)
    elif xpid >= 74 and xpid < 75:
        xpid_offset = xpid - 74
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp296 = tl.load(in_ptr74 + (0))
        tmp297 = tl.broadcast_to(tmp296, [XBLOCK])
        tmp298 = 1.0
        tmp299 = tmp297 + tmp298
        tl.store(out_ptr74 + (tl.full([XBLOCK], 0, tl.int32)), tmp299, None)
    elif xpid >= 75 and xpid < 76:
        xpid_offset = xpid - 75
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp300 = tl.load(in_ptr75 + (0))
        tmp301 = tl.broadcast_to(tmp300, [XBLOCK])
        tmp302 = 1.0
        tmp303 = tmp301 + tmp302
        tl.store(out_ptr75 + (tl.full([XBLOCK], 0, tl.int32)), tmp303, None)
    elif xpid >= 76 and xpid < 77:
        xpid_offset = xpid - 76
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp304 = tl.load(in_ptr76 + (0))
        tmp305 = tl.broadcast_to(tmp304, [XBLOCK])
        tmp306 = 1.0
        tmp307 = tmp305 + tmp306
        tl.store(out_ptr76 + (tl.full([XBLOCK], 0, tl.int32)), tmp307, None)
    elif xpid >= 77 and xpid < 78:
        xpid_offset = xpid - 77
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp308 = tl.load(in_ptr77 + (0))
        tmp309 = tl.broadcast_to(tmp308, [XBLOCK])
        tmp310 = 1.0
        tmp311 = tmp309 + tmp310
        tl.store(out_ptr77 + (tl.full([XBLOCK], 0, tl.int32)), tmp311, None)
    else:
        pass
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/yq/cyqybwtcmz273aeushxtjygnwmd37y3iho36j74bsxp6arppaoke.py
# Source Nodes: [], Original ATen: []

triton_for_fused_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.foreach(
    num_warps=8,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: '*fp32', 35: '*fp32', 36: '*fp32', 37: '*fp32', 38: '*fp32', 39: '*fp32', 40: '*fp32', 41: '*fp32', 42: '*fp32', 43: '*fp32', 44: '*fp32', 45: '*fp32', 46: '*fp32', 47: '*fp32', 48: '*fp32', 49: '*fp32', 50: '*fp32', 51: '*fp32', 52: '*fp32', 53: '*fp32', 54: '*fp32', 55: '*fp32', 56: '*fp32', 57: '*fp32', 58: '*fp32', 59: '*fp32', 60: '*fp32', 61: '*fp32', 62: '*fp32', 63: '*fp32', 64: '*fp32', 65: '*fp32', 66: '*fp32', 67: '*fp32', 68: '*fp32', 69: '*fp32', 70: '*fp32', 71: '*fp32', 72: '*fp32', 73: '*fp32', 74: '*fp32', 75: '*fp32', 76: '*fp32', 77: '*fp32', 78: '*fp32', 79: '*fp32', 80: '*fp32', 81: '*fp32', 82: '*fp32', 83: '*fp32', 84: '*fp32', 85: '*fp32', 86: '*fp32', 87: '*fp32', 88: '*fp32', 89: '*fp32', 90: '*fp32', 91: '*fp32', 92: '*fp32', 93: '*fp32', 94: '*fp32', 95: '*fp32', 96: '*fp32', 97: '*fp32', 98: '*fp32', 99: '*fp32', 100: '*fp32', 101: '*fp32', 102: '*fp32', 103: '*fp32', 104: '*fp32', 105: '*fp32', 106: '*fp32', 107: '*fp32', 108: '*fp32', 109: '*fp32', 110: '*fp32', 111: '*fp32', 112: '*fp32', 113: '*fp32', 114: '*fp32', 115: '*fp32', 116: '*fp32', 117: '*fp32', 118: '*fp32', 119: '*fp32', 120: '*fp32', 121: '*fp32', 122: '*fp32', 123: '*fp32', 124: '*fp32', 125: '*fp32', 126: '*fp32', 127: '*fp32', 128: '*fp32', 129: '*fp32', 130: '*fp32', 131: '*fp32', 132: '*fp32', 133: '*fp32', 134: '*fp32', 135: '*fp32', 136: '*fp32', 137: '*fp32', 138: '*fp32', 139: '*fp32', 140: '*fp32', 141: '*fp32', 142: '*fp32', 143: '*fp32', 144: '*fp32', 145: '*fp32', 146: '*fp32', 147: '*fp32', 148: '*fp32', 149: '*fp32', 150: '*fp32', 151: '*fp32', 152: '*fp32', 153: '*fp32', 154: '*fp32', 155: '*fp32', 156: '*fp32', 157: '*fp32', 158: '*fp32', 159: '*fp32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'kernel_name': 'triton_for_fused_2', 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, in_ptr47, in_ptr48, in_ptr49, in_ptr50, in_ptr51, in_ptr52, in_ptr53, in_ptr54, in_ptr55, in_ptr56, in_ptr57, in_ptr58, in_ptr59, in_ptr60, in_ptr61, in_ptr62, in_ptr63, in_ptr64, in_ptr65, in_ptr66, in_ptr67, in_ptr68, in_ptr69, in_ptr70, in_ptr71, in_ptr72, in_ptr73, in_ptr74, in_ptr75, in_ptr76, in_ptr77, in_ptr78, in_ptr79, in_ptr80, in_ptr81, in_ptr82, in_ptr83, in_ptr84, in_ptr85, in_ptr86, in_ptr87, in_ptr88, in_ptr89, in_ptr90, in_ptr91, in_ptr92, in_ptr93, in_ptr94, in_ptr95, in_ptr96, in_ptr97, in_ptr98, in_ptr99, out_ptr0, out_ptr2, out_ptr3, out_ptr4, out_ptr6, out_ptr7, out_ptr8, out_ptr10, out_ptr11, out_ptr12, out_ptr14, out_ptr15, out_ptr16, out_ptr18, out_ptr19, out_ptr20, out_ptr22, out_ptr23, out_ptr24, out_ptr26, out_ptr27, out_ptr28, out_ptr30, out_ptr31, out_ptr32, out_ptr34, out_ptr35, out_ptr36, out_ptr38, out_ptr39, out_ptr40, out_ptr42, out_ptr43, out_ptr44, out_ptr46, out_ptr47, out_ptr48, out_ptr50, out_ptr51, out_ptr52, out_ptr54, out_ptr55, out_ptr56, out_ptr58, out_ptr59, out_ptr60, out_ptr62, out_ptr63, out_ptr64, out_ptr66, out_ptr67, out_ptr68, out_ptr70, out_ptr71, out_ptr72, out_ptr74, out_ptr75, out_ptr76, out_ptr78, out_ptr79):
    xpid = tl.program_id(0)
    XBLOCK: tl.constexpr = 1024
    if xpid >= 0 and xpid < 10:
        xpid_offset = xpid - 0
        xnumel = 9408
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), xmask)
        tmp1 = tl.load(in_ptr1 + (x0), xmask)
        tmp6 = tl.load(in_ptr2 + (x0), xmask)
        tmp13 = tl.load(in_ptr3 + (x0), xmask)
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
        tmp14 = libdevice.sqrt(tmp12)
        tmp17 = libdevice.pow(tmp7, tmp16)
        tmp18 = 1.0
        tmp19 = tmp17 - tmp18
        tmp20 = -tmp19
        tmp21 = libdevice.sqrt(tmp20)
        tmp22 = tmp14 / tmp21
        tmp23 = 1e-08
        tmp24 = tmp22 + tmp23
        tmp25 = 0.9
        tmp26 = libdevice.pow(tmp25, tmp16)
        tmp27 = tmp26 - tmp18
        tmp28 = 0.001
        tmp29 = tmp27 / tmp28
        tmp30 = 1 / tmp29
        tmp31 = tmp24 / tmp30
        tmp32 = tmp5 / tmp31
        tmp33 = tmp13 + tmp32
        tl.store(out_ptr0 + (x0), tmp5, xmask)
        tl.store(out_ptr2 + (x0), tmp33, xmask)
        tl.store(out_ptr3 + (x0), tmp12, xmask)
    elif xpid >= 10 and xpid < 11:
        xpid_offset = xpid - 10
        xnumel = 64
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
        tmp48 = libdevice.sqrt(tmp46)
        tmp51 = libdevice.pow(tmp41, tmp50)
        tmp52 = 1.0
        tmp53 = tmp51 - tmp52
        tmp54 = -tmp53
        tmp55 = libdevice.sqrt(tmp54)
        tmp56 = tmp48 / tmp55
        tmp57 = 1e-08
        tmp58 = tmp56 + tmp57
        tmp59 = 0.9
        tmp60 = libdevice.pow(tmp59, tmp50)
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
    elif xpid >= 11 and xpid < 12:
        xpid_offset = xpid - 11
        xnumel = 64
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x2 = xindex
        tmp68 = tl.load(in_ptr10 + (x2), xmask)
        tmp69 = tl.load(in_ptr11 + (x2), xmask)
        tmp74 = tl.load(in_ptr12 + (x2), xmask)
        tmp81 = tl.load(in_ptr13 + (x2), xmask)
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
        tmp82 = libdevice.sqrt(tmp80)
        tmp85 = libdevice.pow(tmp75, tmp84)
        tmp86 = 1.0
        tmp87 = tmp85 - tmp86
        tmp88 = -tmp87
        tmp89 = libdevice.sqrt(tmp88)
        tmp90 = tmp82 / tmp89
        tmp91 = 1e-08
        tmp92 = tmp90 + tmp91
        tmp93 = 0.9
        tmp94 = libdevice.pow(tmp93, tmp84)
        tmp95 = tmp94 - tmp86
        tmp96 = 0.001
        tmp97 = tmp95 / tmp96
        tmp98 = 1 / tmp97
        tmp99 = tmp92 / tmp98
        tmp100 = tmp73 / tmp99
        tmp101 = tmp81 + tmp100
        tl.store(out_ptr8 + (x2), tmp73, xmask)
        tl.store(out_ptr10 + (x2), tmp101, xmask)
        tl.store(out_ptr11 + (x2), tmp80, xmask)
    elif xpid >= 12 and xpid < 16:
        xpid_offset = xpid - 12
        xnumel = 4096
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x3 = xindex
        tmp102 = tl.load(in_ptr15 + (x3), None)
        tmp103 = tl.load(in_ptr16 + (x3), None)
        tmp108 = tl.load(in_ptr17 + (x3), None)
        tmp115 = tl.load(in_ptr18 + (x3), None)
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
        tmp116 = libdevice.sqrt(tmp114)
        tmp119 = libdevice.pow(tmp109, tmp118)
        tmp120 = 1.0
        tmp121 = tmp119 - tmp120
        tmp122 = -tmp121
        tmp123 = libdevice.sqrt(tmp122)
        tmp124 = tmp116 / tmp123
        tmp125 = 1e-08
        tmp126 = tmp124 + tmp125
        tmp127 = 0.9
        tmp128 = libdevice.pow(tmp127, tmp118)
        tmp129 = tmp128 - tmp120
        tmp130 = 0.001
        tmp131 = tmp129 / tmp130
        tmp132 = 1 / tmp131
        tmp133 = tmp126 / tmp132
        tmp134 = tmp107 / tmp133
        tmp135 = tmp115 + tmp134
        tl.store(out_ptr12 + (x3), tmp107, None)
        tl.store(out_ptr14 + (x3), tmp135, None)
        tl.store(out_ptr15 + (x3), tmp114, None)
    elif xpid >= 16 and xpid < 17:
        xpid_offset = xpid - 16
        xnumel = 64
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
        tmp150 = libdevice.sqrt(tmp148)
        tmp153 = libdevice.pow(tmp143, tmp152)
        tmp154 = 1.0
        tmp155 = tmp153 - tmp154
        tmp156 = -tmp155
        tmp157 = libdevice.sqrt(tmp156)
        tmp158 = tmp150 / tmp157
        tmp159 = 1e-08
        tmp160 = tmp158 + tmp159
        tmp161 = 0.9
        tmp162 = libdevice.pow(tmp161, tmp152)
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
    elif xpid >= 17 and xpid < 18:
        xpid_offset = xpid - 17
        xnumel = 64
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
        tmp184 = libdevice.sqrt(tmp182)
        tmp187 = libdevice.pow(tmp177, tmp186)
        tmp188 = 1.0
        tmp189 = tmp187 - tmp188
        tmp190 = -tmp189
        tmp191 = libdevice.sqrt(tmp190)
        tmp192 = tmp184 / tmp191
        tmp193 = 1e-08
        tmp194 = tmp192 + tmp193
        tmp195 = 0.9
        tmp196 = libdevice.pow(tmp195, tmp186)
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
    elif xpid >= 18 and xpid < 54:
        xpid_offset = xpid - 18
        xnumel = 36864
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x6 = xindex
        tmp204 = tl.load(in_ptr30 + (x6), None)
        tmp205 = tl.load(in_ptr31 + (x6), None)
        tmp210 = tl.load(in_ptr32 + (x6), None)
        tmp217 = tl.load(in_ptr33 + (x6), None)
        tmp219 = tl.load(in_ptr34 + (0))
        tmp220 = tl.broadcast_to(tmp219, [XBLOCK])
        tmp206 = tmp205 - tmp204
        tmp207 = 0.09999999999999998
        tmp208 = tmp206 * tmp207
        tmp209 = tmp204 + tmp208
        tmp211 = 0.999
        tmp212 = tmp210 * tmp211
        tmp213 = tmp205 * tmp205
        tmp214 = 0.0010000000000000009
        tmp215 = tmp213 * tmp214
        tmp216 = tmp212 + tmp215
        tmp218 = libdevice.sqrt(tmp216)
        tmp221 = libdevice.pow(tmp211, tmp220)
        tmp222 = 1.0
        tmp223 = tmp221 - tmp222
        tmp224 = -tmp223
        tmp225 = libdevice.sqrt(tmp224)
        tmp226 = tmp218 / tmp225
        tmp227 = 1e-08
        tmp228 = tmp226 + tmp227
        tmp229 = 0.9
        tmp230 = libdevice.pow(tmp229, tmp220)
        tmp231 = tmp230 - tmp222
        tmp232 = 0.001
        tmp233 = tmp231 / tmp232
        tmp234 = 1 / tmp233
        tmp235 = tmp228 / tmp234
        tmp236 = tmp209 / tmp235
        tmp237 = tmp217 + tmp236
        tl.store(out_ptr24 + (x6), tmp209, None)
        tl.store(out_ptr26 + (x6), tmp237, None)
        tl.store(out_ptr27 + (x6), tmp216, None)
    elif xpid >= 54 and xpid < 55:
        xpid_offset = xpid - 54
        xnumel = 64
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x7 = xindex
        tmp238 = tl.load(in_ptr35 + (x7), xmask)
        tmp239 = tl.load(in_ptr36 + (x7), xmask)
        tmp244 = tl.load(in_ptr37 + (x7), xmask)
        tmp251 = tl.load(in_ptr38 + (x7), xmask)
        tmp253 = tl.load(in_ptr39 + (0))
        tmp254 = tl.broadcast_to(tmp253, [XBLOCK])
        tmp240 = tmp239 - tmp238
        tmp241 = 0.09999999999999998
        tmp242 = tmp240 * tmp241
        tmp243 = tmp238 + tmp242
        tmp245 = 0.999
        tmp246 = tmp244 * tmp245
        tmp247 = tmp239 * tmp239
        tmp248 = 0.0010000000000000009
        tmp249 = tmp247 * tmp248
        tmp250 = tmp246 + tmp249
        tmp252 = libdevice.sqrt(tmp250)
        tmp255 = libdevice.pow(tmp245, tmp254)
        tmp256 = 1.0
        tmp257 = tmp255 - tmp256
        tmp258 = -tmp257
        tmp259 = libdevice.sqrt(tmp258)
        tmp260 = tmp252 / tmp259
        tmp261 = 1e-08
        tmp262 = tmp260 + tmp261
        tmp263 = 0.9
        tmp264 = libdevice.pow(tmp263, tmp254)
        tmp265 = tmp264 - tmp256
        tmp266 = 0.001
        tmp267 = tmp265 / tmp266
        tmp268 = 1 / tmp267
        tmp269 = tmp262 / tmp268
        tmp270 = tmp243 / tmp269
        tmp271 = tmp251 + tmp270
        tl.store(out_ptr28 + (x7), tmp243, xmask)
        tl.store(out_ptr30 + (x7), tmp271, xmask)
        tl.store(out_ptr31 + (x7), tmp250, xmask)
    elif xpid >= 55 and xpid < 56:
        xpid_offset = xpid - 55
        xnumel = 64
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x8 = xindex
        tmp272 = tl.load(in_ptr40 + (x8), xmask)
        tmp273 = tl.load(in_ptr41 + (x8), xmask)
        tmp278 = tl.load(in_ptr42 + (x8), xmask)
        tmp285 = tl.load(in_ptr43 + (x8), xmask)
        tmp287 = tl.load(in_ptr44 + (0))
        tmp288 = tl.broadcast_to(tmp287, [XBLOCK])
        tmp274 = tmp273 - tmp272
        tmp275 = 0.09999999999999998
        tmp276 = tmp274 * tmp275
        tmp277 = tmp272 + tmp276
        tmp279 = 0.999
        tmp280 = tmp278 * tmp279
        tmp281 = tmp273 * tmp273
        tmp282 = 0.0010000000000000009
        tmp283 = tmp281 * tmp282
        tmp284 = tmp280 + tmp283
        tmp286 = libdevice.sqrt(tmp284)
        tmp289 = libdevice.pow(tmp279, tmp288)
        tmp290 = 1.0
        tmp291 = tmp289 - tmp290
        tmp292 = -tmp291
        tmp293 = libdevice.sqrt(tmp292)
        tmp294 = tmp286 / tmp293
        tmp295 = 1e-08
        tmp296 = tmp294 + tmp295
        tmp297 = 0.9
        tmp298 = libdevice.pow(tmp297, tmp288)
        tmp299 = tmp298 - tmp290
        tmp300 = 0.001
        tmp301 = tmp299 / tmp300
        tmp302 = 1 / tmp301
        tmp303 = tmp296 / tmp302
        tmp304 = tmp277 / tmp303
        tmp305 = tmp285 + tmp304
        tl.store(out_ptr32 + (x8), tmp277, xmask)
        tl.store(out_ptr34 + (x8), tmp305, xmask)
        tl.store(out_ptr35 + (x8), tmp284, xmask)
    elif xpid >= 56 and xpid < 72:
        xpid_offset = xpid - 56
        xnumel = 16384
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x9 = xindex
        tmp306 = tl.load(in_ptr45 + (x9), None)
        tmp307 = tl.load(in_ptr46 + (x9), None)
        tmp312 = tl.load(in_ptr47 + (x9), None)
        tmp319 = tl.load(in_ptr48 + (x9), None)
        tmp321 = tl.load(in_ptr49 + (0))
        tmp322 = tl.broadcast_to(tmp321, [XBLOCK])
        tmp308 = tmp307 - tmp306
        tmp309 = 0.09999999999999998
        tmp310 = tmp308 * tmp309
        tmp311 = tmp306 + tmp310
        tmp313 = 0.999
        tmp314 = tmp312 * tmp313
        tmp315 = tmp307 * tmp307
        tmp316 = 0.0010000000000000009
        tmp317 = tmp315 * tmp316
        tmp318 = tmp314 + tmp317
        tmp320 = libdevice.sqrt(tmp318)
        tmp323 = libdevice.pow(tmp313, tmp322)
        tmp324 = 1.0
        tmp325 = tmp323 - tmp324
        tmp326 = -tmp325
        tmp327 = libdevice.sqrt(tmp326)
        tmp328 = tmp320 / tmp327
        tmp329 = 1e-08
        tmp330 = tmp328 + tmp329
        tmp331 = 0.9
        tmp332 = libdevice.pow(tmp331, tmp322)
        tmp333 = tmp332 - tmp324
        tmp334 = 0.001
        tmp335 = tmp333 / tmp334
        tmp336 = 1 / tmp335
        tmp337 = tmp330 / tmp336
        tmp338 = tmp311 / tmp337
        tmp339 = tmp319 + tmp338
        tl.store(out_ptr36 + (x9), tmp311, None)
        tl.store(out_ptr38 + (x9), tmp339, None)
        tl.store(out_ptr39 + (x9), tmp318, None)
    elif xpid >= 72 and xpid < 73:
        xpid_offset = xpid - 72
        xnumel = 256
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x10 = xindex
        tmp340 = tl.load(in_ptr50 + (x10), xmask)
        tmp341 = tl.load(in_ptr51 + (x10), xmask)
        tmp346 = tl.load(in_ptr52 + (x10), xmask)
        tmp353 = tl.load(in_ptr53 + (x10), xmask)
        tmp355 = tl.load(in_ptr54 + (0))
        tmp356 = tl.broadcast_to(tmp355, [XBLOCK])
        tmp342 = tmp341 - tmp340
        tmp343 = 0.09999999999999998
        tmp344 = tmp342 * tmp343
        tmp345 = tmp340 + tmp344
        tmp347 = 0.999
        tmp348 = tmp346 * tmp347
        tmp349 = tmp341 * tmp341
        tmp350 = 0.0010000000000000009
        tmp351 = tmp349 * tmp350
        tmp352 = tmp348 + tmp351
        tmp354 = libdevice.sqrt(tmp352)
        tmp357 = libdevice.pow(tmp347, tmp356)
        tmp358 = 1.0
        tmp359 = tmp357 - tmp358
        tmp360 = -tmp359
        tmp361 = libdevice.sqrt(tmp360)
        tmp362 = tmp354 / tmp361
        tmp363 = 1e-08
        tmp364 = tmp362 + tmp363
        tmp365 = 0.9
        tmp366 = libdevice.pow(tmp365, tmp356)
        tmp367 = tmp366 - tmp358
        tmp368 = 0.001
        tmp369 = tmp367 / tmp368
        tmp370 = 1 / tmp369
        tmp371 = tmp364 / tmp370
        tmp372 = tmp345 / tmp371
        tmp373 = tmp353 + tmp372
        tl.store(out_ptr40 + (x10), tmp345, xmask)
        tl.store(out_ptr42 + (x10), tmp373, xmask)
        tl.store(out_ptr43 + (x10), tmp352, xmask)
    elif xpid >= 73 and xpid < 74:
        xpid_offset = xpid - 73
        xnumel = 256
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x11 = xindex
        tmp374 = tl.load(in_ptr55 + (x11), xmask)
        tmp375 = tl.load(in_ptr56 + (x11), xmask)
        tmp380 = tl.load(in_ptr57 + (x11), xmask)
        tmp387 = tl.load(in_ptr58 + (x11), xmask)
        tmp389 = tl.load(in_ptr59 + (0))
        tmp390 = tl.broadcast_to(tmp389, [XBLOCK])
        tmp376 = tmp375 - tmp374
        tmp377 = 0.09999999999999998
        tmp378 = tmp376 * tmp377
        tmp379 = tmp374 + tmp378
        tmp381 = 0.999
        tmp382 = tmp380 * tmp381
        tmp383 = tmp375 * tmp375
        tmp384 = 0.0010000000000000009
        tmp385 = tmp383 * tmp384
        tmp386 = tmp382 + tmp385
        tmp388 = libdevice.sqrt(tmp386)
        tmp391 = libdevice.pow(tmp381, tmp390)
        tmp392 = 1.0
        tmp393 = tmp391 - tmp392
        tmp394 = -tmp393
        tmp395 = libdevice.sqrt(tmp394)
        tmp396 = tmp388 / tmp395
        tmp397 = 1e-08
        tmp398 = tmp396 + tmp397
        tmp399 = 0.9
        tmp400 = libdevice.pow(tmp399, tmp390)
        tmp401 = tmp400 - tmp392
        tmp402 = 0.001
        tmp403 = tmp401 / tmp402
        tmp404 = 1 / tmp403
        tmp405 = tmp398 / tmp404
        tmp406 = tmp379 / tmp405
        tmp407 = tmp387 + tmp406
        tl.store(out_ptr44 + (x11), tmp379, xmask)
        tl.store(out_ptr46 + (x11), tmp407, xmask)
        tl.store(out_ptr47 + (x11), tmp386, xmask)
    elif xpid >= 74 and xpid < 90:
        xpid_offset = xpid - 74
        xnumel = 16384
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x12 = xindex
        tmp408 = tl.load(in_ptr60 + (x12), None)
        tmp409 = tl.load(in_ptr61 + (x12), None)
        tmp414 = tl.load(in_ptr62 + (x12), None)
        tmp421 = tl.load(in_ptr63 + (x12), None)
        tmp423 = tl.load(in_ptr64 + (0))
        tmp424 = tl.broadcast_to(tmp423, [XBLOCK])
        tmp410 = tmp409 - tmp408
        tmp411 = 0.09999999999999998
        tmp412 = tmp410 * tmp411
        tmp413 = tmp408 + tmp412
        tmp415 = 0.999
        tmp416 = tmp414 * tmp415
        tmp417 = tmp409 * tmp409
        tmp418 = 0.0010000000000000009
        tmp419 = tmp417 * tmp418
        tmp420 = tmp416 + tmp419
        tmp422 = libdevice.sqrt(tmp420)
        tmp425 = libdevice.pow(tmp415, tmp424)
        tmp426 = 1.0
        tmp427 = tmp425 - tmp426
        tmp428 = -tmp427
        tmp429 = libdevice.sqrt(tmp428)
        tmp430 = tmp422 / tmp429
        tmp431 = 1e-08
        tmp432 = tmp430 + tmp431
        tmp433 = 0.9
        tmp434 = libdevice.pow(tmp433, tmp424)
        tmp435 = tmp434 - tmp426
        tmp436 = 0.001
        tmp437 = tmp435 / tmp436
        tmp438 = 1 / tmp437
        tmp439 = tmp432 / tmp438
        tmp440 = tmp413 / tmp439
        tmp441 = tmp421 + tmp440
        tl.store(out_ptr48 + (x12), tmp413, None)
        tl.store(out_ptr50 + (x12), tmp441, None)
        tl.store(out_ptr51 + (x12), tmp420, None)
    elif xpid >= 90 and xpid < 91:
        xpid_offset = xpid - 90
        xnumel = 256
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x13 = xindex
        tmp442 = tl.load(in_ptr65 + (x13), xmask)
        tmp443 = tl.load(in_ptr66 + (x13), xmask)
        tmp448 = tl.load(in_ptr67 + (x13), xmask)
        tmp455 = tl.load(in_ptr68 + (x13), xmask)
        tmp457 = tl.load(in_ptr69 + (0))
        tmp458 = tl.broadcast_to(tmp457, [XBLOCK])
        tmp444 = tmp443 - tmp442
        tmp445 = 0.09999999999999998
        tmp446 = tmp444 * tmp445
        tmp447 = tmp442 + tmp446
        tmp449 = 0.999
        tmp450 = tmp448 * tmp449
        tmp451 = tmp443 * tmp443
        tmp452 = 0.0010000000000000009
        tmp453 = tmp451 * tmp452
        tmp454 = tmp450 + tmp453
        tmp456 = libdevice.sqrt(tmp454)
        tmp459 = libdevice.pow(tmp449, tmp458)
        tmp460 = 1.0
        tmp461 = tmp459 - tmp460
        tmp462 = -tmp461
        tmp463 = libdevice.sqrt(tmp462)
        tmp464 = tmp456 / tmp463
        tmp465 = 1e-08
        tmp466 = tmp464 + tmp465
        tmp467 = 0.9
        tmp468 = libdevice.pow(tmp467, tmp458)
        tmp469 = tmp468 - tmp460
        tmp470 = 0.001
        tmp471 = tmp469 / tmp470
        tmp472 = 1 / tmp471
        tmp473 = tmp466 / tmp472
        tmp474 = tmp447 / tmp473
        tmp475 = tmp455 + tmp474
        tl.store(out_ptr52 + (x13), tmp447, xmask)
        tl.store(out_ptr54 + (x13), tmp475, xmask)
        tl.store(out_ptr55 + (x13), tmp454, xmask)
    elif xpid >= 91 and xpid < 92:
        xpid_offset = xpid - 91
        xnumel = 256
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x14 = xindex
        tmp476 = tl.load(in_ptr70 + (x14), xmask)
        tmp477 = tl.load(in_ptr71 + (x14), xmask)
        tmp482 = tl.load(in_ptr72 + (x14), xmask)
        tmp489 = tl.load(in_ptr73 + (x14), xmask)
        tmp491 = tl.load(in_ptr74 + (0))
        tmp492 = tl.broadcast_to(tmp491, [XBLOCK])
        tmp478 = tmp477 - tmp476
        tmp479 = 0.09999999999999998
        tmp480 = tmp478 * tmp479
        tmp481 = tmp476 + tmp480
        tmp483 = 0.999
        tmp484 = tmp482 * tmp483
        tmp485 = tmp477 * tmp477
        tmp486 = 0.0010000000000000009
        tmp487 = tmp485 * tmp486
        tmp488 = tmp484 + tmp487
        tmp490 = libdevice.sqrt(tmp488)
        tmp493 = libdevice.pow(tmp483, tmp492)
        tmp494 = 1.0
        tmp495 = tmp493 - tmp494
        tmp496 = -tmp495
        tmp497 = libdevice.sqrt(tmp496)
        tmp498 = tmp490 / tmp497
        tmp499 = 1e-08
        tmp500 = tmp498 + tmp499
        tmp501 = 0.9
        tmp502 = libdevice.pow(tmp501, tmp492)
        tmp503 = tmp502 - tmp494
        tmp504 = 0.001
        tmp505 = tmp503 / tmp504
        tmp506 = 1 / tmp505
        tmp507 = tmp500 / tmp506
        tmp508 = tmp481 / tmp507
        tmp509 = tmp489 + tmp508
        tl.store(out_ptr56 + (x14), tmp481, xmask)
        tl.store(out_ptr58 + (x14), tmp509, xmask)
        tl.store(out_ptr59 + (x14), tmp488, xmask)
    elif xpid >= 92 and xpid < 108:
        xpid_offset = xpid - 92
        xnumel = 16384
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x15 = xindex
        tmp510 = tl.load(in_ptr75 + (x15), None)
        tmp511 = tl.load(in_ptr76 + (x15), None)
        tmp516 = tl.load(in_ptr77 + (x15), None)
        tmp523 = tl.load(in_ptr78 + (x15), None)
        tmp525 = tl.load(in_ptr79 + (0))
        tmp526 = tl.broadcast_to(tmp525, [XBLOCK])
        tmp512 = tmp511 - tmp510
        tmp513 = 0.09999999999999998
        tmp514 = tmp512 * tmp513
        tmp515 = tmp510 + tmp514
        tmp517 = 0.999
        tmp518 = tmp516 * tmp517
        tmp519 = tmp511 * tmp511
        tmp520 = 0.0010000000000000009
        tmp521 = tmp519 * tmp520
        tmp522 = tmp518 + tmp521
        tmp524 = libdevice.sqrt(tmp522)
        tmp527 = libdevice.pow(tmp517, tmp526)
        tmp528 = 1.0
        tmp529 = tmp527 - tmp528
        tmp530 = -tmp529
        tmp531 = libdevice.sqrt(tmp530)
        tmp532 = tmp524 / tmp531
        tmp533 = 1e-08
        tmp534 = tmp532 + tmp533
        tmp535 = 0.9
        tmp536 = libdevice.pow(tmp535, tmp526)
        tmp537 = tmp536 - tmp528
        tmp538 = 0.001
        tmp539 = tmp537 / tmp538
        tmp540 = 1 / tmp539
        tmp541 = tmp534 / tmp540
        tmp542 = tmp515 / tmp541
        tmp543 = tmp523 + tmp542
        tl.store(out_ptr60 + (x15), tmp515, None)
        tl.store(out_ptr62 + (x15), tmp543, None)
        tl.store(out_ptr63 + (x15), tmp522, None)
    elif xpid >= 108 and xpid < 109:
        xpid_offset = xpid - 108
        xnumel = 64
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x16 = xindex
        tmp544 = tl.load(in_ptr80 + (x16), xmask)
        tmp545 = tl.load(in_ptr81 + (x16), xmask)
        tmp550 = tl.load(in_ptr82 + (x16), xmask)
        tmp557 = tl.load(in_ptr83 + (x16), xmask)
        tmp559 = tl.load(in_ptr84 + (0))
        tmp560 = tl.broadcast_to(tmp559, [XBLOCK])
        tmp546 = tmp545 - tmp544
        tmp547 = 0.09999999999999998
        tmp548 = tmp546 * tmp547
        tmp549 = tmp544 + tmp548
        tmp551 = 0.999
        tmp552 = tmp550 * tmp551
        tmp553 = tmp545 * tmp545
        tmp554 = 0.0010000000000000009
        tmp555 = tmp553 * tmp554
        tmp556 = tmp552 + tmp555
        tmp558 = libdevice.sqrt(tmp556)
        tmp561 = libdevice.pow(tmp551, tmp560)
        tmp562 = 1.0
        tmp563 = tmp561 - tmp562
        tmp564 = -tmp563
        tmp565 = libdevice.sqrt(tmp564)
        tmp566 = tmp558 / tmp565
        tmp567 = 1e-08
        tmp568 = tmp566 + tmp567
        tmp569 = 0.9
        tmp570 = libdevice.pow(tmp569, tmp560)
        tmp571 = tmp570 - tmp562
        tmp572 = 0.001
        tmp573 = tmp571 / tmp572
        tmp574 = 1 / tmp573
        tmp575 = tmp568 / tmp574
        tmp576 = tmp549 / tmp575
        tmp577 = tmp557 + tmp576
        tl.store(out_ptr64 + (x16), tmp549, xmask)
        tl.store(out_ptr66 + (x16), tmp577, xmask)
        tl.store(out_ptr67 + (x16), tmp556, xmask)
    elif xpid >= 109 and xpid < 110:
        xpid_offset = xpid - 109
        xnumel = 64
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x17 = xindex
        tmp578 = tl.load(in_ptr85 + (x17), xmask)
        tmp579 = tl.load(in_ptr86 + (x17), xmask)
        tmp584 = tl.load(in_ptr87 + (x17), xmask)
        tmp591 = tl.load(in_ptr88 + (x17), xmask)
        tmp593 = tl.load(in_ptr89 + (0))
        tmp594 = tl.broadcast_to(tmp593, [XBLOCK])
        tmp580 = tmp579 - tmp578
        tmp581 = 0.09999999999999998
        tmp582 = tmp580 * tmp581
        tmp583 = tmp578 + tmp582
        tmp585 = 0.999
        tmp586 = tmp584 * tmp585
        tmp587 = tmp579 * tmp579
        tmp588 = 0.0010000000000000009
        tmp589 = tmp587 * tmp588
        tmp590 = tmp586 + tmp589
        tmp592 = libdevice.sqrt(tmp590)
        tmp595 = libdevice.pow(tmp585, tmp594)
        tmp596 = 1.0
        tmp597 = tmp595 - tmp596
        tmp598 = -tmp597
        tmp599 = libdevice.sqrt(tmp598)
        tmp600 = tmp592 / tmp599
        tmp601 = 1e-08
        tmp602 = tmp600 + tmp601
        tmp603 = 0.9
        tmp604 = libdevice.pow(tmp603, tmp594)
        tmp605 = tmp604 - tmp596
        tmp606 = 0.001
        tmp607 = tmp605 / tmp606
        tmp608 = 1 / tmp607
        tmp609 = tmp602 / tmp608
        tmp610 = tmp583 / tmp609
        tmp611 = tmp591 + tmp610
        tl.store(out_ptr68 + (x17), tmp583, xmask)
        tl.store(out_ptr70 + (x17), tmp611, xmask)
        tl.store(out_ptr71 + (x17), tmp590, xmask)
    elif xpid >= 110 and xpid < 146:
        xpid_offset = xpid - 110
        xnumel = 36864
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x18 = xindex
        tmp612 = tl.load(in_ptr90 + (x18), None)
        tmp613 = tl.load(in_ptr91 + (x18), None)
        tmp618 = tl.load(in_ptr92 + (x18), None)
        tmp625 = tl.load(in_ptr93 + (x18), None)
        tmp627 = tl.load(in_ptr94 + (0))
        tmp628 = tl.broadcast_to(tmp627, [XBLOCK])
        tmp614 = tmp613 - tmp612
        tmp615 = 0.09999999999999998
        tmp616 = tmp614 * tmp615
        tmp617 = tmp612 + tmp616
        tmp619 = 0.999
        tmp620 = tmp618 * tmp619
        tmp621 = tmp613 * tmp613
        tmp622 = 0.0010000000000000009
        tmp623 = tmp621 * tmp622
        tmp624 = tmp620 + tmp623
        tmp626 = libdevice.sqrt(tmp624)
        tmp629 = libdevice.pow(tmp619, tmp628)
        tmp630 = 1.0
        tmp631 = tmp629 - tmp630
        tmp632 = -tmp631
        tmp633 = libdevice.sqrt(tmp632)
        tmp634 = tmp626 / tmp633
        tmp635 = 1e-08
        tmp636 = tmp634 + tmp635
        tmp637 = 0.9
        tmp638 = libdevice.pow(tmp637, tmp628)
        tmp639 = tmp638 - tmp630
        tmp640 = 0.001
        tmp641 = tmp639 / tmp640
        tmp642 = 1 / tmp641
        tmp643 = tmp636 / tmp642
        tmp644 = tmp617 / tmp643
        tmp645 = tmp625 + tmp644
        tl.store(out_ptr72 + (x18), tmp617, None)
        tl.store(out_ptr74 + (x18), tmp645, None)
        tl.store(out_ptr75 + (x18), tmp624, None)
    elif xpid >= 146 and xpid < 147:
        xpid_offset = xpid - 146
        xnumel = 64
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x19 = xindex
        tmp646 = tl.load(in_ptr95 + (x19), xmask)
        tmp647 = tl.load(in_ptr96 + (x19), xmask)
        tmp652 = tl.load(in_ptr97 + (x19), xmask)
        tmp659 = tl.load(in_ptr98 + (x19), xmask)
        tmp661 = tl.load(in_ptr99 + (0))
        tmp662 = tl.broadcast_to(tmp661, [XBLOCK])
        tmp648 = tmp647 - tmp646
        tmp649 = 0.09999999999999998
        tmp650 = tmp648 * tmp649
        tmp651 = tmp646 + tmp650
        tmp653 = 0.999
        tmp654 = tmp652 * tmp653
        tmp655 = tmp647 * tmp647
        tmp656 = 0.0010000000000000009
        tmp657 = tmp655 * tmp656
        tmp658 = tmp654 + tmp657
        tmp660 = libdevice.sqrt(tmp658)
        tmp663 = libdevice.pow(tmp653, tmp662)
        tmp664 = 1.0
        tmp665 = tmp663 - tmp664
        tmp666 = -tmp665
        tmp667 = libdevice.sqrt(tmp666)
        tmp668 = tmp660 / tmp667
        tmp669 = 1e-08
        tmp670 = tmp668 + tmp669
        tmp671 = 0.9
        tmp672 = libdevice.pow(tmp671, tmp662)
        tmp673 = tmp672 - tmp664
        tmp674 = 0.001
        tmp675 = tmp673 / tmp674
        tmp676 = 1 / tmp675
        tmp677 = tmp670 / tmp676
        tmp678 = tmp651 / tmp677
        tmp679 = tmp659 + tmp678
        tl.store(out_ptr76 + (x19), tmp651, xmask)
        tl.store(out_ptr78 + (x19), tmp679, xmask)
        tl.store(out_ptr79 + (x19), tmp658, xmask)
    else:
        pass
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/pa/cpap5hoi7oye25spdins7q7j47ck23abthabl5kto7skoax6zkus.py
# Source Nodes: [], Original ATen: []

triton_for_fused_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.foreach(
    num_warps=8,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: '*fp32', 35: '*fp32', 36: '*fp32', 37: '*fp32', 38: '*fp32', 39: '*fp32', 40: '*fp32', 41: '*fp32', 42: '*fp32', 43: '*fp32', 44: '*fp32', 45: '*fp32', 46: '*fp32', 47: '*fp32', 48: '*fp32', 49: '*fp32', 50: '*fp32', 51: '*fp32', 52: '*fp32', 53: '*fp32', 54: '*fp32', 55: '*fp32', 56: '*fp32', 57: '*fp32', 58: '*fp32', 59: '*fp32', 60: '*fp32', 61: '*fp32', 62: '*fp32', 63: '*fp32', 64: '*fp32', 65: '*fp32', 66: '*fp32', 67: '*fp32', 68: '*fp32', 69: '*fp32', 70: '*fp32', 71: '*fp32', 72: '*fp32', 73: '*fp32', 74: '*fp32', 75: '*fp32', 76: '*fp32', 77: '*fp32', 78: '*fp32', 79: '*fp32', 80: '*fp32', 81: '*fp32', 82: '*fp32', 83: '*fp32', 84: '*fp32', 85: '*fp32', 86: '*fp32', 87: '*fp32', 88: '*fp32', 89: '*fp32', 90: '*fp32', 91: '*fp32', 92: '*fp32', 93: '*fp32', 94: '*fp32', 95: '*fp32', 96: '*fp32', 97: '*fp32', 98: '*fp32', 99: '*fp32', 100: '*fp32', 101: '*fp32', 102: '*fp32', 103: '*fp32', 104: '*fp32', 105: '*fp32', 106: '*fp32', 107: '*fp32', 108: '*fp32', 109: '*fp32', 110: '*fp32', 111: '*fp32', 112: '*fp32', 113: '*fp32', 114: '*fp32', 115: '*fp32', 116: '*fp32', 117: '*fp32', 118: '*fp32', 119: '*fp32', 120: '*fp32', 121: '*fp32', 122: '*fp32', 123: '*fp32', 124: '*fp32', 125: '*fp32', 126: '*fp32', 127: '*fp32', 128: '*fp32', 129: '*fp32', 130: '*fp32', 131: '*fp32', 132: '*fp32', 133: '*fp32', 134: '*fp32', 135: '*fp32', 136: '*fp32', 137: '*fp32', 138: '*fp32', 139: '*fp32', 140: '*fp32', 141: '*fp32', 142: '*fp32', 143: '*fp32', 144: '*fp32', 145: '*fp32', 146: '*fp32', 147: '*fp32', 148: '*fp32', 149: '*fp32', 150: '*fp32', 151: '*fp32', 152: '*fp32', 153: '*fp32', 154: '*fp32', 155: '*fp32', 156: '*fp32', 157: '*fp32', 158: '*fp32', 159: '*fp32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'kernel_name': 'triton_for_fused_3', 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, in_ptr47, in_ptr48, in_ptr49, in_ptr50, in_ptr51, in_ptr52, in_ptr53, in_ptr54, in_ptr55, in_ptr56, in_ptr57, in_ptr58, in_ptr59, in_ptr60, in_ptr61, in_ptr62, in_ptr63, in_ptr64, in_ptr65, in_ptr66, in_ptr67, in_ptr68, in_ptr69, in_ptr70, in_ptr71, in_ptr72, in_ptr73, in_ptr74, in_ptr75, in_ptr76, in_ptr77, in_ptr78, in_ptr79, in_ptr80, in_ptr81, in_ptr82, in_ptr83, in_ptr84, in_ptr85, in_ptr86, in_ptr87, in_ptr88, in_ptr89, in_ptr90, in_ptr91, in_ptr92, in_ptr93, in_ptr94, in_ptr95, in_ptr96, in_ptr97, in_ptr98, in_ptr99, out_ptr0, out_ptr2, out_ptr3, out_ptr4, out_ptr6, out_ptr7, out_ptr8, out_ptr10, out_ptr11, out_ptr12, out_ptr14, out_ptr15, out_ptr16, out_ptr18, out_ptr19, out_ptr20, out_ptr22, out_ptr23, out_ptr24, out_ptr26, out_ptr27, out_ptr28, out_ptr30, out_ptr31, out_ptr32, out_ptr34, out_ptr35, out_ptr36, out_ptr38, out_ptr39, out_ptr40, out_ptr42, out_ptr43, out_ptr44, out_ptr46, out_ptr47, out_ptr48, out_ptr50, out_ptr51, out_ptr52, out_ptr54, out_ptr55, out_ptr56, out_ptr58, out_ptr59, out_ptr60, out_ptr62, out_ptr63, out_ptr64, out_ptr66, out_ptr67, out_ptr68, out_ptr70, out_ptr71, out_ptr72, out_ptr74, out_ptr75, out_ptr76, out_ptr78, out_ptr79):
    xpid = tl.program_id(0)
    XBLOCK: tl.constexpr = 1024
    if xpid >= 0 and xpid < 1:
        xpid_offset = xpid - 0
        xnumel = 64
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), xmask)
        tmp1 = tl.load(in_ptr1 + (x0), xmask)
        tmp6 = tl.load(in_ptr2 + (x0), xmask)
        tmp13 = tl.load(in_ptr3 + (x0), xmask)
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
        tmp14 = libdevice.sqrt(tmp12)
        tmp17 = libdevice.pow(tmp7, tmp16)
        tmp18 = 1.0
        tmp19 = tmp17 - tmp18
        tmp20 = -tmp19
        tmp21 = libdevice.sqrt(tmp20)
        tmp22 = tmp14 / tmp21
        tmp23 = 1e-08
        tmp24 = tmp22 + tmp23
        tmp25 = 0.9
        tmp26 = libdevice.pow(tmp25, tmp16)
        tmp27 = tmp26 - tmp18
        tmp28 = 0.001
        tmp29 = tmp27 / tmp28
        tmp30 = 1 / tmp29
        tmp31 = tmp24 / tmp30
        tmp32 = tmp5 / tmp31
        tmp33 = tmp13 + tmp32
        tl.store(out_ptr0 + (x0), tmp5, xmask)
        tl.store(out_ptr2 + (x0), tmp33, xmask)
        tl.store(out_ptr3 + (x0), tmp12, xmask)
    elif xpid >= 1 and xpid < 17:
        xpid_offset = xpid - 1
        xnumel = 16384
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x1 = xindex
        tmp34 = tl.load(in_ptr5 + (x1), None)
        tmp35 = tl.load(in_ptr6 + (x1), None)
        tmp40 = tl.load(in_ptr7 + (x1), None)
        tmp47 = tl.load(in_ptr8 + (x1), None)
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
        tmp48 = libdevice.sqrt(tmp46)
        tmp51 = libdevice.pow(tmp41, tmp50)
        tmp52 = 1.0
        tmp53 = tmp51 - tmp52
        tmp54 = -tmp53
        tmp55 = libdevice.sqrt(tmp54)
        tmp56 = tmp48 / tmp55
        tmp57 = 1e-08
        tmp58 = tmp56 + tmp57
        tmp59 = 0.9
        tmp60 = libdevice.pow(tmp59, tmp50)
        tmp61 = tmp60 - tmp52
        tmp62 = 0.001
        tmp63 = tmp61 / tmp62
        tmp64 = 1 / tmp63
        tmp65 = tmp58 / tmp64
        tmp66 = tmp39 / tmp65
        tmp67 = tmp47 + tmp66
        tl.store(out_ptr4 + (x1), tmp39, None)
        tl.store(out_ptr6 + (x1), tmp67, None)
        tl.store(out_ptr7 + (x1), tmp46, None)
    elif xpid >= 17 and xpid < 18:
        xpid_offset = xpid - 17
        xnumel = 256
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x2 = xindex
        tmp68 = tl.load(in_ptr10 + (x2), xmask)
        tmp69 = tl.load(in_ptr11 + (x2), xmask)
        tmp74 = tl.load(in_ptr12 + (x2), xmask)
        tmp81 = tl.load(in_ptr13 + (x2), xmask)
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
        tmp82 = libdevice.sqrt(tmp80)
        tmp85 = libdevice.pow(tmp75, tmp84)
        tmp86 = 1.0
        tmp87 = tmp85 - tmp86
        tmp88 = -tmp87
        tmp89 = libdevice.sqrt(tmp88)
        tmp90 = tmp82 / tmp89
        tmp91 = 1e-08
        tmp92 = tmp90 + tmp91
        tmp93 = 0.9
        tmp94 = libdevice.pow(tmp93, tmp84)
        tmp95 = tmp94 - tmp86
        tmp96 = 0.001
        tmp97 = tmp95 / tmp96
        tmp98 = 1 / tmp97
        tmp99 = tmp92 / tmp98
        tmp100 = tmp73 / tmp99
        tmp101 = tmp81 + tmp100
        tl.store(out_ptr8 + (x2), tmp73, xmask)
        tl.store(out_ptr10 + (x2), tmp101, xmask)
        tl.store(out_ptr11 + (x2), tmp80, xmask)
    elif xpid >= 18 and xpid < 19:
        xpid_offset = xpid - 18
        xnumel = 256
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
        tmp116 = libdevice.sqrt(tmp114)
        tmp119 = libdevice.pow(tmp109, tmp118)
        tmp120 = 1.0
        tmp121 = tmp119 - tmp120
        tmp122 = -tmp121
        tmp123 = libdevice.sqrt(tmp122)
        tmp124 = tmp116 / tmp123
        tmp125 = 1e-08
        tmp126 = tmp124 + tmp125
        tmp127 = 0.9
        tmp128 = libdevice.pow(tmp127, tmp118)
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
    elif xpid >= 19 and xpid < 35:
        xpid_offset = xpid - 19
        xnumel = 16384
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x4 = xindex
        tmp136 = tl.load(in_ptr20 + (x4), None)
        tmp137 = tl.load(in_ptr21 + (x4), None)
        tmp142 = tl.load(in_ptr22 + (x4), None)
        tmp149 = tl.load(in_ptr23 + (x4), None)
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
        tmp150 = libdevice.sqrt(tmp148)
        tmp153 = libdevice.pow(tmp143, tmp152)
        tmp154 = 1.0
        tmp155 = tmp153 - tmp154
        tmp156 = -tmp155
        tmp157 = libdevice.sqrt(tmp156)
        tmp158 = tmp150 / tmp157
        tmp159 = 1e-08
        tmp160 = tmp158 + tmp159
        tmp161 = 0.9
        tmp162 = libdevice.pow(tmp161, tmp152)
        tmp163 = tmp162 - tmp154
        tmp164 = 0.001
        tmp165 = tmp163 / tmp164
        tmp166 = 1 / tmp165
        tmp167 = tmp160 / tmp166
        tmp168 = tmp141 / tmp167
        tmp169 = tmp149 + tmp168
        tl.store(out_ptr16 + (x4), tmp141, None)
        tl.store(out_ptr18 + (x4), tmp169, None)
        tl.store(out_ptr19 + (x4), tmp148, None)
    elif xpid >= 35 and xpid < 36:
        xpid_offset = xpid - 35
        xnumel = 64
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
        tmp184 = libdevice.sqrt(tmp182)
        tmp187 = libdevice.pow(tmp177, tmp186)
        tmp188 = 1.0
        tmp189 = tmp187 - tmp188
        tmp190 = -tmp189
        tmp191 = libdevice.sqrt(tmp190)
        tmp192 = tmp184 / tmp191
        tmp193 = 1e-08
        tmp194 = tmp192 + tmp193
        tmp195 = 0.9
        tmp196 = libdevice.pow(tmp195, tmp186)
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
    elif xpid >= 36 and xpid < 37:
        xpid_offset = xpid - 36
        xnumel = 64
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x6 = xindex
        tmp204 = tl.load(in_ptr30 + (x6), xmask)
        tmp205 = tl.load(in_ptr31 + (x6), xmask)
        tmp210 = tl.load(in_ptr32 + (x6), xmask)
        tmp217 = tl.load(in_ptr33 + (x6), xmask)
        tmp219 = tl.load(in_ptr34 + (0))
        tmp220 = tl.broadcast_to(tmp219, [XBLOCK])
        tmp206 = tmp205 - tmp204
        tmp207 = 0.09999999999999998
        tmp208 = tmp206 * tmp207
        tmp209 = tmp204 + tmp208
        tmp211 = 0.999
        tmp212 = tmp210 * tmp211
        tmp213 = tmp205 * tmp205
        tmp214 = 0.0010000000000000009
        tmp215 = tmp213 * tmp214
        tmp216 = tmp212 + tmp215
        tmp218 = libdevice.sqrt(tmp216)
        tmp221 = libdevice.pow(tmp211, tmp220)
        tmp222 = 1.0
        tmp223 = tmp221 - tmp222
        tmp224 = -tmp223
        tmp225 = libdevice.sqrt(tmp224)
        tmp226 = tmp218 / tmp225
        tmp227 = 1e-08
        tmp228 = tmp226 + tmp227
        tmp229 = 0.9
        tmp230 = libdevice.pow(tmp229, tmp220)
        tmp231 = tmp230 - tmp222
        tmp232 = 0.001
        tmp233 = tmp231 / tmp232
        tmp234 = 1 / tmp233
        tmp235 = tmp228 / tmp234
        tmp236 = tmp209 / tmp235
        tmp237 = tmp217 + tmp236
        tl.store(out_ptr24 + (x6), tmp209, xmask)
        tl.store(out_ptr26 + (x6), tmp237, xmask)
        tl.store(out_ptr27 + (x6), tmp216, xmask)
    elif xpid >= 37 and xpid < 73:
        xpid_offset = xpid - 37
        xnumel = 36864
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x7 = xindex
        tmp238 = tl.load(in_ptr35 + (x7), None)
        tmp239 = tl.load(in_ptr36 + (x7), None)
        tmp244 = tl.load(in_ptr37 + (x7), None)
        tmp251 = tl.load(in_ptr38 + (x7), None)
        tmp253 = tl.load(in_ptr39 + (0))
        tmp254 = tl.broadcast_to(tmp253, [XBLOCK])
        tmp240 = tmp239 - tmp238
        tmp241 = 0.09999999999999998
        tmp242 = tmp240 * tmp241
        tmp243 = tmp238 + tmp242
        tmp245 = 0.999
        tmp246 = tmp244 * tmp245
        tmp247 = tmp239 * tmp239
        tmp248 = 0.0010000000000000009
        tmp249 = tmp247 * tmp248
        tmp250 = tmp246 + tmp249
        tmp252 = libdevice.sqrt(tmp250)
        tmp255 = libdevice.pow(tmp245, tmp254)
        tmp256 = 1.0
        tmp257 = tmp255 - tmp256
        tmp258 = -tmp257
        tmp259 = libdevice.sqrt(tmp258)
        tmp260 = tmp252 / tmp259
        tmp261 = 1e-08
        tmp262 = tmp260 + tmp261
        tmp263 = 0.9
        tmp264 = libdevice.pow(tmp263, tmp254)
        tmp265 = tmp264 - tmp256
        tmp266 = 0.001
        tmp267 = tmp265 / tmp266
        tmp268 = 1 / tmp267
        tmp269 = tmp262 / tmp268
        tmp270 = tmp243 / tmp269
        tmp271 = tmp251 + tmp270
        tl.store(out_ptr28 + (x7), tmp243, None)
        tl.store(out_ptr30 + (x7), tmp271, None)
        tl.store(out_ptr31 + (x7), tmp250, None)
    elif xpid >= 73 and xpid < 74:
        xpid_offset = xpid - 73
        xnumel = 64
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x8 = xindex
        tmp272 = tl.load(in_ptr40 + (x8), xmask)
        tmp273 = tl.load(in_ptr41 + (x8), xmask)
        tmp278 = tl.load(in_ptr42 + (x8), xmask)
        tmp285 = tl.load(in_ptr43 + (x8), xmask)
        tmp287 = tl.load(in_ptr44 + (0))
        tmp288 = tl.broadcast_to(tmp287, [XBLOCK])
        tmp274 = tmp273 - tmp272
        tmp275 = 0.09999999999999998
        tmp276 = tmp274 * tmp275
        tmp277 = tmp272 + tmp276
        tmp279 = 0.999
        tmp280 = tmp278 * tmp279
        tmp281 = tmp273 * tmp273
        tmp282 = 0.0010000000000000009
        tmp283 = tmp281 * tmp282
        tmp284 = tmp280 + tmp283
        tmp286 = libdevice.sqrt(tmp284)
        tmp289 = libdevice.pow(tmp279, tmp288)
        tmp290 = 1.0
        tmp291 = tmp289 - tmp290
        tmp292 = -tmp291
        tmp293 = libdevice.sqrt(tmp292)
        tmp294 = tmp286 / tmp293
        tmp295 = 1e-08
        tmp296 = tmp294 + tmp295
        tmp297 = 0.9
        tmp298 = libdevice.pow(tmp297, tmp288)
        tmp299 = tmp298 - tmp290
        tmp300 = 0.001
        tmp301 = tmp299 / tmp300
        tmp302 = 1 / tmp301
        tmp303 = tmp296 / tmp302
        tmp304 = tmp277 / tmp303
        tmp305 = tmp285 + tmp304
        tl.store(out_ptr32 + (x8), tmp277, xmask)
        tl.store(out_ptr34 + (x8), tmp305, xmask)
        tl.store(out_ptr35 + (x8), tmp284, xmask)
    elif xpid >= 74 and xpid < 75:
        xpid_offset = xpid - 74
        xnumel = 64
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x9 = xindex
        tmp306 = tl.load(in_ptr45 + (x9), xmask)
        tmp307 = tl.load(in_ptr46 + (x9), xmask)
        tmp312 = tl.load(in_ptr47 + (x9), xmask)
        tmp319 = tl.load(in_ptr48 + (x9), xmask)
        tmp321 = tl.load(in_ptr49 + (0))
        tmp322 = tl.broadcast_to(tmp321, [XBLOCK])
        tmp308 = tmp307 - tmp306
        tmp309 = 0.09999999999999998
        tmp310 = tmp308 * tmp309
        tmp311 = tmp306 + tmp310
        tmp313 = 0.999
        tmp314 = tmp312 * tmp313
        tmp315 = tmp307 * tmp307
        tmp316 = 0.0010000000000000009
        tmp317 = tmp315 * tmp316
        tmp318 = tmp314 + tmp317
        tmp320 = libdevice.sqrt(tmp318)
        tmp323 = libdevice.pow(tmp313, tmp322)
        tmp324 = 1.0
        tmp325 = tmp323 - tmp324
        tmp326 = -tmp325
        tmp327 = libdevice.sqrt(tmp326)
        tmp328 = tmp320 / tmp327
        tmp329 = 1e-08
        tmp330 = tmp328 + tmp329
        tmp331 = 0.9
        tmp332 = libdevice.pow(tmp331, tmp322)
        tmp333 = tmp332 - tmp324
        tmp334 = 0.001
        tmp335 = tmp333 / tmp334
        tmp336 = 1 / tmp335
        tmp337 = tmp330 / tmp336
        tmp338 = tmp311 / tmp337
        tmp339 = tmp319 + tmp338
        tl.store(out_ptr36 + (x9), tmp311, xmask)
        tl.store(out_ptr38 + (x9), tmp339, xmask)
        tl.store(out_ptr39 + (x9), tmp318, xmask)
    elif xpid >= 75 and xpid < 91:
        xpid_offset = xpid - 75
        xnumel = 16384
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x10 = xindex
        tmp340 = tl.load(in_ptr50 + (x10), None)
        tmp341 = tl.load(in_ptr51 + (x10), None)
        tmp346 = tl.load(in_ptr52 + (x10), None)
        tmp353 = tl.load(in_ptr53 + (x10), None)
        tmp355 = tl.load(in_ptr54 + (0))
        tmp356 = tl.broadcast_to(tmp355, [XBLOCK])
        tmp342 = tmp341 - tmp340
        tmp343 = 0.09999999999999998
        tmp344 = tmp342 * tmp343
        tmp345 = tmp340 + tmp344
        tmp347 = 0.999
        tmp348 = tmp346 * tmp347
        tmp349 = tmp341 * tmp341
        tmp350 = 0.0010000000000000009
        tmp351 = tmp349 * tmp350
        tmp352 = tmp348 + tmp351
        tmp354 = libdevice.sqrt(tmp352)
        tmp357 = libdevice.pow(tmp347, tmp356)
        tmp358 = 1.0
        tmp359 = tmp357 - tmp358
        tmp360 = -tmp359
        tmp361 = libdevice.sqrt(tmp360)
        tmp362 = tmp354 / tmp361
        tmp363 = 1e-08
        tmp364 = tmp362 + tmp363
        tmp365 = 0.9
        tmp366 = libdevice.pow(tmp365, tmp356)
        tmp367 = tmp366 - tmp358
        tmp368 = 0.001
        tmp369 = tmp367 / tmp368
        tmp370 = 1 / tmp369
        tmp371 = tmp364 / tmp370
        tmp372 = tmp345 / tmp371
        tmp373 = tmp353 + tmp372
        tl.store(out_ptr40 + (x10), tmp345, None)
        tl.store(out_ptr42 + (x10), tmp373, None)
        tl.store(out_ptr43 + (x10), tmp352, None)
    elif xpid >= 91 and xpid < 92:
        xpid_offset = xpid - 91
        xnumel = 256
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x11 = xindex
        tmp374 = tl.load(in_ptr55 + (x11), xmask)
        tmp375 = tl.load(in_ptr56 + (x11), xmask)
        tmp380 = tl.load(in_ptr57 + (x11), xmask)
        tmp387 = tl.load(in_ptr58 + (x11), xmask)
        tmp389 = tl.load(in_ptr59 + (0))
        tmp390 = tl.broadcast_to(tmp389, [XBLOCK])
        tmp376 = tmp375 - tmp374
        tmp377 = 0.09999999999999998
        tmp378 = tmp376 * tmp377
        tmp379 = tmp374 + tmp378
        tmp381 = 0.999
        tmp382 = tmp380 * tmp381
        tmp383 = tmp375 * tmp375
        tmp384 = 0.0010000000000000009
        tmp385 = tmp383 * tmp384
        tmp386 = tmp382 + tmp385
        tmp388 = libdevice.sqrt(tmp386)
        tmp391 = libdevice.pow(tmp381, tmp390)
        tmp392 = 1.0
        tmp393 = tmp391 - tmp392
        tmp394 = -tmp393
        tmp395 = libdevice.sqrt(tmp394)
        tmp396 = tmp388 / tmp395
        tmp397 = 1e-08
        tmp398 = tmp396 + tmp397
        tmp399 = 0.9
        tmp400 = libdevice.pow(tmp399, tmp390)
        tmp401 = tmp400 - tmp392
        tmp402 = 0.001
        tmp403 = tmp401 / tmp402
        tmp404 = 1 / tmp403
        tmp405 = tmp398 / tmp404
        tmp406 = tmp379 / tmp405
        tmp407 = tmp387 + tmp406
        tl.store(out_ptr44 + (x11), tmp379, xmask)
        tl.store(out_ptr46 + (x11), tmp407, xmask)
        tl.store(out_ptr47 + (x11), tmp386, xmask)
    elif xpid >= 92 and xpid < 93:
        xpid_offset = xpid - 92
        xnumel = 256
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x12 = xindex
        tmp408 = tl.load(in_ptr60 + (x12), xmask)
        tmp409 = tl.load(in_ptr61 + (x12), xmask)
        tmp414 = tl.load(in_ptr62 + (x12), xmask)
        tmp421 = tl.load(in_ptr63 + (x12), xmask)
        tmp423 = tl.load(in_ptr64 + (0))
        tmp424 = tl.broadcast_to(tmp423, [XBLOCK])
        tmp410 = tmp409 - tmp408
        tmp411 = 0.09999999999999998
        tmp412 = tmp410 * tmp411
        tmp413 = tmp408 + tmp412
        tmp415 = 0.999
        tmp416 = tmp414 * tmp415
        tmp417 = tmp409 * tmp409
        tmp418 = 0.0010000000000000009
        tmp419 = tmp417 * tmp418
        tmp420 = tmp416 + tmp419
        tmp422 = libdevice.sqrt(tmp420)
        tmp425 = libdevice.pow(tmp415, tmp424)
        tmp426 = 1.0
        tmp427 = tmp425 - tmp426
        tmp428 = -tmp427
        tmp429 = libdevice.sqrt(tmp428)
        tmp430 = tmp422 / tmp429
        tmp431 = 1e-08
        tmp432 = tmp430 + tmp431
        tmp433 = 0.9
        tmp434 = libdevice.pow(tmp433, tmp424)
        tmp435 = tmp434 - tmp426
        tmp436 = 0.001
        tmp437 = tmp435 / tmp436
        tmp438 = 1 / tmp437
        tmp439 = tmp432 / tmp438
        tmp440 = tmp413 / tmp439
        tmp441 = tmp421 + tmp440
        tl.store(out_ptr48 + (x12), tmp413, xmask)
        tl.store(out_ptr50 + (x12), tmp441, xmask)
        tl.store(out_ptr51 + (x12), tmp420, xmask)
    elif xpid >= 93 and xpid < 125:
        xpid_offset = xpid - 93
        xnumel = 32768
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x13 = xindex
        tmp442 = tl.load(in_ptr65 + (x13), None)
        tmp443 = tl.load(in_ptr66 + (x13), None)
        tmp448 = tl.load(in_ptr67 + (x13), None)
        tmp455 = tl.load(in_ptr68 + (x13), None)
        tmp457 = tl.load(in_ptr69 + (0))
        tmp458 = tl.broadcast_to(tmp457, [XBLOCK])
        tmp444 = tmp443 - tmp442
        tmp445 = 0.09999999999999998
        tmp446 = tmp444 * tmp445
        tmp447 = tmp442 + tmp446
        tmp449 = 0.999
        tmp450 = tmp448 * tmp449
        tmp451 = tmp443 * tmp443
        tmp452 = 0.0010000000000000009
        tmp453 = tmp451 * tmp452
        tmp454 = tmp450 + tmp453
        tmp456 = libdevice.sqrt(tmp454)
        tmp459 = libdevice.pow(tmp449, tmp458)
        tmp460 = 1.0
        tmp461 = tmp459 - tmp460
        tmp462 = -tmp461
        tmp463 = libdevice.sqrt(tmp462)
        tmp464 = tmp456 / tmp463
        tmp465 = 1e-08
        tmp466 = tmp464 + tmp465
        tmp467 = 0.9
        tmp468 = libdevice.pow(tmp467, tmp458)
        tmp469 = tmp468 - tmp460
        tmp470 = 0.001
        tmp471 = tmp469 / tmp470
        tmp472 = 1 / tmp471
        tmp473 = tmp466 / tmp472
        tmp474 = tmp447 / tmp473
        tmp475 = tmp455 + tmp474
        tl.store(out_ptr52 + (x13), tmp447, None)
        tl.store(out_ptr54 + (x13), tmp475, None)
        tl.store(out_ptr55 + (x13), tmp454, None)
    elif xpid >= 125 and xpid < 126:
        xpid_offset = xpid - 125
        xnumel = 128
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x14 = xindex
        tmp476 = tl.load(in_ptr70 + (x14), xmask)
        tmp477 = tl.load(in_ptr71 + (x14), xmask)
        tmp482 = tl.load(in_ptr72 + (x14), xmask)
        tmp489 = tl.load(in_ptr73 + (x14), xmask)
        tmp491 = tl.load(in_ptr74 + (0))
        tmp492 = tl.broadcast_to(tmp491, [XBLOCK])
        tmp478 = tmp477 - tmp476
        tmp479 = 0.09999999999999998
        tmp480 = tmp478 * tmp479
        tmp481 = tmp476 + tmp480
        tmp483 = 0.999
        tmp484 = tmp482 * tmp483
        tmp485 = tmp477 * tmp477
        tmp486 = 0.0010000000000000009
        tmp487 = tmp485 * tmp486
        tmp488 = tmp484 + tmp487
        tmp490 = libdevice.sqrt(tmp488)
        tmp493 = libdevice.pow(tmp483, tmp492)
        tmp494 = 1.0
        tmp495 = tmp493 - tmp494
        tmp496 = -tmp495
        tmp497 = libdevice.sqrt(tmp496)
        tmp498 = tmp490 / tmp497
        tmp499 = 1e-08
        tmp500 = tmp498 + tmp499
        tmp501 = 0.9
        tmp502 = libdevice.pow(tmp501, tmp492)
        tmp503 = tmp502 - tmp494
        tmp504 = 0.001
        tmp505 = tmp503 / tmp504
        tmp506 = 1 / tmp505
        tmp507 = tmp500 / tmp506
        tmp508 = tmp481 / tmp507
        tmp509 = tmp489 + tmp508
        tl.store(out_ptr56 + (x14), tmp481, xmask)
        tl.store(out_ptr58 + (x14), tmp509, xmask)
        tl.store(out_ptr59 + (x14), tmp488, xmask)
    elif xpid >= 126 and xpid < 127:
        xpid_offset = xpid - 126
        xnumel = 128
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x15 = xindex
        tmp510 = tl.load(in_ptr75 + (x15), xmask)
        tmp511 = tl.load(in_ptr76 + (x15), xmask)
        tmp516 = tl.load(in_ptr77 + (x15), xmask)
        tmp523 = tl.load(in_ptr78 + (x15), xmask)
        tmp525 = tl.load(in_ptr79 + (0))
        tmp526 = tl.broadcast_to(tmp525, [XBLOCK])
        tmp512 = tmp511 - tmp510
        tmp513 = 0.09999999999999998
        tmp514 = tmp512 * tmp513
        tmp515 = tmp510 + tmp514
        tmp517 = 0.999
        tmp518 = tmp516 * tmp517
        tmp519 = tmp511 * tmp511
        tmp520 = 0.0010000000000000009
        tmp521 = tmp519 * tmp520
        tmp522 = tmp518 + tmp521
        tmp524 = libdevice.sqrt(tmp522)
        tmp527 = libdevice.pow(tmp517, tmp526)
        tmp528 = 1.0
        tmp529 = tmp527 - tmp528
        tmp530 = -tmp529
        tmp531 = libdevice.sqrt(tmp530)
        tmp532 = tmp524 / tmp531
        tmp533 = 1e-08
        tmp534 = tmp532 + tmp533
        tmp535 = 0.9
        tmp536 = libdevice.pow(tmp535, tmp526)
        tmp537 = tmp536 - tmp528
        tmp538 = 0.001
        tmp539 = tmp537 / tmp538
        tmp540 = 1 / tmp539
        tmp541 = tmp534 / tmp540
        tmp542 = tmp515 / tmp541
        tmp543 = tmp523 + tmp542
        tl.store(out_ptr60 + (x15), tmp515, xmask)
        tl.store(out_ptr62 + (x15), tmp543, xmask)
        tl.store(out_ptr63 + (x15), tmp522, xmask)
    elif xpid >= 127 and xpid < 271:
        xpid_offset = xpid - 127
        xnumel = 147456
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x16 = xindex
        tmp544 = tl.load(in_ptr80 + (x16), None)
        tmp545 = tl.load(in_ptr81 + (x16), None)
        tmp550 = tl.load(in_ptr82 + (x16), None)
        tmp557 = tl.load(in_ptr83 + (x16), None)
        tmp559 = tl.load(in_ptr84 + (0))
        tmp560 = tl.broadcast_to(tmp559, [XBLOCK])
        tmp546 = tmp545 - tmp544
        tmp547 = 0.09999999999999998
        tmp548 = tmp546 * tmp547
        tmp549 = tmp544 + tmp548
        tmp551 = 0.999
        tmp552 = tmp550 * tmp551
        tmp553 = tmp545 * tmp545
        tmp554 = 0.0010000000000000009
        tmp555 = tmp553 * tmp554
        tmp556 = tmp552 + tmp555
        tmp558 = libdevice.sqrt(tmp556)
        tmp561 = libdevice.pow(tmp551, tmp560)
        tmp562 = 1.0
        tmp563 = tmp561 - tmp562
        tmp564 = -tmp563
        tmp565 = libdevice.sqrt(tmp564)
        tmp566 = tmp558 / tmp565
        tmp567 = 1e-08
        tmp568 = tmp566 + tmp567
        tmp569 = 0.9
        tmp570 = libdevice.pow(tmp569, tmp560)
        tmp571 = tmp570 - tmp562
        tmp572 = 0.001
        tmp573 = tmp571 / tmp572
        tmp574 = 1 / tmp573
        tmp575 = tmp568 / tmp574
        tmp576 = tmp549 / tmp575
        tmp577 = tmp557 + tmp576
        tl.store(out_ptr64 + (x16), tmp549, None)
        tl.store(out_ptr66 + (x16), tmp577, None)
        tl.store(out_ptr67 + (x16), tmp556, None)
    elif xpid >= 271 and xpid < 272:
        xpid_offset = xpid - 271
        xnumel = 128
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x17 = xindex
        tmp578 = tl.load(in_ptr85 + (x17), xmask)
        tmp579 = tl.load(in_ptr86 + (x17), xmask)
        tmp584 = tl.load(in_ptr87 + (x17), xmask)
        tmp591 = tl.load(in_ptr88 + (x17), xmask)
        tmp593 = tl.load(in_ptr89 + (0))
        tmp594 = tl.broadcast_to(tmp593, [XBLOCK])
        tmp580 = tmp579 - tmp578
        tmp581 = 0.09999999999999998
        tmp582 = tmp580 * tmp581
        tmp583 = tmp578 + tmp582
        tmp585 = 0.999
        tmp586 = tmp584 * tmp585
        tmp587 = tmp579 * tmp579
        tmp588 = 0.0010000000000000009
        tmp589 = tmp587 * tmp588
        tmp590 = tmp586 + tmp589
        tmp592 = libdevice.sqrt(tmp590)
        tmp595 = libdevice.pow(tmp585, tmp594)
        tmp596 = 1.0
        tmp597 = tmp595 - tmp596
        tmp598 = -tmp597
        tmp599 = libdevice.sqrt(tmp598)
        tmp600 = tmp592 / tmp599
        tmp601 = 1e-08
        tmp602 = tmp600 + tmp601
        tmp603 = 0.9
        tmp604 = libdevice.pow(tmp603, tmp594)
        tmp605 = tmp604 - tmp596
        tmp606 = 0.001
        tmp607 = tmp605 / tmp606
        tmp608 = 1 / tmp607
        tmp609 = tmp602 / tmp608
        tmp610 = tmp583 / tmp609
        tmp611 = tmp591 + tmp610
        tl.store(out_ptr68 + (x17), tmp583, xmask)
        tl.store(out_ptr70 + (x17), tmp611, xmask)
        tl.store(out_ptr71 + (x17), tmp590, xmask)
    elif xpid >= 272 and xpid < 273:
        xpid_offset = xpid - 272
        xnumel = 128
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x18 = xindex
        tmp612 = tl.load(in_ptr90 + (x18), xmask)
        tmp613 = tl.load(in_ptr91 + (x18), xmask)
        tmp618 = tl.load(in_ptr92 + (x18), xmask)
        tmp625 = tl.load(in_ptr93 + (x18), xmask)
        tmp627 = tl.load(in_ptr94 + (0))
        tmp628 = tl.broadcast_to(tmp627, [XBLOCK])
        tmp614 = tmp613 - tmp612
        tmp615 = 0.09999999999999998
        tmp616 = tmp614 * tmp615
        tmp617 = tmp612 + tmp616
        tmp619 = 0.999
        tmp620 = tmp618 * tmp619
        tmp621 = tmp613 * tmp613
        tmp622 = 0.0010000000000000009
        tmp623 = tmp621 * tmp622
        tmp624 = tmp620 + tmp623
        tmp626 = libdevice.sqrt(tmp624)
        tmp629 = libdevice.pow(tmp619, tmp628)
        tmp630 = 1.0
        tmp631 = tmp629 - tmp630
        tmp632 = -tmp631
        tmp633 = libdevice.sqrt(tmp632)
        tmp634 = tmp626 / tmp633
        tmp635 = 1e-08
        tmp636 = tmp634 + tmp635
        tmp637 = 0.9
        tmp638 = libdevice.pow(tmp637, tmp628)
        tmp639 = tmp638 - tmp630
        tmp640 = 0.001
        tmp641 = tmp639 / tmp640
        tmp642 = 1 / tmp641
        tmp643 = tmp636 / tmp642
        tmp644 = tmp617 / tmp643
        tmp645 = tmp625 + tmp644
        tl.store(out_ptr72 + (x18), tmp617, xmask)
        tl.store(out_ptr74 + (x18), tmp645, xmask)
        tl.store(out_ptr75 + (x18), tmp624, xmask)
    elif xpid >= 273 and xpid < 337:
        xpid_offset = xpid - 273
        xnumel = 65536
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x19 = xindex
        tmp646 = tl.load(in_ptr95 + (x19), None)
        tmp647 = tl.load(in_ptr96 + (x19), None)
        tmp652 = tl.load(in_ptr97 + (x19), None)
        tmp659 = tl.load(in_ptr98 + (x19), None)
        tmp661 = tl.load(in_ptr99 + (0))
        tmp662 = tl.broadcast_to(tmp661, [XBLOCK])
        tmp648 = tmp647 - tmp646
        tmp649 = 0.09999999999999998
        tmp650 = tmp648 * tmp649
        tmp651 = tmp646 + tmp650
        tmp653 = 0.999
        tmp654 = tmp652 * tmp653
        tmp655 = tmp647 * tmp647
        tmp656 = 0.0010000000000000009
        tmp657 = tmp655 * tmp656
        tmp658 = tmp654 + tmp657
        tmp660 = libdevice.sqrt(tmp658)
        tmp663 = libdevice.pow(tmp653, tmp662)
        tmp664 = 1.0
        tmp665 = tmp663 - tmp664
        tmp666 = -tmp665
        tmp667 = libdevice.sqrt(tmp666)
        tmp668 = tmp660 / tmp667
        tmp669 = 1e-08
        tmp670 = tmp668 + tmp669
        tmp671 = 0.9
        tmp672 = libdevice.pow(tmp671, tmp662)
        tmp673 = tmp672 - tmp664
        tmp674 = 0.001
        tmp675 = tmp673 / tmp674
        tmp676 = 1 / tmp675
        tmp677 = tmp670 / tmp676
        tmp678 = tmp651 / tmp677
        tmp679 = tmp659 + tmp678
        tl.store(out_ptr76 + (x19), tmp651, None)
        tl.store(out_ptr78 + (x19), tmp679, None)
        tl.store(out_ptr79 + (x19), tmp658, None)
    else:
        pass
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/zo/czonjiwdhpcvlc2yjj4vkh3w32bucrxvqgkeucvoliwhavbgjfz4.py
# Source Nodes: [], Original ATen: []

triton_for_fused_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.foreach(
    num_warps=8,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: '*fp32', 35: '*fp32', 36: '*fp32', 37: '*fp32', 38: '*fp32', 39: '*fp32', 40: '*fp32', 41: '*fp32', 42: '*fp32', 43: '*fp32', 44: '*fp32', 45: '*fp32', 46: '*fp32', 47: '*fp32', 48: '*fp32', 49: '*fp32', 50: '*fp32', 51: '*fp32', 52: '*fp32', 53: '*fp32', 54: '*fp32', 55: '*fp32', 56: '*fp32', 57: '*fp32', 58: '*fp32', 59: '*fp32', 60: '*fp32', 61: '*fp32', 62: '*fp32', 63: '*fp32', 64: '*fp32', 65: '*fp32', 66: '*fp32', 67: '*fp32', 68: '*fp32', 69: '*fp32', 70: '*fp32', 71: '*fp32', 72: '*fp32', 73: '*fp32', 74: '*fp32', 75: '*fp32', 76: '*fp32', 77: '*fp32', 78: '*fp32', 79: '*fp32', 80: '*fp32', 81: '*fp32', 82: '*fp32', 83: '*fp32', 84: '*fp32', 85: '*fp32', 86: '*fp32', 87: '*fp32', 88: '*fp32', 89: '*fp32', 90: '*fp32', 91: '*fp32', 92: '*fp32', 93: '*fp32', 94: '*fp32', 95: '*fp32', 96: '*fp32', 97: '*fp32', 98: '*fp32', 99: '*fp32', 100: '*fp32', 101: '*fp32', 102: '*fp32', 103: '*fp32', 104: '*fp32', 105: '*fp32', 106: '*fp32', 107: '*fp32', 108: '*fp32', 109: '*fp32', 110: '*fp32', 111: '*fp32', 112: '*fp32', 113: '*fp32', 114: '*fp32', 115: '*fp32', 116: '*fp32', 117: '*fp32', 118: '*fp32', 119: '*fp32', 120: '*fp32', 121: '*fp32', 122: '*fp32', 123: '*fp32', 124: '*fp32', 125: '*fp32', 126: '*fp32', 127: '*fp32', 128: '*fp32', 129: '*fp32', 130: '*fp32', 131: '*fp32', 132: '*fp32', 133: '*fp32', 134: '*fp32', 135: '*fp32', 136: '*fp32', 137: '*fp32', 138: '*fp32', 139: '*fp32', 140: '*fp32', 141: '*fp32', 142: '*fp32', 143: '*fp32', 144: '*fp32', 145: '*fp32', 146: '*fp32', 147: '*fp32', 148: '*fp32', 149: '*fp32', 150: '*fp32', 151: '*fp32', 152: '*fp32', 153: '*fp32', 154: '*fp32', 155: '*fp32', 156: '*fp32', 157: '*fp32', 158: '*fp32', 159: '*fp32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'kernel_name': 'triton_for_fused_4', 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, in_ptr47, in_ptr48, in_ptr49, in_ptr50, in_ptr51, in_ptr52, in_ptr53, in_ptr54, in_ptr55, in_ptr56, in_ptr57, in_ptr58, in_ptr59, in_ptr60, in_ptr61, in_ptr62, in_ptr63, in_ptr64, in_ptr65, in_ptr66, in_ptr67, in_ptr68, in_ptr69, in_ptr70, in_ptr71, in_ptr72, in_ptr73, in_ptr74, in_ptr75, in_ptr76, in_ptr77, in_ptr78, in_ptr79, in_ptr80, in_ptr81, in_ptr82, in_ptr83, in_ptr84, in_ptr85, in_ptr86, in_ptr87, in_ptr88, in_ptr89, in_ptr90, in_ptr91, in_ptr92, in_ptr93, in_ptr94, in_ptr95, in_ptr96, in_ptr97, in_ptr98, in_ptr99, out_ptr0, out_ptr2, out_ptr3, out_ptr4, out_ptr6, out_ptr7, out_ptr8, out_ptr10, out_ptr11, out_ptr12, out_ptr14, out_ptr15, out_ptr16, out_ptr18, out_ptr19, out_ptr20, out_ptr22, out_ptr23, out_ptr24, out_ptr26, out_ptr27, out_ptr28, out_ptr30, out_ptr31, out_ptr32, out_ptr34, out_ptr35, out_ptr36, out_ptr38, out_ptr39, out_ptr40, out_ptr42, out_ptr43, out_ptr44, out_ptr46, out_ptr47, out_ptr48, out_ptr50, out_ptr51, out_ptr52, out_ptr54, out_ptr55, out_ptr56, out_ptr58, out_ptr59, out_ptr60, out_ptr62, out_ptr63, out_ptr64, out_ptr66, out_ptr67, out_ptr68, out_ptr70, out_ptr71, out_ptr72, out_ptr74, out_ptr75, out_ptr76, out_ptr78, out_ptr79):
    xpid = tl.program_id(0)
    XBLOCK: tl.constexpr = 1024
    if xpid >= 0 and xpid < 1:
        xpid_offset = xpid - 0
        xnumel = 512
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), xmask)
        tmp1 = tl.load(in_ptr1 + (x0), xmask)
        tmp6 = tl.load(in_ptr2 + (x0), xmask)
        tmp13 = tl.load(in_ptr3 + (x0), xmask)
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
        tmp14 = libdevice.sqrt(tmp12)
        tmp17 = libdevice.pow(tmp7, tmp16)
        tmp18 = 1.0
        tmp19 = tmp17 - tmp18
        tmp20 = -tmp19
        tmp21 = libdevice.sqrt(tmp20)
        tmp22 = tmp14 / tmp21
        tmp23 = 1e-08
        tmp24 = tmp22 + tmp23
        tmp25 = 0.9
        tmp26 = libdevice.pow(tmp25, tmp16)
        tmp27 = tmp26 - tmp18
        tmp28 = 0.001
        tmp29 = tmp27 / tmp28
        tmp30 = 1 / tmp29
        tmp31 = tmp24 / tmp30
        tmp32 = tmp5 / tmp31
        tmp33 = tmp13 + tmp32
        tl.store(out_ptr0 + (x0), tmp5, xmask)
        tl.store(out_ptr2 + (x0), tmp33, xmask)
        tl.store(out_ptr3 + (x0), tmp12, xmask)
    elif xpid >= 1 and xpid < 2:
        xpid_offset = xpid - 1
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
        tmp48 = libdevice.sqrt(tmp46)
        tmp51 = libdevice.pow(tmp41, tmp50)
        tmp52 = 1.0
        tmp53 = tmp51 - tmp52
        tmp54 = -tmp53
        tmp55 = libdevice.sqrt(tmp54)
        tmp56 = tmp48 / tmp55
        tmp57 = 1e-08
        tmp58 = tmp56 + tmp57
        tmp59 = 0.9
        tmp60 = libdevice.pow(tmp59, tmp50)
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
    elif xpid >= 2 and xpid < 130:
        xpid_offset = xpid - 2
        xnumel = 131072
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
        tmp82 = libdevice.sqrt(tmp80)
        tmp85 = libdevice.pow(tmp75, tmp84)
        tmp86 = 1.0
        tmp87 = tmp85 - tmp86
        tmp88 = -tmp87
        tmp89 = libdevice.sqrt(tmp88)
        tmp90 = tmp82 / tmp89
        tmp91 = 1e-08
        tmp92 = tmp90 + tmp91
        tmp93 = 0.9
        tmp94 = libdevice.pow(tmp93, tmp84)
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
    elif xpid >= 130 and xpid < 131:
        xpid_offset = xpid - 130
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
        tmp116 = libdevice.sqrt(tmp114)
        tmp119 = libdevice.pow(tmp109, tmp118)
        tmp120 = 1.0
        tmp121 = tmp119 - tmp120
        tmp122 = -tmp121
        tmp123 = libdevice.sqrt(tmp122)
        tmp124 = tmp116 / tmp123
        tmp125 = 1e-08
        tmp126 = tmp124 + tmp125
        tmp127 = 0.9
        tmp128 = libdevice.pow(tmp127, tmp118)
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
    elif xpid >= 131 and xpid < 132:
        xpid_offset = xpid - 131
        xnumel = 512
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
        tmp150 = libdevice.sqrt(tmp148)
        tmp153 = libdevice.pow(tmp143, tmp152)
        tmp154 = 1.0
        tmp155 = tmp153 - tmp154
        tmp156 = -tmp155
        tmp157 = libdevice.sqrt(tmp156)
        tmp158 = tmp150 / tmp157
        tmp159 = 1e-08
        tmp160 = tmp158 + tmp159
        tmp161 = 0.9
        tmp162 = libdevice.pow(tmp161, tmp152)
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
    elif xpid >= 132 and xpid < 196:
        xpid_offset = xpid - 132
        xnumel = 65536
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x5 = xindex
        tmp170 = tl.load(in_ptr25 + (x5), None)
        tmp171 = tl.load(in_ptr26 + (x5), None)
        tmp176 = tl.load(in_ptr27 + (x5), None)
        tmp183 = tl.load(in_ptr28 + (x5), None)
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
        tmp184 = libdevice.sqrt(tmp182)
        tmp187 = libdevice.pow(tmp177, tmp186)
        tmp188 = 1.0
        tmp189 = tmp187 - tmp188
        tmp190 = -tmp189
        tmp191 = libdevice.sqrt(tmp190)
        tmp192 = tmp184 / tmp191
        tmp193 = 1e-08
        tmp194 = tmp192 + tmp193
        tmp195 = 0.9
        tmp196 = libdevice.pow(tmp195, tmp186)
        tmp197 = tmp196 - tmp188
        tmp198 = 0.001
        tmp199 = tmp197 / tmp198
        tmp200 = 1 / tmp199
        tmp201 = tmp194 / tmp200
        tmp202 = tmp175 / tmp201
        tmp203 = tmp183 + tmp202
        tl.store(out_ptr20 + (x5), tmp175, None)
        tl.store(out_ptr22 + (x5), tmp203, None)
        tl.store(out_ptr23 + (x5), tmp182, None)
    elif xpid >= 196 and xpid < 197:
        xpid_offset = xpid - 196
        xnumel = 128
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x6 = xindex
        tmp204 = tl.load(in_ptr30 + (x6), xmask)
        tmp205 = tl.load(in_ptr31 + (x6), xmask)
        tmp210 = tl.load(in_ptr32 + (x6), xmask)
        tmp217 = tl.load(in_ptr33 + (x6), xmask)
        tmp219 = tl.load(in_ptr34 + (0))
        tmp220 = tl.broadcast_to(tmp219, [XBLOCK])
        tmp206 = tmp205 - tmp204
        tmp207 = 0.09999999999999998
        tmp208 = tmp206 * tmp207
        tmp209 = tmp204 + tmp208
        tmp211 = 0.999
        tmp212 = tmp210 * tmp211
        tmp213 = tmp205 * tmp205
        tmp214 = 0.0010000000000000009
        tmp215 = tmp213 * tmp214
        tmp216 = tmp212 + tmp215
        tmp218 = libdevice.sqrt(tmp216)
        tmp221 = libdevice.pow(tmp211, tmp220)
        tmp222 = 1.0
        tmp223 = tmp221 - tmp222
        tmp224 = -tmp223
        tmp225 = libdevice.sqrt(tmp224)
        tmp226 = tmp218 / tmp225
        tmp227 = 1e-08
        tmp228 = tmp226 + tmp227
        tmp229 = 0.9
        tmp230 = libdevice.pow(tmp229, tmp220)
        tmp231 = tmp230 - tmp222
        tmp232 = 0.001
        tmp233 = tmp231 / tmp232
        tmp234 = 1 / tmp233
        tmp235 = tmp228 / tmp234
        tmp236 = tmp209 / tmp235
        tmp237 = tmp217 + tmp236
        tl.store(out_ptr24 + (x6), tmp209, xmask)
        tl.store(out_ptr26 + (x6), tmp237, xmask)
        tl.store(out_ptr27 + (x6), tmp216, xmask)
    elif xpid >= 197 and xpid < 198:
        xpid_offset = xpid - 197
        xnumel = 128
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x7 = xindex
        tmp238 = tl.load(in_ptr35 + (x7), xmask)
        tmp239 = tl.load(in_ptr36 + (x7), xmask)
        tmp244 = tl.load(in_ptr37 + (x7), xmask)
        tmp251 = tl.load(in_ptr38 + (x7), xmask)
        tmp253 = tl.load(in_ptr39 + (0))
        tmp254 = tl.broadcast_to(tmp253, [XBLOCK])
        tmp240 = tmp239 - tmp238
        tmp241 = 0.09999999999999998
        tmp242 = tmp240 * tmp241
        tmp243 = tmp238 + tmp242
        tmp245 = 0.999
        tmp246 = tmp244 * tmp245
        tmp247 = tmp239 * tmp239
        tmp248 = 0.0010000000000000009
        tmp249 = tmp247 * tmp248
        tmp250 = tmp246 + tmp249
        tmp252 = libdevice.sqrt(tmp250)
        tmp255 = libdevice.pow(tmp245, tmp254)
        tmp256 = 1.0
        tmp257 = tmp255 - tmp256
        tmp258 = -tmp257
        tmp259 = libdevice.sqrt(tmp258)
        tmp260 = tmp252 / tmp259
        tmp261 = 1e-08
        tmp262 = tmp260 + tmp261
        tmp263 = 0.9
        tmp264 = libdevice.pow(tmp263, tmp254)
        tmp265 = tmp264 - tmp256
        tmp266 = 0.001
        tmp267 = tmp265 / tmp266
        tmp268 = 1 / tmp267
        tmp269 = tmp262 / tmp268
        tmp270 = tmp243 / tmp269
        tmp271 = tmp251 + tmp270
        tl.store(out_ptr28 + (x7), tmp243, xmask)
        tl.store(out_ptr30 + (x7), tmp271, xmask)
        tl.store(out_ptr31 + (x7), tmp250, xmask)
    elif xpid >= 198 and xpid < 342:
        xpid_offset = xpid - 198
        xnumel = 147456
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x8 = xindex
        tmp272 = tl.load(in_ptr40 + (x8), None)
        tmp273 = tl.load(in_ptr41 + (x8), None)
        tmp278 = tl.load(in_ptr42 + (x8), None)
        tmp285 = tl.load(in_ptr43 + (x8), None)
        tmp287 = tl.load(in_ptr44 + (0))
        tmp288 = tl.broadcast_to(tmp287, [XBLOCK])
        tmp274 = tmp273 - tmp272
        tmp275 = 0.09999999999999998
        tmp276 = tmp274 * tmp275
        tmp277 = tmp272 + tmp276
        tmp279 = 0.999
        tmp280 = tmp278 * tmp279
        tmp281 = tmp273 * tmp273
        tmp282 = 0.0010000000000000009
        tmp283 = tmp281 * tmp282
        tmp284 = tmp280 + tmp283
        tmp286 = libdevice.sqrt(tmp284)
        tmp289 = libdevice.pow(tmp279, tmp288)
        tmp290 = 1.0
        tmp291 = tmp289 - tmp290
        tmp292 = -tmp291
        tmp293 = libdevice.sqrt(tmp292)
        tmp294 = tmp286 / tmp293
        tmp295 = 1e-08
        tmp296 = tmp294 + tmp295
        tmp297 = 0.9
        tmp298 = libdevice.pow(tmp297, tmp288)
        tmp299 = tmp298 - tmp290
        tmp300 = 0.001
        tmp301 = tmp299 / tmp300
        tmp302 = 1 / tmp301
        tmp303 = tmp296 / tmp302
        tmp304 = tmp277 / tmp303
        tmp305 = tmp285 + tmp304
        tl.store(out_ptr32 + (x8), tmp277, None)
        tl.store(out_ptr34 + (x8), tmp305, None)
        tl.store(out_ptr35 + (x8), tmp284, None)
    elif xpid >= 342 and xpid < 343:
        xpid_offset = xpid - 342
        xnumel = 128
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x9 = xindex
        tmp306 = tl.load(in_ptr45 + (x9), xmask)
        tmp307 = tl.load(in_ptr46 + (x9), xmask)
        tmp312 = tl.load(in_ptr47 + (x9), xmask)
        tmp319 = tl.load(in_ptr48 + (x9), xmask)
        tmp321 = tl.load(in_ptr49 + (0))
        tmp322 = tl.broadcast_to(tmp321, [XBLOCK])
        tmp308 = tmp307 - tmp306
        tmp309 = 0.09999999999999998
        tmp310 = tmp308 * tmp309
        tmp311 = tmp306 + tmp310
        tmp313 = 0.999
        tmp314 = tmp312 * tmp313
        tmp315 = tmp307 * tmp307
        tmp316 = 0.0010000000000000009
        tmp317 = tmp315 * tmp316
        tmp318 = tmp314 + tmp317
        tmp320 = libdevice.sqrt(tmp318)
        tmp323 = libdevice.pow(tmp313, tmp322)
        tmp324 = 1.0
        tmp325 = tmp323 - tmp324
        tmp326 = -tmp325
        tmp327 = libdevice.sqrt(tmp326)
        tmp328 = tmp320 / tmp327
        tmp329 = 1e-08
        tmp330 = tmp328 + tmp329
        tmp331 = 0.9
        tmp332 = libdevice.pow(tmp331, tmp322)
        tmp333 = tmp332 - tmp324
        tmp334 = 0.001
        tmp335 = tmp333 / tmp334
        tmp336 = 1 / tmp335
        tmp337 = tmp330 / tmp336
        tmp338 = tmp311 / tmp337
        tmp339 = tmp319 + tmp338
        tl.store(out_ptr36 + (x9), tmp311, xmask)
        tl.store(out_ptr38 + (x9), tmp339, xmask)
        tl.store(out_ptr39 + (x9), tmp318, xmask)
    elif xpid >= 343 and xpid < 344:
        xpid_offset = xpid - 343
        xnumel = 128
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x10 = xindex
        tmp340 = tl.load(in_ptr50 + (x10), xmask)
        tmp341 = tl.load(in_ptr51 + (x10), xmask)
        tmp346 = tl.load(in_ptr52 + (x10), xmask)
        tmp353 = tl.load(in_ptr53 + (x10), xmask)
        tmp355 = tl.load(in_ptr54 + (0))
        tmp356 = tl.broadcast_to(tmp355, [XBLOCK])
        tmp342 = tmp341 - tmp340
        tmp343 = 0.09999999999999998
        tmp344 = tmp342 * tmp343
        tmp345 = tmp340 + tmp344
        tmp347 = 0.999
        tmp348 = tmp346 * tmp347
        tmp349 = tmp341 * tmp341
        tmp350 = 0.0010000000000000009
        tmp351 = tmp349 * tmp350
        tmp352 = tmp348 + tmp351
        tmp354 = libdevice.sqrt(tmp352)
        tmp357 = libdevice.pow(tmp347, tmp356)
        tmp358 = 1.0
        tmp359 = tmp357 - tmp358
        tmp360 = -tmp359
        tmp361 = libdevice.sqrt(tmp360)
        tmp362 = tmp354 / tmp361
        tmp363 = 1e-08
        tmp364 = tmp362 + tmp363
        tmp365 = 0.9
        tmp366 = libdevice.pow(tmp365, tmp356)
        tmp367 = tmp366 - tmp358
        tmp368 = 0.001
        tmp369 = tmp367 / tmp368
        tmp370 = 1 / tmp369
        tmp371 = tmp364 / tmp370
        tmp372 = tmp345 / tmp371
        tmp373 = tmp353 + tmp372
        tl.store(out_ptr40 + (x10), tmp345, xmask)
        tl.store(out_ptr42 + (x10), tmp373, xmask)
        tl.store(out_ptr43 + (x10), tmp352, xmask)
    elif xpid >= 344 and xpid < 408:
        xpid_offset = xpid - 344
        xnumel = 65536
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x11 = xindex
        tmp374 = tl.load(in_ptr55 + (x11), None)
        tmp375 = tl.load(in_ptr56 + (x11), None)
        tmp380 = tl.load(in_ptr57 + (x11), None)
        tmp387 = tl.load(in_ptr58 + (x11), None)
        tmp389 = tl.load(in_ptr59 + (0))
        tmp390 = tl.broadcast_to(tmp389, [XBLOCK])
        tmp376 = tmp375 - tmp374
        tmp377 = 0.09999999999999998
        tmp378 = tmp376 * tmp377
        tmp379 = tmp374 + tmp378
        tmp381 = 0.999
        tmp382 = tmp380 * tmp381
        tmp383 = tmp375 * tmp375
        tmp384 = 0.0010000000000000009
        tmp385 = tmp383 * tmp384
        tmp386 = tmp382 + tmp385
        tmp388 = libdevice.sqrt(tmp386)
        tmp391 = libdevice.pow(tmp381, tmp390)
        tmp392 = 1.0
        tmp393 = tmp391 - tmp392
        tmp394 = -tmp393
        tmp395 = libdevice.sqrt(tmp394)
        tmp396 = tmp388 / tmp395
        tmp397 = 1e-08
        tmp398 = tmp396 + tmp397
        tmp399 = 0.9
        tmp400 = libdevice.pow(tmp399, tmp390)
        tmp401 = tmp400 - tmp392
        tmp402 = 0.001
        tmp403 = tmp401 / tmp402
        tmp404 = 1 / tmp403
        tmp405 = tmp398 / tmp404
        tmp406 = tmp379 / tmp405
        tmp407 = tmp387 + tmp406
        tl.store(out_ptr44 + (x11), tmp379, None)
        tl.store(out_ptr46 + (x11), tmp407, None)
        tl.store(out_ptr47 + (x11), tmp386, None)
    elif xpid >= 408 and xpid < 409:
        xpid_offset = xpid - 408
        xnumel = 512
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x12 = xindex
        tmp408 = tl.load(in_ptr60 + (x12), xmask)
        tmp409 = tl.load(in_ptr61 + (x12), xmask)
        tmp414 = tl.load(in_ptr62 + (x12), xmask)
        tmp421 = tl.load(in_ptr63 + (x12), xmask)
        tmp423 = tl.load(in_ptr64 + (0))
        tmp424 = tl.broadcast_to(tmp423, [XBLOCK])
        tmp410 = tmp409 - tmp408
        tmp411 = 0.09999999999999998
        tmp412 = tmp410 * tmp411
        tmp413 = tmp408 + tmp412
        tmp415 = 0.999
        tmp416 = tmp414 * tmp415
        tmp417 = tmp409 * tmp409
        tmp418 = 0.0010000000000000009
        tmp419 = tmp417 * tmp418
        tmp420 = tmp416 + tmp419
        tmp422 = libdevice.sqrt(tmp420)
        tmp425 = libdevice.pow(tmp415, tmp424)
        tmp426 = 1.0
        tmp427 = tmp425 - tmp426
        tmp428 = -tmp427
        tmp429 = libdevice.sqrt(tmp428)
        tmp430 = tmp422 / tmp429
        tmp431 = 1e-08
        tmp432 = tmp430 + tmp431
        tmp433 = 0.9
        tmp434 = libdevice.pow(tmp433, tmp424)
        tmp435 = tmp434 - tmp426
        tmp436 = 0.001
        tmp437 = tmp435 / tmp436
        tmp438 = 1 / tmp437
        tmp439 = tmp432 / tmp438
        tmp440 = tmp413 / tmp439
        tmp441 = tmp421 + tmp440
        tl.store(out_ptr48 + (x12), tmp413, xmask)
        tl.store(out_ptr50 + (x12), tmp441, xmask)
        tl.store(out_ptr51 + (x12), tmp420, xmask)
    elif xpid >= 409 and xpid < 410:
        xpid_offset = xpid - 409
        xnumel = 512
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x13 = xindex
        tmp442 = tl.load(in_ptr65 + (x13), xmask)
        tmp443 = tl.load(in_ptr66 + (x13), xmask)
        tmp448 = tl.load(in_ptr67 + (x13), xmask)
        tmp455 = tl.load(in_ptr68 + (x13), xmask)
        tmp457 = tl.load(in_ptr69 + (0))
        tmp458 = tl.broadcast_to(tmp457, [XBLOCK])
        tmp444 = tmp443 - tmp442
        tmp445 = 0.09999999999999998
        tmp446 = tmp444 * tmp445
        tmp447 = tmp442 + tmp446
        tmp449 = 0.999
        tmp450 = tmp448 * tmp449
        tmp451 = tmp443 * tmp443
        tmp452 = 0.0010000000000000009
        tmp453 = tmp451 * tmp452
        tmp454 = tmp450 + tmp453
        tmp456 = libdevice.sqrt(tmp454)
        tmp459 = libdevice.pow(tmp449, tmp458)
        tmp460 = 1.0
        tmp461 = tmp459 - tmp460
        tmp462 = -tmp461
        tmp463 = libdevice.sqrt(tmp462)
        tmp464 = tmp456 / tmp463
        tmp465 = 1e-08
        tmp466 = tmp464 + tmp465
        tmp467 = 0.9
        tmp468 = libdevice.pow(tmp467, tmp458)
        tmp469 = tmp468 - tmp460
        tmp470 = 0.001
        tmp471 = tmp469 / tmp470
        tmp472 = 1 / tmp471
        tmp473 = tmp466 / tmp472
        tmp474 = tmp447 / tmp473
        tmp475 = tmp455 + tmp474
        tl.store(out_ptr52 + (x13), tmp447, xmask)
        tl.store(out_ptr54 + (x13), tmp475, xmask)
        tl.store(out_ptr55 + (x13), tmp454, xmask)
    elif xpid >= 410 and xpid < 474:
        xpid_offset = xpid - 410
        xnumel = 65536
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x14 = xindex
        tmp476 = tl.load(in_ptr70 + (x14), None)
        tmp477 = tl.load(in_ptr71 + (x14), None)
        tmp482 = tl.load(in_ptr72 + (x14), None)
        tmp489 = tl.load(in_ptr73 + (x14), None)
        tmp491 = tl.load(in_ptr74 + (0))
        tmp492 = tl.broadcast_to(tmp491, [XBLOCK])
        tmp478 = tmp477 - tmp476
        tmp479 = 0.09999999999999998
        tmp480 = tmp478 * tmp479
        tmp481 = tmp476 + tmp480
        tmp483 = 0.999
        tmp484 = tmp482 * tmp483
        tmp485 = tmp477 * tmp477
        tmp486 = 0.0010000000000000009
        tmp487 = tmp485 * tmp486
        tmp488 = tmp484 + tmp487
        tmp490 = libdevice.sqrt(tmp488)
        tmp493 = libdevice.pow(tmp483, tmp492)
        tmp494 = 1.0
        tmp495 = tmp493 - tmp494
        tmp496 = -tmp495
        tmp497 = libdevice.sqrt(tmp496)
        tmp498 = tmp490 / tmp497
        tmp499 = 1e-08
        tmp500 = tmp498 + tmp499
        tmp501 = 0.9
        tmp502 = libdevice.pow(tmp501, tmp492)
        tmp503 = tmp502 - tmp494
        tmp504 = 0.001
        tmp505 = tmp503 / tmp504
        tmp506 = 1 / tmp505
        tmp507 = tmp500 / tmp506
        tmp508 = tmp481 / tmp507
        tmp509 = tmp489 + tmp508
        tl.store(out_ptr56 + (x14), tmp481, None)
        tl.store(out_ptr58 + (x14), tmp509, None)
        tl.store(out_ptr59 + (x14), tmp488, None)
    elif xpid >= 474 and xpid < 475:
        xpid_offset = xpid - 474
        xnumel = 128
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x15 = xindex
        tmp510 = tl.load(in_ptr75 + (x15), xmask)
        tmp511 = tl.load(in_ptr76 + (x15), xmask)
        tmp516 = tl.load(in_ptr77 + (x15), xmask)
        tmp523 = tl.load(in_ptr78 + (x15), xmask)
        tmp525 = tl.load(in_ptr79 + (0))
        tmp526 = tl.broadcast_to(tmp525, [XBLOCK])
        tmp512 = tmp511 - tmp510
        tmp513 = 0.09999999999999998
        tmp514 = tmp512 * tmp513
        tmp515 = tmp510 + tmp514
        tmp517 = 0.999
        tmp518 = tmp516 * tmp517
        tmp519 = tmp511 * tmp511
        tmp520 = 0.0010000000000000009
        tmp521 = tmp519 * tmp520
        tmp522 = tmp518 + tmp521
        tmp524 = libdevice.sqrt(tmp522)
        tmp527 = libdevice.pow(tmp517, tmp526)
        tmp528 = 1.0
        tmp529 = tmp527 - tmp528
        tmp530 = -tmp529
        tmp531 = libdevice.sqrt(tmp530)
        tmp532 = tmp524 / tmp531
        tmp533 = 1e-08
        tmp534 = tmp532 + tmp533
        tmp535 = 0.9
        tmp536 = libdevice.pow(tmp535, tmp526)
        tmp537 = tmp536 - tmp528
        tmp538 = 0.001
        tmp539 = tmp537 / tmp538
        tmp540 = 1 / tmp539
        tmp541 = tmp534 / tmp540
        tmp542 = tmp515 / tmp541
        tmp543 = tmp523 + tmp542
        tl.store(out_ptr60 + (x15), tmp515, xmask)
        tl.store(out_ptr62 + (x15), tmp543, xmask)
        tl.store(out_ptr63 + (x15), tmp522, xmask)
    elif xpid >= 475 and xpid < 476:
        xpid_offset = xpid - 475
        xnumel = 128
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x16 = xindex
        tmp544 = tl.load(in_ptr80 + (x16), xmask)
        tmp545 = tl.load(in_ptr81 + (x16), xmask)
        tmp550 = tl.load(in_ptr82 + (x16), xmask)
        tmp557 = tl.load(in_ptr83 + (x16), xmask)
        tmp559 = tl.load(in_ptr84 + (0))
        tmp560 = tl.broadcast_to(tmp559, [XBLOCK])
        tmp546 = tmp545 - tmp544
        tmp547 = 0.09999999999999998
        tmp548 = tmp546 * tmp547
        tmp549 = tmp544 + tmp548
        tmp551 = 0.999
        tmp552 = tmp550 * tmp551
        tmp553 = tmp545 * tmp545
        tmp554 = 0.0010000000000000009
        tmp555 = tmp553 * tmp554
        tmp556 = tmp552 + tmp555
        tmp558 = libdevice.sqrt(tmp556)
        tmp561 = libdevice.pow(tmp551, tmp560)
        tmp562 = 1.0
        tmp563 = tmp561 - tmp562
        tmp564 = -tmp563
        tmp565 = libdevice.sqrt(tmp564)
        tmp566 = tmp558 / tmp565
        tmp567 = 1e-08
        tmp568 = tmp566 + tmp567
        tmp569 = 0.9
        tmp570 = libdevice.pow(tmp569, tmp560)
        tmp571 = tmp570 - tmp562
        tmp572 = 0.001
        tmp573 = tmp571 / tmp572
        tmp574 = 1 / tmp573
        tmp575 = tmp568 / tmp574
        tmp576 = tmp549 / tmp575
        tmp577 = tmp557 + tmp576
        tl.store(out_ptr64 + (x16), tmp549, xmask)
        tl.store(out_ptr66 + (x16), tmp577, xmask)
        tl.store(out_ptr67 + (x16), tmp556, xmask)
    elif xpid >= 476 and xpid < 620:
        xpid_offset = xpid - 476
        xnumel = 147456
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x17 = xindex
        tmp578 = tl.load(in_ptr85 + (x17), None)
        tmp579 = tl.load(in_ptr86 + (x17), None)
        tmp584 = tl.load(in_ptr87 + (x17), None)
        tmp591 = tl.load(in_ptr88 + (x17), None)
        tmp593 = tl.load(in_ptr89 + (0))
        tmp594 = tl.broadcast_to(tmp593, [XBLOCK])
        tmp580 = tmp579 - tmp578
        tmp581 = 0.09999999999999998
        tmp582 = tmp580 * tmp581
        tmp583 = tmp578 + tmp582
        tmp585 = 0.999
        tmp586 = tmp584 * tmp585
        tmp587 = tmp579 * tmp579
        tmp588 = 0.0010000000000000009
        tmp589 = tmp587 * tmp588
        tmp590 = tmp586 + tmp589
        tmp592 = libdevice.sqrt(tmp590)
        tmp595 = libdevice.pow(tmp585, tmp594)
        tmp596 = 1.0
        tmp597 = tmp595 - tmp596
        tmp598 = -tmp597
        tmp599 = libdevice.sqrt(tmp598)
        tmp600 = tmp592 / tmp599
        tmp601 = 1e-08
        tmp602 = tmp600 + tmp601
        tmp603 = 0.9
        tmp604 = libdevice.pow(tmp603, tmp594)
        tmp605 = tmp604 - tmp596
        tmp606 = 0.001
        tmp607 = tmp605 / tmp606
        tmp608 = 1 / tmp607
        tmp609 = tmp602 / tmp608
        tmp610 = tmp583 / tmp609
        tmp611 = tmp591 + tmp610
        tl.store(out_ptr68 + (x17), tmp583, None)
        tl.store(out_ptr70 + (x17), tmp611, None)
        tl.store(out_ptr71 + (x17), tmp590, None)
    elif xpid >= 620 and xpid < 621:
        xpid_offset = xpid - 620
        xnumel = 128
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x18 = xindex
        tmp612 = tl.load(in_ptr90 + (x18), xmask)
        tmp613 = tl.load(in_ptr91 + (x18), xmask)
        tmp618 = tl.load(in_ptr92 + (x18), xmask)
        tmp625 = tl.load(in_ptr93 + (x18), xmask)
        tmp627 = tl.load(in_ptr94 + (0))
        tmp628 = tl.broadcast_to(tmp627, [XBLOCK])
        tmp614 = tmp613 - tmp612
        tmp615 = 0.09999999999999998
        tmp616 = tmp614 * tmp615
        tmp617 = tmp612 + tmp616
        tmp619 = 0.999
        tmp620 = tmp618 * tmp619
        tmp621 = tmp613 * tmp613
        tmp622 = 0.0010000000000000009
        tmp623 = tmp621 * tmp622
        tmp624 = tmp620 + tmp623
        tmp626 = libdevice.sqrt(tmp624)
        tmp629 = libdevice.pow(tmp619, tmp628)
        tmp630 = 1.0
        tmp631 = tmp629 - tmp630
        tmp632 = -tmp631
        tmp633 = libdevice.sqrt(tmp632)
        tmp634 = tmp626 / tmp633
        tmp635 = 1e-08
        tmp636 = tmp634 + tmp635
        tmp637 = 0.9
        tmp638 = libdevice.pow(tmp637, tmp628)
        tmp639 = tmp638 - tmp630
        tmp640 = 0.001
        tmp641 = tmp639 / tmp640
        tmp642 = 1 / tmp641
        tmp643 = tmp636 / tmp642
        tmp644 = tmp617 / tmp643
        tmp645 = tmp625 + tmp644
        tl.store(out_ptr72 + (x18), tmp617, xmask)
        tl.store(out_ptr74 + (x18), tmp645, xmask)
        tl.store(out_ptr75 + (x18), tmp624, xmask)
    elif xpid >= 621 and xpid < 622:
        xpid_offset = xpid - 621
        xnumel = 128
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x19 = xindex
        tmp646 = tl.load(in_ptr95 + (x19), xmask)
        tmp647 = tl.load(in_ptr96 + (x19), xmask)
        tmp652 = tl.load(in_ptr97 + (x19), xmask)
        tmp659 = tl.load(in_ptr98 + (x19), xmask)
        tmp661 = tl.load(in_ptr99 + (0))
        tmp662 = tl.broadcast_to(tmp661, [XBLOCK])
        tmp648 = tmp647 - tmp646
        tmp649 = 0.09999999999999998
        tmp650 = tmp648 * tmp649
        tmp651 = tmp646 + tmp650
        tmp653 = 0.999
        tmp654 = tmp652 * tmp653
        tmp655 = tmp647 * tmp647
        tmp656 = 0.0010000000000000009
        tmp657 = tmp655 * tmp656
        tmp658 = tmp654 + tmp657
        tmp660 = libdevice.sqrt(tmp658)
        tmp663 = libdevice.pow(tmp653, tmp662)
        tmp664 = 1.0
        tmp665 = tmp663 - tmp664
        tmp666 = -tmp665
        tmp667 = libdevice.sqrt(tmp666)
        tmp668 = tmp660 / tmp667
        tmp669 = 1e-08
        tmp670 = tmp668 + tmp669
        tmp671 = 0.9
        tmp672 = libdevice.pow(tmp671, tmp662)
        tmp673 = tmp672 - tmp664
        tmp674 = 0.001
        tmp675 = tmp673 / tmp674
        tmp676 = 1 / tmp675
        tmp677 = tmp670 / tmp676
        tmp678 = tmp651 / tmp677
        tmp679 = tmp659 + tmp678
        tl.store(out_ptr76 + (x19), tmp651, xmask)
        tl.store(out_ptr78 + (x19), tmp679, xmask)
        tl.store(out_ptr79 + (x19), tmp658, xmask)
    else:
        pass
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/of/cofyviphgrqjgbiy5mog3faqmbqstkmuurenf7gpo7klurvqkwn6.py
# Source Nodes: [], Original ATen: []

triton_for_fused_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.foreach(
    num_warps=8,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: '*fp32', 35: '*fp32', 36: '*fp32', 37: '*fp32', 38: '*fp32', 39: '*fp32', 40: '*fp32', 41: '*fp32', 42: '*fp32', 43: '*fp32', 44: '*fp32', 45: '*fp32', 46: '*fp32', 47: '*fp32', 48: '*fp32', 49: '*fp32', 50: '*fp32', 51: '*fp32', 52: '*fp32', 53: '*fp32', 54: '*fp32', 55: '*fp32', 56: '*fp32', 57: '*fp32', 58: '*fp32', 59: '*fp32', 60: '*fp32', 61: '*fp32', 62: '*fp32', 63: '*fp32', 64: '*fp32', 65: '*fp32', 66: '*fp32', 67: '*fp32', 68: '*fp32', 69: '*fp32', 70: '*fp32', 71: '*fp32', 72: '*fp32', 73: '*fp32', 74: '*fp32', 75: '*fp32', 76: '*fp32', 77: '*fp32', 78: '*fp32', 79: '*fp32', 80: '*fp32', 81: '*fp32', 82: '*fp32', 83: '*fp32', 84: '*fp32', 85: '*fp32', 86: '*fp32', 87: '*fp32', 88: '*fp32', 89: '*fp32', 90: '*fp32', 91: '*fp32', 92: '*fp32', 93: '*fp32', 94: '*fp32', 95: '*fp32', 96: '*fp32', 97: '*fp32', 98: '*fp32', 99: '*fp32', 100: '*fp32', 101: '*fp32', 102: '*fp32', 103: '*fp32', 104: '*fp32', 105: '*fp32', 106: '*fp32', 107: '*fp32', 108: '*fp32', 109: '*fp32', 110: '*fp32', 111: '*fp32', 112: '*fp32', 113: '*fp32', 114: '*fp32', 115: '*fp32', 116: '*fp32', 117: '*fp32', 118: '*fp32', 119: '*fp32', 120: '*fp32', 121: '*fp32', 122: '*fp32', 123: '*fp32', 124: '*fp32', 125: '*fp32', 126: '*fp32', 127: '*fp32', 128: '*fp32', 129: '*fp32', 130: '*fp32', 131: '*fp32', 132: '*fp32', 133: '*fp32', 134: '*fp32', 135: '*fp32', 136: '*fp32', 137: '*fp32', 138: '*fp32', 139: '*fp32', 140: '*fp32', 141: '*fp32', 142: '*fp32', 143: '*fp32', 144: '*fp32', 145: '*fp32', 146: '*fp32', 147: '*fp32', 148: '*fp32', 149: '*fp32', 150: '*fp32', 151: '*fp32', 152: '*fp32', 153: '*fp32', 154: '*fp32', 155: '*fp32', 156: '*fp32', 157: '*fp32', 158: '*fp32', 159: '*fp32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'kernel_name': 'triton_for_fused_5', 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, in_ptr47, in_ptr48, in_ptr49, in_ptr50, in_ptr51, in_ptr52, in_ptr53, in_ptr54, in_ptr55, in_ptr56, in_ptr57, in_ptr58, in_ptr59, in_ptr60, in_ptr61, in_ptr62, in_ptr63, in_ptr64, in_ptr65, in_ptr66, in_ptr67, in_ptr68, in_ptr69, in_ptr70, in_ptr71, in_ptr72, in_ptr73, in_ptr74, in_ptr75, in_ptr76, in_ptr77, in_ptr78, in_ptr79, in_ptr80, in_ptr81, in_ptr82, in_ptr83, in_ptr84, in_ptr85, in_ptr86, in_ptr87, in_ptr88, in_ptr89, in_ptr90, in_ptr91, in_ptr92, in_ptr93, in_ptr94, in_ptr95, in_ptr96, in_ptr97, in_ptr98, in_ptr99, out_ptr0, out_ptr2, out_ptr3, out_ptr4, out_ptr6, out_ptr7, out_ptr8, out_ptr10, out_ptr11, out_ptr12, out_ptr14, out_ptr15, out_ptr16, out_ptr18, out_ptr19, out_ptr20, out_ptr22, out_ptr23, out_ptr24, out_ptr26, out_ptr27, out_ptr28, out_ptr30, out_ptr31, out_ptr32, out_ptr34, out_ptr35, out_ptr36, out_ptr38, out_ptr39, out_ptr40, out_ptr42, out_ptr43, out_ptr44, out_ptr46, out_ptr47, out_ptr48, out_ptr50, out_ptr51, out_ptr52, out_ptr54, out_ptr55, out_ptr56, out_ptr58, out_ptr59, out_ptr60, out_ptr62, out_ptr63, out_ptr64, out_ptr66, out_ptr67, out_ptr68, out_ptr70, out_ptr71, out_ptr72, out_ptr74, out_ptr75, out_ptr76, out_ptr78, out_ptr79):
    xpid = tl.program_id(0)
    XBLOCK: tl.constexpr = 1024
    if xpid >= 0 and xpid < 64:
        xpid_offset = xpid - 0
        xnumel = 65536
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
        tmp14 = libdevice.sqrt(tmp12)
        tmp17 = libdevice.pow(tmp7, tmp16)
        tmp18 = 1.0
        tmp19 = tmp17 - tmp18
        tmp20 = -tmp19
        tmp21 = libdevice.sqrt(tmp20)
        tmp22 = tmp14 / tmp21
        tmp23 = 1e-08
        tmp24 = tmp22 + tmp23
        tmp25 = 0.9
        tmp26 = libdevice.pow(tmp25, tmp16)
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
    elif xpid >= 64 and xpid < 65:
        xpid_offset = xpid - 64
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
        tmp48 = libdevice.sqrt(tmp46)
        tmp51 = libdevice.pow(tmp41, tmp50)
        tmp52 = 1.0
        tmp53 = tmp51 - tmp52
        tmp54 = -tmp53
        tmp55 = libdevice.sqrt(tmp54)
        tmp56 = tmp48 / tmp55
        tmp57 = 1e-08
        tmp58 = tmp56 + tmp57
        tmp59 = 0.9
        tmp60 = libdevice.pow(tmp59, tmp50)
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
    elif xpid >= 65 and xpid < 66:
        xpid_offset = xpid - 65
        xnumel = 512
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x2 = xindex
        tmp68 = tl.load(in_ptr10 + (x2), xmask)
        tmp69 = tl.load(in_ptr11 + (x2), xmask)
        tmp74 = tl.load(in_ptr12 + (x2), xmask)
        tmp81 = tl.load(in_ptr13 + (x2), xmask)
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
        tmp82 = libdevice.sqrt(tmp80)
        tmp85 = libdevice.pow(tmp75, tmp84)
        tmp86 = 1.0
        tmp87 = tmp85 - tmp86
        tmp88 = -tmp87
        tmp89 = libdevice.sqrt(tmp88)
        tmp90 = tmp82 / tmp89
        tmp91 = 1e-08
        tmp92 = tmp90 + tmp91
        tmp93 = 0.9
        tmp94 = libdevice.pow(tmp93, tmp84)
        tmp95 = tmp94 - tmp86
        tmp96 = 0.001
        tmp97 = tmp95 / tmp96
        tmp98 = 1 / tmp97
        tmp99 = tmp92 / tmp98
        tmp100 = tmp73 / tmp99
        tmp101 = tmp81 + tmp100
        tl.store(out_ptr8 + (x2), tmp73, xmask)
        tl.store(out_ptr10 + (x2), tmp101, xmask)
        tl.store(out_ptr11 + (x2), tmp80, xmask)
    elif xpid >= 66 and xpid < 130:
        xpid_offset = xpid - 66
        xnumel = 65536
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x3 = xindex
        tmp102 = tl.load(in_ptr15 + (x3), None)
        tmp103 = tl.load(in_ptr16 + (x3), None)
        tmp108 = tl.load(in_ptr17 + (x3), None)
        tmp115 = tl.load(in_ptr18 + (x3), None)
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
        tmp116 = libdevice.sqrt(tmp114)
        tmp119 = libdevice.pow(tmp109, tmp118)
        tmp120 = 1.0
        tmp121 = tmp119 - tmp120
        tmp122 = -tmp121
        tmp123 = libdevice.sqrt(tmp122)
        tmp124 = tmp116 / tmp123
        tmp125 = 1e-08
        tmp126 = tmp124 + tmp125
        tmp127 = 0.9
        tmp128 = libdevice.pow(tmp127, tmp118)
        tmp129 = tmp128 - tmp120
        tmp130 = 0.001
        tmp131 = tmp129 / tmp130
        tmp132 = 1 / tmp131
        tmp133 = tmp126 / tmp132
        tmp134 = tmp107 / tmp133
        tmp135 = tmp115 + tmp134
        tl.store(out_ptr12 + (x3), tmp107, None)
        tl.store(out_ptr14 + (x3), tmp135, None)
        tl.store(out_ptr15 + (x3), tmp114, None)
    elif xpid >= 130 and xpid < 131:
        xpid_offset = xpid - 130
        xnumel = 128
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
        tmp150 = libdevice.sqrt(tmp148)
        tmp153 = libdevice.pow(tmp143, tmp152)
        tmp154 = 1.0
        tmp155 = tmp153 - tmp154
        tmp156 = -tmp155
        tmp157 = libdevice.sqrt(tmp156)
        tmp158 = tmp150 / tmp157
        tmp159 = 1e-08
        tmp160 = tmp158 + tmp159
        tmp161 = 0.9
        tmp162 = libdevice.pow(tmp161, tmp152)
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
    elif xpid >= 131 and xpid < 132:
        xpid_offset = xpid - 131
        xnumel = 128
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
        tmp184 = libdevice.sqrt(tmp182)
        tmp187 = libdevice.pow(tmp177, tmp186)
        tmp188 = 1.0
        tmp189 = tmp187 - tmp188
        tmp190 = -tmp189
        tmp191 = libdevice.sqrt(tmp190)
        tmp192 = tmp184 / tmp191
        tmp193 = 1e-08
        tmp194 = tmp192 + tmp193
        tmp195 = 0.9
        tmp196 = libdevice.pow(tmp195, tmp186)
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
    elif xpid >= 132 and xpid < 276:
        xpid_offset = xpid - 132
        xnumel = 147456
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x6 = xindex
        tmp204 = tl.load(in_ptr30 + (x6), None)
        tmp205 = tl.load(in_ptr31 + (x6), None)
        tmp210 = tl.load(in_ptr32 + (x6), None)
        tmp217 = tl.load(in_ptr33 + (x6), None)
        tmp219 = tl.load(in_ptr34 + (0))
        tmp220 = tl.broadcast_to(tmp219, [XBLOCK])
        tmp206 = tmp205 - tmp204
        tmp207 = 0.09999999999999998
        tmp208 = tmp206 * tmp207
        tmp209 = tmp204 + tmp208
        tmp211 = 0.999
        tmp212 = tmp210 * tmp211
        tmp213 = tmp205 * tmp205
        tmp214 = 0.0010000000000000009
        tmp215 = tmp213 * tmp214
        tmp216 = tmp212 + tmp215
        tmp218 = libdevice.sqrt(tmp216)
        tmp221 = libdevice.pow(tmp211, tmp220)
        tmp222 = 1.0
        tmp223 = tmp221 - tmp222
        tmp224 = -tmp223
        tmp225 = libdevice.sqrt(tmp224)
        tmp226 = tmp218 / tmp225
        tmp227 = 1e-08
        tmp228 = tmp226 + tmp227
        tmp229 = 0.9
        tmp230 = libdevice.pow(tmp229, tmp220)
        tmp231 = tmp230 - tmp222
        tmp232 = 0.001
        tmp233 = tmp231 / tmp232
        tmp234 = 1 / tmp233
        tmp235 = tmp228 / tmp234
        tmp236 = tmp209 / tmp235
        tmp237 = tmp217 + tmp236
        tl.store(out_ptr24 + (x6), tmp209, None)
        tl.store(out_ptr26 + (x6), tmp237, None)
        tl.store(out_ptr27 + (x6), tmp216, None)
    elif xpid >= 276 and xpid < 277:
        xpid_offset = xpid - 276
        xnumel = 128
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x7 = xindex
        tmp238 = tl.load(in_ptr35 + (x7), xmask)
        tmp239 = tl.load(in_ptr36 + (x7), xmask)
        tmp244 = tl.load(in_ptr37 + (x7), xmask)
        tmp251 = tl.load(in_ptr38 + (x7), xmask)
        tmp253 = tl.load(in_ptr39 + (0))
        tmp254 = tl.broadcast_to(tmp253, [XBLOCK])
        tmp240 = tmp239 - tmp238
        tmp241 = 0.09999999999999998
        tmp242 = tmp240 * tmp241
        tmp243 = tmp238 + tmp242
        tmp245 = 0.999
        tmp246 = tmp244 * tmp245
        tmp247 = tmp239 * tmp239
        tmp248 = 0.0010000000000000009
        tmp249 = tmp247 * tmp248
        tmp250 = tmp246 + tmp249
        tmp252 = libdevice.sqrt(tmp250)
        tmp255 = libdevice.pow(tmp245, tmp254)
        tmp256 = 1.0
        tmp257 = tmp255 - tmp256
        tmp258 = -tmp257
        tmp259 = libdevice.sqrt(tmp258)
        tmp260 = tmp252 / tmp259
        tmp261 = 1e-08
        tmp262 = tmp260 + tmp261
        tmp263 = 0.9
        tmp264 = libdevice.pow(tmp263, tmp254)
        tmp265 = tmp264 - tmp256
        tmp266 = 0.001
        tmp267 = tmp265 / tmp266
        tmp268 = 1 / tmp267
        tmp269 = tmp262 / tmp268
        tmp270 = tmp243 / tmp269
        tmp271 = tmp251 + tmp270
        tl.store(out_ptr28 + (x7), tmp243, xmask)
        tl.store(out_ptr30 + (x7), tmp271, xmask)
        tl.store(out_ptr31 + (x7), tmp250, xmask)
    elif xpid >= 277 and xpid < 278:
        xpid_offset = xpid - 277
        xnumel = 128
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x8 = xindex
        tmp272 = tl.load(in_ptr40 + (x8), xmask)
        tmp273 = tl.load(in_ptr41 + (x8), xmask)
        tmp278 = tl.load(in_ptr42 + (x8), xmask)
        tmp285 = tl.load(in_ptr43 + (x8), xmask)
        tmp287 = tl.load(in_ptr44 + (0))
        tmp288 = tl.broadcast_to(tmp287, [XBLOCK])
        tmp274 = tmp273 - tmp272
        tmp275 = 0.09999999999999998
        tmp276 = tmp274 * tmp275
        tmp277 = tmp272 + tmp276
        tmp279 = 0.999
        tmp280 = tmp278 * tmp279
        tmp281 = tmp273 * tmp273
        tmp282 = 0.0010000000000000009
        tmp283 = tmp281 * tmp282
        tmp284 = tmp280 + tmp283
        tmp286 = libdevice.sqrt(tmp284)
        tmp289 = libdevice.pow(tmp279, tmp288)
        tmp290 = 1.0
        tmp291 = tmp289 - tmp290
        tmp292 = -tmp291
        tmp293 = libdevice.sqrt(tmp292)
        tmp294 = tmp286 / tmp293
        tmp295 = 1e-08
        tmp296 = tmp294 + tmp295
        tmp297 = 0.9
        tmp298 = libdevice.pow(tmp297, tmp288)
        tmp299 = tmp298 - tmp290
        tmp300 = 0.001
        tmp301 = tmp299 / tmp300
        tmp302 = 1 / tmp301
        tmp303 = tmp296 / tmp302
        tmp304 = tmp277 / tmp303
        tmp305 = tmp285 + tmp304
        tl.store(out_ptr32 + (x8), tmp277, xmask)
        tl.store(out_ptr34 + (x8), tmp305, xmask)
        tl.store(out_ptr35 + (x8), tmp284, xmask)
    elif xpid >= 278 and xpid < 342:
        xpid_offset = xpid - 278
        xnumel = 65536
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x9 = xindex
        tmp306 = tl.load(in_ptr45 + (x9), None)
        tmp307 = tl.load(in_ptr46 + (x9), None)
        tmp312 = tl.load(in_ptr47 + (x9), None)
        tmp319 = tl.load(in_ptr48 + (x9), None)
        tmp321 = tl.load(in_ptr49 + (0))
        tmp322 = tl.broadcast_to(tmp321, [XBLOCK])
        tmp308 = tmp307 - tmp306
        tmp309 = 0.09999999999999998
        tmp310 = tmp308 * tmp309
        tmp311 = tmp306 + tmp310
        tmp313 = 0.999
        tmp314 = tmp312 * tmp313
        tmp315 = tmp307 * tmp307
        tmp316 = 0.0010000000000000009
        tmp317 = tmp315 * tmp316
        tmp318 = tmp314 + tmp317
        tmp320 = libdevice.sqrt(tmp318)
        tmp323 = libdevice.pow(tmp313, tmp322)
        tmp324 = 1.0
        tmp325 = tmp323 - tmp324
        tmp326 = -tmp325
        tmp327 = libdevice.sqrt(tmp326)
        tmp328 = tmp320 / tmp327
        tmp329 = 1e-08
        tmp330 = tmp328 + tmp329
        tmp331 = 0.9
        tmp332 = libdevice.pow(tmp331, tmp322)
        tmp333 = tmp332 - tmp324
        tmp334 = 0.001
        tmp335 = tmp333 / tmp334
        tmp336 = 1 / tmp335
        tmp337 = tmp330 / tmp336
        tmp338 = tmp311 / tmp337
        tmp339 = tmp319 + tmp338
        tl.store(out_ptr36 + (x9), tmp311, None)
        tl.store(out_ptr38 + (x9), tmp339, None)
        tl.store(out_ptr39 + (x9), tmp318, None)
    elif xpid >= 342 and xpid < 343:
        xpid_offset = xpid - 342
        xnumel = 512
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x10 = xindex
        tmp340 = tl.load(in_ptr50 + (x10), xmask)
        tmp341 = tl.load(in_ptr51 + (x10), xmask)
        tmp346 = tl.load(in_ptr52 + (x10), xmask)
        tmp353 = tl.load(in_ptr53 + (x10), xmask)
        tmp355 = tl.load(in_ptr54 + (0))
        tmp356 = tl.broadcast_to(tmp355, [XBLOCK])
        tmp342 = tmp341 - tmp340
        tmp343 = 0.09999999999999998
        tmp344 = tmp342 * tmp343
        tmp345 = tmp340 + tmp344
        tmp347 = 0.999
        tmp348 = tmp346 * tmp347
        tmp349 = tmp341 * tmp341
        tmp350 = 0.0010000000000000009
        tmp351 = tmp349 * tmp350
        tmp352 = tmp348 + tmp351
        tmp354 = libdevice.sqrt(tmp352)
        tmp357 = libdevice.pow(tmp347, tmp356)
        tmp358 = 1.0
        tmp359 = tmp357 - tmp358
        tmp360 = -tmp359
        tmp361 = libdevice.sqrt(tmp360)
        tmp362 = tmp354 / tmp361
        tmp363 = 1e-08
        tmp364 = tmp362 + tmp363
        tmp365 = 0.9
        tmp366 = libdevice.pow(tmp365, tmp356)
        tmp367 = tmp366 - tmp358
        tmp368 = 0.001
        tmp369 = tmp367 / tmp368
        tmp370 = 1 / tmp369
        tmp371 = tmp364 / tmp370
        tmp372 = tmp345 / tmp371
        tmp373 = tmp353 + tmp372
        tl.store(out_ptr40 + (x10), tmp345, xmask)
        tl.store(out_ptr42 + (x10), tmp373, xmask)
        tl.store(out_ptr43 + (x10), tmp352, xmask)
    elif xpid >= 343 and xpid < 344:
        xpid_offset = xpid - 343
        xnumel = 512
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x11 = xindex
        tmp374 = tl.load(in_ptr55 + (x11), xmask)
        tmp375 = tl.load(in_ptr56 + (x11), xmask)
        tmp380 = tl.load(in_ptr57 + (x11), xmask)
        tmp387 = tl.load(in_ptr58 + (x11), xmask)
        tmp389 = tl.load(in_ptr59 + (0))
        tmp390 = tl.broadcast_to(tmp389, [XBLOCK])
        tmp376 = tmp375 - tmp374
        tmp377 = 0.09999999999999998
        tmp378 = tmp376 * tmp377
        tmp379 = tmp374 + tmp378
        tmp381 = 0.999
        tmp382 = tmp380 * tmp381
        tmp383 = tmp375 * tmp375
        tmp384 = 0.0010000000000000009
        tmp385 = tmp383 * tmp384
        tmp386 = tmp382 + tmp385
        tmp388 = libdevice.sqrt(tmp386)
        tmp391 = libdevice.pow(tmp381, tmp390)
        tmp392 = 1.0
        tmp393 = tmp391 - tmp392
        tmp394 = -tmp393
        tmp395 = libdevice.sqrt(tmp394)
        tmp396 = tmp388 / tmp395
        tmp397 = 1e-08
        tmp398 = tmp396 + tmp397
        tmp399 = 0.9
        tmp400 = libdevice.pow(tmp399, tmp390)
        tmp401 = tmp400 - tmp392
        tmp402 = 0.001
        tmp403 = tmp401 / tmp402
        tmp404 = 1 / tmp403
        tmp405 = tmp398 / tmp404
        tmp406 = tmp379 / tmp405
        tmp407 = tmp387 + tmp406
        tl.store(out_ptr44 + (x11), tmp379, xmask)
        tl.store(out_ptr46 + (x11), tmp407, xmask)
        tl.store(out_ptr47 + (x11), tmp386, xmask)
    elif xpid >= 344 and xpid < 472:
        xpid_offset = xpid - 344
        xnumel = 131072
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x12 = xindex
        tmp408 = tl.load(in_ptr60 + (x12), None)
        tmp409 = tl.load(in_ptr61 + (x12), None)
        tmp414 = tl.load(in_ptr62 + (x12), None)
        tmp421 = tl.load(in_ptr63 + (x12), None)
        tmp423 = tl.load(in_ptr64 + (0))
        tmp424 = tl.broadcast_to(tmp423, [XBLOCK])
        tmp410 = tmp409 - tmp408
        tmp411 = 0.09999999999999998
        tmp412 = tmp410 * tmp411
        tmp413 = tmp408 + tmp412
        tmp415 = 0.999
        tmp416 = tmp414 * tmp415
        tmp417 = tmp409 * tmp409
        tmp418 = 0.0010000000000000009
        tmp419 = tmp417 * tmp418
        tmp420 = tmp416 + tmp419
        tmp422 = libdevice.sqrt(tmp420)
        tmp425 = libdevice.pow(tmp415, tmp424)
        tmp426 = 1.0
        tmp427 = tmp425 - tmp426
        tmp428 = -tmp427
        tmp429 = libdevice.sqrt(tmp428)
        tmp430 = tmp422 / tmp429
        tmp431 = 1e-08
        tmp432 = tmp430 + tmp431
        tmp433 = 0.9
        tmp434 = libdevice.pow(tmp433, tmp424)
        tmp435 = tmp434 - tmp426
        tmp436 = 0.001
        tmp437 = tmp435 / tmp436
        tmp438 = 1 / tmp437
        tmp439 = tmp432 / tmp438
        tmp440 = tmp413 / tmp439
        tmp441 = tmp421 + tmp440
        tl.store(out_ptr48 + (x12), tmp413, None)
        tl.store(out_ptr50 + (x12), tmp441, None)
        tl.store(out_ptr51 + (x12), tmp420, None)
    elif xpid >= 472 and xpid < 473:
        xpid_offset = xpid - 472
        xnumel = 256
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x13 = xindex
        tmp442 = tl.load(in_ptr65 + (x13), xmask)
        tmp443 = tl.load(in_ptr66 + (x13), xmask)
        tmp448 = tl.load(in_ptr67 + (x13), xmask)
        tmp455 = tl.load(in_ptr68 + (x13), xmask)
        tmp457 = tl.load(in_ptr69 + (0))
        tmp458 = tl.broadcast_to(tmp457, [XBLOCK])
        tmp444 = tmp443 - tmp442
        tmp445 = 0.09999999999999998
        tmp446 = tmp444 * tmp445
        tmp447 = tmp442 + tmp446
        tmp449 = 0.999
        tmp450 = tmp448 * tmp449
        tmp451 = tmp443 * tmp443
        tmp452 = 0.0010000000000000009
        tmp453 = tmp451 * tmp452
        tmp454 = tmp450 + tmp453
        tmp456 = libdevice.sqrt(tmp454)
        tmp459 = libdevice.pow(tmp449, tmp458)
        tmp460 = 1.0
        tmp461 = tmp459 - tmp460
        tmp462 = -tmp461
        tmp463 = libdevice.sqrt(tmp462)
        tmp464 = tmp456 / tmp463
        tmp465 = 1e-08
        tmp466 = tmp464 + tmp465
        tmp467 = 0.9
        tmp468 = libdevice.pow(tmp467, tmp458)
        tmp469 = tmp468 - tmp460
        tmp470 = 0.001
        tmp471 = tmp469 / tmp470
        tmp472 = 1 / tmp471
        tmp473 = tmp466 / tmp472
        tmp474 = tmp447 / tmp473
        tmp475 = tmp455 + tmp474
        tl.store(out_ptr52 + (x13), tmp447, xmask)
        tl.store(out_ptr54 + (x13), tmp475, xmask)
        tl.store(out_ptr55 + (x13), tmp454, xmask)
    elif xpid >= 473 and xpid < 474:
        xpid_offset = xpid - 473
        xnumel = 256
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x14 = xindex
        tmp476 = tl.load(in_ptr70 + (x14), xmask)
        tmp477 = tl.load(in_ptr71 + (x14), xmask)
        tmp482 = tl.load(in_ptr72 + (x14), xmask)
        tmp489 = tl.load(in_ptr73 + (x14), xmask)
        tmp491 = tl.load(in_ptr74 + (0))
        tmp492 = tl.broadcast_to(tmp491, [XBLOCK])
        tmp478 = tmp477 - tmp476
        tmp479 = 0.09999999999999998
        tmp480 = tmp478 * tmp479
        tmp481 = tmp476 + tmp480
        tmp483 = 0.999
        tmp484 = tmp482 * tmp483
        tmp485 = tmp477 * tmp477
        tmp486 = 0.0010000000000000009
        tmp487 = tmp485 * tmp486
        tmp488 = tmp484 + tmp487
        tmp490 = libdevice.sqrt(tmp488)
        tmp493 = libdevice.pow(tmp483, tmp492)
        tmp494 = 1.0
        tmp495 = tmp493 - tmp494
        tmp496 = -tmp495
        tmp497 = libdevice.sqrt(tmp496)
        tmp498 = tmp490 / tmp497
        tmp499 = 1e-08
        tmp500 = tmp498 + tmp499
        tmp501 = 0.9
        tmp502 = libdevice.pow(tmp501, tmp492)
        tmp503 = tmp502 - tmp494
        tmp504 = 0.001
        tmp505 = tmp503 / tmp504
        tmp506 = 1 / tmp505
        tmp507 = tmp500 / tmp506
        tmp508 = tmp481 / tmp507
        tmp509 = tmp489 + tmp508
        tl.store(out_ptr56 + (x14), tmp481, xmask)
        tl.store(out_ptr58 + (x14), tmp509, xmask)
        tl.store(out_ptr59 + (x14), tmp488, xmask)
    elif xpid >= 474 and xpid < 1050:
        xpid_offset = xpid - 474
        xnumel = 589824
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x15 = xindex
        tmp510 = tl.load(in_ptr75 + (x15), None)
        tmp511 = tl.load(in_ptr76 + (x15), None)
        tmp516 = tl.load(in_ptr77 + (x15), None)
        tmp523 = tl.load(in_ptr78 + (x15), None)
        tmp525 = tl.load(in_ptr79 + (0))
        tmp526 = tl.broadcast_to(tmp525, [XBLOCK])
        tmp512 = tmp511 - tmp510
        tmp513 = 0.09999999999999998
        tmp514 = tmp512 * tmp513
        tmp515 = tmp510 + tmp514
        tmp517 = 0.999
        tmp518 = tmp516 * tmp517
        tmp519 = tmp511 * tmp511
        tmp520 = 0.0010000000000000009
        tmp521 = tmp519 * tmp520
        tmp522 = tmp518 + tmp521
        tmp524 = libdevice.sqrt(tmp522)
        tmp527 = libdevice.pow(tmp517, tmp526)
        tmp528 = 1.0
        tmp529 = tmp527 - tmp528
        tmp530 = -tmp529
        tmp531 = libdevice.sqrt(tmp530)
        tmp532 = tmp524 / tmp531
        tmp533 = 1e-08
        tmp534 = tmp532 + tmp533
        tmp535 = 0.9
        tmp536 = libdevice.pow(tmp535, tmp526)
        tmp537 = tmp536 - tmp528
        tmp538 = 0.001
        tmp539 = tmp537 / tmp538
        tmp540 = 1 / tmp539
        tmp541 = tmp534 / tmp540
        tmp542 = tmp515 / tmp541
        tmp543 = tmp523 + tmp542
        tl.store(out_ptr60 + (x15), tmp515, None)
        tl.store(out_ptr62 + (x15), tmp543, None)
        tl.store(out_ptr63 + (x15), tmp522, None)
    elif xpid >= 1050 and xpid < 1051:
        xpid_offset = xpid - 1050
        xnumel = 256
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x16 = xindex
        tmp544 = tl.load(in_ptr80 + (x16), xmask)
        tmp545 = tl.load(in_ptr81 + (x16), xmask)
        tmp550 = tl.load(in_ptr82 + (x16), xmask)
        tmp557 = tl.load(in_ptr83 + (x16), xmask)
        tmp559 = tl.load(in_ptr84 + (0))
        tmp560 = tl.broadcast_to(tmp559, [XBLOCK])
        tmp546 = tmp545 - tmp544
        tmp547 = 0.09999999999999998
        tmp548 = tmp546 * tmp547
        tmp549 = tmp544 + tmp548
        tmp551 = 0.999
        tmp552 = tmp550 * tmp551
        tmp553 = tmp545 * tmp545
        tmp554 = 0.0010000000000000009
        tmp555 = tmp553 * tmp554
        tmp556 = tmp552 + tmp555
        tmp558 = libdevice.sqrt(tmp556)
        tmp561 = libdevice.pow(tmp551, tmp560)
        tmp562 = 1.0
        tmp563 = tmp561 - tmp562
        tmp564 = -tmp563
        tmp565 = libdevice.sqrt(tmp564)
        tmp566 = tmp558 / tmp565
        tmp567 = 1e-08
        tmp568 = tmp566 + tmp567
        tmp569 = 0.9
        tmp570 = libdevice.pow(tmp569, tmp560)
        tmp571 = tmp570 - tmp562
        tmp572 = 0.001
        tmp573 = tmp571 / tmp572
        tmp574 = 1 / tmp573
        tmp575 = tmp568 / tmp574
        tmp576 = tmp549 / tmp575
        tmp577 = tmp557 + tmp576
        tl.store(out_ptr64 + (x16), tmp549, xmask)
        tl.store(out_ptr66 + (x16), tmp577, xmask)
        tl.store(out_ptr67 + (x16), tmp556, xmask)
    elif xpid >= 1051 and xpid < 1052:
        xpid_offset = xpid - 1051
        xnumel = 256
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x17 = xindex
        tmp578 = tl.load(in_ptr85 + (x17), xmask)
        tmp579 = tl.load(in_ptr86 + (x17), xmask)
        tmp584 = tl.load(in_ptr87 + (x17), xmask)
        tmp591 = tl.load(in_ptr88 + (x17), xmask)
        tmp593 = tl.load(in_ptr89 + (0))
        tmp594 = tl.broadcast_to(tmp593, [XBLOCK])
        tmp580 = tmp579 - tmp578
        tmp581 = 0.09999999999999998
        tmp582 = tmp580 * tmp581
        tmp583 = tmp578 + tmp582
        tmp585 = 0.999
        tmp586 = tmp584 * tmp585
        tmp587 = tmp579 * tmp579
        tmp588 = 0.0010000000000000009
        tmp589 = tmp587 * tmp588
        tmp590 = tmp586 + tmp589
        tmp592 = libdevice.sqrt(tmp590)
        tmp595 = libdevice.pow(tmp585, tmp594)
        tmp596 = 1.0
        tmp597 = tmp595 - tmp596
        tmp598 = -tmp597
        tmp599 = libdevice.sqrt(tmp598)
        tmp600 = tmp592 / tmp599
        tmp601 = 1e-08
        tmp602 = tmp600 + tmp601
        tmp603 = 0.9
        tmp604 = libdevice.pow(tmp603, tmp594)
        tmp605 = tmp604 - tmp596
        tmp606 = 0.001
        tmp607 = tmp605 / tmp606
        tmp608 = 1 / tmp607
        tmp609 = tmp602 / tmp608
        tmp610 = tmp583 / tmp609
        tmp611 = tmp591 + tmp610
        tl.store(out_ptr68 + (x17), tmp583, xmask)
        tl.store(out_ptr70 + (x17), tmp611, xmask)
        tl.store(out_ptr71 + (x17), tmp590, xmask)
    elif xpid >= 1052 and xpid < 1308:
        xpid_offset = xpid - 1052
        xnumel = 262144
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x18 = xindex
        tmp612 = tl.load(in_ptr90 + (x18), None)
        tmp613 = tl.load(in_ptr91 + (x18), None)
        tmp618 = tl.load(in_ptr92 + (x18), None)
        tmp625 = tl.load(in_ptr93 + (x18), None)
        tmp627 = tl.load(in_ptr94 + (0))
        tmp628 = tl.broadcast_to(tmp627, [XBLOCK])
        tmp614 = tmp613 - tmp612
        tmp615 = 0.09999999999999998
        tmp616 = tmp614 * tmp615
        tmp617 = tmp612 + tmp616
        tmp619 = 0.999
        tmp620 = tmp618 * tmp619
        tmp621 = tmp613 * tmp613
        tmp622 = 0.0010000000000000009
        tmp623 = tmp621 * tmp622
        tmp624 = tmp620 + tmp623
        tmp626 = libdevice.sqrt(tmp624)
        tmp629 = libdevice.pow(tmp619, tmp628)
        tmp630 = 1.0
        tmp631 = tmp629 - tmp630
        tmp632 = -tmp631
        tmp633 = libdevice.sqrt(tmp632)
        tmp634 = tmp626 / tmp633
        tmp635 = 1e-08
        tmp636 = tmp634 + tmp635
        tmp637 = 0.9
        tmp638 = libdevice.pow(tmp637, tmp628)
        tmp639 = tmp638 - tmp630
        tmp640 = 0.001
        tmp641 = tmp639 / tmp640
        tmp642 = 1 / tmp641
        tmp643 = tmp636 / tmp642
        tmp644 = tmp617 / tmp643
        tmp645 = tmp625 + tmp644
        tl.store(out_ptr72 + (x18), tmp617, None)
        tl.store(out_ptr74 + (x18), tmp645, None)
        tl.store(out_ptr75 + (x18), tmp624, None)
    elif xpid >= 1308 and xpid < 1309:
        xpid_offset = xpid - 1308
        xnumel = 1024
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x19 = xindex
        tmp646 = tl.load(in_ptr95 + (x19), xmask)
        tmp647 = tl.load(in_ptr96 + (x19), xmask)
        tmp652 = tl.load(in_ptr97 + (x19), xmask)
        tmp659 = tl.load(in_ptr98 + (x19), xmask)
        tmp661 = tl.load(in_ptr99 + (0))
        tmp662 = tl.broadcast_to(tmp661, [XBLOCK])
        tmp648 = tmp647 - tmp646
        tmp649 = 0.09999999999999998
        tmp650 = tmp648 * tmp649
        tmp651 = tmp646 + tmp650
        tmp653 = 0.999
        tmp654 = tmp652 * tmp653
        tmp655 = tmp647 * tmp647
        tmp656 = 0.0010000000000000009
        tmp657 = tmp655 * tmp656
        tmp658 = tmp654 + tmp657
        tmp660 = libdevice.sqrt(tmp658)
        tmp663 = libdevice.pow(tmp653, tmp662)
        tmp664 = 1.0
        tmp665 = tmp663 - tmp664
        tmp666 = -tmp665
        tmp667 = libdevice.sqrt(tmp666)
        tmp668 = tmp660 / tmp667
        tmp669 = 1e-08
        tmp670 = tmp668 + tmp669
        tmp671 = 0.9
        tmp672 = libdevice.pow(tmp671, tmp662)
        tmp673 = tmp672 - tmp664
        tmp674 = 0.001
        tmp675 = tmp673 / tmp674
        tmp676 = 1 / tmp675
        tmp677 = tmp670 / tmp676
        tmp678 = tmp651 / tmp677
        tmp679 = tmp659 + tmp678
        tl.store(out_ptr76 + (x19), tmp651, xmask)
        tl.store(out_ptr78 + (x19), tmp679, xmask)
        tl.store(out_ptr79 + (x19), tmp658, xmask)
    else:
        pass
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/nu/cnu7pzyvuelq2wvvzjbjpwtxsmx56tzqeavt5zl45wf4ncqpqh44.py
# Source Nodes: [], Original ATen: []

triton_for_fused_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.foreach(
    num_warps=8,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: '*fp32', 35: '*fp32', 36: '*fp32', 37: '*fp32', 38: '*fp32', 39: '*fp32', 40: '*fp32', 41: '*fp32', 42: '*fp32', 43: '*fp32', 44: '*fp32', 45: '*fp32', 46: '*fp32', 47: '*fp32', 48: '*fp32', 49: '*fp32', 50: '*fp32', 51: '*fp32', 52: '*fp32', 53: '*fp32', 54: '*fp32', 55: '*fp32', 56: '*fp32', 57: '*fp32', 58: '*fp32', 59: '*fp32', 60: '*fp32', 61: '*fp32', 62: '*fp32', 63: '*fp32', 64: '*fp32', 65: '*fp32', 66: '*fp32', 67: '*fp32', 68: '*fp32', 69: '*fp32', 70: '*fp32', 71: '*fp32', 72: '*fp32', 73: '*fp32', 74: '*fp32', 75: '*fp32', 76: '*fp32', 77: '*fp32', 78: '*fp32', 79: '*fp32', 80: '*fp32', 81: '*fp32', 82: '*fp32', 83: '*fp32', 84: '*fp32', 85: '*fp32', 86: '*fp32', 87: '*fp32', 88: '*fp32', 89: '*fp32', 90: '*fp32', 91: '*fp32', 92: '*fp32', 93: '*fp32', 94: '*fp32', 95: '*fp32', 96: '*fp32', 97: '*fp32', 98: '*fp32', 99: '*fp32', 100: '*fp32', 101: '*fp32', 102: '*fp32', 103: '*fp32', 104: '*fp32', 105: '*fp32', 106: '*fp32', 107: '*fp32', 108: '*fp32', 109: '*fp32', 110: '*fp32', 111: '*fp32', 112: '*fp32', 113: '*fp32', 114: '*fp32', 115: '*fp32', 116: '*fp32', 117: '*fp32', 118: '*fp32', 119: '*fp32', 120: '*fp32', 121: '*fp32', 122: '*fp32', 123: '*fp32', 124: '*fp32', 125: '*fp32', 126: '*fp32', 127: '*fp32', 128: '*fp32', 129: '*fp32', 130: '*fp32', 131: '*fp32', 132: '*fp32', 133: '*fp32', 134: '*fp32', 135: '*fp32', 136: '*fp32', 137: '*fp32', 138: '*fp32', 139: '*fp32', 140: '*fp32', 141: '*fp32', 142: '*fp32', 143: '*fp32', 144: '*fp32', 145: '*fp32', 146: '*fp32', 147: '*fp32', 148: '*fp32', 149: '*fp32', 150: '*fp32', 151: '*fp32', 152: '*fp32', 153: '*fp32', 154: '*fp32', 155: '*fp32', 156: '*fp32', 157: '*fp32', 158: '*fp32', 159: '*fp32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'kernel_name': 'triton_for_fused_6', 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, in_ptr47, in_ptr48, in_ptr49, in_ptr50, in_ptr51, in_ptr52, in_ptr53, in_ptr54, in_ptr55, in_ptr56, in_ptr57, in_ptr58, in_ptr59, in_ptr60, in_ptr61, in_ptr62, in_ptr63, in_ptr64, in_ptr65, in_ptr66, in_ptr67, in_ptr68, in_ptr69, in_ptr70, in_ptr71, in_ptr72, in_ptr73, in_ptr74, in_ptr75, in_ptr76, in_ptr77, in_ptr78, in_ptr79, in_ptr80, in_ptr81, in_ptr82, in_ptr83, in_ptr84, in_ptr85, in_ptr86, in_ptr87, in_ptr88, in_ptr89, in_ptr90, in_ptr91, in_ptr92, in_ptr93, in_ptr94, in_ptr95, in_ptr96, in_ptr97, in_ptr98, in_ptr99, out_ptr0, out_ptr2, out_ptr3, out_ptr4, out_ptr6, out_ptr7, out_ptr8, out_ptr10, out_ptr11, out_ptr12, out_ptr14, out_ptr15, out_ptr16, out_ptr18, out_ptr19, out_ptr20, out_ptr22, out_ptr23, out_ptr24, out_ptr26, out_ptr27, out_ptr28, out_ptr30, out_ptr31, out_ptr32, out_ptr34, out_ptr35, out_ptr36, out_ptr38, out_ptr39, out_ptr40, out_ptr42, out_ptr43, out_ptr44, out_ptr46, out_ptr47, out_ptr48, out_ptr50, out_ptr51, out_ptr52, out_ptr54, out_ptr55, out_ptr56, out_ptr58, out_ptr59, out_ptr60, out_ptr62, out_ptr63, out_ptr64, out_ptr66, out_ptr67, out_ptr68, out_ptr70, out_ptr71, out_ptr72, out_ptr74, out_ptr75, out_ptr76, out_ptr78, out_ptr79):
    xpid = tl.program_id(0)
    XBLOCK: tl.constexpr = 1024
    if xpid >= 0 and xpid < 1:
        xpid_offset = xpid - 0
        xnumel = 1024
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), xmask)
        tmp1 = tl.load(in_ptr1 + (x0), xmask)
        tmp6 = tl.load(in_ptr2 + (x0), xmask)
        tmp13 = tl.load(in_ptr3 + (x0), xmask)
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
        tmp14 = libdevice.sqrt(tmp12)
        tmp17 = libdevice.pow(tmp7, tmp16)
        tmp18 = 1.0
        tmp19 = tmp17 - tmp18
        tmp20 = -tmp19
        tmp21 = libdevice.sqrt(tmp20)
        tmp22 = tmp14 / tmp21
        tmp23 = 1e-08
        tmp24 = tmp22 + tmp23
        tmp25 = 0.9
        tmp26 = libdevice.pow(tmp25, tmp16)
        tmp27 = tmp26 - tmp18
        tmp28 = 0.001
        tmp29 = tmp27 / tmp28
        tmp30 = 1 / tmp29
        tmp31 = tmp24 / tmp30
        tmp32 = tmp5 / tmp31
        tmp33 = tmp13 + tmp32
        tl.store(out_ptr0 + (x0), tmp5, xmask)
        tl.store(out_ptr2 + (x0), tmp33, xmask)
        tl.store(out_ptr3 + (x0), tmp12, xmask)
    elif xpid >= 1 and xpid < 513:
        xpid_offset = xpid - 1
        xnumel = 524288
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x1 = xindex
        tmp34 = tl.load(in_ptr5 + (x1), None)
        tmp35 = tl.load(in_ptr6 + (x1), None)
        tmp40 = tl.load(in_ptr7 + (x1), None)
        tmp47 = tl.load(in_ptr8 + (x1), None)
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
        tmp48 = libdevice.sqrt(tmp46)
        tmp51 = libdevice.pow(tmp41, tmp50)
        tmp52 = 1.0
        tmp53 = tmp51 - tmp52
        tmp54 = -tmp53
        tmp55 = libdevice.sqrt(tmp54)
        tmp56 = tmp48 / tmp55
        tmp57 = 1e-08
        tmp58 = tmp56 + tmp57
        tmp59 = 0.9
        tmp60 = libdevice.pow(tmp59, tmp50)
        tmp61 = tmp60 - tmp52
        tmp62 = 0.001
        tmp63 = tmp61 / tmp62
        tmp64 = 1 / tmp63
        tmp65 = tmp58 / tmp64
        tmp66 = tmp39 / tmp65
        tmp67 = tmp47 + tmp66
        tl.store(out_ptr4 + (x1), tmp39, None)
        tl.store(out_ptr6 + (x1), tmp67, None)
        tl.store(out_ptr7 + (x1), tmp46, None)
    elif xpid >= 513 and xpid < 514:
        xpid_offset = xpid - 513
        xnumel = 1024
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x2 = xindex
        tmp68 = tl.load(in_ptr10 + (x2), xmask)
        tmp69 = tl.load(in_ptr11 + (x2), xmask)
        tmp74 = tl.load(in_ptr12 + (x2), xmask)
        tmp81 = tl.load(in_ptr13 + (x2), xmask)
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
        tmp82 = libdevice.sqrt(tmp80)
        tmp85 = libdevice.pow(tmp75, tmp84)
        tmp86 = 1.0
        tmp87 = tmp85 - tmp86
        tmp88 = -tmp87
        tmp89 = libdevice.sqrt(tmp88)
        tmp90 = tmp82 / tmp89
        tmp91 = 1e-08
        tmp92 = tmp90 + tmp91
        tmp93 = 0.9
        tmp94 = libdevice.pow(tmp93, tmp84)
        tmp95 = tmp94 - tmp86
        tmp96 = 0.001
        tmp97 = tmp95 / tmp96
        tmp98 = 1 / tmp97
        tmp99 = tmp92 / tmp98
        tmp100 = tmp73 / tmp99
        tmp101 = tmp81 + tmp100
        tl.store(out_ptr8 + (x2), tmp73, xmask)
        tl.store(out_ptr10 + (x2), tmp101, xmask)
        tl.store(out_ptr11 + (x2), tmp80, xmask)
    elif xpid >= 514 and xpid < 515:
        xpid_offset = xpid - 514
        xnumel = 1024
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
        tmp116 = libdevice.sqrt(tmp114)
        tmp119 = libdevice.pow(tmp109, tmp118)
        tmp120 = 1.0
        tmp121 = tmp119 - tmp120
        tmp122 = -tmp121
        tmp123 = libdevice.sqrt(tmp122)
        tmp124 = tmp116 / tmp123
        tmp125 = 1e-08
        tmp126 = tmp124 + tmp125
        tmp127 = 0.9
        tmp128 = libdevice.pow(tmp127, tmp118)
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
    elif xpid >= 515 and xpid < 771:
        xpid_offset = xpid - 515
        xnumel = 262144
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x4 = xindex
        tmp136 = tl.load(in_ptr20 + (x4), None)
        tmp137 = tl.load(in_ptr21 + (x4), None)
        tmp142 = tl.load(in_ptr22 + (x4), None)
        tmp149 = tl.load(in_ptr23 + (x4), None)
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
        tmp150 = libdevice.sqrt(tmp148)
        tmp153 = libdevice.pow(tmp143, tmp152)
        tmp154 = 1.0
        tmp155 = tmp153 - tmp154
        tmp156 = -tmp155
        tmp157 = libdevice.sqrt(tmp156)
        tmp158 = tmp150 / tmp157
        tmp159 = 1e-08
        tmp160 = tmp158 + tmp159
        tmp161 = 0.9
        tmp162 = libdevice.pow(tmp161, tmp152)
        tmp163 = tmp162 - tmp154
        tmp164 = 0.001
        tmp165 = tmp163 / tmp164
        tmp166 = 1 / tmp165
        tmp167 = tmp160 / tmp166
        tmp168 = tmp141 / tmp167
        tmp169 = tmp149 + tmp168
        tl.store(out_ptr16 + (x4), tmp141, None)
        tl.store(out_ptr18 + (x4), tmp169, None)
        tl.store(out_ptr19 + (x4), tmp148, None)
    elif xpid >= 771 and xpid < 772:
        xpid_offset = xpid - 771
        xnumel = 256
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
        tmp184 = libdevice.sqrt(tmp182)
        tmp187 = libdevice.pow(tmp177, tmp186)
        tmp188 = 1.0
        tmp189 = tmp187 - tmp188
        tmp190 = -tmp189
        tmp191 = libdevice.sqrt(tmp190)
        tmp192 = tmp184 / tmp191
        tmp193 = 1e-08
        tmp194 = tmp192 + tmp193
        tmp195 = 0.9
        tmp196 = libdevice.pow(tmp195, tmp186)
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
    elif xpid >= 772 and xpid < 773:
        xpid_offset = xpid - 772
        xnumel = 256
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x6 = xindex
        tmp204 = tl.load(in_ptr30 + (x6), xmask)
        tmp205 = tl.load(in_ptr31 + (x6), xmask)
        tmp210 = tl.load(in_ptr32 + (x6), xmask)
        tmp217 = tl.load(in_ptr33 + (x6), xmask)
        tmp219 = tl.load(in_ptr34 + (0))
        tmp220 = tl.broadcast_to(tmp219, [XBLOCK])
        tmp206 = tmp205 - tmp204
        tmp207 = 0.09999999999999998
        tmp208 = tmp206 * tmp207
        tmp209 = tmp204 + tmp208
        tmp211 = 0.999
        tmp212 = tmp210 * tmp211
        tmp213 = tmp205 * tmp205
        tmp214 = 0.0010000000000000009
        tmp215 = tmp213 * tmp214
        tmp216 = tmp212 + tmp215
        tmp218 = libdevice.sqrt(tmp216)
        tmp221 = libdevice.pow(tmp211, tmp220)
        tmp222 = 1.0
        tmp223 = tmp221 - tmp222
        tmp224 = -tmp223
        tmp225 = libdevice.sqrt(tmp224)
        tmp226 = tmp218 / tmp225
        tmp227 = 1e-08
        tmp228 = tmp226 + tmp227
        tmp229 = 0.9
        tmp230 = libdevice.pow(tmp229, tmp220)
        tmp231 = tmp230 - tmp222
        tmp232 = 0.001
        tmp233 = tmp231 / tmp232
        tmp234 = 1 / tmp233
        tmp235 = tmp228 / tmp234
        tmp236 = tmp209 / tmp235
        tmp237 = tmp217 + tmp236
        tl.store(out_ptr24 + (x6), tmp209, xmask)
        tl.store(out_ptr26 + (x6), tmp237, xmask)
        tl.store(out_ptr27 + (x6), tmp216, xmask)
    elif xpid >= 773 and xpid < 1349:
        xpid_offset = xpid - 773
        xnumel = 589824
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x7 = xindex
        tmp238 = tl.load(in_ptr35 + (x7), None)
        tmp239 = tl.load(in_ptr36 + (x7), None)
        tmp244 = tl.load(in_ptr37 + (x7), None)
        tmp251 = tl.load(in_ptr38 + (x7), None)
        tmp253 = tl.load(in_ptr39 + (0))
        tmp254 = tl.broadcast_to(tmp253, [XBLOCK])
        tmp240 = tmp239 - tmp238
        tmp241 = 0.09999999999999998
        tmp242 = tmp240 * tmp241
        tmp243 = tmp238 + tmp242
        tmp245 = 0.999
        tmp246 = tmp244 * tmp245
        tmp247 = tmp239 * tmp239
        tmp248 = 0.0010000000000000009
        tmp249 = tmp247 * tmp248
        tmp250 = tmp246 + tmp249
        tmp252 = libdevice.sqrt(tmp250)
        tmp255 = libdevice.pow(tmp245, tmp254)
        tmp256 = 1.0
        tmp257 = tmp255 - tmp256
        tmp258 = -tmp257
        tmp259 = libdevice.sqrt(tmp258)
        tmp260 = tmp252 / tmp259
        tmp261 = 1e-08
        tmp262 = tmp260 + tmp261
        tmp263 = 0.9
        tmp264 = libdevice.pow(tmp263, tmp254)
        tmp265 = tmp264 - tmp256
        tmp266 = 0.001
        tmp267 = tmp265 / tmp266
        tmp268 = 1 / tmp267
        tmp269 = tmp262 / tmp268
        tmp270 = tmp243 / tmp269
        tmp271 = tmp251 + tmp270
        tl.store(out_ptr28 + (x7), tmp243, None)
        tl.store(out_ptr30 + (x7), tmp271, None)
        tl.store(out_ptr31 + (x7), tmp250, None)
    elif xpid >= 1349 and xpid < 1350:
        xpid_offset = xpid - 1349
        xnumel = 256
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x8 = xindex
        tmp272 = tl.load(in_ptr40 + (x8), xmask)
        tmp273 = tl.load(in_ptr41 + (x8), xmask)
        tmp278 = tl.load(in_ptr42 + (x8), xmask)
        tmp285 = tl.load(in_ptr43 + (x8), xmask)
        tmp287 = tl.load(in_ptr44 + (0))
        tmp288 = tl.broadcast_to(tmp287, [XBLOCK])
        tmp274 = tmp273 - tmp272
        tmp275 = 0.09999999999999998
        tmp276 = tmp274 * tmp275
        tmp277 = tmp272 + tmp276
        tmp279 = 0.999
        tmp280 = tmp278 * tmp279
        tmp281 = tmp273 * tmp273
        tmp282 = 0.0010000000000000009
        tmp283 = tmp281 * tmp282
        tmp284 = tmp280 + tmp283
        tmp286 = libdevice.sqrt(tmp284)
        tmp289 = libdevice.pow(tmp279, tmp288)
        tmp290 = 1.0
        tmp291 = tmp289 - tmp290
        tmp292 = -tmp291
        tmp293 = libdevice.sqrt(tmp292)
        tmp294 = tmp286 / tmp293
        tmp295 = 1e-08
        tmp296 = tmp294 + tmp295
        tmp297 = 0.9
        tmp298 = libdevice.pow(tmp297, tmp288)
        tmp299 = tmp298 - tmp290
        tmp300 = 0.001
        tmp301 = tmp299 / tmp300
        tmp302 = 1 / tmp301
        tmp303 = tmp296 / tmp302
        tmp304 = tmp277 / tmp303
        tmp305 = tmp285 + tmp304
        tl.store(out_ptr32 + (x8), tmp277, xmask)
        tl.store(out_ptr34 + (x8), tmp305, xmask)
        tl.store(out_ptr35 + (x8), tmp284, xmask)
    elif xpid >= 1350 and xpid < 1351:
        xpid_offset = xpid - 1350
        xnumel = 256
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x9 = xindex
        tmp306 = tl.load(in_ptr45 + (x9), xmask)
        tmp307 = tl.load(in_ptr46 + (x9), xmask)
        tmp312 = tl.load(in_ptr47 + (x9), xmask)
        tmp319 = tl.load(in_ptr48 + (x9), xmask)
        tmp321 = tl.load(in_ptr49 + (0))
        tmp322 = tl.broadcast_to(tmp321, [XBLOCK])
        tmp308 = tmp307 - tmp306
        tmp309 = 0.09999999999999998
        tmp310 = tmp308 * tmp309
        tmp311 = tmp306 + tmp310
        tmp313 = 0.999
        tmp314 = tmp312 * tmp313
        tmp315 = tmp307 * tmp307
        tmp316 = 0.0010000000000000009
        tmp317 = tmp315 * tmp316
        tmp318 = tmp314 + tmp317
        tmp320 = libdevice.sqrt(tmp318)
        tmp323 = libdevice.pow(tmp313, tmp322)
        tmp324 = 1.0
        tmp325 = tmp323 - tmp324
        tmp326 = -tmp325
        tmp327 = libdevice.sqrt(tmp326)
        tmp328 = tmp320 / tmp327
        tmp329 = 1e-08
        tmp330 = tmp328 + tmp329
        tmp331 = 0.9
        tmp332 = libdevice.pow(tmp331, tmp322)
        tmp333 = tmp332 - tmp324
        tmp334 = 0.001
        tmp335 = tmp333 / tmp334
        tmp336 = 1 / tmp335
        tmp337 = tmp330 / tmp336
        tmp338 = tmp311 / tmp337
        tmp339 = tmp319 + tmp338
        tl.store(out_ptr36 + (x9), tmp311, xmask)
        tl.store(out_ptr38 + (x9), tmp339, xmask)
        tl.store(out_ptr39 + (x9), tmp318, xmask)
    elif xpid >= 1351 and xpid < 1607:
        xpid_offset = xpid - 1351
        xnumel = 262144
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x10 = xindex
        tmp340 = tl.load(in_ptr50 + (x10), None)
        tmp341 = tl.load(in_ptr51 + (x10), None)
        tmp346 = tl.load(in_ptr52 + (x10), None)
        tmp353 = tl.load(in_ptr53 + (x10), None)
        tmp355 = tl.load(in_ptr54 + (0))
        tmp356 = tl.broadcast_to(tmp355, [XBLOCK])
        tmp342 = tmp341 - tmp340
        tmp343 = 0.09999999999999998
        tmp344 = tmp342 * tmp343
        tmp345 = tmp340 + tmp344
        tmp347 = 0.999
        tmp348 = tmp346 * tmp347
        tmp349 = tmp341 * tmp341
        tmp350 = 0.0010000000000000009
        tmp351 = tmp349 * tmp350
        tmp352 = tmp348 + tmp351
        tmp354 = libdevice.sqrt(tmp352)
        tmp357 = libdevice.pow(tmp347, tmp356)
        tmp358 = 1.0
        tmp359 = tmp357 - tmp358
        tmp360 = -tmp359
        tmp361 = libdevice.sqrt(tmp360)
        tmp362 = tmp354 / tmp361
        tmp363 = 1e-08
        tmp364 = tmp362 + tmp363
        tmp365 = 0.9
        tmp366 = libdevice.pow(tmp365, tmp356)
        tmp367 = tmp366 - tmp358
        tmp368 = 0.001
        tmp369 = tmp367 / tmp368
        tmp370 = 1 / tmp369
        tmp371 = tmp364 / tmp370
        tmp372 = tmp345 / tmp371
        tmp373 = tmp353 + tmp372
        tl.store(out_ptr40 + (x10), tmp345, None)
        tl.store(out_ptr42 + (x10), tmp373, None)
        tl.store(out_ptr43 + (x10), tmp352, None)
    elif xpid >= 1607 and xpid < 1608:
        xpid_offset = xpid - 1607
        xnumel = 1024
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x11 = xindex
        tmp374 = tl.load(in_ptr55 + (x11), xmask)
        tmp375 = tl.load(in_ptr56 + (x11), xmask)
        tmp380 = tl.load(in_ptr57 + (x11), xmask)
        tmp387 = tl.load(in_ptr58 + (x11), xmask)
        tmp389 = tl.load(in_ptr59 + (0))
        tmp390 = tl.broadcast_to(tmp389, [XBLOCK])
        tmp376 = tmp375 - tmp374
        tmp377 = 0.09999999999999998
        tmp378 = tmp376 * tmp377
        tmp379 = tmp374 + tmp378
        tmp381 = 0.999
        tmp382 = tmp380 * tmp381
        tmp383 = tmp375 * tmp375
        tmp384 = 0.0010000000000000009
        tmp385 = tmp383 * tmp384
        tmp386 = tmp382 + tmp385
        tmp388 = libdevice.sqrt(tmp386)
        tmp391 = libdevice.pow(tmp381, tmp390)
        tmp392 = 1.0
        tmp393 = tmp391 - tmp392
        tmp394 = -tmp393
        tmp395 = libdevice.sqrt(tmp394)
        tmp396 = tmp388 / tmp395
        tmp397 = 1e-08
        tmp398 = tmp396 + tmp397
        tmp399 = 0.9
        tmp400 = libdevice.pow(tmp399, tmp390)
        tmp401 = tmp400 - tmp392
        tmp402 = 0.001
        tmp403 = tmp401 / tmp402
        tmp404 = 1 / tmp403
        tmp405 = tmp398 / tmp404
        tmp406 = tmp379 / tmp405
        tmp407 = tmp387 + tmp406
        tl.store(out_ptr44 + (x11), tmp379, xmask)
        tl.store(out_ptr46 + (x11), tmp407, xmask)
        tl.store(out_ptr47 + (x11), tmp386, xmask)
    elif xpid >= 1608 and xpid < 1609:
        xpid_offset = xpid - 1608
        xnumel = 1024
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x12 = xindex
        tmp408 = tl.load(in_ptr60 + (x12), xmask)
        tmp409 = tl.load(in_ptr61 + (x12), xmask)
        tmp414 = tl.load(in_ptr62 + (x12), xmask)
        tmp421 = tl.load(in_ptr63 + (x12), xmask)
        tmp423 = tl.load(in_ptr64 + (0))
        tmp424 = tl.broadcast_to(tmp423, [XBLOCK])
        tmp410 = tmp409 - tmp408
        tmp411 = 0.09999999999999998
        tmp412 = tmp410 * tmp411
        tmp413 = tmp408 + tmp412
        tmp415 = 0.999
        tmp416 = tmp414 * tmp415
        tmp417 = tmp409 * tmp409
        tmp418 = 0.0010000000000000009
        tmp419 = tmp417 * tmp418
        tmp420 = tmp416 + tmp419
        tmp422 = libdevice.sqrt(tmp420)
        tmp425 = libdevice.pow(tmp415, tmp424)
        tmp426 = 1.0
        tmp427 = tmp425 - tmp426
        tmp428 = -tmp427
        tmp429 = libdevice.sqrt(tmp428)
        tmp430 = tmp422 / tmp429
        tmp431 = 1e-08
        tmp432 = tmp430 + tmp431
        tmp433 = 0.9
        tmp434 = libdevice.pow(tmp433, tmp424)
        tmp435 = tmp434 - tmp426
        tmp436 = 0.001
        tmp437 = tmp435 / tmp436
        tmp438 = 1 / tmp437
        tmp439 = tmp432 / tmp438
        tmp440 = tmp413 / tmp439
        tmp441 = tmp421 + tmp440
        tl.store(out_ptr48 + (x12), tmp413, xmask)
        tl.store(out_ptr50 + (x12), tmp441, xmask)
        tl.store(out_ptr51 + (x12), tmp420, xmask)
    elif xpid >= 1609 and xpid < 1865:
        xpid_offset = xpid - 1609
        xnumel = 262144
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x13 = xindex
        tmp442 = tl.load(in_ptr65 + (x13), None)
        tmp443 = tl.load(in_ptr66 + (x13), None)
        tmp448 = tl.load(in_ptr67 + (x13), None)
        tmp455 = tl.load(in_ptr68 + (x13), None)
        tmp457 = tl.load(in_ptr69 + (0))
        tmp458 = tl.broadcast_to(tmp457, [XBLOCK])
        tmp444 = tmp443 - tmp442
        tmp445 = 0.09999999999999998
        tmp446 = tmp444 * tmp445
        tmp447 = tmp442 + tmp446
        tmp449 = 0.999
        tmp450 = tmp448 * tmp449
        tmp451 = tmp443 * tmp443
        tmp452 = 0.0010000000000000009
        tmp453 = tmp451 * tmp452
        tmp454 = tmp450 + tmp453
        tmp456 = libdevice.sqrt(tmp454)
        tmp459 = libdevice.pow(tmp449, tmp458)
        tmp460 = 1.0
        tmp461 = tmp459 - tmp460
        tmp462 = -tmp461
        tmp463 = libdevice.sqrt(tmp462)
        tmp464 = tmp456 / tmp463
        tmp465 = 1e-08
        tmp466 = tmp464 + tmp465
        tmp467 = 0.9
        tmp468 = libdevice.pow(tmp467, tmp458)
        tmp469 = tmp468 - tmp460
        tmp470 = 0.001
        tmp471 = tmp469 / tmp470
        tmp472 = 1 / tmp471
        tmp473 = tmp466 / tmp472
        tmp474 = tmp447 / tmp473
        tmp475 = tmp455 + tmp474
        tl.store(out_ptr52 + (x13), tmp447, None)
        tl.store(out_ptr54 + (x13), tmp475, None)
        tl.store(out_ptr55 + (x13), tmp454, None)
    elif xpid >= 1865 and xpid < 1866:
        xpid_offset = xpid - 1865
        xnumel = 256
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x14 = xindex
        tmp476 = tl.load(in_ptr70 + (x14), xmask)
        tmp477 = tl.load(in_ptr71 + (x14), xmask)
        tmp482 = tl.load(in_ptr72 + (x14), xmask)
        tmp489 = tl.load(in_ptr73 + (x14), xmask)
        tmp491 = tl.load(in_ptr74 + (0))
        tmp492 = tl.broadcast_to(tmp491, [XBLOCK])
        tmp478 = tmp477 - tmp476
        tmp479 = 0.09999999999999998
        tmp480 = tmp478 * tmp479
        tmp481 = tmp476 + tmp480
        tmp483 = 0.999
        tmp484 = tmp482 * tmp483
        tmp485 = tmp477 * tmp477
        tmp486 = 0.0010000000000000009
        tmp487 = tmp485 * tmp486
        tmp488 = tmp484 + tmp487
        tmp490 = libdevice.sqrt(tmp488)
        tmp493 = libdevice.pow(tmp483, tmp492)
        tmp494 = 1.0
        tmp495 = tmp493 - tmp494
        tmp496 = -tmp495
        tmp497 = libdevice.sqrt(tmp496)
        tmp498 = tmp490 / tmp497
        tmp499 = 1e-08
        tmp500 = tmp498 + tmp499
        tmp501 = 0.9
        tmp502 = libdevice.pow(tmp501, tmp492)
        tmp503 = tmp502 - tmp494
        tmp504 = 0.001
        tmp505 = tmp503 / tmp504
        tmp506 = 1 / tmp505
        tmp507 = tmp500 / tmp506
        tmp508 = tmp481 / tmp507
        tmp509 = tmp489 + tmp508
        tl.store(out_ptr56 + (x14), tmp481, xmask)
        tl.store(out_ptr58 + (x14), tmp509, xmask)
        tl.store(out_ptr59 + (x14), tmp488, xmask)
    elif xpid >= 1866 and xpid < 1867:
        xpid_offset = xpid - 1866
        xnumel = 256
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x15 = xindex
        tmp510 = tl.load(in_ptr75 + (x15), xmask)
        tmp511 = tl.load(in_ptr76 + (x15), xmask)
        tmp516 = tl.load(in_ptr77 + (x15), xmask)
        tmp523 = tl.load(in_ptr78 + (x15), xmask)
        tmp525 = tl.load(in_ptr79 + (0))
        tmp526 = tl.broadcast_to(tmp525, [XBLOCK])
        tmp512 = tmp511 - tmp510
        tmp513 = 0.09999999999999998
        tmp514 = tmp512 * tmp513
        tmp515 = tmp510 + tmp514
        tmp517 = 0.999
        tmp518 = tmp516 * tmp517
        tmp519 = tmp511 * tmp511
        tmp520 = 0.0010000000000000009
        tmp521 = tmp519 * tmp520
        tmp522 = tmp518 + tmp521
        tmp524 = libdevice.sqrt(tmp522)
        tmp527 = libdevice.pow(tmp517, tmp526)
        tmp528 = 1.0
        tmp529 = tmp527 - tmp528
        tmp530 = -tmp529
        tmp531 = libdevice.sqrt(tmp530)
        tmp532 = tmp524 / tmp531
        tmp533 = 1e-08
        tmp534 = tmp532 + tmp533
        tmp535 = 0.9
        tmp536 = libdevice.pow(tmp535, tmp526)
        tmp537 = tmp536 - tmp528
        tmp538 = 0.001
        tmp539 = tmp537 / tmp538
        tmp540 = 1 / tmp539
        tmp541 = tmp534 / tmp540
        tmp542 = tmp515 / tmp541
        tmp543 = tmp523 + tmp542
        tl.store(out_ptr60 + (x15), tmp515, xmask)
        tl.store(out_ptr62 + (x15), tmp543, xmask)
        tl.store(out_ptr63 + (x15), tmp522, xmask)
    elif xpid >= 1867 and xpid < 2443:
        xpid_offset = xpid - 1867
        xnumel = 589824
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x16 = xindex
        tmp544 = tl.load(in_ptr80 + (x16), None)
        tmp545 = tl.load(in_ptr81 + (x16), None)
        tmp550 = tl.load(in_ptr82 + (x16), None)
        tmp557 = tl.load(in_ptr83 + (x16), None)
        tmp559 = tl.load(in_ptr84 + (0))
        tmp560 = tl.broadcast_to(tmp559, [XBLOCK])
        tmp546 = tmp545 - tmp544
        tmp547 = 0.09999999999999998
        tmp548 = tmp546 * tmp547
        tmp549 = tmp544 + tmp548
        tmp551 = 0.999
        tmp552 = tmp550 * tmp551
        tmp553 = tmp545 * tmp545
        tmp554 = 0.0010000000000000009
        tmp555 = tmp553 * tmp554
        tmp556 = tmp552 + tmp555
        tmp558 = libdevice.sqrt(tmp556)
        tmp561 = libdevice.pow(tmp551, tmp560)
        tmp562 = 1.0
        tmp563 = tmp561 - tmp562
        tmp564 = -tmp563
        tmp565 = libdevice.sqrt(tmp564)
        tmp566 = tmp558 / tmp565
        tmp567 = 1e-08
        tmp568 = tmp566 + tmp567
        tmp569 = 0.9
        tmp570 = libdevice.pow(tmp569, tmp560)
        tmp571 = tmp570 - tmp562
        tmp572 = 0.001
        tmp573 = tmp571 / tmp572
        tmp574 = 1 / tmp573
        tmp575 = tmp568 / tmp574
        tmp576 = tmp549 / tmp575
        tmp577 = tmp557 + tmp576
        tl.store(out_ptr64 + (x16), tmp549, None)
        tl.store(out_ptr66 + (x16), tmp577, None)
        tl.store(out_ptr67 + (x16), tmp556, None)
    elif xpid >= 2443 and xpid < 2444:
        xpid_offset = xpid - 2443
        xnumel = 256
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x17 = xindex
        tmp578 = tl.load(in_ptr85 + (x17), xmask)
        tmp579 = tl.load(in_ptr86 + (x17), xmask)
        tmp584 = tl.load(in_ptr87 + (x17), xmask)
        tmp591 = tl.load(in_ptr88 + (x17), xmask)
        tmp593 = tl.load(in_ptr89 + (0))
        tmp594 = tl.broadcast_to(tmp593, [XBLOCK])
        tmp580 = tmp579 - tmp578
        tmp581 = 0.09999999999999998
        tmp582 = tmp580 * tmp581
        tmp583 = tmp578 + tmp582
        tmp585 = 0.999
        tmp586 = tmp584 * tmp585
        tmp587 = tmp579 * tmp579
        tmp588 = 0.0010000000000000009
        tmp589 = tmp587 * tmp588
        tmp590 = tmp586 + tmp589
        tmp592 = libdevice.sqrt(tmp590)
        tmp595 = libdevice.pow(tmp585, tmp594)
        tmp596 = 1.0
        tmp597 = tmp595 - tmp596
        tmp598 = -tmp597
        tmp599 = libdevice.sqrt(tmp598)
        tmp600 = tmp592 / tmp599
        tmp601 = 1e-08
        tmp602 = tmp600 + tmp601
        tmp603 = 0.9
        tmp604 = libdevice.pow(tmp603, tmp594)
        tmp605 = tmp604 - tmp596
        tmp606 = 0.001
        tmp607 = tmp605 / tmp606
        tmp608 = 1 / tmp607
        tmp609 = tmp602 / tmp608
        tmp610 = tmp583 / tmp609
        tmp611 = tmp591 + tmp610
        tl.store(out_ptr68 + (x17), tmp583, xmask)
        tl.store(out_ptr70 + (x17), tmp611, xmask)
        tl.store(out_ptr71 + (x17), tmp590, xmask)
    elif xpid >= 2444 and xpid < 2445:
        xpid_offset = xpid - 2444
        xnumel = 256
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x18 = xindex
        tmp612 = tl.load(in_ptr90 + (x18), xmask)
        tmp613 = tl.load(in_ptr91 + (x18), xmask)
        tmp618 = tl.load(in_ptr92 + (x18), xmask)
        tmp625 = tl.load(in_ptr93 + (x18), xmask)
        tmp627 = tl.load(in_ptr94 + (0))
        tmp628 = tl.broadcast_to(tmp627, [XBLOCK])
        tmp614 = tmp613 - tmp612
        tmp615 = 0.09999999999999998
        tmp616 = tmp614 * tmp615
        tmp617 = tmp612 + tmp616
        tmp619 = 0.999
        tmp620 = tmp618 * tmp619
        tmp621 = tmp613 * tmp613
        tmp622 = 0.0010000000000000009
        tmp623 = tmp621 * tmp622
        tmp624 = tmp620 + tmp623
        tmp626 = libdevice.sqrt(tmp624)
        tmp629 = libdevice.pow(tmp619, tmp628)
        tmp630 = 1.0
        tmp631 = tmp629 - tmp630
        tmp632 = -tmp631
        tmp633 = libdevice.sqrt(tmp632)
        tmp634 = tmp626 / tmp633
        tmp635 = 1e-08
        tmp636 = tmp634 + tmp635
        tmp637 = 0.9
        tmp638 = libdevice.pow(tmp637, tmp628)
        tmp639 = tmp638 - tmp630
        tmp640 = 0.001
        tmp641 = tmp639 / tmp640
        tmp642 = 1 / tmp641
        tmp643 = tmp636 / tmp642
        tmp644 = tmp617 / tmp643
        tmp645 = tmp625 + tmp644
        tl.store(out_ptr72 + (x18), tmp617, xmask)
        tl.store(out_ptr74 + (x18), tmp645, xmask)
        tl.store(out_ptr75 + (x18), tmp624, xmask)
    elif xpid >= 2445 and xpid < 2701:
        xpid_offset = xpid - 2445
        xnumel = 262144
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x19 = xindex
        tmp646 = tl.load(in_ptr95 + (x19), None)
        tmp647 = tl.load(in_ptr96 + (x19), None)
        tmp652 = tl.load(in_ptr97 + (x19), None)
        tmp659 = tl.load(in_ptr98 + (x19), None)
        tmp661 = tl.load(in_ptr99 + (0))
        tmp662 = tl.broadcast_to(tmp661, [XBLOCK])
        tmp648 = tmp647 - tmp646
        tmp649 = 0.09999999999999998
        tmp650 = tmp648 * tmp649
        tmp651 = tmp646 + tmp650
        tmp653 = 0.999
        tmp654 = tmp652 * tmp653
        tmp655 = tmp647 * tmp647
        tmp656 = 0.0010000000000000009
        tmp657 = tmp655 * tmp656
        tmp658 = tmp654 + tmp657
        tmp660 = libdevice.sqrt(tmp658)
        tmp663 = libdevice.pow(tmp653, tmp662)
        tmp664 = 1.0
        tmp665 = tmp663 - tmp664
        tmp666 = -tmp665
        tmp667 = libdevice.sqrt(tmp666)
        tmp668 = tmp660 / tmp667
        tmp669 = 1e-08
        tmp670 = tmp668 + tmp669
        tmp671 = 0.9
        tmp672 = libdevice.pow(tmp671, tmp662)
        tmp673 = tmp672 - tmp664
        tmp674 = 0.001
        tmp675 = tmp673 / tmp674
        tmp676 = 1 / tmp675
        tmp677 = tmp670 / tmp676
        tmp678 = tmp651 / tmp677
        tmp679 = tmp659 + tmp678
        tl.store(out_ptr76 + (x19), tmp651, None)
        tl.store(out_ptr78 + (x19), tmp679, None)
        tl.store(out_ptr79 + (x19), tmp658, None)
    else:
        pass
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/2m/c2mf45266c2jind3ncfhygeqa5hiez4drw43ezdwfh4daxpsaeya.py
# Source Nodes: [], Original ATen: []

triton_for_fused_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.foreach(
    num_warps=8,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: '*fp32', 35: '*fp32', 36: '*fp32', 37: '*fp32', 38: '*fp32', 39: '*fp32', 40: '*fp32', 41: '*fp32', 42: '*fp32', 43: '*fp32', 44: '*fp32', 45: '*fp32', 46: '*fp32', 47: '*fp32', 48: '*fp32', 49: '*fp32', 50: '*fp32', 51: '*fp32', 52: '*fp32', 53: '*fp32', 54: '*fp32', 55: '*fp32', 56: '*fp32', 57: '*fp32', 58: '*fp32', 59: '*fp32', 60: '*fp32', 61: '*fp32', 62: '*fp32', 63: '*fp32', 64: '*fp32', 65: '*fp32', 66: '*fp32', 67: '*fp32', 68: '*fp32', 69: '*fp32', 70: '*fp32', 71: '*fp32', 72: '*fp32', 73: '*fp32', 74: '*fp32', 75: '*fp32', 76: '*fp32', 77: '*fp32', 78: '*fp32', 79: '*fp32', 80: '*fp32', 81: '*fp32', 82: '*fp32', 83: '*fp32', 84: '*fp32', 85: '*fp32', 86: '*fp32', 87: '*fp32', 88: '*fp32', 89: '*fp32', 90: '*fp32', 91: '*fp32', 92: '*fp32', 93: '*fp32', 94: '*fp32', 95: '*fp32', 96: '*fp32', 97: '*fp32', 98: '*fp32', 99: '*fp32', 100: '*fp32', 101: '*fp32', 102: '*fp32', 103: '*fp32', 104: '*fp32', 105: '*fp32', 106: '*fp32', 107: '*fp32', 108: '*fp32', 109: '*fp32', 110: '*fp32', 111: '*fp32', 112: '*fp32', 113: '*fp32', 114: '*fp32', 115: '*fp32', 116: '*fp32', 117: '*fp32', 118: '*fp32', 119: '*fp32', 120: '*fp32', 121: '*fp32', 122: '*fp32', 123: '*fp32', 124: '*fp32', 125: '*fp32', 126: '*fp32', 127: '*fp32', 128: '*fp32', 129: '*fp32', 130: '*fp32', 131: '*fp32', 132: '*fp32', 133: '*fp32', 134: '*fp32', 135: '*fp32', 136: '*fp32', 137: '*fp32', 138: '*fp32', 139: '*fp32', 140: '*fp32', 141: '*fp32', 142: '*fp32', 143: '*fp32', 144: '*fp32', 145: '*fp32', 146: '*fp32', 147: '*fp32', 148: '*fp32', 149: '*fp32', 150: '*fp32', 151: '*fp32', 152: '*fp32', 153: '*fp32', 154: '*fp32', 155: '*fp32', 156: '*fp32', 157: '*fp32', 158: '*fp32', 159: '*fp32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'kernel_name': 'triton_for_fused_7', 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, in_ptr47, in_ptr48, in_ptr49, in_ptr50, in_ptr51, in_ptr52, in_ptr53, in_ptr54, in_ptr55, in_ptr56, in_ptr57, in_ptr58, in_ptr59, in_ptr60, in_ptr61, in_ptr62, in_ptr63, in_ptr64, in_ptr65, in_ptr66, in_ptr67, in_ptr68, in_ptr69, in_ptr70, in_ptr71, in_ptr72, in_ptr73, in_ptr74, in_ptr75, in_ptr76, in_ptr77, in_ptr78, in_ptr79, in_ptr80, in_ptr81, in_ptr82, in_ptr83, in_ptr84, in_ptr85, in_ptr86, in_ptr87, in_ptr88, in_ptr89, in_ptr90, in_ptr91, in_ptr92, in_ptr93, in_ptr94, in_ptr95, in_ptr96, in_ptr97, in_ptr98, in_ptr99, out_ptr0, out_ptr2, out_ptr3, out_ptr4, out_ptr6, out_ptr7, out_ptr8, out_ptr10, out_ptr11, out_ptr12, out_ptr14, out_ptr15, out_ptr16, out_ptr18, out_ptr19, out_ptr20, out_ptr22, out_ptr23, out_ptr24, out_ptr26, out_ptr27, out_ptr28, out_ptr30, out_ptr31, out_ptr32, out_ptr34, out_ptr35, out_ptr36, out_ptr38, out_ptr39, out_ptr40, out_ptr42, out_ptr43, out_ptr44, out_ptr46, out_ptr47, out_ptr48, out_ptr50, out_ptr51, out_ptr52, out_ptr54, out_ptr55, out_ptr56, out_ptr58, out_ptr59, out_ptr60, out_ptr62, out_ptr63, out_ptr64, out_ptr66, out_ptr67, out_ptr68, out_ptr70, out_ptr71, out_ptr72, out_ptr74, out_ptr75, out_ptr76, out_ptr78, out_ptr79):
    xpid = tl.program_id(0)
    XBLOCK: tl.constexpr = 1024
    if xpid >= 0 and xpid < 1:
        xpid_offset = xpid - 0
        xnumel = 1024
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), xmask)
        tmp1 = tl.load(in_ptr1 + (x0), xmask)
        tmp6 = tl.load(in_ptr2 + (x0), xmask)
        tmp13 = tl.load(in_ptr3 + (x0), xmask)
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
        tmp14 = libdevice.sqrt(tmp12)
        tmp17 = libdevice.pow(tmp7, tmp16)
        tmp18 = 1.0
        tmp19 = tmp17 - tmp18
        tmp20 = -tmp19
        tmp21 = libdevice.sqrt(tmp20)
        tmp22 = tmp14 / tmp21
        tmp23 = 1e-08
        tmp24 = tmp22 + tmp23
        tmp25 = 0.9
        tmp26 = libdevice.pow(tmp25, tmp16)
        tmp27 = tmp26 - tmp18
        tmp28 = 0.001
        tmp29 = tmp27 / tmp28
        tmp30 = 1 / tmp29
        tmp31 = tmp24 / tmp30
        tmp32 = tmp5 / tmp31
        tmp33 = tmp13 + tmp32
        tl.store(out_ptr0 + (x0), tmp5, xmask)
        tl.store(out_ptr2 + (x0), tmp33, xmask)
        tl.store(out_ptr3 + (x0), tmp12, xmask)
    elif xpid >= 1 and xpid < 2:
        xpid_offset = xpid - 1
        xnumel = 1024
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
        tmp48 = libdevice.sqrt(tmp46)
        tmp51 = libdevice.pow(tmp41, tmp50)
        tmp52 = 1.0
        tmp53 = tmp51 - tmp52
        tmp54 = -tmp53
        tmp55 = libdevice.sqrt(tmp54)
        tmp56 = tmp48 / tmp55
        tmp57 = 1e-08
        tmp58 = tmp56 + tmp57
        tmp59 = 0.9
        tmp60 = libdevice.pow(tmp59, tmp50)
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
    elif xpid >= 2 and xpid < 258:
        xpid_offset = xpid - 2
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
        tmp82 = libdevice.sqrt(tmp80)
        tmp85 = libdevice.pow(tmp75, tmp84)
        tmp86 = 1.0
        tmp87 = tmp85 - tmp86
        tmp88 = -tmp87
        tmp89 = libdevice.sqrt(tmp88)
        tmp90 = tmp82 / tmp89
        tmp91 = 1e-08
        tmp92 = tmp90 + tmp91
        tmp93 = 0.9
        tmp94 = libdevice.pow(tmp93, tmp84)
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
    elif xpid >= 258 and xpid < 259:
        xpid_offset = xpid - 258
        xnumel = 256
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
        tmp116 = libdevice.sqrt(tmp114)
        tmp119 = libdevice.pow(tmp109, tmp118)
        tmp120 = 1.0
        tmp121 = tmp119 - tmp120
        tmp122 = -tmp121
        tmp123 = libdevice.sqrt(tmp122)
        tmp124 = tmp116 / tmp123
        tmp125 = 1e-08
        tmp126 = tmp124 + tmp125
        tmp127 = 0.9
        tmp128 = libdevice.pow(tmp127, tmp118)
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
    elif xpid >= 259 and xpid < 260:
        xpid_offset = xpid - 259
        xnumel = 256
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
        tmp150 = libdevice.sqrt(tmp148)
        tmp153 = libdevice.pow(tmp143, tmp152)
        tmp154 = 1.0
        tmp155 = tmp153 - tmp154
        tmp156 = -tmp155
        tmp157 = libdevice.sqrt(tmp156)
        tmp158 = tmp150 / tmp157
        tmp159 = 1e-08
        tmp160 = tmp158 + tmp159
        tmp161 = 0.9
        tmp162 = libdevice.pow(tmp161, tmp152)
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
    elif xpid >= 260 and xpid < 836:
        xpid_offset = xpid - 260
        xnumel = 589824
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x5 = xindex
        tmp170 = tl.load(in_ptr25 + (x5), None)
        tmp171 = tl.load(in_ptr26 + (x5), None)
        tmp176 = tl.load(in_ptr27 + (x5), None)
        tmp183 = tl.load(in_ptr28 + (x5), None)
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
        tmp184 = libdevice.sqrt(tmp182)
        tmp187 = libdevice.pow(tmp177, tmp186)
        tmp188 = 1.0
        tmp189 = tmp187 - tmp188
        tmp190 = -tmp189
        tmp191 = libdevice.sqrt(tmp190)
        tmp192 = tmp184 / tmp191
        tmp193 = 1e-08
        tmp194 = tmp192 + tmp193
        tmp195 = 0.9
        tmp196 = libdevice.pow(tmp195, tmp186)
        tmp197 = tmp196 - tmp188
        tmp198 = 0.001
        tmp199 = tmp197 / tmp198
        tmp200 = 1 / tmp199
        tmp201 = tmp194 / tmp200
        tmp202 = tmp175 / tmp201
        tmp203 = tmp183 + tmp202
        tl.store(out_ptr20 + (x5), tmp175, None)
        tl.store(out_ptr22 + (x5), tmp203, None)
        tl.store(out_ptr23 + (x5), tmp182, None)
    elif xpid >= 836 and xpid < 837:
        xpid_offset = xpid - 836
        xnumel = 256
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x6 = xindex
        tmp204 = tl.load(in_ptr30 + (x6), xmask)
        tmp205 = tl.load(in_ptr31 + (x6), xmask)
        tmp210 = tl.load(in_ptr32 + (x6), xmask)
        tmp217 = tl.load(in_ptr33 + (x6), xmask)
        tmp219 = tl.load(in_ptr34 + (0))
        tmp220 = tl.broadcast_to(tmp219, [XBLOCK])
        tmp206 = tmp205 - tmp204
        tmp207 = 0.09999999999999998
        tmp208 = tmp206 * tmp207
        tmp209 = tmp204 + tmp208
        tmp211 = 0.999
        tmp212 = tmp210 * tmp211
        tmp213 = tmp205 * tmp205
        tmp214 = 0.0010000000000000009
        tmp215 = tmp213 * tmp214
        tmp216 = tmp212 + tmp215
        tmp218 = libdevice.sqrt(tmp216)
        tmp221 = libdevice.pow(tmp211, tmp220)
        tmp222 = 1.0
        tmp223 = tmp221 - tmp222
        tmp224 = -tmp223
        tmp225 = libdevice.sqrt(tmp224)
        tmp226 = tmp218 / tmp225
        tmp227 = 1e-08
        tmp228 = tmp226 + tmp227
        tmp229 = 0.9
        tmp230 = libdevice.pow(tmp229, tmp220)
        tmp231 = tmp230 - tmp222
        tmp232 = 0.001
        tmp233 = tmp231 / tmp232
        tmp234 = 1 / tmp233
        tmp235 = tmp228 / tmp234
        tmp236 = tmp209 / tmp235
        tmp237 = tmp217 + tmp236
        tl.store(out_ptr24 + (x6), tmp209, xmask)
        tl.store(out_ptr26 + (x6), tmp237, xmask)
        tl.store(out_ptr27 + (x6), tmp216, xmask)
    elif xpid >= 837 and xpid < 838:
        xpid_offset = xpid - 837
        xnumel = 256
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x7 = xindex
        tmp238 = tl.load(in_ptr35 + (x7), xmask)
        tmp239 = tl.load(in_ptr36 + (x7), xmask)
        tmp244 = tl.load(in_ptr37 + (x7), xmask)
        tmp251 = tl.load(in_ptr38 + (x7), xmask)
        tmp253 = tl.load(in_ptr39 + (0))
        tmp254 = tl.broadcast_to(tmp253, [XBLOCK])
        tmp240 = tmp239 - tmp238
        tmp241 = 0.09999999999999998
        tmp242 = tmp240 * tmp241
        tmp243 = tmp238 + tmp242
        tmp245 = 0.999
        tmp246 = tmp244 * tmp245
        tmp247 = tmp239 * tmp239
        tmp248 = 0.0010000000000000009
        tmp249 = tmp247 * tmp248
        tmp250 = tmp246 + tmp249
        tmp252 = libdevice.sqrt(tmp250)
        tmp255 = libdevice.pow(tmp245, tmp254)
        tmp256 = 1.0
        tmp257 = tmp255 - tmp256
        tmp258 = -tmp257
        tmp259 = libdevice.sqrt(tmp258)
        tmp260 = tmp252 / tmp259
        tmp261 = 1e-08
        tmp262 = tmp260 + tmp261
        tmp263 = 0.9
        tmp264 = libdevice.pow(tmp263, tmp254)
        tmp265 = tmp264 - tmp256
        tmp266 = 0.001
        tmp267 = tmp265 / tmp266
        tmp268 = 1 / tmp267
        tmp269 = tmp262 / tmp268
        tmp270 = tmp243 / tmp269
        tmp271 = tmp251 + tmp270
        tl.store(out_ptr28 + (x7), tmp243, xmask)
        tl.store(out_ptr30 + (x7), tmp271, xmask)
        tl.store(out_ptr31 + (x7), tmp250, xmask)
    elif xpid >= 838 and xpid < 1094:
        xpid_offset = xpid - 838
        xnumel = 262144
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x8 = xindex
        tmp272 = tl.load(in_ptr40 + (x8), None)
        tmp273 = tl.load(in_ptr41 + (x8), None)
        tmp278 = tl.load(in_ptr42 + (x8), None)
        tmp285 = tl.load(in_ptr43 + (x8), None)
        tmp287 = tl.load(in_ptr44 + (0))
        tmp288 = tl.broadcast_to(tmp287, [XBLOCK])
        tmp274 = tmp273 - tmp272
        tmp275 = 0.09999999999999998
        tmp276 = tmp274 * tmp275
        tmp277 = tmp272 + tmp276
        tmp279 = 0.999
        tmp280 = tmp278 * tmp279
        tmp281 = tmp273 * tmp273
        tmp282 = 0.0010000000000000009
        tmp283 = tmp281 * tmp282
        tmp284 = tmp280 + tmp283
        tmp286 = libdevice.sqrt(tmp284)
        tmp289 = libdevice.pow(tmp279, tmp288)
        tmp290 = 1.0
        tmp291 = tmp289 - tmp290
        tmp292 = -tmp291
        tmp293 = libdevice.sqrt(tmp292)
        tmp294 = tmp286 / tmp293
        tmp295 = 1e-08
        tmp296 = tmp294 + tmp295
        tmp297 = 0.9
        tmp298 = libdevice.pow(tmp297, tmp288)
        tmp299 = tmp298 - tmp290
        tmp300 = 0.001
        tmp301 = tmp299 / tmp300
        tmp302 = 1 / tmp301
        tmp303 = tmp296 / tmp302
        tmp304 = tmp277 / tmp303
        tmp305 = tmp285 + tmp304
        tl.store(out_ptr32 + (x8), tmp277, None)
        tl.store(out_ptr34 + (x8), tmp305, None)
        tl.store(out_ptr35 + (x8), tmp284, None)
    elif xpid >= 1094 and xpid < 1095:
        xpid_offset = xpid - 1094
        xnumel = 1024
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x9 = xindex
        tmp306 = tl.load(in_ptr45 + (x9), xmask)
        tmp307 = tl.load(in_ptr46 + (x9), xmask)
        tmp312 = tl.load(in_ptr47 + (x9), xmask)
        tmp319 = tl.load(in_ptr48 + (x9), xmask)
        tmp321 = tl.load(in_ptr49 + (0))
        tmp322 = tl.broadcast_to(tmp321, [XBLOCK])
        tmp308 = tmp307 - tmp306
        tmp309 = 0.09999999999999998
        tmp310 = tmp308 * tmp309
        tmp311 = tmp306 + tmp310
        tmp313 = 0.999
        tmp314 = tmp312 * tmp313
        tmp315 = tmp307 * tmp307
        tmp316 = 0.0010000000000000009
        tmp317 = tmp315 * tmp316
        tmp318 = tmp314 + tmp317
        tmp320 = libdevice.sqrt(tmp318)
        tmp323 = libdevice.pow(tmp313, tmp322)
        tmp324 = 1.0
        tmp325 = tmp323 - tmp324
        tmp326 = -tmp325
        tmp327 = libdevice.sqrt(tmp326)
        tmp328 = tmp320 / tmp327
        tmp329 = 1e-08
        tmp330 = tmp328 + tmp329
        tmp331 = 0.9
        tmp332 = libdevice.pow(tmp331, tmp322)
        tmp333 = tmp332 - tmp324
        tmp334 = 0.001
        tmp335 = tmp333 / tmp334
        tmp336 = 1 / tmp335
        tmp337 = tmp330 / tmp336
        tmp338 = tmp311 / tmp337
        tmp339 = tmp319 + tmp338
        tl.store(out_ptr36 + (x9), tmp311, xmask)
        tl.store(out_ptr38 + (x9), tmp339, xmask)
        tl.store(out_ptr39 + (x9), tmp318, xmask)
    elif xpid >= 1095 and xpid < 1096:
        xpid_offset = xpid - 1095
        xnumel = 1024
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x10 = xindex
        tmp340 = tl.load(in_ptr50 + (x10), xmask)
        tmp341 = tl.load(in_ptr51 + (x10), xmask)
        tmp346 = tl.load(in_ptr52 + (x10), xmask)
        tmp353 = tl.load(in_ptr53 + (x10), xmask)
        tmp355 = tl.load(in_ptr54 + (0))
        tmp356 = tl.broadcast_to(tmp355, [XBLOCK])
        tmp342 = tmp341 - tmp340
        tmp343 = 0.09999999999999998
        tmp344 = tmp342 * tmp343
        tmp345 = tmp340 + tmp344
        tmp347 = 0.999
        tmp348 = tmp346 * tmp347
        tmp349 = tmp341 * tmp341
        tmp350 = 0.0010000000000000009
        tmp351 = tmp349 * tmp350
        tmp352 = tmp348 + tmp351
        tmp354 = libdevice.sqrt(tmp352)
        tmp357 = libdevice.pow(tmp347, tmp356)
        tmp358 = 1.0
        tmp359 = tmp357 - tmp358
        tmp360 = -tmp359
        tmp361 = libdevice.sqrt(tmp360)
        tmp362 = tmp354 / tmp361
        tmp363 = 1e-08
        tmp364 = tmp362 + tmp363
        tmp365 = 0.9
        tmp366 = libdevice.pow(tmp365, tmp356)
        tmp367 = tmp366 - tmp358
        tmp368 = 0.001
        tmp369 = tmp367 / tmp368
        tmp370 = 1 / tmp369
        tmp371 = tmp364 / tmp370
        tmp372 = tmp345 / tmp371
        tmp373 = tmp353 + tmp372
        tl.store(out_ptr40 + (x10), tmp345, xmask)
        tl.store(out_ptr42 + (x10), tmp373, xmask)
        tl.store(out_ptr43 + (x10), tmp352, xmask)
    elif xpid >= 1096 and xpid < 1352:
        xpid_offset = xpid - 1096
        xnumel = 262144
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x11 = xindex
        tmp374 = tl.load(in_ptr55 + (x11), None)
        tmp375 = tl.load(in_ptr56 + (x11), None)
        tmp380 = tl.load(in_ptr57 + (x11), None)
        tmp387 = tl.load(in_ptr58 + (x11), None)
        tmp389 = tl.load(in_ptr59 + (0))
        tmp390 = tl.broadcast_to(tmp389, [XBLOCK])
        tmp376 = tmp375 - tmp374
        tmp377 = 0.09999999999999998
        tmp378 = tmp376 * tmp377
        tmp379 = tmp374 + tmp378
        tmp381 = 0.999
        tmp382 = tmp380 * tmp381
        tmp383 = tmp375 * tmp375
        tmp384 = 0.0010000000000000009
        tmp385 = tmp383 * tmp384
        tmp386 = tmp382 + tmp385
        tmp388 = libdevice.sqrt(tmp386)
        tmp391 = libdevice.pow(tmp381, tmp390)
        tmp392 = 1.0
        tmp393 = tmp391 - tmp392
        tmp394 = -tmp393
        tmp395 = libdevice.sqrt(tmp394)
        tmp396 = tmp388 / tmp395
        tmp397 = 1e-08
        tmp398 = tmp396 + tmp397
        tmp399 = 0.9
        tmp400 = libdevice.pow(tmp399, tmp390)
        tmp401 = tmp400 - tmp392
        tmp402 = 0.001
        tmp403 = tmp401 / tmp402
        tmp404 = 1 / tmp403
        tmp405 = tmp398 / tmp404
        tmp406 = tmp379 / tmp405
        tmp407 = tmp387 + tmp406
        tl.store(out_ptr44 + (x11), tmp379, None)
        tl.store(out_ptr46 + (x11), tmp407, None)
        tl.store(out_ptr47 + (x11), tmp386, None)
    elif xpid >= 1352 and xpid < 1353:
        xpid_offset = xpid - 1352
        xnumel = 256
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x12 = xindex
        tmp408 = tl.load(in_ptr60 + (x12), xmask)
        tmp409 = tl.load(in_ptr61 + (x12), xmask)
        tmp414 = tl.load(in_ptr62 + (x12), xmask)
        tmp421 = tl.load(in_ptr63 + (x12), xmask)
        tmp423 = tl.load(in_ptr64 + (0))
        tmp424 = tl.broadcast_to(tmp423, [XBLOCK])
        tmp410 = tmp409 - tmp408
        tmp411 = 0.09999999999999998
        tmp412 = tmp410 * tmp411
        tmp413 = tmp408 + tmp412
        tmp415 = 0.999
        tmp416 = tmp414 * tmp415
        tmp417 = tmp409 * tmp409
        tmp418 = 0.0010000000000000009
        tmp419 = tmp417 * tmp418
        tmp420 = tmp416 + tmp419
        tmp422 = libdevice.sqrt(tmp420)
        tmp425 = libdevice.pow(tmp415, tmp424)
        tmp426 = 1.0
        tmp427 = tmp425 - tmp426
        tmp428 = -tmp427
        tmp429 = libdevice.sqrt(tmp428)
        tmp430 = tmp422 / tmp429
        tmp431 = 1e-08
        tmp432 = tmp430 + tmp431
        tmp433 = 0.9
        tmp434 = libdevice.pow(tmp433, tmp424)
        tmp435 = tmp434 - tmp426
        tmp436 = 0.001
        tmp437 = tmp435 / tmp436
        tmp438 = 1 / tmp437
        tmp439 = tmp432 / tmp438
        tmp440 = tmp413 / tmp439
        tmp441 = tmp421 + tmp440
        tl.store(out_ptr48 + (x12), tmp413, xmask)
        tl.store(out_ptr50 + (x12), tmp441, xmask)
        tl.store(out_ptr51 + (x12), tmp420, xmask)
    elif xpid >= 1353 and xpid < 1354:
        xpid_offset = xpid - 1353
        xnumel = 256
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x13 = xindex
        tmp442 = tl.load(in_ptr65 + (x13), xmask)
        tmp443 = tl.load(in_ptr66 + (x13), xmask)
        tmp448 = tl.load(in_ptr67 + (x13), xmask)
        tmp455 = tl.load(in_ptr68 + (x13), xmask)
        tmp457 = tl.load(in_ptr69 + (0))
        tmp458 = tl.broadcast_to(tmp457, [XBLOCK])
        tmp444 = tmp443 - tmp442
        tmp445 = 0.09999999999999998
        tmp446 = tmp444 * tmp445
        tmp447 = tmp442 + tmp446
        tmp449 = 0.999
        tmp450 = tmp448 * tmp449
        tmp451 = tmp443 * tmp443
        tmp452 = 0.0010000000000000009
        tmp453 = tmp451 * tmp452
        tmp454 = tmp450 + tmp453
        tmp456 = libdevice.sqrt(tmp454)
        tmp459 = libdevice.pow(tmp449, tmp458)
        tmp460 = 1.0
        tmp461 = tmp459 - tmp460
        tmp462 = -tmp461
        tmp463 = libdevice.sqrt(tmp462)
        tmp464 = tmp456 / tmp463
        tmp465 = 1e-08
        tmp466 = tmp464 + tmp465
        tmp467 = 0.9
        tmp468 = libdevice.pow(tmp467, tmp458)
        tmp469 = tmp468 - tmp460
        tmp470 = 0.001
        tmp471 = tmp469 / tmp470
        tmp472 = 1 / tmp471
        tmp473 = tmp466 / tmp472
        tmp474 = tmp447 / tmp473
        tmp475 = tmp455 + tmp474
        tl.store(out_ptr52 + (x13), tmp447, xmask)
        tl.store(out_ptr54 + (x13), tmp475, xmask)
        tl.store(out_ptr55 + (x13), tmp454, xmask)
    elif xpid >= 1354 and xpid < 1930:
        xpid_offset = xpid - 1354
        xnumel = 589824
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x14 = xindex
        tmp476 = tl.load(in_ptr70 + (x14), None)
        tmp477 = tl.load(in_ptr71 + (x14), None)
        tmp482 = tl.load(in_ptr72 + (x14), None)
        tmp489 = tl.load(in_ptr73 + (x14), None)
        tmp491 = tl.load(in_ptr74 + (0))
        tmp492 = tl.broadcast_to(tmp491, [XBLOCK])
        tmp478 = tmp477 - tmp476
        tmp479 = 0.09999999999999998
        tmp480 = tmp478 * tmp479
        tmp481 = tmp476 + tmp480
        tmp483 = 0.999
        tmp484 = tmp482 * tmp483
        tmp485 = tmp477 * tmp477
        tmp486 = 0.0010000000000000009
        tmp487 = tmp485 * tmp486
        tmp488 = tmp484 + tmp487
        tmp490 = libdevice.sqrt(tmp488)
        tmp493 = libdevice.pow(tmp483, tmp492)
        tmp494 = 1.0
        tmp495 = tmp493 - tmp494
        tmp496 = -tmp495
        tmp497 = libdevice.sqrt(tmp496)
        tmp498 = tmp490 / tmp497
        tmp499 = 1e-08
        tmp500 = tmp498 + tmp499
        tmp501 = 0.9
        tmp502 = libdevice.pow(tmp501, tmp492)
        tmp503 = tmp502 - tmp494
        tmp504 = 0.001
        tmp505 = tmp503 / tmp504
        tmp506 = 1 / tmp505
        tmp507 = tmp500 / tmp506
        tmp508 = tmp481 / tmp507
        tmp509 = tmp489 + tmp508
        tl.store(out_ptr56 + (x14), tmp481, None)
        tl.store(out_ptr58 + (x14), tmp509, None)
        tl.store(out_ptr59 + (x14), tmp488, None)
    elif xpid >= 1930 and xpid < 1931:
        xpid_offset = xpid - 1930
        xnumel = 256
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x15 = xindex
        tmp510 = tl.load(in_ptr75 + (x15), xmask)
        tmp511 = tl.load(in_ptr76 + (x15), xmask)
        tmp516 = tl.load(in_ptr77 + (x15), xmask)
        tmp523 = tl.load(in_ptr78 + (x15), xmask)
        tmp525 = tl.load(in_ptr79 + (0))
        tmp526 = tl.broadcast_to(tmp525, [XBLOCK])
        tmp512 = tmp511 - tmp510
        tmp513 = 0.09999999999999998
        tmp514 = tmp512 * tmp513
        tmp515 = tmp510 + tmp514
        tmp517 = 0.999
        tmp518 = tmp516 * tmp517
        tmp519 = tmp511 * tmp511
        tmp520 = 0.0010000000000000009
        tmp521 = tmp519 * tmp520
        tmp522 = tmp518 + tmp521
        tmp524 = libdevice.sqrt(tmp522)
        tmp527 = libdevice.pow(tmp517, tmp526)
        tmp528 = 1.0
        tmp529 = tmp527 - tmp528
        tmp530 = -tmp529
        tmp531 = libdevice.sqrt(tmp530)
        tmp532 = tmp524 / tmp531
        tmp533 = 1e-08
        tmp534 = tmp532 + tmp533
        tmp535 = 0.9
        tmp536 = libdevice.pow(tmp535, tmp526)
        tmp537 = tmp536 - tmp528
        tmp538 = 0.001
        tmp539 = tmp537 / tmp538
        tmp540 = 1 / tmp539
        tmp541 = tmp534 / tmp540
        tmp542 = tmp515 / tmp541
        tmp543 = tmp523 + tmp542
        tl.store(out_ptr60 + (x15), tmp515, xmask)
        tl.store(out_ptr62 + (x15), tmp543, xmask)
        tl.store(out_ptr63 + (x15), tmp522, xmask)
    elif xpid >= 1931 and xpid < 1932:
        xpid_offset = xpid - 1931
        xnumel = 256
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x16 = xindex
        tmp544 = tl.load(in_ptr80 + (x16), xmask)
        tmp545 = tl.load(in_ptr81 + (x16), xmask)
        tmp550 = tl.load(in_ptr82 + (x16), xmask)
        tmp557 = tl.load(in_ptr83 + (x16), xmask)
        tmp559 = tl.load(in_ptr84 + (0))
        tmp560 = tl.broadcast_to(tmp559, [XBLOCK])
        tmp546 = tmp545 - tmp544
        tmp547 = 0.09999999999999998
        tmp548 = tmp546 * tmp547
        tmp549 = tmp544 + tmp548
        tmp551 = 0.999
        tmp552 = tmp550 * tmp551
        tmp553 = tmp545 * tmp545
        tmp554 = 0.0010000000000000009
        tmp555 = tmp553 * tmp554
        tmp556 = tmp552 + tmp555
        tmp558 = libdevice.sqrt(tmp556)
        tmp561 = libdevice.pow(tmp551, tmp560)
        tmp562 = 1.0
        tmp563 = tmp561 - tmp562
        tmp564 = -tmp563
        tmp565 = libdevice.sqrt(tmp564)
        tmp566 = tmp558 / tmp565
        tmp567 = 1e-08
        tmp568 = tmp566 + tmp567
        tmp569 = 0.9
        tmp570 = libdevice.pow(tmp569, tmp560)
        tmp571 = tmp570 - tmp562
        tmp572 = 0.001
        tmp573 = tmp571 / tmp572
        tmp574 = 1 / tmp573
        tmp575 = tmp568 / tmp574
        tmp576 = tmp549 / tmp575
        tmp577 = tmp557 + tmp576
        tl.store(out_ptr64 + (x16), tmp549, xmask)
        tl.store(out_ptr66 + (x16), tmp577, xmask)
        tl.store(out_ptr67 + (x16), tmp556, xmask)
    elif xpid >= 1932 and xpid < 2188:
        xpid_offset = xpid - 1932
        xnumel = 262144
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x17 = xindex
        tmp578 = tl.load(in_ptr85 + (x17), None)
        tmp579 = tl.load(in_ptr86 + (x17), None)
        tmp584 = tl.load(in_ptr87 + (x17), None)
        tmp591 = tl.load(in_ptr88 + (x17), None)
        tmp593 = tl.load(in_ptr89 + (0))
        tmp594 = tl.broadcast_to(tmp593, [XBLOCK])
        tmp580 = tmp579 - tmp578
        tmp581 = 0.09999999999999998
        tmp582 = tmp580 * tmp581
        tmp583 = tmp578 + tmp582
        tmp585 = 0.999
        tmp586 = tmp584 * tmp585
        tmp587 = tmp579 * tmp579
        tmp588 = 0.0010000000000000009
        tmp589 = tmp587 * tmp588
        tmp590 = tmp586 + tmp589
        tmp592 = libdevice.sqrt(tmp590)
        tmp595 = libdevice.pow(tmp585, tmp594)
        tmp596 = 1.0
        tmp597 = tmp595 - tmp596
        tmp598 = -tmp597
        tmp599 = libdevice.sqrt(tmp598)
        tmp600 = tmp592 / tmp599
        tmp601 = 1e-08
        tmp602 = tmp600 + tmp601
        tmp603 = 0.9
        tmp604 = libdevice.pow(tmp603, tmp594)
        tmp605 = tmp604 - tmp596
        tmp606 = 0.001
        tmp607 = tmp605 / tmp606
        tmp608 = 1 / tmp607
        tmp609 = tmp602 / tmp608
        tmp610 = tmp583 / tmp609
        tmp611 = tmp591 + tmp610
        tl.store(out_ptr68 + (x17), tmp583, None)
        tl.store(out_ptr70 + (x17), tmp611, None)
        tl.store(out_ptr71 + (x17), tmp590, None)
    elif xpid >= 2188 and xpid < 2189:
        xpid_offset = xpid - 2188
        xnumel = 1024
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x18 = xindex
        tmp612 = tl.load(in_ptr90 + (x18), xmask)
        tmp613 = tl.load(in_ptr91 + (x18), xmask)
        tmp618 = tl.load(in_ptr92 + (x18), xmask)
        tmp625 = tl.load(in_ptr93 + (x18), xmask)
        tmp627 = tl.load(in_ptr94 + (0))
        tmp628 = tl.broadcast_to(tmp627, [XBLOCK])
        tmp614 = tmp613 - tmp612
        tmp615 = 0.09999999999999998
        tmp616 = tmp614 * tmp615
        tmp617 = tmp612 + tmp616
        tmp619 = 0.999
        tmp620 = tmp618 * tmp619
        tmp621 = tmp613 * tmp613
        tmp622 = 0.0010000000000000009
        tmp623 = tmp621 * tmp622
        tmp624 = tmp620 + tmp623
        tmp626 = libdevice.sqrt(tmp624)
        tmp629 = libdevice.pow(tmp619, tmp628)
        tmp630 = 1.0
        tmp631 = tmp629 - tmp630
        tmp632 = -tmp631
        tmp633 = libdevice.sqrt(tmp632)
        tmp634 = tmp626 / tmp633
        tmp635 = 1e-08
        tmp636 = tmp634 + tmp635
        tmp637 = 0.9
        tmp638 = libdevice.pow(tmp637, tmp628)
        tmp639 = tmp638 - tmp630
        tmp640 = 0.001
        tmp641 = tmp639 / tmp640
        tmp642 = 1 / tmp641
        tmp643 = tmp636 / tmp642
        tmp644 = tmp617 / tmp643
        tmp645 = tmp625 + tmp644
        tl.store(out_ptr72 + (x18), tmp617, xmask)
        tl.store(out_ptr74 + (x18), tmp645, xmask)
        tl.store(out_ptr75 + (x18), tmp624, xmask)
    elif xpid >= 2189 and xpid < 2190:
        xpid_offset = xpid - 2189
        xnumel = 1024
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x19 = xindex
        tmp646 = tl.load(in_ptr95 + (x19), xmask)
        tmp647 = tl.load(in_ptr96 + (x19), xmask)
        tmp652 = tl.load(in_ptr97 + (x19), xmask)
        tmp659 = tl.load(in_ptr98 + (x19), xmask)
        tmp661 = tl.load(in_ptr99 + (0))
        tmp662 = tl.broadcast_to(tmp661, [XBLOCK])
        tmp648 = tmp647 - tmp646
        tmp649 = 0.09999999999999998
        tmp650 = tmp648 * tmp649
        tmp651 = tmp646 + tmp650
        tmp653 = 0.999
        tmp654 = tmp652 * tmp653
        tmp655 = tmp647 * tmp647
        tmp656 = 0.0010000000000000009
        tmp657 = tmp655 * tmp656
        tmp658 = tmp654 + tmp657
        tmp660 = libdevice.sqrt(tmp658)
        tmp663 = libdevice.pow(tmp653, tmp662)
        tmp664 = 1.0
        tmp665 = tmp663 - tmp664
        tmp666 = -tmp665
        tmp667 = libdevice.sqrt(tmp666)
        tmp668 = tmp660 / tmp667
        tmp669 = 1e-08
        tmp670 = tmp668 + tmp669
        tmp671 = 0.9
        tmp672 = libdevice.pow(tmp671, tmp662)
        tmp673 = tmp672 - tmp664
        tmp674 = 0.001
        tmp675 = tmp673 / tmp674
        tmp676 = 1 / tmp675
        tmp677 = tmp670 / tmp676
        tmp678 = tmp651 / tmp677
        tmp679 = tmp659 + tmp678
        tl.store(out_ptr76 + (x19), tmp651, xmask)
        tl.store(out_ptr78 + (x19), tmp679, xmask)
        tl.store(out_ptr79 + (x19), tmp658, xmask)
    else:
        pass
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/p3/cp3nhzv7phnnq2aqg5qqfwgiwvsud5azg6qdf5q4mrqubopv43eo.py
# Source Nodes: [], Original ATen: []

triton_for_fused_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.foreach(
    num_warps=8,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: '*fp32', 35: '*fp32', 36: '*fp32', 37: '*fp32', 38: '*fp32', 39: '*fp32', 40: '*fp32', 41: '*fp32', 42: '*fp32', 43: '*fp32', 44: '*fp32', 45: '*fp32', 46: '*fp32', 47: '*fp32', 48: '*fp32', 49: '*fp32', 50: '*fp32', 51: '*fp32', 52: '*fp32', 53: '*fp32', 54: '*fp32', 55: '*fp32', 56: '*fp32', 57: '*fp32', 58: '*fp32', 59: '*fp32', 60: '*fp32', 61: '*fp32', 62: '*fp32', 63: '*fp32', 64: '*fp32', 65: '*fp32', 66: '*fp32', 67: '*fp32', 68: '*fp32', 69: '*fp32', 70: '*fp32', 71: '*fp32', 72: '*fp32', 73: '*fp32', 74: '*fp32', 75: '*fp32', 76: '*fp32', 77: '*fp32', 78: '*fp32', 79: '*fp32', 80: '*fp32', 81: '*fp32', 82: '*fp32', 83: '*fp32', 84: '*fp32', 85: '*fp32', 86: '*fp32', 87: '*fp32', 88: '*fp32', 89: '*fp32', 90: '*fp32', 91: '*fp32', 92: '*fp32', 93: '*fp32', 94: '*fp32', 95: '*fp32', 96: '*fp32', 97: '*fp32', 98: '*fp32', 99: '*fp32', 100: '*fp32', 101: '*fp32', 102: '*fp32', 103: '*fp32', 104: '*fp32', 105: '*fp32', 106: '*fp32', 107: '*fp32', 108: '*fp32', 109: '*fp32', 110: '*fp32', 111: '*fp32', 112: '*fp32', 113: '*fp32', 114: '*fp32', 115: '*fp32', 116: '*fp32', 117: '*fp32', 118: '*fp32', 119: '*fp32', 120: '*fp32', 121: '*fp32', 122: '*fp32', 123: '*fp32', 124: '*fp32', 125: '*fp32', 126: '*fp32', 127: '*fp32', 128: '*fp32', 129: '*fp32', 130: '*fp32', 131: '*fp32', 132: '*fp32', 133: '*fp32', 134: '*fp32', 135: '*fp32', 136: '*fp32', 137: '*fp32', 138: '*fp32', 139: '*fp32', 140: '*fp32', 141: '*fp32', 142: '*fp32', 143: '*fp32', 144: '*fp32', 145: '*fp32', 146: '*fp32', 147: '*fp32', 148: '*fp32', 149: '*fp32', 150: '*fp32', 151: '*fp32', 152: '*fp32', 153: '*fp32', 154: '*fp32', 155: '*fp32', 156: '*fp32', 157: '*fp32', 158: '*fp32', 159: '*fp32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'kernel_name': 'triton_for_fused_8', 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, in_ptr47, in_ptr48, in_ptr49, in_ptr50, in_ptr51, in_ptr52, in_ptr53, in_ptr54, in_ptr55, in_ptr56, in_ptr57, in_ptr58, in_ptr59, in_ptr60, in_ptr61, in_ptr62, in_ptr63, in_ptr64, in_ptr65, in_ptr66, in_ptr67, in_ptr68, in_ptr69, in_ptr70, in_ptr71, in_ptr72, in_ptr73, in_ptr74, in_ptr75, in_ptr76, in_ptr77, in_ptr78, in_ptr79, in_ptr80, in_ptr81, in_ptr82, in_ptr83, in_ptr84, in_ptr85, in_ptr86, in_ptr87, in_ptr88, in_ptr89, in_ptr90, in_ptr91, in_ptr92, in_ptr93, in_ptr94, in_ptr95, in_ptr96, in_ptr97, in_ptr98, in_ptr99, out_ptr0, out_ptr2, out_ptr3, out_ptr4, out_ptr6, out_ptr7, out_ptr8, out_ptr10, out_ptr11, out_ptr12, out_ptr14, out_ptr15, out_ptr16, out_ptr18, out_ptr19, out_ptr20, out_ptr22, out_ptr23, out_ptr24, out_ptr26, out_ptr27, out_ptr28, out_ptr30, out_ptr31, out_ptr32, out_ptr34, out_ptr35, out_ptr36, out_ptr38, out_ptr39, out_ptr40, out_ptr42, out_ptr43, out_ptr44, out_ptr46, out_ptr47, out_ptr48, out_ptr50, out_ptr51, out_ptr52, out_ptr54, out_ptr55, out_ptr56, out_ptr58, out_ptr59, out_ptr60, out_ptr62, out_ptr63, out_ptr64, out_ptr66, out_ptr67, out_ptr68, out_ptr70, out_ptr71, out_ptr72, out_ptr74, out_ptr75, out_ptr76, out_ptr78, out_ptr79):
    xpid = tl.program_id(0)
    XBLOCK: tl.constexpr = 1024
    if xpid >= 0 and xpid < 256:
        xpid_offset = xpid - 0
        xnumel = 262144
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
        tmp14 = libdevice.sqrt(tmp12)
        tmp17 = libdevice.pow(tmp7, tmp16)
        tmp18 = 1.0
        tmp19 = tmp17 - tmp18
        tmp20 = -tmp19
        tmp21 = libdevice.sqrt(tmp20)
        tmp22 = tmp14 / tmp21
        tmp23 = 1e-08
        tmp24 = tmp22 + tmp23
        tmp25 = 0.9
        tmp26 = libdevice.pow(tmp25, tmp16)
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
    elif xpid >= 256 and xpid < 257:
        xpid_offset = xpid - 256
        xnumel = 256
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
        tmp48 = libdevice.sqrt(tmp46)
        tmp51 = libdevice.pow(tmp41, tmp50)
        tmp52 = 1.0
        tmp53 = tmp51 - tmp52
        tmp54 = -tmp53
        tmp55 = libdevice.sqrt(tmp54)
        tmp56 = tmp48 / tmp55
        tmp57 = 1e-08
        tmp58 = tmp56 + tmp57
        tmp59 = 0.9
        tmp60 = libdevice.pow(tmp59, tmp50)
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
    elif xpid >= 257 and xpid < 258:
        xpid_offset = xpid - 257
        xnumel = 256
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x2 = xindex
        tmp68 = tl.load(in_ptr10 + (x2), xmask)
        tmp69 = tl.load(in_ptr11 + (x2), xmask)
        tmp74 = tl.load(in_ptr12 + (x2), xmask)
        tmp81 = tl.load(in_ptr13 + (x2), xmask)
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
        tmp82 = libdevice.sqrt(tmp80)
        tmp85 = libdevice.pow(tmp75, tmp84)
        tmp86 = 1.0
        tmp87 = tmp85 - tmp86
        tmp88 = -tmp87
        tmp89 = libdevice.sqrt(tmp88)
        tmp90 = tmp82 / tmp89
        tmp91 = 1e-08
        tmp92 = tmp90 + tmp91
        tmp93 = 0.9
        tmp94 = libdevice.pow(tmp93, tmp84)
        tmp95 = tmp94 - tmp86
        tmp96 = 0.001
        tmp97 = tmp95 / tmp96
        tmp98 = 1 / tmp97
        tmp99 = tmp92 / tmp98
        tmp100 = tmp73 / tmp99
        tmp101 = tmp81 + tmp100
        tl.store(out_ptr8 + (x2), tmp73, xmask)
        tl.store(out_ptr10 + (x2), tmp101, xmask)
        tl.store(out_ptr11 + (x2), tmp80, xmask)
    elif xpid >= 258 and xpid < 834:
        xpid_offset = xpid - 258
        xnumel = 589824
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x3 = xindex
        tmp102 = tl.load(in_ptr15 + (x3), None)
        tmp103 = tl.load(in_ptr16 + (x3), None)
        tmp108 = tl.load(in_ptr17 + (x3), None)
        tmp115 = tl.load(in_ptr18 + (x3), None)
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
        tmp116 = libdevice.sqrt(tmp114)
        tmp119 = libdevice.pow(tmp109, tmp118)
        tmp120 = 1.0
        tmp121 = tmp119 - tmp120
        tmp122 = -tmp121
        tmp123 = libdevice.sqrt(tmp122)
        tmp124 = tmp116 / tmp123
        tmp125 = 1e-08
        tmp126 = tmp124 + tmp125
        tmp127 = 0.9
        tmp128 = libdevice.pow(tmp127, tmp118)
        tmp129 = tmp128 - tmp120
        tmp130 = 0.001
        tmp131 = tmp129 / tmp130
        tmp132 = 1 / tmp131
        tmp133 = tmp126 / tmp132
        tmp134 = tmp107 / tmp133
        tmp135 = tmp115 + tmp134
        tl.store(out_ptr12 + (x3), tmp107, None)
        tl.store(out_ptr14 + (x3), tmp135, None)
        tl.store(out_ptr15 + (x3), tmp114, None)
    elif xpid >= 834 and xpid < 835:
        xpid_offset = xpid - 834
        xnumel = 256
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
        tmp150 = libdevice.sqrt(tmp148)
        tmp153 = libdevice.pow(tmp143, tmp152)
        tmp154 = 1.0
        tmp155 = tmp153 - tmp154
        tmp156 = -tmp155
        tmp157 = libdevice.sqrt(tmp156)
        tmp158 = tmp150 / tmp157
        tmp159 = 1e-08
        tmp160 = tmp158 + tmp159
        tmp161 = 0.9
        tmp162 = libdevice.pow(tmp161, tmp152)
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
    elif xpid >= 835 and xpid < 836:
        xpid_offset = xpid - 835
        xnumel = 256
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
        tmp184 = libdevice.sqrt(tmp182)
        tmp187 = libdevice.pow(tmp177, tmp186)
        tmp188 = 1.0
        tmp189 = tmp187 - tmp188
        tmp190 = -tmp189
        tmp191 = libdevice.sqrt(tmp190)
        tmp192 = tmp184 / tmp191
        tmp193 = 1e-08
        tmp194 = tmp192 + tmp193
        tmp195 = 0.9
        tmp196 = libdevice.pow(tmp195, tmp186)
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
    elif xpid >= 836 and xpid < 1092:
        xpid_offset = xpid - 836
        xnumel = 262144
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x6 = xindex
        tmp204 = tl.load(in_ptr30 + (x6), None)
        tmp205 = tl.load(in_ptr31 + (x6), None)
        tmp210 = tl.load(in_ptr32 + (x6), None)
        tmp217 = tl.load(in_ptr33 + (x6), None)
        tmp219 = tl.load(in_ptr34 + (0))
        tmp220 = tl.broadcast_to(tmp219, [XBLOCK])
        tmp206 = tmp205 - tmp204
        tmp207 = 0.09999999999999998
        tmp208 = tmp206 * tmp207
        tmp209 = tmp204 + tmp208
        tmp211 = 0.999
        tmp212 = tmp210 * tmp211
        tmp213 = tmp205 * tmp205
        tmp214 = 0.0010000000000000009
        tmp215 = tmp213 * tmp214
        tmp216 = tmp212 + tmp215
        tmp218 = libdevice.sqrt(tmp216)
        tmp221 = libdevice.pow(tmp211, tmp220)
        tmp222 = 1.0
        tmp223 = tmp221 - tmp222
        tmp224 = -tmp223
        tmp225 = libdevice.sqrt(tmp224)
        tmp226 = tmp218 / tmp225
        tmp227 = 1e-08
        tmp228 = tmp226 + tmp227
        tmp229 = 0.9
        tmp230 = libdevice.pow(tmp229, tmp220)
        tmp231 = tmp230 - tmp222
        tmp232 = 0.001
        tmp233 = tmp231 / tmp232
        tmp234 = 1 / tmp233
        tmp235 = tmp228 / tmp234
        tmp236 = tmp209 / tmp235
        tmp237 = tmp217 + tmp236
        tl.store(out_ptr24 + (x6), tmp209, None)
        tl.store(out_ptr26 + (x6), tmp237, None)
        tl.store(out_ptr27 + (x6), tmp216, None)
    elif xpid >= 1092 and xpid < 1093:
        xpid_offset = xpid - 1092
        xnumel = 1024
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x7 = xindex
        tmp238 = tl.load(in_ptr35 + (x7), xmask)
        tmp239 = tl.load(in_ptr36 + (x7), xmask)
        tmp244 = tl.load(in_ptr37 + (x7), xmask)
        tmp251 = tl.load(in_ptr38 + (x7), xmask)
        tmp253 = tl.load(in_ptr39 + (0))
        tmp254 = tl.broadcast_to(tmp253, [XBLOCK])
        tmp240 = tmp239 - tmp238
        tmp241 = 0.09999999999999998
        tmp242 = tmp240 * tmp241
        tmp243 = tmp238 + tmp242
        tmp245 = 0.999
        tmp246 = tmp244 * tmp245
        tmp247 = tmp239 * tmp239
        tmp248 = 0.0010000000000000009
        tmp249 = tmp247 * tmp248
        tmp250 = tmp246 + tmp249
        tmp252 = libdevice.sqrt(tmp250)
        tmp255 = libdevice.pow(tmp245, tmp254)
        tmp256 = 1.0
        tmp257 = tmp255 - tmp256
        tmp258 = -tmp257
        tmp259 = libdevice.sqrt(tmp258)
        tmp260 = tmp252 / tmp259
        tmp261 = 1e-08
        tmp262 = tmp260 + tmp261
        tmp263 = 0.9
        tmp264 = libdevice.pow(tmp263, tmp254)
        tmp265 = tmp264 - tmp256
        tmp266 = 0.001
        tmp267 = tmp265 / tmp266
        tmp268 = 1 / tmp267
        tmp269 = tmp262 / tmp268
        tmp270 = tmp243 / tmp269
        tmp271 = tmp251 + tmp270
        tl.store(out_ptr28 + (x7), tmp243, xmask)
        tl.store(out_ptr30 + (x7), tmp271, xmask)
        tl.store(out_ptr31 + (x7), tmp250, xmask)
    elif xpid >= 1093 and xpid < 1094:
        xpid_offset = xpid - 1093
        xnumel = 1024
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x8 = xindex
        tmp272 = tl.load(in_ptr40 + (x8), xmask)
        tmp273 = tl.load(in_ptr41 + (x8), xmask)
        tmp278 = tl.load(in_ptr42 + (x8), xmask)
        tmp285 = tl.load(in_ptr43 + (x8), xmask)
        tmp287 = tl.load(in_ptr44 + (0))
        tmp288 = tl.broadcast_to(tmp287, [XBLOCK])
        tmp274 = tmp273 - tmp272
        tmp275 = 0.09999999999999998
        tmp276 = tmp274 * tmp275
        tmp277 = tmp272 + tmp276
        tmp279 = 0.999
        tmp280 = tmp278 * tmp279
        tmp281 = tmp273 * tmp273
        tmp282 = 0.0010000000000000009
        tmp283 = tmp281 * tmp282
        tmp284 = tmp280 + tmp283
        tmp286 = libdevice.sqrt(tmp284)
        tmp289 = libdevice.pow(tmp279, tmp288)
        tmp290 = 1.0
        tmp291 = tmp289 - tmp290
        tmp292 = -tmp291
        tmp293 = libdevice.sqrt(tmp292)
        tmp294 = tmp286 / tmp293
        tmp295 = 1e-08
        tmp296 = tmp294 + tmp295
        tmp297 = 0.9
        tmp298 = libdevice.pow(tmp297, tmp288)
        tmp299 = tmp298 - tmp290
        tmp300 = 0.001
        tmp301 = tmp299 / tmp300
        tmp302 = 1 / tmp301
        tmp303 = tmp296 / tmp302
        tmp304 = tmp277 / tmp303
        tmp305 = tmp285 + tmp304
        tl.store(out_ptr32 + (x8), tmp277, xmask)
        tl.store(out_ptr34 + (x8), tmp305, xmask)
        tl.store(out_ptr35 + (x8), tmp284, xmask)
    elif xpid >= 1094 and xpid < 1606:
        xpid_offset = xpid - 1094
        xnumel = 524288
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x9 = xindex
        tmp306 = tl.load(in_ptr45 + (x9), None)
        tmp307 = tl.load(in_ptr46 + (x9), None)
        tmp312 = tl.load(in_ptr47 + (x9), None)
        tmp319 = tl.load(in_ptr48 + (x9), None)
        tmp321 = tl.load(in_ptr49 + (0))
        tmp322 = tl.broadcast_to(tmp321, [XBLOCK])
        tmp308 = tmp307 - tmp306
        tmp309 = 0.09999999999999998
        tmp310 = tmp308 * tmp309
        tmp311 = tmp306 + tmp310
        tmp313 = 0.999
        tmp314 = tmp312 * tmp313
        tmp315 = tmp307 * tmp307
        tmp316 = 0.0010000000000000009
        tmp317 = tmp315 * tmp316
        tmp318 = tmp314 + tmp317
        tmp320 = libdevice.sqrt(tmp318)
        tmp323 = libdevice.pow(tmp313, tmp322)
        tmp324 = 1.0
        tmp325 = tmp323 - tmp324
        tmp326 = -tmp325
        tmp327 = libdevice.sqrt(tmp326)
        tmp328 = tmp320 / tmp327
        tmp329 = 1e-08
        tmp330 = tmp328 + tmp329
        tmp331 = 0.9
        tmp332 = libdevice.pow(tmp331, tmp322)
        tmp333 = tmp332 - tmp324
        tmp334 = 0.001
        tmp335 = tmp333 / tmp334
        tmp336 = 1 / tmp335
        tmp337 = tmp330 / tmp336
        tmp338 = tmp311 / tmp337
        tmp339 = tmp319 + tmp338
        tl.store(out_ptr36 + (x9), tmp311, None)
        tl.store(out_ptr38 + (x9), tmp339, None)
        tl.store(out_ptr39 + (x9), tmp318, None)
    elif xpid >= 1606 and xpid < 1607:
        xpid_offset = xpid - 1606
        xnumel = 512
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x10 = xindex
        tmp340 = tl.load(in_ptr50 + (x10), xmask)
        tmp341 = tl.load(in_ptr51 + (x10), xmask)
        tmp346 = tl.load(in_ptr52 + (x10), xmask)
        tmp353 = tl.load(in_ptr53 + (x10), xmask)
        tmp355 = tl.load(in_ptr54 + (0))
        tmp356 = tl.broadcast_to(tmp355, [XBLOCK])
        tmp342 = tmp341 - tmp340
        tmp343 = 0.09999999999999998
        tmp344 = tmp342 * tmp343
        tmp345 = tmp340 + tmp344
        tmp347 = 0.999
        tmp348 = tmp346 * tmp347
        tmp349 = tmp341 * tmp341
        tmp350 = 0.0010000000000000009
        tmp351 = tmp349 * tmp350
        tmp352 = tmp348 + tmp351
        tmp354 = libdevice.sqrt(tmp352)
        tmp357 = libdevice.pow(tmp347, tmp356)
        tmp358 = 1.0
        tmp359 = tmp357 - tmp358
        tmp360 = -tmp359
        tmp361 = libdevice.sqrt(tmp360)
        tmp362 = tmp354 / tmp361
        tmp363 = 1e-08
        tmp364 = tmp362 + tmp363
        tmp365 = 0.9
        tmp366 = libdevice.pow(tmp365, tmp356)
        tmp367 = tmp366 - tmp358
        tmp368 = 0.001
        tmp369 = tmp367 / tmp368
        tmp370 = 1 / tmp369
        tmp371 = tmp364 / tmp370
        tmp372 = tmp345 / tmp371
        tmp373 = tmp353 + tmp372
        tl.store(out_ptr40 + (x10), tmp345, xmask)
        tl.store(out_ptr42 + (x10), tmp373, xmask)
        tl.store(out_ptr43 + (x10), tmp352, xmask)
    elif xpid >= 1607 and xpid < 1608:
        xpid_offset = xpid - 1607
        xnumel = 512
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x11 = xindex
        tmp374 = tl.load(in_ptr55 + (x11), xmask)
        tmp375 = tl.load(in_ptr56 + (x11), xmask)
        tmp380 = tl.load(in_ptr57 + (x11), xmask)
        tmp387 = tl.load(in_ptr58 + (x11), xmask)
        tmp389 = tl.load(in_ptr59 + (0))
        tmp390 = tl.broadcast_to(tmp389, [XBLOCK])
        tmp376 = tmp375 - tmp374
        tmp377 = 0.09999999999999998
        tmp378 = tmp376 * tmp377
        tmp379 = tmp374 + tmp378
        tmp381 = 0.999
        tmp382 = tmp380 * tmp381
        tmp383 = tmp375 * tmp375
        tmp384 = 0.0010000000000000009
        tmp385 = tmp383 * tmp384
        tmp386 = tmp382 + tmp385
        tmp388 = libdevice.sqrt(tmp386)
        tmp391 = libdevice.pow(tmp381, tmp390)
        tmp392 = 1.0
        tmp393 = tmp391 - tmp392
        tmp394 = -tmp393
        tmp395 = libdevice.sqrt(tmp394)
        tmp396 = tmp388 / tmp395
        tmp397 = 1e-08
        tmp398 = tmp396 + tmp397
        tmp399 = 0.9
        tmp400 = libdevice.pow(tmp399, tmp390)
        tmp401 = tmp400 - tmp392
        tmp402 = 0.001
        tmp403 = tmp401 / tmp402
        tmp404 = 1 / tmp403
        tmp405 = tmp398 / tmp404
        tmp406 = tmp379 / tmp405
        tmp407 = tmp387 + tmp406
        tl.store(out_ptr44 + (x11), tmp379, xmask)
        tl.store(out_ptr46 + (x11), tmp407, xmask)
        tl.store(out_ptr47 + (x11), tmp386, xmask)
    elif xpid >= 1608 and xpid < 3912:
        xpid_offset = xpid - 1608
        xnumel = 2359296
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x12 = xindex
        tmp408 = tl.load(in_ptr60 + (x12), None)
        tmp409 = tl.load(in_ptr61 + (x12), None)
        tmp414 = tl.load(in_ptr62 + (x12), None)
        tmp421 = tl.load(in_ptr63 + (x12), None)
        tmp423 = tl.load(in_ptr64 + (0))
        tmp424 = tl.broadcast_to(tmp423, [XBLOCK])
        tmp410 = tmp409 - tmp408
        tmp411 = 0.09999999999999998
        tmp412 = tmp410 * tmp411
        tmp413 = tmp408 + tmp412
        tmp415 = 0.999
        tmp416 = tmp414 * tmp415
        tmp417 = tmp409 * tmp409
        tmp418 = 0.0010000000000000009
        tmp419 = tmp417 * tmp418
        tmp420 = tmp416 + tmp419
        tmp422 = libdevice.sqrt(tmp420)
        tmp425 = libdevice.pow(tmp415, tmp424)
        tmp426 = 1.0
        tmp427 = tmp425 - tmp426
        tmp428 = -tmp427
        tmp429 = libdevice.sqrt(tmp428)
        tmp430 = tmp422 / tmp429
        tmp431 = 1e-08
        tmp432 = tmp430 + tmp431
        tmp433 = 0.9
        tmp434 = libdevice.pow(tmp433, tmp424)
        tmp435 = tmp434 - tmp426
        tmp436 = 0.001
        tmp437 = tmp435 / tmp436
        tmp438 = 1 / tmp437
        tmp439 = tmp432 / tmp438
        tmp440 = tmp413 / tmp439
        tmp441 = tmp421 + tmp440
        tl.store(out_ptr48 + (x12), tmp413, None)
        tl.store(out_ptr50 + (x12), tmp441, None)
        tl.store(out_ptr51 + (x12), tmp420, None)
    elif xpid >= 3912 and xpid < 3913:
        xpid_offset = xpid - 3912
        xnumel = 512
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x13 = xindex
        tmp442 = tl.load(in_ptr65 + (x13), xmask)
        tmp443 = tl.load(in_ptr66 + (x13), xmask)
        tmp448 = tl.load(in_ptr67 + (x13), xmask)
        tmp455 = tl.load(in_ptr68 + (x13), xmask)
        tmp457 = tl.load(in_ptr69 + (0))
        tmp458 = tl.broadcast_to(tmp457, [XBLOCK])
        tmp444 = tmp443 - tmp442
        tmp445 = 0.09999999999999998
        tmp446 = tmp444 * tmp445
        tmp447 = tmp442 + tmp446
        tmp449 = 0.999
        tmp450 = tmp448 * tmp449
        tmp451 = tmp443 * tmp443
        tmp452 = 0.0010000000000000009
        tmp453 = tmp451 * tmp452
        tmp454 = tmp450 + tmp453
        tmp456 = libdevice.sqrt(tmp454)
        tmp459 = libdevice.pow(tmp449, tmp458)
        tmp460 = 1.0
        tmp461 = tmp459 - tmp460
        tmp462 = -tmp461
        tmp463 = libdevice.sqrt(tmp462)
        tmp464 = tmp456 / tmp463
        tmp465 = 1e-08
        tmp466 = tmp464 + tmp465
        tmp467 = 0.9
        tmp468 = libdevice.pow(tmp467, tmp458)
        tmp469 = tmp468 - tmp460
        tmp470 = 0.001
        tmp471 = tmp469 / tmp470
        tmp472 = 1 / tmp471
        tmp473 = tmp466 / tmp472
        tmp474 = tmp447 / tmp473
        tmp475 = tmp455 + tmp474
        tl.store(out_ptr52 + (x13), tmp447, xmask)
        tl.store(out_ptr54 + (x13), tmp475, xmask)
        tl.store(out_ptr55 + (x13), tmp454, xmask)
    elif xpid >= 3913 and xpid < 3914:
        xpid_offset = xpid - 3913
        xnumel = 512
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x14 = xindex
        tmp476 = tl.load(in_ptr70 + (x14), xmask)
        tmp477 = tl.load(in_ptr71 + (x14), xmask)
        tmp482 = tl.load(in_ptr72 + (x14), xmask)
        tmp489 = tl.load(in_ptr73 + (x14), xmask)
        tmp491 = tl.load(in_ptr74 + (0))
        tmp492 = tl.broadcast_to(tmp491, [XBLOCK])
        tmp478 = tmp477 - tmp476
        tmp479 = 0.09999999999999998
        tmp480 = tmp478 * tmp479
        tmp481 = tmp476 + tmp480
        tmp483 = 0.999
        tmp484 = tmp482 * tmp483
        tmp485 = tmp477 * tmp477
        tmp486 = 0.0010000000000000009
        tmp487 = tmp485 * tmp486
        tmp488 = tmp484 + tmp487
        tmp490 = libdevice.sqrt(tmp488)
        tmp493 = libdevice.pow(tmp483, tmp492)
        tmp494 = 1.0
        tmp495 = tmp493 - tmp494
        tmp496 = -tmp495
        tmp497 = libdevice.sqrt(tmp496)
        tmp498 = tmp490 / tmp497
        tmp499 = 1e-08
        tmp500 = tmp498 + tmp499
        tmp501 = 0.9
        tmp502 = libdevice.pow(tmp501, tmp492)
        tmp503 = tmp502 - tmp494
        tmp504 = 0.001
        tmp505 = tmp503 / tmp504
        tmp506 = 1 / tmp505
        tmp507 = tmp500 / tmp506
        tmp508 = tmp481 / tmp507
        tmp509 = tmp489 + tmp508
        tl.store(out_ptr56 + (x14), tmp481, xmask)
        tl.store(out_ptr58 + (x14), tmp509, xmask)
        tl.store(out_ptr59 + (x14), tmp488, xmask)
    elif xpid >= 3914 and xpid < 4938:
        xpid_offset = xpid - 3914
        xnumel = 1048576
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x15 = xindex
        tmp510 = tl.load(in_ptr75 + (x15), None)
        tmp511 = tl.load(in_ptr76 + (x15), None)
        tmp516 = tl.load(in_ptr77 + (x15), None)
        tmp523 = tl.load(in_ptr78 + (x15), None)
        tmp525 = tl.load(in_ptr79 + (0))
        tmp526 = tl.broadcast_to(tmp525, [XBLOCK])
        tmp512 = tmp511 - tmp510
        tmp513 = 0.09999999999999998
        tmp514 = tmp512 * tmp513
        tmp515 = tmp510 + tmp514
        tmp517 = 0.999
        tmp518 = tmp516 * tmp517
        tmp519 = tmp511 * tmp511
        tmp520 = 0.0010000000000000009
        tmp521 = tmp519 * tmp520
        tmp522 = tmp518 + tmp521
        tmp524 = libdevice.sqrt(tmp522)
        tmp527 = libdevice.pow(tmp517, tmp526)
        tmp528 = 1.0
        tmp529 = tmp527 - tmp528
        tmp530 = -tmp529
        tmp531 = libdevice.sqrt(tmp530)
        tmp532 = tmp524 / tmp531
        tmp533 = 1e-08
        tmp534 = tmp532 + tmp533
        tmp535 = 0.9
        tmp536 = libdevice.pow(tmp535, tmp526)
        tmp537 = tmp536 - tmp528
        tmp538 = 0.001
        tmp539 = tmp537 / tmp538
        tmp540 = 1 / tmp539
        tmp541 = tmp534 / tmp540
        tmp542 = tmp515 / tmp541
        tmp543 = tmp523 + tmp542
        tl.store(out_ptr60 + (x15), tmp515, None)
        tl.store(out_ptr62 + (x15), tmp543, None)
        tl.store(out_ptr63 + (x15), tmp522, None)
    elif xpid >= 4938 and xpid < 4940:
        xpid_offset = xpid - 4938
        xnumel = 2048
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x16 = xindex
        tmp544 = tl.load(in_ptr80 + (x16), None)
        tmp545 = tl.load(in_ptr81 + (x16), None)
        tmp550 = tl.load(in_ptr82 + (x16), None)
        tmp557 = tl.load(in_ptr83 + (x16), None)
        tmp559 = tl.load(in_ptr84 + (0))
        tmp560 = tl.broadcast_to(tmp559, [XBLOCK])
        tmp546 = tmp545 - tmp544
        tmp547 = 0.09999999999999998
        tmp548 = tmp546 * tmp547
        tmp549 = tmp544 + tmp548
        tmp551 = 0.999
        tmp552 = tmp550 * tmp551
        tmp553 = tmp545 * tmp545
        tmp554 = 0.0010000000000000009
        tmp555 = tmp553 * tmp554
        tmp556 = tmp552 + tmp555
        tmp558 = libdevice.sqrt(tmp556)
        tmp561 = libdevice.pow(tmp551, tmp560)
        tmp562 = 1.0
        tmp563 = tmp561 - tmp562
        tmp564 = -tmp563
        tmp565 = libdevice.sqrt(tmp564)
        tmp566 = tmp558 / tmp565
        tmp567 = 1e-08
        tmp568 = tmp566 + tmp567
        tmp569 = 0.9
        tmp570 = libdevice.pow(tmp569, tmp560)
        tmp571 = tmp570 - tmp562
        tmp572 = 0.001
        tmp573 = tmp571 / tmp572
        tmp574 = 1 / tmp573
        tmp575 = tmp568 / tmp574
        tmp576 = tmp549 / tmp575
        tmp577 = tmp557 + tmp576
        tl.store(out_ptr64 + (x16), tmp549, None)
        tl.store(out_ptr66 + (x16), tmp577, None)
        tl.store(out_ptr67 + (x16), tmp556, None)
    elif xpid >= 4940 and xpid < 4942:
        xpid_offset = xpid - 4940
        xnumel = 2048
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x17 = xindex
        tmp578 = tl.load(in_ptr85 + (x17), None)
        tmp579 = tl.load(in_ptr86 + (x17), None)
        tmp584 = tl.load(in_ptr87 + (x17), None)
        tmp591 = tl.load(in_ptr88 + (x17), None)
        tmp593 = tl.load(in_ptr89 + (0))
        tmp594 = tl.broadcast_to(tmp593, [XBLOCK])
        tmp580 = tmp579 - tmp578
        tmp581 = 0.09999999999999998
        tmp582 = tmp580 * tmp581
        tmp583 = tmp578 + tmp582
        tmp585 = 0.999
        tmp586 = tmp584 * tmp585
        tmp587 = tmp579 * tmp579
        tmp588 = 0.0010000000000000009
        tmp589 = tmp587 * tmp588
        tmp590 = tmp586 + tmp589
        tmp592 = libdevice.sqrt(tmp590)
        tmp595 = libdevice.pow(tmp585, tmp594)
        tmp596 = 1.0
        tmp597 = tmp595 - tmp596
        tmp598 = -tmp597
        tmp599 = libdevice.sqrt(tmp598)
        tmp600 = tmp592 / tmp599
        tmp601 = 1e-08
        tmp602 = tmp600 + tmp601
        tmp603 = 0.9
        tmp604 = libdevice.pow(tmp603, tmp594)
        tmp605 = tmp604 - tmp596
        tmp606 = 0.001
        tmp607 = tmp605 / tmp606
        tmp608 = 1 / tmp607
        tmp609 = tmp602 / tmp608
        tmp610 = tmp583 / tmp609
        tmp611 = tmp591 + tmp610
        tl.store(out_ptr68 + (x17), tmp583, None)
        tl.store(out_ptr70 + (x17), tmp611, None)
        tl.store(out_ptr71 + (x17), tmp590, None)
    elif xpid >= 4942 and xpid < 6990:
        xpid_offset = xpid - 4942
        xnumel = 2097152
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x18 = xindex
        tmp612 = tl.load(in_ptr90 + (x18), None)
        tmp613 = tl.load(in_ptr91 + (x18), None)
        tmp618 = tl.load(in_ptr92 + (x18), None)
        tmp625 = tl.load(in_ptr93 + (x18), None)
        tmp627 = tl.load(in_ptr94 + (0))
        tmp628 = tl.broadcast_to(tmp627, [XBLOCK])
        tmp614 = tmp613 - tmp612
        tmp615 = 0.09999999999999998
        tmp616 = tmp614 * tmp615
        tmp617 = tmp612 + tmp616
        tmp619 = 0.999
        tmp620 = tmp618 * tmp619
        tmp621 = tmp613 * tmp613
        tmp622 = 0.0010000000000000009
        tmp623 = tmp621 * tmp622
        tmp624 = tmp620 + tmp623
        tmp626 = libdevice.sqrt(tmp624)
        tmp629 = libdevice.pow(tmp619, tmp628)
        tmp630 = 1.0
        tmp631 = tmp629 - tmp630
        tmp632 = -tmp631
        tmp633 = libdevice.sqrt(tmp632)
        tmp634 = tmp626 / tmp633
        tmp635 = 1e-08
        tmp636 = tmp634 + tmp635
        tmp637 = 0.9
        tmp638 = libdevice.pow(tmp637, tmp628)
        tmp639 = tmp638 - tmp630
        tmp640 = 0.001
        tmp641 = tmp639 / tmp640
        tmp642 = 1 / tmp641
        tmp643 = tmp636 / tmp642
        tmp644 = tmp617 / tmp643
        tmp645 = tmp625 + tmp644
        tl.store(out_ptr72 + (x18), tmp617, None)
        tl.store(out_ptr74 + (x18), tmp645, None)
        tl.store(out_ptr75 + (x18), tmp624, None)
    elif xpid >= 6990 and xpid < 6992:
        xpid_offset = xpid - 6990
        xnumel = 2048
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x19 = xindex
        tmp646 = tl.load(in_ptr95 + (x19), None)
        tmp647 = tl.load(in_ptr96 + (x19), None)
        tmp652 = tl.load(in_ptr97 + (x19), None)
        tmp659 = tl.load(in_ptr98 + (x19), None)
        tmp661 = tl.load(in_ptr99 + (0))
        tmp662 = tl.broadcast_to(tmp661, [XBLOCK])
        tmp648 = tmp647 - tmp646
        tmp649 = 0.09999999999999998
        tmp650 = tmp648 * tmp649
        tmp651 = tmp646 + tmp650
        tmp653 = 0.999
        tmp654 = tmp652 * tmp653
        tmp655 = tmp647 * tmp647
        tmp656 = 0.0010000000000000009
        tmp657 = tmp655 * tmp656
        tmp658 = tmp654 + tmp657
        tmp660 = libdevice.sqrt(tmp658)
        tmp663 = libdevice.pow(tmp653, tmp662)
        tmp664 = 1.0
        tmp665 = tmp663 - tmp664
        tmp666 = -tmp665
        tmp667 = libdevice.sqrt(tmp666)
        tmp668 = tmp660 / tmp667
        tmp669 = 1e-08
        tmp670 = tmp668 + tmp669
        tmp671 = 0.9
        tmp672 = libdevice.pow(tmp671, tmp662)
        tmp673 = tmp672 - tmp664
        tmp674 = 0.001
        tmp675 = tmp673 / tmp674
        tmp676 = 1 / tmp675
        tmp677 = tmp670 / tmp676
        tmp678 = tmp651 / tmp677
        tmp679 = tmp659 + tmp678
        tl.store(out_ptr76 + (x19), tmp651, None)
        tl.store(out_ptr78 + (x19), tmp679, None)
        tl.store(out_ptr79 + (x19), tmp658, None)
    else:
        pass
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/wn/cwngc4vdpchtl4hjdqm5jp3m5wbsnclapn577mkqbt2gkapt7iq5.py
# Source Nodes: [], Original ATen: []

triton_for_fused_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.foreach(
    num_warps=8,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: '*fp32', 35: '*fp32', 36: '*fp32', 37: '*fp32', 38: '*fp32', 39: '*fp32', 40: '*fp32', 41: '*fp32', 42: '*fp32', 43: '*fp32', 44: '*fp32', 45: '*fp32', 46: '*fp32', 47: '*fp32', 48: '*fp32', 49: '*fp32', 50: '*fp32', 51: '*fp32', 52: '*fp32', 53: '*fp32', 54: '*fp32', 55: '*fp32', 56: '*fp32', 57: '*fp32', 58: '*fp32', 59: '*fp32', 60: '*fp32', 61: '*fp32', 62: '*fp32', 63: '*fp32', 64: '*fp32', 65: '*fp32', 66: '*fp32', 67: '*fp32', 68: '*fp32', 69: '*fp32', 70: '*fp32', 71: '*fp32', 72: '*fp32', 73: '*fp32', 74: '*fp32', 75: '*fp32', 76: '*fp32', 77: '*fp32', 78: '*fp32', 79: '*fp32', 80: '*fp32', 81: '*fp32', 82: '*fp32', 83: '*fp32', 84: '*fp32', 85: '*fp32', 86: '*fp32', 87: '*fp32', 88: '*fp32', 89: '*fp32', 90: '*fp32', 91: '*fp32', 92: '*fp32', 93: '*fp32', 94: '*fp32', 95: '*fp32', 96: '*fp32', 97: '*fp32', 98: '*fp32', 99: '*fp32', 100: '*fp32', 101: '*fp32', 102: '*fp32', 103: '*fp32', 104: '*fp32', 105: '*fp32', 106: '*fp32', 107: '*fp32', 108: '*fp32', 109: '*fp32', 110: '*fp32', 111: '*fp32', 112: '*fp32', 113: '*fp32', 114: '*fp32', 115: '*fp32', 116: '*fp32', 117: '*fp32', 118: '*fp32', 119: '*fp32', 120: '*fp32', 121: '*fp32', 122: '*fp32', 123: '*fp32', 124: '*fp32', 125: '*fp32', 126: '*fp32', 127: '*fp32', 128: '*fp32', 129: '*fp32', 130: '*fp32', 131: '*fp32', 132: '*fp32', 133: '*fp32', 134: '*fp32', 135: '*fp32', 136: '*fp32', 137: '*fp32', 138: '*fp32', 139: '*fp32', 140: '*fp32', 141: '*fp32', 142: '*fp32', 143: '*fp32', 144: '*fp32', 145: '*fp32', 146: '*fp32', 147: '*fp32', 148: '*fp32', 149: '*fp32', 150: '*fp32', 151: '*fp32', 152: '*fp32', 153: '*fp32', 154: '*fp32', 155: '*fp32', 156: '*fp32', 157: '*fp32', 158: '*fp32', 159: '*fp32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'kernel_name': 'triton_for_fused_9', 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, in_ptr47, in_ptr48, in_ptr49, in_ptr50, in_ptr51, in_ptr52, in_ptr53, in_ptr54, in_ptr55, in_ptr56, in_ptr57, in_ptr58, in_ptr59, in_ptr60, in_ptr61, in_ptr62, in_ptr63, in_ptr64, in_ptr65, in_ptr66, in_ptr67, in_ptr68, in_ptr69, in_ptr70, in_ptr71, in_ptr72, in_ptr73, in_ptr74, in_ptr75, in_ptr76, in_ptr77, in_ptr78, in_ptr79, in_ptr80, in_ptr81, in_ptr82, in_ptr83, in_ptr84, in_ptr85, in_ptr86, in_ptr87, in_ptr88, in_ptr89, in_ptr90, in_ptr91, in_ptr92, in_ptr93, in_ptr94, in_ptr95, in_ptr96, in_ptr97, in_ptr98, in_ptr99, out_ptr0, out_ptr2, out_ptr3, out_ptr4, out_ptr6, out_ptr7, out_ptr8, out_ptr10, out_ptr11, out_ptr12, out_ptr14, out_ptr15, out_ptr16, out_ptr18, out_ptr19, out_ptr20, out_ptr22, out_ptr23, out_ptr24, out_ptr26, out_ptr27, out_ptr28, out_ptr30, out_ptr31, out_ptr32, out_ptr34, out_ptr35, out_ptr36, out_ptr38, out_ptr39, out_ptr40, out_ptr42, out_ptr43, out_ptr44, out_ptr46, out_ptr47, out_ptr48, out_ptr50, out_ptr51, out_ptr52, out_ptr54, out_ptr55, out_ptr56, out_ptr58, out_ptr59, out_ptr60, out_ptr62, out_ptr63, out_ptr64, out_ptr66, out_ptr67, out_ptr68, out_ptr70, out_ptr71, out_ptr72, out_ptr74, out_ptr75, out_ptr76, out_ptr78, out_ptr79):
    xpid = tl.program_id(0)
    XBLOCK: tl.constexpr = 1024
    if xpid >= 0 and xpid < 2:
        xpid_offset = xpid - 0
        xnumel = 2048
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
        tmp14 = libdevice.sqrt(tmp12)
        tmp17 = libdevice.pow(tmp7, tmp16)
        tmp18 = 1.0
        tmp19 = tmp17 - tmp18
        tmp20 = -tmp19
        tmp21 = libdevice.sqrt(tmp20)
        tmp22 = tmp14 / tmp21
        tmp23 = 1e-08
        tmp24 = tmp22 + tmp23
        tmp25 = 0.9
        tmp26 = libdevice.pow(tmp25, tmp16)
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
    elif xpid >= 2 and xpid < 1026:
        xpid_offset = xpid - 2
        xnumel = 1048576
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x1 = xindex
        tmp34 = tl.load(in_ptr5 + (x1), None)
        tmp35 = tl.load(in_ptr6 + (x1), None)
        tmp40 = tl.load(in_ptr7 + (x1), None)
        tmp47 = tl.load(in_ptr8 + (x1), None)
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
        tmp48 = libdevice.sqrt(tmp46)
        tmp51 = libdevice.pow(tmp41, tmp50)
        tmp52 = 1.0
        tmp53 = tmp51 - tmp52
        tmp54 = -tmp53
        tmp55 = libdevice.sqrt(tmp54)
        tmp56 = tmp48 / tmp55
        tmp57 = 1e-08
        tmp58 = tmp56 + tmp57
        tmp59 = 0.9
        tmp60 = libdevice.pow(tmp59, tmp50)
        tmp61 = tmp60 - tmp52
        tmp62 = 0.001
        tmp63 = tmp61 / tmp62
        tmp64 = 1 / tmp63
        tmp65 = tmp58 / tmp64
        tmp66 = tmp39 / tmp65
        tmp67 = tmp47 + tmp66
        tl.store(out_ptr4 + (x1), tmp39, None)
        tl.store(out_ptr6 + (x1), tmp67, None)
        tl.store(out_ptr7 + (x1), tmp46, None)
    elif xpid >= 1026 and xpid < 1027:
        xpid_offset = xpid - 1026
        xnumel = 512
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x2 = xindex
        tmp68 = tl.load(in_ptr10 + (x2), xmask)
        tmp69 = tl.load(in_ptr11 + (x2), xmask)
        tmp74 = tl.load(in_ptr12 + (x2), xmask)
        tmp81 = tl.load(in_ptr13 + (x2), xmask)
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
        tmp82 = libdevice.sqrt(tmp80)
        tmp85 = libdevice.pow(tmp75, tmp84)
        tmp86 = 1.0
        tmp87 = tmp85 - tmp86
        tmp88 = -tmp87
        tmp89 = libdevice.sqrt(tmp88)
        tmp90 = tmp82 / tmp89
        tmp91 = 1e-08
        tmp92 = tmp90 + tmp91
        tmp93 = 0.9
        tmp94 = libdevice.pow(tmp93, tmp84)
        tmp95 = tmp94 - tmp86
        tmp96 = 0.001
        tmp97 = tmp95 / tmp96
        tmp98 = 1 / tmp97
        tmp99 = tmp92 / tmp98
        tmp100 = tmp73 / tmp99
        tmp101 = tmp81 + tmp100
        tl.store(out_ptr8 + (x2), tmp73, xmask)
        tl.store(out_ptr10 + (x2), tmp101, xmask)
        tl.store(out_ptr11 + (x2), tmp80, xmask)
    elif xpid >= 1027 and xpid < 1028:
        xpid_offset = xpid - 1027
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
        tmp116 = libdevice.sqrt(tmp114)
        tmp119 = libdevice.pow(tmp109, tmp118)
        tmp120 = 1.0
        tmp121 = tmp119 - tmp120
        tmp122 = -tmp121
        tmp123 = libdevice.sqrt(tmp122)
        tmp124 = tmp116 / tmp123
        tmp125 = 1e-08
        tmp126 = tmp124 + tmp125
        tmp127 = 0.9
        tmp128 = libdevice.pow(tmp127, tmp118)
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
    elif xpid >= 1028 and xpid < 3332:
        xpid_offset = xpid - 1028
        xnumel = 2359296
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x4 = xindex
        tmp136 = tl.load(in_ptr20 + (x4), None)
        tmp137 = tl.load(in_ptr21 + (x4), None)
        tmp142 = tl.load(in_ptr22 + (x4), None)
        tmp149 = tl.load(in_ptr23 + (x4), None)
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
        tmp150 = libdevice.sqrt(tmp148)
        tmp153 = libdevice.pow(tmp143, tmp152)
        tmp154 = 1.0
        tmp155 = tmp153 - tmp154
        tmp156 = -tmp155
        tmp157 = libdevice.sqrt(tmp156)
        tmp158 = tmp150 / tmp157
        tmp159 = 1e-08
        tmp160 = tmp158 + tmp159
        tmp161 = 0.9
        tmp162 = libdevice.pow(tmp161, tmp152)
        tmp163 = tmp162 - tmp154
        tmp164 = 0.001
        tmp165 = tmp163 / tmp164
        tmp166 = 1 / tmp165
        tmp167 = tmp160 / tmp166
        tmp168 = tmp141 / tmp167
        tmp169 = tmp149 + tmp168
        tl.store(out_ptr16 + (x4), tmp141, None)
        tl.store(out_ptr18 + (x4), tmp169, None)
        tl.store(out_ptr19 + (x4), tmp148, None)
    elif xpid >= 3332 and xpid < 3333:
        xpid_offset = xpid - 3332
        xnumel = 512
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
        tmp184 = libdevice.sqrt(tmp182)
        tmp187 = libdevice.pow(tmp177, tmp186)
        tmp188 = 1.0
        tmp189 = tmp187 - tmp188
        tmp190 = -tmp189
        tmp191 = libdevice.sqrt(tmp190)
        tmp192 = tmp184 / tmp191
        tmp193 = 1e-08
        tmp194 = tmp192 + tmp193
        tmp195 = 0.9
        tmp196 = libdevice.pow(tmp195, tmp186)
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
    elif xpid >= 3333 and xpid < 3334:
        xpid_offset = xpid - 3333
        xnumel = 512
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x6 = xindex
        tmp204 = tl.load(in_ptr30 + (x6), xmask)
        tmp205 = tl.load(in_ptr31 + (x6), xmask)
        tmp210 = tl.load(in_ptr32 + (x6), xmask)
        tmp217 = tl.load(in_ptr33 + (x6), xmask)
        tmp219 = tl.load(in_ptr34 + (0))
        tmp220 = tl.broadcast_to(tmp219, [XBLOCK])
        tmp206 = tmp205 - tmp204
        tmp207 = 0.09999999999999998
        tmp208 = tmp206 * tmp207
        tmp209 = tmp204 + tmp208
        tmp211 = 0.999
        tmp212 = tmp210 * tmp211
        tmp213 = tmp205 * tmp205
        tmp214 = 0.0010000000000000009
        tmp215 = tmp213 * tmp214
        tmp216 = tmp212 + tmp215
        tmp218 = libdevice.sqrt(tmp216)
        tmp221 = libdevice.pow(tmp211, tmp220)
        tmp222 = 1.0
        tmp223 = tmp221 - tmp222
        tmp224 = -tmp223
        tmp225 = libdevice.sqrt(tmp224)
        tmp226 = tmp218 / tmp225
        tmp227 = 1e-08
        tmp228 = tmp226 + tmp227
        tmp229 = 0.9
        tmp230 = libdevice.pow(tmp229, tmp220)
        tmp231 = tmp230 - tmp222
        tmp232 = 0.001
        tmp233 = tmp231 / tmp232
        tmp234 = 1 / tmp233
        tmp235 = tmp228 / tmp234
        tmp236 = tmp209 / tmp235
        tmp237 = tmp217 + tmp236
        tl.store(out_ptr24 + (x6), tmp209, xmask)
        tl.store(out_ptr26 + (x6), tmp237, xmask)
        tl.store(out_ptr27 + (x6), tmp216, xmask)
    elif xpid >= 3334 and xpid < 4358:
        xpid_offset = xpid - 3334
        xnumel = 1048576
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x7 = xindex
        tmp238 = tl.load(in_ptr35 + (x7), None)
        tmp239 = tl.load(in_ptr36 + (x7), None)
        tmp244 = tl.load(in_ptr37 + (x7), None)
        tmp251 = tl.load(in_ptr38 + (x7), None)
        tmp253 = tl.load(in_ptr39 + (0))
        tmp254 = tl.broadcast_to(tmp253, [XBLOCK])
        tmp240 = tmp239 - tmp238
        tmp241 = 0.09999999999999998
        tmp242 = tmp240 * tmp241
        tmp243 = tmp238 + tmp242
        tmp245 = 0.999
        tmp246 = tmp244 * tmp245
        tmp247 = tmp239 * tmp239
        tmp248 = 0.0010000000000000009
        tmp249 = tmp247 * tmp248
        tmp250 = tmp246 + tmp249
        tmp252 = libdevice.sqrt(tmp250)
        tmp255 = libdevice.pow(tmp245, tmp254)
        tmp256 = 1.0
        tmp257 = tmp255 - tmp256
        tmp258 = -tmp257
        tmp259 = libdevice.sqrt(tmp258)
        tmp260 = tmp252 / tmp259
        tmp261 = 1e-08
        tmp262 = tmp260 + tmp261
        tmp263 = 0.9
        tmp264 = libdevice.pow(tmp263, tmp254)
        tmp265 = tmp264 - tmp256
        tmp266 = 0.001
        tmp267 = tmp265 / tmp266
        tmp268 = 1 / tmp267
        tmp269 = tmp262 / tmp268
        tmp270 = tmp243 / tmp269
        tmp271 = tmp251 + tmp270
        tl.store(out_ptr28 + (x7), tmp243, None)
        tl.store(out_ptr30 + (x7), tmp271, None)
        tl.store(out_ptr31 + (x7), tmp250, None)
    elif xpid >= 4358 and xpid < 4360:
        xpid_offset = xpid - 4358
        xnumel = 2048
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x8 = xindex
        tmp272 = tl.load(in_ptr40 + (x8), None)
        tmp273 = tl.load(in_ptr41 + (x8), None)
        tmp278 = tl.load(in_ptr42 + (x8), None)
        tmp285 = tl.load(in_ptr43 + (x8), None)
        tmp287 = tl.load(in_ptr44 + (0))
        tmp288 = tl.broadcast_to(tmp287, [XBLOCK])
        tmp274 = tmp273 - tmp272
        tmp275 = 0.09999999999999998
        tmp276 = tmp274 * tmp275
        tmp277 = tmp272 + tmp276
        tmp279 = 0.999
        tmp280 = tmp278 * tmp279
        tmp281 = tmp273 * tmp273
        tmp282 = 0.0010000000000000009
        tmp283 = tmp281 * tmp282
        tmp284 = tmp280 + tmp283
        tmp286 = libdevice.sqrt(tmp284)
        tmp289 = libdevice.pow(tmp279, tmp288)
        tmp290 = 1.0
        tmp291 = tmp289 - tmp290
        tmp292 = -tmp291
        tmp293 = libdevice.sqrt(tmp292)
        tmp294 = tmp286 / tmp293
        tmp295 = 1e-08
        tmp296 = tmp294 + tmp295
        tmp297 = 0.9
        tmp298 = libdevice.pow(tmp297, tmp288)
        tmp299 = tmp298 - tmp290
        tmp300 = 0.001
        tmp301 = tmp299 / tmp300
        tmp302 = 1 / tmp301
        tmp303 = tmp296 / tmp302
        tmp304 = tmp277 / tmp303
        tmp305 = tmp285 + tmp304
        tl.store(out_ptr32 + (x8), tmp277, None)
        tl.store(out_ptr34 + (x8), tmp305, None)
        tl.store(out_ptr35 + (x8), tmp284, None)
    elif xpid >= 4360 and xpid < 4362:
        xpid_offset = xpid - 4360
        xnumel = 2048
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x9 = xindex
        tmp306 = tl.load(in_ptr45 + (x9), None)
        tmp307 = tl.load(in_ptr46 + (x9), None)
        tmp312 = tl.load(in_ptr47 + (x9), None)
        tmp319 = tl.load(in_ptr48 + (x9), None)
        tmp321 = tl.load(in_ptr49 + (0))
        tmp322 = tl.broadcast_to(tmp321, [XBLOCK])
        tmp308 = tmp307 - tmp306
        tmp309 = 0.09999999999999998
        tmp310 = tmp308 * tmp309
        tmp311 = tmp306 + tmp310
        tmp313 = 0.999
        tmp314 = tmp312 * tmp313
        tmp315 = tmp307 * tmp307
        tmp316 = 0.0010000000000000009
        tmp317 = tmp315 * tmp316
        tmp318 = tmp314 + tmp317
        tmp320 = libdevice.sqrt(tmp318)
        tmp323 = libdevice.pow(tmp313, tmp322)
        tmp324 = 1.0
        tmp325 = tmp323 - tmp324
        tmp326 = -tmp325
        tmp327 = libdevice.sqrt(tmp326)
        tmp328 = tmp320 / tmp327
        tmp329 = 1e-08
        tmp330 = tmp328 + tmp329
        tmp331 = 0.9
        tmp332 = libdevice.pow(tmp331, tmp322)
        tmp333 = tmp332 - tmp324
        tmp334 = 0.001
        tmp335 = tmp333 / tmp334
        tmp336 = 1 / tmp335
        tmp337 = tmp330 / tmp336
        tmp338 = tmp311 / tmp337
        tmp339 = tmp319 + tmp338
        tl.store(out_ptr36 + (x9), tmp311, None)
        tl.store(out_ptr38 + (x9), tmp339, None)
        tl.store(out_ptr39 + (x9), tmp318, None)
    elif xpid >= 4362 and xpid < 5386:
        xpid_offset = xpid - 4362
        xnumel = 1048576
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x10 = xindex
        tmp340 = tl.load(in_ptr50 + (x10), None)
        tmp341 = tl.load(in_ptr51 + (x10), None)
        tmp346 = tl.load(in_ptr52 + (x10), None)
        tmp353 = tl.load(in_ptr53 + (x10), None)
        tmp355 = tl.load(in_ptr54 + (0))
        tmp356 = tl.broadcast_to(tmp355, [XBLOCK])
        tmp342 = tmp341 - tmp340
        tmp343 = 0.09999999999999998
        tmp344 = tmp342 * tmp343
        tmp345 = tmp340 + tmp344
        tmp347 = 0.999
        tmp348 = tmp346 * tmp347
        tmp349 = tmp341 * tmp341
        tmp350 = 0.0010000000000000009
        tmp351 = tmp349 * tmp350
        tmp352 = tmp348 + tmp351
        tmp354 = libdevice.sqrt(tmp352)
        tmp357 = libdevice.pow(tmp347, tmp356)
        tmp358 = 1.0
        tmp359 = tmp357 - tmp358
        tmp360 = -tmp359
        tmp361 = libdevice.sqrt(tmp360)
        tmp362 = tmp354 / tmp361
        tmp363 = 1e-08
        tmp364 = tmp362 + tmp363
        tmp365 = 0.9
        tmp366 = libdevice.pow(tmp365, tmp356)
        tmp367 = tmp366 - tmp358
        tmp368 = 0.001
        tmp369 = tmp367 / tmp368
        tmp370 = 1 / tmp369
        tmp371 = tmp364 / tmp370
        tmp372 = tmp345 / tmp371
        tmp373 = tmp353 + tmp372
        tl.store(out_ptr40 + (x10), tmp345, None)
        tl.store(out_ptr42 + (x10), tmp373, None)
        tl.store(out_ptr43 + (x10), tmp352, None)
    elif xpid >= 5386 and xpid < 5387:
        xpid_offset = xpid - 5386
        xnumel = 512
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x11 = xindex
        tmp374 = tl.load(in_ptr55 + (x11), xmask)
        tmp375 = tl.load(in_ptr56 + (x11), xmask)
        tmp380 = tl.load(in_ptr57 + (x11), xmask)
        tmp387 = tl.load(in_ptr58 + (x11), xmask)
        tmp389 = tl.load(in_ptr59 + (0))
        tmp390 = tl.broadcast_to(tmp389, [XBLOCK])
        tmp376 = tmp375 - tmp374
        tmp377 = 0.09999999999999998
        tmp378 = tmp376 * tmp377
        tmp379 = tmp374 + tmp378
        tmp381 = 0.999
        tmp382 = tmp380 * tmp381
        tmp383 = tmp375 * tmp375
        tmp384 = 0.0010000000000000009
        tmp385 = tmp383 * tmp384
        tmp386 = tmp382 + tmp385
        tmp388 = libdevice.sqrt(tmp386)
        tmp391 = libdevice.pow(tmp381, tmp390)
        tmp392 = 1.0
        tmp393 = tmp391 - tmp392
        tmp394 = -tmp393
        tmp395 = libdevice.sqrt(tmp394)
        tmp396 = tmp388 / tmp395
        tmp397 = 1e-08
        tmp398 = tmp396 + tmp397
        tmp399 = 0.9
        tmp400 = libdevice.pow(tmp399, tmp390)
        tmp401 = tmp400 - tmp392
        tmp402 = 0.001
        tmp403 = tmp401 / tmp402
        tmp404 = 1 / tmp403
        tmp405 = tmp398 / tmp404
        tmp406 = tmp379 / tmp405
        tmp407 = tmp387 + tmp406
        tl.store(out_ptr44 + (x11), tmp379, xmask)
        tl.store(out_ptr46 + (x11), tmp407, xmask)
        tl.store(out_ptr47 + (x11), tmp386, xmask)
    elif xpid >= 5387 and xpid < 5388:
        xpid_offset = xpid - 5387
        xnumel = 512
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x12 = xindex
        tmp408 = tl.load(in_ptr60 + (x12), xmask)
        tmp409 = tl.load(in_ptr61 + (x12), xmask)
        tmp414 = tl.load(in_ptr62 + (x12), xmask)
        tmp421 = tl.load(in_ptr63 + (x12), xmask)
        tmp423 = tl.load(in_ptr64 + (0))
        tmp424 = tl.broadcast_to(tmp423, [XBLOCK])
        tmp410 = tmp409 - tmp408
        tmp411 = 0.09999999999999998
        tmp412 = tmp410 * tmp411
        tmp413 = tmp408 + tmp412
        tmp415 = 0.999
        tmp416 = tmp414 * tmp415
        tmp417 = tmp409 * tmp409
        tmp418 = 0.0010000000000000009
        tmp419 = tmp417 * tmp418
        tmp420 = tmp416 + tmp419
        tmp422 = libdevice.sqrt(tmp420)
        tmp425 = libdevice.pow(tmp415, tmp424)
        tmp426 = 1.0
        tmp427 = tmp425 - tmp426
        tmp428 = -tmp427
        tmp429 = libdevice.sqrt(tmp428)
        tmp430 = tmp422 / tmp429
        tmp431 = 1e-08
        tmp432 = tmp430 + tmp431
        tmp433 = 0.9
        tmp434 = libdevice.pow(tmp433, tmp424)
        tmp435 = tmp434 - tmp426
        tmp436 = 0.001
        tmp437 = tmp435 / tmp436
        tmp438 = 1 / tmp437
        tmp439 = tmp432 / tmp438
        tmp440 = tmp413 / tmp439
        tmp441 = tmp421 + tmp440
        tl.store(out_ptr48 + (x12), tmp413, xmask)
        tl.store(out_ptr50 + (x12), tmp441, xmask)
        tl.store(out_ptr51 + (x12), tmp420, xmask)
    elif xpid >= 5388 and xpid < 7692:
        xpid_offset = xpid - 5388
        xnumel = 2359296
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x13 = xindex
        tmp442 = tl.load(in_ptr65 + (x13), None)
        tmp443 = tl.load(in_ptr66 + (x13), None)
        tmp448 = tl.load(in_ptr67 + (x13), None)
        tmp455 = tl.load(in_ptr68 + (x13), None)
        tmp457 = tl.load(in_ptr69 + (0))
        tmp458 = tl.broadcast_to(tmp457, [XBLOCK])
        tmp444 = tmp443 - tmp442
        tmp445 = 0.09999999999999998
        tmp446 = tmp444 * tmp445
        tmp447 = tmp442 + tmp446
        tmp449 = 0.999
        tmp450 = tmp448 * tmp449
        tmp451 = tmp443 * tmp443
        tmp452 = 0.0010000000000000009
        tmp453 = tmp451 * tmp452
        tmp454 = tmp450 + tmp453
        tmp456 = libdevice.sqrt(tmp454)
        tmp459 = libdevice.pow(tmp449, tmp458)
        tmp460 = 1.0
        tmp461 = tmp459 - tmp460
        tmp462 = -tmp461
        tmp463 = libdevice.sqrt(tmp462)
        tmp464 = tmp456 / tmp463
        tmp465 = 1e-08
        tmp466 = tmp464 + tmp465
        tmp467 = 0.9
        tmp468 = libdevice.pow(tmp467, tmp458)
        tmp469 = tmp468 - tmp460
        tmp470 = 0.001
        tmp471 = tmp469 / tmp470
        tmp472 = 1 / tmp471
        tmp473 = tmp466 / tmp472
        tmp474 = tmp447 / tmp473
        tmp475 = tmp455 + tmp474
        tl.store(out_ptr52 + (x13), tmp447, None)
        tl.store(out_ptr54 + (x13), tmp475, None)
        tl.store(out_ptr55 + (x13), tmp454, None)
    elif xpid >= 7692 and xpid < 7693:
        xpid_offset = xpid - 7692
        xnumel = 512
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x14 = xindex
        tmp476 = tl.load(in_ptr70 + (x14), xmask)
        tmp477 = tl.load(in_ptr71 + (x14), xmask)
        tmp482 = tl.load(in_ptr72 + (x14), xmask)
        tmp489 = tl.load(in_ptr73 + (x14), xmask)
        tmp491 = tl.load(in_ptr74 + (0))
        tmp492 = tl.broadcast_to(tmp491, [XBLOCK])
        tmp478 = tmp477 - tmp476
        tmp479 = 0.09999999999999998
        tmp480 = tmp478 * tmp479
        tmp481 = tmp476 + tmp480
        tmp483 = 0.999
        tmp484 = tmp482 * tmp483
        tmp485 = tmp477 * tmp477
        tmp486 = 0.0010000000000000009
        tmp487 = tmp485 * tmp486
        tmp488 = tmp484 + tmp487
        tmp490 = libdevice.sqrt(tmp488)
        tmp493 = libdevice.pow(tmp483, tmp492)
        tmp494 = 1.0
        tmp495 = tmp493 - tmp494
        tmp496 = -tmp495
        tmp497 = libdevice.sqrt(tmp496)
        tmp498 = tmp490 / tmp497
        tmp499 = 1e-08
        tmp500 = tmp498 + tmp499
        tmp501 = 0.9
        tmp502 = libdevice.pow(tmp501, tmp492)
        tmp503 = tmp502 - tmp494
        tmp504 = 0.001
        tmp505 = tmp503 / tmp504
        tmp506 = 1 / tmp505
        tmp507 = tmp500 / tmp506
        tmp508 = tmp481 / tmp507
        tmp509 = tmp489 + tmp508
        tl.store(out_ptr56 + (x14), tmp481, xmask)
        tl.store(out_ptr58 + (x14), tmp509, xmask)
        tl.store(out_ptr59 + (x14), tmp488, xmask)
    elif xpid >= 7693 and xpid < 7694:
        xpid_offset = xpid - 7693
        xnumel = 512
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x15 = xindex
        tmp510 = tl.load(in_ptr75 + (x15), xmask)
        tmp511 = tl.load(in_ptr76 + (x15), xmask)
        tmp516 = tl.load(in_ptr77 + (x15), xmask)
        tmp523 = tl.load(in_ptr78 + (x15), xmask)
        tmp525 = tl.load(in_ptr79 + (0))
        tmp526 = tl.broadcast_to(tmp525, [XBLOCK])
        tmp512 = tmp511 - tmp510
        tmp513 = 0.09999999999999998
        tmp514 = tmp512 * tmp513
        tmp515 = tmp510 + tmp514
        tmp517 = 0.999
        tmp518 = tmp516 * tmp517
        tmp519 = tmp511 * tmp511
        tmp520 = 0.0010000000000000009
        tmp521 = tmp519 * tmp520
        tmp522 = tmp518 + tmp521
        tmp524 = libdevice.sqrt(tmp522)
        tmp527 = libdevice.pow(tmp517, tmp526)
        tmp528 = 1.0
        tmp529 = tmp527 - tmp528
        tmp530 = -tmp529
        tmp531 = libdevice.sqrt(tmp530)
        tmp532 = tmp524 / tmp531
        tmp533 = 1e-08
        tmp534 = tmp532 + tmp533
        tmp535 = 0.9
        tmp536 = libdevice.pow(tmp535, tmp526)
        tmp537 = tmp536 - tmp528
        tmp538 = 0.001
        tmp539 = tmp537 / tmp538
        tmp540 = 1 / tmp539
        tmp541 = tmp534 / tmp540
        tmp542 = tmp515 / tmp541
        tmp543 = tmp523 + tmp542
        tl.store(out_ptr60 + (x15), tmp515, xmask)
        tl.store(out_ptr62 + (x15), tmp543, xmask)
        tl.store(out_ptr63 + (x15), tmp522, xmask)
    elif xpid >= 7694 and xpid < 8718:
        xpid_offset = xpid - 7694
        xnumel = 1048576
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x16 = xindex
        tmp544 = tl.load(in_ptr80 + (x16), None)
        tmp545 = tl.load(in_ptr81 + (x16), None)
        tmp550 = tl.load(in_ptr82 + (x16), None)
        tmp557 = tl.load(in_ptr83 + (x16), None)
        tmp559 = tl.load(in_ptr84 + (0))
        tmp560 = tl.broadcast_to(tmp559, [XBLOCK])
        tmp546 = tmp545 - tmp544
        tmp547 = 0.09999999999999998
        tmp548 = tmp546 * tmp547
        tmp549 = tmp544 + tmp548
        tmp551 = 0.999
        tmp552 = tmp550 * tmp551
        tmp553 = tmp545 * tmp545
        tmp554 = 0.0010000000000000009
        tmp555 = tmp553 * tmp554
        tmp556 = tmp552 + tmp555
        tmp558 = libdevice.sqrt(tmp556)
        tmp561 = libdevice.pow(tmp551, tmp560)
        tmp562 = 1.0
        tmp563 = tmp561 - tmp562
        tmp564 = -tmp563
        tmp565 = libdevice.sqrt(tmp564)
        tmp566 = tmp558 / tmp565
        tmp567 = 1e-08
        tmp568 = tmp566 + tmp567
        tmp569 = 0.9
        tmp570 = libdevice.pow(tmp569, tmp560)
        tmp571 = tmp570 - tmp562
        tmp572 = 0.001
        tmp573 = tmp571 / tmp572
        tmp574 = 1 / tmp573
        tmp575 = tmp568 / tmp574
        tmp576 = tmp549 / tmp575
        tmp577 = tmp557 + tmp576
        tl.store(out_ptr64 + (x16), tmp549, None)
        tl.store(out_ptr66 + (x16), tmp577, None)
        tl.store(out_ptr67 + (x16), tmp556, None)
    elif xpid >= 8718 and xpid < 8720:
        xpid_offset = xpid - 8718
        xnumel = 2048
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x17 = xindex
        tmp578 = tl.load(in_ptr85 + (x17), None)
        tmp579 = tl.load(in_ptr86 + (x17), None)
        tmp584 = tl.load(in_ptr87 + (x17), None)
        tmp591 = tl.load(in_ptr88 + (x17), None)
        tmp593 = tl.load(in_ptr89 + (0))
        tmp594 = tl.broadcast_to(tmp593, [XBLOCK])
        tmp580 = tmp579 - tmp578
        tmp581 = 0.09999999999999998
        tmp582 = tmp580 * tmp581
        tmp583 = tmp578 + tmp582
        tmp585 = 0.999
        tmp586 = tmp584 * tmp585
        tmp587 = tmp579 * tmp579
        tmp588 = 0.0010000000000000009
        tmp589 = tmp587 * tmp588
        tmp590 = tmp586 + tmp589
        tmp592 = libdevice.sqrt(tmp590)
        tmp595 = libdevice.pow(tmp585, tmp594)
        tmp596 = 1.0
        tmp597 = tmp595 - tmp596
        tmp598 = -tmp597
        tmp599 = libdevice.sqrt(tmp598)
        tmp600 = tmp592 / tmp599
        tmp601 = 1e-08
        tmp602 = tmp600 + tmp601
        tmp603 = 0.9
        tmp604 = libdevice.pow(tmp603, tmp594)
        tmp605 = tmp604 - tmp596
        tmp606 = 0.001
        tmp607 = tmp605 / tmp606
        tmp608 = 1 / tmp607
        tmp609 = tmp602 / tmp608
        tmp610 = tmp583 / tmp609
        tmp611 = tmp591 + tmp610
        tl.store(out_ptr68 + (x17), tmp583, None)
        tl.store(out_ptr70 + (x17), tmp611, None)
        tl.store(out_ptr71 + (x17), tmp590, None)
    elif xpid >= 8720 and xpid < 8722:
        xpid_offset = xpid - 8720
        xnumel = 2048
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x18 = xindex
        tmp612 = tl.load(in_ptr90 + (x18), None)
        tmp613 = tl.load(in_ptr91 + (x18), None)
        tmp618 = tl.load(in_ptr92 + (x18), None)
        tmp625 = tl.load(in_ptr93 + (x18), None)
        tmp627 = tl.load(in_ptr94 + (0))
        tmp628 = tl.broadcast_to(tmp627, [XBLOCK])
        tmp614 = tmp613 - tmp612
        tmp615 = 0.09999999999999998
        tmp616 = tmp614 * tmp615
        tmp617 = tmp612 + tmp616
        tmp619 = 0.999
        tmp620 = tmp618 * tmp619
        tmp621 = tmp613 * tmp613
        tmp622 = 0.0010000000000000009
        tmp623 = tmp621 * tmp622
        tmp624 = tmp620 + tmp623
        tmp626 = libdevice.sqrt(tmp624)
        tmp629 = libdevice.pow(tmp619, tmp628)
        tmp630 = 1.0
        tmp631 = tmp629 - tmp630
        tmp632 = -tmp631
        tmp633 = libdevice.sqrt(tmp632)
        tmp634 = tmp626 / tmp633
        tmp635 = 1e-08
        tmp636 = tmp634 + tmp635
        tmp637 = 0.9
        tmp638 = libdevice.pow(tmp637, tmp628)
        tmp639 = tmp638 - tmp630
        tmp640 = 0.001
        tmp641 = tmp639 / tmp640
        tmp642 = 1 / tmp641
        tmp643 = tmp636 / tmp642
        tmp644 = tmp617 / tmp643
        tmp645 = tmp625 + tmp644
        tl.store(out_ptr72 + (x18), tmp617, None)
        tl.store(out_ptr74 + (x18), tmp645, None)
        tl.store(out_ptr75 + (x18), tmp624, None)
    elif xpid >= 8722 and xpid < 10722:
        xpid_offset = xpid - 8722
        xnumel = 2048000
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x19 = xindex
        tmp646 = tl.load(in_ptr95 + (x19), None)
        tmp647 = tl.load(in_ptr96 + (x19), None)
        tmp652 = tl.load(in_ptr97 + (x19), None)
        tmp659 = tl.load(in_ptr98 + (x19), None)
        tmp661 = tl.load(in_ptr99 + (0))
        tmp662 = tl.broadcast_to(tmp661, [XBLOCK])
        tmp648 = tmp647 - tmp646
        tmp649 = 0.09999999999999998
        tmp650 = tmp648 * tmp649
        tmp651 = tmp646 + tmp650
        tmp653 = 0.999
        tmp654 = tmp652 * tmp653
        tmp655 = tmp647 * tmp647
        tmp656 = 0.0010000000000000009
        tmp657 = tmp655 * tmp656
        tmp658 = tmp654 + tmp657
        tmp660 = libdevice.sqrt(tmp658)
        tmp663 = libdevice.pow(tmp653, tmp662)
        tmp664 = 1.0
        tmp665 = tmp663 - tmp664
        tmp666 = -tmp665
        tmp667 = libdevice.sqrt(tmp666)
        tmp668 = tmp660 / tmp667
        tmp669 = 1e-08
        tmp670 = tmp668 + tmp669
        tmp671 = 0.9
        tmp672 = libdevice.pow(tmp671, tmp662)
        tmp673 = tmp672 - tmp664
        tmp674 = 0.001
        tmp675 = tmp673 / tmp674
        tmp676 = 1 / tmp675
        tmp677 = tmp670 / tmp676
        tmp678 = tmp651 / tmp677
        tmp679 = tmp659 + tmp678
        tl.store(out_ptr76 + (x19), tmp651, None)
        tl.store(out_ptr78 + (x19), tmp679, None)
        tl.store(out_ptr79 + (x19), tmp658, None)
    else:
        pass
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/kc/ckcwvjy4dcdqubcfyto7ezznmvfngv7bqf2uo4hglhrojetremba.py
# Source Nodes: [], Original ATen: []

triton_for_fused_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.foreach(
    num_warps=8,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'kernel_name': 'triton_for_fused_10', 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr2, out_ptr3):
    xpid = tl.program_id(0)
    XBLOCK: tl.constexpr = 1024
    if xpid >= 0 and xpid < 1:
        xpid_offset = xpid - 0
        xnumel = 1000
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), xmask)
        tmp1 = tl.load(in_ptr1 + (x0), xmask)
        tmp6 = tl.load(in_ptr2 + (x0), xmask)
        tmp13 = tl.load(in_ptr3 + (x0), xmask)
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
        tmp14 = libdevice.sqrt(tmp12)
        tmp17 = libdevice.pow(tmp7, tmp16)
        tmp18 = 1.0
        tmp19 = tmp17 - tmp18
        tmp20 = -tmp19
        tmp21 = libdevice.sqrt(tmp20)
        tmp22 = tmp14 / tmp21
        tmp23 = 1e-08
        tmp24 = tmp22 + tmp23
        tmp25 = 0.9
        tmp26 = libdevice.pow(tmp25, tmp16)
        tmp27 = tmp26 - tmp18
        tmp28 = 0.001
        tmp29 = tmp27 / tmp28
        tmp30 = 1 / tmp29
        tmp31 = tmp24 / tmp30
        tmp32 = tmp5 / tmp31
        tmp33 = tmp13 + tmp32
        tl.store(out_ptr0 + (x0), tmp5, xmask)
        tl.store(out_ptr2 + (x0), tmp33, xmask)
        tl.store(out_ptr3 + (x0), tmp12, xmask)
    else:
        pass
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1, arg728_1, arg729_1, arg730_1, arg731_1, arg732_1, arg733_1, arg734_1, arg735_1, arg736_1, arg737_1, arg738_1, arg739_1, arg740_1, arg741_1, arg742_1, arg743_1, arg744_1, arg745_1, arg746_1, arg747_1, arg748_1, arg749_1, arg750_1, arg751_1, arg752_1, arg753_1, arg754_1, arg755_1, arg756_1, arg757_1, arg758_1, arg759_1, arg760_1, arg761_1, arg762_1, arg763_1, arg764_1, arg765_1, arg766_1, arg767_1, arg768_1, arg769_1, arg770_1, arg771_1, arg772_1, arg773_1, arg774_1, arg775_1, arg776_1, arg777_1, arg778_1, arg779_1, arg780_1, arg781_1, arg782_1, arg783_1, arg784_1, arg785_1, arg786_1, arg787_1, arg788_1, arg789_1, arg790_1, arg791_1, arg792_1, arg793_1, arg794_1, arg795_1, arg796_1, arg797_1, arg798_1, arg799_1, arg800_1, arg801_1, arg802_1, arg803_1, arg804_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(arg1_1, (64, ), (1, ))
    assert_size_stride(arg2_1, (64, ), (1, ))
    assert_size_stride(arg3_1, (64, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(arg4_1, (64, ), (1, ))
    assert_size_stride(arg5_1, (64, ), (1, ))
    assert_size_stride(arg6_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg7_1, (64, ), (1, ))
    assert_size_stride(arg8_1, (64, ), (1, ))
    assert_size_stride(arg9_1, (256, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(arg10_1, (256, ), (1, ))
    assert_size_stride(arg11_1, (256, ), (1, ))
    assert_size_stride(arg12_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg13_1, (256, ), (1, ))
    assert_size_stride(arg14_1, (256, ), (1, ))
    assert_size_stride(arg15_1, (64, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg16_1, (64, ), (1, ))
    assert_size_stride(arg17_1, (64, ), (1, ))
    assert_size_stride(arg18_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg19_1, (64, ), (1, ))
    assert_size_stride(arg20_1, (64, ), (1, ))
    assert_size_stride(arg21_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg22_1, (256, ), (1, ))
    assert_size_stride(arg23_1, (256, ), (1, ))
    assert_size_stride(arg24_1, (64, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg25_1, (64, ), (1, ))
    assert_size_stride(arg26_1, (64, ), (1, ))
    assert_size_stride(arg27_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg28_1, (64, ), (1, ))
    assert_size_stride(arg29_1, (64, ), (1, ))
    assert_size_stride(arg30_1, (256, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(arg31_1, (256, ), (1, ))
    assert_size_stride(arg32_1, (256, ), (1, ))
    assert_size_stride(arg33_1, (128, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg34_1, (128, ), (1, ))
    assert_size_stride(arg35_1, (128, ), (1, ))
    assert_size_stride(arg36_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg37_1, (128, ), (1, ))
    assert_size_stride(arg38_1, (128, ), (1, ))
    assert_size_stride(arg39_1, (512, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(arg40_1, (512, ), (1, ))
    assert_size_stride(arg41_1, (512, ), (1, ))
    assert_size_stride(arg42_1, (512, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg43_1, (512, ), (1, ))
    assert_size_stride(arg44_1, (512, ), (1, ))
    assert_size_stride(arg45_1, (128, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg46_1, (128, ), (1, ))
    assert_size_stride(arg47_1, (128, ), (1, ))
    assert_size_stride(arg48_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg49_1, (128, ), (1, ))
    assert_size_stride(arg50_1, (128, ), (1, ))
    assert_size_stride(arg51_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg52_1, (512, ), (1, ))
    assert_size_stride(arg53_1, (512, ), (1, ))
    assert_size_stride(arg54_1, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg55_1, (128, ), (1, ))
    assert_size_stride(arg56_1, (128, ), (1, ))
    assert_size_stride(arg57_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg58_1, (128, ), (1, ))
    assert_size_stride(arg59_1, (128, ), (1, ))
    assert_size_stride(arg60_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg61_1, (512, ), (1, ))
    assert_size_stride(arg62_1, (512, ), (1, ))
    assert_size_stride(arg63_1, (128, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg64_1, (128, ), (1, ))
    assert_size_stride(arg65_1, (128, ), (1, ))
    assert_size_stride(arg66_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg67_1, (128, ), (1, ))
    assert_size_stride(arg68_1, (128, ), (1, ))
    assert_size_stride(arg69_1, (512, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(arg70_1, (512, ), (1, ))
    assert_size_stride(arg71_1, (512, ), (1, ))
    assert_size_stride(arg72_1, (256, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg73_1, (256, ), (1, ))
    assert_size_stride(arg74_1, (256, ), (1, ))
    assert_size_stride(arg75_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg76_1, (256, ), (1, ))
    assert_size_stride(arg77_1, (256, ), (1, ))
    assert_size_stride(arg78_1, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg79_1, (1024, ), (1, ))
    assert_size_stride(arg80_1, (1024, ), (1, ))
    assert_size_stride(arg81_1, (1024, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg82_1, (1024, ), (1, ))
    assert_size_stride(arg83_1, (1024, ), (1, ))
    assert_size_stride(arg84_1, (256, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(arg85_1, (256, ), (1, ))
    assert_size_stride(arg86_1, (256, ), (1, ))
    assert_size_stride(arg87_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg88_1, (256, ), (1, ))
    assert_size_stride(arg89_1, (256, ), (1, ))
    assert_size_stride(arg90_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg91_1, (1024, ), (1, ))
    assert_size_stride(arg92_1, (1024, ), (1, ))
    assert_size_stride(arg93_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg94_1, (256, ), (1, ))
    assert_size_stride(arg95_1, (256, ), (1, ))
    assert_size_stride(arg96_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg97_1, (256, ), (1, ))
    assert_size_stride(arg98_1, (256, ), (1, ))
    assert_size_stride(arg99_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg100_1, (1024, ), (1, ))
    assert_size_stride(arg101_1, (1024, ), (1, ))
    assert_size_stride(arg102_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg103_1, (256, ), (1, ))
    assert_size_stride(arg104_1, (256, ), (1, ))
    assert_size_stride(arg105_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg106_1, (256, ), (1, ))
    assert_size_stride(arg107_1, (256, ), (1, ))
    assert_size_stride(arg108_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg109_1, (1024, ), (1, ))
    assert_size_stride(arg110_1, (1024, ), (1, ))
    assert_size_stride(arg111_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg112_1, (256, ), (1, ))
    assert_size_stride(arg113_1, (256, ), (1, ))
    assert_size_stride(arg114_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg115_1, (256, ), (1, ))
    assert_size_stride(arg116_1, (256, ), (1, ))
    assert_size_stride(arg117_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg118_1, (1024, ), (1, ))
    assert_size_stride(arg119_1, (1024, ), (1, ))
    assert_size_stride(arg120_1, (256, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(arg121_1, (256, ), (1, ))
    assert_size_stride(arg122_1, (256, ), (1, ))
    assert_size_stride(arg123_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg124_1, (256, ), (1, ))
    assert_size_stride(arg125_1, (256, ), (1, ))
    assert_size_stride(arg126_1, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg127_1, (1024, ), (1, ))
    assert_size_stride(arg128_1, (1024, ), (1, ))
    assert_size_stride(arg129_1, (512, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(arg130_1, (512, ), (1, ))
    assert_size_stride(arg131_1, (512, ), (1, ))
    assert_size_stride(arg132_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg133_1, (512, ), (1, ))
    assert_size_stride(arg134_1, (512, ), (1, ))
    assert_size_stride(arg135_1, (2048, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg136_1, (2048, ), (1, ))
    assert_size_stride(arg137_1, (2048, ), (1, ))
    assert_size_stride(arg138_1, (2048, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(arg139_1, (2048, ), (1, ))
    assert_size_stride(arg140_1, (2048, ), (1, ))
    assert_size_stride(arg141_1, (512, 2048, 1, 1), (2048, 1, 2048, 2048))
    assert_size_stride(arg142_1, (512, ), (1, ))
    assert_size_stride(arg143_1, (512, ), (1, ))
    assert_size_stride(arg144_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg145_1, (512, ), (1, ))
    assert_size_stride(arg146_1, (512, ), (1, ))
    assert_size_stride(arg147_1, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg148_1, (2048, ), (1, ))
    assert_size_stride(arg149_1, (2048, ), (1, ))
    assert_size_stride(arg150_1, (512, 2048, 1, 1), (2048, 1, 2048, 2048))
    assert_size_stride(arg151_1, (512, ), (1, ))
    assert_size_stride(arg152_1, (512, ), (1, ))
    assert_size_stride(arg153_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg154_1, (512, ), (1, ))
    assert_size_stride(arg155_1, (512, ), (1, ))
    assert_size_stride(arg156_1, (2048, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg157_1, (2048, ), (1, ))
    assert_size_stride(arg158_1, (2048, ), (1, ))
    assert_size_stride(arg159_1, (1000, 2048), (2048, 1))
    assert_size_stride(arg160_1, (1000, ), (1, ))
    assert_size_stride(arg161_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(arg162_1, (64, ), (1, ))
    assert_size_stride(arg163_1, (64, ), (1, ))
    assert_size_stride(arg164_1, (64, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(arg165_1, (64, ), (1, ))
    assert_size_stride(arg166_1, (64, ), (1, ))
    assert_size_stride(arg167_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg168_1, (64, ), (1, ))
    assert_size_stride(arg169_1, (64, ), (1, ))
    assert_size_stride(arg170_1, (256, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(arg171_1, (256, ), (1, ))
    assert_size_stride(arg172_1, (256, ), (1, ))
    assert_size_stride(arg173_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg174_1, (256, ), (1, ))
    assert_size_stride(arg175_1, (256, ), (1, ))
    assert_size_stride(arg176_1, (64, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg177_1, (64, ), (1, ))
    assert_size_stride(arg178_1, (64, ), (1, ))
    assert_size_stride(arg179_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg180_1, (64, ), (1, ))
    assert_size_stride(arg181_1, (64, ), (1, ))
    assert_size_stride(arg182_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg183_1, (256, ), (1, ))
    assert_size_stride(arg184_1, (256, ), (1, ))
    assert_size_stride(arg185_1, (64, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg186_1, (64, ), (1, ))
    assert_size_stride(arg187_1, (64, ), (1, ))
    assert_size_stride(arg188_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg189_1, (64, ), (1, ))
    assert_size_stride(arg190_1, (64, ), (1, ))
    assert_size_stride(arg191_1, (256, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(arg192_1, (256, ), (1, ))
    assert_size_stride(arg193_1, (256, ), (1, ))
    assert_size_stride(arg194_1, (128, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg195_1, (128, ), (1, ))
    assert_size_stride(arg196_1, (128, ), (1, ))
    assert_size_stride(arg197_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg198_1, (128, ), (1, ))
    assert_size_stride(arg199_1, (128, ), (1, ))
    assert_size_stride(arg200_1, (512, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(arg201_1, (512, ), (1, ))
    assert_size_stride(arg202_1, (512, ), (1, ))
    assert_size_stride(arg203_1, (512, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg204_1, (512, ), (1, ))
    assert_size_stride(arg205_1, (512, ), (1, ))
    assert_size_stride(arg206_1, (128, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg207_1, (128, ), (1, ))
    assert_size_stride(arg208_1, (128, ), (1, ))
    assert_size_stride(arg209_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg210_1, (128, ), (1, ))
    assert_size_stride(arg211_1, (128, ), (1, ))
    assert_size_stride(arg212_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg213_1, (512, ), (1, ))
    assert_size_stride(arg214_1, (512, ), (1, ))
    assert_size_stride(arg215_1, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg216_1, (128, ), (1, ))
    assert_size_stride(arg217_1, (128, ), (1, ))
    assert_size_stride(arg218_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg219_1, (128, ), (1, ))
    assert_size_stride(arg220_1, (128, ), (1, ))
    assert_size_stride(arg221_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg222_1, (512, ), (1, ))
    assert_size_stride(arg223_1, (512, ), (1, ))
    assert_size_stride(arg224_1, (128, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg225_1, (128, ), (1, ))
    assert_size_stride(arg226_1, (128, ), (1, ))
    assert_size_stride(arg227_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg228_1, (128, ), (1, ))
    assert_size_stride(arg229_1, (128, ), (1, ))
    assert_size_stride(arg230_1, (512, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(arg231_1, (512, ), (1, ))
    assert_size_stride(arg232_1, (512, ), (1, ))
    assert_size_stride(arg233_1, (256, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg234_1, (256, ), (1, ))
    assert_size_stride(arg235_1, (256, ), (1, ))
    assert_size_stride(arg236_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg237_1, (256, ), (1, ))
    assert_size_stride(arg238_1, (256, ), (1, ))
    assert_size_stride(arg239_1, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg240_1, (1024, ), (1, ))
    assert_size_stride(arg241_1, (1024, ), (1, ))
    assert_size_stride(arg242_1, (1024, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg243_1, (1024, ), (1, ))
    assert_size_stride(arg244_1, (1024, ), (1, ))
    assert_size_stride(arg245_1, (256, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(arg246_1, (256, ), (1, ))
    assert_size_stride(arg247_1, (256, ), (1, ))
    assert_size_stride(arg248_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg249_1, (256, ), (1, ))
    assert_size_stride(arg250_1, (256, ), (1, ))
    assert_size_stride(arg251_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg252_1, (1024, ), (1, ))
    assert_size_stride(arg253_1, (1024, ), (1, ))
    assert_size_stride(arg254_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg255_1, (256, ), (1, ))
    assert_size_stride(arg256_1, (256, ), (1, ))
    assert_size_stride(arg257_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg258_1, (256, ), (1, ))
    assert_size_stride(arg259_1, (256, ), (1, ))
    assert_size_stride(arg260_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg261_1, (1024, ), (1, ))
    assert_size_stride(arg262_1, (1024, ), (1, ))
    assert_size_stride(arg263_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg264_1, (256, ), (1, ))
    assert_size_stride(arg265_1, (256, ), (1, ))
    assert_size_stride(arg266_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg267_1, (256, ), (1, ))
    assert_size_stride(arg268_1, (256, ), (1, ))
    assert_size_stride(arg269_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg270_1, (1024, ), (1, ))
    assert_size_stride(arg271_1, (1024, ), (1, ))
    assert_size_stride(arg272_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg273_1, (256, ), (1, ))
    assert_size_stride(arg274_1, (256, ), (1, ))
    assert_size_stride(arg275_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg276_1, (256, ), (1, ))
    assert_size_stride(arg277_1, (256, ), (1, ))
    assert_size_stride(arg278_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg279_1, (1024, ), (1, ))
    assert_size_stride(arg280_1, (1024, ), (1, ))
    assert_size_stride(arg281_1, (256, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(arg282_1, (256, ), (1, ))
    assert_size_stride(arg283_1, (256, ), (1, ))
    assert_size_stride(arg284_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg285_1, (256, ), (1, ))
    assert_size_stride(arg286_1, (256, ), (1, ))
    assert_size_stride(arg287_1, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg288_1, (1024, ), (1, ))
    assert_size_stride(arg289_1, (1024, ), (1, ))
    assert_size_stride(arg290_1, (512, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(arg291_1, (512, ), (1, ))
    assert_size_stride(arg292_1, (512, ), (1, ))
    assert_size_stride(arg293_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg294_1, (512, ), (1, ))
    assert_size_stride(arg295_1, (512, ), (1, ))
    assert_size_stride(arg296_1, (2048, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg297_1, (2048, ), (1, ))
    assert_size_stride(arg298_1, (2048, ), (1, ))
    assert_size_stride(arg299_1, (2048, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(arg300_1, (2048, ), (1, ))
    assert_size_stride(arg301_1, (2048, ), (1, ))
    assert_size_stride(arg302_1, (512, 2048, 1, 1), (2048, 1, 2048, 2048))
    assert_size_stride(arg303_1, (512, ), (1, ))
    assert_size_stride(arg304_1, (512, ), (1, ))
    assert_size_stride(arg305_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg306_1, (512, ), (1, ))
    assert_size_stride(arg307_1, (512, ), (1, ))
    assert_size_stride(arg308_1, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg309_1, (2048, ), (1, ))
    assert_size_stride(arg310_1, (2048, ), (1, ))
    assert_size_stride(arg311_1, (512, 2048, 1, 1), (2048, 1, 2048, 2048))
    assert_size_stride(arg312_1, (512, ), (1, ))
    assert_size_stride(arg313_1, (512, ), (1, ))
    assert_size_stride(arg314_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg315_1, (512, ), (1, ))
    assert_size_stride(arg316_1, (512, ), (1, ))
    assert_size_stride(arg317_1, (2048, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg318_1, (2048, ), (1, ))
    assert_size_stride(arg319_1, (2048, ), (1, ))
    assert_size_stride(arg320_1, (1000, 2048), (2048, 1))
    assert_size_stride(arg321_1, (1000, ), (1, ))
    assert_size_stride(arg322_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(arg323_1, (64, ), (1, ))
    assert_size_stride(arg324_1, (64, ), (1, ))
    assert_size_stride(arg325_1, (64, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(arg326_1, (64, ), (1, ))
    assert_size_stride(arg327_1, (64, ), (1, ))
    assert_size_stride(arg328_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg329_1, (64, ), (1, ))
    assert_size_stride(arg330_1, (64, ), (1, ))
    assert_size_stride(arg331_1, (256, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(arg332_1, (256, ), (1, ))
    assert_size_stride(arg333_1, (256, ), (1, ))
    assert_size_stride(arg334_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg335_1, (256, ), (1, ))
    assert_size_stride(arg336_1, (256, ), (1, ))
    assert_size_stride(arg337_1, (64, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg338_1, (64, ), (1, ))
    assert_size_stride(arg339_1, (64, ), (1, ))
    assert_size_stride(arg340_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg341_1, (64, ), (1, ))
    assert_size_stride(arg342_1, (64, ), (1, ))
    assert_size_stride(arg343_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg344_1, (256, ), (1, ))
    assert_size_stride(arg345_1, (256, ), (1, ))
    assert_size_stride(arg346_1, (64, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg347_1, (64, ), (1, ))
    assert_size_stride(arg348_1, (64, ), (1, ))
    assert_size_stride(arg349_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg350_1, (64, ), (1, ))
    assert_size_stride(arg351_1, (64, ), (1, ))
    assert_size_stride(arg352_1, (256, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(arg353_1, (256, ), (1, ))
    assert_size_stride(arg354_1, (256, ), (1, ))
    assert_size_stride(arg355_1, (128, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg356_1, (128, ), (1, ))
    assert_size_stride(arg357_1, (128, ), (1, ))
    assert_size_stride(arg358_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg359_1, (128, ), (1, ))
    assert_size_stride(arg360_1, (128, ), (1, ))
    assert_size_stride(arg361_1, (512, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(arg362_1, (512, ), (1, ))
    assert_size_stride(arg363_1, (512, ), (1, ))
    assert_size_stride(arg364_1, (512, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg365_1, (512, ), (1, ))
    assert_size_stride(arg366_1, (512, ), (1, ))
    assert_size_stride(arg367_1, (128, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg368_1, (128, ), (1, ))
    assert_size_stride(arg369_1, (128, ), (1, ))
    assert_size_stride(arg370_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg371_1, (128, ), (1, ))
    assert_size_stride(arg372_1, (128, ), (1, ))
    assert_size_stride(arg373_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg374_1, (512, ), (1, ))
    assert_size_stride(arg375_1, (512, ), (1, ))
    assert_size_stride(arg376_1, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg377_1, (128, ), (1, ))
    assert_size_stride(arg378_1, (128, ), (1, ))
    assert_size_stride(arg379_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg380_1, (128, ), (1, ))
    assert_size_stride(arg381_1, (128, ), (1, ))
    assert_size_stride(arg382_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg383_1, (512, ), (1, ))
    assert_size_stride(arg384_1, (512, ), (1, ))
    assert_size_stride(arg385_1, (128, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg386_1, (128, ), (1, ))
    assert_size_stride(arg387_1, (128, ), (1, ))
    assert_size_stride(arg388_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg389_1, (128, ), (1, ))
    assert_size_stride(arg390_1, (128, ), (1, ))
    assert_size_stride(arg391_1, (512, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(arg392_1, (512, ), (1, ))
    assert_size_stride(arg393_1, (512, ), (1, ))
    assert_size_stride(arg394_1, (256, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg395_1, (256, ), (1, ))
    assert_size_stride(arg396_1, (256, ), (1, ))
    assert_size_stride(arg397_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg398_1, (256, ), (1, ))
    assert_size_stride(arg399_1, (256, ), (1, ))
    assert_size_stride(arg400_1, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg401_1, (1024, ), (1, ))
    assert_size_stride(arg402_1, (1024, ), (1, ))
    assert_size_stride(arg403_1, (1024, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg404_1, (1024, ), (1, ))
    assert_size_stride(arg405_1, (1024, ), (1, ))
    assert_size_stride(arg406_1, (256, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(arg407_1, (256, ), (1, ))
    assert_size_stride(arg408_1, (256, ), (1, ))
    assert_size_stride(arg409_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg410_1, (256, ), (1, ))
    assert_size_stride(arg411_1, (256, ), (1, ))
    assert_size_stride(arg412_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg413_1, (1024, ), (1, ))
    assert_size_stride(arg414_1, (1024, ), (1, ))
    assert_size_stride(arg415_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg416_1, (256, ), (1, ))
    assert_size_stride(arg417_1, (256, ), (1, ))
    assert_size_stride(arg418_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg419_1, (256, ), (1, ))
    assert_size_stride(arg420_1, (256, ), (1, ))
    assert_size_stride(arg421_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg422_1, (1024, ), (1, ))
    assert_size_stride(arg423_1, (1024, ), (1, ))
    assert_size_stride(arg424_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg425_1, (256, ), (1, ))
    assert_size_stride(arg426_1, (256, ), (1, ))
    assert_size_stride(arg427_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg428_1, (256, ), (1, ))
    assert_size_stride(arg429_1, (256, ), (1, ))
    assert_size_stride(arg430_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg431_1, (1024, ), (1, ))
    assert_size_stride(arg432_1, (1024, ), (1, ))
    assert_size_stride(arg433_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg434_1, (256, ), (1, ))
    assert_size_stride(arg435_1, (256, ), (1, ))
    assert_size_stride(arg436_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg437_1, (256, ), (1, ))
    assert_size_stride(arg438_1, (256, ), (1, ))
    assert_size_stride(arg439_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg440_1, (1024, ), (1, ))
    assert_size_stride(arg441_1, (1024, ), (1, ))
    assert_size_stride(arg442_1, (256, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(arg443_1, (256, ), (1, ))
    assert_size_stride(arg444_1, (256, ), (1, ))
    assert_size_stride(arg445_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg446_1, (256, ), (1, ))
    assert_size_stride(arg447_1, (256, ), (1, ))
    assert_size_stride(arg448_1, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg449_1, (1024, ), (1, ))
    assert_size_stride(arg450_1, (1024, ), (1, ))
    assert_size_stride(arg451_1, (512, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(arg452_1, (512, ), (1, ))
    assert_size_stride(arg453_1, (512, ), (1, ))
    assert_size_stride(arg454_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg455_1, (512, ), (1, ))
    assert_size_stride(arg456_1, (512, ), (1, ))
    assert_size_stride(arg457_1, (2048, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg458_1, (2048, ), (1, ))
    assert_size_stride(arg459_1, (2048, ), (1, ))
    assert_size_stride(arg460_1, (2048, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(arg461_1, (2048, ), (1, ))
    assert_size_stride(arg462_1, (2048, ), (1, ))
    assert_size_stride(arg463_1, (512, 2048, 1, 1), (2048, 1, 2048, 2048))
    assert_size_stride(arg464_1, (512, ), (1, ))
    assert_size_stride(arg465_1, (512, ), (1, ))
    assert_size_stride(arg466_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg467_1, (512, ), (1, ))
    assert_size_stride(arg468_1, (512, ), (1, ))
    assert_size_stride(arg469_1, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg470_1, (2048, ), (1, ))
    assert_size_stride(arg471_1, (2048, ), (1, ))
    assert_size_stride(arg472_1, (512, 2048, 1, 1), (2048, 1, 2048, 2048))
    assert_size_stride(arg473_1, (512, ), (1, ))
    assert_size_stride(arg474_1, (512, ), (1, ))
    assert_size_stride(arg475_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg476_1, (512, ), (1, ))
    assert_size_stride(arg477_1, (512, ), (1, ))
    assert_size_stride(arg478_1, (2048, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg479_1, (2048, ), (1, ))
    assert_size_stride(arg480_1, (2048, ), (1, ))
    assert_size_stride(arg481_1, (1000, 2048), (2048, 1))
    assert_size_stride(arg482_1, (1000, ), (1, ))
    assert_size_stride(arg483_1, (), ())
    assert_size_stride(arg484_1, (), ())
    assert_size_stride(arg485_1, (), ())
    assert_size_stride(arg486_1, (), ())
    assert_size_stride(arg487_1, (), ())
    assert_size_stride(arg488_1, (), ())
    assert_size_stride(arg489_1, (), ())
    assert_size_stride(arg490_1, (), ())
    assert_size_stride(arg491_1, (), ())
    assert_size_stride(arg492_1, (), ())
    assert_size_stride(arg493_1, (), ())
    assert_size_stride(arg494_1, (), ())
    assert_size_stride(arg495_1, (), ())
    assert_size_stride(arg496_1, (), ())
    assert_size_stride(arg497_1, (), ())
    assert_size_stride(arg498_1, (), ())
    assert_size_stride(arg499_1, (), ())
    assert_size_stride(arg500_1, (), ())
    assert_size_stride(arg501_1, (), ())
    assert_size_stride(arg502_1, (), ())
    assert_size_stride(arg503_1, (), ())
    assert_size_stride(arg504_1, (), ())
    assert_size_stride(arg505_1, (), ())
    assert_size_stride(arg506_1, (), ())
    assert_size_stride(arg507_1, (), ())
    assert_size_stride(arg508_1, (), ())
    assert_size_stride(arg509_1, (), ())
    assert_size_stride(arg510_1, (), ())
    assert_size_stride(arg511_1, (), ())
    assert_size_stride(arg512_1, (), ())
    assert_size_stride(arg513_1, (), ())
    assert_size_stride(arg514_1, (), ())
    assert_size_stride(arg515_1, (), ())
    assert_size_stride(arg516_1, (), ())
    assert_size_stride(arg517_1, (), ())
    assert_size_stride(arg518_1, (), ())
    assert_size_stride(arg519_1, (), ())
    assert_size_stride(arg520_1, (), ())
    assert_size_stride(arg521_1, (), ())
    assert_size_stride(arg522_1, (), ())
    assert_size_stride(arg523_1, (), ())
    assert_size_stride(arg524_1, (), ())
    assert_size_stride(arg525_1, (), ())
    assert_size_stride(arg526_1, (), ())
    assert_size_stride(arg527_1, (), ())
    assert_size_stride(arg528_1, (), ())
    assert_size_stride(arg529_1, (), ())
    assert_size_stride(arg530_1, (), ())
    assert_size_stride(arg531_1, (), ())
    assert_size_stride(arg532_1, (), ())
    assert_size_stride(arg533_1, (), ())
    assert_size_stride(arg534_1, (), ())
    assert_size_stride(arg535_1, (), ())
    assert_size_stride(arg536_1, (), ())
    assert_size_stride(arg537_1, (), ())
    assert_size_stride(arg538_1, (), ())
    assert_size_stride(arg539_1, (), ())
    assert_size_stride(arg540_1, (), ())
    assert_size_stride(arg541_1, (), ())
    assert_size_stride(arg542_1, (), ())
    assert_size_stride(arg543_1, (), ())
    assert_size_stride(arg544_1, (), ())
    assert_size_stride(arg545_1, (), ())
    assert_size_stride(arg546_1, (), ())
    assert_size_stride(arg547_1, (), ())
    assert_size_stride(arg548_1, (), ())
    assert_size_stride(arg549_1, (), ())
    assert_size_stride(arg550_1, (), ())
    assert_size_stride(arg551_1, (), ())
    assert_size_stride(arg552_1, (), ())
    assert_size_stride(arg553_1, (), ())
    assert_size_stride(arg554_1, (), ())
    assert_size_stride(arg555_1, (), ())
    assert_size_stride(arg556_1, (), ())
    assert_size_stride(arg557_1, (), ())
    assert_size_stride(arg558_1, (), ())
    assert_size_stride(arg559_1, (), ())
    assert_size_stride(arg560_1, (), ())
    assert_size_stride(arg561_1, (), ())
    assert_size_stride(arg562_1, (), ())
    assert_size_stride(arg563_1, (), ())
    assert_size_stride(arg564_1, (), ())
    assert_size_stride(arg565_1, (), ())
    assert_size_stride(arg566_1, (), ())
    assert_size_stride(arg567_1, (), ())
    assert_size_stride(arg568_1, (), ())
    assert_size_stride(arg569_1, (), ())
    assert_size_stride(arg570_1, (), ())
    assert_size_stride(arg571_1, (), ())
    assert_size_stride(arg572_1, (), ())
    assert_size_stride(arg573_1, (), ())
    assert_size_stride(arg574_1, (), ())
    assert_size_stride(arg575_1, (), ())
    assert_size_stride(arg576_1, (), ())
    assert_size_stride(arg577_1, (), ())
    assert_size_stride(arg578_1, (), ())
    assert_size_stride(arg579_1, (), ())
    assert_size_stride(arg580_1, (), ())
    assert_size_stride(arg581_1, (), ())
    assert_size_stride(arg582_1, (), ())
    assert_size_stride(arg583_1, (), ())
    assert_size_stride(arg584_1, (), ())
    assert_size_stride(arg585_1, (), ())
    assert_size_stride(arg586_1, (), ())
    assert_size_stride(arg587_1, (), ())
    assert_size_stride(arg588_1, (), ())
    assert_size_stride(arg589_1, (), ())
    assert_size_stride(arg590_1, (), ())
    assert_size_stride(arg591_1, (), ())
    assert_size_stride(arg592_1, (), ())
    assert_size_stride(arg593_1, (), ())
    assert_size_stride(arg594_1, (), ())
    assert_size_stride(arg595_1, (), ())
    assert_size_stride(arg596_1, (), ())
    assert_size_stride(arg597_1, (), ())
    assert_size_stride(arg598_1, (), ())
    assert_size_stride(arg599_1, (), ())
    assert_size_stride(arg600_1, (), ())
    assert_size_stride(arg601_1, (), ())
    assert_size_stride(arg602_1, (), ())
    assert_size_stride(arg603_1, (), ())
    assert_size_stride(arg604_1, (), ())
    assert_size_stride(arg605_1, (), ())
    assert_size_stride(arg606_1, (), ())
    assert_size_stride(arg607_1, (), ())
    assert_size_stride(arg608_1, (), ())
    assert_size_stride(arg609_1, (), ())
    assert_size_stride(arg610_1, (), ())
    assert_size_stride(arg611_1, (), ())
    assert_size_stride(arg612_1, (), ())
    assert_size_stride(arg613_1, (), ())
    assert_size_stride(arg614_1, (), ())
    assert_size_stride(arg615_1, (), ())
    assert_size_stride(arg616_1, (), ())
    assert_size_stride(arg617_1, (), ())
    assert_size_stride(arg618_1, (), ())
    assert_size_stride(arg619_1, (), ())
    assert_size_stride(arg620_1, (), ())
    assert_size_stride(arg621_1, (), ())
    assert_size_stride(arg622_1, (), ())
    assert_size_stride(arg623_1, (), ())
    assert_size_stride(arg624_1, (), ())
    assert_size_stride(arg625_1, (), ())
    assert_size_stride(arg626_1, (), ())
    assert_size_stride(arg627_1, (), ())
    assert_size_stride(arg628_1, (), ())
    assert_size_stride(arg629_1, (), ())
    assert_size_stride(arg630_1, (), ())
    assert_size_stride(arg631_1, (), ())
    assert_size_stride(arg632_1, (), ())
    assert_size_stride(arg633_1, (), ())
    assert_size_stride(arg634_1, (), ())
    assert_size_stride(arg635_1, (), ())
    assert_size_stride(arg636_1, (), ())
    assert_size_stride(arg637_1, (), ())
    assert_size_stride(arg638_1, (), ())
    assert_size_stride(arg639_1, (), ())
    assert_size_stride(arg640_1, (), ())
    assert_size_stride(arg641_1, (), ())
    assert_size_stride(arg642_1, (), ())
    assert_size_stride(arg643_1, (), ())
    assert_size_stride(arg644_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(arg645_1, (64, ), (1, ))
    assert_size_stride(arg646_1, (64, ), (1, ))
    assert_size_stride(arg647_1, (64, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(arg648_1, (64, ), (1, ))
    assert_size_stride(arg649_1, (64, ), (1, ))
    assert_size_stride(arg650_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg651_1, (64, ), (1, ))
    assert_size_stride(arg652_1, (64, ), (1, ))
    assert_size_stride(arg653_1, (256, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(arg654_1, (256, ), (1, ))
    assert_size_stride(arg655_1, (256, ), (1, ))
    assert_size_stride(arg656_1, (256, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(arg657_1, (256, ), (1, ))
    assert_size_stride(arg658_1, (256, ), (1, ))
    assert_size_stride(arg659_1, (64, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg660_1, (64, ), (1, ))
    assert_size_stride(arg661_1, (64, ), (1, ))
    assert_size_stride(arg662_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg663_1, (64, ), (1, ))
    assert_size_stride(arg664_1, (64, ), (1, ))
    assert_size_stride(arg665_1, (256, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(arg666_1, (256, ), (1, ))
    assert_size_stride(arg667_1, (256, ), (1, ))
    assert_size_stride(arg668_1, (64, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg669_1, (64, ), (1, ))
    assert_size_stride(arg670_1, (64, ), (1, ))
    assert_size_stride(arg671_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg672_1, (64, ), (1, ))
    assert_size_stride(arg673_1, (64, ), (1, ))
    assert_size_stride(arg674_1, (256, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(arg675_1, (256, ), (1, ))
    assert_size_stride(arg676_1, (256, ), (1, ))
    assert_size_stride(arg677_1, (128, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg678_1, (128, ), (1, ))
    assert_size_stride(arg679_1, (128, ), (1, ))
    assert_size_stride(arg680_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg681_1, (128, ), (1, ))
    assert_size_stride(arg682_1, (128, ), (1, ))
    assert_size_stride(arg683_1, (512, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(arg684_1, (512, ), (1, ))
    assert_size_stride(arg685_1, (512, ), (1, ))
    assert_size_stride(arg686_1, (512, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg687_1, (512, ), (1, ))
    assert_size_stride(arg688_1, (512, ), (1, ))
    assert_size_stride(arg689_1, (128, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg690_1, (128, ), (1, ))
    assert_size_stride(arg691_1, (128, ), (1, ))
    assert_size_stride(arg692_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg693_1, (128, ), (1, ))
    assert_size_stride(arg694_1, (128, ), (1, ))
    assert_size_stride(arg695_1, (512, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(arg696_1, (512, ), (1, ))
    assert_size_stride(arg697_1, (512, ), (1, ))
    assert_size_stride(arg698_1, (128, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg699_1, (128, ), (1, ))
    assert_size_stride(arg700_1, (128, ), (1, ))
    assert_size_stride(arg701_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg702_1, (128, ), (1, ))
    assert_size_stride(arg703_1, (128, ), (1, ))
    assert_size_stride(arg704_1, (512, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(arg705_1, (512, ), (1, ))
    assert_size_stride(arg706_1, (512, ), (1, ))
    assert_size_stride(arg707_1, (128, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg708_1, (128, ), (1, ))
    assert_size_stride(arg709_1, (128, ), (1, ))
    assert_size_stride(arg710_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg711_1, (128, ), (1, ))
    assert_size_stride(arg712_1, (128, ), (1, ))
    assert_size_stride(arg713_1, (512, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(arg714_1, (512, ), (1, ))
    assert_size_stride(arg715_1, (512, ), (1, ))
    assert_size_stride(arg716_1, (256, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg717_1, (256, ), (1, ))
    assert_size_stride(arg718_1, (256, ), (1, ))
    assert_size_stride(arg719_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg720_1, (256, ), (1, ))
    assert_size_stride(arg721_1, (256, ), (1, ))
    assert_size_stride(arg722_1, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg723_1, (1024, ), (1, ))
    assert_size_stride(arg724_1, (1024, ), (1, ))
    assert_size_stride(arg725_1, (1024, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg726_1, (1024, ), (1, ))
    assert_size_stride(arg727_1, (1024, ), (1, ))
    assert_size_stride(arg728_1, (256, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(arg729_1, (256, ), (1, ))
    assert_size_stride(arg730_1, (256, ), (1, ))
    assert_size_stride(arg731_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg732_1, (256, ), (1, ))
    assert_size_stride(arg733_1, (256, ), (1, ))
    assert_size_stride(arg734_1, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg735_1, (1024, ), (1, ))
    assert_size_stride(arg736_1, (1024, ), (1, ))
    assert_size_stride(arg737_1, (256, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(arg738_1, (256, ), (1, ))
    assert_size_stride(arg739_1, (256, ), (1, ))
    assert_size_stride(arg740_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg741_1, (256, ), (1, ))
    assert_size_stride(arg742_1, (256, ), (1, ))
    assert_size_stride(arg743_1, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg744_1, (1024, ), (1, ))
    assert_size_stride(arg745_1, (1024, ), (1, ))
    assert_size_stride(arg746_1, (256, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(arg747_1, (256, ), (1, ))
    assert_size_stride(arg748_1, (256, ), (1, ))
    assert_size_stride(arg749_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg750_1, (256, ), (1, ))
    assert_size_stride(arg751_1, (256, ), (1, ))
    assert_size_stride(arg752_1, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg753_1, (1024, ), (1, ))
    assert_size_stride(arg754_1, (1024, ), (1, ))
    assert_size_stride(arg755_1, (256, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(arg756_1, (256, ), (1, ))
    assert_size_stride(arg757_1, (256, ), (1, ))
    assert_size_stride(arg758_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg759_1, (256, ), (1, ))
    assert_size_stride(arg760_1, (256, ), (1, ))
    assert_size_stride(arg761_1, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg762_1, (1024, ), (1, ))
    assert_size_stride(arg763_1, (1024, ), (1, ))
    assert_size_stride(arg764_1, (256, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(arg765_1, (256, ), (1, ))
    assert_size_stride(arg766_1, (256, ), (1, ))
    assert_size_stride(arg767_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg768_1, (256, ), (1, ))
    assert_size_stride(arg769_1, (256, ), (1, ))
    assert_size_stride(arg770_1, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg771_1, (1024, ), (1, ))
    assert_size_stride(arg772_1, (1024, ), (1, ))
    assert_size_stride(arg773_1, (512, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(arg774_1, (512, ), (1, ))
    assert_size_stride(arg775_1, (512, ), (1, ))
    assert_size_stride(arg776_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg777_1, (512, ), (1, ))
    assert_size_stride(arg778_1, (512, ), (1, ))
    assert_size_stride(arg779_1, (2048, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg780_1, (2048, ), (1, ))
    assert_size_stride(arg781_1, (2048, ), (1, ))
    assert_size_stride(arg782_1, (2048, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(arg783_1, (2048, ), (1, ))
    assert_size_stride(arg784_1, (2048, ), (1, ))
    assert_size_stride(arg785_1, (512, 2048, 1, 1), (2048, 1, 2048, 2048))
    assert_size_stride(arg786_1, (512, ), (1, ))
    assert_size_stride(arg787_1, (512, ), (1, ))
    assert_size_stride(arg788_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg789_1, (512, ), (1, ))
    assert_size_stride(arg790_1, (512, ), (1, ))
    assert_size_stride(arg791_1, (2048, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg792_1, (2048, ), (1, ))
    assert_size_stride(arg793_1, (2048, ), (1, ))
    assert_size_stride(arg794_1, (512, 2048, 1, 1), (2048, 1, 2048, 2048))
    assert_size_stride(arg795_1, (512, ), (1, ))
    assert_size_stride(arg796_1, (512, ), (1, ))
    assert_size_stride(arg797_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg798_1, (512, ), (1, ))
    assert_size_stride(arg799_1, (512, ), (1, ))
    assert_size_stride(arg800_1, (2048, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg801_1, (2048, ), (1, ))
    assert_size_stride(arg802_1, (2048, ), (1, ))
    assert_size_stride(arg803_1, (1000, 2048), (2048, 1))
    assert_size_stride(arg804_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_for_fused_0.run(arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, grid=((83, 1, 1)), stream=stream0)
        # Source Nodes: [], Original ATen: []
        triton_for_fused_1.run(arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, grid=((78, 1, 1)), stream=stream0)
        # Source Nodes: [], Original ATen: []
        triton_for_fused_2.run(arg161_1, arg644_1, arg322_1, arg0_1, arg483_1, arg162_1, arg645_1, arg323_1, arg1_1, arg484_1, arg163_1, arg646_1, arg324_1, arg2_1, arg485_1, arg164_1, arg647_1, arg325_1, arg3_1, arg486_1, arg165_1, arg648_1, arg326_1, arg4_1, arg487_1, arg166_1, arg649_1, arg327_1, arg5_1, arg488_1, arg167_1, arg650_1, arg328_1, arg6_1, arg489_1, arg168_1, arg651_1, arg329_1, arg7_1, arg490_1, arg169_1, arg652_1, arg330_1, arg8_1, arg491_1, arg170_1, arg653_1, arg331_1, arg9_1, arg492_1, arg171_1, arg654_1, arg332_1, arg10_1, arg493_1, arg172_1, arg655_1, arg333_1, arg11_1, arg494_1, arg173_1, arg656_1, arg334_1, arg12_1, arg495_1, arg174_1, arg657_1, arg335_1, arg13_1, arg496_1, arg175_1, arg658_1, arg336_1, arg14_1, arg497_1, arg176_1, arg659_1, arg337_1, arg15_1, arg498_1, arg177_1, arg660_1, arg338_1, arg16_1, arg499_1, arg178_1, arg661_1, arg339_1, arg17_1, arg500_1, arg179_1, arg662_1, arg340_1, arg18_1, arg501_1, arg180_1, arg663_1, arg341_1, arg19_1, arg502_1, arg161_1, arg0_1, arg322_1, arg162_1, arg1_1, arg323_1, arg163_1, arg2_1, arg324_1, arg164_1, arg3_1, arg325_1, arg165_1, arg4_1, arg326_1, arg166_1, arg5_1, arg327_1, arg167_1, arg6_1, arg328_1, arg168_1, arg7_1, arg329_1, arg169_1, arg8_1, arg330_1, arg170_1, arg9_1, arg331_1, arg171_1, arg10_1, arg332_1, arg172_1, arg11_1, arg333_1, arg173_1, arg12_1, arg334_1, arg174_1, arg13_1, arg335_1, arg175_1, arg14_1, arg336_1, arg176_1, arg15_1, arg337_1, arg177_1, arg16_1, arg338_1, arg178_1, arg17_1, arg339_1, arg179_1, arg18_1, arg340_1, arg180_1, arg19_1, arg341_1, grid=((147, 1, 1)), stream=stream0)
        # Source Nodes: [], Original ATen: []
        triton_for_fused_3.run(arg181_1, arg664_1, arg342_1, arg20_1, arg503_1, arg182_1, arg665_1, arg343_1, arg21_1, arg504_1, arg183_1, arg666_1, arg344_1, arg22_1, arg505_1, arg184_1, arg667_1, arg345_1, arg23_1, arg506_1, arg185_1, arg668_1, arg346_1, arg24_1, arg507_1, arg186_1, arg669_1, arg347_1, arg25_1, arg508_1, arg187_1, arg670_1, arg348_1, arg26_1, arg509_1, arg188_1, arg671_1, arg349_1, arg27_1, arg510_1, arg189_1, arg672_1, arg350_1, arg28_1, arg511_1, arg190_1, arg673_1, arg351_1, arg29_1, arg512_1, arg191_1, arg674_1, arg352_1, arg30_1, arg513_1, arg192_1, arg675_1, arg353_1, arg31_1, arg514_1, arg193_1, arg676_1, arg354_1, arg32_1, arg515_1, arg194_1, arg677_1, arg355_1, arg33_1, arg516_1, arg195_1, arg678_1, arg356_1, arg34_1, arg517_1, arg196_1, arg679_1, arg357_1, arg35_1, arg518_1, arg197_1, arg680_1, arg358_1, arg36_1, arg519_1, arg198_1, arg681_1, arg359_1, arg37_1, arg520_1, arg199_1, arg682_1, arg360_1, arg38_1, arg521_1, arg200_1, arg683_1, arg361_1, arg39_1, arg522_1, arg181_1, arg20_1, arg342_1, arg182_1, arg21_1, arg343_1, arg183_1, arg22_1, arg344_1, arg184_1, arg23_1, arg345_1, arg185_1, arg24_1, arg346_1, arg186_1, arg25_1, arg347_1, arg187_1, arg26_1, arg348_1, arg188_1, arg27_1, arg349_1, arg189_1, arg28_1, arg350_1, arg190_1, arg29_1, arg351_1, arg191_1, arg30_1, arg352_1, arg192_1, arg31_1, arg353_1, arg193_1, arg32_1, arg354_1, arg194_1, arg33_1, arg355_1, arg195_1, arg34_1, arg356_1, arg196_1, arg35_1, arg357_1, arg197_1, arg36_1, arg358_1, arg198_1, arg37_1, arg359_1, arg199_1, arg38_1, arg360_1, arg200_1, arg39_1, arg361_1, grid=((337, 1, 1)), stream=stream0)
        # Source Nodes: [], Original ATen: []
        triton_for_fused_4.run(arg201_1, arg684_1, arg362_1, arg40_1, arg523_1, arg202_1, arg685_1, arg363_1, arg41_1, arg524_1, arg203_1, arg686_1, arg364_1, arg42_1, arg525_1, arg204_1, arg687_1, arg365_1, arg43_1, arg526_1, arg205_1, arg688_1, arg366_1, arg44_1, arg527_1, arg206_1, arg689_1, arg367_1, arg45_1, arg528_1, arg207_1, arg690_1, arg368_1, arg46_1, arg529_1, arg208_1, arg691_1, arg369_1, arg47_1, arg530_1, arg209_1, arg692_1, arg370_1, arg48_1, arg531_1, arg210_1, arg693_1, arg371_1, arg49_1, arg532_1, arg211_1, arg694_1, arg372_1, arg50_1, arg533_1, arg212_1, arg695_1, arg373_1, arg51_1, arg534_1, arg213_1, arg696_1, arg374_1, arg52_1, arg535_1, arg214_1, arg697_1, arg375_1, arg53_1, arg536_1, arg215_1, arg698_1, arg376_1, arg54_1, arg537_1, arg216_1, arg699_1, arg377_1, arg55_1, arg538_1, arg217_1, arg700_1, arg378_1, arg56_1, arg539_1, arg218_1, arg701_1, arg379_1, arg57_1, arg540_1, arg219_1, arg702_1, arg380_1, arg58_1, arg541_1, arg220_1, arg703_1, arg381_1, arg59_1, arg542_1, arg201_1, arg40_1, arg362_1, arg202_1, arg41_1, arg363_1, arg203_1, arg42_1, arg364_1, arg204_1, arg43_1, arg365_1, arg205_1, arg44_1, arg366_1, arg206_1, arg45_1, arg367_1, arg207_1, arg46_1, arg368_1, arg208_1, arg47_1, arg369_1, arg209_1, arg48_1, arg370_1, arg210_1, arg49_1, arg371_1, arg211_1, arg50_1, arg372_1, arg212_1, arg51_1, arg373_1, arg213_1, arg52_1, arg374_1, arg214_1, arg53_1, arg375_1, arg215_1, arg54_1, arg376_1, arg216_1, arg55_1, arg377_1, arg217_1, arg56_1, arg378_1, arg218_1, arg57_1, arg379_1, arg219_1, arg58_1, arg380_1, arg220_1, arg59_1, arg381_1, grid=((622, 1, 1)), stream=stream0)
        # Source Nodes: [], Original ATen: []
        triton_for_fused_5.run(arg221_1, arg704_1, arg382_1, arg60_1, arg543_1, arg222_1, arg705_1, arg383_1, arg61_1, arg544_1, arg223_1, arg706_1, arg384_1, arg62_1, arg545_1, arg224_1, arg707_1, arg385_1, arg63_1, arg546_1, arg225_1, arg708_1, arg386_1, arg64_1, arg547_1, arg226_1, arg709_1, arg387_1, arg65_1, arg548_1, arg227_1, arg710_1, arg388_1, arg66_1, arg549_1, arg228_1, arg711_1, arg389_1, arg67_1, arg550_1, arg229_1, arg712_1, arg390_1, arg68_1, arg551_1, arg230_1, arg713_1, arg391_1, arg69_1, arg552_1, arg231_1, arg714_1, arg392_1, arg70_1, arg553_1, arg232_1, arg715_1, arg393_1, arg71_1, arg554_1, arg233_1, arg716_1, arg394_1, arg72_1, arg555_1, arg234_1, arg717_1, arg395_1, arg73_1, arg556_1, arg235_1, arg718_1, arg396_1, arg74_1, arg557_1, arg236_1, arg719_1, arg397_1, arg75_1, arg558_1, arg237_1, arg720_1, arg398_1, arg76_1, arg559_1, arg238_1, arg721_1, arg399_1, arg77_1, arg560_1, arg239_1, arg722_1, arg400_1, arg78_1, arg561_1, arg240_1, arg723_1, arg401_1, arg79_1, arg562_1, arg221_1, arg60_1, arg382_1, arg222_1, arg61_1, arg383_1, arg223_1, arg62_1, arg384_1, arg224_1, arg63_1, arg385_1, arg225_1, arg64_1, arg386_1, arg226_1, arg65_1, arg387_1, arg227_1, arg66_1, arg388_1, arg228_1, arg67_1, arg389_1, arg229_1, arg68_1, arg390_1, arg230_1, arg69_1, arg391_1, arg231_1, arg70_1, arg392_1, arg232_1, arg71_1, arg393_1, arg233_1, arg72_1, arg394_1, arg234_1, arg73_1, arg395_1, arg235_1, arg74_1, arg396_1, arg236_1, arg75_1, arg397_1, arg237_1, arg76_1, arg398_1, arg238_1, arg77_1, arg399_1, arg239_1, arg78_1, arg400_1, arg240_1, arg79_1, arg401_1, grid=((1309, 1, 1)), stream=stream0)
        # Source Nodes: [], Original ATen: []
        triton_for_fused_6.run(arg241_1, arg724_1, arg402_1, arg80_1, arg563_1, arg242_1, arg725_1, arg403_1, arg81_1, arg564_1, arg243_1, arg726_1, arg404_1, arg82_1, arg565_1, arg244_1, arg727_1, arg405_1, arg83_1, arg566_1, arg245_1, arg728_1, arg406_1, arg84_1, arg567_1, arg246_1, arg729_1, arg407_1, arg85_1, arg568_1, arg247_1, arg730_1, arg408_1, arg86_1, arg569_1, arg248_1, arg731_1, arg409_1, arg87_1, arg570_1, arg249_1, arg732_1, arg410_1, arg88_1, arg571_1, arg250_1, arg733_1, arg411_1, arg89_1, arg572_1, arg251_1, arg734_1, arg412_1, arg90_1, arg573_1, arg252_1, arg735_1, arg413_1, arg91_1, arg574_1, arg253_1, arg736_1, arg414_1, arg92_1, arg575_1, arg254_1, arg737_1, arg415_1, arg93_1, arg576_1, arg255_1, arg738_1, arg416_1, arg94_1, arg577_1, arg256_1, arg739_1, arg417_1, arg95_1, arg578_1, arg257_1, arg740_1, arg418_1, arg96_1, arg579_1, arg258_1, arg741_1, arg419_1, arg97_1, arg580_1, arg259_1, arg742_1, arg420_1, arg98_1, arg581_1, arg260_1, arg743_1, arg421_1, arg99_1, arg582_1, arg241_1, arg80_1, arg402_1, arg242_1, arg81_1, arg403_1, arg243_1, arg82_1, arg404_1, arg244_1, arg83_1, arg405_1, arg245_1, arg84_1, arg406_1, arg246_1, arg85_1, arg407_1, arg247_1, arg86_1, arg408_1, arg248_1, arg87_1, arg409_1, arg249_1, arg88_1, arg410_1, arg250_1, arg89_1, arg411_1, arg251_1, arg90_1, arg412_1, arg252_1, arg91_1, arg413_1, arg253_1, arg92_1, arg414_1, arg254_1, arg93_1, arg415_1, arg255_1, arg94_1, arg416_1, arg256_1, arg95_1, arg417_1, arg257_1, arg96_1, arg418_1, arg258_1, arg97_1, arg419_1, arg259_1, arg98_1, arg420_1, arg260_1, arg99_1, arg421_1, grid=((2701, 1, 1)), stream=stream0)
        # Source Nodes: [], Original ATen: []
        triton_for_fused_7.run(arg261_1, arg744_1, arg422_1, arg100_1, arg583_1, arg262_1, arg745_1, arg423_1, arg101_1, arg584_1, arg263_1, arg746_1, arg424_1, arg102_1, arg585_1, arg264_1, arg747_1, arg425_1, arg103_1, arg586_1, arg265_1, arg748_1, arg426_1, arg104_1, arg587_1, arg266_1, arg749_1, arg427_1, arg105_1, arg588_1, arg267_1, arg750_1, arg428_1, arg106_1, arg589_1, arg268_1, arg751_1, arg429_1, arg107_1, arg590_1, arg269_1, arg752_1, arg430_1, arg108_1, arg591_1, arg270_1, arg753_1, arg431_1, arg109_1, arg592_1, arg271_1, arg754_1, arg432_1, arg110_1, arg593_1, arg272_1, arg755_1, arg433_1, arg111_1, arg594_1, arg273_1, arg756_1, arg434_1, arg112_1, arg595_1, arg274_1, arg757_1, arg435_1, arg113_1, arg596_1, arg275_1, arg758_1, arg436_1, arg114_1, arg597_1, arg276_1, arg759_1, arg437_1, arg115_1, arg598_1, arg277_1, arg760_1, arg438_1, arg116_1, arg599_1, arg278_1, arg761_1, arg439_1, arg117_1, arg600_1, arg279_1, arg762_1, arg440_1, arg118_1, arg601_1, arg280_1, arg763_1, arg441_1, arg119_1, arg602_1, arg261_1, arg100_1, arg422_1, arg262_1, arg101_1, arg423_1, arg263_1, arg102_1, arg424_1, arg264_1, arg103_1, arg425_1, arg265_1, arg104_1, arg426_1, arg266_1, arg105_1, arg427_1, arg267_1, arg106_1, arg428_1, arg268_1, arg107_1, arg429_1, arg269_1, arg108_1, arg430_1, arg270_1, arg109_1, arg431_1, arg271_1, arg110_1, arg432_1, arg272_1, arg111_1, arg433_1, arg273_1, arg112_1, arg434_1, arg274_1, arg113_1, arg435_1, arg275_1, arg114_1, arg436_1, arg276_1, arg115_1, arg437_1, arg277_1, arg116_1, arg438_1, arg278_1, arg117_1, arg439_1, arg279_1, arg118_1, arg440_1, arg280_1, arg119_1, arg441_1, grid=((2190, 1, 1)), stream=stream0)
        # Source Nodes: [], Original ATen: []
        triton_for_fused_8.run(arg281_1, arg764_1, arg442_1, arg120_1, arg603_1, arg282_1, arg765_1, arg443_1, arg121_1, arg604_1, arg283_1, arg766_1, arg444_1, arg122_1, arg605_1, arg284_1, arg767_1, arg445_1, arg123_1, arg606_1, arg285_1, arg768_1, arg446_1, arg124_1, arg607_1, arg286_1, arg769_1, arg447_1, arg125_1, arg608_1, arg287_1, arg770_1, arg448_1, arg126_1, arg609_1, arg288_1, arg771_1, arg449_1, arg127_1, arg610_1, arg289_1, arg772_1, arg450_1, arg128_1, arg611_1, arg290_1, arg773_1, arg451_1, arg129_1, arg612_1, arg291_1, arg774_1, arg452_1, arg130_1, arg613_1, arg292_1, arg775_1, arg453_1, arg131_1, arg614_1, arg293_1, arg776_1, arg454_1, arg132_1, arg615_1, arg294_1, arg777_1, arg455_1, arg133_1, arg616_1, arg295_1, arg778_1, arg456_1, arg134_1, arg617_1, arg296_1, arg779_1, arg457_1, arg135_1, arg618_1, arg297_1, arg780_1, arg458_1, arg136_1, arg619_1, arg298_1, arg781_1, arg459_1, arg137_1, arg620_1, arg299_1, arg782_1, arg460_1, arg138_1, arg621_1, arg300_1, arg783_1, arg461_1, arg139_1, arg622_1, arg281_1, arg120_1, arg442_1, arg282_1, arg121_1, arg443_1, arg283_1, arg122_1, arg444_1, arg284_1, arg123_1, arg445_1, arg285_1, arg124_1, arg446_1, arg286_1, arg125_1, arg447_1, arg287_1, arg126_1, arg448_1, arg288_1, arg127_1, arg449_1, arg289_1, arg128_1, arg450_1, arg290_1, arg129_1, arg451_1, arg291_1, arg130_1, arg452_1, arg292_1, arg131_1, arg453_1, arg293_1, arg132_1, arg454_1, arg294_1, arg133_1, arg455_1, arg295_1, arg134_1, arg456_1, arg296_1, arg135_1, arg457_1, arg297_1, arg136_1, arg458_1, arg298_1, arg137_1, arg459_1, arg299_1, arg138_1, arg460_1, arg300_1, arg139_1, arg461_1, grid=((6992, 1, 1)), stream=stream0)
        # Source Nodes: [], Original ATen: []
        triton_for_fused_9.run(arg301_1, arg784_1, arg462_1, arg140_1, arg623_1, arg302_1, arg785_1, arg463_1, arg141_1, arg624_1, arg303_1, arg786_1, arg464_1, arg142_1, arg625_1, arg304_1, arg787_1, arg465_1, arg143_1, arg626_1, arg305_1, arg788_1, arg466_1, arg144_1, arg627_1, arg306_1, arg789_1, arg467_1, arg145_1, arg628_1, arg307_1, arg790_1, arg468_1, arg146_1, arg629_1, arg308_1, arg791_1, arg469_1, arg147_1, arg630_1, arg309_1, arg792_1, arg470_1, arg148_1, arg631_1, arg310_1, arg793_1, arg471_1, arg149_1, arg632_1, arg311_1, arg794_1, arg472_1, arg150_1, arg633_1, arg312_1, arg795_1, arg473_1, arg151_1, arg634_1, arg313_1, arg796_1, arg474_1, arg152_1, arg635_1, arg314_1, arg797_1, arg475_1, arg153_1, arg636_1, arg315_1, arg798_1, arg476_1, arg154_1, arg637_1, arg316_1, arg799_1, arg477_1, arg155_1, arg638_1, arg317_1, arg800_1, arg478_1, arg156_1, arg639_1, arg318_1, arg801_1, arg479_1, arg157_1, arg640_1, arg319_1, arg802_1, arg480_1, arg158_1, arg641_1, arg320_1, arg803_1, arg481_1, arg159_1, arg642_1, arg301_1, arg140_1, arg462_1, arg302_1, arg141_1, arg463_1, arg303_1, arg142_1, arg464_1, arg304_1, arg143_1, arg465_1, arg305_1, arg144_1, arg466_1, arg306_1, arg145_1, arg467_1, arg307_1, arg146_1, arg468_1, arg308_1, arg147_1, arg469_1, arg309_1, arg148_1, arg470_1, arg310_1, arg149_1, arg471_1, arg311_1, arg150_1, arg472_1, arg312_1, arg151_1, arg473_1, arg313_1, arg152_1, arg474_1, arg314_1, arg153_1, arg475_1, arg315_1, arg154_1, arg476_1, arg316_1, arg155_1, arg477_1, arg317_1, arg156_1, arg478_1, arg318_1, arg157_1, arg479_1, arg319_1, arg158_1, arg480_1, arg320_1, arg159_1, arg481_1, grid=((10722, 1, 1)), stream=stream0)
        # Source Nodes: [], Original ATen: []
        triton_for_fused_10.run(arg321_1, arg804_1, arg482_1, arg160_1, arg643_1, arg321_1, arg160_1, arg482_1, grid=((1, 1, 1)), stream=stream0)
        del arg0_1
        del arg100_1
        del arg101_1
        del arg102_1
        del arg103_1
        del arg104_1
        del arg105_1
        del arg106_1
        del arg107_1
        del arg108_1
        del arg109_1
        del arg10_1
        del arg110_1
        del arg111_1
        del arg112_1
        del arg113_1
        del arg114_1
        del arg115_1
        del arg116_1
        del arg117_1
        del arg118_1
        del arg119_1
        del arg11_1
        del arg120_1
        del arg121_1
        del arg122_1
        del arg123_1
        del arg124_1
        del arg125_1
        del arg126_1
        del arg127_1
        del arg128_1
        del arg129_1
        del arg12_1
        del arg130_1
        del arg131_1
        del arg132_1
        del arg133_1
        del arg134_1
        del arg135_1
        del arg136_1
        del arg137_1
        del arg138_1
        del arg139_1
        del arg13_1
        del arg140_1
        del arg141_1
        del arg142_1
        del arg143_1
        del arg144_1
        del arg145_1
        del arg146_1
        del arg147_1
        del arg148_1
        del arg149_1
        del arg14_1
        del arg150_1
        del arg151_1
        del arg152_1
        del arg153_1
        del arg154_1
        del arg155_1
        del arg156_1
        del arg157_1
        del arg158_1
        del arg159_1
        del arg15_1
        del arg160_1
        del arg161_1
        del arg162_1
        del arg163_1
        del arg164_1
        del arg165_1
        del arg166_1
        del arg167_1
        del arg168_1
        del arg169_1
        del arg16_1
        del arg170_1
        del arg171_1
        del arg172_1
        del arg173_1
        del arg174_1
        del arg175_1
        del arg176_1
        del arg177_1
        del arg178_1
        del arg179_1
        del arg17_1
        del arg180_1
        del arg181_1
        del arg182_1
        del arg183_1
        del arg184_1
        del arg185_1
        del arg186_1
        del arg187_1
        del arg188_1
        del arg189_1
        del arg18_1
        del arg190_1
        del arg191_1
        del arg192_1
        del arg193_1
        del arg194_1
        del arg195_1
        del arg196_1
        del arg197_1
        del arg198_1
        del arg199_1
        del arg19_1
        del arg1_1
        del arg200_1
        del arg201_1
        del arg202_1
        del arg203_1
        del arg204_1
        del arg205_1
        del arg206_1
        del arg207_1
        del arg208_1
        del arg209_1
        del arg20_1
        del arg210_1
        del arg211_1
        del arg212_1
        del arg213_1
        del arg214_1
        del arg215_1
        del arg216_1
        del arg217_1
        del arg218_1
        del arg219_1
        del arg21_1
        del arg220_1
        del arg221_1
        del arg222_1
        del arg223_1
        del arg224_1
        del arg225_1
        del arg226_1
        del arg227_1
        del arg228_1
        del arg229_1
        del arg22_1
        del arg230_1
        del arg231_1
        del arg232_1
        del arg233_1
        del arg234_1
        del arg235_1
        del arg236_1
        del arg237_1
        del arg238_1
        del arg239_1
        del arg23_1
        del arg240_1
        del arg241_1
        del arg242_1
        del arg243_1
        del arg244_1
        del arg245_1
        del arg246_1
        del arg247_1
        del arg248_1
        del arg249_1
        del arg24_1
        del arg250_1
        del arg251_1
        del arg252_1
        del arg253_1
        del arg254_1
        del arg255_1
        del arg256_1
        del arg257_1
        del arg258_1
        del arg259_1
        del arg25_1
        del arg260_1
        del arg261_1
        del arg262_1
        del arg263_1
        del arg264_1
        del arg265_1
        del arg266_1
        del arg267_1
        del arg268_1
        del arg269_1
        del arg26_1
        del arg270_1
        del arg271_1
        del arg272_1
        del arg273_1
        del arg274_1
        del arg275_1
        del arg276_1
        del arg277_1
        del arg278_1
        del arg279_1
        del arg27_1
        del arg280_1
        del arg281_1
        del arg282_1
        del arg283_1
        del arg284_1
        del arg285_1
        del arg286_1
        del arg287_1
        del arg288_1
        del arg289_1
        del arg28_1
        del arg290_1
        del arg291_1
        del arg292_1
        del arg293_1
        del arg294_1
        del arg295_1
        del arg296_1
        del arg297_1
        del arg298_1
        del arg299_1
        del arg29_1
        del arg2_1
        del arg300_1
        del arg301_1
        del arg302_1
        del arg303_1
        del arg304_1
        del arg305_1
        del arg306_1
        del arg307_1
        del arg308_1
        del arg309_1
        del arg30_1
        del arg310_1
        del arg311_1
        del arg312_1
        del arg313_1
        del arg314_1
        del arg315_1
        del arg316_1
        del arg317_1
        del arg318_1
        del arg319_1
        del arg31_1
        del arg320_1
        del arg321_1
        del arg322_1
        del arg323_1
        del arg324_1
        del arg325_1
        del arg326_1
        del arg327_1
        del arg328_1
        del arg329_1
        del arg32_1
        del arg330_1
        del arg331_1
        del arg332_1
        del arg333_1
        del arg334_1
        del arg335_1
        del arg336_1
        del arg337_1
        del arg338_1
        del arg339_1
        del arg33_1
        del arg340_1
        del arg341_1
        del arg342_1
        del arg343_1
        del arg344_1
        del arg345_1
        del arg346_1
        del arg347_1
        del arg348_1
        del arg349_1
        del arg34_1
        del arg350_1
        del arg351_1
        del arg352_1
        del arg353_1
        del arg354_1
        del arg355_1
        del arg356_1
        del arg357_1
        del arg358_1
        del arg359_1
        del arg35_1
        del arg360_1
        del arg361_1
        del arg362_1
        del arg363_1
        del arg364_1
        del arg365_1
        del arg366_1
        del arg367_1
        del arg368_1
        del arg369_1
        del arg36_1
        del arg370_1
        del arg371_1
        del arg372_1
        del arg373_1
        del arg374_1
        del arg375_1
        del arg376_1
        del arg377_1
        del arg378_1
        del arg379_1
        del arg37_1
        del arg380_1
        del arg381_1
        del arg382_1
        del arg383_1
        del arg384_1
        del arg385_1
        del arg386_1
        del arg387_1
        del arg388_1
        del arg389_1
        del arg38_1
        del arg390_1
        del arg391_1
        del arg392_1
        del arg393_1
        del arg394_1
        del arg395_1
        del arg396_1
        del arg397_1
        del arg398_1
        del arg399_1
        del arg39_1
        del arg3_1
        del arg400_1
        del arg401_1
        del arg402_1
        del arg403_1
        del arg404_1
        del arg405_1
        del arg406_1
        del arg407_1
        del arg408_1
        del arg409_1
        del arg40_1
        del arg410_1
        del arg411_1
        del arg412_1
        del arg413_1
        del arg414_1
        del arg415_1
        del arg416_1
        del arg417_1
        del arg418_1
        del arg419_1
        del arg41_1
        del arg420_1
        del arg421_1
        del arg422_1
        del arg423_1
        del arg424_1
        del arg425_1
        del arg426_1
        del arg427_1
        del arg428_1
        del arg429_1
        del arg42_1
        del arg430_1
        del arg431_1
        del arg432_1
        del arg433_1
        del arg434_1
        del arg435_1
        del arg436_1
        del arg437_1
        del arg438_1
        del arg439_1
        del arg43_1
        del arg440_1
        del arg441_1
        del arg442_1
        del arg443_1
        del arg444_1
        del arg445_1
        del arg446_1
        del arg447_1
        del arg448_1
        del arg449_1
        del arg44_1
        del arg450_1
        del arg451_1
        del arg452_1
        del arg453_1
        del arg454_1
        del arg455_1
        del arg456_1
        del arg457_1
        del arg458_1
        del arg459_1
        del arg45_1
        del arg460_1
        del arg461_1
        del arg462_1
        del arg463_1
        del arg464_1
        del arg465_1
        del arg466_1
        del arg467_1
        del arg468_1
        del arg469_1
        del arg46_1
        del arg470_1
        del arg471_1
        del arg472_1
        del arg473_1
        del arg474_1
        del arg475_1
        del arg476_1
        del arg477_1
        del arg478_1
        del arg479_1
        del arg47_1
        del arg480_1
        del arg481_1
        del arg482_1
        del arg483_1
        del arg484_1
        del arg485_1
        del arg486_1
        del arg487_1
        del arg488_1
        del arg489_1
        del arg48_1
        del arg490_1
        del arg491_1
        del arg492_1
        del arg493_1
        del arg494_1
        del arg495_1
        del arg496_1
        del arg497_1
        del arg498_1
        del arg499_1
        del arg49_1
        del arg4_1
        del arg500_1
        del arg501_1
        del arg502_1
        del arg503_1
        del arg504_1
        del arg505_1
        del arg506_1
        del arg507_1
        del arg508_1
        del arg509_1
        del arg50_1
        del arg510_1
        del arg511_1
        del arg512_1
        del arg513_1
        del arg514_1
        del arg515_1
        del arg516_1
        del arg517_1
        del arg518_1
        del arg519_1
        del arg51_1
        del arg520_1
        del arg521_1
        del arg522_1
        del arg523_1
        del arg524_1
        del arg525_1
        del arg526_1
        del arg527_1
        del arg528_1
        del arg529_1
        del arg52_1
        del arg530_1
        del arg531_1
        del arg532_1
        del arg533_1
        del arg534_1
        del arg535_1
        del arg536_1
        del arg537_1
        del arg538_1
        del arg539_1
        del arg53_1
        del arg540_1
        del arg541_1
        del arg542_1
        del arg543_1
        del arg544_1
        del arg545_1
        del arg546_1
        del arg547_1
        del arg548_1
        del arg549_1
        del arg54_1
        del arg550_1
        del arg551_1
        del arg552_1
        del arg553_1
        del arg554_1
        del arg555_1
        del arg556_1
        del arg557_1
        del arg558_1
        del arg559_1
        del arg55_1
        del arg560_1
        del arg561_1
        del arg562_1
        del arg563_1
        del arg564_1
        del arg565_1
        del arg566_1
        del arg567_1
        del arg568_1
        del arg569_1
        del arg56_1
        del arg570_1
        del arg571_1
        del arg572_1
        del arg573_1
        del arg574_1
        del arg575_1
        del arg576_1
        del arg577_1
        del arg578_1
        del arg579_1
        del arg57_1
        del arg580_1
        del arg581_1
        del arg582_1
        del arg583_1
        del arg584_1
        del arg585_1
        del arg586_1
        del arg587_1
        del arg588_1
        del arg589_1
        del arg58_1
        del arg590_1
        del arg591_1
        del arg592_1
        del arg593_1
        del arg594_1
        del arg595_1
        del arg596_1
        del arg597_1
        del arg598_1
        del arg599_1
        del arg59_1
        del arg5_1
        del arg600_1
        del arg601_1
        del arg602_1
        del arg603_1
        del arg604_1
        del arg605_1
        del arg606_1
        del arg607_1
        del arg608_1
        del arg609_1
        del arg60_1
        del arg610_1
        del arg611_1
        del arg612_1
        del arg613_1
        del arg614_1
        del arg615_1
        del arg616_1
        del arg617_1
        del arg618_1
        del arg619_1
        del arg61_1
        del arg620_1
        del arg621_1
        del arg622_1
        del arg623_1
        del arg624_1
        del arg625_1
        del arg626_1
        del arg627_1
        del arg628_1
        del arg629_1
        del arg62_1
        del arg630_1
        del arg631_1
        del arg632_1
        del arg633_1
        del arg634_1
        del arg635_1
        del arg636_1
        del arg637_1
        del arg638_1
        del arg639_1
        del arg63_1
        del arg640_1
        del arg641_1
        del arg642_1
        del arg643_1
        del arg644_1
        del arg645_1
        del arg646_1
        del arg647_1
        del arg648_1
        del arg649_1
        del arg64_1
        del arg650_1
        del arg651_1
        del arg652_1
        del arg653_1
        del arg654_1
        del arg655_1
        del arg656_1
        del arg657_1
        del arg658_1
        del arg659_1
        del arg65_1
        del arg660_1
        del arg661_1
        del arg662_1
        del arg663_1
        del arg664_1
        del arg665_1
        del arg666_1
        del arg667_1
        del arg668_1
        del arg669_1
        del arg66_1
        del arg670_1
        del arg671_1
        del arg672_1
        del arg673_1
        del arg674_1
        del arg675_1
        del arg676_1
        del arg677_1
        del arg678_1
        del arg679_1
        del arg67_1
        del arg680_1
        del arg681_1
        del arg682_1
        del arg683_1
        del arg684_1
        del arg685_1
        del arg686_1
        del arg687_1
        del arg688_1
        del arg689_1
        del arg68_1
        del arg690_1
        del arg691_1
        del arg692_1
        del arg693_1
        del arg694_1
        del arg695_1
        del arg696_1
        del arg697_1
        del arg698_1
        del arg699_1
        del arg69_1
        del arg6_1
        del arg700_1
        del arg701_1
        del arg702_1
        del arg703_1
        del arg704_1
        del arg705_1
        del arg706_1
        del arg707_1
        del arg708_1
        del arg709_1
        del arg70_1
        del arg710_1
        del arg711_1
        del arg712_1
        del arg713_1
        del arg714_1
        del arg715_1
        del arg716_1
        del arg717_1
        del arg718_1
        del arg719_1
        del arg71_1
        del arg720_1
        del arg721_1
        del arg722_1
        del arg723_1
        del arg724_1
        del arg725_1
        del arg726_1
        del arg727_1
        del arg728_1
        del arg729_1
        del arg72_1
        del arg730_1
        del arg731_1
        del arg732_1
        del arg733_1
        del arg734_1
        del arg735_1
        del arg736_1
        del arg737_1
        del arg738_1
        del arg739_1
        del arg73_1
        del arg740_1
        del arg741_1
        del arg742_1
        del arg743_1
        del arg744_1
        del arg745_1
        del arg746_1
        del arg747_1
        del arg748_1
        del arg749_1
        del arg74_1
        del arg750_1
        del arg751_1
        del arg752_1
        del arg753_1
        del arg754_1
        del arg755_1
        del arg756_1
        del arg757_1
        del arg758_1
        del arg759_1
        del arg75_1
        del arg760_1
        del arg761_1
        del arg762_1
        del arg763_1
        del arg764_1
        del arg765_1
        del arg766_1
        del arg767_1
        del arg768_1
        del arg769_1
        del arg76_1
        del arg770_1
        del arg771_1
        del arg772_1
        del arg773_1
        del arg774_1
        del arg775_1
        del arg776_1
        del arg777_1
        del arg778_1
        del arg779_1
        del arg77_1
        del arg780_1
        del arg781_1
        del arg782_1
        del arg783_1
        del arg784_1
        del arg785_1
        del arg786_1
        del arg787_1
        del arg788_1
        del arg789_1
        del arg78_1
        del arg790_1
        del arg791_1
        del arg792_1
        del arg793_1
        del arg794_1
        del arg795_1
        del arg796_1
        del arg797_1
        del arg798_1
        del arg799_1
        del arg79_1
        del arg7_1
        del arg800_1
        del arg801_1
        del arg802_1
        del arg803_1
        del arg804_1
        del arg80_1
        del arg81_1
        del arg82_1
        del arg83_1
        del arg84_1
        del arg85_1
        del arg86_1
        del arg87_1
        del arg88_1
        del arg89_1
        del arg8_1
        del arg90_1
        del arg91_1
        del arg92_1
        del arg93_1
        del arg94_1
        del arg95_1
        del arg96_1
        del arg97_1
        del arg98_1
        del arg99_1
        del arg9_1
    return ()


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((256, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((64, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((64, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((256, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((128, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((512, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((512, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((128, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((128, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((512, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((256, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((1024, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((2048, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((2048, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((64, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((256, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((64, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((64, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((256, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((128, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((512, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((512, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((128, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((128, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((512, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((256, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((1024, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((2048, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((2048, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((64, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((256, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((64, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((64, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((256, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((128, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((512, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((512, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((128, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((128, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg386_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg389_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg390_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg391_1 = rand_strided((512, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    arg392_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg393_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg394_1 = rand_strided((256, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg395_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg396_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg397_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg398_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg399_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg400_1 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg401_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg402_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg403_1 = rand_strided((1024, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg404_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg405_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg406_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg407_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg408_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg409_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg410_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg411_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg412_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg413_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg414_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg415_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg416_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg417_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg418_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg419_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg420_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg421_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg422_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg423_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg424_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg425_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg426_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg427_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg428_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg429_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg430_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg431_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg432_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg433_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg434_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg435_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg436_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg437_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg438_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg439_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg440_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg441_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg442_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg443_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg444_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg445_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg446_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg447_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg448_1 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg449_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg450_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg451_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg452_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg453_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg454_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg455_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg456_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg457_1 = rand_strided((2048, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg458_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg459_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg460_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg461_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg462_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg463_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda:0', dtype=torch.float32)
    arg464_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg465_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg466_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg467_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg468_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg469_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg470_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg471_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg472_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda:0', dtype=torch.float32)
    arg473_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg474_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg475_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg476_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg477_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg478_1 = rand_strided((2048, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg479_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg480_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg481_1 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg482_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg483_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg484_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg485_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg486_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg487_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg488_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg489_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg490_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg491_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg492_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg493_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg494_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg495_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg496_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg497_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg498_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg499_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg500_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg501_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg502_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg503_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg504_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg505_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg506_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg507_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg508_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg509_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg510_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg511_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg512_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg513_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg514_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg515_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg516_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg517_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg518_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg519_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg520_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg521_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg522_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg523_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg524_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg525_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg526_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg527_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg528_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg529_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg530_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg531_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg532_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg533_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg534_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg535_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg536_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg537_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg538_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg539_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg540_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg541_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg542_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg543_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg544_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg545_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg546_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg547_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg548_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg549_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg550_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg551_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg552_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg553_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg554_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg555_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg556_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg557_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg558_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg559_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg560_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg561_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg562_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg563_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg564_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg565_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg566_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg567_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg568_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg569_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg570_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg571_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg572_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg573_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg574_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg575_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg576_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg577_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg578_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg579_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg580_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg581_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg582_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg583_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg584_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg585_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg586_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg587_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg588_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg589_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg590_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg591_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg592_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg593_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg594_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg595_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg596_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg597_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg598_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg599_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg600_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg601_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg602_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg603_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg604_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg605_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg606_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg607_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg608_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg609_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg610_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg611_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg612_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg613_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg614_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg615_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg616_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg617_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg618_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg619_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg620_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg621_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg622_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg623_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg624_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg625_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg626_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg627_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg628_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg629_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg630_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg631_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg632_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg633_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg634_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg635_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg636_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg637_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg638_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg639_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg640_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg641_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg642_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg643_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg644_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg645_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg646_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg647_1 = rand_strided((64, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.float32)
    arg648_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg649_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg650_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg651_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg652_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg653_1 = rand_strided((256, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.float32)
    arg654_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg655_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg656_1 = rand_strided((256, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.float32)
    arg657_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg658_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg659_1 = rand_strided((64, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg660_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg661_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg662_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg663_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg664_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg665_1 = rand_strided((256, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.float32)
    arg666_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg667_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg668_1 = rand_strided((64, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg669_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg670_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg671_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg672_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg673_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg674_1 = rand_strided((256, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.float32)
    arg675_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg676_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg677_1 = rand_strided((128, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg678_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg679_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg680_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg681_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg682_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg683_1 = rand_strided((512, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    arg684_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg685_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg686_1 = rand_strided((512, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg687_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg688_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg689_1 = rand_strided((128, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg690_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg691_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg692_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg693_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg694_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg695_1 = rand_strided((512, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    arg696_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg697_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg698_1 = rand_strided((128, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg699_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg700_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg701_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg702_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg703_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg704_1 = rand_strided((512, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    arg705_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg706_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg707_1 = rand_strided((128, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg708_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg709_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg710_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg711_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg712_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg713_1 = rand_strided((512, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    arg714_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg715_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg716_1 = rand_strided((256, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg717_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg718_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg719_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg720_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg721_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg722_1 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg723_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg724_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg725_1 = rand_strided((1024, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg726_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg727_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg728_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg729_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg730_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg731_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg732_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg733_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg734_1 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg735_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg736_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg737_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg738_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg739_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg740_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg741_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg742_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg743_1 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg744_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg745_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg746_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg747_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg748_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg749_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg750_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg751_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg752_1 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg753_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg754_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg755_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg756_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg757_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg758_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg759_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg760_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg761_1 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg762_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg763_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg764_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg765_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg766_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg767_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg768_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg769_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg770_1 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg771_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg772_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg773_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg774_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg775_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg776_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg777_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg778_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg779_1 = rand_strided((2048, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg780_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg781_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg782_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg783_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg784_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg785_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda:0', dtype=torch.float32)
    arg786_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg787_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg788_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg789_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg790_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg791_1 = rand_strided((2048, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg792_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg793_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg794_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda:0', dtype=torch.float32)
    arg795_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg796_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg797_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg798_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg799_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg800_1 = rand_strided((2048, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg801_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg802_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg803_1 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg804_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1, arg728_1, arg729_1, arg730_1, arg731_1, arg732_1, arg733_1, arg734_1, arg735_1, arg736_1, arg737_1, arg738_1, arg739_1, arg740_1, arg741_1, arg742_1, arg743_1, arg744_1, arg745_1, arg746_1, arg747_1, arg748_1, arg749_1, arg750_1, arg751_1, arg752_1, arg753_1, arg754_1, arg755_1, arg756_1, arg757_1, arg758_1, arg759_1, arg760_1, arg761_1, arg762_1, arg763_1, arg764_1, arg765_1, arg766_1, arg767_1, arg768_1, arg769_1, arg770_1, arg771_1, arg772_1, arg773_1, arg774_1, arg775_1, arg776_1, arg777_1, arg778_1, arg779_1, arg780_1, arg781_1, arg782_1, arg783_1, arg784_1, arg785_1, arg786_1, arg787_1, arg788_1, arg789_1, arg790_1, arg791_1, arg792_1, arg793_1, arg794_1, arg795_1, arg796_1, arg797_1, arg798_1, arg799_1, arg800_1, arg801_1, arg802_1, arg803_1, arg804_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
