
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


# kernel path: /tmp/torchinductor_zhang402/4u/c4urco2hwt64z3rkugquekid7mnezz3gqesvkbofvm7crdp7sdba.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: '*fp32', 35: '*fp32', 36: '*fp32', 37: '*fp32', 38: '*fp32', 39: '*fp32', 40: '*fp32', 41: '*fp32', 42: '*fp32', 43: '*fp32', 44: '*fp32', 45: '*fp32', 46: '*fp32', 47: '*fp32', 48: '*fp32', 49: '*fp32', 50: '*fp32', 51: '*fp32', 52: '*fp32', 53: '*fp32', 54: '*fp32', 55: '*fp32', 56: '*fp32', 57: '*fp32', 58: '*fp32', 59: '*fp32', 60: '*fp32', 61: '*fp32', 62: '*fp32', 63: '*fp32', 64: '*fp32', 65: '*fp32', 66: '*fp32', 67: '*fp32', 68: '*fp32', 69: '*fp32', 70: '*fp32', 71: '*fp32', 72: '*fp32', 73: '*fp32', 74: '*fp32', 75: '*fp32', 76: '*fp32', 77: '*fp32', 78: '*fp32', 79: '*fp32', 80: '*fp32', 81: '*fp32', 82: '*fp32', 83: '*fp32', 84: '*fp32', 85: '*fp32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'kernel_name': 'triton_for_fused_1', 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, out_ptr13, out_ptr14, out_ptr15, out_ptr16, out_ptr17, out_ptr18, out_ptr19, out_ptr20, out_ptr21, out_ptr22, out_ptr23, out_ptr24, out_ptr25, out_ptr26, out_ptr27, out_ptr28, out_ptr29, out_ptr30, out_ptr31, out_ptr32, out_ptr33, out_ptr34, out_ptr35, out_ptr36, out_ptr37, out_ptr38, out_ptr39, out_ptr40, out_ptr41, out_ptr42):
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
    else:
        pass
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/6k/c6kvmcnnt3iqfn5svkds2ajpybk7orvasnpbcq3zjsw3j7c5yptv.py
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
    if xpid >= 0 and xpid < 1:
        xpid_offset = xpid - 0
        xnumel = 864
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
        xnumel = 32
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
    elif xpid >= 2 and xpid < 3:
        xpid_offset = xpid - 2
        xnumel = 32
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
    elif xpid >= 3 and xpid < 12:
        xpid_offset = xpid - 3
        xnumel = 9216
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
    elif xpid >= 12 and xpid < 13:
        xpid_offset = xpid - 12
        xnumel = 32
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
    elif xpid >= 13 and xpid < 14:
        xpid_offset = xpid - 13
        xnumel = 32
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
    elif xpid >= 14 and xpid < 32:
        xpid_offset = xpid - 14
        xnumel = 18432
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
    elif xpid >= 32 and xpid < 33:
        xpid_offset = xpid - 32
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
    elif xpid >= 33 and xpid < 34:
        xpid_offset = xpid - 33
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
    elif xpid >= 34 and xpid < 39:
        xpid_offset = xpid - 34
        xnumel = 5120
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
    elif xpid >= 39 and xpid < 40:
        xpid_offset = xpid - 39
        xnumel = 80
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
    elif xpid >= 40 and xpid < 41:
        xpid_offset = xpid - 40
        xnumel = 80
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
    elif xpid >= 41 and xpid < 176:
        xpid_offset = xpid - 41
        xnumel = 138240
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
    elif xpid >= 176 and xpid < 177:
        xpid_offset = xpid - 176
        xnumel = 192
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
    elif xpid >= 177 and xpid < 178:
        xpid_offset = xpid - 177
        xnumel = 192
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
    elif xpid >= 178 and xpid < 190:
        xpid_offset = xpid - 178
        xnumel = 12288
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
    elif xpid >= 190 and xpid < 191:
        xpid_offset = xpid - 190
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
    elif xpid >= 191 and xpid < 192:
        xpid_offset = xpid - 191
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
    elif xpid >= 192 and xpid < 201:
        xpid_offset = xpid - 192
        xnumel = 9216
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
    elif xpid >= 201 and xpid < 202:
        xpid_offset = xpid - 201
        xnumel = 48
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


# kernel path: /tmp/torchinductor_zhang402/2h/c2hpsokbnj2fdeg3t2d2w4blmwucm2ghejflge64oyfy4cls4qpv.py
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
        xnumel = 48
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
    elif xpid >= 1 and xpid < 76:
        xpid_offset = xpid - 1
        xnumel = 76800
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
    elif xpid >= 76 and xpid < 77:
        xpid_offset = xpid - 76
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
    elif xpid >= 77 and xpid < 78:
        xpid_offset = xpid - 77
        xnumel = 64
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
    elif xpid >= 78 and xpid < 90:
        xpid_offset = xpid - 78
        xnumel = 12288
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
    elif xpid >= 90 and xpid < 91:
        xpid_offset = xpid - 90
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
    elif xpid >= 91 and xpid < 92:
        xpid_offset = xpid - 91
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
    elif xpid >= 92 and xpid < 146:
        xpid_offset = xpid - 92
        xnumel = 55296
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
    elif xpid >= 146 and xpid < 147:
        xpid_offset = xpid - 146
        xnumel = 96
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
    elif xpid >= 147 and xpid < 148:
        xpid_offset = xpid - 147
        xnumel = 96
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
    elif xpid >= 148 and xpid < 229:
        xpid_offset = xpid - 148
        xnumel = 82944
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
    elif xpid >= 229 and xpid < 230:
        xpid_offset = xpid - 229
        xnumel = 96
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
    elif xpid >= 230 and xpid < 231:
        xpid_offset = xpid - 230
        xnumel = 96
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
    elif xpid >= 231 and xpid < 237:
        xpid_offset = xpid - 231
        xnumel = 6144
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
    elif xpid >= 237 and xpid < 238:
        xpid_offset = xpid - 237
        xnumel = 32
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
    elif xpid >= 238 and xpid < 239:
        xpid_offset = xpid - 238
        xnumel = 32
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
    elif xpid >= 239 and xpid < 255:
        xpid_offset = xpid - 239
        xnumel = 16384
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
    elif xpid >= 255 and xpid < 256:
        xpid_offset = xpid - 255
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
    elif xpid >= 256 and xpid < 257:
        xpid_offset = xpid - 256
        xnumel = 64
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
    elif xpid >= 257 and xpid < 269:
        xpid_offset = xpid - 257
        xnumel = 12288
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


# kernel path: /tmp/torchinductor_zhang402/c6/cc67dn3y5vuyd3gm647hqfjo6myvaukpolg6hyb24fxz7ftbcqo4.py
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
        xnumel = 48
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
        xnumel = 48
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
    elif xpid >= 2 and xpid < 77:
        xpid_offset = xpid - 2
        xnumel = 76800
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
    elif xpid >= 77 and xpid < 78:
        xpid_offset = xpid - 77
        xnumel = 64
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
    elif xpid >= 78 and xpid < 79:
        xpid_offset = xpid - 78
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
    elif xpid >= 79 and xpid < 95:
        xpid_offset = xpid - 79
        xnumel = 16384
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
    elif xpid >= 95 and xpid < 96:
        xpid_offset = xpid - 95
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
    elif xpid >= 96 and xpid < 97:
        xpid_offset = xpid - 96
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
    elif xpid >= 97 and xpid < 151:
        xpid_offset = xpid - 97
        xnumel = 55296
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
    elif xpid >= 151 and xpid < 152:
        xpid_offset = xpid - 151
        xnumel = 96
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
    elif xpid >= 152 and xpid < 153:
        xpid_offset = xpid - 152
        xnumel = 96
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
    elif xpid >= 153 and xpid < 234:
        xpid_offset = xpid - 153
        xnumel = 82944
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
    elif xpid >= 234 and xpid < 235:
        xpid_offset = xpid - 234
        xnumel = 96
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
    elif xpid >= 235 and xpid < 236:
        xpid_offset = xpid - 235
        xnumel = 96
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
    elif xpid >= 236 and xpid < 252:
        xpid_offset = xpid - 236
        xnumel = 16384
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
    elif xpid >= 252 and xpid < 253:
        xpid_offset = xpid - 252
        xnumel = 64
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
    elif xpid >= 253 and xpid < 254:
        xpid_offset = xpid - 253
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
    elif xpid >= 254 and xpid < 272:
        xpid_offset = xpid - 254
        xnumel = 18432
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
    elif xpid >= 272 and xpid < 273:
        xpid_offset = xpid - 272
        xnumel = 64
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
    elif xpid >= 273 and xpid < 274:
        xpid_offset = xpid - 273
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


# kernel path: /tmp/torchinductor_zhang402/lq/clq5b5lu2thvrpin2k4yvdpl46jd3tgusu4ysdo63u6bmgmpt3ve.py
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
    if xpid >= 0 and xpid < 14:
        xpid_offset = xpid - 0
        xnumel = 13824
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
    elif xpid >= 14 and xpid < 15:
        xpid_offset = xpid - 14
        xnumel = 48
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
    elif xpid >= 15 and xpid < 16:
        xpid_offset = xpid - 15
        xnumel = 48
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
    elif xpid >= 16 and xpid < 91:
        xpid_offset = xpid - 16
        xnumel = 76800
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
    elif xpid >= 91 and xpid < 92:
        xpid_offset = xpid - 91
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
    elif xpid >= 92 and xpid < 93:
        xpid_offset = xpid - 92
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
    elif xpid >= 93 and xpid < 111:
        xpid_offset = xpid - 93
        xnumel = 18432
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
    elif xpid >= 111 and xpid < 112:
        xpid_offset = xpid - 111
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
    elif xpid >= 112 and xpid < 113:
        xpid_offset = xpid - 112
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
    elif xpid >= 113 and xpid < 167:
        xpid_offset = xpid - 113
        xnumel = 55296
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
    elif xpid >= 167 and xpid < 168:
        xpid_offset = xpid - 167
        xnumel = 96
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
    elif xpid >= 168 and xpid < 169:
        xpid_offset = xpid - 168
        xnumel = 96
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
    elif xpid >= 169 and xpid < 250:
        xpid_offset = xpid - 169
        xnumel = 82944
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
    elif xpid >= 250 and xpid < 251:
        xpid_offset = xpid - 250
        xnumel = 96
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
    elif xpid >= 251 and xpid < 252:
        xpid_offset = xpid - 251
        xnumel = 96
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
    elif xpid >= 252 and xpid < 270:
        xpid_offset = xpid - 252
        xnumel = 18432
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
    elif xpid >= 270 and xpid < 271:
        xpid_offset = xpid - 270
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
    elif xpid >= 271 and xpid < 272:
        xpid_offset = xpid - 271
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
    elif xpid >= 272 and xpid < 1244:
        xpid_offset = xpid - 272
        xnumel = 995328
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
    elif xpid >= 1244 and xpid < 1245:
        xpid_offset = xpid - 1244
        xnumel = 384
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


# kernel path: /tmp/torchinductor_zhang402/xx/cxxdhr3j3r5ipvutkvdu75t3uzap5od6c6j5kud3zjj5paehp26y.py
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
        xnumel = 384
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
    elif xpid >= 1 and xpid < 19:
        xpid_offset = xpid - 1
        xnumel = 18432
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
    elif xpid >= 19 and xpid < 20:
        xpid_offset = xpid - 19
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
    elif xpid >= 20 and xpid < 21:
        xpid_offset = xpid - 20
        xnumel = 64
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
    elif xpid >= 21 and xpid < 75:
        xpid_offset = xpid - 21
        xnumel = 55296
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
    elif xpid >= 75 and xpid < 76:
        xpid_offset = xpid - 75
        xnumel = 96
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
    elif xpid >= 76 and xpid < 77:
        xpid_offset = xpid - 76
        xnumel = 96
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
    elif xpid >= 77 and xpid < 158:
        xpid_offset = xpid - 77
        xnumel = 82944
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
    elif xpid >= 158 and xpid < 159:
        xpid_offset = xpid - 158
        xnumel = 96
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
    elif xpid >= 159 and xpid < 160:
        xpid_offset = xpid - 159
        xnumel = 96
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
    elif xpid >= 160 and xpid < 304:
        xpid_offset = xpid - 160
        xnumel = 147456
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
    elif xpid >= 304 and xpid < 305:
        xpid_offset = xpid - 304
        xnumel = 192
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
    elif xpid >= 305 and xpid < 306:
        xpid_offset = xpid - 305
        xnumel = 192
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
    elif xpid >= 306 and xpid < 402:
        xpid_offset = xpid - 306
        xnumel = 98304
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
    elif xpid >= 402 and xpid < 403:
        xpid_offset = xpid - 402
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
    elif xpid >= 403 and xpid < 404:
        xpid_offset = xpid - 403
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
    elif xpid >= 404 and xpid < 516:
        xpid_offset = xpid - 404
        xnumel = 114688
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
    elif xpid >= 516 and xpid < 517:
        xpid_offset = xpid - 516
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
    elif xpid >= 517 and xpid < 518:
        xpid_offset = xpid - 517
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
    elif xpid >= 518 and xpid < 686:
        xpid_offset = xpid - 518
        xnumel = 172032
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


# kernel path: /tmp/torchinductor_zhang402/am/cam7hnt5k4nyiu3bb7qkacigwou2cs2qpunpvyj7v53towryh2wt.py
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
        xnumel = 192
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
        xnumel = 192
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
    elif xpid >= 2 and xpid < 98:
        xpid_offset = xpid - 2
        xnumel = 98304
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
    elif xpid >= 98 and xpid < 99:
        xpid_offset = xpid - 98
        xnumel = 128
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
    elif xpid >= 99 and xpid < 100:
        xpid_offset = xpid - 99
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
    elif xpid >= 100 and xpid < 212:
        xpid_offset = xpid - 100
        xnumel = 114688
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
    elif xpid >= 212 and xpid < 213:
        xpid_offset = xpid - 212
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
    elif xpid >= 213 and xpid < 214:
        xpid_offset = xpid - 213
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
    elif xpid >= 214 and xpid < 326:
        xpid_offset = xpid - 214
        xnumel = 114688
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
    elif xpid >= 326 and xpid < 327:
        xpid_offset = xpid - 326
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
    elif xpid >= 327 and xpid < 328:
        xpid_offset = xpid - 327
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
    elif xpid >= 328 and xpid < 440:
        xpid_offset = xpid - 328
        xnumel = 114688
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
    elif xpid >= 440 and xpid < 441:
        xpid_offset = xpid - 440
        xnumel = 128
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
    elif xpid >= 441 and xpid < 442:
        xpid_offset = xpid - 441
        xnumel = 128
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
    elif xpid >= 442 and xpid < 610:
        xpid_offset = xpid - 442
        xnumel = 172032
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
    elif xpid >= 610 and xpid < 611:
        xpid_offset = xpid - 610
        xnumel = 192
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
    elif xpid >= 611 and xpid < 612:
        xpid_offset = xpid - 611
        xnumel = 192
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
    elif xpid >= 612 and xpid < 756:
        xpid_offset = xpid - 612
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
    elif xpid >= 756 and xpid < 757:
        xpid_offset = xpid - 756
        xnumel = 192
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
    elif xpid >= 757 and xpid < 758:
        xpid_offset = xpid - 757
        xnumel = 192
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


# kernel path: /tmp/torchinductor_zhang402/22/c22piiuxk4bnw7nx2jz2mv426qozv3ldsxnd6hjalb43wtn3iuk7.py
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
    if xpid >= 0 and xpid < 144:
        xpid_offset = xpid - 0
        xnumel = 147456
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
    elif xpid >= 144 and xpid < 145:
        xpid_offset = xpid - 144
        xnumel = 192
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
    elif xpid >= 145 and xpid < 146:
        xpid_offset = xpid - 145
        xnumel = 192
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
    elif xpid >= 146 and xpid < 266:
        xpid_offset = xpid - 146
        xnumel = 122880
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
    elif xpid >= 266 and xpid < 267:
        xpid_offset = xpid - 266
        xnumel = 160
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
    elif xpid >= 267 and xpid < 268:
        xpid_offset = xpid - 267
        xnumel = 160
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
    elif xpid >= 268 and xpid < 443:
        xpid_offset = xpid - 268
        xnumel = 179200
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
    elif xpid >= 443 and xpid < 444:
        xpid_offset = xpid - 443
        xnumel = 160
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
    elif xpid >= 444 and xpid < 445:
        xpid_offset = xpid - 444
        xnumel = 160
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
    elif xpid >= 445 and xpid < 655:
        xpid_offset = xpid - 445
        xnumel = 215040
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
    elif xpid >= 655 and xpid < 656:
        xpid_offset = xpid - 655
        xnumel = 192
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
    elif xpid >= 656 and xpid < 657:
        xpid_offset = xpid - 656
        xnumel = 192
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
    elif xpid >= 657 and xpid < 777:
        xpid_offset = xpid - 657
        xnumel = 122880
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
    elif xpid >= 777 and xpid < 778:
        xpid_offset = xpid - 777
        xnumel = 160
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
    elif xpid >= 778 and xpid < 779:
        xpid_offset = xpid - 778
        xnumel = 160
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
    elif xpid >= 779 and xpid < 954:
        xpid_offset = xpid - 779
        xnumel = 179200
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
    elif xpid >= 954 and xpid < 955:
        xpid_offset = xpid - 954
        xnumel = 160
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
    elif xpid >= 955 and xpid < 956:
        xpid_offset = xpid - 955
        xnumel = 160
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
    elif xpid >= 956 and xpid < 1131:
        xpid_offset = xpid - 956
        xnumel = 179200
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
    elif xpid >= 1131 and xpid < 1132:
        xpid_offset = xpid - 1131
        xnumel = 160
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


# kernel path: /tmp/torchinductor_zhang402/j4/cj4xm4pfbdvy737353tadutbend7nglvvlyt6e4jv6xsenjgd7ss.py
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
    if xpid >= 0 and xpid < 1:
        xpid_offset = xpid - 0
        xnumel = 160
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
    elif xpid >= 1 and xpid < 176:
        xpid_offset = xpid - 1
        xnumel = 179200
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
    elif xpid >= 176 and xpid < 177:
        xpid_offset = xpid - 176
        xnumel = 160
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
    elif xpid >= 177 and xpid < 178:
        xpid_offset = xpid - 177
        xnumel = 160
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
    elif xpid >= 178 and xpid < 388:
        xpid_offset = xpid - 178
        xnumel = 215040
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
    elif xpid >= 388 and xpid < 389:
        xpid_offset = xpid - 388
        xnumel = 192
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
    elif xpid >= 389 and xpid < 390:
        xpid_offset = xpid - 389
        xnumel = 192
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
    elif xpid >= 390 and xpid < 534:
        xpid_offset = xpid - 390
        xnumel = 147456
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
    elif xpid >= 534 and xpid < 535:
        xpid_offset = xpid - 534
        xnumel = 192
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
    elif xpid >= 535 and xpid < 536:
        xpid_offset = xpid - 535
        xnumel = 192
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
    elif xpid >= 536 and xpid < 680:
        xpid_offset = xpid - 536
        xnumel = 147456
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
    elif xpid >= 680 and xpid < 681:
        xpid_offset = xpid - 680
        xnumel = 192
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
    elif xpid >= 681 and xpid < 682:
        xpid_offset = xpid - 681
        xnumel = 192
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
    elif xpid >= 682 and xpid < 802:
        xpid_offset = xpid - 682
        xnumel = 122880
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
    elif xpid >= 802 and xpid < 803:
        xpid_offset = xpid - 802
        xnumel = 160
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
    elif xpid >= 803 and xpid < 804:
        xpid_offset = xpid - 803
        xnumel = 160
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
    elif xpid >= 804 and xpid < 979:
        xpid_offset = xpid - 804
        xnumel = 179200
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
    elif xpid >= 979 and xpid < 980:
        xpid_offset = xpid - 979
        xnumel = 160
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
    elif xpid >= 980 and xpid < 981:
        xpid_offset = xpid - 980
        xnumel = 160
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
    elif xpid >= 981 and xpid < 1191:
        xpid_offset = xpid - 981
        xnumel = 215040
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


# kernel path: /tmp/torchinductor_zhang402/oc/cocnewetpftv32uxm5x3x6cw5p4e6ssv3gwnva55ehxfwgcrd3qz.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: '*fp32', 35: '*fp32', 36: '*fp32', 37: '*fp32', 38: '*fp32', 39: '*fp32', 40: '*fp32', 41: '*fp32', 42: '*fp32', 43: '*fp32', 44: '*fp32', 45: '*fp32', 46: '*fp32', 47: '*fp32', 48: '*fp32', 49: '*fp32', 50: '*fp32', 51: '*fp32', 52: '*fp32', 53: '*fp32', 54: '*fp32', 55: '*fp32', 56: '*fp32', 57: '*fp32', 58: '*fp32', 59: '*fp32', 60: '*fp32', 61: '*fp32', 62: '*fp32', 63: '*fp32', 64: '*fp32', 65: '*fp32', 66: '*fp32', 67: '*fp32', 68: '*fp32', 69: '*fp32', 70: '*fp32', 71: '*fp32', 72: '*fp32', 73: '*fp32', 74: '*fp32', 75: '*fp32', 76: '*fp32', 77: '*fp32', 78: '*fp32', 79: '*fp32', 80: '*fp32', 81: '*fp32', 82: '*fp32', 83: '*fp32', 84: '*fp32', 85: '*fp32', 86: '*fp32', 87: '*fp32', 88: '*fp32', 89: '*fp32', 90: '*fp32', 91: '*fp32', 92: '*fp32', 93: '*fp32', 94: '*fp32', 95: '*fp32', 96: '*fp32', 97: '*fp32', 98: '*fp32', 99: '*fp32', 100: '*fp32', 101: '*fp32', 102: '*fp32', 103: '*fp32', 104: '*fp32', 105: '*fp32', 106: '*fp32', 107: '*fp32', 108: '*fp32', 109: '*fp32', 110: '*fp32', 111: '*fp32', 112: '*fp32', 113: '*fp32', 114: '*fp32', 115: '*fp32', 116: '*fp32', 117: '*fp32', 118: '*fp32', 119: '*fp32', 120: '*fp32', 121: '*fp32', 122: '*fp32', 123: '*fp32', 124: '*fp32', 125: '*fp32', 126: '*fp32', 127: '*fp32', 128: '*fp32', 129: '*fp32', 130: '*fp32', 131: '*fp32', 132: '*fp32', 133: '*fp32', 134: '*fp32', 135: '*fp32', 136: '*fp32', 137: '*fp32', 138: '*fp32', 139: '*fp32', 140: '*fp32', 141: '*fp32', 142: '*fp32', 143: '*fp32', 144: '*fp32', 145: '*fp32', 146: '*fp32', 147: '*fp32', 148: '*fp32', 149: '*fp32', 150: '*fp32', 151: '*fp32', 152: '*fp32', 153: '*fp32', 154: '*fp32', 155: '*fp32', 156: '*fp32', 157: '*fp32', 158: '*fp32', 159: '*fp32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'kernel_name': 'triton_for_fused_10', 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, in_ptr47, in_ptr48, in_ptr49, in_ptr50, in_ptr51, in_ptr52, in_ptr53, in_ptr54, in_ptr55, in_ptr56, in_ptr57, in_ptr58, in_ptr59, in_ptr60, in_ptr61, in_ptr62, in_ptr63, in_ptr64, in_ptr65, in_ptr66, in_ptr67, in_ptr68, in_ptr69, in_ptr70, in_ptr71, in_ptr72, in_ptr73, in_ptr74, in_ptr75, in_ptr76, in_ptr77, in_ptr78, in_ptr79, in_ptr80, in_ptr81, in_ptr82, in_ptr83, in_ptr84, in_ptr85, in_ptr86, in_ptr87, in_ptr88, in_ptr89, in_ptr90, in_ptr91, in_ptr92, in_ptr93, in_ptr94, in_ptr95, in_ptr96, in_ptr97, in_ptr98, in_ptr99, out_ptr0, out_ptr2, out_ptr3, out_ptr4, out_ptr6, out_ptr7, out_ptr8, out_ptr10, out_ptr11, out_ptr12, out_ptr14, out_ptr15, out_ptr16, out_ptr18, out_ptr19, out_ptr20, out_ptr22, out_ptr23, out_ptr24, out_ptr26, out_ptr27, out_ptr28, out_ptr30, out_ptr31, out_ptr32, out_ptr34, out_ptr35, out_ptr36, out_ptr38, out_ptr39, out_ptr40, out_ptr42, out_ptr43, out_ptr44, out_ptr46, out_ptr47, out_ptr48, out_ptr50, out_ptr51, out_ptr52, out_ptr54, out_ptr55, out_ptr56, out_ptr58, out_ptr59, out_ptr60, out_ptr62, out_ptr63, out_ptr64, out_ptr66, out_ptr67, out_ptr68, out_ptr70, out_ptr71, out_ptr72, out_ptr74, out_ptr75, out_ptr76, out_ptr78, out_ptr79):
    xpid = tl.program_id(0)
    XBLOCK: tl.constexpr = 1024
    if xpid >= 0 and xpid < 1:
        xpid_offset = xpid - 0
        xnumel = 192
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
        xnumel = 192
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
    elif xpid >= 2 and xpid < 122:
        xpid_offset = xpid - 2
        xnumel = 122880
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
    elif xpid >= 122 and xpid < 123:
        xpid_offset = xpid - 122
        xnumel = 160
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
    elif xpid >= 123 and xpid < 124:
        xpid_offset = xpid - 123
        xnumel = 160
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
    elif xpid >= 124 and xpid < 299:
        xpid_offset = xpid - 124
        xnumel = 179200
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
    elif xpid >= 299 and xpid < 300:
        xpid_offset = xpid - 299
        xnumel = 160
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
    elif xpid >= 300 and xpid < 301:
        xpid_offset = xpid - 300
        xnumel = 160
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
    elif xpid >= 301 and xpid < 476:
        xpid_offset = xpid - 301
        xnumel = 179200
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
    elif xpid >= 476 and xpid < 477:
        xpid_offset = xpid - 476
        xnumel = 160
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
    elif xpid >= 477 and xpid < 478:
        xpid_offset = xpid - 477
        xnumel = 160
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
    elif xpid >= 478 and xpid < 653:
        xpid_offset = xpid - 478
        xnumel = 179200
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
    elif xpid >= 653 and xpid < 654:
        xpid_offset = xpid - 653
        xnumel = 160
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
    elif xpid >= 654 and xpid < 655:
        xpid_offset = xpid - 654
        xnumel = 160
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
    elif xpid >= 655 and xpid < 865:
        xpid_offset = xpid - 655
        xnumel = 215040
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
    elif xpid >= 865 and xpid < 866:
        xpid_offset = xpid - 865
        xnumel = 192
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
    elif xpid >= 866 and xpid < 867:
        xpid_offset = xpid - 866
        xnumel = 192
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
    elif xpid >= 867 and xpid < 1011:
        xpid_offset = xpid - 867
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
    elif xpid >= 1011 and xpid < 1012:
        xpid_offset = xpid - 1011
        xnumel = 192
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
    elif xpid >= 1012 and xpid < 1013:
        xpid_offset = xpid - 1012
        xnumel = 192
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


# kernel path: /tmp/torchinductor_zhang402/6y/c6y6bsy6wxvgdmaiwzolwxmedg4r7h4qui7ima3l42ksbuidhv53.py
# Source Nodes: [], Original ATen: []

triton_for_fused_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'kernel_name': 'triton_for_fused_11', 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, in_ptr47, in_ptr48, in_ptr49, in_ptr50, in_ptr51, in_ptr52, in_ptr53, in_ptr54, in_ptr55, in_ptr56, in_ptr57, in_ptr58, in_ptr59, in_ptr60, in_ptr61, in_ptr62, in_ptr63, in_ptr64, in_ptr65, in_ptr66, in_ptr67, in_ptr68, in_ptr69, in_ptr70, in_ptr71, in_ptr72, in_ptr73, in_ptr74, in_ptr75, in_ptr76, in_ptr77, in_ptr78, in_ptr79, in_ptr80, in_ptr81, in_ptr82, in_ptr83, in_ptr84, in_ptr85, in_ptr86, in_ptr87, in_ptr88, in_ptr89, in_ptr90, in_ptr91, in_ptr92, in_ptr93, in_ptr94, in_ptr95, in_ptr96, in_ptr97, in_ptr98, in_ptr99, out_ptr0, out_ptr2, out_ptr3, out_ptr4, out_ptr6, out_ptr7, out_ptr8, out_ptr10, out_ptr11, out_ptr12, out_ptr14, out_ptr15, out_ptr16, out_ptr18, out_ptr19, out_ptr20, out_ptr22, out_ptr23, out_ptr24, out_ptr26, out_ptr27, out_ptr28, out_ptr30, out_ptr31, out_ptr32, out_ptr34, out_ptr35, out_ptr36, out_ptr38, out_ptr39, out_ptr40, out_ptr42, out_ptr43, out_ptr44, out_ptr46, out_ptr47, out_ptr48, out_ptr50, out_ptr51, out_ptr52, out_ptr54, out_ptr55, out_ptr56, out_ptr58, out_ptr59, out_ptr60, out_ptr62, out_ptr63, out_ptr64, out_ptr66, out_ptr67, out_ptr68, out_ptr70, out_ptr71, out_ptr72, out_ptr74, out_ptr75, out_ptr76, out_ptr78, out_ptr79):
    xpid = tl.program_id(0)
    XBLOCK: tl.constexpr = 1024
    if xpid >= 0 and xpid < 144:
        xpid_offset = xpid - 0
        xnumel = 147456
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
    elif xpid >= 144 and xpid < 145:
        xpid_offset = xpid - 144
        xnumel = 192
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
    elif xpid >= 145 and xpid < 146:
        xpid_offset = xpid - 145
        xnumel = 192
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
    elif xpid >= 146 and xpid < 290:
        xpid_offset = xpid - 146
        xnumel = 147456
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
    elif xpid >= 290 and xpid < 291:
        xpid_offset = xpid - 290
        xnumel = 192
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
    elif xpid >= 291 and xpid < 292:
        xpid_offset = xpid - 291
        xnumel = 192
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
    elif xpid >= 292 and xpid < 544:
        xpid_offset = xpid - 292
        xnumel = 258048
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
    elif xpid >= 544 and xpid < 545:
        xpid_offset = xpid - 544
        xnumel = 192
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
    elif xpid >= 545 and xpid < 546:
        xpid_offset = xpid - 545
        xnumel = 192
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
    elif xpid >= 546 and xpid < 798:
        xpid_offset = xpid - 546
        xnumel = 258048
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
    elif xpid >= 798 and xpid < 799:
        xpid_offset = xpid - 798
        xnumel = 192
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
    elif xpid >= 799 and xpid < 800:
        xpid_offset = xpid - 799
        xnumel = 192
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
    elif xpid >= 800 and xpid < 944:
        xpid_offset = xpid - 800
        xnumel = 147456
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
    elif xpid >= 944 and xpid < 945:
        xpid_offset = xpid - 944
        xnumel = 192
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
    elif xpid >= 945 and xpid < 946:
        xpid_offset = xpid - 945
        xnumel = 192
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
    elif xpid >= 946 and xpid < 1198:
        xpid_offset = xpid - 946
        xnumel = 258048
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
    elif xpid >= 1198 and xpid < 1199:
        xpid_offset = xpid - 1198
        xnumel = 192
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
    elif xpid >= 1199 and xpid < 1200:
        xpid_offset = xpid - 1199
        xnumel = 192
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
    elif xpid >= 1200 and xpid < 1452:
        xpid_offset = xpid - 1200
        xnumel = 258048
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
    elif xpid >= 1452 and xpid < 1453:
        xpid_offset = xpid - 1452
        xnumel = 192
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


# kernel path: /tmp/torchinductor_zhang402/rw/crwo2w3risk6666adiwtm66b3nagjrxmcnxeb3b6e2xr3lyzuye6.py
# Source Nodes: [], Original ATen: []

triton_for_fused_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'kernel_name': 'triton_for_fused_12', 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, in_ptr47, in_ptr48, in_ptr49, in_ptr50, in_ptr51, in_ptr52, in_ptr53, in_ptr54, in_ptr55, in_ptr56, in_ptr57, in_ptr58, in_ptr59, in_ptr60, in_ptr61, in_ptr62, in_ptr63, in_ptr64, in_ptr65, in_ptr66, in_ptr67, in_ptr68, in_ptr69, in_ptr70, in_ptr71, in_ptr72, in_ptr73, in_ptr74, in_ptr75, in_ptr76, in_ptr77, in_ptr78, in_ptr79, in_ptr80, in_ptr81, in_ptr82, in_ptr83, in_ptr84, in_ptr85, in_ptr86, in_ptr87, in_ptr88, in_ptr89, in_ptr90, in_ptr91, in_ptr92, in_ptr93, in_ptr94, in_ptr95, in_ptr96, in_ptr97, in_ptr98, in_ptr99, out_ptr0, out_ptr2, out_ptr3, out_ptr4, out_ptr6, out_ptr7, out_ptr8, out_ptr10, out_ptr11, out_ptr12, out_ptr14, out_ptr15, out_ptr16, out_ptr18, out_ptr19, out_ptr20, out_ptr22, out_ptr23, out_ptr24, out_ptr26, out_ptr27, out_ptr28, out_ptr30, out_ptr31, out_ptr32, out_ptr34, out_ptr35, out_ptr36, out_ptr38, out_ptr39, out_ptr40, out_ptr42, out_ptr43, out_ptr44, out_ptr46, out_ptr47, out_ptr48, out_ptr50, out_ptr51, out_ptr52, out_ptr54, out_ptr55, out_ptr56, out_ptr58, out_ptr59, out_ptr60, out_ptr62, out_ptr63, out_ptr64, out_ptr66, out_ptr67, out_ptr68, out_ptr70, out_ptr71, out_ptr72, out_ptr74, out_ptr75, out_ptr76, out_ptr78, out_ptr79):
    xpid = tl.program_id(0)
    XBLOCK: tl.constexpr = 1024
    if xpid >= 0 and xpid < 1:
        xpid_offset = xpid - 0
        xnumel = 192
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
    elif xpid >= 1 and xpid < 253:
        xpid_offset = xpid - 1
        xnumel = 258048
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
    elif xpid >= 253 and xpid < 254:
        xpid_offset = xpid - 253
        xnumel = 192
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
    elif xpid >= 254 and xpid < 255:
        xpid_offset = xpid - 254
        xnumel = 192
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
    elif xpid >= 255 and xpid < 507:
        xpid_offset = xpid - 255
        xnumel = 258048
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
    elif xpid >= 507 and xpid < 508:
        xpid_offset = xpid - 507
        xnumel = 192
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
    elif xpid >= 508 and xpid < 509:
        xpid_offset = xpid - 508
        xnumel = 192
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
    elif xpid >= 509 and xpid < 653:
        xpid_offset = xpid - 509
        xnumel = 147456
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
    elif xpid >= 653 and xpid < 654:
        xpid_offset = xpid - 653
        xnumel = 192
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
    elif xpid >= 654 and xpid < 655:
        xpid_offset = xpid - 654
        xnumel = 192
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
    elif xpid >= 655 and xpid < 751:
        xpid_offset = xpid - 655
        xnumel = 98304
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
    elif xpid >= 751 and xpid < 752:
        xpid_offset = xpid - 751
        xnumel = 128
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
    elif xpid >= 752 and xpid < 753:
        xpid_offset = xpid - 752
        xnumel = 128
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
    elif xpid >= 753 and xpid < 3153:
        xpid_offset = xpid - 753
        xnumel = 2457600
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
    elif xpid >= 3153 and xpid < 3154:
        xpid_offset = xpid - 3153
        xnumel = 768
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
    elif xpid >= 3154 and xpid < 3155:
        xpid_offset = xpid - 3154
        xnumel = 768
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
    elif xpid >= 3155 and xpid < 3905:
        xpid_offset = xpid - 3155
        xnumel = 768000
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
    elif xpid >= 3905 and xpid < 3906:
        xpid_offset = xpid - 3905
        xnumel = 1000
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
    elif xpid >= 3906 and xpid < 4050:
        xpid_offset = xpid - 3906
        xnumel = 147456
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
    elif xpid >= 4050 and xpid < 4051:
        xpid_offset = xpid - 4050
        xnumel = 192
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


# kernel path: /tmp/torchinductor_zhang402/vd/cvducu4i2pf66rl5ywhipmepqxt6altsyjbtj4jwfk6otu5utgib.py
# Source Nodes: [], Original ATen: []

triton_for_fused_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'kernel_name': 'triton_for_fused_13', 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, in_ptr47, in_ptr48, in_ptr49, in_ptr50, in_ptr51, in_ptr52, in_ptr53, in_ptr54, in_ptr55, in_ptr56, in_ptr57, in_ptr58, in_ptr59, in_ptr60, in_ptr61, in_ptr62, in_ptr63, in_ptr64, in_ptr65, in_ptr66, in_ptr67, in_ptr68, in_ptr69, in_ptr70, in_ptr71, in_ptr72, in_ptr73, in_ptr74, in_ptr75, in_ptr76, in_ptr77, in_ptr78, in_ptr79, in_ptr80, in_ptr81, in_ptr82, in_ptr83, in_ptr84, in_ptr85, in_ptr86, in_ptr87, in_ptr88, in_ptr89, in_ptr90, in_ptr91, in_ptr92, in_ptr93, in_ptr94, in_ptr95, in_ptr96, in_ptr97, in_ptr98, in_ptr99, out_ptr0, out_ptr2, out_ptr3, out_ptr4, out_ptr6, out_ptr7, out_ptr8, out_ptr10, out_ptr11, out_ptr12, out_ptr14, out_ptr15, out_ptr16, out_ptr18, out_ptr19, out_ptr20, out_ptr22, out_ptr23, out_ptr24, out_ptr26, out_ptr27, out_ptr28, out_ptr30, out_ptr31, out_ptr32, out_ptr34, out_ptr35, out_ptr36, out_ptr38, out_ptr39, out_ptr40, out_ptr42, out_ptr43, out_ptr44, out_ptr46, out_ptr47, out_ptr48, out_ptr50, out_ptr51, out_ptr52, out_ptr54, out_ptr55, out_ptr56, out_ptr58, out_ptr59, out_ptr60, out_ptr62, out_ptr63, out_ptr64, out_ptr66, out_ptr67, out_ptr68, out_ptr70, out_ptr71, out_ptr72, out_ptr74, out_ptr75, out_ptr76, out_ptr78, out_ptr79):
    xpid = tl.program_id(0)
    XBLOCK: tl.constexpr = 1024
    if xpid >= 0 and xpid < 1:
        xpid_offset = xpid - 0
        xnumel = 192
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
    elif xpid >= 1 and xpid < 541:
        xpid_offset = xpid - 1
        xnumel = 552960
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
    elif xpid >= 541 and xpid < 542:
        xpid_offset = xpid - 541
        xnumel = 320
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
    elif xpid >= 542 and xpid < 543:
        xpid_offset = xpid - 542
        xnumel = 320
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
    elif xpid >= 543 and xpid < 687:
        xpid_offset = xpid - 543
        xnumel = 147456
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
    elif xpid >= 687 and xpid < 688:
        xpid_offset = xpid - 687
        xnumel = 192
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
    elif xpid >= 688 and xpid < 689:
        xpid_offset = xpid - 688
        xnumel = 192
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
    elif xpid >= 689 and xpid < 941:
        xpid_offset = xpid - 689
        xnumel = 258048
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
    elif xpid >= 941 and xpid < 942:
        xpid_offset = xpid - 941
        xnumel = 192
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
    elif xpid >= 942 and xpid < 943:
        xpid_offset = xpid - 942
        xnumel = 192
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
    elif xpid >= 943 and xpid < 1195:
        xpid_offset = xpid - 943
        xnumel = 258048
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
    elif xpid >= 1195 and xpid < 1196:
        xpid_offset = xpid - 1195
        xnumel = 192
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
    elif xpid >= 1196 and xpid < 1197:
        xpid_offset = xpid - 1196
        xnumel = 192
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
    elif xpid >= 1197 and xpid < 1521:
        xpid_offset = xpid - 1197
        xnumel = 331776
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
    elif xpid >= 1521 and xpid < 1522:
        xpid_offset = xpid - 1521
        xnumel = 192
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
    elif xpid >= 1522 and xpid < 1523:
        xpid_offset = xpid - 1522
        xnumel = 192
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
    elif xpid >= 1523 and xpid < 1923:
        xpid_offset = xpid - 1523
        xnumel = 409600
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
    elif xpid >= 1923 and xpid < 1924:
        xpid_offset = xpid - 1923
        xnumel = 320
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
    elif xpid >= 1924 and xpid < 1925:
        xpid_offset = xpid - 1924
        xnumel = 320
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
    elif xpid >= 1925 and xpid < 2405:
        xpid_offset = xpid - 1925
        xnumel = 491520
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


# kernel path: /tmp/torchinductor_zhang402/2b/c2bjehgbr55j6sd2mmnclidh7haijn56s6oy7jqqtgxivqcccpo2.py
# Source Nodes: [], Original ATen: []

triton_for_fused_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'kernel_name': 'triton_for_fused_14', 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, in_ptr47, in_ptr48, in_ptr49, in_ptr50, in_ptr51, in_ptr52, in_ptr53, in_ptr54, in_ptr55, in_ptr56, in_ptr57, in_ptr58, in_ptr59, in_ptr60, in_ptr61, in_ptr62, in_ptr63, in_ptr64, in_ptr65, in_ptr66, in_ptr67, in_ptr68, in_ptr69, in_ptr70, in_ptr71, in_ptr72, in_ptr73, in_ptr74, in_ptr75, in_ptr76, in_ptr77, in_ptr78, in_ptr79, in_ptr80, in_ptr81, in_ptr82, in_ptr83, in_ptr84, in_ptr85, in_ptr86, in_ptr87, in_ptr88, in_ptr89, in_ptr90, in_ptr91, in_ptr92, in_ptr93, in_ptr94, in_ptr95, in_ptr96, in_ptr97, in_ptr98, in_ptr99, out_ptr0, out_ptr2, out_ptr3, out_ptr4, out_ptr6, out_ptr7, out_ptr8, out_ptr10, out_ptr11, out_ptr12, out_ptr14, out_ptr15, out_ptr16, out_ptr18, out_ptr19, out_ptr20, out_ptr22, out_ptr23, out_ptr24, out_ptr26, out_ptr27, out_ptr28, out_ptr30, out_ptr31, out_ptr32, out_ptr34, out_ptr35, out_ptr36, out_ptr38, out_ptr39, out_ptr40, out_ptr42, out_ptr43, out_ptr44, out_ptr46, out_ptr47, out_ptr48, out_ptr50, out_ptr51, out_ptr52, out_ptr54, out_ptr55, out_ptr56, out_ptr58, out_ptr59, out_ptr60, out_ptr62, out_ptr63, out_ptr64, out_ptr66, out_ptr67, out_ptr68, out_ptr70, out_ptr71, out_ptr72, out_ptr74, out_ptr75, out_ptr76, out_ptr78, out_ptr79):
    xpid = tl.program_id(0)
    XBLOCK: tl.constexpr = 1024
    if xpid >= 0 and xpid < 1:
        xpid_offset = xpid - 0
        xnumel = 384
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
        xnumel = 384
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
    elif xpid >= 2 and xpid < 434:
        xpid_offset = xpid - 2
        xnumel = 442368
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
    elif xpid >= 434 and xpid < 435:
        xpid_offset = xpid - 434
        xnumel = 384
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
    elif xpid >= 435 and xpid < 436:
        xpid_offset = xpid - 435
        xnumel = 384
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
    elif xpid >= 436 and xpid < 868:
        xpid_offset = xpid - 436
        xnumel = 442368
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
    elif xpid >= 868 and xpid < 869:
        xpid_offset = xpid - 868
        xnumel = 384
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
    elif xpid >= 869 and xpid < 870:
        xpid_offset = xpid - 869
        xnumel = 384
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
    elif xpid >= 870 and xpid < 1430:
        xpid_offset = xpid - 870
        xnumel = 573440
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
    elif xpid >= 1430 and xpid < 1431:
        xpid_offset = xpid - 1430
        xnumel = 448
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
    elif xpid >= 1431 and xpid < 1432:
        xpid_offset = xpid - 1431
        xnumel = 448
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
    elif xpid >= 1432 and xpid < 2944:
        xpid_offset = xpid - 1432
        xnumel = 1548288
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
    elif xpid >= 2944 and xpid < 2945:
        xpid_offset = xpid - 2944
        xnumel = 384
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
    elif xpid >= 2945 and xpid < 2946:
        xpid_offset = xpid - 2945
        xnumel = 384
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
    elif xpid >= 2946 and xpid < 3378:
        xpid_offset = xpid - 2946
        xnumel = 442368
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
    elif xpid >= 3378 and xpid < 3379:
        xpid_offset = xpid - 3378
        xnumel = 384
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
    elif xpid >= 3379 and xpid < 3380:
        xpid_offset = xpid - 3379
        xnumel = 384
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
    elif xpid >= 3380 and xpid < 3812:
        xpid_offset = xpid - 3380
        xnumel = 442368
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
    elif xpid >= 3812 and xpid < 3813:
        xpid_offset = xpid - 3812
        xnumel = 384
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
    elif xpid >= 3813 and xpid < 3814:
        xpid_offset = xpid - 3813
        xnumel = 384
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


# kernel path: /tmp/torchinductor_zhang402/hu/chuyieepzcex4tgvmsvu732nrbaklb4knzalrq76aczhujum2es6.py
# Source Nodes: [], Original ATen: []

triton_for_fused_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'kernel_name': 'triton_for_fused_15', 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, in_ptr47, in_ptr48, in_ptr49, in_ptr50, in_ptr51, in_ptr52, in_ptr53, in_ptr54, in_ptr55, in_ptr56, in_ptr57, in_ptr58, in_ptr59, in_ptr60, in_ptr61, in_ptr62, in_ptr63, in_ptr64, in_ptr65, in_ptr66, in_ptr67, in_ptr68, in_ptr69, in_ptr70, in_ptr71, in_ptr72, in_ptr73, in_ptr74, in_ptr75, in_ptr76, in_ptr77, in_ptr78, in_ptr79, in_ptr80, in_ptr81, in_ptr82, in_ptr83, in_ptr84, in_ptr85, in_ptr86, in_ptr87, in_ptr88, in_ptr89, in_ptr90, in_ptr91, in_ptr92, in_ptr93, in_ptr94, in_ptr95, in_ptr96, in_ptr97, in_ptr98, in_ptr99, out_ptr0, out_ptr2, out_ptr3, out_ptr4, out_ptr6, out_ptr7, out_ptr8, out_ptr10, out_ptr11, out_ptr12, out_ptr14, out_ptr15, out_ptr16, out_ptr18, out_ptr19, out_ptr20, out_ptr22, out_ptr23, out_ptr24, out_ptr26, out_ptr27, out_ptr28, out_ptr30, out_ptr31, out_ptr32, out_ptr34, out_ptr35, out_ptr36, out_ptr38, out_ptr39, out_ptr40, out_ptr42, out_ptr43, out_ptr44, out_ptr46, out_ptr47, out_ptr48, out_ptr50, out_ptr51, out_ptr52, out_ptr54, out_ptr55, out_ptr56, out_ptr58, out_ptr59, out_ptr60, out_ptr62, out_ptr63, out_ptr64, out_ptr66, out_ptr67, out_ptr68, out_ptr70, out_ptr71, out_ptr72, out_ptr74, out_ptr75, out_ptr76, out_ptr78, out_ptr79):
    xpid = tl.program_id(0)
    XBLOCK: tl.constexpr = 1024
    if xpid >= 0 and xpid < 240:
        xpid_offset = xpid - 0
        xnumel = 245760
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
    elif xpid >= 240 and xpid < 241:
        xpid_offset = xpid - 240
        xnumel = 192
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
    elif xpid >= 241 and xpid < 242:
        xpid_offset = xpid - 241
        xnumel = 192
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
    elif xpid >= 242 and xpid < 882:
        xpid_offset = xpid - 242
        xnumel = 655360
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
    elif xpid >= 882 and xpid < 883:
        xpid_offset = xpid - 882
        xnumel = 320
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
    elif xpid >= 883 and xpid < 884:
        xpid_offset = xpid - 883
        xnumel = 320
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
    elif xpid >= 884 and xpid < 1652:
        xpid_offset = xpid - 884
        xnumel = 786432
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
    elif xpid >= 1652 and xpid < 1653:
        xpid_offset = xpid - 1652
        xnumel = 384
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
    elif xpid >= 1653 and xpid < 1654:
        xpid_offset = xpid - 1653
        xnumel = 384
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
    elif xpid >= 1654 and xpid < 2086:
        xpid_offset = xpid - 1654
        xnumel = 442368
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
    elif xpid >= 2086 and xpid < 2087:
        xpid_offset = xpid - 2086
        xnumel = 384
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
    elif xpid >= 2087 and xpid < 2088:
        xpid_offset = xpid - 2087
        xnumel = 384
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
    elif xpid >= 2088 and xpid < 2520:
        xpid_offset = xpid - 2088
        xnumel = 442368
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
    elif xpid >= 2520 and xpid < 2521:
        xpid_offset = xpid - 2520
        xnumel = 384
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
    elif xpid >= 2521 and xpid < 2522:
        xpid_offset = xpid - 2521
        xnumel = 384
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
    elif xpid >= 2522 and xpid < 3418:
        xpid_offset = xpid - 2522
        xnumel = 917504
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
    elif xpid >= 3418 and xpid < 3419:
        xpid_offset = xpid - 3418
        xnumel = 448
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
    elif xpid >= 3419 and xpid < 3420:
        xpid_offset = xpid - 3419
        xnumel = 448
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
    elif xpid >= 3420 and xpid < 4932:
        xpid_offset = xpid - 3420
        xnumel = 1548288
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
    elif xpid >= 4932 and xpid < 4933:
        xpid_offset = xpid - 4932
        xnumel = 384
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


# kernel path: /tmp/torchinductor_zhang402/zm/czmwsxjkwycwrng4egbynzfkkqqh3mzjrgpwzmwmc3px4xv3tijs.py
# Source Nodes: [], Original ATen: []

triton_for_fused_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: '*fp32', 35: '*fp32', 36: '*fp32', 37: '*fp32', 38: '*fp32', 39: '*fp32', 40: '*fp32', 41: '*fp32', 42: '*fp32', 43: '*fp32', 44: '*fp32', 45: '*fp32', 46: '*fp32', 47: '*fp32', 48: '*fp32', 49: '*fp32', 50: '*fp32', 51: '*fp32', 52: '*fp32', 53: '*fp32', 54: '*fp32', 55: '*fp32', 56: '*fp32', 57: '*fp32', 58: '*fp32', 59: '*fp32', 60: '*fp32', 61: '*fp32', 62: '*fp32', 63: '*fp32', 64: '*fp32', 65: '*fp32', 66: '*fp32', 67: '*fp32', 68: '*fp32', 69: '*fp32', 70: '*fp32', 71: '*fp32', 72: '*fp32', 73: '*fp32', 74: '*fp32', 75: '*fp32', 76: '*fp32', 77: '*fp32', 78: '*fp32', 79: '*fp32', 80: '*fp32', 81: '*fp32', 82: '*fp32', 83: '*fp32', 84: '*fp32', 85: '*fp32', 86: '*fp32', 87: '*fp32', 88: '*fp32', 89: '*fp32', 90: '*fp32', 91: '*fp32', 92: '*fp32', 93: '*fp32', 94: '*fp32', 95: '*fp32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'kernel_name': 'triton_for_fused_16', 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, in_ptr47, in_ptr48, in_ptr49, in_ptr50, in_ptr51, in_ptr52, in_ptr53, in_ptr54, in_ptr55, in_ptr56, in_ptr57, in_ptr58, in_ptr59, out_ptr0, out_ptr2, out_ptr3, out_ptr4, out_ptr6, out_ptr7, out_ptr8, out_ptr10, out_ptr11, out_ptr12, out_ptr14, out_ptr15, out_ptr16, out_ptr18, out_ptr19, out_ptr20, out_ptr22, out_ptr23, out_ptr24, out_ptr26, out_ptr27, out_ptr28, out_ptr30, out_ptr31, out_ptr32, out_ptr34, out_ptr35, out_ptr36, out_ptr38, out_ptr39, out_ptr40, out_ptr42, out_ptr43, out_ptr44, out_ptr46, out_ptr47):
    xpid = tl.program_id(0)
    XBLOCK: tl.constexpr = 1024
    if xpid >= 0 and xpid < 1:
        xpid_offset = xpid - 0
        xnumel = 384
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
    elif xpid >= 1 and xpid < 433:
        xpid_offset = xpid - 1
        xnumel = 442368
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
    elif xpid >= 433 and xpid < 434:
        xpid_offset = xpid - 433
        xnumel = 384
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
    elif xpid >= 434 and xpid < 435:
        xpid_offset = xpid - 434
        xnumel = 384
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
    elif xpid >= 435 and xpid < 867:
        xpid_offset = xpid - 435
        xnumel = 442368
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
    elif xpid >= 867 and xpid < 868:
        xpid_offset = xpid - 867
        xnumel = 384
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
    elif xpid >= 868 and xpid < 869:
        xpid_offset = xpid - 868
        xnumel = 384
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
    elif xpid >= 869 and xpid < 1253:
        xpid_offset = xpid - 869
        xnumel = 393216
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
    elif xpid >= 1253 and xpid < 1254:
        xpid_offset = xpid - 1253
        xnumel = 192
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
    elif xpid >= 1254 and xpid < 1255:
        xpid_offset = xpid - 1254
        xnumel = 192
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
    elif xpid >= 1255 and xpid < 3255:
        xpid_offset = xpid - 1255
        xnumel = 2048000
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
    elif xpid >= 3255 and xpid < 3256:
        xpid_offset = xpid - 3255
        xnumel = 1000
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
    else:
        pass
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1, arg728_1, arg729_1, arg730_1, arg731_1, arg732_1, arg733_1, arg734_1, arg735_1, arg736_1, arg737_1, arg738_1, arg739_1, arg740_1, arg741_1, arg742_1, arg743_1, arg744_1, arg745_1, arg746_1, arg747_1, arg748_1, arg749_1, arg750_1, arg751_1, arg752_1, arg753_1, arg754_1, arg755_1, arg756_1, arg757_1, arg758_1, arg759_1, arg760_1, arg761_1, arg762_1, arg763_1, arg764_1, arg765_1, arg766_1, arg767_1, arg768_1, arg769_1, arg770_1, arg771_1, arg772_1, arg773_1, arg774_1, arg775_1, arg776_1, arg777_1, arg778_1, arg779_1, arg780_1, arg781_1, arg782_1, arg783_1, arg784_1, arg785_1, arg786_1, arg787_1, arg788_1, arg789_1, arg790_1, arg791_1, arg792_1, arg793_1, arg794_1, arg795_1, arg796_1, arg797_1, arg798_1, arg799_1, arg800_1, arg801_1, arg802_1, arg803_1, arg804_1, arg805_1, arg806_1, arg807_1, arg808_1, arg809_1, arg810_1, arg811_1, arg812_1, arg813_1, arg814_1, arg815_1, arg816_1, arg817_1, arg818_1, arg819_1, arg820_1, arg821_1, arg822_1, arg823_1, arg824_1, arg825_1, arg826_1, arg827_1, arg828_1, arg829_1, arg830_1, arg831_1, arg832_1, arg833_1, arg834_1, arg835_1, arg836_1, arg837_1, arg838_1, arg839_1, arg840_1, arg841_1, arg842_1, arg843_1, arg844_1, arg845_1, arg846_1, arg847_1, arg848_1, arg849_1, arg850_1, arg851_1, arg852_1, arg853_1, arg854_1, arg855_1, arg856_1, arg857_1, arg858_1, arg859_1, arg860_1, arg861_1, arg862_1, arg863_1, arg864_1, arg865_1, arg866_1, arg867_1, arg868_1, arg869_1, arg870_1, arg871_1, arg872_1, arg873_1, arg874_1, arg875_1, arg876_1, arg877_1, arg878_1, arg879_1, arg880_1, arg881_1, arg882_1, arg883_1, arg884_1, arg885_1, arg886_1, arg887_1, arg888_1, arg889_1, arg890_1, arg891_1, arg892_1, arg893_1, arg894_1, arg895_1, arg896_1, arg897_1, arg898_1, arg899_1, arg900_1, arg901_1, arg902_1, arg903_1, arg904_1, arg905_1, arg906_1, arg907_1, arg908_1, arg909_1, arg910_1, arg911_1, arg912_1, arg913_1, arg914_1, arg915_1, arg916_1, arg917_1, arg918_1, arg919_1, arg920_1, arg921_1, arg922_1, arg923_1, arg924_1, arg925_1, arg926_1, arg927_1, arg928_1, arg929_1, arg930_1, arg931_1, arg932_1, arg933_1, arg934_1, arg935_1, arg936_1, arg937_1, arg938_1, arg939_1, arg940_1, arg941_1, arg942_1, arg943_1, arg944_1, arg945_1, arg946_1, arg947_1, arg948_1, arg949_1, arg950_1, arg951_1, arg952_1, arg953_1, arg954_1, arg955_1, arg956_1, arg957_1, arg958_1, arg959_1, arg960_1, arg961_1, arg962_1, arg963_1, arg964_1, arg965_1, arg966_1, arg967_1, arg968_1, arg969_1, arg970_1, arg971_1, arg972_1, arg973_1, arg974_1, arg975_1, arg976_1, arg977_1, arg978_1, arg979_1, arg980_1, arg981_1, arg982_1, arg983_1, arg984_1, arg985_1, arg986_1, arg987_1, arg988_1, arg989_1, arg990_1, arg991_1, arg992_1, arg993_1, arg994_1, arg995_1, arg996_1, arg997_1, arg998_1, arg999_1, arg1000_1, arg1001_1, arg1002_1, arg1003_1, arg1004_1, arg1005_1, arg1006_1, arg1007_1, arg1008_1, arg1009_1, arg1010_1, arg1011_1, arg1012_1, arg1013_1, arg1014_1, arg1015_1, arg1016_1, arg1017_1, arg1018_1, arg1019_1, arg1020_1, arg1021_1, arg1022_1, arg1023_1, arg1024_1, arg1025_1, arg1026_1, arg1027_1, arg1028_1, arg1029_1, arg1030_1, arg1031_1, arg1032_1, arg1033_1, arg1034_1, arg1035_1, arg1036_1, arg1037_1, arg1038_1, arg1039_1, arg1040_1, arg1041_1, arg1042_1, arg1043_1, arg1044_1, arg1045_1, arg1046_1, arg1047_1, arg1048_1, arg1049_1, arg1050_1, arg1051_1, arg1052_1, arg1053_1, arg1054_1, arg1055_1, arg1056_1, arg1057_1, arg1058_1, arg1059_1, arg1060_1, arg1061_1, arg1062_1, arg1063_1, arg1064_1, arg1065_1, arg1066_1, arg1067_1, arg1068_1, arg1069_1, arg1070_1, arg1071_1, arg1072_1, arg1073_1, arg1074_1, arg1075_1, arg1076_1, arg1077_1, arg1078_1, arg1079_1, arg1080_1, arg1081_1, arg1082_1, arg1083_1, arg1084_1, arg1085_1, arg1086_1, arg1087_1, arg1088_1, arg1089_1, arg1090_1, arg1091_1, arg1092_1, arg1093_1, arg1094_1, arg1095_1, arg1096_1, arg1097_1, arg1098_1, arg1099_1, arg1100_1, arg1101_1, arg1102_1, arg1103_1, arg1104_1, arg1105_1, arg1106_1, arg1107_1, arg1108_1, arg1109_1, arg1110_1, arg1111_1, arg1112_1, arg1113_1, arg1114_1, arg1115_1, arg1116_1, arg1117_1, arg1118_1, arg1119_1, arg1120_1, arg1121_1, arg1122_1, arg1123_1, arg1124_1, arg1125_1, arg1126_1, arg1127_1, arg1128_1, arg1129_1, arg1130_1, arg1131_1, arg1132_1, arg1133_1, arg1134_1, arg1135_1, arg1136_1, arg1137_1, arg1138_1, arg1139_1, arg1140_1, arg1141_1, arg1142_1, arg1143_1, arg1144_1, arg1145_1, arg1146_1, arg1147_1, arg1148_1, arg1149_1, arg1150_1, arg1151_1, arg1152_1, arg1153_1, arg1154_1, arg1155_1, arg1156_1, arg1157_1, arg1158_1, arg1159_1, arg1160_1, arg1161_1, arg1162_1, arg1163_1, arg1164_1, arg1165_1, arg1166_1, arg1167_1, arg1168_1, arg1169_1, arg1170_1, arg1171_1, arg1172_1, arg1173_1, arg1174_1, arg1175_1, arg1176_1, arg1177_1, arg1178_1, arg1179_1, arg1180_1, arg1181_1, arg1182_1, arg1183_1, arg1184_1, arg1185_1, arg1186_1, arg1187_1, arg1188_1, arg1189_1, arg1190_1, arg1191_1, arg1192_1, arg1193_1, arg1194_1, arg1195_1, arg1196_1, arg1197_1, arg1198_1, arg1199_1, arg1200_1, arg1201_1, arg1202_1, arg1203_1, arg1204_1, arg1205_1, arg1206_1, arg1207_1, arg1208_1, arg1209_1, arg1210_1, arg1211_1, arg1212_1, arg1213_1, arg1214_1, arg1215_1, arg1216_1, arg1217_1, arg1218_1, arg1219_1, arg1220_1, arg1221_1, arg1222_1, arg1223_1, arg1224_1, arg1225_1, arg1226_1, arg1227_1, arg1228_1, arg1229_1, arg1230_1, arg1231_1, arg1232_1, arg1233_1, arg1234_1, arg1235_1, arg1236_1, arg1237_1, arg1238_1, arg1239_1, arg1240_1, arg1241_1, arg1242_1, arg1243_1, arg1244_1, arg1245_1, arg1246_1, arg1247_1, arg1248_1, arg1249_1, arg1250_1, arg1251_1, arg1252_1, arg1253_1, arg1254_1, arg1255_1, arg1256_1, arg1257_1, arg1258_1, arg1259_1, arg1260_1, arg1261_1, arg1262_1, arg1263_1, arg1264_1, arg1265_1, arg1266_1, arg1267_1, arg1268_1, arg1269_1, arg1270_1, arg1271_1, arg1272_1, arg1273_1, arg1274_1, arg1275_1, arg1276_1, arg1277_1, arg1278_1, arg1279_1, arg1280_1, arg1281_1, arg1282_1, arg1283_1, arg1284_1, arg1285_1, arg1286_1, arg1287_1, arg1288_1, arg1289_1, arg1290_1, arg1291_1, arg1292_1, arg1293_1, arg1294_1, arg1295_1, arg1296_1, arg1297_1, arg1298_1, arg1299_1, arg1300_1, arg1301_1, arg1302_1, arg1303_1, arg1304_1, arg1305_1, arg1306_1, arg1307_1, arg1308_1, arg1309_1, arg1310_1, arg1311_1, arg1312_1, arg1313_1, arg1314_1, arg1315_1, arg1316_1, arg1317_1, arg1318_1, arg1319_1, arg1320_1, arg1321_1, arg1322_1, arg1323_1, arg1324_1, arg1325_1, arg1326_1, arg1327_1, arg1328_1, arg1329_1, arg1330_1, arg1331_1, arg1332_1, arg1333_1, arg1334_1, arg1335_1, arg1336_1, arg1337_1, arg1338_1, arg1339_1, arg1340_1, arg1341_1, arg1342_1, arg1343_1, arg1344_1, arg1345_1, arg1346_1, arg1347_1, arg1348_1, arg1349_1, arg1350_1, arg1351_1, arg1352_1, arg1353_1, arg1354_1, arg1355_1, arg1356_1, arg1357_1, arg1358_1, arg1359_1, arg1360_1, arg1361_1, arg1362_1, arg1363_1, arg1364_1, arg1365_1, arg1366_1, arg1367_1, arg1368_1, arg1369_1, arg1370_1, arg1371_1, arg1372_1, arg1373_1, arg1374_1, arg1375_1, arg1376_1, arg1377_1, arg1378_1, arg1379_1, arg1380_1, arg1381_1, arg1382_1, arg1383_1, arg1384_1, arg1385_1, arg1386_1, arg1387_1, arg1388_1, arg1389_1, arg1390_1, arg1391_1, arg1392_1, arg1393_1, arg1394_1, arg1395_1, arg1396_1, arg1397_1, arg1398_1, arg1399_1, arg1400_1, arg1401_1, arg1402_1, arg1403_1, arg1404_1, arg1405_1, arg1406_1, arg1407_1, arg1408_1, arg1409_1, arg1410_1, arg1411_1, arg1412_1, arg1413_1, arg1414_1, arg1415_1, arg1416_1, arg1417_1, arg1418_1, arg1419_1, arg1420_1, arg1421_1, arg1422_1, arg1423_1, arg1424_1, arg1425_1, arg1426_1, arg1427_1, arg1428_1, arg1429_1, arg1430_1, arg1431_1, arg1432_1, arg1433_1, arg1434_1, arg1435_1, arg1436_1, arg1437_1, arg1438_1, arg1439_1, arg1440_1, arg1441_1, arg1442_1, arg1443_1, arg1444_1, arg1445_1, arg1446_1, arg1447_1, arg1448_1, arg1449_1, arg1450_1, arg1451_1, arg1452_1, arg1453_1, arg1454_1, arg1455_1, arg1456_1, arg1457_1, arg1458_1, arg1459_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg1_1, (32, ), (1, ))
    assert_size_stride(arg2_1, (32, ), (1, ))
    assert_size_stride(arg3_1, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg4_1, (32, ), (1, ))
    assert_size_stride(arg5_1, (32, ), (1, ))
    assert_size_stride(arg6_1, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg7_1, (64, ), (1, ))
    assert_size_stride(arg8_1, (64, ), (1, ))
    assert_size_stride(arg9_1, (80, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(arg10_1, (80, ), (1, ))
    assert_size_stride(arg11_1, (80, ), (1, ))
    assert_size_stride(arg12_1, (192, 80, 3, 3), (720, 9, 3, 1))
    assert_size_stride(arg13_1, (192, ), (1, ))
    assert_size_stride(arg14_1, (192, ), (1, ))
    assert_size_stride(arg15_1, (64, 192, 1, 1), (192, 1, 192, 192))
    assert_size_stride(arg16_1, (64, ), (1, ))
    assert_size_stride(arg17_1, (64, ), (1, ))
    assert_size_stride(arg18_1, (48, 192, 1, 1), (192, 1, 192, 192))
    assert_size_stride(arg19_1, (48, ), (1, ))
    assert_size_stride(arg20_1, (48, ), (1, ))
    assert_size_stride(arg21_1, (64, 48, 5, 5), (1200, 25, 5, 1))
    assert_size_stride(arg22_1, (64, ), (1, ))
    assert_size_stride(arg23_1, (64, ), (1, ))
    assert_size_stride(arg24_1, (64, 192, 1, 1), (192, 1, 192, 192))
    assert_size_stride(arg25_1, (64, ), (1, ))
    assert_size_stride(arg26_1, (64, ), (1, ))
    assert_size_stride(arg27_1, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg28_1, (96, ), (1, ))
    assert_size_stride(arg29_1, (96, ), (1, ))
    assert_size_stride(arg30_1, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(arg31_1, (96, ), (1, ))
    assert_size_stride(arg32_1, (96, ), (1, ))
    assert_size_stride(arg33_1, (32, 192, 1, 1), (192, 1, 192, 192))
    assert_size_stride(arg34_1, (32, ), (1, ))
    assert_size_stride(arg35_1, (32, ), (1, ))
    assert_size_stride(arg36_1, (64, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg37_1, (64, ), (1, ))
    assert_size_stride(arg38_1, (64, ), (1, ))
    assert_size_stride(arg39_1, (48, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg40_1, (48, ), (1, ))
    assert_size_stride(arg41_1, (48, ), (1, ))
    assert_size_stride(arg42_1, (64, 48, 5, 5), (1200, 25, 5, 1))
    assert_size_stride(arg43_1, (64, ), (1, ))
    assert_size_stride(arg44_1, (64, ), (1, ))
    assert_size_stride(arg45_1, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg46_1, (64, ), (1, ))
    assert_size_stride(arg47_1, (64, ), (1, ))
    assert_size_stride(arg48_1, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg49_1, (96, ), (1, ))
    assert_size_stride(arg50_1, (96, ), (1, ))
    assert_size_stride(arg51_1, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(arg52_1, (96, ), (1, ))
    assert_size_stride(arg53_1, (96, ), (1, ))
    assert_size_stride(arg54_1, (64, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg55_1, (64, ), (1, ))
    assert_size_stride(arg56_1, (64, ), (1, ))
    assert_size_stride(arg57_1, (64, 288, 1, 1), (288, 1, 288, 288))
    assert_size_stride(arg58_1, (64, ), (1, ))
    assert_size_stride(arg59_1, (64, ), (1, ))
    assert_size_stride(arg60_1, (48, 288, 1, 1), (288, 1, 288, 288))
    assert_size_stride(arg61_1, (48, ), (1, ))
    assert_size_stride(arg62_1, (48, ), (1, ))
    assert_size_stride(arg63_1, (64, 48, 5, 5), (1200, 25, 5, 1))
    assert_size_stride(arg64_1, (64, ), (1, ))
    assert_size_stride(arg65_1, (64, ), (1, ))
    assert_size_stride(arg66_1, (64, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(arg67_1, (64, ), (1, ))
    assert_size_stride(arg68_1, (64, ), (1, ))
    assert_size_stride(arg69_1, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg70_1, (96, ), (1, ))
    assert_size_stride(arg71_1, (96, ), (1, ))
    assert_size_stride(arg72_1, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(arg73_1, (96, ), (1, ))
    assert_size_stride(arg74_1, (96, ), (1, ))
    assert_size_stride(arg75_1, (64, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(arg76_1, (64, ), (1, ))
    assert_size_stride(arg77_1, (64, ), (1, ))
    assert_size_stride(arg78_1, (384, 288, 3, 3), (2592, 9, 3, 1))
    assert_size_stride(arg79_1, (384, ), (1, ))
    assert_size_stride(arg80_1, (384, ), (1, ))
    assert_size_stride(arg81_1, (64, 288, 1, 1), (288, 1, 288, 288))
    assert_size_stride(arg82_1, (64, ), (1, ))
    assert_size_stride(arg83_1, (64, ), (1, ))
    assert_size_stride(arg84_1, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg85_1, (96, ), (1, ))
    assert_size_stride(arg86_1, (96, ), (1, ))
    assert_size_stride(arg87_1, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(arg88_1, (96, ), (1, ))
    assert_size_stride(arg89_1, (96, ), (1, ))
    assert_size_stride(arg90_1, (192, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(arg91_1, (192, ), (1, ))
    assert_size_stride(arg92_1, (192, ), (1, ))
    assert_size_stride(arg93_1, (128, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(arg94_1, (128, ), (1, ))
    assert_size_stride(arg95_1, (128, ), (1, ))
    assert_size_stride(arg96_1, (128, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(arg97_1, (128, ), (1, ))
    assert_size_stride(arg98_1, (128, ), (1, ))
    assert_size_stride(arg99_1, (192, 128, 7, 1), (896, 7, 1, 1))
    assert_size_stride(arg100_1, (192, ), (1, ))
    assert_size_stride(arg101_1, (192, ), (1, ))
    assert_size_stride(arg102_1, (128, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(arg103_1, (128, ), (1, ))
    assert_size_stride(arg104_1, (128, ), (1, ))
    assert_size_stride(arg105_1, (128, 128, 7, 1), (896, 7, 1, 1))
    assert_size_stride(arg106_1, (128, ), (1, ))
    assert_size_stride(arg107_1, (128, ), (1, ))
    assert_size_stride(arg108_1, (128, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(arg109_1, (128, ), (1, ))
    assert_size_stride(arg110_1, (128, ), (1, ))
    assert_size_stride(arg111_1, (128, 128, 7, 1), (896, 7, 1, 1))
    assert_size_stride(arg112_1, (128, ), (1, ))
    assert_size_stride(arg113_1, (128, ), (1, ))
    assert_size_stride(arg114_1, (192, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(arg115_1, (192, ), (1, ))
    assert_size_stride(arg116_1, (192, ), (1, ))
    assert_size_stride(arg117_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg118_1, (192, ), (1, ))
    assert_size_stride(arg119_1, (192, ), (1, ))
    assert_size_stride(arg120_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg121_1, (192, ), (1, ))
    assert_size_stride(arg122_1, (192, ), (1, ))
    assert_size_stride(arg123_1, (160, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(arg124_1, (160, ), (1, ))
    assert_size_stride(arg125_1, (160, ), (1, ))
    assert_size_stride(arg126_1, (160, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(arg127_1, (160, ), (1, ))
    assert_size_stride(arg128_1, (160, ), (1, ))
    assert_size_stride(arg129_1, (192, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(arg130_1, (192, ), (1, ))
    assert_size_stride(arg131_1, (192, ), (1, ))
    assert_size_stride(arg132_1, (160, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg133_1, (160, ), (1, ))
    assert_size_stride(arg134_1, (160, ), (1, ))
    assert_size_stride(arg135_1, (160, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(arg136_1, (160, ), (1, ))
    assert_size_stride(arg137_1, (160, ), (1, ))
    assert_size_stride(arg138_1, (160, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(arg139_1, (160, ), (1, ))
    assert_size_stride(arg140_1, (160, ), (1, ))
    assert_size_stride(arg141_1, (160, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(arg142_1, (160, ), (1, ))
    assert_size_stride(arg143_1, (160, ), (1, ))
    assert_size_stride(arg144_1, (192, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(arg145_1, (192, ), (1, ))
    assert_size_stride(arg146_1, (192, ), (1, ))
    assert_size_stride(arg147_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg148_1, (192, ), (1, ))
    assert_size_stride(arg149_1, (192, ), (1, ))
    assert_size_stride(arg150_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg151_1, (192, ), (1, ))
    assert_size_stride(arg152_1, (192, ), (1, ))
    assert_size_stride(arg153_1, (160, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg154_1, (160, ), (1, ))
    assert_size_stride(arg155_1, (160, ), (1, ))
    assert_size_stride(arg156_1, (160, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(arg157_1, (160, ), (1, ))
    assert_size_stride(arg158_1, (160, ), (1, ))
    assert_size_stride(arg159_1, (192, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(arg160_1, (192, ), (1, ))
    assert_size_stride(arg161_1, (192, ), (1, ))
    assert_size_stride(arg162_1, (160, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(arg163_1, (160, ), (1, ))
    assert_size_stride(arg164_1, (160, ), (1, ))
    assert_size_stride(arg165_1, (160, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(arg166_1, (160, ), (1, ))
    assert_size_stride(arg167_1, (160, ), (1, ))
    assert_size_stride(arg168_1, (160, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(arg169_1, (160, ), (1, ))
    assert_size_stride(arg170_1, (160, ), (1, ))
    assert_size_stride(arg171_1, (160, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(arg172_1, (160, ), (1, ))
    assert_size_stride(arg173_1, (160, ), (1, ))
    assert_size_stride(arg174_1, (192, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(arg175_1, (192, ), (1, ))
    assert_size_stride(arg176_1, (192, ), (1, ))
    assert_size_stride(arg177_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg178_1, (192, ), (1, ))
    assert_size_stride(arg179_1, (192, ), (1, ))
    assert_size_stride(arg180_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg181_1, (192, ), (1, ))
    assert_size_stride(arg182_1, (192, ), (1, ))
    assert_size_stride(arg183_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg184_1, (192, ), (1, ))
    assert_size_stride(arg185_1, (192, ), (1, ))
    assert_size_stride(arg186_1, (192, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(arg187_1, (192, ), (1, ))
    assert_size_stride(arg188_1, (192, ), (1, ))
    assert_size_stride(arg189_1, (192, 192, 7, 1), (1344, 7, 1, 1))
    assert_size_stride(arg190_1, (192, ), (1, ))
    assert_size_stride(arg191_1, (192, ), (1, ))
    assert_size_stride(arg192_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg193_1, (192, ), (1, ))
    assert_size_stride(arg194_1, (192, ), (1, ))
    assert_size_stride(arg195_1, (192, 192, 7, 1), (1344, 7, 1, 1))
    assert_size_stride(arg196_1, (192, ), (1, ))
    assert_size_stride(arg197_1, (192, ), (1, ))
    assert_size_stride(arg198_1, (192, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(arg199_1, (192, ), (1, ))
    assert_size_stride(arg200_1, (192, ), (1, ))
    assert_size_stride(arg201_1, (192, 192, 7, 1), (1344, 7, 1, 1))
    assert_size_stride(arg202_1, (192, ), (1, ))
    assert_size_stride(arg203_1, (192, ), (1, ))
    assert_size_stride(arg204_1, (192, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(arg205_1, (192, ), (1, ))
    assert_size_stride(arg206_1, (192, ), (1, ))
    assert_size_stride(arg207_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg208_1, (192, ), (1, ))
    assert_size_stride(arg209_1, (192, ), (1, ))
    assert_size_stride(arg210_1, (128, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(arg211_1, (128, ), (1, ))
    assert_size_stride(arg212_1, (128, ), (1, ))
    assert_size_stride(arg213_1, (768, 128, 5, 5), (3200, 25, 5, 1))
    assert_size_stride(arg214_1, (768, ), (1, ))
    assert_size_stride(arg215_1, (768, ), (1, ))
    assert_size_stride(arg216_1, (1000, 768), (768, 1))
    assert_size_stride(arg217_1, (1000, ), (1, ))
    assert_size_stride(arg218_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg219_1, (192, ), (1, ))
    assert_size_stride(arg220_1, (192, ), (1, ))
    assert_size_stride(arg221_1, (320, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg222_1, (320, ), (1, ))
    assert_size_stride(arg223_1, (320, ), (1, ))
    assert_size_stride(arg224_1, (192, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(arg225_1, (192, ), (1, ))
    assert_size_stride(arg226_1, (192, ), (1, ))
    assert_size_stride(arg227_1, (192, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(arg228_1, (192, ), (1, ))
    assert_size_stride(arg229_1, (192, ), (1, ))
    assert_size_stride(arg230_1, (192, 192, 7, 1), (1344, 7, 1, 1))
    assert_size_stride(arg231_1, (192, ), (1, ))
    assert_size_stride(arg232_1, (192, ), (1, ))
    assert_size_stride(arg233_1, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg234_1, (192, ), (1, ))
    assert_size_stride(arg235_1, (192, ), (1, ))
    assert_size_stride(arg236_1, (320, 1280, 1, 1), (1280, 1, 1280, 1280))
    assert_size_stride(arg237_1, (320, ), (1, ))
    assert_size_stride(arg238_1, (320, ), (1, ))
    assert_size_stride(arg239_1, (384, 1280, 1, 1), (1280, 1, 1280, 1280))
    assert_size_stride(arg240_1, (384, ), (1, ))
    assert_size_stride(arg241_1, (384, ), (1, ))
    assert_size_stride(arg242_1, (384, 384, 1, 3), (1152, 3, 3, 1))
    assert_size_stride(arg243_1, (384, ), (1, ))
    assert_size_stride(arg244_1, (384, ), (1, ))
    assert_size_stride(arg245_1, (384, 384, 3, 1), (1152, 3, 1, 1))
    assert_size_stride(arg246_1, (384, ), (1, ))
    assert_size_stride(arg247_1, (384, ), (1, ))
    assert_size_stride(arg248_1, (448, 1280, 1, 1), (1280, 1, 1280, 1280))
    assert_size_stride(arg249_1, (448, ), (1, ))
    assert_size_stride(arg250_1, (448, ), (1, ))
    assert_size_stride(arg251_1, (384, 448, 3, 3), (4032, 9, 3, 1))
    assert_size_stride(arg252_1, (384, ), (1, ))
    assert_size_stride(arg253_1, (384, ), (1, ))
    assert_size_stride(arg254_1, (384, 384, 1, 3), (1152, 3, 3, 1))
    assert_size_stride(arg255_1, (384, ), (1, ))
    assert_size_stride(arg256_1, (384, ), (1, ))
    assert_size_stride(arg257_1, (384, 384, 3, 1), (1152, 3, 1, 1))
    assert_size_stride(arg258_1, (384, ), (1, ))
    assert_size_stride(arg259_1, (384, ), (1, ))
    assert_size_stride(arg260_1, (192, 1280, 1, 1), (1280, 1, 1280, 1280))
    assert_size_stride(arg261_1, (192, ), (1, ))
    assert_size_stride(arg262_1, (192, ), (1, ))
    assert_size_stride(arg263_1, (320, 2048, 1, 1), (2048, 1, 2048, 2048))
    assert_size_stride(arg264_1, (320, ), (1, ))
    assert_size_stride(arg265_1, (320, ), (1, ))
    assert_size_stride(arg266_1, (384, 2048, 1, 1), (2048, 1, 2048, 2048))
    assert_size_stride(arg267_1, (384, ), (1, ))
    assert_size_stride(arg268_1, (384, ), (1, ))
    assert_size_stride(arg269_1, (384, 384, 1, 3), (1152, 3, 3, 1))
    assert_size_stride(arg270_1, (384, ), (1, ))
    assert_size_stride(arg271_1, (384, ), (1, ))
    assert_size_stride(arg272_1, (384, 384, 3, 1), (1152, 3, 1, 1))
    assert_size_stride(arg273_1, (384, ), (1, ))
    assert_size_stride(arg274_1, (384, ), (1, ))
    assert_size_stride(arg275_1, (448, 2048, 1, 1), (2048, 1, 2048, 2048))
    assert_size_stride(arg276_1, (448, ), (1, ))
    assert_size_stride(arg277_1, (448, ), (1, ))
    assert_size_stride(arg278_1, (384, 448, 3, 3), (4032, 9, 3, 1))
    assert_size_stride(arg279_1, (384, ), (1, ))
    assert_size_stride(arg280_1, (384, ), (1, ))
    assert_size_stride(arg281_1, (384, 384, 1, 3), (1152, 3, 3, 1))
    assert_size_stride(arg282_1, (384, ), (1, ))
    assert_size_stride(arg283_1, (384, ), (1, ))
    assert_size_stride(arg284_1, (384, 384, 3, 1), (1152, 3, 1, 1))
    assert_size_stride(arg285_1, (384, ), (1, ))
    assert_size_stride(arg286_1, (384, ), (1, ))
    assert_size_stride(arg287_1, (192, 2048, 1, 1), (2048, 1, 2048, 2048))
    assert_size_stride(arg288_1, (192, ), (1, ))
    assert_size_stride(arg289_1, (192, ), (1, ))
    assert_size_stride(arg290_1, (1000, 2048), (2048, 1))
    assert_size_stride(arg291_1, (1000, ), (1, ))
    assert_size_stride(arg292_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg293_1, (32, ), (1, ))
    assert_size_stride(arg294_1, (32, ), (1, ))
    assert_size_stride(arg295_1, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg296_1, (32, ), (1, ))
    assert_size_stride(arg297_1, (32, ), (1, ))
    assert_size_stride(arg298_1, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg299_1, (64, ), (1, ))
    assert_size_stride(arg300_1, (64, ), (1, ))
    assert_size_stride(arg301_1, (80, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(arg302_1, (80, ), (1, ))
    assert_size_stride(arg303_1, (80, ), (1, ))
    assert_size_stride(arg304_1, (192, 80, 3, 3), (720, 9, 3, 1))
    assert_size_stride(arg305_1, (192, ), (1, ))
    assert_size_stride(arg306_1, (192, ), (1, ))
    assert_size_stride(arg307_1, (64, 192, 1, 1), (192, 1, 192, 192))
    assert_size_stride(arg308_1, (64, ), (1, ))
    assert_size_stride(arg309_1, (64, ), (1, ))
    assert_size_stride(arg310_1, (48, 192, 1, 1), (192, 1, 192, 192))
    assert_size_stride(arg311_1, (48, ), (1, ))
    assert_size_stride(arg312_1, (48, ), (1, ))
    assert_size_stride(arg313_1, (64, 48, 5, 5), (1200, 25, 5, 1))
    assert_size_stride(arg314_1, (64, ), (1, ))
    assert_size_stride(arg315_1, (64, ), (1, ))
    assert_size_stride(arg316_1, (64, 192, 1, 1), (192, 1, 192, 192))
    assert_size_stride(arg317_1, (64, ), (1, ))
    assert_size_stride(arg318_1, (64, ), (1, ))
    assert_size_stride(arg319_1, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg320_1, (96, ), (1, ))
    assert_size_stride(arg321_1, (96, ), (1, ))
    assert_size_stride(arg322_1, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(arg323_1, (96, ), (1, ))
    assert_size_stride(arg324_1, (96, ), (1, ))
    assert_size_stride(arg325_1, (32, 192, 1, 1), (192, 1, 192, 192))
    assert_size_stride(arg326_1, (32, ), (1, ))
    assert_size_stride(arg327_1, (32, ), (1, ))
    assert_size_stride(arg328_1, (64, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg329_1, (64, ), (1, ))
    assert_size_stride(arg330_1, (64, ), (1, ))
    assert_size_stride(arg331_1, (48, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg332_1, (48, ), (1, ))
    assert_size_stride(arg333_1, (48, ), (1, ))
    assert_size_stride(arg334_1, (64, 48, 5, 5), (1200, 25, 5, 1))
    assert_size_stride(arg335_1, (64, ), (1, ))
    assert_size_stride(arg336_1, (64, ), (1, ))
    assert_size_stride(arg337_1, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg338_1, (64, ), (1, ))
    assert_size_stride(arg339_1, (64, ), (1, ))
    assert_size_stride(arg340_1, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg341_1, (96, ), (1, ))
    assert_size_stride(arg342_1, (96, ), (1, ))
    assert_size_stride(arg343_1, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(arg344_1, (96, ), (1, ))
    assert_size_stride(arg345_1, (96, ), (1, ))
    assert_size_stride(arg346_1, (64, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg347_1, (64, ), (1, ))
    assert_size_stride(arg348_1, (64, ), (1, ))
    assert_size_stride(arg349_1, (64, 288, 1, 1), (288, 1, 288, 288))
    assert_size_stride(arg350_1, (64, ), (1, ))
    assert_size_stride(arg351_1, (64, ), (1, ))
    assert_size_stride(arg352_1, (48, 288, 1, 1), (288, 1, 288, 288))
    assert_size_stride(arg353_1, (48, ), (1, ))
    assert_size_stride(arg354_1, (48, ), (1, ))
    assert_size_stride(arg355_1, (64, 48, 5, 5), (1200, 25, 5, 1))
    assert_size_stride(arg356_1, (64, ), (1, ))
    assert_size_stride(arg357_1, (64, ), (1, ))
    assert_size_stride(arg358_1, (64, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(arg359_1, (64, ), (1, ))
    assert_size_stride(arg360_1, (64, ), (1, ))
    assert_size_stride(arg361_1, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg362_1, (96, ), (1, ))
    assert_size_stride(arg363_1, (96, ), (1, ))
    assert_size_stride(arg364_1, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(arg365_1, (96, ), (1, ))
    assert_size_stride(arg366_1, (96, ), (1, ))
    assert_size_stride(arg367_1, (64, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(arg368_1, (64, ), (1, ))
    assert_size_stride(arg369_1, (64, ), (1, ))
    assert_size_stride(arg370_1, (384, 288, 3, 3), (2592, 9, 3, 1))
    assert_size_stride(arg371_1, (384, ), (1, ))
    assert_size_stride(arg372_1, (384, ), (1, ))
    assert_size_stride(arg373_1, (64, 288, 1, 1), (288, 1, 288, 288))
    assert_size_stride(arg374_1, (64, ), (1, ))
    assert_size_stride(arg375_1, (64, ), (1, ))
    assert_size_stride(arg376_1, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg377_1, (96, ), (1, ))
    assert_size_stride(arg378_1, (96, ), (1, ))
    assert_size_stride(arg379_1, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(arg380_1, (96, ), (1, ))
    assert_size_stride(arg381_1, (96, ), (1, ))
    assert_size_stride(arg382_1, (192, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(arg383_1, (192, ), (1, ))
    assert_size_stride(arg384_1, (192, ), (1, ))
    assert_size_stride(arg385_1, (128, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(arg386_1, (128, ), (1, ))
    assert_size_stride(arg387_1, (128, ), (1, ))
    assert_size_stride(arg388_1, (128, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(arg389_1, (128, ), (1, ))
    assert_size_stride(arg390_1, (128, ), (1, ))
    assert_size_stride(arg391_1, (192, 128, 7, 1), (896, 7, 1, 1))
    assert_size_stride(arg392_1, (192, ), (1, ))
    assert_size_stride(arg393_1, (192, ), (1, ))
    assert_size_stride(arg394_1, (128, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(arg395_1, (128, ), (1, ))
    assert_size_stride(arg396_1, (128, ), (1, ))
    assert_size_stride(arg397_1, (128, 128, 7, 1), (896, 7, 1, 1))
    assert_size_stride(arg398_1, (128, ), (1, ))
    assert_size_stride(arg399_1, (128, ), (1, ))
    assert_size_stride(arg400_1, (128, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(arg401_1, (128, ), (1, ))
    assert_size_stride(arg402_1, (128, ), (1, ))
    assert_size_stride(arg403_1, (128, 128, 7, 1), (896, 7, 1, 1))
    assert_size_stride(arg404_1, (128, ), (1, ))
    assert_size_stride(arg405_1, (128, ), (1, ))
    assert_size_stride(arg406_1, (192, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(arg407_1, (192, ), (1, ))
    assert_size_stride(arg408_1, (192, ), (1, ))
    assert_size_stride(arg409_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg410_1, (192, ), (1, ))
    assert_size_stride(arg411_1, (192, ), (1, ))
    assert_size_stride(arg412_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg413_1, (192, ), (1, ))
    assert_size_stride(arg414_1, (192, ), (1, ))
    assert_size_stride(arg415_1, (160, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(arg416_1, (160, ), (1, ))
    assert_size_stride(arg417_1, (160, ), (1, ))
    assert_size_stride(arg418_1, (160, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(arg419_1, (160, ), (1, ))
    assert_size_stride(arg420_1, (160, ), (1, ))
    assert_size_stride(arg421_1, (192, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(arg422_1, (192, ), (1, ))
    assert_size_stride(arg423_1, (192, ), (1, ))
    assert_size_stride(arg424_1, (160, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg425_1, (160, ), (1, ))
    assert_size_stride(arg426_1, (160, ), (1, ))
    assert_size_stride(arg427_1, (160, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(arg428_1, (160, ), (1, ))
    assert_size_stride(arg429_1, (160, ), (1, ))
    assert_size_stride(arg430_1, (160, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(arg431_1, (160, ), (1, ))
    assert_size_stride(arg432_1, (160, ), (1, ))
    assert_size_stride(arg433_1, (160, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(arg434_1, (160, ), (1, ))
    assert_size_stride(arg435_1, (160, ), (1, ))
    assert_size_stride(arg436_1, (192, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(arg437_1, (192, ), (1, ))
    assert_size_stride(arg438_1, (192, ), (1, ))
    assert_size_stride(arg439_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg440_1, (192, ), (1, ))
    assert_size_stride(arg441_1, (192, ), (1, ))
    assert_size_stride(arg442_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg443_1, (192, ), (1, ))
    assert_size_stride(arg444_1, (192, ), (1, ))
    assert_size_stride(arg445_1, (160, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg446_1, (160, ), (1, ))
    assert_size_stride(arg447_1, (160, ), (1, ))
    assert_size_stride(arg448_1, (160, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(arg449_1, (160, ), (1, ))
    assert_size_stride(arg450_1, (160, ), (1, ))
    assert_size_stride(arg451_1, (192, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(arg452_1, (192, ), (1, ))
    assert_size_stride(arg453_1, (192, ), (1, ))
    assert_size_stride(arg454_1, (160, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(arg455_1, (160, ), (1, ))
    assert_size_stride(arg456_1, (160, ), (1, ))
    assert_size_stride(arg457_1, (160, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(arg458_1, (160, ), (1, ))
    assert_size_stride(arg459_1, (160, ), (1, ))
    assert_size_stride(arg460_1, (160, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(arg461_1, (160, ), (1, ))
    assert_size_stride(arg462_1, (160, ), (1, ))
    assert_size_stride(arg463_1, (160, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(arg464_1, (160, ), (1, ))
    assert_size_stride(arg465_1, (160, ), (1, ))
    assert_size_stride(arg466_1, (192, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(arg467_1, (192, ), (1, ))
    assert_size_stride(arg468_1, (192, ), (1, ))
    assert_size_stride(arg469_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg470_1, (192, ), (1, ))
    assert_size_stride(arg471_1, (192, ), (1, ))
    assert_size_stride(arg472_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg473_1, (192, ), (1, ))
    assert_size_stride(arg474_1, (192, ), (1, ))
    assert_size_stride(arg475_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg476_1, (192, ), (1, ))
    assert_size_stride(arg477_1, (192, ), (1, ))
    assert_size_stride(arg478_1, (192, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(arg479_1, (192, ), (1, ))
    assert_size_stride(arg480_1, (192, ), (1, ))
    assert_size_stride(arg481_1, (192, 192, 7, 1), (1344, 7, 1, 1))
    assert_size_stride(arg482_1, (192, ), (1, ))
    assert_size_stride(arg483_1, (192, ), (1, ))
    assert_size_stride(arg484_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg485_1, (192, ), (1, ))
    assert_size_stride(arg486_1, (192, ), (1, ))
    assert_size_stride(arg487_1, (192, 192, 7, 1), (1344, 7, 1, 1))
    assert_size_stride(arg488_1, (192, ), (1, ))
    assert_size_stride(arg489_1, (192, ), (1, ))
    assert_size_stride(arg490_1, (192, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(arg491_1, (192, ), (1, ))
    assert_size_stride(arg492_1, (192, ), (1, ))
    assert_size_stride(arg493_1, (192, 192, 7, 1), (1344, 7, 1, 1))
    assert_size_stride(arg494_1, (192, ), (1, ))
    assert_size_stride(arg495_1, (192, ), (1, ))
    assert_size_stride(arg496_1, (192, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(arg497_1, (192, ), (1, ))
    assert_size_stride(arg498_1, (192, ), (1, ))
    assert_size_stride(arg499_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg500_1, (192, ), (1, ))
    assert_size_stride(arg501_1, (192, ), (1, ))
    assert_size_stride(arg502_1, (128, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(arg503_1, (128, ), (1, ))
    assert_size_stride(arg504_1, (128, ), (1, ))
    assert_size_stride(arg505_1, (768, 128, 5, 5), (3200, 25, 5, 1))
    assert_size_stride(arg506_1, (768, ), (1, ))
    assert_size_stride(arg507_1, (768, ), (1, ))
    assert_size_stride(arg508_1, (1000, 768), (768, 1))
    assert_size_stride(arg509_1, (1000, ), (1, ))
    assert_size_stride(arg510_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg511_1, (192, ), (1, ))
    assert_size_stride(arg512_1, (192, ), (1, ))
    assert_size_stride(arg513_1, (320, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg514_1, (320, ), (1, ))
    assert_size_stride(arg515_1, (320, ), (1, ))
    assert_size_stride(arg516_1, (192, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(arg517_1, (192, ), (1, ))
    assert_size_stride(arg518_1, (192, ), (1, ))
    assert_size_stride(arg519_1, (192, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(arg520_1, (192, ), (1, ))
    assert_size_stride(arg521_1, (192, ), (1, ))
    assert_size_stride(arg522_1, (192, 192, 7, 1), (1344, 7, 1, 1))
    assert_size_stride(arg523_1, (192, ), (1, ))
    assert_size_stride(arg524_1, (192, ), (1, ))
    assert_size_stride(arg525_1, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg526_1, (192, ), (1, ))
    assert_size_stride(arg527_1, (192, ), (1, ))
    assert_size_stride(arg528_1, (320, 1280, 1, 1), (1280, 1, 1280, 1280))
    assert_size_stride(arg529_1, (320, ), (1, ))
    assert_size_stride(arg530_1, (320, ), (1, ))
    assert_size_stride(arg531_1, (384, 1280, 1, 1), (1280, 1, 1280, 1280))
    assert_size_stride(arg532_1, (384, ), (1, ))
    assert_size_stride(arg533_1, (384, ), (1, ))
    assert_size_stride(arg534_1, (384, 384, 1, 3), (1152, 3, 3, 1))
    assert_size_stride(arg535_1, (384, ), (1, ))
    assert_size_stride(arg536_1, (384, ), (1, ))
    assert_size_stride(arg537_1, (384, 384, 3, 1), (1152, 3, 1, 1))
    assert_size_stride(arg538_1, (384, ), (1, ))
    assert_size_stride(arg539_1, (384, ), (1, ))
    assert_size_stride(arg540_1, (448, 1280, 1, 1), (1280, 1, 1280, 1280))
    assert_size_stride(arg541_1, (448, ), (1, ))
    assert_size_stride(arg542_1, (448, ), (1, ))
    assert_size_stride(arg543_1, (384, 448, 3, 3), (4032, 9, 3, 1))
    assert_size_stride(arg544_1, (384, ), (1, ))
    assert_size_stride(arg545_1, (384, ), (1, ))
    assert_size_stride(arg546_1, (384, 384, 1, 3), (1152, 3, 3, 1))
    assert_size_stride(arg547_1, (384, ), (1, ))
    assert_size_stride(arg548_1, (384, ), (1, ))
    assert_size_stride(arg549_1, (384, 384, 3, 1), (1152, 3, 1, 1))
    assert_size_stride(arg550_1, (384, ), (1, ))
    assert_size_stride(arg551_1, (384, ), (1, ))
    assert_size_stride(arg552_1, (192, 1280, 1, 1), (1280, 1, 1280, 1280))
    assert_size_stride(arg553_1, (192, ), (1, ))
    assert_size_stride(arg554_1, (192, ), (1, ))
    assert_size_stride(arg555_1, (320, 2048, 1, 1), (2048, 1, 2048, 2048))
    assert_size_stride(arg556_1, (320, ), (1, ))
    assert_size_stride(arg557_1, (320, ), (1, ))
    assert_size_stride(arg558_1, (384, 2048, 1, 1), (2048, 1, 2048, 2048))
    assert_size_stride(arg559_1, (384, ), (1, ))
    assert_size_stride(arg560_1, (384, ), (1, ))
    assert_size_stride(arg561_1, (384, 384, 1, 3), (1152, 3, 3, 1))
    assert_size_stride(arg562_1, (384, ), (1, ))
    assert_size_stride(arg563_1, (384, ), (1, ))
    assert_size_stride(arg564_1, (384, 384, 3, 1), (1152, 3, 1, 1))
    assert_size_stride(arg565_1, (384, ), (1, ))
    assert_size_stride(arg566_1, (384, ), (1, ))
    assert_size_stride(arg567_1, (448, 2048, 1, 1), (2048, 1, 2048, 2048))
    assert_size_stride(arg568_1, (448, ), (1, ))
    assert_size_stride(arg569_1, (448, ), (1, ))
    assert_size_stride(arg570_1, (384, 448, 3, 3), (4032, 9, 3, 1))
    assert_size_stride(arg571_1, (384, ), (1, ))
    assert_size_stride(arg572_1, (384, ), (1, ))
    assert_size_stride(arg573_1, (384, 384, 1, 3), (1152, 3, 3, 1))
    assert_size_stride(arg574_1, (384, ), (1, ))
    assert_size_stride(arg575_1, (384, ), (1, ))
    assert_size_stride(arg576_1, (384, 384, 3, 1), (1152, 3, 1, 1))
    assert_size_stride(arg577_1, (384, ), (1, ))
    assert_size_stride(arg578_1, (384, ), (1, ))
    assert_size_stride(arg579_1, (192, 2048, 1, 1), (2048, 1, 2048, 2048))
    assert_size_stride(arg580_1, (192, ), (1, ))
    assert_size_stride(arg581_1, (192, ), (1, ))
    assert_size_stride(arg582_1, (1000, 2048), (2048, 1))
    assert_size_stride(arg583_1, (1000, ), (1, ))
    assert_size_stride(arg584_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg585_1, (32, ), (1, ))
    assert_size_stride(arg586_1, (32, ), (1, ))
    assert_size_stride(arg587_1, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg588_1, (32, ), (1, ))
    assert_size_stride(arg589_1, (32, ), (1, ))
    assert_size_stride(arg590_1, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg591_1, (64, ), (1, ))
    assert_size_stride(arg592_1, (64, ), (1, ))
    assert_size_stride(arg593_1, (80, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(arg594_1, (80, ), (1, ))
    assert_size_stride(arg595_1, (80, ), (1, ))
    assert_size_stride(arg596_1, (192, 80, 3, 3), (720, 9, 3, 1))
    assert_size_stride(arg597_1, (192, ), (1, ))
    assert_size_stride(arg598_1, (192, ), (1, ))
    assert_size_stride(arg599_1, (64, 192, 1, 1), (192, 1, 192, 192))
    assert_size_stride(arg600_1, (64, ), (1, ))
    assert_size_stride(arg601_1, (64, ), (1, ))
    assert_size_stride(arg602_1, (48, 192, 1, 1), (192, 1, 192, 192))
    assert_size_stride(arg603_1, (48, ), (1, ))
    assert_size_stride(arg604_1, (48, ), (1, ))
    assert_size_stride(arg605_1, (64, 48, 5, 5), (1200, 25, 5, 1))
    assert_size_stride(arg606_1, (64, ), (1, ))
    assert_size_stride(arg607_1, (64, ), (1, ))
    assert_size_stride(arg608_1, (64, 192, 1, 1), (192, 1, 192, 192))
    assert_size_stride(arg609_1, (64, ), (1, ))
    assert_size_stride(arg610_1, (64, ), (1, ))
    assert_size_stride(arg611_1, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg612_1, (96, ), (1, ))
    assert_size_stride(arg613_1, (96, ), (1, ))
    assert_size_stride(arg614_1, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(arg615_1, (96, ), (1, ))
    assert_size_stride(arg616_1, (96, ), (1, ))
    assert_size_stride(arg617_1, (32, 192, 1, 1), (192, 1, 192, 192))
    assert_size_stride(arg618_1, (32, ), (1, ))
    assert_size_stride(arg619_1, (32, ), (1, ))
    assert_size_stride(arg620_1, (64, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg621_1, (64, ), (1, ))
    assert_size_stride(arg622_1, (64, ), (1, ))
    assert_size_stride(arg623_1, (48, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg624_1, (48, ), (1, ))
    assert_size_stride(arg625_1, (48, ), (1, ))
    assert_size_stride(arg626_1, (64, 48, 5, 5), (1200, 25, 5, 1))
    assert_size_stride(arg627_1, (64, ), (1, ))
    assert_size_stride(arg628_1, (64, ), (1, ))
    assert_size_stride(arg629_1, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg630_1, (64, ), (1, ))
    assert_size_stride(arg631_1, (64, ), (1, ))
    assert_size_stride(arg632_1, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg633_1, (96, ), (1, ))
    assert_size_stride(arg634_1, (96, ), (1, ))
    assert_size_stride(arg635_1, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(arg636_1, (96, ), (1, ))
    assert_size_stride(arg637_1, (96, ), (1, ))
    assert_size_stride(arg638_1, (64, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg639_1, (64, ), (1, ))
    assert_size_stride(arg640_1, (64, ), (1, ))
    assert_size_stride(arg641_1, (64, 288, 1, 1), (288, 1, 288, 288))
    assert_size_stride(arg642_1, (64, ), (1, ))
    assert_size_stride(arg643_1, (64, ), (1, ))
    assert_size_stride(arg644_1, (48, 288, 1, 1), (288, 1, 288, 288))
    assert_size_stride(arg645_1, (48, ), (1, ))
    assert_size_stride(arg646_1, (48, ), (1, ))
    assert_size_stride(arg647_1, (64, 48, 5, 5), (1200, 25, 5, 1))
    assert_size_stride(arg648_1, (64, ), (1, ))
    assert_size_stride(arg649_1, (64, ), (1, ))
    assert_size_stride(arg650_1, (64, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(arg651_1, (64, ), (1, ))
    assert_size_stride(arg652_1, (64, ), (1, ))
    assert_size_stride(arg653_1, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg654_1, (96, ), (1, ))
    assert_size_stride(arg655_1, (96, ), (1, ))
    assert_size_stride(arg656_1, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(arg657_1, (96, ), (1, ))
    assert_size_stride(arg658_1, (96, ), (1, ))
    assert_size_stride(arg659_1, (64, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(arg660_1, (64, ), (1, ))
    assert_size_stride(arg661_1, (64, ), (1, ))
    assert_size_stride(arg662_1, (384, 288, 3, 3), (2592, 9, 3, 1))
    assert_size_stride(arg663_1, (384, ), (1, ))
    assert_size_stride(arg664_1, (384, ), (1, ))
    assert_size_stride(arg665_1, (64, 288, 1, 1), (288, 1, 288, 288))
    assert_size_stride(arg666_1, (64, ), (1, ))
    assert_size_stride(arg667_1, (64, ), (1, ))
    assert_size_stride(arg668_1, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg669_1, (96, ), (1, ))
    assert_size_stride(arg670_1, (96, ), (1, ))
    assert_size_stride(arg671_1, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(arg672_1, (96, ), (1, ))
    assert_size_stride(arg673_1, (96, ), (1, ))
    assert_size_stride(arg674_1, (192, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(arg675_1, (192, ), (1, ))
    assert_size_stride(arg676_1, (192, ), (1, ))
    assert_size_stride(arg677_1, (128, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(arg678_1, (128, ), (1, ))
    assert_size_stride(arg679_1, (128, ), (1, ))
    assert_size_stride(arg680_1, (128, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(arg681_1, (128, ), (1, ))
    assert_size_stride(arg682_1, (128, ), (1, ))
    assert_size_stride(arg683_1, (192, 128, 7, 1), (896, 7, 1, 1))
    assert_size_stride(arg684_1, (192, ), (1, ))
    assert_size_stride(arg685_1, (192, ), (1, ))
    assert_size_stride(arg686_1, (128, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(arg687_1, (128, ), (1, ))
    assert_size_stride(arg688_1, (128, ), (1, ))
    assert_size_stride(arg689_1, (128, 128, 7, 1), (896, 7, 1, 1))
    assert_size_stride(arg690_1, (128, ), (1, ))
    assert_size_stride(arg691_1, (128, ), (1, ))
    assert_size_stride(arg692_1, (128, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(arg693_1, (128, ), (1, ))
    assert_size_stride(arg694_1, (128, ), (1, ))
    assert_size_stride(arg695_1, (128, 128, 7, 1), (896, 7, 1, 1))
    assert_size_stride(arg696_1, (128, ), (1, ))
    assert_size_stride(arg697_1, (128, ), (1, ))
    assert_size_stride(arg698_1, (192, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(arg699_1, (192, ), (1, ))
    assert_size_stride(arg700_1, (192, ), (1, ))
    assert_size_stride(arg701_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg702_1, (192, ), (1, ))
    assert_size_stride(arg703_1, (192, ), (1, ))
    assert_size_stride(arg704_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg705_1, (192, ), (1, ))
    assert_size_stride(arg706_1, (192, ), (1, ))
    assert_size_stride(arg707_1, (160, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(arg708_1, (160, ), (1, ))
    assert_size_stride(arg709_1, (160, ), (1, ))
    assert_size_stride(arg710_1, (160, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(arg711_1, (160, ), (1, ))
    assert_size_stride(arg712_1, (160, ), (1, ))
    assert_size_stride(arg713_1, (192, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(arg714_1, (192, ), (1, ))
    assert_size_stride(arg715_1, (192, ), (1, ))
    assert_size_stride(arg716_1, (160, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg717_1, (160, ), (1, ))
    assert_size_stride(arg718_1, (160, ), (1, ))
    assert_size_stride(arg719_1, (160, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(arg720_1, (160, ), (1, ))
    assert_size_stride(arg721_1, (160, ), (1, ))
    assert_size_stride(arg722_1, (160, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(arg723_1, (160, ), (1, ))
    assert_size_stride(arg724_1, (160, ), (1, ))
    assert_size_stride(arg725_1, (160, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(arg726_1, (160, ), (1, ))
    assert_size_stride(arg727_1, (160, ), (1, ))
    assert_size_stride(arg728_1, (192, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(arg729_1, (192, ), (1, ))
    assert_size_stride(arg730_1, (192, ), (1, ))
    assert_size_stride(arg731_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg732_1, (192, ), (1, ))
    assert_size_stride(arg733_1, (192, ), (1, ))
    assert_size_stride(arg734_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg735_1, (192, ), (1, ))
    assert_size_stride(arg736_1, (192, ), (1, ))
    assert_size_stride(arg737_1, (160, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg738_1, (160, ), (1, ))
    assert_size_stride(arg739_1, (160, ), (1, ))
    assert_size_stride(arg740_1, (160, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(arg741_1, (160, ), (1, ))
    assert_size_stride(arg742_1, (160, ), (1, ))
    assert_size_stride(arg743_1, (192, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(arg744_1, (192, ), (1, ))
    assert_size_stride(arg745_1, (192, ), (1, ))
    assert_size_stride(arg746_1, (160, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(arg747_1, (160, ), (1, ))
    assert_size_stride(arg748_1, (160, ), (1, ))
    assert_size_stride(arg749_1, (160, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(arg750_1, (160, ), (1, ))
    assert_size_stride(arg751_1, (160, ), (1, ))
    assert_size_stride(arg752_1, (160, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(arg753_1, (160, ), (1, ))
    assert_size_stride(arg754_1, (160, ), (1, ))
    assert_size_stride(arg755_1, (160, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(arg756_1, (160, ), (1, ))
    assert_size_stride(arg757_1, (160, ), (1, ))
    assert_size_stride(arg758_1, (192, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(arg759_1, (192, ), (1, ))
    assert_size_stride(arg760_1, (192, ), (1, ))
    assert_size_stride(arg761_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg762_1, (192, ), (1, ))
    assert_size_stride(arg763_1, (192, ), (1, ))
    assert_size_stride(arg764_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg765_1, (192, ), (1, ))
    assert_size_stride(arg766_1, (192, ), (1, ))
    assert_size_stride(arg767_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg768_1, (192, ), (1, ))
    assert_size_stride(arg769_1, (192, ), (1, ))
    assert_size_stride(arg770_1, (192, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(arg771_1, (192, ), (1, ))
    assert_size_stride(arg772_1, (192, ), (1, ))
    assert_size_stride(arg773_1, (192, 192, 7, 1), (1344, 7, 1, 1))
    assert_size_stride(arg774_1, (192, ), (1, ))
    assert_size_stride(arg775_1, (192, ), (1, ))
    assert_size_stride(arg776_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg777_1, (192, ), (1, ))
    assert_size_stride(arg778_1, (192, ), (1, ))
    assert_size_stride(arg779_1, (192, 192, 7, 1), (1344, 7, 1, 1))
    assert_size_stride(arg780_1, (192, ), (1, ))
    assert_size_stride(arg781_1, (192, ), (1, ))
    assert_size_stride(arg782_1, (192, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(arg783_1, (192, ), (1, ))
    assert_size_stride(arg784_1, (192, ), (1, ))
    assert_size_stride(arg785_1, (192, 192, 7, 1), (1344, 7, 1, 1))
    assert_size_stride(arg786_1, (192, ), (1, ))
    assert_size_stride(arg787_1, (192, ), (1, ))
    assert_size_stride(arg788_1, (192, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(arg789_1, (192, ), (1, ))
    assert_size_stride(arg790_1, (192, ), (1, ))
    assert_size_stride(arg791_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg792_1, (192, ), (1, ))
    assert_size_stride(arg793_1, (192, ), (1, ))
    assert_size_stride(arg794_1, (128, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(arg795_1, (128, ), (1, ))
    assert_size_stride(arg796_1, (128, ), (1, ))
    assert_size_stride(arg797_1, (768, 128, 5, 5), (3200, 25, 5, 1))
    assert_size_stride(arg798_1, (768, ), (1, ))
    assert_size_stride(arg799_1, (768, ), (1, ))
    assert_size_stride(arg800_1, (1000, 768), (768, 1))
    assert_size_stride(arg801_1, (1000, ), (1, ))
    assert_size_stride(arg802_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg803_1, (192, ), (1, ))
    assert_size_stride(arg804_1, (192, ), (1, ))
    assert_size_stride(arg805_1, (320, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg806_1, (320, ), (1, ))
    assert_size_stride(arg807_1, (320, ), (1, ))
    assert_size_stride(arg808_1, (192, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(arg809_1, (192, ), (1, ))
    assert_size_stride(arg810_1, (192, ), (1, ))
    assert_size_stride(arg811_1, (192, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(arg812_1, (192, ), (1, ))
    assert_size_stride(arg813_1, (192, ), (1, ))
    assert_size_stride(arg814_1, (192, 192, 7, 1), (1344, 7, 1, 1))
    assert_size_stride(arg815_1, (192, ), (1, ))
    assert_size_stride(arg816_1, (192, ), (1, ))
    assert_size_stride(arg817_1, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg818_1, (192, ), (1, ))
    assert_size_stride(arg819_1, (192, ), (1, ))
    assert_size_stride(arg820_1, (320, 1280, 1, 1), (1280, 1, 1280, 1280))
    assert_size_stride(arg821_1, (320, ), (1, ))
    assert_size_stride(arg822_1, (320, ), (1, ))
    assert_size_stride(arg823_1, (384, 1280, 1, 1), (1280, 1, 1280, 1280))
    assert_size_stride(arg824_1, (384, ), (1, ))
    assert_size_stride(arg825_1, (384, ), (1, ))
    assert_size_stride(arg826_1, (384, 384, 1, 3), (1152, 3, 3, 1))
    assert_size_stride(arg827_1, (384, ), (1, ))
    assert_size_stride(arg828_1, (384, ), (1, ))
    assert_size_stride(arg829_1, (384, 384, 3, 1), (1152, 3, 1, 1))
    assert_size_stride(arg830_1, (384, ), (1, ))
    assert_size_stride(arg831_1, (384, ), (1, ))
    assert_size_stride(arg832_1, (448, 1280, 1, 1), (1280, 1, 1280, 1280))
    assert_size_stride(arg833_1, (448, ), (1, ))
    assert_size_stride(arg834_1, (448, ), (1, ))
    assert_size_stride(arg835_1, (384, 448, 3, 3), (4032, 9, 3, 1))
    assert_size_stride(arg836_1, (384, ), (1, ))
    assert_size_stride(arg837_1, (384, ), (1, ))
    assert_size_stride(arg838_1, (384, 384, 1, 3), (1152, 3, 3, 1))
    assert_size_stride(arg839_1, (384, ), (1, ))
    assert_size_stride(arg840_1, (384, ), (1, ))
    assert_size_stride(arg841_1, (384, 384, 3, 1), (1152, 3, 1, 1))
    assert_size_stride(arg842_1, (384, ), (1, ))
    assert_size_stride(arg843_1, (384, ), (1, ))
    assert_size_stride(arg844_1, (192, 1280, 1, 1), (1280, 1, 1280, 1280))
    assert_size_stride(arg845_1, (192, ), (1, ))
    assert_size_stride(arg846_1, (192, ), (1, ))
    assert_size_stride(arg847_1, (320, 2048, 1, 1), (2048, 1, 2048, 2048))
    assert_size_stride(arg848_1, (320, ), (1, ))
    assert_size_stride(arg849_1, (320, ), (1, ))
    assert_size_stride(arg850_1, (384, 2048, 1, 1), (2048, 1, 2048, 2048))
    assert_size_stride(arg851_1, (384, ), (1, ))
    assert_size_stride(arg852_1, (384, ), (1, ))
    assert_size_stride(arg853_1, (384, 384, 1, 3), (1152, 3, 3, 1))
    assert_size_stride(arg854_1, (384, ), (1, ))
    assert_size_stride(arg855_1, (384, ), (1, ))
    assert_size_stride(arg856_1, (384, 384, 3, 1), (1152, 3, 1, 1))
    assert_size_stride(arg857_1, (384, ), (1, ))
    assert_size_stride(arg858_1, (384, ), (1, ))
    assert_size_stride(arg859_1, (448, 2048, 1, 1), (2048, 1, 2048, 2048))
    assert_size_stride(arg860_1, (448, ), (1, ))
    assert_size_stride(arg861_1, (448, ), (1, ))
    assert_size_stride(arg862_1, (384, 448, 3, 3), (4032, 9, 3, 1))
    assert_size_stride(arg863_1, (384, ), (1, ))
    assert_size_stride(arg864_1, (384, ), (1, ))
    assert_size_stride(arg865_1, (384, 384, 1, 3), (1152, 3, 3, 1))
    assert_size_stride(arg866_1, (384, ), (1, ))
    assert_size_stride(arg867_1, (384, ), (1, ))
    assert_size_stride(arg868_1, (384, 384, 3, 1), (1152, 3, 1, 1))
    assert_size_stride(arg869_1, (384, ), (1, ))
    assert_size_stride(arg870_1, (384, ), (1, ))
    assert_size_stride(arg871_1, (192, 2048, 1, 1), (2048, 1, 2048, 2048))
    assert_size_stride(arg872_1, (192, ), (1, ))
    assert_size_stride(arg873_1, (192, ), (1, ))
    assert_size_stride(arg874_1, (1000, 2048), (2048, 1))
    assert_size_stride(arg875_1, (1000, ), (1, ))
    assert_size_stride(arg876_1, (), ())
    assert_size_stride(arg877_1, (), ())
    assert_size_stride(arg878_1, (), ())
    assert_size_stride(arg879_1, (), ())
    assert_size_stride(arg880_1, (), ())
    assert_size_stride(arg881_1, (), ())
    assert_size_stride(arg882_1, (), ())
    assert_size_stride(arg883_1, (), ())
    assert_size_stride(arg884_1, (), ())
    assert_size_stride(arg885_1, (), ())
    assert_size_stride(arg886_1, (), ())
    assert_size_stride(arg887_1, (), ())
    assert_size_stride(arg888_1, (), ())
    assert_size_stride(arg889_1, (), ())
    assert_size_stride(arg890_1, (), ())
    assert_size_stride(arg891_1, (), ())
    assert_size_stride(arg892_1, (), ())
    assert_size_stride(arg893_1, (), ())
    assert_size_stride(arg894_1, (), ())
    assert_size_stride(arg895_1, (), ())
    assert_size_stride(arg896_1, (), ())
    assert_size_stride(arg897_1, (), ())
    assert_size_stride(arg898_1, (), ())
    assert_size_stride(arg899_1, (), ())
    assert_size_stride(arg900_1, (), ())
    assert_size_stride(arg901_1, (), ())
    assert_size_stride(arg902_1, (), ())
    assert_size_stride(arg903_1, (), ())
    assert_size_stride(arg904_1, (), ())
    assert_size_stride(arg905_1, (), ())
    assert_size_stride(arg906_1, (), ())
    assert_size_stride(arg907_1, (), ())
    assert_size_stride(arg908_1, (), ())
    assert_size_stride(arg909_1, (), ())
    assert_size_stride(arg910_1, (), ())
    assert_size_stride(arg911_1, (), ())
    assert_size_stride(arg912_1, (), ())
    assert_size_stride(arg913_1, (), ())
    assert_size_stride(arg914_1, (), ())
    assert_size_stride(arg915_1, (), ())
    assert_size_stride(arg916_1, (), ())
    assert_size_stride(arg917_1, (), ())
    assert_size_stride(arg918_1, (), ())
    assert_size_stride(arg919_1, (), ())
    assert_size_stride(arg920_1, (), ())
    assert_size_stride(arg921_1, (), ())
    assert_size_stride(arg922_1, (), ())
    assert_size_stride(arg923_1, (), ())
    assert_size_stride(arg924_1, (), ())
    assert_size_stride(arg925_1, (), ())
    assert_size_stride(arg926_1, (), ())
    assert_size_stride(arg927_1, (), ())
    assert_size_stride(arg928_1, (), ())
    assert_size_stride(arg929_1, (), ())
    assert_size_stride(arg930_1, (), ())
    assert_size_stride(arg931_1, (), ())
    assert_size_stride(arg932_1, (), ())
    assert_size_stride(arg933_1, (), ())
    assert_size_stride(arg934_1, (), ())
    assert_size_stride(arg935_1, (), ())
    assert_size_stride(arg936_1, (), ())
    assert_size_stride(arg937_1, (), ())
    assert_size_stride(arg938_1, (), ())
    assert_size_stride(arg939_1, (), ())
    assert_size_stride(arg940_1, (), ())
    assert_size_stride(arg941_1, (), ())
    assert_size_stride(arg942_1, (), ())
    assert_size_stride(arg943_1, (), ())
    assert_size_stride(arg944_1, (), ())
    assert_size_stride(arg945_1, (), ())
    assert_size_stride(arg946_1, (), ())
    assert_size_stride(arg947_1, (), ())
    assert_size_stride(arg948_1, (), ())
    assert_size_stride(arg949_1, (), ())
    assert_size_stride(arg950_1, (), ())
    assert_size_stride(arg951_1, (), ())
    assert_size_stride(arg952_1, (), ())
    assert_size_stride(arg953_1, (), ())
    assert_size_stride(arg954_1, (), ())
    assert_size_stride(arg955_1, (), ())
    assert_size_stride(arg956_1, (), ())
    assert_size_stride(arg957_1, (), ())
    assert_size_stride(arg958_1, (), ())
    assert_size_stride(arg959_1, (), ())
    assert_size_stride(arg960_1, (), ())
    assert_size_stride(arg961_1, (), ())
    assert_size_stride(arg962_1, (), ())
    assert_size_stride(arg963_1, (), ())
    assert_size_stride(arg964_1, (), ())
    assert_size_stride(arg965_1, (), ())
    assert_size_stride(arg966_1, (), ())
    assert_size_stride(arg967_1, (), ())
    assert_size_stride(arg968_1, (), ())
    assert_size_stride(arg969_1, (), ())
    assert_size_stride(arg970_1, (), ())
    assert_size_stride(arg971_1, (), ())
    assert_size_stride(arg972_1, (), ())
    assert_size_stride(arg973_1, (), ())
    assert_size_stride(arg974_1, (), ())
    assert_size_stride(arg975_1, (), ())
    assert_size_stride(arg976_1, (), ())
    assert_size_stride(arg977_1, (), ())
    assert_size_stride(arg978_1, (), ())
    assert_size_stride(arg979_1, (), ())
    assert_size_stride(arg980_1, (), ())
    assert_size_stride(arg981_1, (), ())
    assert_size_stride(arg982_1, (), ())
    assert_size_stride(arg983_1, (), ())
    assert_size_stride(arg984_1, (), ())
    assert_size_stride(arg985_1, (), ())
    assert_size_stride(arg986_1, (), ())
    assert_size_stride(arg987_1, (), ())
    assert_size_stride(arg988_1, (), ())
    assert_size_stride(arg989_1, (), ())
    assert_size_stride(arg990_1, (), ())
    assert_size_stride(arg991_1, (), ())
    assert_size_stride(arg992_1, (), ())
    assert_size_stride(arg993_1, (), ())
    assert_size_stride(arg994_1, (), ())
    assert_size_stride(arg995_1, (), ())
    assert_size_stride(arg996_1, (), ())
    assert_size_stride(arg997_1, (), ())
    assert_size_stride(arg998_1, (), ())
    assert_size_stride(arg999_1, (), ())
    assert_size_stride(arg1000_1, (), ())
    assert_size_stride(arg1001_1, (), ())
    assert_size_stride(arg1002_1, (), ())
    assert_size_stride(arg1003_1, (), ())
    assert_size_stride(arg1004_1, (), ())
    assert_size_stride(arg1005_1, (), ())
    assert_size_stride(arg1006_1, (), ())
    assert_size_stride(arg1007_1, (), ())
    assert_size_stride(arg1008_1, (), ())
    assert_size_stride(arg1009_1, (), ())
    assert_size_stride(arg1010_1, (), ())
    assert_size_stride(arg1011_1, (), ())
    assert_size_stride(arg1012_1, (), ())
    assert_size_stride(arg1013_1, (), ())
    assert_size_stride(arg1014_1, (), ())
    assert_size_stride(arg1015_1, (), ())
    assert_size_stride(arg1016_1, (), ())
    assert_size_stride(arg1017_1, (), ())
    assert_size_stride(arg1018_1, (), ())
    assert_size_stride(arg1019_1, (), ())
    assert_size_stride(arg1020_1, (), ())
    assert_size_stride(arg1021_1, (), ())
    assert_size_stride(arg1022_1, (), ())
    assert_size_stride(arg1023_1, (), ())
    assert_size_stride(arg1024_1, (), ())
    assert_size_stride(arg1025_1, (), ())
    assert_size_stride(arg1026_1, (), ())
    assert_size_stride(arg1027_1, (), ())
    assert_size_stride(arg1028_1, (), ())
    assert_size_stride(arg1029_1, (), ())
    assert_size_stride(arg1030_1, (), ())
    assert_size_stride(arg1031_1, (), ())
    assert_size_stride(arg1032_1, (), ())
    assert_size_stride(arg1033_1, (), ())
    assert_size_stride(arg1034_1, (), ())
    assert_size_stride(arg1035_1, (), ())
    assert_size_stride(arg1036_1, (), ())
    assert_size_stride(arg1037_1, (), ())
    assert_size_stride(arg1038_1, (), ())
    assert_size_stride(arg1039_1, (), ())
    assert_size_stride(arg1040_1, (), ())
    assert_size_stride(arg1041_1, (), ())
    assert_size_stride(arg1042_1, (), ())
    assert_size_stride(arg1043_1, (), ())
    assert_size_stride(arg1044_1, (), ())
    assert_size_stride(arg1045_1, (), ())
    assert_size_stride(arg1046_1, (), ())
    assert_size_stride(arg1047_1, (), ())
    assert_size_stride(arg1048_1, (), ())
    assert_size_stride(arg1049_1, (), ())
    assert_size_stride(arg1050_1, (), ())
    assert_size_stride(arg1051_1, (), ())
    assert_size_stride(arg1052_1, (), ())
    assert_size_stride(arg1053_1, (), ())
    assert_size_stride(arg1054_1, (), ())
    assert_size_stride(arg1055_1, (), ())
    assert_size_stride(arg1056_1, (), ())
    assert_size_stride(arg1057_1, (), ())
    assert_size_stride(arg1058_1, (), ())
    assert_size_stride(arg1059_1, (), ())
    assert_size_stride(arg1060_1, (), ())
    assert_size_stride(arg1061_1, (), ())
    assert_size_stride(arg1062_1, (), ())
    assert_size_stride(arg1063_1, (), ())
    assert_size_stride(arg1064_1, (), ())
    assert_size_stride(arg1065_1, (), ())
    assert_size_stride(arg1066_1, (), ())
    assert_size_stride(arg1067_1, (), ())
    assert_size_stride(arg1068_1, (), ())
    assert_size_stride(arg1069_1, (), ())
    assert_size_stride(arg1070_1, (), ())
    assert_size_stride(arg1071_1, (), ())
    assert_size_stride(arg1072_1, (), ())
    assert_size_stride(arg1073_1, (), ())
    assert_size_stride(arg1074_1, (), ())
    assert_size_stride(arg1075_1, (), ())
    assert_size_stride(arg1076_1, (), ())
    assert_size_stride(arg1077_1, (), ())
    assert_size_stride(arg1078_1, (), ())
    assert_size_stride(arg1079_1, (), ())
    assert_size_stride(arg1080_1, (), ())
    assert_size_stride(arg1081_1, (), ())
    assert_size_stride(arg1082_1, (), ())
    assert_size_stride(arg1083_1, (), ())
    assert_size_stride(arg1084_1, (), ())
    assert_size_stride(arg1085_1, (), ())
    assert_size_stride(arg1086_1, (), ())
    assert_size_stride(arg1087_1, (), ())
    assert_size_stride(arg1088_1, (), ())
    assert_size_stride(arg1089_1, (), ())
    assert_size_stride(arg1090_1, (), ())
    assert_size_stride(arg1091_1, (), ())
    assert_size_stride(arg1092_1, (), ())
    assert_size_stride(arg1093_1, (), ())
    assert_size_stride(arg1094_1, (), ())
    assert_size_stride(arg1095_1, (), ())
    assert_size_stride(arg1096_1, (), ())
    assert_size_stride(arg1097_1, (), ())
    assert_size_stride(arg1098_1, (), ())
    assert_size_stride(arg1099_1, (), ())
    assert_size_stride(arg1100_1, (), ())
    assert_size_stride(arg1101_1, (), ())
    assert_size_stride(arg1102_1, (), ())
    assert_size_stride(arg1103_1, (), ())
    assert_size_stride(arg1104_1, (), ())
    assert_size_stride(arg1105_1, (), ())
    assert_size_stride(arg1106_1, (), ())
    assert_size_stride(arg1107_1, (), ())
    assert_size_stride(arg1108_1, (), ())
    assert_size_stride(arg1109_1, (), ())
    assert_size_stride(arg1110_1, (), ())
    assert_size_stride(arg1111_1, (), ())
    assert_size_stride(arg1112_1, (), ())
    assert_size_stride(arg1113_1, (), ())
    assert_size_stride(arg1114_1, (), ())
    assert_size_stride(arg1115_1, (), ())
    assert_size_stride(arg1116_1, (), ())
    assert_size_stride(arg1117_1, (), ())
    assert_size_stride(arg1118_1, (), ())
    assert_size_stride(arg1119_1, (), ())
    assert_size_stride(arg1120_1, (), ())
    assert_size_stride(arg1121_1, (), ())
    assert_size_stride(arg1122_1, (), ())
    assert_size_stride(arg1123_1, (), ())
    assert_size_stride(arg1124_1, (), ())
    assert_size_stride(arg1125_1, (), ())
    assert_size_stride(arg1126_1, (), ())
    assert_size_stride(arg1127_1, (), ())
    assert_size_stride(arg1128_1, (), ())
    assert_size_stride(arg1129_1, (), ())
    assert_size_stride(arg1130_1, (), ())
    assert_size_stride(arg1131_1, (), ())
    assert_size_stride(arg1132_1, (), ())
    assert_size_stride(arg1133_1, (), ())
    assert_size_stride(arg1134_1, (), ())
    assert_size_stride(arg1135_1, (), ())
    assert_size_stride(arg1136_1, (), ())
    assert_size_stride(arg1137_1, (), ())
    assert_size_stride(arg1138_1, (), ())
    assert_size_stride(arg1139_1, (), ())
    assert_size_stride(arg1140_1, (), ())
    assert_size_stride(arg1141_1, (), ())
    assert_size_stride(arg1142_1, (), ())
    assert_size_stride(arg1143_1, (), ())
    assert_size_stride(arg1144_1, (), ())
    assert_size_stride(arg1145_1, (), ())
    assert_size_stride(arg1146_1, (), ())
    assert_size_stride(arg1147_1, (), ())
    assert_size_stride(arg1148_1, (), ())
    assert_size_stride(arg1149_1, (), ())
    assert_size_stride(arg1150_1, (), ())
    assert_size_stride(arg1151_1, (), ())
    assert_size_stride(arg1152_1, (), ())
    assert_size_stride(arg1153_1, (), ())
    assert_size_stride(arg1154_1, (), ())
    assert_size_stride(arg1155_1, (), ())
    assert_size_stride(arg1156_1, (), ())
    assert_size_stride(arg1157_1, (), ())
    assert_size_stride(arg1158_1, (), ())
    assert_size_stride(arg1159_1, (), ())
    assert_size_stride(arg1160_1, (), ())
    assert_size_stride(arg1161_1, (), ())
    assert_size_stride(arg1162_1, (), ())
    assert_size_stride(arg1163_1, (), ())
    assert_size_stride(arg1164_1, (), ())
    assert_size_stride(arg1165_1, (), ())
    assert_size_stride(arg1166_1, (), ())
    assert_size_stride(arg1167_1, (), ())
    assert_size_stride(arg1168_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg1169_1, (32, ), (1, ))
    assert_size_stride(arg1170_1, (32, ), (1, ))
    assert_size_stride(arg1171_1, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg1172_1, (32, ), (1, ))
    assert_size_stride(arg1173_1, (32, ), (1, ))
    assert_size_stride(arg1174_1, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg1175_1, (64, ), (1, ))
    assert_size_stride(arg1176_1, (64, ), (1, ))
    assert_size_stride(arg1177_1, (80, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(arg1178_1, (80, ), (1, ))
    assert_size_stride(arg1179_1, (80, ), (1, ))
    assert_size_stride(arg1180_1, (192, 80, 3, 3), (720, 9, 3, 1))
    assert_size_stride(arg1181_1, (192, ), (1, ))
    assert_size_stride(arg1182_1, (192, ), (1, ))
    assert_size_stride(arg1183_1, (64, 192, 1, 1), (192, 1, 192, 192))
    assert_size_stride(arg1184_1, (64, ), (1, ))
    assert_size_stride(arg1185_1, (64, ), (1, ))
    assert_size_stride(arg1186_1, (48, 192, 1, 1), (192, 1, 192, 192))
    assert_size_stride(arg1187_1, (48, ), (1, ))
    assert_size_stride(arg1188_1, (48, ), (1, ))
    assert_size_stride(arg1189_1, (64, 48, 5, 5), (1200, 25, 5, 1))
    assert_size_stride(arg1190_1, (64, ), (1, ))
    assert_size_stride(arg1191_1, (64, ), (1, ))
    assert_size_stride(arg1192_1, (64, 192, 1, 1), (192, 1, 192, 192))
    assert_size_stride(arg1193_1, (64, ), (1, ))
    assert_size_stride(arg1194_1, (64, ), (1, ))
    assert_size_stride(arg1195_1, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg1196_1, (96, ), (1, ))
    assert_size_stride(arg1197_1, (96, ), (1, ))
    assert_size_stride(arg1198_1, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(arg1199_1, (96, ), (1, ))
    assert_size_stride(arg1200_1, (96, ), (1, ))
    assert_size_stride(arg1201_1, (32, 192, 1, 1), (192, 1, 192, 192))
    assert_size_stride(arg1202_1, (32, ), (1, ))
    assert_size_stride(arg1203_1, (32, ), (1, ))
    assert_size_stride(arg1204_1, (64, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg1205_1, (64, ), (1, ))
    assert_size_stride(arg1206_1, (64, ), (1, ))
    assert_size_stride(arg1207_1, (48, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg1208_1, (48, ), (1, ))
    assert_size_stride(arg1209_1, (48, ), (1, ))
    assert_size_stride(arg1210_1, (64, 48, 5, 5), (1200, 25, 5, 1))
    assert_size_stride(arg1211_1, (64, ), (1, ))
    assert_size_stride(arg1212_1, (64, ), (1, ))
    assert_size_stride(arg1213_1, (64, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg1214_1, (64, ), (1, ))
    assert_size_stride(arg1215_1, (64, ), (1, ))
    assert_size_stride(arg1216_1, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg1217_1, (96, ), (1, ))
    assert_size_stride(arg1218_1, (96, ), (1, ))
    assert_size_stride(arg1219_1, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(arg1220_1, (96, ), (1, ))
    assert_size_stride(arg1221_1, (96, ), (1, ))
    assert_size_stride(arg1222_1, (64, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg1223_1, (64, ), (1, ))
    assert_size_stride(arg1224_1, (64, ), (1, ))
    assert_size_stride(arg1225_1, (64, 288, 1, 1), (288, 1, 288, 288))
    assert_size_stride(arg1226_1, (64, ), (1, ))
    assert_size_stride(arg1227_1, (64, ), (1, ))
    assert_size_stride(arg1228_1, (48, 288, 1, 1), (288, 1, 288, 288))
    assert_size_stride(arg1229_1, (48, ), (1, ))
    assert_size_stride(arg1230_1, (48, ), (1, ))
    assert_size_stride(arg1231_1, (64, 48, 5, 5), (1200, 25, 5, 1))
    assert_size_stride(arg1232_1, (64, ), (1, ))
    assert_size_stride(arg1233_1, (64, ), (1, ))
    assert_size_stride(arg1234_1, (64, 288, 1, 1), (288, 1, 288, 288))
    assert_size_stride(arg1235_1, (64, ), (1, ))
    assert_size_stride(arg1236_1, (64, ), (1, ))
    assert_size_stride(arg1237_1, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg1238_1, (96, ), (1, ))
    assert_size_stride(arg1239_1, (96, ), (1, ))
    assert_size_stride(arg1240_1, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(arg1241_1, (96, ), (1, ))
    assert_size_stride(arg1242_1, (96, ), (1, ))
    assert_size_stride(arg1243_1, (64, 288, 1, 1), (288, 1, 288, 288))
    assert_size_stride(arg1244_1, (64, ), (1, ))
    assert_size_stride(arg1245_1, (64, ), (1, ))
    assert_size_stride(arg1246_1, (384, 288, 3, 3), (2592, 9, 3, 1))
    assert_size_stride(arg1247_1, (384, ), (1, ))
    assert_size_stride(arg1248_1, (384, ), (1, ))
    assert_size_stride(arg1249_1, (64, 288, 1, 1), (288, 1, 288, 288))
    assert_size_stride(arg1250_1, (64, ), (1, ))
    assert_size_stride(arg1251_1, (64, ), (1, ))
    assert_size_stride(arg1252_1, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg1253_1, (96, ), (1, ))
    assert_size_stride(arg1254_1, (96, ), (1, ))
    assert_size_stride(arg1255_1, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(arg1256_1, (96, ), (1, ))
    assert_size_stride(arg1257_1, (96, ), (1, ))
    assert_size_stride(arg1258_1, (192, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(arg1259_1, (192, ), (1, ))
    assert_size_stride(arg1260_1, (192, ), (1, ))
    assert_size_stride(arg1261_1, (128, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(arg1262_1, (128, ), (1, ))
    assert_size_stride(arg1263_1, (128, ), (1, ))
    assert_size_stride(arg1264_1, (128, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(arg1265_1, (128, ), (1, ))
    assert_size_stride(arg1266_1, (128, ), (1, ))
    assert_size_stride(arg1267_1, (192, 128, 7, 1), (896, 7, 1, 1))
    assert_size_stride(arg1268_1, (192, ), (1, ))
    assert_size_stride(arg1269_1, (192, ), (1, ))
    assert_size_stride(arg1270_1, (128, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(arg1271_1, (128, ), (1, ))
    assert_size_stride(arg1272_1, (128, ), (1, ))
    assert_size_stride(arg1273_1, (128, 128, 7, 1), (896, 7, 1, 1))
    assert_size_stride(arg1274_1, (128, ), (1, ))
    assert_size_stride(arg1275_1, (128, ), (1, ))
    assert_size_stride(arg1276_1, (128, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(arg1277_1, (128, ), (1, ))
    assert_size_stride(arg1278_1, (128, ), (1, ))
    assert_size_stride(arg1279_1, (128, 128, 7, 1), (896, 7, 1, 1))
    assert_size_stride(arg1280_1, (128, ), (1, ))
    assert_size_stride(arg1281_1, (128, ), (1, ))
    assert_size_stride(arg1282_1, (192, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(arg1283_1, (192, ), (1, ))
    assert_size_stride(arg1284_1, (192, ), (1, ))
    assert_size_stride(arg1285_1, (192, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(arg1286_1, (192, ), (1, ))
    assert_size_stride(arg1287_1, (192, ), (1, ))
    assert_size_stride(arg1288_1, (192, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(arg1289_1, (192, ), (1, ))
    assert_size_stride(arg1290_1, (192, ), (1, ))
    assert_size_stride(arg1291_1, (160, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(arg1292_1, (160, ), (1, ))
    assert_size_stride(arg1293_1, (160, ), (1, ))
    assert_size_stride(arg1294_1, (160, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(arg1295_1, (160, ), (1, ))
    assert_size_stride(arg1296_1, (160, ), (1, ))
    assert_size_stride(arg1297_1, (192, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(arg1298_1, (192, ), (1, ))
    assert_size_stride(arg1299_1, (192, ), (1, ))
    assert_size_stride(arg1300_1, (160, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(arg1301_1, (160, ), (1, ))
    assert_size_stride(arg1302_1, (160, ), (1, ))
    assert_size_stride(arg1303_1, (160, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(arg1304_1, (160, ), (1, ))
    assert_size_stride(arg1305_1, (160, ), (1, ))
    assert_size_stride(arg1306_1, (160, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(arg1307_1, (160, ), (1, ))
    assert_size_stride(arg1308_1, (160, ), (1, ))
    assert_size_stride(arg1309_1, (160, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(arg1310_1, (160, ), (1, ))
    assert_size_stride(arg1311_1, (160, ), (1, ))
    assert_size_stride(arg1312_1, (192, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(arg1313_1, (192, ), (1, ))
    assert_size_stride(arg1314_1, (192, ), (1, ))
    assert_size_stride(arg1315_1, (192, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(arg1316_1, (192, ), (1, ))
    assert_size_stride(arg1317_1, (192, ), (1, ))
    assert_size_stride(arg1318_1, (192, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(arg1319_1, (192, ), (1, ))
    assert_size_stride(arg1320_1, (192, ), (1, ))
    assert_size_stride(arg1321_1, (160, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(arg1322_1, (160, ), (1, ))
    assert_size_stride(arg1323_1, (160, ), (1, ))
    assert_size_stride(arg1324_1, (160, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(arg1325_1, (160, ), (1, ))
    assert_size_stride(arg1326_1, (160, ), (1, ))
    assert_size_stride(arg1327_1, (192, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(arg1328_1, (192, ), (1, ))
    assert_size_stride(arg1329_1, (192, ), (1, ))
    assert_size_stride(arg1330_1, (160, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(arg1331_1, (160, ), (1, ))
    assert_size_stride(arg1332_1, (160, ), (1, ))
    assert_size_stride(arg1333_1, (160, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(arg1334_1, (160, ), (1, ))
    assert_size_stride(arg1335_1, (160, ), (1, ))
    assert_size_stride(arg1336_1, (160, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(arg1337_1, (160, ), (1, ))
    assert_size_stride(arg1338_1, (160, ), (1, ))
    assert_size_stride(arg1339_1, (160, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(arg1340_1, (160, ), (1, ))
    assert_size_stride(arg1341_1, (160, ), (1, ))
    assert_size_stride(arg1342_1, (192, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(arg1343_1, (192, ), (1, ))
    assert_size_stride(arg1344_1, (192, ), (1, ))
    assert_size_stride(arg1345_1, (192, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(arg1346_1, (192, ), (1, ))
    assert_size_stride(arg1347_1, (192, ), (1, ))
    assert_size_stride(arg1348_1, (192, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(arg1349_1, (192, ), (1, ))
    assert_size_stride(arg1350_1, (192, ), (1, ))
    assert_size_stride(arg1351_1, (192, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(arg1352_1, (192, ), (1, ))
    assert_size_stride(arg1353_1, (192, ), (1, ))
    assert_size_stride(arg1354_1, (192, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(arg1355_1, (192, ), (1, ))
    assert_size_stride(arg1356_1, (192, ), (1, ))
    assert_size_stride(arg1357_1, (192, 192, 7, 1), (1344, 7, 1, 1))
    assert_size_stride(arg1358_1, (192, ), (1, ))
    assert_size_stride(arg1359_1, (192, ), (1, ))
    assert_size_stride(arg1360_1, (192, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(arg1361_1, (192, ), (1, ))
    assert_size_stride(arg1362_1, (192, ), (1, ))
    assert_size_stride(arg1363_1, (192, 192, 7, 1), (1344, 7, 1, 1))
    assert_size_stride(arg1364_1, (192, ), (1, ))
    assert_size_stride(arg1365_1, (192, ), (1, ))
    assert_size_stride(arg1366_1, (192, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(arg1367_1, (192, ), (1, ))
    assert_size_stride(arg1368_1, (192, ), (1, ))
    assert_size_stride(arg1369_1, (192, 192, 7, 1), (1344, 7, 1, 1))
    assert_size_stride(arg1370_1, (192, ), (1, ))
    assert_size_stride(arg1371_1, (192, ), (1, ))
    assert_size_stride(arg1372_1, (192, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(arg1373_1, (192, ), (1, ))
    assert_size_stride(arg1374_1, (192, ), (1, ))
    assert_size_stride(arg1375_1, (192, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(arg1376_1, (192, ), (1, ))
    assert_size_stride(arg1377_1, (192, ), (1, ))
    assert_size_stride(arg1378_1, (128, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(arg1379_1, (128, ), (1, ))
    assert_size_stride(arg1380_1, (128, ), (1, ))
    assert_size_stride(arg1381_1, (768, 128, 5, 5), (3200, 25, 5, 1))
    assert_size_stride(arg1382_1, (768, ), (1, ))
    assert_size_stride(arg1383_1, (768, ), (1, ))
    assert_size_stride(arg1384_1, (1000, 768), (768, 1))
    assert_size_stride(arg1385_1, (1000, ), (1, ))
    assert_size_stride(arg1386_1, (192, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(arg1387_1, (192, ), (1, ))
    assert_size_stride(arg1388_1, (192, ), (1, ))
    assert_size_stride(arg1389_1, (320, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg1390_1, (320, ), (1, ))
    assert_size_stride(arg1391_1, (320, ), (1, ))
    assert_size_stride(arg1392_1, (192, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(arg1393_1, (192, ), (1, ))
    assert_size_stride(arg1394_1, (192, ), (1, ))
    assert_size_stride(arg1395_1, (192, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(arg1396_1, (192, ), (1, ))
    assert_size_stride(arg1397_1, (192, ), (1, ))
    assert_size_stride(arg1398_1, (192, 192, 7, 1), (1344, 7, 1, 1))
    assert_size_stride(arg1399_1, (192, ), (1, ))
    assert_size_stride(arg1400_1, (192, ), (1, ))
    assert_size_stride(arg1401_1, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg1402_1, (192, ), (1, ))
    assert_size_stride(arg1403_1, (192, ), (1, ))
    assert_size_stride(arg1404_1, (320, 1280, 1, 1), (1280, 1, 1280, 1280))
    assert_size_stride(arg1405_1, (320, ), (1, ))
    assert_size_stride(arg1406_1, (320, ), (1, ))
    assert_size_stride(arg1407_1, (384, 1280, 1, 1), (1280, 1, 1280, 1280))
    assert_size_stride(arg1408_1, (384, ), (1, ))
    assert_size_stride(arg1409_1, (384, ), (1, ))
    assert_size_stride(arg1410_1, (384, 384, 1, 3), (1152, 3, 3, 1))
    assert_size_stride(arg1411_1, (384, ), (1, ))
    assert_size_stride(arg1412_1, (384, ), (1, ))
    assert_size_stride(arg1413_1, (384, 384, 3, 1), (1152, 3, 1, 1))
    assert_size_stride(arg1414_1, (384, ), (1, ))
    assert_size_stride(arg1415_1, (384, ), (1, ))
    assert_size_stride(arg1416_1, (448, 1280, 1, 1), (1280, 1, 1280, 1280))
    assert_size_stride(arg1417_1, (448, ), (1, ))
    assert_size_stride(arg1418_1, (448, ), (1, ))
    assert_size_stride(arg1419_1, (384, 448, 3, 3), (4032, 9, 3, 1))
    assert_size_stride(arg1420_1, (384, ), (1, ))
    assert_size_stride(arg1421_1, (384, ), (1, ))
    assert_size_stride(arg1422_1, (384, 384, 1, 3), (1152, 3, 3, 1))
    assert_size_stride(arg1423_1, (384, ), (1, ))
    assert_size_stride(arg1424_1, (384, ), (1, ))
    assert_size_stride(arg1425_1, (384, 384, 3, 1), (1152, 3, 1, 1))
    assert_size_stride(arg1426_1, (384, ), (1, ))
    assert_size_stride(arg1427_1, (384, ), (1, ))
    assert_size_stride(arg1428_1, (192, 1280, 1, 1), (1280, 1, 1280, 1280))
    assert_size_stride(arg1429_1, (192, ), (1, ))
    assert_size_stride(arg1430_1, (192, ), (1, ))
    assert_size_stride(arg1431_1, (320, 2048, 1, 1), (2048, 1, 2048, 2048))
    assert_size_stride(arg1432_1, (320, ), (1, ))
    assert_size_stride(arg1433_1, (320, ), (1, ))
    assert_size_stride(arg1434_1, (384, 2048, 1, 1), (2048, 1, 2048, 2048))
    assert_size_stride(arg1435_1, (384, ), (1, ))
    assert_size_stride(arg1436_1, (384, ), (1, ))
    assert_size_stride(arg1437_1, (384, 384, 1, 3), (1152, 3, 3, 1))
    assert_size_stride(arg1438_1, (384, ), (1, ))
    assert_size_stride(arg1439_1, (384, ), (1, ))
    assert_size_stride(arg1440_1, (384, 384, 3, 1), (1152, 3, 1, 1))
    assert_size_stride(arg1441_1, (384, ), (1, ))
    assert_size_stride(arg1442_1, (384, ), (1, ))
    assert_size_stride(arg1443_1, (448, 2048, 1, 1), (2048, 1, 2048, 2048))
    assert_size_stride(arg1444_1, (448, ), (1, ))
    assert_size_stride(arg1445_1, (448, ), (1, ))
    assert_size_stride(arg1446_1, (384, 448, 3, 3), (4032, 9, 3, 1))
    assert_size_stride(arg1447_1, (384, ), (1, ))
    assert_size_stride(arg1448_1, (384, ), (1, ))
    assert_size_stride(arg1449_1, (384, 384, 1, 3), (1152, 3, 3, 1))
    assert_size_stride(arg1450_1, (384, ), (1, ))
    assert_size_stride(arg1451_1, (384, ), (1, ))
    assert_size_stride(arg1452_1, (384, 384, 3, 1), (1152, 3, 1, 1))
    assert_size_stride(arg1453_1, (384, ), (1, ))
    assert_size_stride(arg1454_1, (384, ), (1, ))
    assert_size_stride(arg1455_1, (192, 2048, 1, 1), (2048, 1, 2048, 2048))
    assert_size_stride(arg1456_1, (192, ), (1, ))
    assert_size_stride(arg1457_1, (192, ), (1, ))
    assert_size_stride(arg1458_1, (1000, 2048), (2048, 1))
    assert_size_stride(arg1459_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_for_fused_0.run(arg876_1, arg877_1, arg878_1, arg879_1, arg880_1, arg881_1, arg882_1, arg883_1, arg884_1, arg885_1, arg886_1, arg887_1, arg888_1, arg889_1, arg890_1, arg891_1, arg892_1, arg893_1, arg894_1, arg895_1, arg896_1, arg897_1, arg898_1, arg899_1, arg900_1, arg901_1, arg902_1, arg903_1, arg904_1, arg905_1, arg906_1, arg907_1, arg908_1, arg909_1, arg910_1, arg911_1, arg912_1, arg913_1, arg914_1, arg915_1, arg916_1, arg917_1, arg918_1, arg919_1, arg920_1, arg921_1, arg922_1, arg923_1, arg924_1, arg925_1, arg926_1, arg927_1, arg928_1, arg929_1, arg930_1, arg931_1, arg932_1, arg933_1, arg934_1, arg935_1, arg936_1, arg937_1, arg938_1, arg939_1, arg940_1, arg941_1, arg942_1, arg943_1, arg944_1, arg945_1, arg946_1, arg947_1, arg948_1, arg949_1, arg950_1, arg951_1, arg952_1, arg953_1, arg954_1, arg955_1, arg956_1, arg957_1, arg958_1, arg876_1, arg877_1, arg878_1, arg879_1, arg880_1, arg881_1, arg882_1, arg883_1, arg884_1, arg885_1, arg886_1, arg887_1, arg888_1, arg889_1, arg890_1, arg891_1, arg892_1, arg893_1, arg894_1, arg895_1, arg896_1, arg897_1, arg898_1, arg899_1, arg900_1, arg901_1, arg902_1, arg903_1, arg904_1, arg905_1, arg906_1, arg907_1, arg908_1, arg909_1, arg910_1, arg911_1, arg912_1, arg913_1, arg914_1, arg915_1, arg916_1, arg917_1, arg918_1, arg919_1, arg920_1, arg921_1, arg922_1, arg923_1, arg924_1, arg925_1, arg926_1, arg927_1, arg928_1, arg929_1, arg930_1, arg931_1, arg932_1, arg933_1, arg934_1, arg935_1, arg936_1, arg937_1, arg938_1, arg939_1, arg940_1, arg941_1, arg942_1, arg943_1, arg944_1, arg945_1, arg946_1, arg947_1, arg948_1, arg949_1, arg950_1, arg951_1, arg952_1, arg953_1, arg954_1, arg955_1, arg956_1, arg957_1, arg958_1, grid=((83, 1, 1)), stream=stream0)
        # Source Nodes: [], Original ATen: []
        triton_for_fused_0.run(arg959_1, arg960_1, arg961_1, arg962_1, arg963_1, arg964_1, arg965_1, arg966_1, arg967_1, arg968_1, arg969_1, arg970_1, arg971_1, arg972_1, arg973_1, arg974_1, arg975_1, arg976_1, arg977_1, arg978_1, arg979_1, arg980_1, arg981_1, arg982_1, arg983_1, arg984_1, arg985_1, arg986_1, arg987_1, arg988_1, arg989_1, arg990_1, arg991_1, arg992_1, arg993_1, arg994_1, arg995_1, arg996_1, arg997_1, arg998_1, arg999_1, arg1000_1, arg1001_1, arg1002_1, arg1003_1, arg1004_1, arg1005_1, arg1006_1, arg1007_1, arg1008_1, arg1009_1, arg1010_1, arg1011_1, arg1012_1, arg1013_1, arg1014_1, arg1015_1, arg1016_1, arg1017_1, arg1018_1, arg1019_1, arg1020_1, arg1021_1, arg1022_1, arg1023_1, arg1024_1, arg1025_1, arg1026_1, arg1027_1, arg1028_1, arg1029_1, arg1030_1, arg1031_1, arg1032_1, arg1033_1, arg1034_1, arg1035_1, arg1036_1, arg1037_1, arg1038_1, arg1039_1, arg1040_1, arg1041_1, arg959_1, arg960_1, arg961_1, arg962_1, arg963_1, arg964_1, arg965_1, arg966_1, arg967_1, arg968_1, arg969_1, arg970_1, arg971_1, arg972_1, arg973_1, arg974_1, arg975_1, arg976_1, arg977_1, arg978_1, arg979_1, arg980_1, arg981_1, arg982_1, arg983_1, arg984_1, arg985_1, arg986_1, arg987_1, arg988_1, arg989_1, arg990_1, arg991_1, arg992_1, arg993_1, arg994_1, arg995_1, arg996_1, arg997_1, arg998_1, arg999_1, arg1000_1, arg1001_1, arg1002_1, arg1003_1, arg1004_1, arg1005_1, arg1006_1, arg1007_1, arg1008_1, arg1009_1, arg1010_1, arg1011_1, arg1012_1, arg1013_1, arg1014_1, arg1015_1, arg1016_1, arg1017_1, arg1018_1, arg1019_1, arg1020_1, arg1021_1, arg1022_1, arg1023_1, arg1024_1, arg1025_1, arg1026_1, arg1027_1, arg1028_1, arg1029_1, arg1030_1, arg1031_1, arg1032_1, arg1033_1, arg1034_1, arg1035_1, arg1036_1, arg1037_1, arg1038_1, arg1039_1, arg1040_1, arg1041_1, grid=((83, 1, 1)), stream=stream0)
        # Source Nodes: [], Original ATen: []
        triton_for_fused_0.run(arg1042_1, arg1043_1, arg1044_1, arg1045_1, arg1046_1, arg1047_1, arg1048_1, arg1049_1, arg1050_1, arg1051_1, arg1052_1, arg1053_1, arg1054_1, arg1055_1, arg1056_1, arg1057_1, arg1058_1, arg1059_1, arg1060_1, arg1061_1, arg1062_1, arg1063_1, arg1064_1, arg1065_1, arg1066_1, arg1067_1, arg1068_1, arg1069_1, arg1070_1, arg1071_1, arg1072_1, arg1073_1, arg1074_1, arg1075_1, arg1076_1, arg1077_1, arg1078_1, arg1079_1, arg1080_1, arg1081_1, arg1082_1, arg1083_1, arg1084_1, arg1085_1, arg1086_1, arg1087_1, arg1088_1, arg1089_1, arg1090_1, arg1091_1, arg1092_1, arg1093_1, arg1094_1, arg1095_1, arg1096_1, arg1097_1, arg1098_1, arg1099_1, arg1100_1, arg1101_1, arg1102_1, arg1103_1, arg1104_1, arg1105_1, arg1106_1, arg1107_1, arg1108_1, arg1109_1, arg1110_1, arg1111_1, arg1112_1, arg1113_1, arg1114_1, arg1115_1, arg1116_1, arg1117_1, arg1118_1, arg1119_1, arg1120_1, arg1121_1, arg1122_1, arg1123_1, arg1124_1, arg1042_1, arg1043_1, arg1044_1, arg1045_1, arg1046_1, arg1047_1, arg1048_1, arg1049_1, arg1050_1, arg1051_1, arg1052_1, arg1053_1, arg1054_1, arg1055_1, arg1056_1, arg1057_1, arg1058_1, arg1059_1, arg1060_1, arg1061_1, arg1062_1, arg1063_1, arg1064_1, arg1065_1, arg1066_1, arg1067_1, arg1068_1, arg1069_1, arg1070_1, arg1071_1, arg1072_1, arg1073_1, arg1074_1, arg1075_1, arg1076_1, arg1077_1, arg1078_1, arg1079_1, arg1080_1, arg1081_1, arg1082_1, arg1083_1, arg1084_1, arg1085_1, arg1086_1, arg1087_1, arg1088_1, arg1089_1, arg1090_1, arg1091_1, arg1092_1, arg1093_1, arg1094_1, arg1095_1, arg1096_1, arg1097_1, arg1098_1, arg1099_1, arg1100_1, arg1101_1, arg1102_1, arg1103_1, arg1104_1, arg1105_1, arg1106_1, arg1107_1, arg1108_1, arg1109_1, arg1110_1, arg1111_1, arg1112_1, arg1113_1, arg1114_1, arg1115_1, arg1116_1, arg1117_1, arg1118_1, arg1119_1, arg1120_1, arg1121_1, arg1122_1, arg1123_1, arg1124_1, grid=((83, 1, 1)), stream=stream0)
        # Source Nodes: [], Original ATen: []
        triton_for_fused_1.run(arg1125_1, arg1126_1, arg1127_1, arg1128_1, arg1129_1, arg1130_1, arg1131_1, arg1132_1, arg1133_1, arg1134_1, arg1135_1, arg1136_1, arg1137_1, arg1138_1, arg1139_1, arg1140_1, arg1141_1, arg1142_1, arg1143_1, arg1144_1, arg1145_1, arg1146_1, arg1147_1, arg1148_1, arg1149_1, arg1150_1, arg1151_1, arg1152_1, arg1153_1, arg1154_1, arg1155_1, arg1156_1, arg1157_1, arg1158_1, arg1159_1, arg1160_1, arg1161_1, arg1162_1, arg1163_1, arg1164_1, arg1165_1, arg1166_1, arg1167_1, arg1125_1, arg1126_1, arg1127_1, arg1128_1, arg1129_1, arg1130_1, arg1131_1, arg1132_1, arg1133_1, arg1134_1, arg1135_1, arg1136_1, arg1137_1, arg1138_1, arg1139_1, arg1140_1, arg1141_1, arg1142_1, arg1143_1, arg1144_1, arg1145_1, arg1146_1, arg1147_1, arg1148_1, arg1149_1, arg1150_1, arg1151_1, arg1152_1, arg1153_1, arg1154_1, arg1155_1, arg1156_1, arg1157_1, arg1158_1, arg1159_1, arg1160_1, arg1161_1, arg1162_1, arg1163_1, arg1164_1, arg1165_1, arg1166_1, arg1167_1, grid=((43, 1, 1)), stream=stream0)
        # Source Nodes: [], Original ATen: []
        triton_for_fused_2.run(arg292_1, arg1168_1, arg584_1, arg0_1, arg876_1, arg293_1, arg1169_1, arg585_1, arg1_1, arg877_1, arg294_1, arg1170_1, arg586_1, arg2_1, arg878_1, arg295_1, arg1171_1, arg587_1, arg3_1, arg879_1, arg296_1, arg1172_1, arg588_1, arg4_1, arg880_1, arg297_1, arg1173_1, arg589_1, arg5_1, arg881_1, arg298_1, arg1174_1, arg590_1, arg6_1, arg882_1, arg299_1, arg1175_1, arg591_1, arg7_1, arg883_1, arg300_1, arg1176_1, arg592_1, arg8_1, arg884_1, arg301_1, arg1177_1, arg593_1, arg9_1, arg885_1, arg302_1, arg1178_1, arg594_1, arg10_1, arg886_1, arg303_1, arg1179_1, arg595_1, arg11_1, arg887_1, arg304_1, arg1180_1, arg596_1, arg12_1, arg888_1, arg305_1, arg1181_1, arg597_1, arg13_1, arg889_1, arg306_1, arg1182_1, arg598_1, arg14_1, arg890_1, arg307_1, arg1183_1, arg599_1, arg15_1, arg891_1, arg308_1, arg1184_1, arg600_1, arg16_1, arg892_1, arg309_1, arg1185_1, arg601_1, arg17_1, arg893_1, arg310_1, arg1186_1, arg602_1, arg18_1, arg894_1, arg311_1, arg1187_1, arg603_1, arg19_1, arg895_1, arg292_1, arg0_1, arg584_1, arg293_1, arg1_1, arg585_1, arg294_1, arg2_1, arg586_1, arg295_1, arg3_1, arg587_1, arg296_1, arg4_1, arg588_1, arg297_1, arg5_1, arg589_1, arg298_1, arg6_1, arg590_1, arg299_1, arg7_1, arg591_1, arg300_1, arg8_1, arg592_1, arg301_1, arg9_1, arg593_1, arg302_1, arg10_1, arg594_1, arg303_1, arg11_1, arg595_1, arg304_1, arg12_1, arg596_1, arg305_1, arg13_1, arg597_1, arg306_1, arg14_1, arg598_1, arg307_1, arg15_1, arg599_1, arg308_1, arg16_1, arg600_1, arg309_1, arg17_1, arg601_1, arg310_1, arg18_1, arg602_1, arg311_1, arg19_1, arg603_1, grid=((202, 1, 1)), stream=stream0)
        # Source Nodes: [], Original ATen: []
        triton_for_fused_3.run(arg312_1, arg1188_1, arg604_1, arg20_1, arg896_1, arg313_1, arg1189_1, arg605_1, arg21_1, arg897_1, arg314_1, arg1190_1, arg606_1, arg22_1, arg898_1, arg315_1, arg1191_1, arg607_1, arg23_1, arg899_1, arg316_1, arg1192_1, arg608_1, arg24_1, arg900_1, arg317_1, arg1193_1, arg609_1, arg25_1, arg901_1, arg318_1, arg1194_1, arg610_1, arg26_1, arg902_1, arg319_1, arg1195_1, arg611_1, arg27_1, arg903_1, arg320_1, arg1196_1, arg612_1, arg28_1, arg904_1, arg321_1, arg1197_1, arg613_1, arg29_1, arg905_1, arg322_1, arg1198_1, arg614_1, arg30_1, arg906_1, arg323_1, arg1199_1, arg615_1, arg31_1, arg907_1, arg324_1, arg1200_1, arg616_1, arg32_1, arg908_1, arg325_1, arg1201_1, arg617_1, arg33_1, arg909_1, arg326_1, arg1202_1, arg618_1, arg34_1, arg910_1, arg327_1, arg1203_1, arg619_1, arg35_1, arg911_1, arg328_1, arg1204_1, arg620_1, arg36_1, arg912_1, arg329_1, arg1205_1, arg621_1, arg37_1, arg913_1, arg330_1, arg1206_1, arg622_1, arg38_1, arg914_1, arg331_1, arg1207_1, arg623_1, arg39_1, arg915_1, arg312_1, arg20_1, arg604_1, arg313_1, arg21_1, arg605_1, arg314_1, arg22_1, arg606_1, arg315_1, arg23_1, arg607_1, arg316_1, arg24_1, arg608_1, arg317_1, arg25_1, arg609_1, arg318_1, arg26_1, arg610_1, arg319_1, arg27_1, arg611_1, arg320_1, arg28_1, arg612_1, arg321_1, arg29_1, arg613_1, arg322_1, arg30_1, arg614_1, arg323_1, arg31_1, arg615_1, arg324_1, arg32_1, arg616_1, arg325_1, arg33_1, arg617_1, arg326_1, arg34_1, arg618_1, arg327_1, arg35_1, arg619_1, arg328_1, arg36_1, arg620_1, arg329_1, arg37_1, arg621_1, arg330_1, arg38_1, arg622_1, arg331_1, arg39_1, arg623_1, grid=((269, 1, 1)), stream=stream0)
        # Source Nodes: [], Original ATen: []
        triton_for_fused_4.run(arg332_1, arg1208_1, arg624_1, arg40_1, arg916_1, arg333_1, arg1209_1, arg625_1, arg41_1, arg917_1, arg334_1, arg1210_1, arg626_1, arg42_1, arg918_1, arg335_1, arg1211_1, arg627_1, arg43_1, arg919_1, arg336_1, arg1212_1, arg628_1, arg44_1, arg920_1, arg337_1, arg1213_1, arg629_1, arg45_1, arg921_1, arg338_1, arg1214_1, arg630_1, arg46_1, arg922_1, arg339_1, arg1215_1, arg631_1, arg47_1, arg923_1, arg340_1, arg1216_1, arg632_1, arg48_1, arg924_1, arg341_1, arg1217_1, arg633_1, arg49_1, arg925_1, arg342_1, arg1218_1, arg634_1, arg50_1, arg926_1, arg343_1, arg1219_1, arg635_1, arg51_1, arg927_1, arg344_1, arg1220_1, arg636_1, arg52_1, arg928_1, arg345_1, arg1221_1, arg637_1, arg53_1, arg929_1, arg346_1, arg1222_1, arg638_1, arg54_1, arg930_1, arg347_1, arg1223_1, arg639_1, arg55_1, arg931_1, arg348_1, arg1224_1, arg640_1, arg56_1, arg932_1, arg349_1, arg1225_1, arg641_1, arg57_1, arg933_1, arg350_1, arg1226_1, arg642_1, arg58_1, arg934_1, arg351_1, arg1227_1, arg643_1, arg59_1, arg935_1, arg332_1, arg40_1, arg624_1, arg333_1, arg41_1, arg625_1, arg334_1, arg42_1, arg626_1, arg335_1, arg43_1, arg627_1, arg336_1, arg44_1, arg628_1, arg337_1, arg45_1, arg629_1, arg338_1, arg46_1, arg630_1, arg339_1, arg47_1, arg631_1, arg340_1, arg48_1, arg632_1, arg341_1, arg49_1, arg633_1, arg342_1, arg50_1, arg634_1, arg343_1, arg51_1, arg635_1, arg344_1, arg52_1, arg636_1, arg345_1, arg53_1, arg637_1, arg346_1, arg54_1, arg638_1, arg347_1, arg55_1, arg639_1, arg348_1, arg56_1, arg640_1, arg349_1, arg57_1, arg641_1, arg350_1, arg58_1, arg642_1, arg351_1, arg59_1, arg643_1, grid=((274, 1, 1)), stream=stream0)
        # Source Nodes: [], Original ATen: []
        triton_for_fused_5.run(arg352_1, arg1228_1, arg644_1, arg60_1, arg936_1, arg353_1, arg1229_1, arg645_1, arg61_1, arg937_1, arg354_1, arg1230_1, arg646_1, arg62_1, arg938_1, arg355_1, arg1231_1, arg647_1, arg63_1, arg939_1, arg356_1, arg1232_1, arg648_1, arg64_1, arg940_1, arg357_1, arg1233_1, arg649_1, arg65_1, arg941_1, arg358_1, arg1234_1, arg650_1, arg66_1, arg942_1, arg359_1, arg1235_1, arg651_1, arg67_1, arg943_1, arg360_1, arg1236_1, arg652_1, arg68_1, arg944_1, arg361_1, arg1237_1, arg653_1, arg69_1, arg945_1, arg362_1, arg1238_1, arg654_1, arg70_1, arg946_1, arg363_1, arg1239_1, arg655_1, arg71_1, arg947_1, arg364_1, arg1240_1, arg656_1, arg72_1, arg948_1, arg365_1, arg1241_1, arg657_1, arg73_1, arg949_1, arg366_1, arg1242_1, arg658_1, arg74_1, arg950_1, arg367_1, arg1243_1, arg659_1, arg75_1, arg951_1, arg368_1, arg1244_1, arg660_1, arg76_1, arg952_1, arg369_1, arg1245_1, arg661_1, arg77_1, arg953_1, arg370_1, arg1246_1, arg662_1, arg78_1, arg954_1, arg371_1, arg1247_1, arg663_1, arg79_1, arg955_1, arg352_1, arg60_1, arg644_1, arg353_1, arg61_1, arg645_1, arg354_1, arg62_1, arg646_1, arg355_1, arg63_1, arg647_1, arg356_1, arg64_1, arg648_1, arg357_1, arg65_1, arg649_1, arg358_1, arg66_1, arg650_1, arg359_1, arg67_1, arg651_1, arg360_1, arg68_1, arg652_1, arg361_1, arg69_1, arg653_1, arg362_1, arg70_1, arg654_1, arg363_1, arg71_1, arg655_1, arg364_1, arg72_1, arg656_1, arg365_1, arg73_1, arg657_1, arg366_1, arg74_1, arg658_1, arg367_1, arg75_1, arg659_1, arg368_1, arg76_1, arg660_1, arg369_1, arg77_1, arg661_1, arg370_1, arg78_1, arg662_1, arg371_1, arg79_1, arg663_1, grid=((1245, 1, 1)), stream=stream0)
        # Source Nodes: [], Original ATen: []
        triton_for_fused_6.run(arg372_1, arg1248_1, arg664_1, arg80_1, arg956_1, arg373_1, arg1249_1, arg665_1, arg81_1, arg957_1, arg374_1, arg1250_1, arg666_1, arg82_1, arg958_1, arg375_1, arg1251_1, arg667_1, arg83_1, arg959_1, arg376_1, arg1252_1, arg668_1, arg84_1, arg960_1, arg377_1, arg1253_1, arg669_1, arg85_1, arg961_1, arg378_1, arg1254_1, arg670_1, arg86_1, arg962_1, arg379_1, arg1255_1, arg671_1, arg87_1, arg963_1, arg380_1, arg1256_1, arg672_1, arg88_1, arg964_1, arg381_1, arg1257_1, arg673_1, arg89_1, arg965_1, arg382_1, arg1258_1, arg674_1, arg90_1, arg966_1, arg383_1, arg1259_1, arg675_1, arg91_1, arg967_1, arg384_1, arg1260_1, arg676_1, arg92_1, arg968_1, arg385_1, arg1261_1, arg677_1, arg93_1, arg969_1, arg386_1, arg1262_1, arg678_1, arg94_1, arg970_1, arg387_1, arg1263_1, arg679_1, arg95_1, arg971_1, arg388_1, arg1264_1, arg680_1, arg96_1, arg972_1, arg389_1, arg1265_1, arg681_1, arg97_1, arg973_1, arg390_1, arg1266_1, arg682_1, arg98_1, arg974_1, arg391_1, arg1267_1, arg683_1, arg99_1, arg975_1, arg372_1, arg80_1, arg664_1, arg373_1, arg81_1, arg665_1, arg374_1, arg82_1, arg666_1, arg375_1, arg83_1, arg667_1, arg376_1, arg84_1, arg668_1, arg377_1, arg85_1, arg669_1, arg378_1, arg86_1, arg670_1, arg379_1, arg87_1, arg671_1, arg380_1, arg88_1, arg672_1, arg381_1, arg89_1, arg673_1, arg382_1, arg90_1, arg674_1, arg383_1, arg91_1, arg675_1, arg384_1, arg92_1, arg676_1, arg385_1, arg93_1, arg677_1, arg386_1, arg94_1, arg678_1, arg387_1, arg95_1, arg679_1, arg388_1, arg96_1, arg680_1, arg389_1, arg97_1, arg681_1, arg390_1, arg98_1, arg682_1, arg391_1, arg99_1, arg683_1, grid=((686, 1, 1)), stream=stream0)
        # Source Nodes: [], Original ATen: []
        triton_for_fused_7.run(arg392_1, arg1268_1, arg684_1, arg100_1, arg976_1, arg393_1, arg1269_1, arg685_1, arg101_1, arg977_1, arg394_1, arg1270_1, arg686_1, arg102_1, arg978_1, arg395_1, arg1271_1, arg687_1, arg103_1, arg979_1, arg396_1, arg1272_1, arg688_1, arg104_1, arg980_1, arg397_1, arg1273_1, arg689_1, arg105_1, arg981_1, arg398_1, arg1274_1, arg690_1, arg106_1, arg982_1, arg399_1, arg1275_1, arg691_1, arg107_1, arg983_1, arg400_1, arg1276_1, arg692_1, arg108_1, arg984_1, arg401_1, arg1277_1, arg693_1, arg109_1, arg985_1, arg402_1, arg1278_1, arg694_1, arg110_1, arg986_1, arg403_1, arg1279_1, arg695_1, arg111_1, arg987_1, arg404_1, arg1280_1, arg696_1, arg112_1, arg988_1, arg405_1, arg1281_1, arg697_1, arg113_1, arg989_1, arg406_1, arg1282_1, arg698_1, arg114_1, arg990_1, arg407_1, arg1283_1, arg699_1, arg115_1, arg991_1, arg408_1, arg1284_1, arg700_1, arg116_1, arg992_1, arg409_1, arg1285_1, arg701_1, arg117_1, arg993_1, arg410_1, arg1286_1, arg702_1, arg118_1, arg994_1, arg411_1, arg1287_1, arg703_1, arg119_1, arg995_1, arg392_1, arg100_1, arg684_1, arg393_1, arg101_1, arg685_1, arg394_1, arg102_1, arg686_1, arg395_1, arg103_1, arg687_1, arg396_1, arg104_1, arg688_1, arg397_1, arg105_1, arg689_1, arg398_1, arg106_1, arg690_1, arg399_1, arg107_1, arg691_1, arg400_1, arg108_1, arg692_1, arg401_1, arg109_1, arg693_1, arg402_1, arg110_1, arg694_1, arg403_1, arg111_1, arg695_1, arg404_1, arg112_1, arg696_1, arg405_1, arg113_1, arg697_1, arg406_1, arg114_1, arg698_1, arg407_1, arg115_1, arg699_1, arg408_1, arg116_1, arg700_1, arg409_1, arg117_1, arg701_1, arg410_1, arg118_1, arg702_1, arg411_1, arg119_1, arg703_1, grid=((758, 1, 1)), stream=stream0)
        # Source Nodes: [], Original ATen: []
        triton_for_fused_8.run(arg412_1, arg1288_1, arg704_1, arg120_1, arg996_1, arg413_1, arg1289_1, arg705_1, arg121_1, arg997_1, arg414_1, arg1290_1, arg706_1, arg122_1, arg998_1, arg415_1, arg1291_1, arg707_1, arg123_1, arg999_1, arg416_1, arg1292_1, arg708_1, arg124_1, arg1000_1, arg417_1, arg1293_1, arg709_1, arg125_1, arg1001_1, arg418_1, arg1294_1, arg710_1, arg126_1, arg1002_1, arg419_1, arg1295_1, arg711_1, arg127_1, arg1003_1, arg420_1, arg1296_1, arg712_1, arg128_1, arg1004_1, arg421_1, arg1297_1, arg713_1, arg129_1, arg1005_1, arg422_1, arg1298_1, arg714_1, arg130_1, arg1006_1, arg423_1, arg1299_1, arg715_1, arg131_1, arg1007_1, arg424_1, arg1300_1, arg716_1, arg132_1, arg1008_1, arg425_1, arg1301_1, arg717_1, arg133_1, arg1009_1, arg426_1, arg1302_1, arg718_1, arg134_1, arg1010_1, arg427_1, arg1303_1, arg719_1, arg135_1, arg1011_1, arg428_1, arg1304_1, arg720_1, arg136_1, arg1012_1, arg429_1, arg1305_1, arg721_1, arg137_1, arg1013_1, arg430_1, arg1306_1, arg722_1, arg138_1, arg1014_1, arg431_1, arg1307_1, arg723_1, arg139_1, arg1015_1, arg412_1, arg120_1, arg704_1, arg413_1, arg121_1, arg705_1, arg414_1, arg122_1, arg706_1, arg415_1, arg123_1, arg707_1, arg416_1, arg124_1, arg708_1, arg417_1, arg125_1, arg709_1, arg418_1, arg126_1, arg710_1, arg419_1, arg127_1, arg711_1, arg420_1, arg128_1, arg712_1, arg421_1, arg129_1, arg713_1, arg422_1, arg130_1, arg714_1, arg423_1, arg131_1, arg715_1, arg424_1, arg132_1, arg716_1, arg425_1, arg133_1, arg717_1, arg426_1, arg134_1, arg718_1, arg427_1, arg135_1, arg719_1, arg428_1, arg136_1, arg720_1, arg429_1, arg137_1, arg721_1, arg430_1, arg138_1, arg722_1, arg431_1, arg139_1, arg723_1, grid=((1132, 1, 1)), stream=stream0)
        # Source Nodes: [], Original ATen: []
        triton_for_fused_9.run(arg432_1, arg1308_1, arg724_1, arg140_1, arg1016_1, arg433_1, arg1309_1, arg725_1, arg141_1, arg1017_1, arg434_1, arg1310_1, arg726_1, arg142_1, arg1018_1, arg435_1, arg1311_1, arg727_1, arg143_1, arg1019_1, arg436_1, arg1312_1, arg728_1, arg144_1, arg1020_1, arg437_1, arg1313_1, arg729_1, arg145_1, arg1021_1, arg438_1, arg1314_1, arg730_1, arg146_1, arg1022_1, arg439_1, arg1315_1, arg731_1, arg147_1, arg1023_1, arg440_1, arg1316_1, arg732_1, arg148_1, arg1024_1, arg441_1, arg1317_1, arg733_1, arg149_1, arg1025_1, arg442_1, arg1318_1, arg734_1, arg150_1, arg1026_1, arg443_1, arg1319_1, arg735_1, arg151_1, arg1027_1, arg444_1, arg1320_1, arg736_1, arg152_1, arg1028_1, arg445_1, arg1321_1, arg737_1, arg153_1, arg1029_1, arg446_1, arg1322_1, arg738_1, arg154_1, arg1030_1, arg447_1, arg1323_1, arg739_1, arg155_1, arg1031_1, arg448_1, arg1324_1, arg740_1, arg156_1, arg1032_1, arg449_1, arg1325_1, arg741_1, arg157_1, arg1033_1, arg450_1, arg1326_1, arg742_1, arg158_1, arg1034_1, arg451_1, arg1327_1, arg743_1, arg159_1, arg1035_1, arg432_1, arg140_1, arg724_1, arg433_1, arg141_1, arg725_1, arg434_1, arg142_1, arg726_1, arg435_1, arg143_1, arg727_1, arg436_1, arg144_1, arg728_1, arg437_1, arg145_1, arg729_1, arg438_1, arg146_1, arg730_1, arg439_1, arg147_1, arg731_1, arg440_1, arg148_1, arg732_1, arg441_1, arg149_1, arg733_1, arg442_1, arg150_1, arg734_1, arg443_1, arg151_1, arg735_1, arg444_1, arg152_1, arg736_1, arg445_1, arg153_1, arg737_1, arg446_1, arg154_1, arg738_1, arg447_1, arg155_1, arg739_1, arg448_1, arg156_1, arg740_1, arg449_1, arg157_1, arg741_1, arg450_1, arg158_1, arg742_1, arg451_1, arg159_1, arg743_1, grid=((1191, 1, 1)), stream=stream0)
        # Source Nodes: [], Original ATen: []
        triton_for_fused_10.run(arg452_1, arg1328_1, arg744_1, arg160_1, arg1036_1, arg453_1, arg1329_1, arg745_1, arg161_1, arg1037_1, arg454_1, arg1330_1, arg746_1, arg162_1, arg1038_1, arg455_1, arg1331_1, arg747_1, arg163_1, arg1039_1, arg456_1, arg1332_1, arg748_1, arg164_1, arg1040_1, arg457_1, arg1333_1, arg749_1, arg165_1, arg1041_1, arg458_1, arg1334_1, arg750_1, arg166_1, arg1042_1, arg459_1, arg1335_1, arg751_1, arg167_1, arg1043_1, arg460_1, arg1336_1, arg752_1, arg168_1, arg1044_1, arg461_1, arg1337_1, arg753_1, arg169_1, arg1045_1, arg462_1, arg1338_1, arg754_1, arg170_1, arg1046_1, arg463_1, arg1339_1, arg755_1, arg171_1, arg1047_1, arg464_1, arg1340_1, arg756_1, arg172_1, arg1048_1, arg465_1, arg1341_1, arg757_1, arg173_1, arg1049_1, arg466_1, arg1342_1, arg758_1, arg174_1, arg1050_1, arg467_1, arg1343_1, arg759_1, arg175_1, arg1051_1, arg468_1, arg1344_1, arg760_1, arg176_1, arg1052_1, arg469_1, arg1345_1, arg761_1, arg177_1, arg1053_1, arg470_1, arg1346_1, arg762_1, arg178_1, arg1054_1, arg471_1, arg1347_1, arg763_1, arg179_1, arg1055_1, arg452_1, arg160_1, arg744_1, arg453_1, arg161_1, arg745_1, arg454_1, arg162_1, arg746_1, arg455_1, arg163_1, arg747_1, arg456_1, arg164_1, arg748_1, arg457_1, arg165_1, arg749_1, arg458_1, arg166_1, arg750_1, arg459_1, arg167_1, arg751_1, arg460_1, arg168_1, arg752_1, arg461_1, arg169_1, arg753_1, arg462_1, arg170_1, arg754_1, arg463_1, arg171_1, arg755_1, arg464_1, arg172_1, arg756_1, arg465_1, arg173_1, arg757_1, arg466_1, arg174_1, arg758_1, arg467_1, arg175_1, arg759_1, arg468_1, arg176_1, arg760_1, arg469_1, arg177_1, arg761_1, arg470_1, arg178_1, arg762_1, arg471_1, arg179_1, arg763_1, grid=((1013, 1, 1)), stream=stream0)
        # Source Nodes: [], Original ATen: []
        triton_for_fused_11.run(arg472_1, arg1348_1, arg764_1, arg180_1, arg1056_1, arg473_1, arg1349_1, arg765_1, arg181_1, arg1057_1, arg474_1, arg1350_1, arg766_1, arg182_1, arg1058_1, arg475_1, arg1351_1, arg767_1, arg183_1, arg1059_1, arg476_1, arg1352_1, arg768_1, arg184_1, arg1060_1, arg477_1, arg1353_1, arg769_1, arg185_1, arg1061_1, arg478_1, arg1354_1, arg770_1, arg186_1, arg1062_1, arg479_1, arg1355_1, arg771_1, arg187_1, arg1063_1, arg480_1, arg1356_1, arg772_1, arg188_1, arg1064_1, arg481_1, arg1357_1, arg773_1, arg189_1, arg1065_1, arg482_1, arg1358_1, arg774_1, arg190_1, arg1066_1, arg483_1, arg1359_1, arg775_1, arg191_1, arg1067_1, arg484_1, arg1360_1, arg776_1, arg192_1, arg1068_1, arg485_1, arg1361_1, arg777_1, arg193_1, arg1069_1, arg486_1, arg1362_1, arg778_1, arg194_1, arg1070_1, arg487_1, arg1363_1, arg779_1, arg195_1, arg1071_1, arg488_1, arg1364_1, arg780_1, arg196_1, arg1072_1, arg489_1, arg1365_1, arg781_1, arg197_1, arg1073_1, arg490_1, arg1366_1, arg782_1, arg198_1, arg1074_1, arg491_1, arg1367_1, arg783_1, arg199_1, arg1075_1, arg472_1, arg180_1, arg764_1, arg473_1, arg181_1, arg765_1, arg474_1, arg182_1, arg766_1, arg475_1, arg183_1, arg767_1, arg476_1, arg184_1, arg768_1, arg477_1, arg185_1, arg769_1, arg478_1, arg186_1, arg770_1, arg479_1, arg187_1, arg771_1, arg480_1, arg188_1, arg772_1, arg481_1, arg189_1, arg773_1, arg482_1, arg190_1, arg774_1, arg483_1, arg191_1, arg775_1, arg484_1, arg192_1, arg776_1, arg485_1, arg193_1, arg777_1, arg486_1, arg194_1, arg778_1, arg487_1, arg195_1, arg779_1, arg488_1, arg196_1, arg780_1, arg489_1, arg197_1, arg781_1, arg490_1, arg198_1, arg782_1, arg491_1, arg199_1, arg783_1, grid=((1453, 1, 1)), stream=stream0)
        # Source Nodes: [], Original ATen: []
        triton_for_fused_12.run(arg492_1, arg1368_1, arg784_1, arg200_1, arg1076_1, arg493_1, arg1369_1, arg785_1, arg201_1, arg1077_1, arg494_1, arg1370_1, arg786_1, arg202_1, arg1078_1, arg495_1, arg1371_1, arg787_1, arg203_1, arg1079_1, arg496_1, arg1372_1, arg788_1, arg204_1, arg1080_1, arg497_1, arg1373_1, arg789_1, arg205_1, arg1081_1, arg498_1, arg1374_1, arg790_1, arg206_1, arg1082_1, arg499_1, arg1375_1, arg791_1, arg207_1, arg1083_1, arg500_1, arg1376_1, arg792_1, arg208_1, arg1084_1, arg501_1, arg1377_1, arg793_1, arg209_1, arg1085_1, arg502_1, arg1378_1, arg794_1, arg210_1, arg1086_1, arg503_1, arg1379_1, arg795_1, arg211_1, arg1087_1, arg504_1, arg1380_1, arg796_1, arg212_1, arg1088_1, arg505_1, arg1381_1, arg797_1, arg213_1, arg1089_1, arg506_1, arg1382_1, arg798_1, arg214_1, arg1090_1, arg507_1, arg1383_1, arg799_1, arg215_1, arg1091_1, arg508_1, arg1384_1, arg800_1, arg216_1, arg1092_1, arg509_1, arg1385_1, arg801_1, arg217_1, arg1093_1, arg510_1, arg1386_1, arg802_1, arg218_1, arg1094_1, arg511_1, arg1387_1, arg803_1, arg219_1, arg1095_1, arg492_1, arg200_1, arg784_1, arg493_1, arg201_1, arg785_1, arg494_1, arg202_1, arg786_1, arg495_1, arg203_1, arg787_1, arg496_1, arg204_1, arg788_1, arg497_1, arg205_1, arg789_1, arg498_1, arg206_1, arg790_1, arg499_1, arg207_1, arg791_1, arg500_1, arg208_1, arg792_1, arg501_1, arg209_1, arg793_1, arg502_1, arg210_1, arg794_1, arg503_1, arg211_1, arg795_1, arg504_1, arg212_1, arg796_1, arg505_1, arg213_1, arg797_1, arg506_1, arg214_1, arg798_1, arg507_1, arg215_1, arg799_1, arg508_1, arg216_1, arg800_1, arg509_1, arg217_1, arg801_1, arg510_1, arg218_1, arg802_1, arg511_1, arg219_1, arg803_1, grid=((4051, 1, 1)), stream=stream0)
        # Source Nodes: [], Original ATen: []
        triton_for_fused_13.run(arg512_1, arg1388_1, arg804_1, arg220_1, arg1096_1, arg513_1, arg1389_1, arg805_1, arg221_1, arg1097_1, arg514_1, arg1390_1, arg806_1, arg222_1, arg1098_1, arg515_1, arg1391_1, arg807_1, arg223_1, arg1099_1, arg516_1, arg1392_1, arg808_1, arg224_1, arg1100_1, arg517_1, arg1393_1, arg809_1, arg225_1, arg1101_1, arg518_1, arg1394_1, arg810_1, arg226_1, arg1102_1, arg519_1, arg1395_1, arg811_1, arg227_1, arg1103_1, arg520_1, arg1396_1, arg812_1, arg228_1, arg1104_1, arg521_1, arg1397_1, arg813_1, arg229_1, arg1105_1, arg522_1, arg1398_1, arg814_1, arg230_1, arg1106_1, arg523_1, arg1399_1, arg815_1, arg231_1, arg1107_1, arg524_1, arg1400_1, arg816_1, arg232_1, arg1108_1, arg525_1, arg1401_1, arg817_1, arg233_1, arg1109_1, arg526_1, arg1402_1, arg818_1, arg234_1, arg1110_1, arg527_1, arg1403_1, arg819_1, arg235_1, arg1111_1, arg528_1, arg1404_1, arg820_1, arg236_1, arg1112_1, arg529_1, arg1405_1, arg821_1, arg237_1, arg1113_1, arg530_1, arg1406_1, arg822_1, arg238_1, arg1114_1, arg531_1, arg1407_1, arg823_1, arg239_1, arg1115_1, arg512_1, arg220_1, arg804_1, arg513_1, arg221_1, arg805_1, arg514_1, arg222_1, arg806_1, arg515_1, arg223_1, arg807_1, arg516_1, arg224_1, arg808_1, arg517_1, arg225_1, arg809_1, arg518_1, arg226_1, arg810_1, arg519_1, arg227_1, arg811_1, arg520_1, arg228_1, arg812_1, arg521_1, arg229_1, arg813_1, arg522_1, arg230_1, arg814_1, arg523_1, arg231_1, arg815_1, arg524_1, arg232_1, arg816_1, arg525_1, arg233_1, arg817_1, arg526_1, arg234_1, arg818_1, arg527_1, arg235_1, arg819_1, arg528_1, arg236_1, arg820_1, arg529_1, arg237_1, arg821_1, arg530_1, arg238_1, arg822_1, arg531_1, arg239_1, arg823_1, grid=((2405, 1, 1)), stream=stream0)
        # Source Nodes: [], Original ATen: []
        triton_for_fused_14.run(arg532_1, arg1408_1, arg824_1, arg240_1, arg1116_1, arg533_1, arg1409_1, arg825_1, arg241_1, arg1117_1, arg534_1, arg1410_1, arg826_1, arg242_1, arg1118_1, arg535_1, arg1411_1, arg827_1, arg243_1, arg1119_1, arg536_1, arg1412_1, arg828_1, arg244_1, arg1120_1, arg537_1, arg1413_1, arg829_1, arg245_1, arg1121_1, arg538_1, arg1414_1, arg830_1, arg246_1, arg1122_1, arg539_1, arg1415_1, arg831_1, arg247_1, arg1123_1, arg540_1, arg1416_1, arg832_1, arg248_1, arg1124_1, arg541_1, arg1417_1, arg833_1, arg249_1, arg1125_1, arg542_1, arg1418_1, arg834_1, arg250_1, arg1126_1, arg543_1, arg1419_1, arg835_1, arg251_1, arg1127_1, arg544_1, arg1420_1, arg836_1, arg252_1, arg1128_1, arg545_1, arg1421_1, arg837_1, arg253_1, arg1129_1, arg546_1, arg1422_1, arg838_1, arg254_1, arg1130_1, arg547_1, arg1423_1, arg839_1, arg255_1, arg1131_1, arg548_1, arg1424_1, arg840_1, arg256_1, arg1132_1, arg549_1, arg1425_1, arg841_1, arg257_1, arg1133_1, arg550_1, arg1426_1, arg842_1, arg258_1, arg1134_1, arg551_1, arg1427_1, arg843_1, arg259_1, arg1135_1, arg532_1, arg240_1, arg824_1, arg533_1, arg241_1, arg825_1, arg534_1, arg242_1, arg826_1, arg535_1, arg243_1, arg827_1, arg536_1, arg244_1, arg828_1, arg537_1, arg245_1, arg829_1, arg538_1, arg246_1, arg830_1, arg539_1, arg247_1, arg831_1, arg540_1, arg248_1, arg832_1, arg541_1, arg249_1, arg833_1, arg542_1, arg250_1, arg834_1, arg543_1, arg251_1, arg835_1, arg544_1, arg252_1, arg836_1, arg545_1, arg253_1, arg837_1, arg546_1, arg254_1, arg838_1, arg547_1, arg255_1, arg839_1, arg548_1, arg256_1, arg840_1, arg549_1, arg257_1, arg841_1, arg550_1, arg258_1, arg842_1, arg551_1, arg259_1, arg843_1, grid=((3814, 1, 1)), stream=stream0)
        # Source Nodes: [], Original ATen: []
        triton_for_fused_15.run(arg552_1, arg1428_1, arg844_1, arg260_1, arg1136_1, arg553_1, arg1429_1, arg845_1, arg261_1, arg1137_1, arg554_1, arg1430_1, arg846_1, arg262_1, arg1138_1, arg555_1, arg1431_1, arg847_1, arg263_1, arg1139_1, arg556_1, arg1432_1, arg848_1, arg264_1, arg1140_1, arg557_1, arg1433_1, arg849_1, arg265_1, arg1141_1, arg558_1, arg1434_1, arg850_1, arg266_1, arg1142_1, arg559_1, arg1435_1, arg851_1, arg267_1, arg1143_1, arg560_1, arg1436_1, arg852_1, arg268_1, arg1144_1, arg561_1, arg1437_1, arg853_1, arg269_1, arg1145_1, arg562_1, arg1438_1, arg854_1, arg270_1, arg1146_1, arg563_1, arg1439_1, arg855_1, arg271_1, arg1147_1, arg564_1, arg1440_1, arg856_1, arg272_1, arg1148_1, arg565_1, arg1441_1, arg857_1, arg273_1, arg1149_1, arg566_1, arg1442_1, arg858_1, arg274_1, arg1150_1, arg567_1, arg1443_1, arg859_1, arg275_1, arg1151_1, arg568_1, arg1444_1, arg860_1, arg276_1, arg1152_1, arg569_1, arg1445_1, arg861_1, arg277_1, arg1153_1, arg570_1, arg1446_1, arg862_1, arg278_1, arg1154_1, arg571_1, arg1447_1, arg863_1, arg279_1, arg1155_1, arg552_1, arg260_1, arg844_1, arg553_1, arg261_1, arg845_1, arg554_1, arg262_1, arg846_1, arg555_1, arg263_1, arg847_1, arg556_1, arg264_1, arg848_1, arg557_1, arg265_1, arg849_1, arg558_1, arg266_1, arg850_1, arg559_1, arg267_1, arg851_1, arg560_1, arg268_1, arg852_1, arg561_1, arg269_1, arg853_1, arg562_1, arg270_1, arg854_1, arg563_1, arg271_1, arg855_1, arg564_1, arg272_1, arg856_1, arg565_1, arg273_1, arg857_1, arg566_1, arg274_1, arg858_1, arg567_1, arg275_1, arg859_1, arg568_1, arg276_1, arg860_1, arg569_1, arg277_1, arg861_1, arg570_1, arg278_1, arg862_1, arg571_1, arg279_1, arg863_1, grid=((4933, 1, 1)), stream=stream0)
        # Source Nodes: [], Original ATen: []
        triton_for_fused_16.run(arg572_1, arg1448_1, arg864_1, arg280_1, arg1156_1, arg573_1, arg1449_1, arg865_1, arg281_1, arg1157_1, arg574_1, arg1450_1, arg866_1, arg282_1, arg1158_1, arg575_1, arg1451_1, arg867_1, arg283_1, arg1159_1, arg576_1, arg1452_1, arg868_1, arg284_1, arg1160_1, arg577_1, arg1453_1, arg869_1, arg285_1, arg1161_1, arg578_1, arg1454_1, arg870_1, arg286_1, arg1162_1, arg579_1, arg1455_1, arg871_1, arg287_1, arg1163_1, arg580_1, arg1456_1, arg872_1, arg288_1, arg1164_1, arg581_1, arg1457_1, arg873_1, arg289_1, arg1165_1, arg582_1, arg1458_1, arg874_1, arg290_1, arg1166_1, arg583_1, arg1459_1, arg875_1, arg291_1, arg1167_1, arg572_1, arg280_1, arg864_1, arg573_1, arg281_1, arg865_1, arg574_1, arg282_1, arg866_1, arg575_1, arg283_1, arg867_1, arg576_1, arg284_1, arg868_1, arg577_1, arg285_1, arg869_1, arg578_1, arg286_1, arg870_1, arg579_1, arg287_1, arg871_1, arg580_1, arg288_1, arg872_1, arg581_1, arg289_1, arg873_1, arg582_1, arg290_1, arg874_1, arg583_1, arg291_1, arg875_1, grid=((3256, 1, 1)), stream=stream0)
        del arg0_1
        del arg1000_1
        del arg1001_1
        del arg1002_1
        del arg1003_1
        del arg1004_1
        del arg1005_1
        del arg1006_1
        del arg1007_1
        del arg1008_1
        del arg1009_1
        del arg100_1
        del arg1010_1
        del arg1011_1
        del arg1012_1
        del arg1013_1
        del arg1014_1
        del arg1015_1
        del arg1016_1
        del arg1017_1
        del arg1018_1
        del arg1019_1
        del arg101_1
        del arg1020_1
        del arg1021_1
        del arg1022_1
        del arg1023_1
        del arg1024_1
        del arg1025_1
        del arg1026_1
        del arg1027_1
        del arg1028_1
        del arg1029_1
        del arg102_1
        del arg1030_1
        del arg1031_1
        del arg1032_1
        del arg1033_1
        del arg1034_1
        del arg1035_1
        del arg1036_1
        del arg1037_1
        del arg1038_1
        del arg1039_1
        del arg103_1
        del arg1040_1
        del arg1041_1
        del arg1042_1
        del arg1043_1
        del arg1044_1
        del arg1045_1
        del arg1046_1
        del arg1047_1
        del arg1048_1
        del arg1049_1
        del arg104_1
        del arg1050_1
        del arg1051_1
        del arg1052_1
        del arg1053_1
        del arg1054_1
        del arg1055_1
        del arg1056_1
        del arg1057_1
        del arg1058_1
        del arg1059_1
        del arg105_1
        del arg1060_1
        del arg1061_1
        del arg1062_1
        del arg1063_1
        del arg1064_1
        del arg1065_1
        del arg1066_1
        del arg1067_1
        del arg1068_1
        del arg1069_1
        del arg106_1
        del arg1070_1
        del arg1071_1
        del arg1072_1
        del arg1073_1
        del arg1074_1
        del arg1075_1
        del arg1076_1
        del arg1077_1
        del arg1078_1
        del arg1079_1
        del arg107_1
        del arg1080_1
        del arg1081_1
        del arg1082_1
        del arg1083_1
        del arg1084_1
        del arg1085_1
        del arg1086_1
        del arg1087_1
        del arg1088_1
        del arg1089_1
        del arg108_1
        del arg1090_1
        del arg1091_1
        del arg1092_1
        del arg1093_1
        del arg1094_1
        del arg1095_1
        del arg1096_1
        del arg1097_1
        del arg1098_1
        del arg1099_1
        del arg109_1
        del arg10_1
        del arg1100_1
        del arg1101_1
        del arg1102_1
        del arg1103_1
        del arg1104_1
        del arg1105_1
        del arg1106_1
        del arg1107_1
        del arg1108_1
        del arg1109_1
        del arg110_1
        del arg1110_1
        del arg1111_1
        del arg1112_1
        del arg1113_1
        del arg1114_1
        del arg1115_1
        del arg1116_1
        del arg1117_1
        del arg1118_1
        del arg1119_1
        del arg111_1
        del arg1120_1
        del arg1121_1
        del arg1122_1
        del arg1123_1
        del arg1124_1
        del arg1125_1
        del arg1126_1
        del arg1127_1
        del arg1128_1
        del arg1129_1
        del arg112_1
        del arg1130_1
        del arg1131_1
        del arg1132_1
        del arg1133_1
        del arg1134_1
        del arg1135_1
        del arg1136_1
        del arg1137_1
        del arg1138_1
        del arg1139_1
        del arg113_1
        del arg1140_1
        del arg1141_1
        del arg1142_1
        del arg1143_1
        del arg1144_1
        del arg1145_1
        del arg1146_1
        del arg1147_1
        del arg1148_1
        del arg1149_1
        del arg114_1
        del arg1150_1
        del arg1151_1
        del arg1152_1
        del arg1153_1
        del arg1154_1
        del arg1155_1
        del arg1156_1
        del arg1157_1
        del arg1158_1
        del arg1159_1
        del arg115_1
        del arg1160_1
        del arg1161_1
        del arg1162_1
        del arg1163_1
        del arg1164_1
        del arg1165_1
        del arg1166_1
        del arg1167_1
        del arg1168_1
        del arg1169_1
        del arg116_1
        del arg1170_1
        del arg1171_1
        del arg1172_1
        del arg1173_1
        del arg1174_1
        del arg1175_1
        del arg1176_1
        del arg1177_1
        del arg1178_1
        del arg1179_1
        del arg117_1
        del arg1180_1
        del arg1181_1
        del arg1182_1
        del arg1183_1
        del arg1184_1
        del arg1185_1
        del arg1186_1
        del arg1187_1
        del arg1188_1
        del arg1189_1
        del arg118_1
        del arg1190_1
        del arg1191_1
        del arg1192_1
        del arg1193_1
        del arg1194_1
        del arg1195_1
        del arg1196_1
        del arg1197_1
        del arg1198_1
        del arg1199_1
        del arg119_1
        del arg11_1
        del arg1200_1
        del arg1201_1
        del arg1202_1
        del arg1203_1
        del arg1204_1
        del arg1205_1
        del arg1206_1
        del arg1207_1
        del arg1208_1
        del arg1209_1
        del arg120_1
        del arg1210_1
        del arg1211_1
        del arg1212_1
        del arg1213_1
        del arg1214_1
        del arg1215_1
        del arg1216_1
        del arg1217_1
        del arg1218_1
        del arg1219_1
        del arg121_1
        del arg1220_1
        del arg1221_1
        del arg1222_1
        del arg1223_1
        del arg1224_1
        del arg1225_1
        del arg1226_1
        del arg1227_1
        del arg1228_1
        del arg1229_1
        del arg122_1
        del arg1230_1
        del arg1231_1
        del arg1232_1
        del arg1233_1
        del arg1234_1
        del arg1235_1
        del arg1236_1
        del arg1237_1
        del arg1238_1
        del arg1239_1
        del arg123_1
        del arg1240_1
        del arg1241_1
        del arg1242_1
        del arg1243_1
        del arg1244_1
        del arg1245_1
        del arg1246_1
        del arg1247_1
        del arg1248_1
        del arg1249_1
        del arg124_1
        del arg1250_1
        del arg1251_1
        del arg1252_1
        del arg1253_1
        del arg1254_1
        del arg1255_1
        del arg1256_1
        del arg1257_1
        del arg1258_1
        del arg1259_1
        del arg125_1
        del arg1260_1
        del arg1261_1
        del arg1262_1
        del arg1263_1
        del arg1264_1
        del arg1265_1
        del arg1266_1
        del arg1267_1
        del arg1268_1
        del arg1269_1
        del arg126_1
        del arg1270_1
        del arg1271_1
        del arg1272_1
        del arg1273_1
        del arg1274_1
        del arg1275_1
        del arg1276_1
        del arg1277_1
        del arg1278_1
        del arg1279_1
        del arg127_1
        del arg1280_1
        del arg1281_1
        del arg1282_1
        del arg1283_1
        del arg1284_1
        del arg1285_1
        del arg1286_1
        del arg1287_1
        del arg1288_1
        del arg1289_1
        del arg128_1
        del arg1290_1
        del arg1291_1
        del arg1292_1
        del arg1293_1
        del arg1294_1
        del arg1295_1
        del arg1296_1
        del arg1297_1
        del arg1298_1
        del arg1299_1
        del arg129_1
        del arg12_1
        del arg1300_1
        del arg1301_1
        del arg1302_1
        del arg1303_1
        del arg1304_1
        del arg1305_1
        del arg1306_1
        del arg1307_1
        del arg1308_1
        del arg1309_1
        del arg130_1
        del arg1310_1
        del arg1311_1
        del arg1312_1
        del arg1313_1
        del arg1314_1
        del arg1315_1
        del arg1316_1
        del arg1317_1
        del arg1318_1
        del arg1319_1
        del arg131_1
        del arg1320_1
        del arg1321_1
        del arg1322_1
        del arg1323_1
        del arg1324_1
        del arg1325_1
        del arg1326_1
        del arg1327_1
        del arg1328_1
        del arg1329_1
        del arg132_1
        del arg1330_1
        del arg1331_1
        del arg1332_1
        del arg1333_1
        del arg1334_1
        del arg1335_1
        del arg1336_1
        del arg1337_1
        del arg1338_1
        del arg1339_1
        del arg133_1
        del arg1340_1
        del arg1341_1
        del arg1342_1
        del arg1343_1
        del arg1344_1
        del arg1345_1
        del arg1346_1
        del arg1347_1
        del arg1348_1
        del arg1349_1
        del arg134_1
        del arg1350_1
        del arg1351_1
        del arg1352_1
        del arg1353_1
        del arg1354_1
        del arg1355_1
        del arg1356_1
        del arg1357_1
        del arg1358_1
        del arg1359_1
        del arg135_1
        del arg1360_1
        del arg1361_1
        del arg1362_1
        del arg1363_1
        del arg1364_1
        del arg1365_1
        del arg1366_1
        del arg1367_1
        del arg1368_1
        del arg1369_1
        del arg136_1
        del arg1370_1
        del arg1371_1
        del arg1372_1
        del arg1373_1
        del arg1374_1
        del arg1375_1
        del arg1376_1
        del arg1377_1
        del arg1378_1
        del arg1379_1
        del arg137_1
        del arg1380_1
        del arg1381_1
        del arg1382_1
        del arg1383_1
        del arg1384_1
        del arg1385_1
        del arg1386_1
        del arg1387_1
        del arg1388_1
        del arg1389_1
        del arg138_1
        del arg1390_1
        del arg1391_1
        del arg1392_1
        del arg1393_1
        del arg1394_1
        del arg1395_1
        del arg1396_1
        del arg1397_1
        del arg1398_1
        del arg1399_1
        del arg139_1
        del arg13_1
        del arg1400_1
        del arg1401_1
        del arg1402_1
        del arg1403_1
        del arg1404_1
        del arg1405_1
        del arg1406_1
        del arg1407_1
        del arg1408_1
        del arg1409_1
        del arg140_1
        del arg1410_1
        del arg1411_1
        del arg1412_1
        del arg1413_1
        del arg1414_1
        del arg1415_1
        del arg1416_1
        del arg1417_1
        del arg1418_1
        del arg1419_1
        del arg141_1
        del arg1420_1
        del arg1421_1
        del arg1422_1
        del arg1423_1
        del arg1424_1
        del arg1425_1
        del arg1426_1
        del arg1427_1
        del arg1428_1
        del arg1429_1
        del arg142_1
        del arg1430_1
        del arg1431_1
        del arg1432_1
        del arg1433_1
        del arg1434_1
        del arg1435_1
        del arg1436_1
        del arg1437_1
        del arg1438_1
        del arg1439_1
        del arg143_1
        del arg1440_1
        del arg1441_1
        del arg1442_1
        del arg1443_1
        del arg1444_1
        del arg1445_1
        del arg1446_1
        del arg1447_1
        del arg1448_1
        del arg1449_1
        del arg144_1
        del arg1450_1
        del arg1451_1
        del arg1452_1
        del arg1453_1
        del arg1454_1
        del arg1455_1
        del arg1456_1
        del arg1457_1
        del arg1458_1
        del arg1459_1
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
        del arg805_1
        del arg806_1
        del arg807_1
        del arg808_1
        del arg809_1
        del arg80_1
        del arg810_1
        del arg811_1
        del arg812_1
        del arg813_1
        del arg814_1
        del arg815_1
        del arg816_1
        del arg817_1
        del arg818_1
        del arg819_1
        del arg81_1
        del arg820_1
        del arg821_1
        del arg822_1
        del arg823_1
        del arg824_1
        del arg825_1
        del arg826_1
        del arg827_1
        del arg828_1
        del arg829_1
        del arg82_1
        del arg830_1
        del arg831_1
        del arg832_1
        del arg833_1
        del arg834_1
        del arg835_1
        del arg836_1
        del arg837_1
        del arg838_1
        del arg839_1
        del arg83_1
        del arg840_1
        del arg841_1
        del arg842_1
        del arg843_1
        del arg844_1
        del arg845_1
        del arg846_1
        del arg847_1
        del arg848_1
        del arg849_1
        del arg84_1
        del arg850_1
        del arg851_1
        del arg852_1
        del arg853_1
        del arg854_1
        del arg855_1
        del arg856_1
        del arg857_1
        del arg858_1
        del arg859_1
        del arg85_1
        del arg860_1
        del arg861_1
        del arg862_1
        del arg863_1
        del arg864_1
        del arg865_1
        del arg866_1
        del arg867_1
        del arg868_1
        del arg869_1
        del arg86_1
        del arg870_1
        del arg871_1
        del arg872_1
        del arg873_1
        del arg874_1
        del arg875_1
        del arg876_1
        del arg877_1
        del arg878_1
        del arg879_1
        del arg87_1
        del arg880_1
        del arg881_1
        del arg882_1
        del arg883_1
        del arg884_1
        del arg885_1
        del arg886_1
        del arg887_1
        del arg888_1
        del arg889_1
        del arg88_1
        del arg890_1
        del arg891_1
        del arg892_1
        del arg893_1
        del arg894_1
        del arg895_1
        del arg896_1
        del arg897_1
        del arg898_1
        del arg899_1
        del arg89_1
        del arg8_1
        del arg900_1
        del arg901_1
        del arg902_1
        del arg903_1
        del arg904_1
        del arg905_1
        del arg906_1
        del arg907_1
        del arg908_1
        del arg909_1
        del arg90_1
        del arg910_1
        del arg911_1
        del arg912_1
        del arg913_1
        del arg914_1
        del arg915_1
        del arg916_1
        del arg917_1
        del arg918_1
        del arg919_1
        del arg91_1
        del arg920_1
        del arg921_1
        del arg922_1
        del arg923_1
        del arg924_1
        del arg925_1
        del arg926_1
        del arg927_1
        del arg928_1
        del arg929_1
        del arg92_1
        del arg930_1
        del arg931_1
        del arg932_1
        del arg933_1
        del arg934_1
        del arg935_1
        del arg936_1
        del arg937_1
        del arg938_1
        del arg939_1
        del arg93_1
        del arg940_1
        del arg941_1
        del arg942_1
        del arg943_1
        del arg944_1
        del arg945_1
        del arg946_1
        del arg947_1
        del arg948_1
        del arg949_1
        del arg94_1
        del arg950_1
        del arg951_1
        del arg952_1
        del arg953_1
        del arg954_1
        del arg955_1
        del arg956_1
        del arg957_1
        del arg958_1
        del arg959_1
        del arg95_1
        del arg960_1
        del arg961_1
        del arg962_1
        del arg963_1
        del arg964_1
        del arg965_1
        del arg966_1
        del arg967_1
        del arg968_1
        del arg969_1
        del arg96_1
        del arg970_1
        del arg971_1
        del arg972_1
        del arg973_1
        del arg974_1
        del arg975_1
        del arg976_1
        del arg977_1
        del arg978_1
        del arg979_1
        del arg97_1
        del arg980_1
        del arg981_1
        del arg982_1
        del arg983_1
        del arg984_1
        del arg985_1
        del arg986_1
        del arg987_1
        del arg988_1
        del arg989_1
        del arg98_1
        del arg990_1
        del arg991_1
        del arg992_1
        del arg993_1
        del arg994_1
        del arg995_1
        del arg996_1
        del arg997_1
        del arg998_1
        del arg999_1
        del arg99_1
        del arg9_1
    return ()


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((80, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((192, 80, 3, 3), (720, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((64, 192, 1, 1), (192, 1, 192, 192), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((48, 192, 1, 1), (192, 1, 192, 192), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((64, 48, 5, 5), (1200, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((64, 192, 1, 1), (192, 1, 192, 192), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((32, 192, 1, 1), (192, 1, 192, 192), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((64, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((48, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((64, 48, 5, 5), (1200, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((64, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((64, 288, 1, 1), (288, 1, 288, 288), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((48, 288, 1, 1), (288, 1, 288, 288), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((64, 48, 5, 5), (1200, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((64, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((64, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((384, 288, 3, 3), (2592, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((64, 288, 1, 1), (288, 1, 288, 288), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((192, 768, 1, 1), (768, 1, 768, 768), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((128, 768, 1, 1), (768, 1, 768, 768), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((128, 128, 1, 7), (896, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((192, 128, 7, 1), (896, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((128, 768, 1, 1), (768, 1, 768, 768), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((128, 128, 7, 1), (896, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((128, 128, 1, 7), (896, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((128, 128, 7, 1), (896, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((192, 128, 1, 7), (896, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((160, 768, 1, 1), (768, 1, 768, 768), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((160, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((192, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((160, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((160, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((160, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((160, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((192, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((160, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((160, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((192, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((160, 768, 1, 1), (768, 1, 768, 768), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((160, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((160, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((160, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((192, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((192, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((192, 192, 7, 1), (1344, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((192, 192, 7, 1), (1344, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((192, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((192, 192, 7, 1), (1344, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((192, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((128, 768, 1, 1), (768, 1, 768, 768), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((768, 128, 5, 5), (3200, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((320, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((192, 768, 1, 1), (768, 1, 768, 768), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((192, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((192, 192, 7, 1), (1344, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((320, 1280, 1, 1), (1280, 1, 1280, 1280), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((384, 1280, 1, 1), (1280, 1, 1280, 1280), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((384, 384, 1, 3), (1152, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((384, 384, 3, 1), (1152, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((448, 1280, 1, 1), (1280, 1, 1280, 1280), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((384, 448, 3, 3), (4032, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((384, 384, 1, 3), (1152, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((384, 384, 3, 1), (1152, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((192, 1280, 1, 1), (1280, 1, 1280, 1280), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((320, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((384, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((384, 384, 1, 3), (1152, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((384, 384, 3, 1), (1152, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((448, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((384, 448, 3, 3), (4032, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((384, 384, 1, 3), (1152, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((384, 384, 3, 1), (1152, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((192, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((80, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((192, 80, 3, 3), (720, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((64, 192, 1, 1), (192, 1, 192, 192), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((48, 192, 1, 1), (192, 1, 192, 192), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((64, 48, 5, 5), (1200, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((64, 192, 1, 1), (192, 1, 192, 192), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((32, 192, 1, 1), (192, 1, 192, 192), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((64, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((48, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((64, 48, 5, 5), (1200, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((64, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((64, 288, 1, 1), (288, 1, 288, 288), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((48, 288, 1, 1), (288, 1, 288, 288), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((64, 48, 5, 5), (1200, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((64, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((64, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((384, 288, 3, 3), (2592, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((64, 288, 1, 1), (288, 1, 288, 288), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((192, 768, 1, 1), (768, 1, 768, 768), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((128, 768, 1, 1), (768, 1, 768, 768), device='cuda:0', dtype=torch.float32)
    arg386_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((128, 128, 1, 7), (896, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg389_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg390_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg391_1 = rand_strided((192, 128, 7, 1), (896, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg392_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg393_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg394_1 = rand_strided((128, 768, 1, 1), (768, 1, 768, 768), device='cuda:0', dtype=torch.float32)
    arg395_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg396_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg397_1 = rand_strided((128, 128, 7, 1), (896, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg398_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg399_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg400_1 = rand_strided((128, 128, 1, 7), (896, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg401_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg402_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg403_1 = rand_strided((128, 128, 7, 1), (896, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg404_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg405_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg406_1 = rand_strided((192, 128, 1, 7), (896, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg407_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg408_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg409_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg410_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg411_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg412_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg413_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg414_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg415_1 = rand_strided((160, 768, 1, 1), (768, 1, 768, 768), device='cuda:0', dtype=torch.float32)
    arg416_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg417_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg418_1 = rand_strided((160, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg419_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg420_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg421_1 = rand_strided((192, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg422_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg423_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg424_1 = rand_strided((160, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg425_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg426_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg427_1 = rand_strided((160, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg428_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg429_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg430_1 = rand_strided((160, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg431_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg432_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg433_1 = rand_strided((160, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg434_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg435_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg436_1 = rand_strided((192, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg437_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg438_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg439_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg440_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg441_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg442_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg443_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg444_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg445_1 = rand_strided((160, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg446_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg447_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg448_1 = rand_strided((160, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg449_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg450_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg451_1 = rand_strided((192, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg452_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg453_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg454_1 = rand_strided((160, 768, 1, 1), (768, 1, 768, 768), device='cuda:0', dtype=torch.float32)
    arg455_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg456_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg457_1 = rand_strided((160, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg458_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg459_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg460_1 = rand_strided((160, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg461_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg462_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg463_1 = rand_strided((160, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg464_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg465_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg466_1 = rand_strided((192, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg467_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg468_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg469_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg470_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg471_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg472_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg473_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg474_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg475_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg476_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg477_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg478_1 = rand_strided((192, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg479_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg480_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg481_1 = rand_strided((192, 192, 7, 1), (1344, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg482_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg483_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg484_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg485_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg486_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg487_1 = rand_strided((192, 192, 7, 1), (1344, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg488_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg489_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg490_1 = rand_strided((192, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg491_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg492_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg493_1 = rand_strided((192, 192, 7, 1), (1344, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg494_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg495_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg496_1 = rand_strided((192, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg497_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg498_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg499_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg500_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg501_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg502_1 = rand_strided((128, 768, 1, 1), (768, 1, 768, 768), device='cuda:0', dtype=torch.float32)
    arg503_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg504_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg505_1 = rand_strided((768, 128, 5, 5), (3200, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg506_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg507_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg508_1 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg509_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg510_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg511_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg512_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg513_1 = rand_strided((320, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg514_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg515_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg516_1 = rand_strided((192, 768, 1, 1), (768, 1, 768, 768), device='cuda:0', dtype=torch.float32)
    arg517_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg518_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg519_1 = rand_strided((192, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg520_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg521_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg522_1 = rand_strided((192, 192, 7, 1), (1344, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg523_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg524_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg525_1 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg526_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg527_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg528_1 = rand_strided((320, 1280, 1, 1), (1280, 1, 1280, 1280), device='cuda:0', dtype=torch.float32)
    arg529_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg530_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg531_1 = rand_strided((384, 1280, 1, 1), (1280, 1, 1280, 1280), device='cuda:0', dtype=torch.float32)
    arg532_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg533_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg534_1 = rand_strided((384, 384, 1, 3), (1152, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    arg535_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg536_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg537_1 = rand_strided((384, 384, 3, 1), (1152, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    arg538_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg539_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg540_1 = rand_strided((448, 1280, 1, 1), (1280, 1, 1280, 1280), device='cuda:0', dtype=torch.float32)
    arg541_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg542_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg543_1 = rand_strided((384, 448, 3, 3), (4032, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg544_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg545_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg546_1 = rand_strided((384, 384, 1, 3), (1152, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    arg547_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg548_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg549_1 = rand_strided((384, 384, 3, 1), (1152, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    arg550_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg551_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg552_1 = rand_strided((192, 1280, 1, 1), (1280, 1, 1280, 1280), device='cuda:0', dtype=torch.float32)
    arg553_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg554_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg555_1 = rand_strided((320, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda:0', dtype=torch.float32)
    arg556_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg557_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg558_1 = rand_strided((384, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda:0', dtype=torch.float32)
    arg559_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg560_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg561_1 = rand_strided((384, 384, 1, 3), (1152, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    arg562_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg563_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg564_1 = rand_strided((384, 384, 3, 1), (1152, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    arg565_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg566_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg567_1 = rand_strided((448, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda:0', dtype=torch.float32)
    arg568_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg569_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg570_1 = rand_strided((384, 448, 3, 3), (4032, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg571_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg572_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg573_1 = rand_strided((384, 384, 1, 3), (1152, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    arg574_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg575_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg576_1 = rand_strided((384, 384, 3, 1), (1152, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    arg577_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg578_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg579_1 = rand_strided((192, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda:0', dtype=torch.float32)
    arg580_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg581_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg582_1 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg583_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg584_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg585_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg586_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg587_1 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg588_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg589_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg590_1 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg591_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg592_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg593_1 = rand_strided((80, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.float32)
    arg594_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg595_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg596_1 = rand_strided((192, 80, 3, 3), (720, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg597_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg598_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg599_1 = rand_strided((64, 192, 1, 1), (192, 1, 192, 192), device='cuda:0', dtype=torch.float32)
    arg600_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg601_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg602_1 = rand_strided((48, 192, 1, 1), (192, 1, 192, 192), device='cuda:0', dtype=torch.float32)
    arg603_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg604_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg605_1 = rand_strided((64, 48, 5, 5), (1200, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg606_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg607_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg608_1 = rand_strided((64, 192, 1, 1), (192, 1, 192, 192), device='cuda:0', dtype=torch.float32)
    arg609_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg610_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg611_1 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg612_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg613_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg614_1 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg615_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg616_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg617_1 = rand_strided((32, 192, 1, 1), (192, 1, 192, 192), device='cuda:0', dtype=torch.float32)
    arg618_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg619_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg620_1 = rand_strided((64, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg621_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg622_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg623_1 = rand_strided((48, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg624_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg625_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg626_1 = rand_strided((64, 48, 5, 5), (1200, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg627_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg628_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg629_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg630_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg631_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg632_1 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg633_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg634_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg635_1 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg636_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg637_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg638_1 = rand_strided((64, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg639_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg640_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg641_1 = rand_strided((64, 288, 1, 1), (288, 1, 288, 288), device='cuda:0', dtype=torch.float32)
    arg642_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg643_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg644_1 = rand_strided((48, 288, 1, 1), (288, 1, 288, 288), device='cuda:0', dtype=torch.float32)
    arg645_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg646_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg647_1 = rand_strided((64, 48, 5, 5), (1200, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg648_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg649_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg650_1 = rand_strided((64, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg651_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg652_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg653_1 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg654_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg655_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg656_1 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg657_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg658_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg659_1 = rand_strided((64, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg660_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg661_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg662_1 = rand_strided((384, 288, 3, 3), (2592, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg663_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg664_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg665_1 = rand_strided((64, 288, 1, 1), (288, 1, 288, 288), device='cuda:0', dtype=torch.float32)
    arg666_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg667_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg668_1 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg669_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg670_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg671_1 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg672_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg673_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg674_1 = rand_strided((192, 768, 1, 1), (768, 1, 768, 768), device='cuda:0', dtype=torch.float32)
    arg675_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg676_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg677_1 = rand_strided((128, 768, 1, 1), (768, 1, 768, 768), device='cuda:0', dtype=torch.float32)
    arg678_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg679_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg680_1 = rand_strided((128, 128, 1, 7), (896, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg681_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg682_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg683_1 = rand_strided((192, 128, 7, 1), (896, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg684_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg685_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg686_1 = rand_strided((128, 768, 1, 1), (768, 1, 768, 768), device='cuda:0', dtype=torch.float32)
    arg687_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg688_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg689_1 = rand_strided((128, 128, 7, 1), (896, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg690_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg691_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg692_1 = rand_strided((128, 128, 1, 7), (896, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg693_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg694_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg695_1 = rand_strided((128, 128, 7, 1), (896, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg696_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg697_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg698_1 = rand_strided((192, 128, 1, 7), (896, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg699_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg700_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg701_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg702_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg703_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg704_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg705_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg706_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg707_1 = rand_strided((160, 768, 1, 1), (768, 1, 768, 768), device='cuda:0', dtype=torch.float32)
    arg708_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg709_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg710_1 = rand_strided((160, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg711_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg712_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg713_1 = rand_strided((192, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg714_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg715_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg716_1 = rand_strided((160, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg717_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg718_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg719_1 = rand_strided((160, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg720_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg721_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg722_1 = rand_strided((160, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg723_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg724_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg725_1 = rand_strided((160, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg726_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg727_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg728_1 = rand_strided((192, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg729_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg730_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg731_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg732_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg733_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg734_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg735_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg736_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg737_1 = rand_strided((160, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg738_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg739_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg740_1 = rand_strided((160, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg741_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg742_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg743_1 = rand_strided((192, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg744_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg745_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg746_1 = rand_strided((160, 768, 1, 1), (768, 1, 768, 768), device='cuda:0', dtype=torch.float32)
    arg747_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg748_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg749_1 = rand_strided((160, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg750_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg751_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg752_1 = rand_strided((160, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg753_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg754_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg755_1 = rand_strided((160, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg756_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg757_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg758_1 = rand_strided((192, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg759_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg760_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg761_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg762_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg763_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg764_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg765_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg766_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg767_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg768_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg769_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg770_1 = rand_strided((192, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg771_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg772_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg773_1 = rand_strided((192, 192, 7, 1), (1344, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg774_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg775_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg776_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg777_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg778_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg779_1 = rand_strided((192, 192, 7, 1), (1344, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg780_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg781_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg782_1 = rand_strided((192, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg783_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg784_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg785_1 = rand_strided((192, 192, 7, 1), (1344, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg786_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg787_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg788_1 = rand_strided((192, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg789_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg790_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg791_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg792_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg793_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg794_1 = rand_strided((128, 768, 1, 1), (768, 1, 768, 768), device='cuda:0', dtype=torch.float32)
    arg795_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg796_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg797_1 = rand_strided((768, 128, 5, 5), (3200, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg798_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg799_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg800_1 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg801_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg802_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg803_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg804_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg805_1 = rand_strided((320, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg806_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg807_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg808_1 = rand_strided((192, 768, 1, 1), (768, 1, 768, 768), device='cuda:0', dtype=torch.float32)
    arg809_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg810_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg811_1 = rand_strided((192, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg812_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg813_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg814_1 = rand_strided((192, 192, 7, 1), (1344, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg815_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg816_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg817_1 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg818_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg819_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg820_1 = rand_strided((320, 1280, 1, 1), (1280, 1, 1280, 1280), device='cuda:0', dtype=torch.float32)
    arg821_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg822_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg823_1 = rand_strided((384, 1280, 1, 1), (1280, 1, 1280, 1280), device='cuda:0', dtype=torch.float32)
    arg824_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg825_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg826_1 = rand_strided((384, 384, 1, 3), (1152, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    arg827_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg828_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg829_1 = rand_strided((384, 384, 3, 1), (1152, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    arg830_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg831_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg832_1 = rand_strided((448, 1280, 1, 1), (1280, 1, 1280, 1280), device='cuda:0', dtype=torch.float32)
    arg833_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg834_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg835_1 = rand_strided((384, 448, 3, 3), (4032, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg836_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg837_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg838_1 = rand_strided((384, 384, 1, 3), (1152, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    arg839_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg840_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg841_1 = rand_strided((384, 384, 3, 1), (1152, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    arg842_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg843_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg844_1 = rand_strided((192, 1280, 1, 1), (1280, 1, 1280, 1280), device='cuda:0', dtype=torch.float32)
    arg845_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg846_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg847_1 = rand_strided((320, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda:0', dtype=torch.float32)
    arg848_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg849_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg850_1 = rand_strided((384, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda:0', dtype=torch.float32)
    arg851_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg852_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg853_1 = rand_strided((384, 384, 1, 3), (1152, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    arg854_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg855_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg856_1 = rand_strided((384, 384, 3, 1), (1152, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    arg857_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg858_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg859_1 = rand_strided((448, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda:0', dtype=torch.float32)
    arg860_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg861_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg862_1 = rand_strided((384, 448, 3, 3), (4032, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg863_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg864_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg865_1 = rand_strided((384, 384, 1, 3), (1152, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    arg866_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg867_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg868_1 = rand_strided((384, 384, 3, 1), (1152, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    arg869_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg870_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg871_1 = rand_strided((192, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda:0', dtype=torch.float32)
    arg872_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg873_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg874_1 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg875_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg876_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg877_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg878_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg879_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg880_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg881_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg882_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg883_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg884_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg885_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg886_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg887_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg888_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg889_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg890_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg891_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg892_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg893_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg894_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg895_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg896_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg897_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg898_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg899_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg900_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg901_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg902_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg903_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg904_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg905_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg906_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg907_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg908_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg909_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg910_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg911_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg912_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg913_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg914_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg915_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg916_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg917_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg918_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg919_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg920_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg921_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg922_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg923_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg924_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg925_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg926_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg927_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg928_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg929_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg930_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg931_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg932_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg933_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg934_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg935_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg936_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg937_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg938_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg939_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg940_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg941_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg942_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg943_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg944_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg945_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg946_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg947_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg948_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg949_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg950_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg951_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg952_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg953_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg954_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg955_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg956_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg957_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg958_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg959_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg960_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg961_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg962_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg963_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg964_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg965_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg966_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg967_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg968_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg969_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg970_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg971_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg972_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg973_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg974_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg975_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg976_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg977_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg978_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg979_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg980_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg981_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg982_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg983_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg984_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg985_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg986_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg987_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg988_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg989_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg990_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg991_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg992_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg993_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg994_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg995_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg996_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg997_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg998_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg999_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1000_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1001_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1002_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1003_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1004_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1005_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1006_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1007_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1008_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1009_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1010_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1011_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1012_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1013_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1014_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1015_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1016_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1017_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1018_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1019_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1020_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1021_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1022_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1023_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1024_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1025_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1026_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1027_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1028_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1029_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1030_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1031_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1032_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1033_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1034_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1035_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1036_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1037_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1038_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1039_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1040_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1041_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1042_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1043_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1044_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1045_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1046_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1047_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1048_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1049_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1050_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1051_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1052_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1053_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1054_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1055_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1056_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1057_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1058_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1059_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1060_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1061_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1062_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1063_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1064_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1065_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1066_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1067_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1068_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1069_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1070_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1071_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1072_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1073_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1074_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1075_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1076_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1077_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1078_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1079_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1080_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1081_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1082_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1083_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1084_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1085_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1086_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1087_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1088_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1089_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1090_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1091_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1092_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1093_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1094_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1095_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1096_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1097_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1098_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1099_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1100_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1101_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1102_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1103_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1104_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1105_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1106_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1107_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1108_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1109_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1110_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1111_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1112_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1113_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1114_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1115_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1116_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1117_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1118_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1119_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1120_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1121_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1122_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1123_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1124_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1125_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1126_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1127_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1128_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1129_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1130_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1131_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1132_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1133_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1134_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1135_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1136_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1137_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1138_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1139_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1140_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1141_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1142_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1143_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1144_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1145_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1146_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1147_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1148_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1149_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1150_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1151_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1152_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1153_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1154_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1155_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1156_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1157_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1158_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1159_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1160_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1161_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1162_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1163_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1164_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1165_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1166_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1167_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg1168_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1169_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1170_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1171_1 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1172_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1173_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1174_1 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1175_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1176_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1177_1 = rand_strided((80, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.float32)
    arg1178_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1179_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1180_1 = rand_strided((192, 80, 3, 3), (720, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1181_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1182_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1183_1 = rand_strided((64, 192, 1, 1), (192, 1, 192, 192), device='cuda:0', dtype=torch.float32)
    arg1184_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1185_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1186_1 = rand_strided((48, 192, 1, 1), (192, 1, 192, 192), device='cuda:0', dtype=torch.float32)
    arg1187_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1188_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1189_1 = rand_strided((64, 48, 5, 5), (1200, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg1190_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1191_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1192_1 = rand_strided((64, 192, 1, 1), (192, 1, 192, 192), device='cuda:0', dtype=torch.float32)
    arg1193_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1194_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1195_1 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1196_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1197_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1198_1 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1199_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1200_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1201_1 = rand_strided((32, 192, 1, 1), (192, 1, 192, 192), device='cuda:0', dtype=torch.float32)
    arg1202_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1203_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1204_1 = rand_strided((64, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg1205_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1206_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1207_1 = rand_strided((48, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg1208_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1209_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1210_1 = rand_strided((64, 48, 5, 5), (1200, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg1211_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1212_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1213_1 = rand_strided((64, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg1214_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1215_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1216_1 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1217_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1218_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1219_1 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1220_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1221_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1222_1 = rand_strided((64, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg1223_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1224_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1225_1 = rand_strided((64, 288, 1, 1), (288, 1, 288, 288), device='cuda:0', dtype=torch.float32)
    arg1226_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1227_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1228_1 = rand_strided((48, 288, 1, 1), (288, 1, 288, 288), device='cuda:0', dtype=torch.float32)
    arg1229_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1230_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1231_1 = rand_strided((64, 48, 5, 5), (1200, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg1232_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1233_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1234_1 = rand_strided((64, 288, 1, 1), (288, 1, 288, 288), device='cuda:0', dtype=torch.float32)
    arg1235_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1236_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1237_1 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1238_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1239_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1240_1 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1241_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1242_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1243_1 = rand_strided((64, 288, 1, 1), (288, 1, 288, 288), device='cuda:0', dtype=torch.float32)
    arg1244_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1245_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1246_1 = rand_strided((384, 288, 3, 3), (2592, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1247_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1248_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1249_1 = rand_strided((64, 288, 1, 1), (288, 1, 288, 288), device='cuda:0', dtype=torch.float32)
    arg1250_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1251_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1252_1 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1253_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1254_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1255_1 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1256_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1257_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1258_1 = rand_strided((192, 768, 1, 1), (768, 1, 768, 768), device='cuda:0', dtype=torch.float32)
    arg1259_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1260_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1261_1 = rand_strided((128, 768, 1, 1), (768, 1, 768, 768), device='cuda:0', dtype=torch.float32)
    arg1262_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1263_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1264_1 = rand_strided((128, 128, 1, 7), (896, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg1265_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1266_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1267_1 = rand_strided((192, 128, 7, 1), (896, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1268_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1269_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1270_1 = rand_strided((128, 768, 1, 1), (768, 1, 768, 768), device='cuda:0', dtype=torch.float32)
    arg1271_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1272_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1273_1 = rand_strided((128, 128, 7, 1), (896, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1274_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1275_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1276_1 = rand_strided((128, 128, 1, 7), (896, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg1277_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1278_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1279_1 = rand_strided((128, 128, 7, 1), (896, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1280_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1281_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1282_1 = rand_strided((192, 128, 1, 7), (896, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg1283_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1284_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1285_1 = rand_strided((192, 768, 1, 1), (768, 1, 768, 768), device='cuda:0', dtype=torch.float32)
    arg1286_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1287_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1288_1 = rand_strided((192, 768, 1, 1), (768, 1, 768, 768), device='cuda:0', dtype=torch.float32)
    arg1289_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1290_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1291_1 = rand_strided((160, 768, 1, 1), (768, 1, 768, 768), device='cuda:0', dtype=torch.float32)
    arg1292_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1293_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1294_1 = rand_strided((160, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg1295_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1296_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1297_1 = rand_strided((192, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1298_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1299_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1300_1 = rand_strided((160, 768, 1, 1), (768, 1, 768, 768), device='cuda:0', dtype=torch.float32)
    arg1301_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1302_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1303_1 = rand_strided((160, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1304_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1305_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1306_1 = rand_strided((160, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg1307_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1308_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1309_1 = rand_strided((160, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1310_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1311_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1312_1 = rand_strided((192, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg1313_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1314_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1315_1 = rand_strided((192, 768, 1, 1), (768, 1, 768, 768), device='cuda:0', dtype=torch.float32)
    arg1316_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1317_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1318_1 = rand_strided((192, 768, 1, 1), (768, 1, 768, 768), device='cuda:0', dtype=torch.float32)
    arg1319_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1320_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1321_1 = rand_strided((160, 768, 1, 1), (768, 1, 768, 768), device='cuda:0', dtype=torch.float32)
    arg1322_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1323_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1324_1 = rand_strided((160, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg1325_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1326_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1327_1 = rand_strided((192, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1328_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1329_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1330_1 = rand_strided((160, 768, 1, 1), (768, 1, 768, 768), device='cuda:0', dtype=torch.float32)
    arg1331_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1332_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1333_1 = rand_strided((160, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1334_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1335_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1336_1 = rand_strided((160, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg1337_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1338_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1339_1 = rand_strided((160, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1340_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1341_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1342_1 = rand_strided((192, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg1343_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1344_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1345_1 = rand_strided((192, 768, 1, 1), (768, 1, 768, 768), device='cuda:0', dtype=torch.float32)
    arg1346_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1347_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1348_1 = rand_strided((192, 768, 1, 1), (768, 1, 768, 768), device='cuda:0', dtype=torch.float32)
    arg1349_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1350_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1351_1 = rand_strided((192, 768, 1, 1), (768, 1, 768, 768), device='cuda:0', dtype=torch.float32)
    arg1352_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1353_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1354_1 = rand_strided((192, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg1355_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1356_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1357_1 = rand_strided((192, 192, 7, 1), (1344, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1358_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1359_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1360_1 = rand_strided((192, 768, 1, 1), (768, 1, 768, 768), device='cuda:0', dtype=torch.float32)
    arg1361_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1362_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1363_1 = rand_strided((192, 192, 7, 1), (1344, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1364_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1365_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1366_1 = rand_strided((192, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg1367_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1368_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1369_1 = rand_strided((192, 192, 7, 1), (1344, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1370_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1371_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1372_1 = rand_strided((192, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg1373_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1374_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1375_1 = rand_strided((192, 768, 1, 1), (768, 1, 768, 768), device='cuda:0', dtype=torch.float32)
    arg1376_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1377_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1378_1 = rand_strided((128, 768, 1, 1), (768, 1, 768, 768), device='cuda:0', dtype=torch.float32)
    arg1379_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1380_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1381_1 = rand_strided((768, 128, 5, 5), (3200, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg1382_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1383_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1384_1 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg1385_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1386_1 = rand_strided((192, 768, 1, 1), (768, 1, 768, 768), device='cuda:0', dtype=torch.float32)
    arg1387_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1388_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1389_1 = rand_strided((320, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1390_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1391_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1392_1 = rand_strided((192, 768, 1, 1), (768, 1, 768, 768), device='cuda:0', dtype=torch.float32)
    arg1393_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1394_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1395_1 = rand_strided((192, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg1396_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1397_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1398_1 = rand_strided((192, 192, 7, 1), (1344, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1399_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1400_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1401_1 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1402_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1403_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1404_1 = rand_strided((320, 1280, 1, 1), (1280, 1, 1280, 1280), device='cuda:0', dtype=torch.float32)
    arg1405_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1406_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1407_1 = rand_strided((384, 1280, 1, 1), (1280, 1, 1280, 1280), device='cuda:0', dtype=torch.float32)
    arg1408_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1409_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1410_1 = rand_strided((384, 384, 1, 3), (1152, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1411_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1412_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1413_1 = rand_strided((384, 384, 3, 1), (1152, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1414_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1415_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1416_1 = rand_strided((448, 1280, 1, 1), (1280, 1, 1280, 1280), device='cuda:0', dtype=torch.float32)
    arg1417_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1418_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1419_1 = rand_strided((384, 448, 3, 3), (4032, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1420_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1421_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1422_1 = rand_strided((384, 384, 1, 3), (1152, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1423_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1424_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1425_1 = rand_strided((384, 384, 3, 1), (1152, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1426_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1427_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1428_1 = rand_strided((192, 1280, 1, 1), (1280, 1, 1280, 1280), device='cuda:0', dtype=torch.float32)
    arg1429_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1430_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1431_1 = rand_strided((320, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda:0', dtype=torch.float32)
    arg1432_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1433_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1434_1 = rand_strided((384, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda:0', dtype=torch.float32)
    arg1435_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1436_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1437_1 = rand_strided((384, 384, 1, 3), (1152, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1438_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1439_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1440_1 = rand_strided((384, 384, 3, 1), (1152, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1441_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1442_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1443_1 = rand_strided((448, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda:0', dtype=torch.float32)
    arg1444_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1445_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1446_1 = rand_strided((384, 448, 3, 3), (4032, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1447_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1448_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1449_1 = rand_strided((384, 384, 1, 3), (1152, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1450_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1451_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1452_1 = rand_strided((384, 384, 3, 1), (1152, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1453_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1454_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1455_1 = rand_strided((192, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda:0', dtype=torch.float32)
    arg1456_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1457_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1458_1 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg1459_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1, arg728_1, arg729_1, arg730_1, arg731_1, arg732_1, arg733_1, arg734_1, arg735_1, arg736_1, arg737_1, arg738_1, arg739_1, arg740_1, arg741_1, arg742_1, arg743_1, arg744_1, arg745_1, arg746_1, arg747_1, arg748_1, arg749_1, arg750_1, arg751_1, arg752_1, arg753_1, arg754_1, arg755_1, arg756_1, arg757_1, arg758_1, arg759_1, arg760_1, arg761_1, arg762_1, arg763_1, arg764_1, arg765_1, arg766_1, arg767_1, arg768_1, arg769_1, arg770_1, arg771_1, arg772_1, arg773_1, arg774_1, arg775_1, arg776_1, arg777_1, arg778_1, arg779_1, arg780_1, arg781_1, arg782_1, arg783_1, arg784_1, arg785_1, arg786_1, arg787_1, arg788_1, arg789_1, arg790_1, arg791_1, arg792_1, arg793_1, arg794_1, arg795_1, arg796_1, arg797_1, arg798_1, arg799_1, arg800_1, arg801_1, arg802_1, arg803_1, arg804_1, arg805_1, arg806_1, arg807_1, arg808_1, arg809_1, arg810_1, arg811_1, arg812_1, arg813_1, arg814_1, arg815_1, arg816_1, arg817_1, arg818_1, arg819_1, arg820_1, arg821_1, arg822_1, arg823_1, arg824_1, arg825_1, arg826_1, arg827_1, arg828_1, arg829_1, arg830_1, arg831_1, arg832_1, arg833_1, arg834_1, arg835_1, arg836_1, arg837_1, arg838_1, arg839_1, arg840_1, arg841_1, arg842_1, arg843_1, arg844_1, arg845_1, arg846_1, arg847_1, arg848_1, arg849_1, arg850_1, arg851_1, arg852_1, arg853_1, arg854_1, arg855_1, arg856_1, arg857_1, arg858_1, arg859_1, arg860_1, arg861_1, arg862_1, arg863_1, arg864_1, arg865_1, arg866_1, arg867_1, arg868_1, arg869_1, arg870_1, arg871_1, arg872_1, arg873_1, arg874_1, arg875_1, arg876_1, arg877_1, arg878_1, arg879_1, arg880_1, arg881_1, arg882_1, arg883_1, arg884_1, arg885_1, arg886_1, arg887_1, arg888_1, arg889_1, arg890_1, arg891_1, arg892_1, arg893_1, arg894_1, arg895_1, arg896_1, arg897_1, arg898_1, arg899_1, arg900_1, arg901_1, arg902_1, arg903_1, arg904_1, arg905_1, arg906_1, arg907_1, arg908_1, arg909_1, arg910_1, arg911_1, arg912_1, arg913_1, arg914_1, arg915_1, arg916_1, arg917_1, arg918_1, arg919_1, arg920_1, arg921_1, arg922_1, arg923_1, arg924_1, arg925_1, arg926_1, arg927_1, arg928_1, arg929_1, arg930_1, arg931_1, arg932_1, arg933_1, arg934_1, arg935_1, arg936_1, arg937_1, arg938_1, arg939_1, arg940_1, arg941_1, arg942_1, arg943_1, arg944_1, arg945_1, arg946_1, arg947_1, arg948_1, arg949_1, arg950_1, arg951_1, arg952_1, arg953_1, arg954_1, arg955_1, arg956_1, arg957_1, arg958_1, arg959_1, arg960_1, arg961_1, arg962_1, arg963_1, arg964_1, arg965_1, arg966_1, arg967_1, arg968_1, arg969_1, arg970_1, arg971_1, arg972_1, arg973_1, arg974_1, arg975_1, arg976_1, arg977_1, arg978_1, arg979_1, arg980_1, arg981_1, arg982_1, arg983_1, arg984_1, arg985_1, arg986_1, arg987_1, arg988_1, arg989_1, arg990_1, arg991_1, arg992_1, arg993_1, arg994_1, arg995_1, arg996_1, arg997_1, arg998_1, arg999_1, arg1000_1, arg1001_1, arg1002_1, arg1003_1, arg1004_1, arg1005_1, arg1006_1, arg1007_1, arg1008_1, arg1009_1, arg1010_1, arg1011_1, arg1012_1, arg1013_1, arg1014_1, arg1015_1, arg1016_1, arg1017_1, arg1018_1, arg1019_1, arg1020_1, arg1021_1, arg1022_1, arg1023_1, arg1024_1, arg1025_1, arg1026_1, arg1027_1, arg1028_1, arg1029_1, arg1030_1, arg1031_1, arg1032_1, arg1033_1, arg1034_1, arg1035_1, arg1036_1, arg1037_1, arg1038_1, arg1039_1, arg1040_1, arg1041_1, arg1042_1, arg1043_1, arg1044_1, arg1045_1, arg1046_1, arg1047_1, arg1048_1, arg1049_1, arg1050_1, arg1051_1, arg1052_1, arg1053_1, arg1054_1, arg1055_1, arg1056_1, arg1057_1, arg1058_1, arg1059_1, arg1060_1, arg1061_1, arg1062_1, arg1063_1, arg1064_1, arg1065_1, arg1066_1, arg1067_1, arg1068_1, arg1069_1, arg1070_1, arg1071_1, arg1072_1, arg1073_1, arg1074_1, arg1075_1, arg1076_1, arg1077_1, arg1078_1, arg1079_1, arg1080_1, arg1081_1, arg1082_1, arg1083_1, arg1084_1, arg1085_1, arg1086_1, arg1087_1, arg1088_1, arg1089_1, arg1090_1, arg1091_1, arg1092_1, arg1093_1, arg1094_1, arg1095_1, arg1096_1, arg1097_1, arg1098_1, arg1099_1, arg1100_1, arg1101_1, arg1102_1, arg1103_1, arg1104_1, arg1105_1, arg1106_1, arg1107_1, arg1108_1, arg1109_1, arg1110_1, arg1111_1, arg1112_1, arg1113_1, arg1114_1, arg1115_1, arg1116_1, arg1117_1, arg1118_1, arg1119_1, arg1120_1, arg1121_1, arg1122_1, arg1123_1, arg1124_1, arg1125_1, arg1126_1, arg1127_1, arg1128_1, arg1129_1, arg1130_1, arg1131_1, arg1132_1, arg1133_1, arg1134_1, arg1135_1, arg1136_1, arg1137_1, arg1138_1, arg1139_1, arg1140_1, arg1141_1, arg1142_1, arg1143_1, arg1144_1, arg1145_1, arg1146_1, arg1147_1, arg1148_1, arg1149_1, arg1150_1, arg1151_1, arg1152_1, arg1153_1, arg1154_1, arg1155_1, arg1156_1, arg1157_1, arg1158_1, arg1159_1, arg1160_1, arg1161_1, arg1162_1, arg1163_1, arg1164_1, arg1165_1, arg1166_1, arg1167_1, arg1168_1, arg1169_1, arg1170_1, arg1171_1, arg1172_1, arg1173_1, arg1174_1, arg1175_1, arg1176_1, arg1177_1, arg1178_1, arg1179_1, arg1180_1, arg1181_1, arg1182_1, arg1183_1, arg1184_1, arg1185_1, arg1186_1, arg1187_1, arg1188_1, arg1189_1, arg1190_1, arg1191_1, arg1192_1, arg1193_1, arg1194_1, arg1195_1, arg1196_1, arg1197_1, arg1198_1, arg1199_1, arg1200_1, arg1201_1, arg1202_1, arg1203_1, arg1204_1, arg1205_1, arg1206_1, arg1207_1, arg1208_1, arg1209_1, arg1210_1, arg1211_1, arg1212_1, arg1213_1, arg1214_1, arg1215_1, arg1216_1, arg1217_1, arg1218_1, arg1219_1, arg1220_1, arg1221_1, arg1222_1, arg1223_1, arg1224_1, arg1225_1, arg1226_1, arg1227_1, arg1228_1, arg1229_1, arg1230_1, arg1231_1, arg1232_1, arg1233_1, arg1234_1, arg1235_1, arg1236_1, arg1237_1, arg1238_1, arg1239_1, arg1240_1, arg1241_1, arg1242_1, arg1243_1, arg1244_1, arg1245_1, arg1246_1, arg1247_1, arg1248_1, arg1249_1, arg1250_1, arg1251_1, arg1252_1, arg1253_1, arg1254_1, arg1255_1, arg1256_1, arg1257_1, arg1258_1, arg1259_1, arg1260_1, arg1261_1, arg1262_1, arg1263_1, arg1264_1, arg1265_1, arg1266_1, arg1267_1, arg1268_1, arg1269_1, arg1270_1, arg1271_1, arg1272_1, arg1273_1, arg1274_1, arg1275_1, arg1276_1, arg1277_1, arg1278_1, arg1279_1, arg1280_1, arg1281_1, arg1282_1, arg1283_1, arg1284_1, arg1285_1, arg1286_1, arg1287_1, arg1288_1, arg1289_1, arg1290_1, arg1291_1, arg1292_1, arg1293_1, arg1294_1, arg1295_1, arg1296_1, arg1297_1, arg1298_1, arg1299_1, arg1300_1, arg1301_1, arg1302_1, arg1303_1, arg1304_1, arg1305_1, arg1306_1, arg1307_1, arg1308_1, arg1309_1, arg1310_1, arg1311_1, arg1312_1, arg1313_1, arg1314_1, arg1315_1, arg1316_1, arg1317_1, arg1318_1, arg1319_1, arg1320_1, arg1321_1, arg1322_1, arg1323_1, arg1324_1, arg1325_1, arg1326_1, arg1327_1, arg1328_1, arg1329_1, arg1330_1, arg1331_1, arg1332_1, arg1333_1, arg1334_1, arg1335_1, arg1336_1, arg1337_1, arg1338_1, arg1339_1, arg1340_1, arg1341_1, arg1342_1, arg1343_1, arg1344_1, arg1345_1, arg1346_1, arg1347_1, arg1348_1, arg1349_1, arg1350_1, arg1351_1, arg1352_1, arg1353_1, arg1354_1, arg1355_1, arg1356_1, arg1357_1, arg1358_1, arg1359_1, arg1360_1, arg1361_1, arg1362_1, arg1363_1, arg1364_1, arg1365_1, arg1366_1, arg1367_1, arg1368_1, arg1369_1, arg1370_1, arg1371_1, arg1372_1, arg1373_1, arg1374_1, arg1375_1, arg1376_1, arg1377_1, arg1378_1, arg1379_1, arg1380_1, arg1381_1, arg1382_1, arg1383_1, arg1384_1, arg1385_1, arg1386_1, arg1387_1, arg1388_1, arg1389_1, arg1390_1, arg1391_1, arg1392_1, arg1393_1, arg1394_1, arg1395_1, arg1396_1, arg1397_1, arg1398_1, arg1399_1, arg1400_1, arg1401_1, arg1402_1, arg1403_1, arg1404_1, arg1405_1, arg1406_1, arg1407_1, arg1408_1, arg1409_1, arg1410_1, arg1411_1, arg1412_1, arg1413_1, arg1414_1, arg1415_1, arg1416_1, arg1417_1, arg1418_1, arg1419_1, arg1420_1, arg1421_1, arg1422_1, arg1423_1, arg1424_1, arg1425_1, arg1426_1, arg1427_1, arg1428_1, arg1429_1, arg1430_1, arg1431_1, arg1432_1, arg1433_1, arg1434_1, arg1435_1, arg1436_1, arg1437_1, arg1438_1, arg1439_1, arg1440_1, arg1441_1, arg1442_1, arg1443_1, arg1444_1, arg1445_1, arg1446_1, arg1447_1, arg1448_1, arg1449_1, arg1450_1, arg1451_1, arg1452_1, arg1453_1, arg1454_1, arg1455_1, arg1456_1, arg1457_1, arg1458_1, arg1459_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
