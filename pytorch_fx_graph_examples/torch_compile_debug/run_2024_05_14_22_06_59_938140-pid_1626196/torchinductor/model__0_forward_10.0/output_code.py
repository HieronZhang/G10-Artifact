
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


# kernel path: /tmp/torchinductor_zhang402/a7/ca7ihxvfiqamlxbf5iadmiuxc6e5ld7qxekibqz7n3kdwx6jtsk7.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[128, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 96
    xnumel = 9
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = (yindex // 3)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (27*y1)), tmp0, xmask & ymask)
''', device_str='cuda')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, split_scan_grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream


# kernel path: /tmp/torchinductor_zhang402/b6/cb6nk5gg6ogae7dueu3asmsyooiehxu6uqsygcx25ou5hhlnkcwd.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[1024, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 9
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 32
    y1 = (yindex // 32)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (32*x2) + (288*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/pm/cpmqadc7antgkez3xzz54urgwc4cwd4l442dnkdeacfnwt7vf6a7.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[2048, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 9
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 32
    y1 = (yindex // 32)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (32*x2) + (288*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/mb/cmbhpu5ilvkip2pqhojnkivjsuir7ahinlpud2yqdmlpijqqocbv.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[16384, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 15360
    xnumel = 9
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 80
    y1 = (yindex // 80)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (80*x2) + (720*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/l7/cl7dy66ibkzzcxjv75uys35cmah24tjmnkkbv2luh3k5ei6ikq6x.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[4096, 32], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_4', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 25
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 48
    y1 = (yindex // 48)
    tmp0 = tl.load(in_ptr0 + (x2 + (25*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (48*x2) + (1200*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/fb/cfbvmlmijtoad5f3ltjpctxwpdbcctacwchf2cuo25ddowvf62t3.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[8192, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_5', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (576*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/57/c575trqfek5i5zvgoj2p2exrhjxqv54iu6usiyb464ccaaoi6llm.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[16384, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_6', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9216
    xnumel = 9
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 96
    y1 = (yindex // 96)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (96*x2) + (864*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/lg/clgkkg3clewmcdx36synk5zvdkmqiictuxzttlbzglk5cvmyjs64.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[131072, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_7', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 110592
    xnumel = 9
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 288
    y1 = (yindex // 288)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (288*x2) + (2592*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/ji/cjipzoteldjdneci5u36mnpzkwzdpprrpycg2chamvvst3qtptmu.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[16384, 8], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_8', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 7
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    tmp0 = tl.load(in_ptr0 + (x2 + (7*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (128*x2) + (896*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/bg/cbg6gvaxwywhsmfmygyz7zz5icpgszrbbbfmg65y2n5lechipbia.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[32768, 8], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_9', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24576
    xnumel = 7
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    tmp0 = tl.load(in_ptr0 + (x2 + (7*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (128*x2) + (896*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/rv/crvsifdod4o4zndy32s77cg7xtk2euzjouddpjauzi24gf6id5oy.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[32768, 8], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_10', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25600
    xnumel = 7
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 160
    y1 = (yindex // 160)
    tmp0 = tl.load(in_ptr0 + (x2 + (7*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (160*x2) + (1120*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/oe/coeuvcit2y5c3kb5ew4xmdai7ut34umx36z43llruy6gj3uhrzt5.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[32768, 8], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_11', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 30720
    xnumel = 7
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 160
    y1 = (yindex // 160)
    tmp0 = tl.load(in_ptr0 + (x2 + (7*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (160*x2) + (1120*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/zs/czsguqestkugqzba4p7vjusyn24vqryzjjdgvt3d2ny4563pnisu.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[65536, 8], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_12', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 36864
    xnumel = 7
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 192
    y1 = (yindex // 192)
    tmp0 = tl.load(in_ptr0 + (x2 + (7*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (192*x2) + (1344*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/ch/cchkptzmbygzrpg3q37rq6dxstbd7r5j6otquqlkk643qy3kx7cw.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[131072, 32], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_13', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 98304
    xnumel = 25
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    tmp0 = tl.load(in_ptr0 + (x2 + (25*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (128*x2) + (3200*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/7r/c7rozi2woe737zfhlcjg3lsdz63iz62cfjdqeeu3n67wlyok7rdi.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[65536, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_14', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 61440
    xnumel = 9
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 192
    y1 = (yindex // 192)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (192*x2) + (1728*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/ht/chtzaq4iwcddqz5a2sh4hjrh2f6wsduiy2ovym3nfu3hsx2suld3.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[65536, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_15', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 36864
    xnumel = 9
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 192
    y1 = (yindex // 192)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (192*x2) + (1728*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/wb/cwbffyzxjk6ec5kufhlkypby4ff5gewnrlezlmoa6745tvbzigrk.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[262144, 4], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_16', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 147456
    xnumel = 3
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 384
    y1 = (yindex // 384)
    tmp0 = tl.load(in_ptr0 + (x2 + (3*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (384*x2) + (1152*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/jq/cjquws63iuawcaqixhgmd3b34duhohmkvk42cxmkjeo7tttivk3f.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[262144, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_17', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 172032
    xnumel = 9
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 448
    y1 = (yindex // 448)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (448*x2) + (4032*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/tw/ctwxvtfpqpaa6usb5tvp2u7zk2nedwezbjpatpx2atjqw6zwhkmt.py
# Source Nodes: [cat_31], Original ATen: [aten.cat]
# cat_31 => cat
triton_poi_fused_cat_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_18', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8582496
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 3
    x1 = (xindex // 3) % 89401
    x2 = (xindex // 268203)
    x3 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x1 + (268203*x2)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = 0.458
    tmp7 = tmp5 * tmp6
    tmp8 = -0.030000000000000027
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 2, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tmp12 & tmp14
    tmp16 = tl.load(in_ptr0 + (89401 + x1 + (268203*x2)), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = 0.448
    tmp18 = tmp16 * tmp17
    tmp19 = -0.08799999999999997
    tmp20 = tmp18 + tmp19
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp15, tmp20, tmp21)
    tmp23 = tmp0 >= tmp13
    tmp24 = tl.full([1], 3, tl.int64)
    tmp25 = tmp0 < tmp24
    tmp26 = tl.load(in_ptr0 + (178802 + x1 + (268203*x2)), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp27 = 0.45
    tmp28 = tmp26 * tmp27
    tmp29 = -0.18799999999999994
    tmp30 = tmp28 + tmp29
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp23, tmp30, tmp31)
    tmp33 = tl.where(tmp15, tmp22, tmp32)
    tmp34 = tl.where(tmp4, tmp11, tmp33)
    tl.store(out_ptr0 + (x3), tmp34, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/6c/c6c3ibbrrjstci3jqsbrqqlitbk7amntulonkv5ro5dhbvf3rts3.py
# Source Nodes: [x_2], Original ATen: [aten._native_batch_norm_legit_functional]
# x_2 => var_mean
triton_red_fused__native_batch_norm_legit_functional_19 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_19', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 27616
    rnumel = 824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 32)
    x0 = xindex % 32
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (824*x1)
        tmp1 = tl.full([1, 1], 710432, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (32*((r2 + (824*x1)) % 710432))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/v7/cv76qepdxwva637lbn2bbdl57vsedulyedlqgjjejnvaqn4yqwbe.py
# Source Nodes: [x_2], Original ATen: [aten._native_batch_norm_legit_functional]
# x_2 => var_mean
triton_red_fused__native_batch_norm_legit_functional_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_20', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 224
    rnumel = 124
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 7
    x1 = (xindex // 7)
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (124*x0)
        tmp1 = tl.full([1, 1], 863, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (32*r2) + (3968*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.load(in_ptr1 + (x1 + (32*r2) + (3968*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = tl.load(in_ptr2 + (x1 + (32*r2) + (3968*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x1 + (32*x0)), tmp15, xmask)
    tl.store(out_ptr1 + (x1 + (32*x0)), tmp16, xmask)
    tl.store(out_ptr2 + (x1 + (32*x0)), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/vf/cvf52nj4rkcatvdeqeax5kmquk5k4kew5gntsxukxqdny7ta4qpq.py
# Source Nodes: [x_2], Original ATen: [aten._native_batch_norm_legit_functional]
# x_2 => add_4, add_5, add_6, mul_4, mul_5, mul_6, mul_7, mul_8, rsqrt, var_mean
triton_per_fused__native_batch_norm_legit_functional_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[32, 8],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_21', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
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
    tmp0 = tl.load(in_ptr0 + (x0 + (32*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (32*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (32*r1)), rmask & xmask, other=0.0)
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
    tmp16 = 710432.0
    tmp17 = tmp14 / tmp16
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000014075962338
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/3e/c3e5rcggvaamapwdl2pge2bt4twsruyovq7v7frx6xuh56qtn7gf.py
# Source Nodes: [x_2, x_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_2 => add_4, add_7, mul_3, mul_9, rsqrt, sub, var_mean
# x_3 => relu
triton_poi_fused__native_batch_norm_legit_functional_relu_22 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_22', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 22733824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 710432.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/4v/c4vddmbeayotgnbzlspgmxo344ox7g5sxrc4v7l434an7hupuv4d.py
# Source Nodes: [x_5], Original ATen: [aten._native_batch_norm_legit_functional]
# x_5 => var_mean_1
triton_red_fused__native_batch_norm_legit_functional_23 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_23', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 28224
    rnumel = 784
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (32*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
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
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/ez/ceza7n3wh5txkmzaz7idjjvyvc7hgdai7igyana544hwwrc5qcvi.py
# Source Nodes: [x_5], Original ATen: [aten._native_batch_norm_legit_functional]
# x_5 => var_mean_1
triton_red_fused__native_batch_norm_legit_functional_24 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_24', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 224
    rnumel = 126
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
        tmp0 = tl.load(in_ptr0 + (x1 + (32*r2) + (4032*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (32*r2) + (4032*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (32*r2) + (4032*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x1 + (32*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (32*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (32*x0)), tmp8, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/ix/cixavpwzeloykujbzuoqdbm52ipvjbqw62mjhx6fpsixakerug4p.py
# Source Nodes: [x_5], Original ATen: [aten._native_batch_norm_legit_functional]
# x_5 => add_10, add_11, add_9, mul_11, mul_12, mul_13, mul_14, mul_15, rsqrt_1, var_mean_1
triton_per_fused__native_batch_norm_legit_functional_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[32, 8],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_25', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
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
    tmp0 = tl.load(in_ptr0 + (x0 + (32*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (32*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (32*r1)), rmask & xmask, other=0.0)
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
    tmp16 = 691488.0
    tmp17 = tmp14 / tmp16
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000014461587854
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/jk/cjkot7ngoubkayisp3aydton36qms6xlktknnb3djndn6gnzfjdk.py
# Source Nodes: [x_5, x_6], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_5 => add_12, add_9, mul_10, mul_16, rsqrt_1, sub_1, var_mean_1
# x_6 => relu_1
triton_poi_fused__native_batch_norm_legit_functional_relu_26 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_26', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 22127616
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 691488.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/4p/c4pzkwyemzjcomwythzy5uccthtdqz2lreju4reffsieuj25vzc3.py
# Source Nodes: [x_8], Original ATen: [aten._native_batch_norm_legit_functional]
# x_8 => var_mean_2
triton_red_fused__native_batch_norm_legit_functional_27 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_27', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 56448
    rnumel = 784
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
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (50176*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
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
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/pn/cpnawkt7cdyvdhhzkfgh6vuuwnj7qnisty5vb7jb22edscpduxdj.py
# Source Nodes: [x_8], Original ATen: [aten._native_batch_norm_legit_functional]
# x_8 => var_mean_2
triton_red_fused__native_batch_norm_legit_functional_28 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_28', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 448
    rnumel = 126
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
        tmp0 = tl.load(in_ptr0 + (x1 + (64*r2) + (8064*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (64*r2) + (8064*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (64*r2) + (8064*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/a6/ca6e7sz6nes2g4wmxoxd5gtpvyikxxst5hwdaax25c4yvyrbzxl2.py
# Source Nodes: [x_8], Original ATen: [aten._native_batch_norm_legit_functional]
# x_8 => add_14, add_15, add_16, mul_18, mul_19, mul_20, mul_21, mul_22, rsqrt_2, var_mean_2
triton_per_fused__native_batch_norm_legit_functional_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[64, 8],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_29', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
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
    roffset = 0
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
    tmp16 = 691488.0
    tmp17 = tmp14 / tmp16
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000014461587854
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/r2/cr2fu4pf7wphucohz5jujwbf7ffxkc5asimgd4dwekdlpg3nzmpy.py
# Source Nodes: [x_8, x_9], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_8 => add_14, add_17, mul_17, mul_23, rsqrt_2, sub_2, var_mean_2
# x_9 => relu_2
triton_poi_fused__native_batch_norm_legit_functional_relu_30 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_30', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 44255232
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
    tmp4 = 691488.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/7w/c7w7sqgoymofxjmgd4alovc4je366yibgobro76pa3tfnyvf373d.py
# Source Nodes: [x_10], Original ATen: [aten.max_pool2d_with_indices]
# x_10 => getitem_6, getitem_7
triton_poi_fused_max_pool2d_with_indices_31 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_31', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10913792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 73
    x2 = (xindex // 4672) % 73
    x3 = (xindex // 341056)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*x1) + (18816*x2) + (1382976*x3)), None)
    tmp1 = tl.load(in_ptr0 + (64 + x0 + (128*x1) + (18816*x2) + (1382976*x3)), None)
    tmp3 = tl.load(in_ptr0 + (128 + x0 + (128*x1) + (18816*x2) + (1382976*x3)), None)
    tmp5 = tl.load(in_ptr0 + (9408 + x0 + (128*x1) + (18816*x2) + (1382976*x3)), None)
    tmp7 = tl.load(in_ptr0 + (9472 + x0 + (128*x1) + (18816*x2) + (1382976*x3)), None)
    tmp9 = tl.load(in_ptr0 + (9536 + x0 + (128*x1) + (18816*x2) + (1382976*x3)), None)
    tmp11 = tl.load(in_ptr0 + (18816 + x0 + (128*x1) + (18816*x2) + (1382976*x3)), None)
    tmp13 = tl.load(in_ptr0 + (18880 + x0 + (128*x1) + (18816*x2) + (1382976*x3)), None)
    tmp15 = tl.load(in_ptr0 + (18944 + x0 + (128*x1) + (18816*x2) + (1382976*x3)), None)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tmp17 = tmp1 > tmp0
    tmp18 = 1 + (2*x1) + (294*x2)
    tmp19 = (2*x1) + (294*x2)
    tmp20 = tl.where(tmp17, tmp18, tmp19)
    tmp21 = tmp3 > tmp2
    tmp22 = 2 + (2*x1) + (294*x2)
    tmp23 = tl.where(tmp21, tmp22, tmp20)
    tmp24 = tmp5 > tmp4
    tmp25 = 147 + (2*x1) + (294*x2)
    tmp26 = tl.where(tmp24, tmp25, tmp23)
    tmp27 = tmp7 > tmp6
    tmp28 = 148 + (2*x1) + (294*x2)
    tmp29 = tl.where(tmp27, tmp28, tmp26)
    tmp30 = tmp9 > tmp8
    tmp31 = 149 + (2*x1) + (294*x2)
    tmp32 = tl.where(tmp30, tmp31, tmp29)
    tmp33 = tmp11 > tmp10
    tmp34 = 294 + (2*x1) + (294*x2)
    tmp35 = tl.where(tmp33, tmp34, tmp32)
    tmp36 = tmp13 > tmp12
    tmp37 = 295 + (2*x1) + (294*x2)
    tmp38 = tl.where(tmp36, tmp37, tmp35)
    tmp39 = tmp15 > tmp14
    tmp40 = 296 + (2*x1) + (294*x2)
    tmp41 = tl.where(tmp39, tmp40, tmp38)
    tl.store(out_ptr0 + (x4), tmp16, None)
    tl.store(out_ptr1 + (x4), tmp41, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/dm/cdmwbkf4c2xilbwzlt23t6cfxzyxkh5ygfifd6agufrhyzadh72d.py
# Source Nodes: [x_12], Original ATen: [aten._native_batch_norm_legit_functional]
# x_12 => var_mean_3
triton_red_fused__native_batch_norm_legit_functional_32 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_32', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 106640
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 80)
    x0 = xindex % 80
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x1)
        tmp1 = tl.full([1, 1], 170528, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (80*((r2 + (128*x1)) % 170528))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/wm/cwmvsdei76gr5brpxw7eq6q6igxpkkyn7exqidc7awiksjaxkpha.py
# Source Nodes: [x_12], Original ATen: [aten._native_batch_norm_legit_functional]
# x_12 => var_mean_3
triton_red_fused__native_batch_norm_legit_functional_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_33', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 880
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 11
    x1 = (xindex // 11)
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (122*x0)
        tmp1 = tl.full([1, 1], 1333, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (80*r2) + (9760*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.load(in_ptr1 + (x1 + (80*r2) + (9760*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = tl.load(in_ptr2 + (x1 + (80*r2) + (9760*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x1 + (80*x0)), tmp15, xmask)
    tl.store(out_ptr1 + (x1 + (80*x0)), tmp16, xmask)
    tl.store(out_ptr2 + (x1 + (80*x0)), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/yz/cyztfpv42w3jemzs2kwj5glfahlrtxjj33vy6iuhzeevnj7hx4rw.py
# Source Nodes: [x_12], Original ATen: [aten._native_batch_norm_legit_functional]
# x_12 => add_19, add_20, add_21, mul_25, mul_26, mul_27, mul_28, mul_29, rsqrt_3, var_mean_3
triton_per_fused__native_batch_norm_legit_functional_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[128, 16],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_34', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 80
    rnumel = 11
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (80*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (80*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (80*r1)), rmask & xmask, other=0.0)
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
    tmp16 = 170528.0
    tmp17 = tmp14 / tmp16
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000058641740017
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/e6/ce6hxzcbf7rlu5lefcsytezymawnufwwf72kj3mbx66okestpi6m.py
# Source Nodes: [x_12, x_13], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_12 => add_19, add_22, mul_24, mul_30, rsqrt_3, sub_3, var_mean_3
# x_13 => relu_3
triton_poi_fused__native_batch_norm_legit_functional_relu_35 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_35', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 13642240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 80
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 170528.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/qv/cqvcoq4w5ndjqpciimmxzpxn5u5oplxbvaclpawahhj77d4sx45b.py
# Source Nodes: [x_15], Original ATen: [aten._native_batch_norm_legit_functional]
# x_15 => var_mean_4
triton_red_fused__native_batch_norm_legit_functional_36 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_36', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 82560
    rnumel = 376
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 192)
    x0 = xindex % 192
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (376*x1)
        tmp1 = tl.full([1, 1], 161312, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (192*((r2 + (376*x1)) % 161312))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/co/cco7ylkhrjkejiwihoajn6f2p22djffasvdeke7cgwyqta4xhrtl.py
# Source Nodes: [x_15], Original ATen: [aten._native_batch_norm_legit_functional]
# x_15 => var_mean_4
triton_red_fused__native_batch_norm_legit_functional_37 = async_compile.triton('triton_', '''
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
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_37', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 108
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 4
    x1 = (xindex // 4)
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (108*x0)
        tmp1 = tl.full([1, 1], 430, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (192*r2) + (20736*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.load(in_ptr1 + (x1 + (192*r2) + (20736*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = tl.load(in_ptr2 + (x1 + (192*r2) + (20736*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x1 + (192*x0)), tmp15, xmask)
    tl.store(out_ptr1 + (x1 + (192*x0)), tmp16, xmask)
    tl.store(out_ptr2 + (x1 + (192*x0)), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/m2/cm2lo5jioy6lbj4dtjo4f3xd3h4ukdnf4zev36n2mgx5oiikkuiw.py
# Source Nodes: [x_15], Original ATen: [aten._native_batch_norm_legit_functional]
# x_15 => add_24, add_25, add_26, mul_32, mul_33, mul_34, mul_35, mul_36, rsqrt_4, var_mean_4
triton_per_fused__native_batch_norm_legit_functional_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[256, 4],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_38', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (192*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (192*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (192*r1)), rmask & xmask, other=0.0)
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
    tmp16 = 161312.0
    tmp17 = tmp14 / tmp16
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000061992052618
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/po/cpogt27jpt2x724h6p4c57r2v7fn37zztmqhk4xrubqe6gvxa63h.py
# Source Nodes: [x_15, x_16], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_15 => add_24, add_27, mul_31, mul_37, rsqrt_4, sub_4, var_mean_4
# x_16 => relu_4
triton_poi_fused__native_batch_norm_legit_functional_relu_39 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_39', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 30971904
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 192
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 161312.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/yy/cyyeuv6dxzcgykabt2bbzr6dtmahodm7wqra7bs7vdeoxqydrlax.py
# Source Nodes: [x_17], Original ATen: [aten.max_pool2d_with_indices]
# x_17 => getitem_12, getitem_13
triton_poi_fused_max_pool2d_with_indices_40 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_40', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7526400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 192
    x1 = (xindex // 192) % 35
    x2 = (xindex // 6720) % 35
    x3 = (xindex // 235200)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (384*x1) + (27264*x2) + (967872*x3)), None)
    tmp1 = tl.load(in_ptr0 + (192 + x0 + (384*x1) + (27264*x2) + (967872*x3)), None)
    tmp3 = tl.load(in_ptr0 + (384 + x0 + (384*x1) + (27264*x2) + (967872*x3)), None)
    tmp5 = tl.load(in_ptr0 + (13632 + x0 + (384*x1) + (27264*x2) + (967872*x3)), None)
    tmp7 = tl.load(in_ptr0 + (13824 + x0 + (384*x1) + (27264*x2) + (967872*x3)), None)
    tmp9 = tl.load(in_ptr0 + (14016 + x0 + (384*x1) + (27264*x2) + (967872*x3)), None)
    tmp11 = tl.load(in_ptr0 + (27264 + x0 + (384*x1) + (27264*x2) + (967872*x3)), None)
    tmp13 = tl.load(in_ptr0 + (27456 + x0 + (384*x1) + (27264*x2) + (967872*x3)), None)
    tmp15 = tl.load(in_ptr0 + (27648 + x0 + (384*x1) + (27264*x2) + (967872*x3)), None)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tmp17 = tmp1 > tmp0
    tmp18 = 1 + (2*x1) + (142*x2)
    tmp19 = (2*x1) + (142*x2)
    tmp20 = tl.where(tmp17, tmp18, tmp19)
    tmp21 = tmp3 > tmp2
    tmp22 = 2 + (2*x1) + (142*x2)
    tmp23 = tl.where(tmp21, tmp22, tmp20)
    tmp24 = tmp5 > tmp4
    tmp25 = 71 + (2*x1) + (142*x2)
    tmp26 = tl.where(tmp24, tmp25, tmp23)
    tmp27 = tmp7 > tmp6
    tmp28 = 72 + (2*x1) + (142*x2)
    tmp29 = tl.where(tmp27, tmp28, tmp26)
    tmp30 = tmp9 > tmp8
    tmp31 = 73 + (2*x1) + (142*x2)
    tmp32 = tl.where(tmp30, tmp31, tmp29)
    tmp33 = tmp11 > tmp10
    tmp34 = 142 + (2*x1) + (142*x2)
    tmp35 = tl.where(tmp33, tmp34, tmp32)
    tmp36 = tmp13 > tmp12
    tmp37 = 143 + (2*x1) + (142*x2)
    tmp38 = tl.where(tmp36, tmp37, tmp35)
    tmp39 = tmp15 > tmp14
    tmp40 = 144 + (2*x1) + (142*x2)
    tmp41 = tl.where(tmp39, tmp40, tmp38)
    tl.store(out_ptr0 + (x4), tmp16, None)
    tl.store(out_ptr1 + (x4), tmp41, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/4w/c4wip62ogkdh65qvy5yz447dterd6hshf42ytbusupiuv67fk5uf.py
# Source Nodes: [x_19], Original ATen: [aten._native_batch_norm_legit_functional]
# x_19 => var_mean_5
triton_red_fused__native_batch_norm_legit_functional_41 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_41', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 19648
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 64)
    x0 = xindex % 64
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
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
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/4t/c4t4txlamxeoimzsry45zgjtfcikitutpvsvlo4oeqlkyopcab2y.py
# Source Nodes: [x_19], Original ATen: [aten._native_batch_norm_legit_functional]
# x_19 => var_mean_5
triton_red_fused__native_batch_norm_legit_functional_42 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_42', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 103
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 3
    x1 = (xindex // 3)
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (103*x0)
        tmp1 = tl.full([1, 1], 307, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (64*r2) + (6592*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.load(in_ptr1 + (x1 + (64*r2) + (6592*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = tl.load(in_ptr2 + (x1 + (64*r2) + (6592*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x1 + (64*x0)), tmp15, xmask)
    tl.store(out_ptr1 + (x1 + (64*x0)), tmp16, xmask)
    tl.store(out_ptr2 + (x1 + (64*x0)), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/mo/cmo2lz5bf6u4scjp6urs2eegkqoo2hytjy7zgedw3qg63playidx.py
# Source Nodes: [x_19], Original ATen: [aten._native_batch_norm_legit_functional]
# x_19 => add_29, add_30, add_31, mul_39, mul_40, mul_41, mul_42, mul_43, rsqrt_5, var_mean_5
triton_per_fused__native_batch_norm_legit_functional_43 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[64, 4],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_43', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 3
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
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
    tmp16 = 39200.0
    tmp17 = tmp14 / tmp16
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000255108548688
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/to/ctoxhvn6dxd45p664c4mefnaaigosl2e4jrb27nyxwnv5vep6iyy.py
# Source Nodes: [branch1x1, x_19], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# branch1x1 => relu_5
# x_19 => add_29, add_32, mul_38, mul_44, rsqrt_5, sub_5, var_mean_5
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_44 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_44', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 1225
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (y0 + (64*x2) + (78400*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 39200.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x2 + (1225*y0) + (313600*y1)), tmp14, xmask)
    tl.store(out_ptr1 + (y0 + (64*x2) + (78400*y1)), tmp16, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/7k/c7kga7wzbingheivzfxeaiahskkiarr3wdkuj5jmkddimkrjbc55.py
# Source Nodes: [x_21], Original ATen: [aten._native_batch_norm_legit_functional]
# x_21 => var_mean_6
triton_red_fused__native_batch_norm_legit_functional_45 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_45', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 14736
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 48)
    x0 = xindex % 48
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x1)
        tmp1 = tl.full([1, 1], 39200, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (48*((r2 + (128*x1)) % 39200))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/r4/cr4kfk3y52iiykrqtbt5xh4ywqaxiifw5chudp343jgly2csc4b2.py
# Source Nodes: [x_21], Original ATen: [aten._native_batch_norm_legit_functional]
# x_21 => var_mean_6
triton_red_fused__native_batch_norm_legit_functional_46 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_46', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 144
    rnumel = 103
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 3
    x1 = (xindex // 3)
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (103*x0)
        tmp1 = tl.full([1, 1], 307, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (48*r2) + (4944*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.load(in_ptr1 + (x1 + (48*r2) + (4944*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = tl.load(in_ptr2 + (x1 + (48*r2) + (4944*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x1 + (48*x0)), tmp15, xmask)
    tl.store(out_ptr1 + (x1 + (48*x0)), tmp16, xmask)
    tl.store(out_ptr2 + (x1 + (48*x0)), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/ry/crypyowkauzqgxhmafssntbeehbr2puvnx3icozptpatpsbxgt2w.py
# Source Nodes: [x_21], Original ATen: [aten._native_batch_norm_legit_functional]
# x_21 => add_34, add_35, add_36, mul_46, mul_47, mul_48, mul_49, mul_50, rsqrt_6, var_mean_6
triton_per_fused__native_batch_norm_legit_functional_47 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[64, 4],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_47', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 48
    rnumel = 3
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (48*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (48*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (48*r1)), rmask & xmask, other=0.0)
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
    tmp16 = 39200.0
    tmp17 = tmp14 / tmp16
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000255108548688
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/7s/c7srpnhlif6lc3mbacos57obmufgmq4lvwkl7v5tjx63rldgxx6x.py
# Source Nodes: [branch5x5, x_21], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# branch5x5 => relu_6
# x_21 => add_34, add_37, mul_45, mul_51, rsqrt_6, sub_6, var_mean_6
triton_poi_fused__native_batch_norm_legit_functional_relu_48 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_48', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1881600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 48
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 39200.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/2d/c2dgc3z2klk4itrjuku6aoc6mjx6bari4zrkrfplxb4zmfwpbesi.py
# Source Nodes: [branch3x3dbl, x_25], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# branch3x3dbl => relu_8
# x_25 => add_44, add_47, mul_59, mul_65, rsqrt_8, sub_8, var_mean_8
triton_poi_fused__native_batch_norm_legit_functional_relu_49 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_49', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2508800
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
    tmp4 = 39200.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/km/ckmujfs5itj2mmgaah2vcp3wn5sk2cq6ttphxqebi76rwpa2umu5.py
# Source Nodes: [x_27], Original ATen: [aten._native_batch_norm_legit_functional]
# x_27 => var_mean_9
triton_red_fused__native_batch_norm_legit_functional_50 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_50', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 29472
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 96)
    x0 = xindex % 96
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
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
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/uv/cuv3yqkwk2gcsbehbjcuduyvtuom7fexuhr6xgly6exak77jo6pz.py
# Source Nodes: [x_27], Original ATen: [aten._native_batch_norm_legit_functional]
# x_27 => var_mean_9
triton_red_fused__native_batch_norm_legit_functional_51 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_51', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 288
    rnumel = 103
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 3
    x1 = (xindex // 3)
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (103*x0)
        tmp1 = tl.full([1, 1], 307, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (96*r2) + (9888*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.load(in_ptr1 + (x1 + (96*r2) + (9888*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = tl.load(in_ptr2 + (x1 + (96*r2) + (9888*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x1 + (96*x0)), tmp15, xmask)
    tl.store(out_ptr1 + (x1 + (96*x0)), tmp16, xmask)
    tl.store(out_ptr2 + (x1 + (96*x0)), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/ol/coletpfvg5yk5oplhblgltof4au66cespwcrl7od6dmnjovqwngg.py
# Source Nodes: [x_27], Original ATen: [aten._native_batch_norm_legit_functional]
# x_27 => add_49, add_50, add_51, mul_67, mul_68, mul_69, mul_70, mul_71, rsqrt_9, var_mean_9
triton_per_fused__native_batch_norm_legit_functional_52 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[128, 4],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_52', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 3
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (96*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (96*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (96*r1)), rmask & xmask, other=0.0)
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
    tmp16 = 39200.0
    tmp17 = tmp14 / tmp16
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000255108548688
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/dw/cdwyggjqhjdhjera6yuc7iy7p6fcl262emn5uybsei4humbz6vnz.py
# Source Nodes: [branch3x3dbl_1, x_27], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# branch3x3dbl_1 => relu_9
# x_27 => add_49, add_52, mul_66, mul_72, rsqrt_9, sub_9, var_mean_9
triton_poi_fused__native_batch_norm_legit_functional_relu_53 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_53', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3763200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 96
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 39200.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/ik/cikeq42vmabzngu7sbgi5fmugfdgluyq6dy33l24i7ydgqkzkle2.py
# Source Nodes: [branch3x3dbl_2, x_29], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# branch3x3dbl_2 => relu_10
# x_29 => add_54, add_57, mul_73, mul_79, rsqrt_10, sub_10, var_mean_10
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_54 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[4096, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_54', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 1225
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 96
    y1 = (yindex // 96)
    tmp0 = tl.load(in_ptr0 + (y0 + (96*x2) + (117600*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 39200.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x2 + (1225*y0) + (313600*y1)), tmp14, xmask)
    tl.store(out_ptr1 + (y0 + (96*x2) + (117600*y1)), tmp16, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/6k/c6k4bblsa7qrodc6bzlqeygwstfl2xuw7pydp7hbun7as2hejsww.py
# Source Nodes: [branch_pool], Original ATen: [aten.avg_pool2d]
# branch_pool => avg_pool2d
triton_poi_fused_avg_pool2d_55 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_55', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
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
    x3 = (xindex // 35)
    x2 = xindex % 35
    x5 = xindex
    y0 = yindex % 192
    y1 = (yindex // 192)
    tmp0 = (-1) + x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 35, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x2
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-6912) + y0 + (192*x5) + (235200*y1)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = x2
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-6720) + y0 + (192*x5) + (235200*y1)), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + x2
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-6528) + y0 + (192*x5) + (235200*y1)), tmp27 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = x3
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-192) + y0 + (192*x5) + (235200*y1)), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (y0 + (192*x5) + (235200*y1)), tmp41 & xmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (192 + y0 + (192*x5) + (235200*y1)), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + x3
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (6528 + y0 + (192*x5) + (235200*y1)), tmp55 & xmask, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (6720 + y0 + (192*x5) + (235200*y1)), tmp60 & xmask, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (6912 + y0 + (192*x5) + (235200*y1)), tmp65 & xmask, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = tl.full([1, 1], -1, tl.int64)
    tmp71 = tmp0 >= tmp70
    tmp72 = tl.full([1, 1], 36, tl.int64)
    tmp73 = tmp0 < tmp72
    tmp74 = tmp71 & tmp73
    tmp75 = tmp6 >= tmp70
    tmp76 = tmp6 < tmp72
    tmp77 = tmp75 & tmp76
    tmp78 = tmp74 & tmp77
    tmp79 = tl.broadcast_to((-1) + x3, [XBLOCK, YBLOCK])
    tmp80 = tmp79 >= tmp1
    tmp81 = tmp79 < tmp3
    tmp82 = tmp80 & tmp81
    tmp83 = tl.broadcast_to((-1) + x2, [XBLOCK, YBLOCK])
    tmp84 = tmp83 >= tmp1
    tmp85 = tmp83 < tmp3
    tmp86 = tmp84 & tmp85
    tmp87 = tmp82 & tmp86
    tmp88 = tmp87 & tmp78
    tmp89 = 1.0
    tmp90 = tl.full(tmp89.shape, 1.0, tmp89.dtype)
    tmp91 = tl.where(tmp88, tmp89, tmp90)
    tmp92 = tl.full(tmp91.shape, 0.0, tmp91.dtype)
    tmp93 = tl.where(tmp78, tmp91, tmp92)
    tmp94 = tmp14 >= tmp70
    tmp95 = tmp14 < tmp72
    tmp96 = tmp94 & tmp95
    tmp97 = tmp74 & tmp96
    tmp98 = tl.broadcast_to(x2, [XBLOCK, YBLOCK])
    tmp99 = tmp98 >= tmp1
    tmp100 = tmp98 < tmp3
    tmp101 = tmp99 & tmp100
    tmp102 = tmp82 & tmp101
    tmp103 = tmp102 & tmp97
    tmp104 = tl.where(tmp103, tmp89, tmp90)
    tmp105 = tl.full(tmp104.shape, 0.0, tmp104.dtype)
    tmp106 = tl.where(tmp97, tmp104, tmp105)
    tmp107 = tmp106 + tmp93
    tmp108 = tmp23 >= tmp70
    tmp109 = tmp23 < tmp72
    tmp110 = tmp108 & tmp109
    tmp111 = tmp74 & tmp110
    tmp112 = tl.broadcast_to(1 + x2, [XBLOCK, YBLOCK])
    tmp113 = tmp112 >= tmp1
    tmp114 = tmp112 < tmp3
    tmp115 = tmp113 & tmp114
    tmp116 = tmp82 & tmp115
    tmp117 = tmp116 & tmp111
    tmp118 = tl.where(tmp117, tmp89, tmp90)
    tmp119 = tl.full(tmp118.shape, 0.0, tmp118.dtype)
    tmp120 = tl.where(tmp111, tmp118, tmp119)
    tmp121 = tmp120 + tmp107
    tmp122 = tmp32 >= tmp70
    tmp123 = tmp32 < tmp72
    tmp124 = tmp122 & tmp123
    tmp125 = tmp124 & tmp77
    tmp126 = tl.broadcast_to(x3, [XBLOCK, YBLOCK])
    tmp127 = tmp126 >= tmp1
    tmp128 = tmp126 < tmp3
    tmp129 = tmp127 & tmp128
    tmp130 = tmp129 & tmp86
    tmp131 = tmp130 & tmp125
    tmp132 = tl.where(tmp131, tmp89, tmp90)
    tmp133 = tl.full(tmp132.shape, 0.0, tmp132.dtype)
    tmp134 = tl.where(tmp125, tmp132, tmp133)
    tmp135 = tmp134 + tmp121
    tmp136 = tmp124 & tmp96
    tmp137 = tmp129 & tmp101
    tmp138 = tmp137 & tmp136
    tmp139 = tl.where(tmp138, tmp89, tmp90)
    tmp140 = tl.full(tmp139.shape, 0.0, tmp139.dtype)
    tmp141 = tl.where(tmp136, tmp139, tmp140)
    tmp142 = tmp141 + tmp135
    tmp143 = tmp124 & tmp110
    tmp144 = tmp129 & tmp115
    tmp145 = tmp144 & tmp143
    tmp146 = tl.where(tmp145, tmp89, tmp90)
    tmp147 = tl.full(tmp146.shape, 0.0, tmp146.dtype)
    tmp148 = tl.where(tmp143, tmp146, tmp147)
    tmp149 = tmp148 + tmp142
    tmp150 = tmp51 >= tmp70
    tmp151 = tmp51 < tmp72
    tmp152 = tmp150 & tmp151
    tmp153 = tmp152 & tmp77
    tmp154 = tl.broadcast_to(1 + x3, [XBLOCK, YBLOCK])
    tmp155 = tmp154 >= tmp1
    tmp156 = tmp154 < tmp3
    tmp157 = tmp155 & tmp156
    tmp158 = tmp157 & tmp86
    tmp159 = tmp158 & tmp153
    tmp160 = tl.where(tmp159, tmp89, tmp90)
    tmp161 = tl.full(tmp160.shape, 0.0, tmp160.dtype)
    tmp162 = tl.where(tmp153, tmp160, tmp161)
    tmp163 = tmp162 + tmp149
    tmp164 = tmp152 & tmp96
    tmp165 = tmp157 & tmp101
    tmp166 = tmp165 & tmp164
    tmp167 = tl.where(tmp166, tmp89, tmp90)
    tmp168 = tl.full(tmp167.shape, 0.0, tmp167.dtype)
    tmp169 = tl.where(tmp164, tmp167, tmp168)
    tmp170 = tmp169 + tmp163
    tmp171 = tmp152 & tmp110
    tmp172 = tmp157 & tmp115
    tmp173 = tmp172 & tmp171
    tmp174 = tl.where(tmp173, tmp89, tmp90)
    tmp175 = tl.full(tmp174.shape, 0.0, tmp174.dtype)
    tmp176 = tl.where(tmp171, tmp174, tmp175)
    tmp177 = tmp176 + tmp170
    tmp178 = tmp69 / tmp177
    tl.store(out_ptr0 + (y0 + (192*x5) + (235200*y1)), tmp178, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/vo/cvoufx6ng3nbe4fv7366e6j65cq7gycikkopsfe3pvdt5jifnqj4.py
# Source Nodes: [x_31], Original ATen: [aten._native_batch_norm_legit_functional]
# x_31 => var_mean_11
triton_red_fused__native_batch_norm_legit_functional_56 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_56', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9824
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 32)
    x0 = xindex % 32
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
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
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/iv/civz6vf232xzo2nv3ndoxp6q5qmachmv2bey6yu66becbwxoi5yc.py
# Source Nodes: [x_31], Original ATen: [aten._native_batch_norm_legit_functional]
# x_31 => var_mean_11
triton_red_fused__native_batch_norm_legit_functional_57 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_57', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 103
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 3
    x1 = (xindex // 3)
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (103*x0)
        tmp1 = tl.full([1, 1], 307, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (32*r2) + (3296*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.load(in_ptr1 + (x1 + (32*r2) + (3296*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = tl.load(in_ptr2 + (x1 + (32*r2) + (3296*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x1 + (32*x0)), tmp15, xmask)
    tl.store(out_ptr1 + (x1 + (32*x0)), tmp16, xmask)
    tl.store(out_ptr2 + (x1 + (32*x0)), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/d2/cd26z6g6wtmkmnyeewrpyxetnhzdlkrxarfsfvu552vpbb6odzss.py
# Source Nodes: [x_31], Original ATen: [aten._native_batch_norm_legit_functional]
# x_31 => add_59, add_60, add_61, mul_81, mul_82, mul_83, mul_84, mul_85, rsqrt_11, var_mean_11
triton_per_fused__native_batch_norm_legit_functional_58 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[32, 4],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_58', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 3
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (32*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (32*r1)), rmask & xmask, other=0.0)
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
    tmp16 = 39200.0
    tmp17 = tmp14 / tmp16
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000255108548688
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/ps/cps5xplqcqa5t5cbs6hwbqpo723ktcvcekpr3qpfpmi4bklyrriy.py
# Source Nodes: [branch_pool_1, x_31], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# branch_pool_1 => relu_11
# x_31 => add_59, add_62, mul_80, mul_86, rsqrt_11, sub_11, var_mean_11
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_59 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[1024, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_59', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 1225
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 32
    y1 = (yindex // 32)
    tmp0 = tl.load(in_ptr0 + (y0 + (32*x2) + (39200*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 39200.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x2 + (1225*y0) + (313600*y1)), tmp14, xmask)
    tl.store(out_ptr1 + (y0 + (32*x2) + (39200*y1)), tmp16, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/ck/ccku5gucyzpzkq72srpnhaucxcoo2y4s4ymqtcmlnbj2dtfmzjln.py
# Source Nodes: [cat_30], Original ATen: [aten.cat]
# cat_30 => cat_1
triton_poi_fused_cat_60 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_60', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
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
    x2 = xindex
    y3 = yindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    tmp0 = tl.load(in_ptr0 + (x2 + (1225*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (256*x2) + (313600*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/5b/c5bmqfzbbqtsh7pk7v77nx75hpsmbfauy5euzs2wjq6qcc6ahput.py
# Source Nodes: [branch1x1_1, x_34], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# branch1x1_1 => relu_12
# x_34 => add_64, add_67, mul_87, mul_93, rsqrt_12, sub_12, var_mean_12
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 1225
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (y0 + (64*x2) + (78400*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 39200.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x2 + (1225*y0) + (352800*y1)), tmp14, xmask)
    tl.store(out_ptr1 + (y0 + (64*x2) + (78400*y1)), tmp16, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/ky/ckyxc532fhzp6a2bcjers2l24rbvp54fuvrllk6ncrf5tgwywwdc.py
# Source Nodes: [branch3x3dbl_5, x_44], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# branch3x3dbl_5 => relu_17
# x_44 => add_89, add_92, mul_122, mul_128, rsqrt_17, sub_17, var_mean_17
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_62 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[4096, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_62', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 1225
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 96
    y1 = (yindex // 96)
    tmp0 = tl.load(in_ptr0 + (y0 + (96*x2) + (117600*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 39200.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x2 + (1225*y0) + (352800*y1)), tmp14, xmask)
    tl.store(out_ptr1 + (y0 + (96*x2) + (117600*y1)), tmp16, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/k2/ck2yvd3kvto5yg6aefoxkomofjs26rawewanot55i5ib4q247pyg.py
# Source Nodes: [branch_pool_2], Original ATen: [aten.avg_pool2d]
# branch_pool_2 => avg_pool2d_1
triton_poi_fused_avg_pool2d_63 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_63', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10035200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 8960) % 35
    x1 = (xindex // 256) % 35
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 35, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-9216) + x6), tmp10, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = x1
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-8960) + x6), tmp18, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + x1
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-8704) + x6), tmp27, other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = x2
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-256) + x6), tmp36, other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (x6), tmp41, other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (256 + x6), tmp46, other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + x2
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (8704 + x6), tmp55, other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (8960 + x6), tmp60, other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (9216 + x6), tmp65, other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = tl.full([1], -1, tl.int64)
    tmp71 = tmp0 >= tmp70
    tmp72 = tl.full([1], 36, tl.int64)
    tmp73 = tmp0 < tmp72
    tmp74 = tmp71 & tmp73
    tmp75 = tmp6 >= tmp70
    tmp76 = tmp6 < tmp72
    tmp77 = tmp75 & tmp76
    tmp78 = tmp74 & tmp77
    tmp79 = tmp10 & tmp78
    tmp80 = 1.0
    tmp81 = tl.full(tmp80.shape, 1.0, tmp80.dtype)
    tmp82 = tl.where(tmp79, tmp80, tmp81)
    tmp83 = tl.full(tmp82.shape, 0.0, tmp82.dtype)
    tmp84 = tl.where(tmp78, tmp82, tmp83)
    tmp85 = tmp14 >= tmp70
    tmp86 = tmp14 < tmp72
    tmp87 = tmp85 & tmp86
    tmp88 = tmp74 & tmp87
    tmp89 = tmp18 & tmp88
    tmp90 = tl.where(tmp89, tmp80, tmp81)
    tmp91 = tl.full(tmp90.shape, 0.0, tmp90.dtype)
    tmp92 = tl.where(tmp88, tmp90, tmp91)
    tmp93 = tmp92 + tmp84
    tmp94 = tmp23 >= tmp70
    tmp95 = tmp23 < tmp72
    tmp96 = tmp94 & tmp95
    tmp97 = tmp74 & tmp96
    tmp98 = tmp27 & tmp97
    tmp99 = tl.where(tmp98, tmp80, tmp81)
    tmp100 = tl.full(tmp99.shape, 0.0, tmp99.dtype)
    tmp101 = tl.where(tmp97, tmp99, tmp100)
    tmp102 = tmp101 + tmp93
    tmp103 = tmp32 >= tmp70
    tmp104 = tmp32 < tmp72
    tmp105 = tmp103 & tmp104
    tmp106 = tmp105 & tmp77
    tmp107 = tmp36 & tmp106
    tmp108 = tl.where(tmp107, tmp80, tmp81)
    tmp109 = tl.full(tmp108.shape, 0.0, tmp108.dtype)
    tmp110 = tl.where(tmp106, tmp108, tmp109)
    tmp111 = tmp110 + tmp102
    tmp112 = tmp105 & tmp87
    tmp113 = tmp41 & tmp112
    tmp114 = tl.where(tmp113, tmp80, tmp81)
    tmp115 = tl.full(tmp114.shape, 0.0, tmp114.dtype)
    tmp116 = tl.where(tmp112, tmp114, tmp115)
    tmp117 = tmp116 + tmp111
    tmp118 = tmp105 & tmp96
    tmp119 = tmp46 & tmp118
    tmp120 = tl.where(tmp119, tmp80, tmp81)
    tmp121 = tl.full(tmp120.shape, 0.0, tmp120.dtype)
    tmp122 = tl.where(tmp118, tmp120, tmp121)
    tmp123 = tmp122 + tmp117
    tmp124 = tmp51 >= tmp70
    tmp125 = tmp51 < tmp72
    tmp126 = tmp124 & tmp125
    tmp127 = tmp126 & tmp77
    tmp128 = tmp55 & tmp127
    tmp129 = tl.where(tmp128, tmp80, tmp81)
    tmp130 = tl.full(tmp129.shape, 0.0, tmp129.dtype)
    tmp131 = tl.where(tmp127, tmp129, tmp130)
    tmp132 = tmp131 + tmp123
    tmp133 = tmp126 & tmp87
    tmp134 = tmp60 & tmp133
    tmp135 = tl.where(tmp134, tmp80, tmp81)
    tmp136 = tl.full(tmp135.shape, 0.0, tmp135.dtype)
    tmp137 = tl.where(tmp133, tmp135, tmp136)
    tmp138 = tmp137 + tmp132
    tmp139 = tmp126 & tmp96
    tmp140 = tmp65 & tmp139
    tmp141 = tl.where(tmp140, tmp80, tmp81)
    tmp142 = tl.full(tmp141.shape, 0.0, tmp141.dtype)
    tmp143 = tl.where(tmp139, tmp141, tmp142)
    tmp144 = tmp143 + tmp138
    tmp145 = tmp69 / tmp144
    tl.store(out_ptr0 + (x6), tmp145, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/u3/cu3q4qptmxzusrb3mhaow4iskk25pvak3j52gdcv2ks764k6qlqs.py
# Source Nodes: [cat_29], Original ATen: [aten.cat]
# cat_29 => cat_2
triton_poi_fused_cat_64 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_64', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
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
    x2 = xindex
    y3 = yindex
    y0 = yindex % 288
    y1 = (yindex // 288)
    tmp0 = tl.load(in_ptr0 + (x2 + (1225*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (288*x2) + (352800*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/dl/cdllcb4ijht75k6hitj5mil35psk572qevxbe4kogw2m2p3rjlh2.py
# Source Nodes: [branch_pool_4], Original ATen: [aten.avg_pool2d]
# branch_pool_4 => avg_pool2d_2
triton_poi_fused_avg_pool2d_65 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_65', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 11289600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 10080) % 35
    x1 = (xindex // 288) % 35
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 35, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-10368) + x6), tmp10 & xmask, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = x1
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-10080) + x6), tmp18 & xmask, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + x1
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-9792) + x6), tmp27 & xmask, other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = x2
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-288) + x6), tmp36 & xmask, other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (x6), tmp41 & xmask, other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (288 + x6), tmp46 & xmask, other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + x2
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (9792 + x6), tmp55 & xmask, other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (10080 + x6), tmp60 & xmask, other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (10368 + x6), tmp65 & xmask, other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = tl.full([1], -1, tl.int64)
    tmp71 = tmp0 >= tmp70
    tmp72 = tl.full([1], 36, tl.int64)
    tmp73 = tmp0 < tmp72
    tmp74 = tmp71 & tmp73
    tmp75 = tmp6 >= tmp70
    tmp76 = tmp6 < tmp72
    tmp77 = tmp75 & tmp76
    tmp78 = tmp74 & tmp77
    tmp79 = tmp10 & tmp78
    tmp80 = 1.0
    tmp81 = tl.full(tmp80.shape, 1.0, tmp80.dtype)
    tmp82 = tl.where(tmp79, tmp80, tmp81)
    tmp83 = tl.full(tmp82.shape, 0.0, tmp82.dtype)
    tmp84 = tl.where(tmp78, tmp82, tmp83)
    tmp85 = tmp14 >= tmp70
    tmp86 = tmp14 < tmp72
    tmp87 = tmp85 & tmp86
    tmp88 = tmp74 & tmp87
    tmp89 = tmp18 & tmp88
    tmp90 = tl.where(tmp89, tmp80, tmp81)
    tmp91 = tl.full(tmp90.shape, 0.0, tmp90.dtype)
    tmp92 = tl.where(tmp88, tmp90, tmp91)
    tmp93 = tmp92 + tmp84
    tmp94 = tmp23 >= tmp70
    tmp95 = tmp23 < tmp72
    tmp96 = tmp94 & tmp95
    tmp97 = tmp74 & tmp96
    tmp98 = tmp27 & tmp97
    tmp99 = tl.where(tmp98, tmp80, tmp81)
    tmp100 = tl.full(tmp99.shape, 0.0, tmp99.dtype)
    tmp101 = tl.where(tmp97, tmp99, tmp100)
    tmp102 = tmp101 + tmp93
    tmp103 = tmp32 >= tmp70
    tmp104 = tmp32 < tmp72
    tmp105 = tmp103 & tmp104
    tmp106 = tmp105 & tmp77
    tmp107 = tmp36 & tmp106
    tmp108 = tl.where(tmp107, tmp80, tmp81)
    tmp109 = tl.full(tmp108.shape, 0.0, tmp108.dtype)
    tmp110 = tl.where(tmp106, tmp108, tmp109)
    tmp111 = tmp110 + tmp102
    tmp112 = tmp105 & tmp87
    tmp113 = tmp41 & tmp112
    tmp114 = tl.where(tmp113, tmp80, tmp81)
    tmp115 = tl.full(tmp114.shape, 0.0, tmp114.dtype)
    tmp116 = tl.where(tmp112, tmp114, tmp115)
    tmp117 = tmp116 + tmp111
    tmp118 = tmp105 & tmp96
    tmp119 = tmp46 & tmp118
    tmp120 = tl.where(tmp119, tmp80, tmp81)
    tmp121 = tl.full(tmp120.shape, 0.0, tmp120.dtype)
    tmp122 = tl.where(tmp118, tmp120, tmp121)
    tmp123 = tmp122 + tmp117
    tmp124 = tmp51 >= tmp70
    tmp125 = tmp51 < tmp72
    tmp126 = tmp124 & tmp125
    tmp127 = tmp126 & tmp77
    tmp128 = tmp55 & tmp127
    tmp129 = tl.where(tmp128, tmp80, tmp81)
    tmp130 = tl.full(tmp129.shape, 0.0, tmp129.dtype)
    tmp131 = tl.where(tmp127, tmp129, tmp130)
    tmp132 = tmp131 + tmp123
    tmp133 = tmp126 & tmp87
    tmp134 = tmp60 & tmp133
    tmp135 = tl.where(tmp134, tmp80, tmp81)
    tmp136 = tl.full(tmp135.shape, 0.0, tmp135.dtype)
    tmp137 = tl.where(tmp133, tmp135, tmp136)
    tmp138 = tmp137 + tmp132
    tmp139 = tmp126 & tmp96
    tmp140 = tmp65 & tmp139
    tmp141 = tl.where(tmp140, tmp80, tmp81)
    tmp142 = tl.full(tmp141.shape, 0.0, tmp141.dtype)
    tmp143 = tl.where(tmp139, tmp141, tmp142)
    tmp144 = tmp143 + tmp138
    tmp145 = tmp69 / tmp144
    tl.store(out_ptr0 + (x6), tmp145, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/ks/cksyu2ec5uytboblrbhc5qhjxrmq5xyazalvgnc6u7v5nrpq36ub.py
# Source Nodes: [x_64], Original ATen: [aten._native_batch_norm_legit_functional]
# x_64 => var_mean_26
triton_red_fused__native_batch_norm_legit_functional_66 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_66', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 28032
    rnumel = 127
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 384)
    x0 = xindex % 384
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (127*x1)
        tmp1 = tl.full([1, 1], 9248, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (384*((r2 + (127*x1)) % 9248))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/qk/cqkdkfz5tmlqridyifd652r3d5e7ijpwkbclqd2xr4ewzphhm4eg.py
# Source Nodes: [x_64], Original ATen: [aten._native_batch_norm_legit_functional]
# x_64 => add_134, add_135, add_136, mul_186, mul_187, mul_188, mul_189, mul_190, rsqrt_26, var_mean_26
triton_red_fused__native_batch_norm_legit_functional_67 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_67', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 73
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
        tmp0 = tl.load(in_ptr0 + (x0 + (384*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (384*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x0 + (384*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp9 = 9248.0
    tmp10 = tmp7 / tmp9
    tmp11 = 0.001
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.rsqrt(tmp12)
    tmp14 = 0.1
    tmp15 = tmp6 * tmp14
    tmp17 = 0.9
    tmp18 = tmp16 * tmp17
    tmp19 = tmp15 + tmp18
    tmp20 = 1.0001081431815724
    tmp21 = tmp10 * tmp20
    tmp22 = tmp21 * tmp14
    tmp24 = tmp23 * tmp17
    tmp25 = tmp22 + tmp24
    tl.store(out_ptr2 + (x0), tmp13, xmask)
    tl.store(out_ptr4 + (x0), tmp19, xmask)
    tl.store(out_ptr6 + (x0), tmp25, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/6a/c6anqo3pjlrhpnrtrbv4thsev7dzunye2uj4q4bgg6ft7talump2.py
# Source Nodes: [branch3x3, x_64], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# branch3x3 => relu_26
# x_64 => add_134, add_137, mul_185, mul_191, rsqrt_26, sub_26, var_mean_26
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_68 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_68', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12288
    xnumel = 289
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 384
    y1 = (yindex // 384)
    tmp0 = tl.load(in_ptr0 + (y0 + (384*x2) + (110976*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 9248.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x2 + (289*y0) + (221952*y1)), tmp14, xmask)
    tl.store(out_ptr1 + (y0 + (384*x2) + (110976*y1)), tmp16, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/py/cpyepatrqmvrrqqviqeaopt2npay5tk4vc5jjxj4g3a2iy5t3kzr.py
# Source Nodes: [x_70], Original ATen: [aten._native_batch_norm_legit_functional]
# x_70 => var_mean_29
triton_red_fused__native_batch_norm_legit_functional_69 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_69', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7008
    rnumel = 127
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 96)
    x0 = xindex % 96
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (127*x1)
        tmp1 = tl.full([1, 1], 9248, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (96*((r2 + (127*x1)) % 9248))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/7z/c7zpiy32gy243uacey7aijncgnvh3wkayiybub67qhcsijtdq4yk.py
# Source Nodes: [x_70], Original ATen: [aten._native_batch_norm_legit_functional]
# x_70 => add_149, add_150, add_151, mul_207, mul_208, mul_209, mul_210, mul_211, rsqrt_29, var_mean_29
triton_red_fused__native_batch_norm_legit_functional_70 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_70', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 73
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
        tmp0 = tl.load(in_ptr0 + (x0 + (96*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (96*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x0 + (96*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp9 = 9248.0
    tmp10 = tmp7 / tmp9
    tmp11 = 0.001
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.rsqrt(tmp12)
    tmp14 = 0.1
    tmp15 = tmp6 * tmp14
    tmp17 = 0.9
    tmp18 = tmp16 * tmp17
    tmp19 = tmp15 + tmp18
    tmp20 = 1.0001081431815724
    tmp21 = tmp10 * tmp20
    tmp22 = tmp21 * tmp14
    tmp24 = tmp23 * tmp17
    tmp25 = tmp22 + tmp24
    tl.store(out_ptr2 + (x0), tmp13, xmask)
    tl.store(out_ptr4 + (x0), tmp19, xmask)
    tl.store(out_ptr6 + (x0), tmp25, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/np/cnpniwddgfduad7ooxwqdqf7wlbom4ix6z5qojwgpnwp3p52fjpk.py
# Source Nodes: [branch3x3dbl_11, x_70], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# branch3x3dbl_11 => relu_29
# x_70 => add_149, add_152, mul_206, mul_212, rsqrt_29, sub_29, var_mean_29
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_71 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[4096, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_71', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 289
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 96
    y1 = (yindex // 96)
    tmp0 = tl.load(in_ptr0 + (y0 + (96*x2) + (27744*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 9248.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x2 + (289*y0) + (221952*y1)), tmp14, xmask)
    tl.store(out_ptr1 + (y0 + (96*x2) + (27744*y1)), tmp16, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/gq/cgqgpuh5kiy4tetrajfafxnb3na7bdagryacxvby3wyf3msvpwpd.py
# Source Nodes: [branch_pool_6], Original ATen: [aten.max_pool2d_with_indices]
# branch_pool_6 => max_pool2d_with_indices_2
triton_poi_fused_max_pool2d_with_indices_72 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[16384, 512], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_72', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9216
    xnumel = 289
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 17
    x3 = (xindex // 17)
    y0 = yindex % 288
    y1 = (yindex // 288)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (y0 + (576*x2) + (20160*x3) + (352800*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (288 + y0 + (576*x2) + (20160*x3) + (352800*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (576 + y0 + (576*x2) + (20160*x3) + (352800*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (10080 + y0 + (576*x2) + (20160*x3) + (352800*y1)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (10368 + y0 + (576*x2) + (20160*x3) + (352800*y1)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (10656 + y0 + (576*x2) + (20160*x3) + (352800*y1)), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (20160 + y0 + (576*x2) + (20160*x3) + (352800*y1)), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (20448 + y0 + (576*x2) + (20160*x3) + (352800*y1)), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (20736 + y0 + (576*x2) + (20160*x3) + (352800*y1)), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tl.store(out_ptr0 + (x4 + (289*y0) + (221952*y1)), tmp16, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/cl/cclthgnpy4e3dbsme7uk3soyibyb72fjui2onpkl6m33le6sbrit.py
# Source Nodes: [branch_pool_6], Original ATen: [aten.max_pool2d_with_indices]
# branch_pool_6 => getitem_65
triton_poi_fused_max_pool2d_with_indices_73 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_73', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2663424
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 288
    x1 = (xindex // 288) % 17
    x2 = (xindex // 4896) % 17
    x3 = (xindex // 83232)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (576*x1) + (20160*x2) + (352800*x3)), xmask)
    tmp1 = tl.load(in_ptr0 + (288 + x0 + (576*x1) + (20160*x2) + (352800*x3)), xmask)
    tmp7 = tl.load(in_ptr0 + (576 + x0 + (576*x1) + (20160*x2) + (352800*x3)), xmask)
    tmp12 = tl.load(in_ptr0 + (10080 + x0 + (576*x1) + (20160*x2) + (352800*x3)), xmask)
    tmp17 = tl.load(in_ptr0 + (10368 + x0 + (576*x1) + (20160*x2) + (352800*x3)), xmask)
    tmp22 = tl.load(in_ptr0 + (10656 + x0 + (576*x1) + (20160*x2) + (352800*x3)), xmask)
    tmp27 = tl.load(in_ptr0 + (20160 + x0 + (576*x1) + (20160*x2) + (352800*x3)), xmask)
    tmp32 = tl.load(in_ptr0 + (20448 + x0 + (576*x1) + (20160*x2) + (352800*x3)), xmask)
    tmp37 = tl.load(in_ptr0 + (20736 + x0 + (576*x1) + (20160*x2) + (352800*x3)), xmask)
    tmp2 = tmp1 > tmp0
    tmp3 = 1 + (2*x1) + (70*x2)
    tmp4 = (2*x1) + (70*x2)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = triton_helpers.maximum(tmp1, tmp0)
    tmp8 = tmp7 > tmp6
    tmp9 = 2 + (2*x1) + (70*x2)
    tmp10 = tl.where(tmp8, tmp9, tmp5)
    tmp11 = triton_helpers.maximum(tmp7, tmp6)
    tmp13 = tmp12 > tmp11
    tmp14 = 35 + (2*x1) + (70*x2)
    tmp15 = tl.where(tmp13, tmp14, tmp10)
    tmp16 = triton_helpers.maximum(tmp12, tmp11)
    tmp18 = tmp17 > tmp16
    tmp19 = 36 + (2*x1) + (70*x2)
    tmp20 = tl.where(tmp18, tmp19, tmp15)
    tmp21 = triton_helpers.maximum(tmp17, tmp16)
    tmp23 = tmp22 > tmp21
    tmp24 = 37 + (2*x1) + (70*x2)
    tmp25 = tl.where(tmp23, tmp24, tmp20)
    tmp26 = triton_helpers.maximum(tmp22, tmp21)
    tmp28 = tmp27 > tmp26
    tmp29 = 70 + (2*x1) + (70*x2)
    tmp30 = tl.where(tmp28, tmp29, tmp25)
    tmp31 = triton_helpers.maximum(tmp27, tmp26)
    tmp33 = tmp32 > tmp31
    tmp34 = 71 + (2*x1) + (70*x2)
    tmp35 = tl.where(tmp33, tmp34, tmp30)
    tmp36 = triton_helpers.maximum(tmp32, tmp31)
    tmp38 = tmp37 > tmp36
    tmp39 = 72 + (2*x1) + (70*x2)
    tmp40 = tl.where(tmp38, tmp39, tmp35)
    tmp41 = triton_helpers.maximum(tmp37, tmp36)
    tl.store(out_ptr0 + (x4), tmp40, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/xp/cxpno7bdhe5jouhb7hbw5ksh62n6ylw7vpbqqgxr4wnfapvupra4.py
# Source Nodes: [cat_27], Original ATen: [aten.cat]
# cat_27 => cat_4
triton_poi_fused_cat_74 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_74', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
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
    x2 = xindex
    y3 = yindex
    y0 = yindex % 768
    y1 = (yindex // 768)
    tmp0 = tl.load(in_ptr0 + (x2 + (289*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (768*x2) + (221952*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/nx/cnxqiszeysuwmeyl475fhfxtemljfx3xa7lbffaxk6zkbdvassmc.py
# Source Nodes: [x_73], Original ATen: [aten._native_batch_norm_legit_functional]
# x_73 => var_mean_30
triton_red_fused__native_batch_norm_legit_functional_75 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_75', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 14016
    rnumel = 127
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 192)
    x0 = xindex % 192
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
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
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/kt/cktcb2zzxzygzfbdyauwgaw3peahmkx7ayy4olla7vmxnpwvii35.py
# Source Nodes: [x_73], Original ATen: [aten._native_batch_norm_legit_functional]
# x_73 => add_154, add_155, add_156, mul_214, mul_215, mul_216, mul_217, mul_218, rsqrt_30, var_mean_30
triton_red_fused__native_batch_norm_legit_functional_76 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_76', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 73
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
        tmp0 = tl.load(in_ptr0 + (x0 + (192*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (192*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x0 + (192*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp9 = 9248.0
    tmp10 = tmp7 / tmp9
    tmp11 = 0.001
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.rsqrt(tmp12)
    tmp14 = 0.1
    tmp15 = tmp6 * tmp14
    tmp17 = 0.9
    tmp18 = tmp16 * tmp17
    tmp19 = tmp15 + tmp18
    tmp20 = 1.0001081431815724
    tmp21 = tmp10 * tmp20
    tmp22 = tmp21 * tmp14
    tmp24 = tmp23 * tmp17
    tmp25 = tmp22 + tmp24
    tl.store(out_ptr2 + (x0), tmp13, xmask)
    tl.store(out_ptr4 + (x0), tmp19, xmask)
    tl.store(out_ptr6 + (x0), tmp25, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/t2/ct27k5pxxtrxfefw4pywrdtra3thfjcqabwnozmxdvrpk37nbhlt.py
# Source Nodes: [branch1x1_3, x_73], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# branch1x1_3 => relu_30
# x_73 => add_154, add_157, mul_213, mul_219, rsqrt_30, sub_30, var_mean_30
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_77 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[8192, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_77', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 289
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 192
    y1 = (yindex // 192)
    tmp0 = tl.load(in_ptr0 + (y0 + (192*x2) + (55488*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 9248.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x2 + (289*y0) + (221952*y1)), tmp14, xmask)
    tl.store(out_ptr1 + (y0 + (192*x2) + (55488*y1)), tmp16, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/gg/cggbp6rdhswqt3dwnmeabiehtmmz7sh75gxawsy2ir2darfz6bjh.py
# Source Nodes: [x_75], Original ATen: [aten._native_batch_norm_legit_functional]
# x_75 => var_mean_31
triton_red_fused__native_batch_norm_legit_functional_78 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_78', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9344
    rnumel = 127
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 128)
    x0 = xindex % 128
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (127*x1)
        tmp1 = tl.full([1, 1], 9248, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (128*((r2 + (127*x1)) % 9248))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/5v/c5vxphh7b24ftxnbbefs7rpe2zmbdca4hfr5h2tqedcrxi7tnyqo.py
# Source Nodes: [x_75], Original ATen: [aten._native_batch_norm_legit_functional]
# x_75 => add_159, add_160, add_161, mul_221, mul_222, mul_223, mul_224, mul_225, rsqrt_31, var_mean_31
triton_red_fused__native_batch_norm_legit_functional_79 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_79', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 73
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
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (128*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x0 + (128*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp9 = 9248.0
    tmp10 = tmp7 / tmp9
    tmp11 = 0.001
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.rsqrt(tmp12)
    tmp14 = 0.1
    tmp15 = tmp6 * tmp14
    tmp17 = 0.9
    tmp18 = tmp16 * tmp17
    tmp19 = tmp15 + tmp18
    tmp20 = 1.0001081431815724
    tmp21 = tmp10 * tmp20
    tmp22 = tmp21 * tmp14
    tmp24 = tmp23 * tmp17
    tmp25 = tmp22 + tmp24
    tl.store(out_ptr2 + (x0), tmp13, xmask)
    tl.store(out_ptr4 + (x0), tmp19, xmask)
    tl.store(out_ptr6 + (x0), tmp25, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/vs/cvsaemd4fs5kz5fxyfin2qidntsupegwqvpmahgzmqpb4zxwyeyp.py
# Source Nodes: [branch7x7, x_75], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# branch7x7 => relu_31
# x_75 => add_159, add_162, mul_220, mul_226, rsqrt_31, sub_31, var_mean_31
triton_poi_fused__native_batch_norm_legit_functional_relu_80 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_80', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1183744
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
    tmp4 = 9248.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/e6/ce6b7f2xvlmszxnw5zd32qv5l3a6cxqx7wo5b6rb3yf3jsll2mb2.py
# Source Nodes: [branch_pool_7], Original ATen: [aten.avg_pool2d]
# branch_pool_7 => avg_pool2d_3
triton_poi_fused_avg_pool2d_81 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_81', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7102464
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 13056) % 17
    x1 = (xindex // 768) % 17
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 17, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-13824) + x6), tmp10, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = x1
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-13056) + x6), tmp18, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + x1
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-12288) + x6), tmp27, other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = x2
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-768) + x6), tmp36, other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (x6), tmp41, other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (768 + x6), tmp46, other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + x2
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (12288 + x6), tmp55, other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (13056 + x6), tmp60, other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (13824 + x6), tmp65, other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = tl.full([1], -1, tl.int64)
    tmp71 = tmp0 >= tmp70
    tmp72 = tl.full([1], 18, tl.int64)
    tmp73 = tmp0 < tmp72
    tmp74 = tmp71 & tmp73
    tmp75 = tmp6 >= tmp70
    tmp76 = tmp6 < tmp72
    tmp77 = tmp75 & tmp76
    tmp78 = tmp74 & tmp77
    tmp79 = tmp10 & tmp78
    tmp80 = 1.0
    tmp81 = tl.full(tmp80.shape, 1.0, tmp80.dtype)
    tmp82 = tl.where(tmp79, tmp80, tmp81)
    tmp83 = tl.full(tmp82.shape, 0.0, tmp82.dtype)
    tmp84 = tl.where(tmp78, tmp82, tmp83)
    tmp85 = tmp14 >= tmp70
    tmp86 = tmp14 < tmp72
    tmp87 = tmp85 & tmp86
    tmp88 = tmp74 & tmp87
    tmp89 = tmp18 & tmp88
    tmp90 = tl.where(tmp89, tmp80, tmp81)
    tmp91 = tl.full(tmp90.shape, 0.0, tmp90.dtype)
    tmp92 = tl.where(tmp88, tmp90, tmp91)
    tmp93 = tmp92 + tmp84
    tmp94 = tmp23 >= tmp70
    tmp95 = tmp23 < tmp72
    tmp96 = tmp94 & tmp95
    tmp97 = tmp74 & tmp96
    tmp98 = tmp27 & tmp97
    tmp99 = tl.where(tmp98, tmp80, tmp81)
    tmp100 = tl.full(tmp99.shape, 0.0, tmp99.dtype)
    tmp101 = tl.where(tmp97, tmp99, tmp100)
    tmp102 = tmp101 + tmp93
    tmp103 = tmp32 >= tmp70
    tmp104 = tmp32 < tmp72
    tmp105 = tmp103 & tmp104
    tmp106 = tmp105 & tmp77
    tmp107 = tmp36 & tmp106
    tmp108 = tl.where(tmp107, tmp80, tmp81)
    tmp109 = tl.full(tmp108.shape, 0.0, tmp108.dtype)
    tmp110 = tl.where(tmp106, tmp108, tmp109)
    tmp111 = tmp110 + tmp102
    tmp112 = tmp105 & tmp87
    tmp113 = tmp41 & tmp112
    tmp114 = tl.where(tmp113, tmp80, tmp81)
    tmp115 = tl.full(tmp114.shape, 0.0, tmp114.dtype)
    tmp116 = tl.where(tmp112, tmp114, tmp115)
    tmp117 = tmp116 + tmp111
    tmp118 = tmp105 & tmp96
    tmp119 = tmp46 & tmp118
    tmp120 = tl.where(tmp119, tmp80, tmp81)
    tmp121 = tl.full(tmp120.shape, 0.0, tmp120.dtype)
    tmp122 = tl.where(tmp118, tmp120, tmp121)
    tmp123 = tmp122 + tmp117
    tmp124 = tmp51 >= tmp70
    tmp125 = tmp51 < tmp72
    tmp126 = tmp124 & tmp125
    tmp127 = tmp126 & tmp77
    tmp128 = tmp55 & tmp127
    tmp129 = tl.where(tmp128, tmp80, tmp81)
    tmp130 = tl.full(tmp129.shape, 0.0, tmp129.dtype)
    tmp131 = tl.where(tmp127, tmp129, tmp130)
    tmp132 = tmp131 + tmp123
    tmp133 = tmp126 & tmp87
    tmp134 = tmp60 & tmp133
    tmp135 = tl.where(tmp134, tmp80, tmp81)
    tmp136 = tl.full(tmp135.shape, 0.0, tmp135.dtype)
    tmp137 = tl.where(tmp133, tmp135, tmp136)
    tmp138 = tmp137 + tmp132
    tmp139 = tmp126 & tmp96
    tmp140 = tmp65 & tmp139
    tmp141 = tl.where(tmp140, tmp80, tmp81)
    tmp142 = tl.full(tmp141.shape, 0.0, tmp141.dtype)
    tmp143 = tl.where(tmp139, tmp141, tmp142)
    tmp144 = tmp143 + tmp138
    tmp145 = tmp69 / tmp144
    tl.store(out_ptr0 + (x6), tmp145, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/wq/cwqmgzhzbztvupzpfq5ppkyivtz5dvaq7nf3z6cvwnvjna2ote5d.py
# Source Nodes: [x_96], Original ATen: [aten._native_batch_norm_legit_functional]
# x_96 => var_mean_41
triton_red_fused__native_batch_norm_legit_functional_82 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_82', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 11680
    rnumel = 127
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 160)
    x0 = xindex % 160
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (127*x1)
        tmp1 = tl.full([1, 1], 9248, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (160*((r2 + (127*x1)) % 9248))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/ig/cigqxlfmjktdwerojscm7mcajx5i7km4raf7dbjfjkxrt6j2glhq.py
# Source Nodes: [x_96], Original ATen: [aten._native_batch_norm_legit_functional]
# x_96 => add_209, add_210, add_211, mul_291, mul_292, mul_293, mul_294, mul_295, rsqrt_41, var_mean_41
triton_red_fused__native_batch_norm_legit_functional_83 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_83', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 160
    rnumel = 73
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
        tmp0 = tl.load(in_ptr0 + (x0 + (160*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (160*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x0 + (160*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp9 = 9248.0
    tmp10 = tmp7 / tmp9
    tmp11 = 0.001
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.rsqrt(tmp12)
    tmp14 = 0.1
    tmp15 = tmp6 * tmp14
    tmp17 = 0.9
    tmp18 = tmp16 * tmp17
    tmp19 = tmp15 + tmp18
    tmp20 = 1.0001081431815724
    tmp21 = tmp10 * tmp20
    tmp22 = tmp21 * tmp14
    tmp24 = tmp23 * tmp17
    tmp25 = tmp22 + tmp24
    tl.store(out_ptr2 + (x0), tmp13, xmask)
    tl.store(out_ptr4 + (x0), tmp19, xmask)
    tl.store(out_ptr6 + (x0), tmp25, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/k5/ck5t73te5hpif3ng3nhq47ymvol4nsffjuhqk7efonqjwts6ygly.py
# Source Nodes: [branch7x7_3, x_96], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# branch7x7_3 => relu_41
# x_96 => add_209, add_212, mul_290, mul_296, rsqrt_41, sub_41, var_mean_41
triton_poi_fused__native_batch_norm_legit_functional_relu_84 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_84', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1479680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 160
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 9248.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/ar/car6mnrrjfr2uuxi6ks2jnqwklhjkpbdgz3si737j7npw6oscgcx.py
# Source Nodes: [branch7x7_9, x_138], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# branch7x7_9 => relu_61
# x_138 => add_309, add_312, mul_430, mul_436, rsqrt_61, sub_61, var_mean_61
triton_poi_fused__native_batch_norm_legit_functional_relu_85 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_85', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1775616
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 192
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 9248.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/vy/cvybunkjqinyxuav5rynh3wdl27emzfqc6szrdy4o5wzvx2xtifb.py
# Source Nodes: [x_156], Original ATen: [aten.avg_pool2d]
# x_156 => avg_pool2d_7
triton_poi_fused_avg_pool2d_86 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_86', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 614400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 768
    x1 = (xindex // 768) % 5
    x2 = (xindex // 3840) % 5
    x3 = (xindex // 19200)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (2304*x1) + (39168*x2) + (221952*x3)), None)
    tmp1 = tl.load(in_ptr0 + (768 + x0 + (2304*x1) + (39168*x2) + (221952*x3)), None)
    tmp3 = tl.load(in_ptr0 + (1536 + x0 + (2304*x1) + (39168*x2) + (221952*x3)), None)
    tmp5 = tl.load(in_ptr0 + (2304 + x0 + (2304*x1) + (39168*x2) + (221952*x3)), None)
    tmp7 = tl.load(in_ptr0 + (3072 + x0 + (2304*x1) + (39168*x2) + (221952*x3)), None)
    tmp9 = tl.load(in_ptr0 + (13056 + x0 + (2304*x1) + (39168*x2) + (221952*x3)), None)
    tmp11 = tl.load(in_ptr0 + (13824 + x0 + (2304*x1) + (39168*x2) + (221952*x3)), None)
    tmp13 = tl.load(in_ptr0 + (14592 + x0 + (2304*x1) + (39168*x2) + (221952*x3)), None)
    tmp15 = tl.load(in_ptr0 + (15360 + x0 + (2304*x1) + (39168*x2) + (221952*x3)), None)
    tmp17 = tl.load(in_ptr0 + (16128 + x0 + (2304*x1) + (39168*x2) + (221952*x3)), None)
    tmp19 = tl.load(in_ptr0 + (26112 + x0 + (2304*x1) + (39168*x2) + (221952*x3)), None)
    tmp21 = tl.load(in_ptr0 + (26880 + x0 + (2304*x1) + (39168*x2) + (221952*x3)), None)
    tmp23 = tl.load(in_ptr0 + (27648 + x0 + (2304*x1) + (39168*x2) + (221952*x3)), None)
    tmp25 = tl.load(in_ptr0 + (28416 + x0 + (2304*x1) + (39168*x2) + (221952*x3)), None)
    tmp27 = tl.load(in_ptr0 + (29184 + x0 + (2304*x1) + (39168*x2) + (221952*x3)), None)
    tmp29 = tl.load(in_ptr0 + (39168 + x0 + (2304*x1) + (39168*x2) + (221952*x3)), None)
    tmp31 = tl.load(in_ptr0 + (39936 + x0 + (2304*x1) + (39168*x2) + (221952*x3)), None)
    tmp33 = tl.load(in_ptr0 + (40704 + x0 + (2304*x1) + (39168*x2) + (221952*x3)), None)
    tmp35 = tl.load(in_ptr0 + (41472 + x0 + (2304*x1) + (39168*x2) + (221952*x3)), None)
    tmp37 = tl.load(in_ptr0 + (42240 + x0 + (2304*x1) + (39168*x2) + (221952*x3)), None)
    tmp39 = tl.load(in_ptr0 + (52224 + x0 + (2304*x1) + (39168*x2) + (221952*x3)), None)
    tmp41 = tl.load(in_ptr0 + (52992 + x0 + (2304*x1) + (39168*x2) + (221952*x3)), None)
    tmp43 = tl.load(in_ptr0 + (53760 + x0 + (2304*x1) + (39168*x2) + (221952*x3)), None)
    tmp45 = tl.load(in_ptr0 + (54528 + x0 + (2304*x1) + (39168*x2) + (221952*x3)), None)
    tmp47 = tl.load(in_ptr0 + (55296 + x0 + (2304*x1) + (39168*x2) + (221952*x3)), None)
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp16 = tmp15 + tmp14
    tmp18 = tmp17 + tmp16
    tmp20 = tmp19 + tmp18
    tmp22 = tmp21 + tmp20
    tmp24 = tmp23 + tmp22
    tmp26 = tmp25 + tmp24
    tmp28 = tmp27 + tmp26
    tmp30 = tmp29 + tmp28
    tmp32 = tmp31 + tmp30
    tmp34 = tmp33 + tmp32
    tmp36 = tmp35 + tmp34
    tmp38 = tmp37 + tmp36
    tmp40 = tmp39 + tmp38
    tmp42 = tmp41 + tmp40
    tmp44 = tmp43 + tmp42
    tmp46 = tmp45 + tmp44
    tmp48 = tmp47 + tmp46
    tmp49 = 0.04
    tmp50 = tmp48 * tmp49
    tl.store(out_ptr0 + (x4), tmp50, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/dx/cdxchhsreh4l37tx2hz2umtumin6s5hbnfxhogz4yqlcotlfu52w.py
# Source Nodes: [x_158], Original ATen: [aten._native_batch_norm_legit_functional]
# x_158 => var_mean_70
triton_red_fused__native_batch_norm_legit_functional_87 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_87', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 896
    rnumel = 115
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 128)
    x0 = xindex % 128
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (115*x1)
        tmp1 = tl.full([1, 1], 800, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (128*(r2 % 5)) + (640*(((r2 + (115*x1)) // 5) % 160))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/wg/cwgavky4r25x4ssfycssxgjc25pvnvnhpektdc7klmsqjqjfmn57.py
# Source Nodes: [x_158], Original ATen: [aten._native_batch_norm_legit_functional]
# x_158 => add_354, add_355, add_356, mul_494, mul_495, mul_496, mul_497, mul_498, rsqrt_70, var_mean_70
triton_per_fused__native_batch_norm_legit_functional_88 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_88', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
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
    roffset = 0
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
    tmp16 = 800.0
    tmp17 = tmp14 / tmp16
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0012515644555695
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/b5/cb5czvxmmk3pavpf7yj6xqhlae23ehpn7xixbd4g5dhsssmj77ej.py
# Source Nodes: [x_158, x_159], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_158 => add_354, add_357, mul_493, mul_499, rsqrt_70, sub_70, var_mean_70
# x_159 => relu_70
triton_poi_fused__native_batch_norm_legit_functional_relu_89 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_89', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102400
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
    tmp4 = 800.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/nr/cnrk7tlzisxo474af6l6uf3of45zhdx5iwuorfpy2uyx4ep6nhsw.py
# Source Nodes: [x_161], Original ATen: [aten._native_batch_norm_legit_functional]
# x_161 => add_359, add_360, add_361, mul_501, mul_502, mul_503, mul_504, mul_505, rsqrt_71, var_mean_71
triton_per_fused__native_batch_norm_legit_functional_90 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_90', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 32, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = 32.0
    tmp18 = tmp16 / tmp17
    tmp19 = 0.001
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.rsqrt(tmp20)
    tmp22 = 0.1
    tmp23 = tmp10 * tmp22
    tmp25 = 0.9
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 + tmp26
    tmp28 = 1.032258064516129
    tmp29 = tmp18 * tmp28
    tmp30 = tmp29 * tmp22
    tmp32 = tmp31 * tmp25
    tmp33 = tmp30 + tmp32
    tl.store(out_ptr2 + (x0), tmp21, xmask)
    tl.store(out_ptr4 + (x0), tmp27, xmask)
    tl.store(out_ptr6 + (x0), tmp33, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/vb/cvbvthx7wfdp6swxcacvvynji7rvwrgg7vtt5adc3bpvnafatdai.py
# Source Nodes: [x_161, x_162, x_163], Original ATen: [aten._native_batch_norm_legit_functional, aten.mean, aten.relu, aten.threshold_backward]
# x_161 => add_359, add_362, mul_500, mul_506, rsqrt_71, sub_71, var_mean_71
# x_162 => relu_71
# x_163 => mean
triton_poi_fused__native_batch_norm_legit_functional_mean_relu_threshold_backward_91 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_mean_relu_threshold_backward_91', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 32.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = 1.0
    tmp16 = tmp14 / tmp15
    tmp17 = 0.0
    tmp18 = tmp14 <= tmp17
    tl.store(out_ptr1 + (x2), tmp16, None)
    tl.store(out_ptr2 + (x2), tmp18, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/rd/crd7so3ay2zg2gls6hguzrzknk2f2mzbgxmrdzxwrvqiuq6a5nxa.py
# Source Nodes: [x_169], Original ATen: [aten._native_batch_norm_legit_functional]
# x_169 => var_mean_73
triton_red_fused__native_batch_norm_legit_functional_92 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_92', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5120
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 320
    x1 = (xindex // 320)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (320*r2) + (40960*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
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
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/e2/ce24qovos4264fykhsatfpihak6xc7awsyimwyyf5664nffdxjrm.py
# Source Nodes: [x_169], Original ATen: [aten._native_batch_norm_legit_functional]
# x_169 => add_369, add_370, add_371, mul_515, mul_516, mul_517, mul_518, mul_519, rsqrt_73, var_mean_73
triton_per_fused__native_batch_norm_legit_functional_93 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_93', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x0 + (320*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (320*r1)), rmask & xmask, other=0.0)
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
    tmp16 = 2048.0
    tmp17 = tmp14 / tmp16
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0004885197850513
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/2p/c2pnwkjrzburxpztyykfat3qqyfhpyvinywht7xi2v764nbsguys.py
# Source Nodes: [branch3x3_2, x_169], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# branch3x3_2 => relu_73
# x_169 => add_369, add_372, mul_514, mul_520, rsqrt_73, sub_73, var_mean_73
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_94 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[16384, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_94', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 10240
    xnumel = 64
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 320
    y1 = (yindex // 320)
    tmp0 = tl.load(in_ptr0 + (y0 + (320*x2) + (20480*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 2048.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x2 + (64*y0) + (81920*y1)), tmp14, xmask)
    tl.store(out_ptr1 + (y0 + (320*x2) + (20480*y1)), tmp16, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/ib/ciba2khezratxvnbh7x34hbtn2urgq66jeljxkh56vcaxygebff3.py
# Source Nodes: [x_177], Original ATen: [aten._native_batch_norm_legit_functional]
# x_177 => var_mean_77
triton_red_fused__native_batch_norm_legit_functional_95 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_95', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 192
    x1 = (xindex // 192)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (192*r2) + (24576*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
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
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/n5/cn5gelhnmtnsp3db5vedb4mszuvucxrf265ooxjtwpjrpmvbuzba.py
# Source Nodes: [x_177], Original ATen: [aten._native_batch_norm_legit_functional]
# x_177 => add_389, add_390, add_391, mul_543, mul_544, mul_545, mul_546, mul_547, rsqrt_77, var_mean_77
triton_per_fused__native_batch_norm_legit_functional_96 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_96', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x0 + (192*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (192*r1)), rmask & xmask, other=0.0)
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
    tmp16 = 2048.0
    tmp17 = tmp14 / tmp16
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0004885197850513
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/gm/cgmvujkuzs4kivx2hiplururw6xwwgnnmnbw3mnl3vwcdaqb2tik.py
# Source Nodes: [branch7x7x3_3, x_177], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# branch7x7x3_3 => relu_77
# x_177 => add_389, add_392, mul_542, mul_548, rsqrt_77, sub_77, var_mean_77
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_97 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[8192, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_97', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 64
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 192
    y1 = (yindex // 192)
    tmp0 = tl.load(in_ptr0 + (y0 + (192*x2) + (12288*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 2048.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x2 + (64*y0) + (81920*y1)), tmp14, xmask)
    tl.store(out_ptr1 + (y0 + (192*x2) + (12288*y1)), tmp16, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/2x/c2xtqarhi4hvs7jpittujhzp6zjkkhjo4tzcjrtrvd3rixgrjtmd.py
# Source Nodes: [branch_pool_15], Original ATen: [aten.max_pool2d_with_indices]
# branch_pool_15 => max_pool2d_with_indices_3
triton_poi_fused_max_pool2d_with_indices_98 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[32768, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_98', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24576
    xnumel = 64
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 8
    x3 = (xindex // 8)
    y0 = yindex % 768
    y1 = (yindex // 768)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (y0 + (1536*x2) + (26112*x3) + (221952*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (768 + y0 + (1536*x2) + (26112*x3) + (221952*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1536 + y0 + (1536*x2) + (26112*x3) + (221952*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (13056 + y0 + (1536*x2) + (26112*x3) + (221952*y1)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (13824 + y0 + (1536*x2) + (26112*x3) + (221952*y1)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (14592 + y0 + (1536*x2) + (26112*x3) + (221952*y1)), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (26112 + y0 + (1536*x2) + (26112*x3) + (221952*y1)), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (26880 + y0 + (1536*x2) + (26112*x3) + (221952*y1)), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (27648 + y0 + (1536*x2) + (26112*x3) + (221952*y1)), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tl.store(out_ptr0 + (x4 + (64*y0) + (81920*y1)), tmp16, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/sl/csllu5hkypxunzqvapge3u6dkrgh4dhquaujubym4iyxzck2q7rk.py
# Source Nodes: [branch_pool_15], Original ATen: [aten.max_pool2d_with_indices]
# branch_pool_15 => getitem_163
triton_poi_fused_max_pool2d_with_indices_99 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_99', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 768
    x1 = (xindex // 768) % 8
    x2 = (xindex // 6144) % 8
    x3 = (xindex // 49152)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1536*x1) + (26112*x2) + (221952*x3)), None)
    tmp1 = tl.load(in_ptr0 + (768 + x0 + (1536*x1) + (26112*x2) + (221952*x3)), None)
    tmp7 = tl.load(in_ptr0 + (1536 + x0 + (1536*x1) + (26112*x2) + (221952*x3)), None)
    tmp12 = tl.load(in_ptr0 + (13056 + x0 + (1536*x1) + (26112*x2) + (221952*x3)), None)
    tmp17 = tl.load(in_ptr0 + (13824 + x0 + (1536*x1) + (26112*x2) + (221952*x3)), None)
    tmp22 = tl.load(in_ptr0 + (14592 + x0 + (1536*x1) + (26112*x2) + (221952*x3)), None)
    tmp27 = tl.load(in_ptr0 + (26112 + x0 + (1536*x1) + (26112*x2) + (221952*x3)), None)
    tmp32 = tl.load(in_ptr0 + (26880 + x0 + (1536*x1) + (26112*x2) + (221952*x3)), None)
    tmp37 = tl.load(in_ptr0 + (27648 + x0 + (1536*x1) + (26112*x2) + (221952*x3)), None)
    tmp2 = tmp1 > tmp0
    tmp3 = 1 + (2*x1) + (34*x2)
    tmp4 = (2*x1) + (34*x2)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = triton_helpers.maximum(tmp1, tmp0)
    tmp8 = tmp7 > tmp6
    tmp9 = 2 + (2*x1) + (34*x2)
    tmp10 = tl.where(tmp8, tmp9, tmp5)
    tmp11 = triton_helpers.maximum(tmp7, tmp6)
    tmp13 = tmp12 > tmp11
    tmp14 = 17 + (2*x1) + (34*x2)
    tmp15 = tl.where(tmp13, tmp14, tmp10)
    tmp16 = triton_helpers.maximum(tmp12, tmp11)
    tmp18 = tmp17 > tmp16
    tmp19 = 18 + (2*x1) + (34*x2)
    tmp20 = tl.where(tmp18, tmp19, tmp15)
    tmp21 = triton_helpers.maximum(tmp17, tmp16)
    tmp23 = tmp22 > tmp21
    tmp24 = 19 + (2*x1) + (34*x2)
    tmp25 = tl.where(tmp23, tmp24, tmp20)
    tmp26 = triton_helpers.maximum(tmp22, tmp21)
    tmp28 = tmp27 > tmp26
    tmp29 = 34 + (2*x1) + (34*x2)
    tmp30 = tl.where(tmp28, tmp29, tmp25)
    tmp31 = triton_helpers.maximum(tmp27, tmp26)
    tmp33 = tmp32 > tmp31
    tmp34 = 35 + (2*x1) + (34*x2)
    tmp35 = tl.where(tmp33, tmp34, tmp30)
    tmp36 = triton_helpers.maximum(tmp32, tmp31)
    tmp38 = tmp37 > tmp36
    tmp39 = 36 + (2*x1) + (34*x2)
    tmp40 = tl.where(tmp38, tmp39, tmp35)
    tmp41 = triton_helpers.maximum(tmp37, tmp36)
    tl.store(out_ptr0 + (x4), tmp40, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/yl/cyltw2ojibxgtjej3d7mb4b6rg33oy4drlciuvkfdog3k7molhep.py
# Source Nodes: [cat_22], Original ATen: [aten.cat]
# cat_22 => cat_9
triton_poi_fused_cat_100 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_100', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
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
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1280
    y1 = (yindex // 1280)
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (1280*x2) + (81920*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/7w/c7wyxkutpdjjlxxkrd2kacj6lrsd47oqcdviz5ya6la4agkzxmqq.py
# Source Nodes: [branch1x1_7, x_180], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# branch1x1_7 => relu_78
# x_180 => add_394, add_397, mul_549, mul_555, rsqrt_78, sub_78, var_mean_78
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_101 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[16384, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_101', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 10240
    xnumel = 64
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 320
    y1 = (yindex // 320)
    tmp0 = tl.load(in_ptr0 + (y0 + (320*x2) + (20480*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 2048.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x2 + (64*y0) + (131072*y1)), tmp14, xmask)
    tl.store(out_ptr1 + (y0 + (320*x2) + (20480*y1)), tmp16, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/vy/cvyabpx6d7we7oi4nyoe5zop5kzycqwthssbzzk6rcm4auf2why7.py
# Source Nodes: [x_182], Original ATen: [aten._native_batch_norm_legit_functional]
# x_182 => var_mean_79
triton_red_fused__native_batch_norm_legit_functional_102 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_102', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 384
    x1 = (xindex // 384)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (384*r2) + (49152*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
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
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/sd/csd6yngd4lnz6hksftkluljtxwrb5iyl6nnupvb2cd6gi7itoe6o.py
# Source Nodes: [x_182], Original ATen: [aten._native_batch_norm_legit_functional]
# x_182 => add_399, add_400, add_401, mul_557, mul_558, mul_559, mul_560, mul_561, rsqrt_79, var_mean_79
triton_per_fused__native_batch_norm_legit_functional_103 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_103', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x0 + (384*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (384*r1)), rmask & xmask, other=0.0)
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
    tmp16 = 2048.0
    tmp17 = tmp14 / tmp16
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0004885197850513
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/ga/cgagsdkqhrck37fhwbwvook5yjyaggnprn6tz2z5qccxdr3vmu7c.py
# Source Nodes: [branch3x3_3, x_182], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# branch3x3_3 => relu_79
# x_182 => add_399, add_402, mul_556, mul_562, rsqrt_79, sub_79, var_mean_79
triton_poi_fused__native_batch_norm_legit_functional_relu_104 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_104', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 384
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 2048.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/4v/c4vmwx42n5yomfwyc7jjfagtc5p3usap3mbs4g7ztk6crjnfxj5x.py
# Source Nodes: [relu_80, x_184], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# relu_80 => relu_80
# x_184 => add_404, add_407, mul_563, mul_569, rsqrt_80, sub_80, var_mean_80
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_105 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[16384, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_105', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12288
    xnumel = 64
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 384
    y1 = (yindex // 384)
    tmp0 = tl.load(in_ptr0 + (y0 + (384*x2) + (24576*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 2048.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x2 + (64*y0) + (49152*y1)), tmp14, xmask)
    tl.store(out_ptr1 + (y0 + (384*x2) + (24576*y1)), tmp16, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/cn/ccnnb2ohraw6gnvrnlcswkgqop7u44lpnqhkdi4zrvrihbjsfrfm.py
# Source Nodes: [x_188], Original ATen: [aten._native_batch_norm_legit_functional]
# x_188 => var_mean_82
triton_red_fused__native_batch_norm_legit_functional_106 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_106', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7168
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 448
    x1 = (xindex // 448)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (448*r2) + (57344*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
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
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/7o/c7owxwoc4v4rwsufd6tg2sd4fvfg75aluwr35s4pqixcks7qoytd.py
# Source Nodes: [x_188], Original ATen: [aten._native_batch_norm_legit_functional]
# x_188 => add_414, add_415, add_416, mul_578, mul_579, mul_580, mul_581, mul_582, rsqrt_82, var_mean_82
triton_per_fused__native_batch_norm_legit_functional_107 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_107', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x0 + (448*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (448*r1)), rmask & xmask, other=0.0)
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
    tmp16 = 2048.0
    tmp17 = tmp14 / tmp16
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0004885197850513
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/xb/cxbv6aybyhp7hglm3pa6rvonytdiuxqjh7n6xurmag2upudm7oqk.py
# Source Nodes: [branch3x3dbl_12, x_188], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# branch3x3dbl_12 => relu_82
# x_188 => add_414, add_417, mul_577, mul_583, rsqrt_82, sub_82, var_mean_82
triton_poi_fused__native_batch_norm_legit_functional_relu_108 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_108', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 917504
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 448
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 2048.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/cq/ccqpazdzgnvfhbvqz5vpkybukpm2vg65s77sug5pwftnwgrxubrz.py
# Source Nodes: [branch_pool_16], Original ATen: [aten.avg_pool2d]
# branch_pool_16 => avg_pool2d_8
triton_poi_fused_avg_pool2d_109 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_109', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2621440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 10240) % 8
    x1 = (xindex // 1280) % 8
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-11520) + x6), tmp10, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = x1
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-10240) + x6), tmp18, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + x1
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-8960) + x6), tmp27, other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = x2
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-1280) + x6), tmp36, other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (x6), tmp41, other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (1280 + x6), tmp46, other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + x2
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (8960 + x6), tmp55, other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (10240 + x6), tmp60, other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (11520 + x6), tmp65, other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = tl.full([1], -1, tl.int64)
    tmp71 = tmp0 >= tmp70
    tmp72 = tl.full([1], 9, tl.int64)
    tmp73 = tmp0 < tmp72
    tmp74 = tmp71 & tmp73
    tmp75 = tmp6 >= tmp70
    tmp76 = tmp6 < tmp72
    tmp77 = tmp75 & tmp76
    tmp78 = tmp74 & tmp77
    tmp79 = tmp10 & tmp78
    tmp80 = 1.0
    tmp81 = tl.full(tmp80.shape, 1.0, tmp80.dtype)
    tmp82 = tl.where(tmp79, tmp80, tmp81)
    tmp83 = tl.full(tmp82.shape, 0.0, tmp82.dtype)
    tmp84 = tl.where(tmp78, tmp82, tmp83)
    tmp85 = tmp14 >= tmp70
    tmp86 = tmp14 < tmp72
    tmp87 = tmp85 & tmp86
    tmp88 = tmp74 & tmp87
    tmp89 = tmp18 & tmp88
    tmp90 = tl.where(tmp89, tmp80, tmp81)
    tmp91 = tl.full(tmp90.shape, 0.0, tmp90.dtype)
    tmp92 = tl.where(tmp88, tmp90, tmp91)
    tmp93 = tmp92 + tmp84
    tmp94 = tmp23 >= tmp70
    tmp95 = tmp23 < tmp72
    tmp96 = tmp94 & tmp95
    tmp97 = tmp74 & tmp96
    tmp98 = tmp27 & tmp97
    tmp99 = tl.where(tmp98, tmp80, tmp81)
    tmp100 = tl.full(tmp99.shape, 0.0, tmp99.dtype)
    tmp101 = tl.where(tmp97, tmp99, tmp100)
    tmp102 = tmp101 + tmp93
    tmp103 = tmp32 >= tmp70
    tmp104 = tmp32 < tmp72
    tmp105 = tmp103 & tmp104
    tmp106 = tmp105 & tmp77
    tmp107 = tmp36 & tmp106
    tmp108 = tl.where(tmp107, tmp80, tmp81)
    tmp109 = tl.full(tmp108.shape, 0.0, tmp108.dtype)
    tmp110 = tl.where(tmp106, tmp108, tmp109)
    tmp111 = tmp110 + tmp102
    tmp112 = tmp105 & tmp87
    tmp113 = tmp41 & tmp112
    tmp114 = tl.where(tmp113, tmp80, tmp81)
    tmp115 = tl.full(tmp114.shape, 0.0, tmp114.dtype)
    tmp116 = tl.where(tmp112, tmp114, tmp115)
    tmp117 = tmp116 + tmp111
    tmp118 = tmp105 & tmp96
    tmp119 = tmp46 & tmp118
    tmp120 = tl.where(tmp119, tmp80, tmp81)
    tmp121 = tl.full(tmp120.shape, 0.0, tmp120.dtype)
    tmp122 = tl.where(tmp118, tmp120, tmp121)
    tmp123 = tmp122 + tmp117
    tmp124 = tmp51 >= tmp70
    tmp125 = tmp51 < tmp72
    tmp126 = tmp124 & tmp125
    tmp127 = tmp126 & tmp77
    tmp128 = tmp55 & tmp127
    tmp129 = tl.where(tmp128, tmp80, tmp81)
    tmp130 = tl.full(tmp129.shape, 0.0, tmp129.dtype)
    tmp131 = tl.where(tmp127, tmp129, tmp130)
    tmp132 = tmp131 + tmp123
    tmp133 = tmp126 & tmp87
    tmp134 = tmp60 & tmp133
    tmp135 = tl.where(tmp134, tmp80, tmp81)
    tmp136 = tl.full(tmp135.shape, 0.0, tmp135.dtype)
    tmp137 = tl.where(tmp133, tmp135, tmp136)
    tmp138 = tmp137 + tmp132
    tmp139 = tmp126 & tmp96
    tmp140 = tmp65 & tmp139
    tmp141 = tl.where(tmp140, tmp80, tmp81)
    tmp142 = tl.full(tmp141.shape, 0.0, tmp141.dtype)
    tmp143 = tl.where(tmp139, tmp141, tmp142)
    tmp144 = tmp143 + tmp138
    tmp145 = tmp69 / tmp144
    tl.store(out_ptr0 + (x6), tmp145, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/th/cthb5lbj6iojrljfraqorbhvmg6whdywywaxfgdnitkfcwynsan6.py
# Source Nodes: [branch_pool_17, x_196], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# branch_pool_17 => relu_86
# x_196 => add_434, add_437, mul_605, mul_611, rsqrt_86, sub_86, var_mean_86
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_110 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[8192, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_110', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 64
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 192
    y1 = (yindex // 192)
    tmp0 = tl.load(in_ptr0 + (y0 + (192*x2) + (12288*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 2048.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x2 + (64*y0) + (131072*y1)), tmp14, xmask)
    tl.store(out_ptr1 + (y0 + (192*x2) + (12288*y1)), tmp16, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/hn/chnpd6hq273rmwefwg2lbmpadibzhbjkzr25u2rpa6pvpdna3ysi.py
# Source Nodes: [cat_19], Original ATen: [aten.cat]
# cat_19 => cat_12
triton_poi_fused_cat_111 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_111', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 49152
    x1 = (xindex // 49152)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + (131072*x1)), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/vr/cvracnrcokxgzzoyzp6p4wu2ayhrluibik5wffjkumspibovmfij.py
# Source Nodes: [cat_19], Original ATen: [aten.cat]
# cat_19 => cat_12
triton_poi_fused_cat_112 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_112', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
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
    x2 = xindex
    y3 = yindex
    y0 = yindex % 2048
    y1 = (yindex // 2048)
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (2048*x2) + (131072*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/pl/cplvu3ba2nfupxhb3yyw2knesb66xu73a7rtahsp6wyflpysylvw.py
# Source Nodes: [branch_pool_18], Original ATen: [aten.avg_pool2d]
# branch_pool_18 => avg_pool2d_9
triton_poi_fused_avg_pool2d_113 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_113', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 16384) % 8
    x1 = (xindex // 2048) % 8
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-18432) + x6), tmp10, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = x1
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-16384) + x6), tmp18, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + x1
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-14336) + x6), tmp27, other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = x2
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-2048) + x6), tmp36, other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (x6), tmp41, other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (2048 + x6), tmp46, other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + x2
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (14336 + x6), tmp55, other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (16384 + x6), tmp60, other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (18432 + x6), tmp65, other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = tl.full([1], -1, tl.int64)
    tmp71 = tmp0 >= tmp70
    tmp72 = tl.full([1], 9, tl.int64)
    tmp73 = tmp0 < tmp72
    tmp74 = tmp71 & tmp73
    tmp75 = tmp6 >= tmp70
    tmp76 = tmp6 < tmp72
    tmp77 = tmp75 & tmp76
    tmp78 = tmp74 & tmp77
    tmp79 = tmp10 & tmp78
    tmp80 = 1.0
    tmp81 = tl.full(tmp80.shape, 1.0, tmp80.dtype)
    tmp82 = tl.where(tmp79, tmp80, tmp81)
    tmp83 = tl.full(tmp82.shape, 0.0, tmp82.dtype)
    tmp84 = tl.where(tmp78, tmp82, tmp83)
    tmp85 = tmp14 >= tmp70
    tmp86 = tmp14 < tmp72
    tmp87 = tmp85 & tmp86
    tmp88 = tmp74 & tmp87
    tmp89 = tmp18 & tmp88
    tmp90 = tl.where(tmp89, tmp80, tmp81)
    tmp91 = tl.full(tmp90.shape, 0.0, tmp90.dtype)
    tmp92 = tl.where(tmp88, tmp90, tmp91)
    tmp93 = tmp92 + tmp84
    tmp94 = tmp23 >= tmp70
    tmp95 = tmp23 < tmp72
    tmp96 = tmp94 & tmp95
    tmp97 = tmp74 & tmp96
    tmp98 = tmp27 & tmp97
    tmp99 = tl.where(tmp98, tmp80, tmp81)
    tmp100 = tl.full(tmp99.shape, 0.0, tmp99.dtype)
    tmp101 = tl.where(tmp97, tmp99, tmp100)
    tmp102 = tmp101 + tmp93
    tmp103 = tmp32 >= tmp70
    tmp104 = tmp32 < tmp72
    tmp105 = tmp103 & tmp104
    tmp106 = tmp105 & tmp77
    tmp107 = tmp36 & tmp106
    tmp108 = tl.where(tmp107, tmp80, tmp81)
    tmp109 = tl.full(tmp108.shape, 0.0, tmp108.dtype)
    tmp110 = tl.where(tmp106, tmp108, tmp109)
    tmp111 = tmp110 + tmp102
    tmp112 = tmp105 & tmp87
    tmp113 = tmp41 & tmp112
    tmp114 = tl.where(tmp113, tmp80, tmp81)
    tmp115 = tl.full(tmp114.shape, 0.0, tmp114.dtype)
    tmp116 = tl.where(tmp112, tmp114, tmp115)
    tmp117 = tmp116 + tmp111
    tmp118 = tmp105 & tmp96
    tmp119 = tmp46 & tmp118
    tmp120 = tl.where(tmp119, tmp80, tmp81)
    tmp121 = tl.full(tmp120.shape, 0.0, tmp120.dtype)
    tmp122 = tl.where(tmp118, tmp120, tmp121)
    tmp123 = tmp122 + tmp117
    tmp124 = tmp51 >= tmp70
    tmp125 = tmp51 < tmp72
    tmp126 = tmp124 & tmp125
    tmp127 = tmp126 & tmp77
    tmp128 = tmp55 & tmp127
    tmp129 = tl.where(tmp128, tmp80, tmp81)
    tmp130 = tl.full(tmp129.shape, 0.0, tmp129.dtype)
    tmp131 = tl.where(tmp127, tmp129, tmp130)
    tmp132 = tmp131 + tmp123
    tmp133 = tmp126 & tmp87
    tmp134 = tmp60 & tmp133
    tmp135 = tl.where(tmp134, tmp80, tmp81)
    tmp136 = tl.full(tmp135.shape, 0.0, tmp135.dtype)
    tmp137 = tl.where(tmp133, tmp135, tmp136)
    tmp138 = tmp137 + tmp132
    tmp139 = tmp126 & tmp96
    tmp140 = tmp65 & tmp139
    tmp141 = tl.where(tmp140, tmp80, tmp81)
    tmp142 = tl.full(tmp141.shape, 0.0, tmp141.dtype)
    tmp143 = tl.where(tmp139, tmp141, tmp142)
    tmp144 = tmp143 + tmp138
    tmp145 = tmp69 / tmp144
    tl.store(out_ptr0 + (x6), tmp145, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/il/cilirpbtuf7auftyqkmc2zdxln4d7yhgthflt2t3qewh2h6hzhg5.py
# Source Nodes: [x_217, x_218], Original ATen: [aten.mean, aten.native_dropout]
# x_217 => mean_1
# x_218 => gt, mul_675, mul_676
triton_per_fused_mean_native_dropout_114 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[65536, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: '*i1', 4: 'i32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_native_dropout_114', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, load_seed_offset, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
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
    tmp0 = tl.load(in_ptr0 + (r1 + (64*x0)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = tl.load(in_ptr1 + load_seed_offset)
    tmp6 = x0
    tmp7 = tl.rand(tmp5, (tmp6).to(tl.uint32))
    tmp8 = 0.5
    tmp9 = tmp7 > tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp11 = 64.0
    tmp12 = tmp4 / tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = 2.0
    tmp15 = tmp13 * tmp14
    tl.store(out_ptr1 + (x0), tmp9, None)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_zhang402/ml/cmlkucab2w5gpxd3eoolef5kqwegw4cr6avyujercgtwmrlzjizv.py
# Source Nodes: [x_2], Original ATen: [aten.add]
# x_2 => add_3
triton_poi_fused_add_115 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[1], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {2: 1}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=(2,), ids_of_folded_args=(2,), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_115', 'mutated_arg_names': ['in_ptr0', 'out_ptr1'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
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
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581 = args
    args.clear()
    assert_size_stride(primals_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (32, ), (1, ))
    assert_size_stride(primals_3, (32, ), (1, ))
    assert_size_stride(primals_4, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_5, (32, ), (1, ))
    assert_size_stride(primals_6, (32, ), (1, ))
    assert_size_stride(primals_7, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_8, (64, ), (1, ))
    assert_size_stride(primals_9, (64, ), (1, ))
    assert_size_stride(primals_10, (80, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_11, (80, ), (1, ))
    assert_size_stride(primals_12, (80, ), (1, ))
    assert_size_stride(primals_13, (192, 80, 3, 3), (720, 9, 3, 1))
    assert_size_stride(primals_14, (192, ), (1, ))
    assert_size_stride(primals_15, (192, ), (1, ))
    assert_size_stride(primals_16, (64, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_17, (64, ), (1, ))
    assert_size_stride(primals_18, (64, ), (1, ))
    assert_size_stride(primals_19, (48, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_20, (48, ), (1, ))
    assert_size_stride(primals_21, (48, ), (1, ))
    assert_size_stride(primals_22, (64, 48, 5, 5), (1200, 25, 5, 1))
    assert_size_stride(primals_23, (64, ), (1, ))
    assert_size_stride(primals_24, (64, ), (1, ))
    assert_size_stride(primals_25, (64, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_26, (64, ), (1, ))
    assert_size_stride(primals_27, (64, ), (1, ))
    assert_size_stride(primals_28, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_29, (96, ), (1, ))
    assert_size_stride(primals_30, (96, ), (1, ))
    assert_size_stride(primals_31, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_32, (96, ), (1, ))
    assert_size_stride(primals_33, (96, ), (1, ))
    assert_size_stride(primals_34, (32, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_35, (32, ), (1, ))
    assert_size_stride(primals_36, (32, ), (1, ))
    assert_size_stride(primals_37, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_38, (64, ), (1, ))
    assert_size_stride(primals_39, (64, ), (1, ))
    assert_size_stride(primals_40, (48, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_41, (48, ), (1, ))
    assert_size_stride(primals_42, (48, ), (1, ))
    assert_size_stride(primals_43, (64, 48, 5, 5), (1200, 25, 5, 1))
    assert_size_stride(primals_44, (64, ), (1, ))
    assert_size_stride(primals_45, (64, ), (1, ))
    assert_size_stride(primals_46, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_47, (64, ), (1, ))
    assert_size_stride(primals_48, (64, ), (1, ))
    assert_size_stride(primals_49, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_50, (96, ), (1, ))
    assert_size_stride(primals_51, (96, ), (1, ))
    assert_size_stride(primals_52, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_53, (96, ), (1, ))
    assert_size_stride(primals_54, (96, ), (1, ))
    assert_size_stride(primals_55, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_56, (64, ), (1, ))
    assert_size_stride(primals_57, (64, ), (1, ))
    assert_size_stride(primals_58, (64, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(primals_59, (64, ), (1, ))
    assert_size_stride(primals_60, (64, ), (1, ))
    assert_size_stride(primals_61, (48, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(primals_62, (48, ), (1, ))
    assert_size_stride(primals_63, (48, ), (1, ))
    assert_size_stride(primals_64, (64, 48, 5, 5), (1200, 25, 5, 1))
    assert_size_stride(primals_65, (64, ), (1, ))
    assert_size_stride(primals_66, (64, ), (1, ))
    assert_size_stride(primals_67, (64, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(primals_68, (64, ), (1, ))
    assert_size_stride(primals_69, (64, ), (1, ))
    assert_size_stride(primals_70, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_71, (96, ), (1, ))
    assert_size_stride(primals_72, (96, ), (1, ))
    assert_size_stride(primals_73, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_74, (96, ), (1, ))
    assert_size_stride(primals_75, (96, ), (1, ))
    assert_size_stride(primals_76, (64, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(primals_77, (64, ), (1, ))
    assert_size_stride(primals_78, (64, ), (1, ))
    assert_size_stride(primals_79, (384, 288, 3, 3), (2592, 9, 3, 1))
    assert_size_stride(primals_80, (384, ), (1, ))
    assert_size_stride(primals_81, (384, ), (1, ))
    assert_size_stride(primals_82, (64, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(primals_83, (64, ), (1, ))
    assert_size_stride(primals_84, (64, ), (1, ))
    assert_size_stride(primals_85, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_86, (96, ), (1, ))
    assert_size_stride(primals_87, (96, ), (1, ))
    assert_size_stride(primals_88, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_89, (96, ), (1, ))
    assert_size_stride(primals_90, (96, ), (1, ))
    assert_size_stride(primals_91, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_92, (192, ), (1, ))
    assert_size_stride(primals_93, (192, ), (1, ))
    assert_size_stride(primals_94, (128, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_95, (128, ), (1, ))
    assert_size_stride(primals_96, (128, ), (1, ))
    assert_size_stride(primals_97, (128, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(primals_98, (128, ), (1, ))
    assert_size_stride(primals_99, (128, ), (1, ))
    assert_size_stride(primals_100, (192, 128, 7, 1), (896, 7, 1, 1))
    assert_size_stride(primals_101, (192, ), (1, ))
    assert_size_stride(primals_102, (192, ), (1, ))
    assert_size_stride(primals_103, (128, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_104, (128, ), (1, ))
    assert_size_stride(primals_105, (128, ), (1, ))
    assert_size_stride(primals_106, (128, 128, 7, 1), (896, 7, 1, 1))
    assert_size_stride(primals_107, (128, ), (1, ))
    assert_size_stride(primals_108, (128, ), (1, ))
    assert_size_stride(primals_109, (128, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(primals_110, (128, ), (1, ))
    assert_size_stride(primals_111, (128, ), (1, ))
    assert_size_stride(primals_112, (128, 128, 7, 1), (896, 7, 1, 1))
    assert_size_stride(primals_113, (128, ), (1, ))
    assert_size_stride(primals_114, (128, ), (1, ))
    assert_size_stride(primals_115, (192, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(primals_116, (192, ), (1, ))
    assert_size_stride(primals_117, (192, ), (1, ))
    assert_size_stride(primals_118, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_119, (192, ), (1, ))
    assert_size_stride(primals_120, (192, ), (1, ))
    assert_size_stride(primals_121, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_122, (192, ), (1, ))
    assert_size_stride(primals_123, (192, ), (1, ))
    assert_size_stride(primals_124, (160, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_125, (160, ), (1, ))
    assert_size_stride(primals_126, (160, ), (1, ))
    assert_size_stride(primals_127, (160, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(primals_128, (160, ), (1, ))
    assert_size_stride(primals_129, (160, ), (1, ))
    assert_size_stride(primals_130, (192, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(primals_131, (192, ), (1, ))
    assert_size_stride(primals_132, (192, ), (1, ))
    assert_size_stride(primals_133, (160, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_134, (160, ), (1, ))
    assert_size_stride(primals_135, (160, ), (1, ))
    assert_size_stride(primals_136, (160, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(primals_137, (160, ), (1, ))
    assert_size_stride(primals_138, (160, ), (1, ))
    assert_size_stride(primals_139, (160, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(primals_140, (160, ), (1, ))
    assert_size_stride(primals_141, (160, ), (1, ))
    assert_size_stride(primals_142, (160, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(primals_143, (160, ), (1, ))
    assert_size_stride(primals_144, (160, ), (1, ))
    assert_size_stride(primals_145, (192, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(primals_146, (192, ), (1, ))
    assert_size_stride(primals_147, (192, ), (1, ))
    assert_size_stride(primals_148, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_149, (192, ), (1, ))
    assert_size_stride(primals_150, (192, ), (1, ))
    assert_size_stride(primals_151, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_152, (192, ), (1, ))
    assert_size_stride(primals_153, (192, ), (1, ))
    assert_size_stride(primals_154, (160, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_155, (160, ), (1, ))
    assert_size_stride(primals_156, (160, ), (1, ))
    assert_size_stride(primals_157, (160, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(primals_158, (160, ), (1, ))
    assert_size_stride(primals_159, (160, ), (1, ))
    assert_size_stride(primals_160, (192, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(primals_161, (192, ), (1, ))
    assert_size_stride(primals_162, (192, ), (1, ))
    assert_size_stride(primals_163, (160, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_164, (160, ), (1, ))
    assert_size_stride(primals_165, (160, ), (1, ))
    assert_size_stride(primals_166, (160, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(primals_167, (160, ), (1, ))
    assert_size_stride(primals_168, (160, ), (1, ))
    assert_size_stride(primals_169, (160, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(primals_170, (160, ), (1, ))
    assert_size_stride(primals_171, (160, ), (1, ))
    assert_size_stride(primals_172, (160, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(primals_173, (160, ), (1, ))
    assert_size_stride(primals_174, (160, ), (1, ))
    assert_size_stride(primals_175, (192, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(primals_176, (192, ), (1, ))
    assert_size_stride(primals_177, (192, ), (1, ))
    assert_size_stride(primals_178, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_179, (192, ), (1, ))
    assert_size_stride(primals_180, (192, ), (1, ))
    assert_size_stride(primals_181, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_182, (192, ), (1, ))
    assert_size_stride(primals_183, (192, ), (1, ))
    assert_size_stride(primals_184, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_185, (192, ), (1, ))
    assert_size_stride(primals_186, (192, ), (1, ))
    assert_size_stride(primals_187, (192, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(primals_188, (192, ), (1, ))
    assert_size_stride(primals_189, (192, ), (1, ))
    assert_size_stride(primals_190, (192, 192, 7, 1), (1344, 7, 1, 1))
    assert_size_stride(primals_191, (192, ), (1, ))
    assert_size_stride(primals_192, (192, ), (1, ))
    assert_size_stride(primals_193, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_194, (192, ), (1, ))
    assert_size_stride(primals_195, (192, ), (1, ))
    assert_size_stride(primals_196, (192, 192, 7, 1), (1344, 7, 1, 1))
    assert_size_stride(primals_197, (192, ), (1, ))
    assert_size_stride(primals_198, (192, ), (1, ))
    assert_size_stride(primals_199, (192, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(primals_200, (192, ), (1, ))
    assert_size_stride(primals_201, (192, ), (1, ))
    assert_size_stride(primals_202, (192, 192, 7, 1), (1344, 7, 1, 1))
    assert_size_stride(primals_203, (192, ), (1, ))
    assert_size_stride(primals_204, (192, ), (1, ))
    assert_size_stride(primals_205, (192, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(primals_206, (192, ), (1, ))
    assert_size_stride(primals_207, (192, ), (1, ))
    assert_size_stride(primals_208, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_209, (192, ), (1, ))
    assert_size_stride(primals_210, (192, ), (1, ))
    assert_size_stride(primals_211, (128, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_212, (128, ), (1, ))
    assert_size_stride(primals_213, (128, ), (1, ))
    assert_size_stride(primals_214, (768, 128, 5, 5), (3200, 25, 5, 1))
    assert_size_stride(primals_215, (768, ), (1, ))
    assert_size_stride(primals_216, (768, ), (1, ))
    assert_size_stride(primals_217, (1000, 768), (768, 1))
    assert_size_stride(primals_218, (1000, ), (1, ))
    assert_size_stride(primals_219, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_220, (192, ), (1, ))
    assert_size_stride(primals_221, (192, ), (1, ))
    assert_size_stride(primals_222, (320, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_223, (320, ), (1, ))
    assert_size_stride(primals_224, (320, ), (1, ))
    assert_size_stride(primals_225, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_226, (192, ), (1, ))
    assert_size_stride(primals_227, (192, ), (1, ))
    assert_size_stride(primals_228, (192, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(primals_229, (192, ), (1, ))
    assert_size_stride(primals_230, (192, ), (1, ))
    assert_size_stride(primals_231, (192, 192, 7, 1), (1344, 7, 1, 1))
    assert_size_stride(primals_232, (192, ), (1, ))
    assert_size_stride(primals_233, (192, ), (1, ))
    assert_size_stride(primals_234, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_235, (192, ), (1, ))
    assert_size_stride(primals_236, (192, ), (1, ))
    assert_size_stride(primals_237, (320, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(primals_238, (320, ), (1, ))
    assert_size_stride(primals_239, (320, ), (1, ))
    assert_size_stride(primals_240, (384, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(primals_241, (384, ), (1, ))
    assert_size_stride(primals_242, (384, ), (1, ))
    assert_size_stride(primals_243, (384, 384, 1, 3), (1152, 3, 3, 1))
    assert_size_stride(primals_244, (384, ), (1, ))
    assert_size_stride(primals_245, (384, ), (1, ))
    assert_size_stride(primals_246, (384, 384, 3, 1), (1152, 3, 1, 1))
    assert_size_stride(primals_247, (384, ), (1, ))
    assert_size_stride(primals_248, (384, ), (1, ))
    assert_size_stride(primals_249, (448, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(primals_250, (448, ), (1, ))
    assert_size_stride(primals_251, (448, ), (1, ))
    assert_size_stride(primals_252, (384, 448, 3, 3), (4032, 9, 3, 1))
    assert_size_stride(primals_253, (384, ), (1, ))
    assert_size_stride(primals_254, (384, ), (1, ))
    assert_size_stride(primals_255, (384, 384, 1, 3), (1152, 3, 3, 1))
    assert_size_stride(primals_256, (384, ), (1, ))
    assert_size_stride(primals_257, (384, ), (1, ))
    assert_size_stride(primals_258, (384, 384, 3, 1), (1152, 3, 1, 1))
    assert_size_stride(primals_259, (384, ), (1, ))
    assert_size_stride(primals_260, (384, ), (1, ))
    assert_size_stride(primals_261, (192, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(primals_262, (192, ), (1, ))
    assert_size_stride(primals_263, (192, ), (1, ))
    assert_size_stride(primals_264, (320, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_265, (320, ), (1, ))
    assert_size_stride(primals_266, (320, ), (1, ))
    assert_size_stride(primals_267, (384, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_268, (384, ), (1, ))
    assert_size_stride(primals_269, (384, ), (1, ))
    assert_size_stride(primals_270, (384, 384, 1, 3), (1152, 3, 3, 1))
    assert_size_stride(primals_271, (384, ), (1, ))
    assert_size_stride(primals_272, (384, ), (1, ))
    assert_size_stride(primals_273, (384, 384, 3, 1), (1152, 3, 1, 1))
    assert_size_stride(primals_274, (384, ), (1, ))
    assert_size_stride(primals_275, (384, ), (1, ))
    assert_size_stride(primals_276, (448, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_277, (448, ), (1, ))
    assert_size_stride(primals_278, (448, ), (1, ))
    assert_size_stride(primals_279, (384, 448, 3, 3), (4032, 9, 3, 1))
    assert_size_stride(primals_280, (384, ), (1, ))
    assert_size_stride(primals_281, (384, ), (1, ))
    assert_size_stride(primals_282, (384, 384, 1, 3), (1152, 3, 3, 1))
    assert_size_stride(primals_283, (384, ), (1, ))
    assert_size_stride(primals_284, (384, ), (1, ))
    assert_size_stride(primals_285, (384, 384, 3, 1), (1152, 3, 1, 1))
    assert_size_stride(primals_286, (384, ), (1, ))
    assert_size_stride(primals_287, (384, ), (1, ))
    assert_size_stride(primals_288, (192, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_289, (192, ), (1, ))
    assert_size_stride(primals_290, (192, ), (1, ))
    assert_size_stride(primals_291, (1000, 2048), (2048, 1))
    assert_size_stride(primals_292, (1000, ), (1, ))
    assert_size_stride(primals_293, (32, ), (1, ))
    assert_size_stride(primals_294, (32, ), (1, ))
    assert_size_stride(primals_295, (), ())
    assert_size_stride(primals_296, (32, ), (1, ))
    assert_size_stride(primals_297, (32, ), (1, ))
    assert_size_stride(primals_298, (), ())
    assert_size_stride(primals_299, (64, ), (1, ))
    assert_size_stride(primals_300, (64, ), (1, ))
    assert_size_stride(primals_301, (), ())
    assert_size_stride(primals_302, (80, ), (1, ))
    assert_size_stride(primals_303, (80, ), (1, ))
    assert_size_stride(primals_304, (), ())
    assert_size_stride(primals_305, (192, ), (1, ))
    assert_size_stride(primals_306, (192, ), (1, ))
    assert_size_stride(primals_307, (), ())
    assert_size_stride(primals_308, (64, ), (1, ))
    assert_size_stride(primals_309, (64, ), (1, ))
    assert_size_stride(primals_310, (), ())
    assert_size_stride(primals_311, (48, ), (1, ))
    assert_size_stride(primals_312, (48, ), (1, ))
    assert_size_stride(primals_313, (), ())
    assert_size_stride(primals_314, (64, ), (1, ))
    assert_size_stride(primals_315, (64, ), (1, ))
    assert_size_stride(primals_316, (), ())
    assert_size_stride(primals_317, (64, ), (1, ))
    assert_size_stride(primals_318, (64, ), (1, ))
    assert_size_stride(primals_319, (), ())
    assert_size_stride(primals_320, (96, ), (1, ))
    assert_size_stride(primals_321, (96, ), (1, ))
    assert_size_stride(primals_322, (), ())
    assert_size_stride(primals_323, (96, ), (1, ))
    assert_size_stride(primals_324, (96, ), (1, ))
    assert_size_stride(primals_325, (), ())
    assert_size_stride(primals_326, (32, ), (1, ))
    assert_size_stride(primals_327, (32, ), (1, ))
    assert_size_stride(primals_328, (), ())
    assert_size_stride(primals_329, (64, ), (1, ))
    assert_size_stride(primals_330, (64, ), (1, ))
    assert_size_stride(primals_331, (), ())
    assert_size_stride(primals_332, (48, ), (1, ))
    assert_size_stride(primals_333, (48, ), (1, ))
    assert_size_stride(primals_334, (), ())
    assert_size_stride(primals_335, (64, ), (1, ))
    assert_size_stride(primals_336, (64, ), (1, ))
    assert_size_stride(primals_337, (), ())
    assert_size_stride(primals_338, (64, ), (1, ))
    assert_size_stride(primals_339, (64, ), (1, ))
    assert_size_stride(primals_340, (), ())
    assert_size_stride(primals_341, (96, ), (1, ))
    assert_size_stride(primals_342, (96, ), (1, ))
    assert_size_stride(primals_343, (), ())
    assert_size_stride(primals_344, (96, ), (1, ))
    assert_size_stride(primals_345, (96, ), (1, ))
    assert_size_stride(primals_346, (), ())
    assert_size_stride(primals_347, (64, ), (1, ))
    assert_size_stride(primals_348, (64, ), (1, ))
    assert_size_stride(primals_349, (), ())
    assert_size_stride(primals_350, (64, ), (1, ))
    assert_size_stride(primals_351, (64, ), (1, ))
    assert_size_stride(primals_352, (), ())
    assert_size_stride(primals_353, (48, ), (1, ))
    assert_size_stride(primals_354, (48, ), (1, ))
    assert_size_stride(primals_355, (), ())
    assert_size_stride(primals_356, (64, ), (1, ))
    assert_size_stride(primals_357, (64, ), (1, ))
    assert_size_stride(primals_358, (), ())
    assert_size_stride(primals_359, (64, ), (1, ))
    assert_size_stride(primals_360, (64, ), (1, ))
    assert_size_stride(primals_361, (), ())
    assert_size_stride(primals_362, (96, ), (1, ))
    assert_size_stride(primals_363, (96, ), (1, ))
    assert_size_stride(primals_364, (), ())
    assert_size_stride(primals_365, (96, ), (1, ))
    assert_size_stride(primals_366, (96, ), (1, ))
    assert_size_stride(primals_367, (), ())
    assert_size_stride(primals_368, (64, ), (1, ))
    assert_size_stride(primals_369, (64, ), (1, ))
    assert_size_stride(primals_370, (), ())
    assert_size_stride(primals_371, (384, ), (1, ))
    assert_size_stride(primals_372, (384, ), (1, ))
    assert_size_stride(primals_373, (), ())
    assert_size_stride(primals_374, (64, ), (1, ))
    assert_size_stride(primals_375, (64, ), (1, ))
    assert_size_stride(primals_376, (), ())
    assert_size_stride(primals_377, (96, ), (1, ))
    assert_size_stride(primals_378, (96, ), (1, ))
    assert_size_stride(primals_379, (), ())
    assert_size_stride(primals_380, (96, ), (1, ))
    assert_size_stride(primals_381, (96, ), (1, ))
    assert_size_stride(primals_382, (), ())
    assert_size_stride(primals_383, (192, ), (1, ))
    assert_size_stride(primals_384, (192, ), (1, ))
    assert_size_stride(primals_385, (), ())
    assert_size_stride(primals_386, (128, ), (1, ))
    assert_size_stride(primals_387, (128, ), (1, ))
    assert_size_stride(primals_388, (), ())
    assert_size_stride(primals_389, (128, ), (1, ))
    assert_size_stride(primals_390, (128, ), (1, ))
    assert_size_stride(primals_391, (), ())
    assert_size_stride(primals_392, (192, ), (1, ))
    assert_size_stride(primals_393, (192, ), (1, ))
    assert_size_stride(primals_394, (), ())
    assert_size_stride(primals_395, (128, ), (1, ))
    assert_size_stride(primals_396, (128, ), (1, ))
    assert_size_stride(primals_397, (), ())
    assert_size_stride(primals_398, (128, ), (1, ))
    assert_size_stride(primals_399, (128, ), (1, ))
    assert_size_stride(primals_400, (), ())
    assert_size_stride(primals_401, (128, ), (1, ))
    assert_size_stride(primals_402, (128, ), (1, ))
    assert_size_stride(primals_403, (), ())
    assert_size_stride(primals_404, (128, ), (1, ))
    assert_size_stride(primals_405, (128, ), (1, ))
    assert_size_stride(primals_406, (), ())
    assert_size_stride(primals_407, (192, ), (1, ))
    assert_size_stride(primals_408, (192, ), (1, ))
    assert_size_stride(primals_409, (), ())
    assert_size_stride(primals_410, (192, ), (1, ))
    assert_size_stride(primals_411, (192, ), (1, ))
    assert_size_stride(primals_412, (), ())
    assert_size_stride(primals_413, (192, ), (1, ))
    assert_size_stride(primals_414, (192, ), (1, ))
    assert_size_stride(primals_415, (), ())
    assert_size_stride(primals_416, (160, ), (1, ))
    assert_size_stride(primals_417, (160, ), (1, ))
    assert_size_stride(primals_418, (), ())
    assert_size_stride(primals_419, (160, ), (1, ))
    assert_size_stride(primals_420, (160, ), (1, ))
    assert_size_stride(primals_421, (), ())
    assert_size_stride(primals_422, (192, ), (1, ))
    assert_size_stride(primals_423, (192, ), (1, ))
    assert_size_stride(primals_424, (), ())
    assert_size_stride(primals_425, (160, ), (1, ))
    assert_size_stride(primals_426, (160, ), (1, ))
    assert_size_stride(primals_427, (), ())
    assert_size_stride(primals_428, (160, ), (1, ))
    assert_size_stride(primals_429, (160, ), (1, ))
    assert_size_stride(primals_430, (), ())
    assert_size_stride(primals_431, (160, ), (1, ))
    assert_size_stride(primals_432, (160, ), (1, ))
    assert_size_stride(primals_433, (), ())
    assert_size_stride(primals_434, (160, ), (1, ))
    assert_size_stride(primals_435, (160, ), (1, ))
    assert_size_stride(primals_436, (), ())
    assert_size_stride(primals_437, (192, ), (1, ))
    assert_size_stride(primals_438, (192, ), (1, ))
    assert_size_stride(primals_439, (), ())
    assert_size_stride(primals_440, (192, ), (1, ))
    assert_size_stride(primals_441, (192, ), (1, ))
    assert_size_stride(primals_442, (), ())
    assert_size_stride(primals_443, (192, ), (1, ))
    assert_size_stride(primals_444, (192, ), (1, ))
    assert_size_stride(primals_445, (), ())
    assert_size_stride(primals_446, (160, ), (1, ))
    assert_size_stride(primals_447, (160, ), (1, ))
    assert_size_stride(primals_448, (), ())
    assert_size_stride(primals_449, (160, ), (1, ))
    assert_size_stride(primals_450, (160, ), (1, ))
    assert_size_stride(primals_451, (), ())
    assert_size_stride(primals_452, (192, ), (1, ))
    assert_size_stride(primals_453, (192, ), (1, ))
    assert_size_stride(primals_454, (), ())
    assert_size_stride(primals_455, (160, ), (1, ))
    assert_size_stride(primals_456, (160, ), (1, ))
    assert_size_stride(primals_457, (), ())
    assert_size_stride(primals_458, (160, ), (1, ))
    assert_size_stride(primals_459, (160, ), (1, ))
    assert_size_stride(primals_460, (), ())
    assert_size_stride(primals_461, (160, ), (1, ))
    assert_size_stride(primals_462, (160, ), (1, ))
    assert_size_stride(primals_463, (), ())
    assert_size_stride(primals_464, (160, ), (1, ))
    assert_size_stride(primals_465, (160, ), (1, ))
    assert_size_stride(primals_466, (), ())
    assert_size_stride(primals_467, (192, ), (1, ))
    assert_size_stride(primals_468, (192, ), (1, ))
    assert_size_stride(primals_469, (), ())
    assert_size_stride(primals_470, (192, ), (1, ))
    assert_size_stride(primals_471, (192, ), (1, ))
    assert_size_stride(primals_472, (), ())
    assert_size_stride(primals_473, (192, ), (1, ))
    assert_size_stride(primals_474, (192, ), (1, ))
    assert_size_stride(primals_475, (), ())
    assert_size_stride(primals_476, (192, ), (1, ))
    assert_size_stride(primals_477, (192, ), (1, ))
    assert_size_stride(primals_478, (), ())
    assert_size_stride(primals_479, (192, ), (1, ))
    assert_size_stride(primals_480, (192, ), (1, ))
    assert_size_stride(primals_481, (), ())
    assert_size_stride(primals_482, (192, ), (1, ))
    assert_size_stride(primals_483, (192, ), (1, ))
    assert_size_stride(primals_484, (), ())
    assert_size_stride(primals_485, (192, ), (1, ))
    assert_size_stride(primals_486, (192, ), (1, ))
    assert_size_stride(primals_487, (), ())
    assert_size_stride(primals_488, (192, ), (1, ))
    assert_size_stride(primals_489, (192, ), (1, ))
    assert_size_stride(primals_490, (), ())
    assert_size_stride(primals_491, (192, ), (1, ))
    assert_size_stride(primals_492, (192, ), (1, ))
    assert_size_stride(primals_493, (), ())
    assert_size_stride(primals_494, (192, ), (1, ))
    assert_size_stride(primals_495, (192, ), (1, ))
    assert_size_stride(primals_496, (), ())
    assert_size_stride(primals_497, (192, ), (1, ))
    assert_size_stride(primals_498, (192, ), (1, ))
    assert_size_stride(primals_499, (), ())
    assert_size_stride(primals_500, (192, ), (1, ))
    assert_size_stride(primals_501, (192, ), (1, ))
    assert_size_stride(primals_502, (), ())
    assert_size_stride(primals_503, (128, ), (1, ))
    assert_size_stride(primals_504, (128, ), (1, ))
    assert_size_stride(primals_505, (), ())
    assert_size_stride(primals_506, (768, ), (1, ))
    assert_size_stride(primals_507, (768, ), (1, ))
    assert_size_stride(primals_508, (), ())
    assert_size_stride(primals_509, (192, ), (1, ))
    assert_size_stride(primals_510, (192, ), (1, ))
    assert_size_stride(primals_511, (), ())
    assert_size_stride(primals_512, (320, ), (1, ))
    assert_size_stride(primals_513, (320, ), (1, ))
    assert_size_stride(primals_514, (), ())
    assert_size_stride(primals_515, (192, ), (1, ))
    assert_size_stride(primals_516, (192, ), (1, ))
    assert_size_stride(primals_517, (), ())
    assert_size_stride(primals_518, (192, ), (1, ))
    assert_size_stride(primals_519, (192, ), (1, ))
    assert_size_stride(primals_520, (), ())
    assert_size_stride(primals_521, (192, ), (1, ))
    assert_size_stride(primals_522, (192, ), (1, ))
    assert_size_stride(primals_523, (), ())
    assert_size_stride(primals_524, (192, ), (1, ))
    assert_size_stride(primals_525, (192, ), (1, ))
    assert_size_stride(primals_526, (), ())
    assert_size_stride(primals_527, (320, ), (1, ))
    assert_size_stride(primals_528, (320, ), (1, ))
    assert_size_stride(primals_529, (), ())
    assert_size_stride(primals_530, (384, ), (1, ))
    assert_size_stride(primals_531, (384, ), (1, ))
    assert_size_stride(primals_532, (), ())
    assert_size_stride(primals_533, (384, ), (1, ))
    assert_size_stride(primals_534, (384, ), (1, ))
    assert_size_stride(primals_535, (), ())
    assert_size_stride(primals_536, (384, ), (1, ))
    assert_size_stride(primals_537, (384, ), (1, ))
    assert_size_stride(primals_538, (), ())
    assert_size_stride(primals_539, (448, ), (1, ))
    assert_size_stride(primals_540, (448, ), (1, ))
    assert_size_stride(primals_541, (), ())
    assert_size_stride(primals_542, (384, ), (1, ))
    assert_size_stride(primals_543, (384, ), (1, ))
    assert_size_stride(primals_544, (), ())
    assert_size_stride(primals_545, (384, ), (1, ))
    assert_size_stride(primals_546, (384, ), (1, ))
    assert_size_stride(primals_547, (), ())
    assert_size_stride(primals_548, (384, ), (1, ))
    assert_size_stride(primals_549, (384, ), (1, ))
    assert_size_stride(primals_550, (), ())
    assert_size_stride(primals_551, (192, ), (1, ))
    assert_size_stride(primals_552, (192, ), (1, ))
    assert_size_stride(primals_553, (), ())
    assert_size_stride(primals_554, (320, ), (1, ))
    assert_size_stride(primals_555, (320, ), (1, ))
    assert_size_stride(primals_556, (), ())
    assert_size_stride(primals_557, (384, ), (1, ))
    assert_size_stride(primals_558, (384, ), (1, ))
    assert_size_stride(primals_559, (), ())
    assert_size_stride(primals_560, (384, ), (1, ))
    assert_size_stride(primals_561, (384, ), (1, ))
    assert_size_stride(primals_562, (), ())
    assert_size_stride(primals_563, (384, ), (1, ))
    assert_size_stride(primals_564, (384, ), (1, ))
    assert_size_stride(primals_565, (), ())
    assert_size_stride(primals_566, (448, ), (1, ))
    assert_size_stride(primals_567, (448, ), (1, ))
    assert_size_stride(primals_568, (), ())
    assert_size_stride(primals_569, (384, ), (1, ))
    assert_size_stride(primals_570, (384, ), (1, ))
    assert_size_stride(primals_571, (), ())
    assert_size_stride(primals_572, (384, ), (1, ))
    assert_size_stride(primals_573, (384, ), (1, ))
    assert_size_stride(primals_574, (), ())
    assert_size_stride(primals_575, (384, ), (1, ))
    assert_size_stride(primals_576, (384, ), (1, ))
    assert_size_stride(primals_577, (), ())
    assert_size_stride(primals_578, (192, ), (1, ))
    assert_size_stride(primals_579, (192, ), (1, ))
    assert_size_stride(primals_580, (), ())
    assert_size_stride(primals_581, (32, 3, 299, 299), (268203, 89401, 299, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_1, buf0, 96, 9, grid=grid(96, 9), stream=stream0)
        del primals_1
        buf1 = empty_strided_cuda((32, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_4, buf1, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_4
        buf2 = empty_strided_cuda((64, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_7, buf2, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del primals_7
        buf3 = empty_strided_cuda((192, 80, 3, 3), (720, 1, 240, 80), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_13, buf3, 15360, 9, grid=grid(15360, 9), stream=stream0)
        del primals_13
        buf4 = empty_strided_cuda((64, 48, 5, 5), (1200, 1, 240, 48), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_22, buf4, 3072, 25, grid=grid(3072, 25), stream=stream0)
        del primals_22
        buf5 = empty_strided_cuda((96, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_28, buf5, 6144, 9, grid=grid(6144, 9), stream=stream0)
        del primals_28
        buf6 = empty_strided_cuda((96, 96, 3, 3), (864, 1, 288, 96), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_6.run(primals_31, buf6, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del primals_31
        buf7 = empty_strided_cuda((64, 48, 5, 5), (1200, 1, 240, 48), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_43, buf7, 3072, 25, grid=grid(3072, 25), stream=stream0)
        del primals_43
        buf8 = empty_strided_cuda((96, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_49, buf8, 6144, 9, grid=grid(6144, 9), stream=stream0)
        del primals_49
        buf9 = empty_strided_cuda((96, 96, 3, 3), (864, 1, 288, 96), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_6.run(primals_52, buf9, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del primals_52
        buf10 = empty_strided_cuda((64, 48, 5, 5), (1200, 1, 240, 48), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_64, buf10, 3072, 25, grid=grid(3072, 25), stream=stream0)
        del primals_64
        buf11 = empty_strided_cuda((96, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_70, buf11, 6144, 9, grid=grid(6144, 9), stream=stream0)
        del primals_70
        buf12 = empty_strided_cuda((96, 96, 3, 3), (864, 1, 288, 96), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_6.run(primals_73, buf12, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del primals_73
        buf13 = empty_strided_cuda((384, 288, 3, 3), (2592, 1, 864, 288), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_7.run(primals_79, buf13, 110592, 9, grid=grid(110592, 9), stream=stream0)
        del primals_79
        buf14 = empty_strided_cuda((96, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_85, buf14, 6144, 9, grid=grid(6144, 9), stream=stream0)
        del primals_85
        buf15 = empty_strided_cuda((96, 96, 3, 3), (864, 1, 288, 96), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_6.run(primals_88, buf15, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del primals_88
        buf16 = empty_strided_cuda((128, 128, 1, 7), (896, 1, 896, 128), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_8.run(primals_97, buf16, 16384, 7, grid=grid(16384, 7), stream=stream0)
        del primals_97
        buf17 = empty_strided_cuda((192, 128, 7, 1), (896, 1, 128, 128), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(primals_100, buf17, 24576, 7, grid=grid(24576, 7), stream=stream0)
        del primals_100
        buf18 = empty_strided_cuda((128, 128, 7, 1), (896, 1, 128, 128), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_8.run(primals_106, buf18, 16384, 7, grid=grid(16384, 7), stream=stream0)
        del primals_106
        buf19 = empty_strided_cuda((128, 128, 1, 7), (896, 1, 896, 128), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_8.run(primals_109, buf19, 16384, 7, grid=grid(16384, 7), stream=stream0)
        del primals_109
        buf20 = empty_strided_cuda((128, 128, 7, 1), (896, 1, 128, 128), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_8.run(primals_112, buf20, 16384, 7, grid=grid(16384, 7), stream=stream0)
        del primals_112
        buf21 = empty_strided_cuda((192, 128, 1, 7), (896, 1, 896, 128), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(primals_115, buf21, 24576, 7, grid=grid(24576, 7), stream=stream0)
        del primals_115
        buf22 = empty_strided_cuda((160, 160, 1, 7), (1120, 1, 1120, 160), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_10.run(primals_127, buf22, 25600, 7, grid=grid(25600, 7), stream=stream0)
        del primals_127
        buf23 = empty_strided_cuda((192, 160, 7, 1), (1120, 1, 160, 160), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_11.run(primals_130, buf23, 30720, 7, grid=grid(30720, 7), stream=stream0)
        del primals_130
        buf24 = empty_strided_cuda((160, 160, 7, 1), (1120, 1, 160, 160), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_10.run(primals_136, buf24, 25600, 7, grid=grid(25600, 7), stream=stream0)
        del primals_136
        buf25 = empty_strided_cuda((160, 160, 1, 7), (1120, 1, 1120, 160), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_10.run(primals_139, buf25, 25600, 7, grid=grid(25600, 7), stream=stream0)
        del primals_139
        buf26 = empty_strided_cuda((160, 160, 7, 1), (1120, 1, 160, 160), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_10.run(primals_142, buf26, 25600, 7, grid=grid(25600, 7), stream=stream0)
        del primals_142
        buf27 = empty_strided_cuda((192, 160, 1, 7), (1120, 1, 1120, 160), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_11.run(primals_145, buf27, 30720, 7, grid=grid(30720, 7), stream=stream0)
        del primals_145
        buf28 = empty_strided_cuda((160, 160, 1, 7), (1120, 1, 1120, 160), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_10.run(primals_157, buf28, 25600, 7, grid=grid(25600, 7), stream=stream0)
        del primals_157
        buf29 = empty_strided_cuda((192, 160, 7, 1), (1120, 1, 160, 160), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_11.run(primals_160, buf29, 30720, 7, grid=grid(30720, 7), stream=stream0)
        del primals_160
        buf30 = empty_strided_cuda((160, 160, 7, 1), (1120, 1, 160, 160), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_10.run(primals_166, buf30, 25600, 7, grid=grid(25600, 7), stream=stream0)
        del primals_166
        buf31 = empty_strided_cuda((160, 160, 1, 7), (1120, 1, 1120, 160), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_10.run(primals_169, buf31, 25600, 7, grid=grid(25600, 7), stream=stream0)
        del primals_169
        buf32 = empty_strided_cuda((160, 160, 7, 1), (1120, 1, 160, 160), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_10.run(primals_172, buf32, 25600, 7, grid=grid(25600, 7), stream=stream0)
        del primals_172
        buf33 = empty_strided_cuda((192, 160, 1, 7), (1120, 1, 1120, 160), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_11.run(primals_175, buf33, 30720, 7, grid=grid(30720, 7), stream=stream0)
        del primals_175
        buf34 = empty_strided_cuda((192, 192, 1, 7), (1344, 1, 1344, 192), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_12.run(primals_187, buf34, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del primals_187
        buf35 = empty_strided_cuda((192, 192, 7, 1), (1344, 1, 192, 192), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_12.run(primals_190, buf35, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del primals_190
        buf36 = empty_strided_cuda((192, 192, 7, 1), (1344, 1, 192, 192), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_12.run(primals_196, buf36, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del primals_196
        buf37 = empty_strided_cuda((192, 192, 1, 7), (1344, 1, 1344, 192), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_12.run(primals_199, buf37, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del primals_199
        buf38 = empty_strided_cuda((192, 192, 7, 1), (1344, 1, 192, 192), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_12.run(primals_202, buf38, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del primals_202
        buf39 = empty_strided_cuda((192, 192, 1, 7), (1344, 1, 1344, 192), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_12.run(primals_205, buf39, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del primals_205
        buf40 = empty_strided_cuda((768, 128, 5, 5), (3200, 1, 640, 128), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_13.run(primals_214, buf40, 98304, 25, grid=grid(98304, 25), stream=stream0)
        del primals_214
        buf41 = empty_strided_cuda((320, 192, 3, 3), (1728, 1, 576, 192), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(primals_222, buf41, 61440, 9, grid=grid(61440, 9), stream=stream0)
        del primals_222
        buf42 = empty_strided_cuda((192, 192, 1, 7), (1344, 1, 1344, 192), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_12.run(primals_228, buf42, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del primals_228
        buf43 = empty_strided_cuda((192, 192, 7, 1), (1344, 1, 192, 192), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_12.run(primals_231, buf43, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del primals_231
        buf44 = empty_strided_cuda((192, 192, 3, 3), (1728, 1, 576, 192), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_15.run(primals_234, buf44, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del primals_234
        buf45 = empty_strided_cuda((384, 384, 1, 3), (1152, 1, 1152, 384), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_16.run(primals_243, buf45, 147456, 3, grid=grid(147456, 3), stream=stream0)
        del primals_243
        buf46 = empty_strided_cuda((384, 384, 3, 1), (1152, 1, 384, 384), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_16.run(primals_246, buf46, 147456, 3, grid=grid(147456, 3), stream=stream0)
        del primals_246
        buf47 = empty_strided_cuda((384, 448, 3, 3), (4032, 1, 1344, 448), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_17.run(primals_252, buf47, 172032, 9, grid=grid(172032, 9), stream=stream0)
        del primals_252
        buf48 = empty_strided_cuda((384, 384, 1, 3), (1152, 1, 1152, 384), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_16.run(primals_255, buf48, 147456, 3, grid=grid(147456, 3), stream=stream0)
        del primals_255
        buf49 = empty_strided_cuda((384, 384, 3, 1), (1152, 1, 384, 384), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_16.run(primals_258, buf49, 147456, 3, grid=grid(147456, 3), stream=stream0)
        del primals_258
        buf50 = empty_strided_cuda((384, 384, 1, 3), (1152, 1, 1152, 384), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_16.run(primals_270, buf50, 147456, 3, grid=grid(147456, 3), stream=stream0)
        del primals_270
        buf51 = empty_strided_cuda((384, 384, 3, 1), (1152, 1, 384, 384), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_16.run(primals_273, buf51, 147456, 3, grid=grid(147456, 3), stream=stream0)
        del primals_273
        buf52 = empty_strided_cuda((384, 448, 3, 3), (4032, 1, 1344, 448), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_17.run(primals_279, buf52, 172032, 9, grid=grid(172032, 9), stream=stream0)
        del primals_279
        buf53 = empty_strided_cuda((384, 384, 1, 3), (1152, 1, 1152, 384), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_16.run(primals_282, buf53, 147456, 3, grid=grid(147456, 3), stream=stream0)
        del primals_282
        buf54 = empty_strided_cuda((384, 384, 3, 1), (1152, 1, 384, 384), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_16.run(primals_285, buf54, 147456, 3, grid=grid(147456, 3), stream=stream0)
        del primals_285
        buf55 = empty_strided_cuda((32, 3, 299, 299), (268203, 1, 897, 3), torch.float32)
        # Source Nodes: [cat_31], Original ATen: [aten.cat]
        triton_poi_fused_cat_18.run(primals_581, buf55, 8582496, grid=grid(8582496), stream=stream0)
        del primals_581
        # Source Nodes: [x_1], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, buf0, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (32, 32, 149, 149), (710432, 1, 4768, 32))
        buf57 = empty_strided_cuda((1, 32, 1, 1, 863), (27616, 1, 27616, 27616, 32), torch.float32)
        buf58 = empty_strided_cuda((1, 32, 1, 1, 863), (27616, 1, 27616, 27616, 32), torch.float32)
        buf59 = empty_strided_cuda((1, 32, 1, 1, 863), (27616, 1, 27616, 27616, 32), torch.float32)
        # Source Nodes: [x_2], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf56, buf57, buf58, buf59, 27616, 824, grid=grid(27616), stream=stream0)
        buf60 = empty_strided_cuda((1, 32, 1, 1, 7), (224, 1, 224, 224, 32), torch.float32)
        buf61 = empty_strided_cuda((1, 32, 1, 1, 7), (224, 1, 224, 224, 32), torch.float32)
        buf62 = empty_strided_cuda((1, 32, 1, 1, 7), (224, 1, 224, 224, 32), torch.float32)
        # Source Nodes: [x_2], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf57, buf58, buf59, buf60, buf61, buf62, 224, 124, grid=grid(224), stream=stream0)
        del buf57
        del buf58
        del buf59
        buf63 = empty_strided_cuda((1, 32, 1, 1), (32, 1, 32, 32), torch.float32)
        buf64 = empty_strided_cuda((1, 32, 1, 1), (32, 1, 32, 32), torch.float32)
        buf66 = empty_strided_cuda((1, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        # Source Nodes: [x_2], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_21.run(buf60, buf61, buf62, primals_293, primals_294, buf63, buf64, buf66, primals_293, primals_294, 32, 7, grid=grid(32), stream=stream0)
        del primals_293
        del primals_294
        buf67 = empty_strided_cuda((32, 32, 149, 149), (710432, 1, 4768, 32), torch.float32)
        # Source Nodes: [x_2, x_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_22.run(buf56, buf63, buf64, primals_2, primals_3, buf67, 22733824, grid=grid(22733824), stream=stream0)
        del primals_3
        # Source Nodes: [x_4], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf67, buf1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (32, 32, 147, 147), (691488, 1, 4704, 32))
        buf69 = empty_strided_cuda((1, 32, 1, 1, 882), (28224, 1, 28224, 28224, 32), torch.float32)
        buf70 = empty_strided_cuda((1, 32, 1, 1, 882), (28224, 1, 28224, 28224, 32), torch.float32)
        buf71 = empty_strided_cuda((1, 32, 1, 1, 882), (28224, 1, 28224, 28224, 32), torch.float32)
        # Source Nodes: [x_5], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_23.run(buf68, buf69, buf70, buf71, 28224, 784, grid=grid(28224), stream=stream0)
        buf72 = buf62; del buf62  # reuse
        buf73 = buf61; del buf61  # reuse
        buf74 = buf60; del buf60  # reuse
        # Source Nodes: [x_5], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_24.run(buf69, buf70, buf71, buf72, buf73, buf74, 224, 126, grid=grid(224), stream=stream0)
        del buf69
        del buf70
        del buf71
        buf75 = buf64; del buf64  # reuse
        buf76 = empty_strided_cuda((1, 32, 1, 1), (32, 1, 32, 32), torch.float32)
        buf78 = empty_strided_cuda((1, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        # Source Nodes: [x_5], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_25.run(buf72, buf73, buf74, primals_296, primals_297, buf75, buf76, buf78, primals_296, primals_297, 32, 7, grid=grid(32), stream=stream0)
        del buf72
        del buf73
        del buf74
        del primals_296
        del primals_297
        buf79 = empty_strided_cuda((32, 32, 147, 147), (691488, 1, 4704, 32), torch.float32)
        # Source Nodes: [x_5, x_6], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_26.run(buf68, buf75, buf76, primals_5, primals_6, buf79, 22127616, grid=grid(22127616), stream=stream0)
        del primals_6
        # Source Nodes: [x_7], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf79, buf2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (32, 64, 147, 147), (1382976, 1, 9408, 64))
        buf81 = empty_strided_cuda((1, 64, 1, 1, 882), (56448, 1, 56448, 56448, 64), torch.float32)
        buf82 = empty_strided_cuda((1, 64, 1, 1, 882), (56448, 1, 56448, 56448, 64), torch.float32)
        buf83 = empty_strided_cuda((1, 64, 1, 1, 882), (56448, 1, 56448, 56448, 64), torch.float32)
        # Source Nodes: [x_8], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_27.run(buf80, buf81, buf82, buf83, 56448, 784, grid=grid(56448), stream=stream0)
        buf84 = empty_strided_cuda((1, 64, 1, 1, 7), (448, 1, 448, 448, 64), torch.float32)
        buf85 = empty_strided_cuda((1, 64, 1, 1, 7), (448, 1, 448, 448, 64), torch.float32)
        buf86 = empty_strided_cuda((1, 64, 1, 1, 7), (448, 1, 448, 448, 64), torch.float32)
        # Source Nodes: [x_8], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_28.run(buf81, buf82, buf83, buf84, buf85, buf86, 448, 126, grid=grid(448), stream=stream0)
        del buf81
        del buf82
        del buf83
        buf87 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        buf88 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        buf90 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 1, 1), torch.float32)
        # Source Nodes: [x_8], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_29.run(buf84, buf85, buf86, primals_299, primals_300, buf87, buf88, buf90, primals_299, primals_300, 64, 7, grid=grid(64), stream=stream0)
        del primals_299
        del primals_300
        buf91 = empty_strided_cuda((32, 64, 147, 147), (1382976, 1, 9408, 64), torch.float32)
        # Source Nodes: [x_8, x_9], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_30.run(buf80, buf87, buf88, primals_8, primals_9, buf91, 44255232, grid=grid(44255232), stream=stream0)
        del primals_9
        buf92 = empty_strided_cuda((32, 64, 73, 73), (341056, 1, 4672, 64), torch.float32)
        buf93 = empty_strided_cuda((32, 64, 73, 73), (341056, 1, 4672, 64), torch.int64)
        # Source Nodes: [x_10], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_31.run(buf91, buf92, buf93, 10913792, grid=grid(10913792), stream=stream0)
        # Source Nodes: [x_11], Original ATen: [aten.convolution]
        buf94 = extern_kernels.convolution(buf92, primals_10, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf94, (32, 80, 73, 73), (426320, 1, 5840, 80))
        buf95 = empty_strided_cuda((1, 80, 1, 1, 1333), (106640, 1, 106640, 106640, 80), torch.float32)
        buf96 = empty_strided_cuda((1, 80, 1, 1, 1333), (106640, 1, 106640, 106640, 80), torch.float32)
        buf97 = empty_strided_cuda((1, 80, 1, 1, 1333), (106640, 1, 106640, 106640, 80), torch.float32)
        # Source Nodes: [x_12], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_32.run(buf94, buf95, buf96, buf97, 106640, 128, grid=grid(106640), stream=stream0)
        buf98 = empty_strided_cuda((1, 80, 1, 1, 11), (880, 1, 880, 880, 80), torch.float32)
        buf99 = empty_strided_cuda((1, 80, 1, 1, 11), (880, 1, 880, 880, 80), torch.float32)
        buf100 = empty_strided_cuda((1, 80, 1, 1, 11), (880, 1, 880, 880, 80), torch.float32)
        # Source Nodes: [x_12], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_33.run(buf95, buf96, buf97, buf98, buf99, buf100, 880, 122, grid=grid(880), stream=stream0)
        del buf95
        del buf96
        del buf97
        buf101 = empty_strided_cuda((1, 80, 1, 1), (80, 1, 80, 80), torch.float32)
        buf102 = empty_strided_cuda((1, 80, 1, 1), (80, 1, 80, 80), torch.float32)
        buf104 = empty_strided_cuda((1, 80, 1, 1), (80, 1, 1, 1), torch.float32)
        # Source Nodes: [x_12], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_34.run(buf98, buf99, buf100, primals_302, primals_303, buf101, buf102, buf104, primals_302, primals_303, 80, 11, grid=grid(80), stream=stream0)
        del buf100
        del buf98
        del buf99
        del primals_302
        del primals_303
        buf105 = empty_strided_cuda((32, 80, 73, 73), (426320, 1, 5840, 80), torch.float32)
        # Source Nodes: [x_12, x_13], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_35.run(buf94, buf101, buf102, primals_11, primals_12, buf105, 13642240, grid=grid(13642240), stream=stream0)
        del buf102
        del primals_12
        # Source Nodes: [x_14], Original ATen: [aten.convolution]
        buf106 = extern_kernels.convolution(buf105, buf3, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf106, (32, 192, 71, 71), (967872, 1, 13632, 192))
        buf107 = empty_strided_cuda((1, 192, 1, 1, 430), (82560, 1, 82560, 82560, 192), torch.float32)
        buf108 = empty_strided_cuda((1, 192, 1, 1, 430), (82560, 1, 82560, 82560, 192), torch.float32)
        buf109 = empty_strided_cuda((1, 192, 1, 1, 430), (82560, 1, 82560, 82560, 192), torch.float32)
        # Source Nodes: [x_15], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_36.run(buf106, buf107, buf108, buf109, 82560, 376, grid=grid(82560), stream=stream0)
        buf110 = empty_strided_cuda((1, 192, 1, 1, 4), (768, 1, 768, 768, 192), torch.float32)
        buf111 = empty_strided_cuda((1, 192, 1, 1, 4), (768, 1, 768, 768, 192), torch.float32)
        buf112 = empty_strided_cuda((1, 192, 1, 1, 4), (768, 1, 768, 768, 192), torch.float32)
        # Source Nodes: [x_15], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_37.run(buf107, buf108, buf109, buf110, buf111, buf112, 768, 108, grid=grid(768), stream=stream0)
        del buf107
        del buf108
        del buf109
        buf113 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 192, 192), torch.float32)
        buf114 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 192, 192), torch.float32)
        buf116 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 1, 1), torch.float32)
        # Source Nodes: [x_15], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_38.run(buf110, buf111, buf112, primals_305, primals_306, buf113, buf114, buf116, primals_305, primals_306, 192, 4, grid=grid(192), stream=stream0)
        del primals_305
        del primals_306
        buf117 = empty_strided_cuda((32, 192, 71, 71), (967872, 1, 13632, 192), torch.float32)
        # Source Nodes: [x_15, x_16], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_39.run(buf106, buf113, buf114, primals_14, primals_15, buf117, 30971904, grid=grid(30971904), stream=stream0)
        del primals_15
        buf118 = empty_strided_cuda((32, 192, 35, 35), (235200, 1, 6720, 192), torch.float32)
        buf119 = empty_strided_cuda((32, 192, 35, 35), (235200, 1, 6720, 192), torch.int64)
        # Source Nodes: [x_17], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_40.run(buf117, buf118, buf119, 7526400, grid=grid(7526400), stream=stream0)
        # Source Nodes: [x_18], Original ATen: [aten.convolution]
        buf120 = extern_kernels.convolution(buf118, primals_16, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf120, (32, 64, 35, 35), (78400, 1, 2240, 64))
        buf121 = empty_strided_cuda((1, 64, 1, 1, 307), (19648, 1, 19648, 19648, 64), torch.float32)
        buf122 = empty_strided_cuda((1, 64, 1, 1, 307), (19648, 1, 19648, 19648, 64), torch.float32)
        buf123 = empty_strided_cuda((1, 64, 1, 1, 307), (19648, 1, 19648, 19648, 64), torch.float32)
        # Source Nodes: [x_19], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_41.run(buf120, buf121, buf122, buf123, 19648, 128, grid=grid(19648), stream=stream0)
        buf124 = reinterpret_tensor(buf114, (1, 64, 1, 1, 3), (192, 1, 192, 192, 64), 0); del buf114  # reuse
        buf125 = empty_strided_cuda((1, 64, 1, 1, 3), (192, 1, 192, 192, 64), torch.float32)
        buf126 = empty_strided_cuda((1, 64, 1, 1, 3), (192, 1, 192, 192, 64), torch.float32)
        # Source Nodes: [x_19], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf121, buf122, buf123, buf124, buf125, buf126, 192, 103, grid=grid(192), stream=stream0)
        buf127 = buf88; del buf88  # reuse
        buf128 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        buf130 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 1, 1), torch.float32)
        # Source Nodes: [x_19], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf124, buf125, buf126, primals_308, primals_309, buf127, buf128, buf130, primals_308, primals_309, 64, 3, grid=grid(64), stream=stream0)
        del primals_308
        del primals_309
        buf205 = empty_strided_cuda((32, 256, 35, 35), (313600, 1225, 35, 1), torch.float32)
        buf131 = reinterpret_tensor(buf205, (32, 64, 35, 35), (313600, 1225, 35, 1), 0)  # alias
        buf1100 = empty_strided_cuda((32, 64, 35, 35), (78400, 1, 2240, 64), torch.bool)
        # Source Nodes: [branch1x1, x_19], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_44.run(buf120, buf127, buf128, primals_17, primals_18, buf131, buf1100, 2048, 1225, grid=grid(2048, 1225), stream=stream0)
        del primals_18
        # Source Nodes: [x_20], Original ATen: [aten.convolution]
        buf132 = extern_kernels.convolution(buf118, primals_19, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf132, (32, 48, 35, 35), (58800, 1, 1680, 48))
        buf133 = empty_strided_cuda((1, 48, 1, 1, 307), (14736, 1, 14736, 14736, 48), torch.float32)
        buf134 = empty_strided_cuda((1, 48, 1, 1, 307), (14736, 1, 14736, 14736, 48), torch.float32)
        buf135 = empty_strided_cuda((1, 48, 1, 1, 307), (14736, 1, 14736, 14736, 48), torch.float32)
        # Source Nodes: [x_21], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_45.run(buf132, buf133, buf134, buf135, 14736, 128, grid=grid(14736), stream=stream0)
        buf136 = empty_strided_cuda((1, 48, 1, 1, 3), (144, 1, 144, 144, 48), torch.float32)
        buf137 = empty_strided_cuda((1, 48, 1, 1, 3), (144, 1, 144, 144, 48), torch.float32)
        buf138 = empty_strided_cuda((1, 48, 1, 1, 3), (144, 1, 144, 144, 48), torch.float32)
        # Source Nodes: [x_21], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_46.run(buf133, buf134, buf135, buf136, buf137, buf138, 144, 103, grid=grid(144), stream=stream0)
        buf139 = empty_strided_cuda((1, 48, 1, 1), (48, 1, 48, 48), torch.float32)
        buf140 = empty_strided_cuda((1, 48, 1, 1), (48, 1, 48, 48), torch.float32)
        buf142 = empty_strided_cuda((1, 48, 1, 1), (48, 1, 1, 1), torch.float32)
        # Source Nodes: [x_21], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_47.run(buf136, buf137, buf138, primals_311, primals_312, buf139, buf140, buf142, primals_311, primals_312, 48, 3, grid=grid(48), stream=stream0)
        del primals_311
        del primals_312
        buf143 = empty_strided_cuda((32, 48, 35, 35), (58800, 1, 1680, 48), torch.float32)
        # Source Nodes: [branch5x5, x_21], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_48.run(buf132, buf139, buf140, primals_20, primals_21, buf143, 1881600, grid=grid(1881600), stream=stream0)
        del primals_21
        # Source Nodes: [x_22], Original ATen: [aten.convolution]
        buf144 = extern_kernels.convolution(buf143, buf4, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf144, (32, 64, 35, 35), (78400, 1, 2240, 64))
        buf145 = buf123; del buf123  # reuse
        buf146 = buf122; del buf122  # reuse
        buf147 = buf121; del buf121  # reuse
        # Source Nodes: [x_23], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_41.run(buf144, buf145, buf146, buf147, 19648, 128, grid=grid(19648), stream=stream0)
        buf148 = buf126; del buf126  # reuse
        buf149 = buf125; del buf125  # reuse
        buf150 = buf124; del buf124  # reuse
        # Source Nodes: [x_23], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf145, buf146, buf147, buf148, buf149, buf150, 192, 103, grid=grid(192), stream=stream0)
        buf151 = buf128; del buf128  # reuse
        buf152 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        buf154 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 1, 1), torch.float32)
        # Source Nodes: [x_23], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf148, buf149, buf150, primals_314, primals_315, buf151, buf152, buf154, primals_314, primals_315, 64, 3, grid=grid(64), stream=stream0)
        del primals_314
        del primals_315
        buf155 = reinterpret_tensor(buf205, (32, 64, 35, 35), (313600, 1225, 35, 1), 78400)  # alias
        buf1099 = empty_strided_cuda((32, 64, 35, 35), (78400, 1, 2240, 64), torch.bool)
        # Source Nodes: [branch5x5_1, x_23], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_44.run(buf144, buf151, buf152, primals_23, primals_24, buf155, buf1099, 2048, 1225, grid=grid(2048, 1225), stream=stream0)
        del primals_24
        # Source Nodes: [x_24], Original ATen: [aten.convolution]
        buf156 = extern_kernels.convolution(buf118, primals_25, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (32, 64, 35, 35), (78400, 1, 2240, 64))
        buf157 = buf147; del buf147  # reuse
        buf158 = buf146; del buf146  # reuse
        buf159 = buf145; del buf145  # reuse
        # Source Nodes: [x_25], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_41.run(buf156, buf157, buf158, buf159, 19648, 128, grid=grid(19648), stream=stream0)
        buf160 = buf150; del buf150  # reuse
        buf161 = buf149; del buf149  # reuse
        buf162 = buf148; del buf148  # reuse
        # Source Nodes: [x_25], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf157, buf158, buf159, buf160, buf161, buf162, 192, 103, grid=grid(192), stream=stream0)
        buf163 = buf152; del buf152  # reuse
        buf164 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        buf166 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 1, 1), torch.float32)
        # Source Nodes: [x_25], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf160, buf161, buf162, primals_317, primals_318, buf163, buf164, buf166, primals_317, primals_318, 64, 3, grid=grid(64), stream=stream0)
        del primals_317
        del primals_318
        buf167 = empty_strided_cuda((32, 64, 35, 35), (78400, 1, 2240, 64), torch.float32)
        # Source Nodes: [branch3x3dbl, x_25], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_49.run(buf156, buf163, buf164, primals_26, primals_27, buf167, 2508800, grid=grid(2508800), stream=stream0)
        del primals_27
        # Source Nodes: [x_26], Original ATen: [aten.convolution]
        buf168 = extern_kernels.convolution(buf167, buf5, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf168, (32, 96, 35, 35), (117600, 1, 3360, 96))
        buf169 = empty_strided_cuda((1, 96, 1, 1, 307), (29472, 1, 29472, 29472, 96), torch.float32)
        buf170 = empty_strided_cuda((1, 96, 1, 1, 307), (29472, 1, 29472, 29472, 96), torch.float32)
        buf171 = empty_strided_cuda((1, 96, 1, 1, 307), (29472, 1, 29472, 29472, 96), torch.float32)
        # Source Nodes: [x_27], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_50.run(buf168, buf169, buf170, buf171, 29472, 128, grid=grid(29472), stream=stream0)
        buf172 = empty_strided_cuda((1, 96, 1, 1, 3), (288, 1, 288, 288, 96), torch.float32)
        buf173 = empty_strided_cuda((1, 96, 1, 1, 3), (288, 1, 288, 288, 96), torch.float32)
        buf174 = empty_strided_cuda((1, 96, 1, 1, 3), (288, 1, 288, 288, 96), torch.float32)
        # Source Nodes: [x_27], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf169, buf170, buf171, buf172, buf173, buf174, 288, 103, grid=grid(288), stream=stream0)
        buf175 = empty_strided_cuda((1, 96, 1, 1), (96, 1, 96, 96), torch.float32)
        buf176 = empty_strided_cuda((1, 96, 1, 1), (96, 1, 96, 96), torch.float32)
        buf178 = empty_strided_cuda((1, 96, 1, 1), (96, 1, 1, 1), torch.float32)
        # Source Nodes: [x_27], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_52.run(buf172, buf173, buf174, primals_320, primals_321, buf175, buf176, buf178, primals_320, primals_321, 96, 3, grid=grid(96), stream=stream0)
        del primals_320
        del primals_321
        buf179 = empty_strided_cuda((32, 96, 35, 35), (117600, 1, 3360, 96), torch.float32)
        # Source Nodes: [branch3x3dbl_1, x_27], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_53.run(buf168, buf175, buf176, primals_29, primals_30, buf179, 3763200, grid=grid(3763200), stream=stream0)
        del primals_30
        # Source Nodes: [x_28], Original ATen: [aten.convolution]
        buf180 = extern_kernels.convolution(buf179, buf6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf180, (32, 96, 35, 35), (117600, 1, 3360, 96))
        buf181 = buf171; del buf171  # reuse
        buf182 = buf170; del buf170  # reuse
        buf183 = buf169; del buf169  # reuse
        # Source Nodes: [x_29], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_50.run(buf180, buf181, buf182, buf183, 29472, 128, grid=grid(29472), stream=stream0)
        buf184 = buf174; del buf174  # reuse
        buf185 = buf173; del buf173  # reuse
        buf186 = buf172; del buf172  # reuse
        # Source Nodes: [x_29], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf181, buf182, buf183, buf184, buf185, buf186, 288, 103, grid=grid(288), stream=stream0)
        buf187 = buf176; del buf176  # reuse
        buf188 = empty_strided_cuda((1, 96, 1, 1), (96, 1, 96, 96), torch.float32)
        buf190 = empty_strided_cuda((1, 96, 1, 1), (96, 1, 1, 1), torch.float32)
        # Source Nodes: [x_29], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_52.run(buf184, buf185, buf186, primals_323, primals_324, buf187, buf188, buf190, primals_323, primals_324, 96, 3, grid=grid(96), stream=stream0)
        del primals_323
        del primals_324
        buf191 = reinterpret_tensor(buf205, (32, 96, 35, 35), (313600, 1225, 35, 1), 156800)  # alias
        buf1098 = empty_strided_cuda((32, 96, 35, 35), (117600, 1, 3360, 96), torch.bool)
        # Source Nodes: [branch3x3dbl_2, x_29], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_54.run(buf180, buf187, buf188, primals_32, primals_33, buf191, buf1098, 3072, 1225, grid=grid(3072, 1225), stream=stream0)
        del primals_33
        buf192 = empty_strided_cuda((32, 192, 35, 35), (235200, 1, 6720, 192), torch.float32)
        # Source Nodes: [branch_pool], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_55.run(buf118, buf192, 6144, 1225, grid=grid(6144, 1225), stream=stream0)
        # Source Nodes: [x_30], Original ATen: [aten.convolution]
        buf193 = extern_kernels.convolution(buf192, primals_34, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf193, (32, 32, 35, 35), (39200, 1, 1120, 32))
        buf194 = empty_strided_cuda((1, 32, 1, 1, 307), (9824, 1, 9824, 9824, 32), torch.float32)
        buf195 = empty_strided_cuda((1, 32, 1, 1, 307), (9824, 1, 9824, 9824, 32), torch.float32)
        buf196 = empty_strided_cuda((1, 32, 1, 1, 307), (9824, 1, 9824, 9824, 32), torch.float32)
        # Source Nodes: [x_31], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_56.run(buf193, buf194, buf195, buf196, 9824, 128, grid=grid(9824), stream=stream0)
        buf197 = reinterpret_tensor(buf188, (1, 32, 1, 1, 3), (96, 1, 96, 96, 32), 0); del buf188  # reuse
        buf198 = empty_strided_cuda((1, 32, 1, 1, 3), (96, 1, 96, 96, 32), torch.float32)
        buf199 = empty_strided_cuda((1, 32, 1, 1, 3), (96, 1, 96, 96, 32), torch.float32)
        # Source Nodes: [x_31], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_57.run(buf194, buf195, buf196, buf197, buf198, buf199, 96, 103, grid=grid(96), stream=stream0)
        del buf194
        del buf195
        del buf196
        buf200 = buf76; del buf76  # reuse
        buf201 = empty_strided_cuda((1, 32, 1, 1), (32, 1, 32, 32), torch.float32)
        buf203 = empty_strided_cuda((1, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        # Source Nodes: [x_31], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_58.run(buf197, buf198, buf199, primals_326, primals_327, buf200, buf201, buf203, primals_326, primals_327, 32, 3, grid=grid(32), stream=stream0)
        del primals_326
        del primals_327
        buf204 = reinterpret_tensor(buf205, (32, 32, 35, 35), (313600, 1225, 35, 1), 274400)  # alias
        buf1097 = empty_strided_cuda((32, 32, 35, 35), (39200, 1, 1120, 32), torch.bool)
        # Source Nodes: [branch_pool_1, x_31], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_59.run(buf193, buf200, buf201, primals_35, primals_36, buf204, buf1097, 1024, 1225, grid=grid(1024, 1225), stream=stream0)
        del buf201
        del primals_36
        buf206 = empty_strided_cuda((32, 256, 35, 35), (313600, 1, 8960, 256), torch.float32)
        # Source Nodes: [cat_30], Original ATen: [aten.cat]
        triton_poi_fused_cat_60.run(buf205, buf206, 8192, 1225, grid=grid(8192, 1225), stream=stream0)
        del buf131
        del buf155
        del buf191
        del buf204
        # Source Nodes: [x_33], Original ATen: [aten.convolution]
        buf207 = extern_kernels.convolution(buf206, primals_37, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf207, (32, 64, 35, 35), (78400, 1, 2240, 64))
        buf208 = buf159; del buf159  # reuse
        buf209 = buf158; del buf158  # reuse
        buf210 = buf157; del buf157  # reuse
        # Source Nodes: [x_34], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_41.run(buf207, buf208, buf209, buf210, 19648, 128, grid=grid(19648), stream=stream0)
        buf211 = buf162; del buf162  # reuse
        buf212 = buf161; del buf161  # reuse
        buf213 = buf160; del buf160  # reuse
        # Source Nodes: [x_34], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf208, buf209, buf210, buf211, buf212, buf213, 192, 103, grid=grid(192), stream=stream0)
        buf214 = buf164; del buf164  # reuse
        buf215 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        buf217 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 1, 1), torch.float32)
        # Source Nodes: [x_34], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf211, buf212, buf213, primals_329, primals_330, buf214, buf215, buf217, primals_329, primals_330, 64, 3, grid=grid(64), stream=stream0)
        del primals_329
        del primals_330
        buf292 = empty_strided_cuda((32, 288, 35, 35), (352800, 1225, 35, 1), torch.float32)
        buf218 = reinterpret_tensor(buf292, (32, 64, 35, 35), (352800, 1225, 35, 1), 0)  # alias
        buf1096 = empty_strided_cuda((32, 64, 35, 35), (78400, 1, 2240, 64), torch.bool)
        # Source Nodes: [branch1x1_1, x_34], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf207, buf214, buf215, primals_38, primals_39, buf218, buf1096, 2048, 1225, grid=grid(2048, 1225), stream=stream0)
        del primals_39
        # Source Nodes: [x_35], Original ATen: [aten.convolution]
        buf219 = extern_kernels.convolution(buf206, primals_40, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf219, (32, 48, 35, 35), (58800, 1, 1680, 48))
        buf220 = buf135; del buf135  # reuse
        buf221 = buf134; del buf134  # reuse
        buf222 = buf133; del buf133  # reuse
        # Source Nodes: [x_36], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_45.run(buf219, buf220, buf221, buf222, 14736, 128, grid=grid(14736), stream=stream0)
        buf223 = buf138; del buf138  # reuse
        buf224 = buf137; del buf137  # reuse
        buf225 = buf136; del buf136  # reuse
        # Source Nodes: [x_36], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_46.run(buf220, buf221, buf222, buf223, buf224, buf225, 144, 103, grid=grid(144), stream=stream0)
        buf226 = buf140; del buf140  # reuse
        buf227 = empty_strided_cuda((1, 48, 1, 1), (48, 1, 48, 48), torch.float32)
        buf229 = empty_strided_cuda((1, 48, 1, 1), (48, 1, 1, 1), torch.float32)
        # Source Nodes: [x_36], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_47.run(buf223, buf224, buf225, primals_332, primals_333, buf226, buf227, buf229, primals_332, primals_333, 48, 3, grid=grid(48), stream=stream0)
        del primals_332
        del primals_333
        buf230 = empty_strided_cuda((32, 48, 35, 35), (58800, 1, 1680, 48), torch.float32)
        # Source Nodes: [branch5x5_2, x_36], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_48.run(buf219, buf226, buf227, primals_41, primals_42, buf230, 1881600, grid=grid(1881600), stream=stream0)
        del primals_42
        # Source Nodes: [x_37], Original ATen: [aten.convolution]
        buf231 = extern_kernels.convolution(buf230, buf7, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf231, (32, 64, 35, 35), (78400, 1, 2240, 64))
        buf232 = buf210; del buf210  # reuse
        buf233 = buf209; del buf209  # reuse
        buf234 = buf208; del buf208  # reuse
        # Source Nodes: [x_38], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_41.run(buf231, buf232, buf233, buf234, 19648, 128, grid=grid(19648), stream=stream0)
        buf235 = buf213; del buf213  # reuse
        buf236 = buf212; del buf212  # reuse
        buf237 = buf211; del buf211  # reuse
        # Source Nodes: [x_38], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf232, buf233, buf234, buf235, buf236, buf237, 192, 103, grid=grid(192), stream=stream0)
        buf238 = buf215; del buf215  # reuse
        buf239 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        buf241 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 1, 1), torch.float32)
        # Source Nodes: [x_38], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf235, buf236, buf237, primals_335, primals_336, buf238, buf239, buf241, primals_335, primals_336, 64, 3, grid=grid(64), stream=stream0)
        del primals_335
        del primals_336
        buf242 = reinterpret_tensor(buf292, (32, 64, 35, 35), (352800, 1225, 35, 1), 78400)  # alias
        buf1095 = empty_strided_cuda((32, 64, 35, 35), (78400, 1, 2240, 64), torch.bool)
        # Source Nodes: [branch5x5_3, x_38], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf231, buf238, buf239, primals_44, primals_45, buf242, buf1095, 2048, 1225, grid=grid(2048, 1225), stream=stream0)
        del primals_45
        # Source Nodes: [x_39], Original ATen: [aten.convolution]
        buf243 = extern_kernels.convolution(buf206, primals_46, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf243, (32, 64, 35, 35), (78400, 1, 2240, 64))
        buf244 = buf234; del buf234  # reuse
        buf245 = buf233; del buf233  # reuse
        buf246 = buf232; del buf232  # reuse
        # Source Nodes: [x_40], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_41.run(buf243, buf244, buf245, buf246, 19648, 128, grid=grid(19648), stream=stream0)
        buf247 = buf237; del buf237  # reuse
        buf248 = buf236; del buf236  # reuse
        buf249 = buf235; del buf235  # reuse
        # Source Nodes: [x_40], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf244, buf245, buf246, buf247, buf248, buf249, 192, 103, grid=grid(192), stream=stream0)
        buf250 = buf239; del buf239  # reuse
        buf251 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        buf253 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 1, 1), torch.float32)
        # Source Nodes: [x_40], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf247, buf248, buf249, primals_338, primals_339, buf250, buf251, buf253, primals_338, primals_339, 64, 3, grid=grid(64), stream=stream0)
        del primals_338
        del primals_339
        buf254 = empty_strided_cuda((32, 64, 35, 35), (78400, 1, 2240, 64), torch.float32)
        # Source Nodes: [branch3x3dbl_3, x_40], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_49.run(buf243, buf250, buf251, primals_47, primals_48, buf254, 2508800, grid=grid(2508800), stream=stream0)
        del primals_48
        # Source Nodes: [x_41], Original ATen: [aten.convolution]
        buf255 = extern_kernels.convolution(buf254, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf255, (32, 96, 35, 35), (117600, 1, 3360, 96))
        buf256 = buf183; del buf183  # reuse
        buf257 = buf182; del buf182  # reuse
        buf258 = buf181; del buf181  # reuse
        # Source Nodes: [x_42], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_50.run(buf255, buf256, buf257, buf258, 29472, 128, grid=grid(29472), stream=stream0)
        buf259 = buf186; del buf186  # reuse
        buf260 = buf185; del buf185  # reuse
        buf261 = buf184; del buf184  # reuse
        # Source Nodes: [x_42], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf256, buf257, buf258, buf259, buf260, buf261, 288, 103, grid=grid(288), stream=stream0)
        buf262 = reinterpret_tensor(buf199, (1, 96, 1, 1), (96, 1, 96, 96), 0); del buf199  # reuse
        buf263 = reinterpret_tensor(buf198, (1, 96, 1, 1), (96, 1, 96, 96), 0); del buf198  # reuse
        buf265 = reinterpret_tensor(buf197, (1, 96, 1, 1), (96, 1, 1, 1), 0); del buf197  # reuse
        # Source Nodes: [x_42], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_52.run(buf259, buf260, buf261, primals_341, primals_342, buf262, buf263, buf265, primals_341, primals_342, 96, 3, grid=grid(96), stream=stream0)
        del primals_341
        del primals_342
        buf266 = empty_strided_cuda((32, 96, 35, 35), (117600, 1, 3360, 96), torch.float32)
        # Source Nodes: [branch3x3dbl_4, x_42], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_53.run(buf255, buf262, buf263, primals_50, primals_51, buf266, 3763200, grid=grid(3763200), stream=stream0)
        del primals_51
        # Source Nodes: [x_43], Original ATen: [aten.convolution]
        buf267 = extern_kernels.convolution(buf266, buf9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf267, (32, 96, 35, 35), (117600, 1, 3360, 96))
        buf268 = buf258; del buf258  # reuse
        buf269 = buf257; del buf257  # reuse
        buf270 = buf256; del buf256  # reuse
        # Source Nodes: [x_44], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_50.run(buf267, buf268, buf269, buf270, 29472, 128, grid=grid(29472), stream=stream0)
        buf271 = buf261; del buf261  # reuse
        buf272 = buf260; del buf260  # reuse
        buf273 = buf259; del buf259  # reuse
        # Source Nodes: [x_44], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf268, buf269, buf270, buf271, buf272, buf273, 288, 103, grid=grid(288), stream=stream0)
        buf274 = buf263; del buf263  # reuse
        buf275 = empty_strided_cuda((1, 96, 1, 1), (96, 1, 96, 96), torch.float32)
        buf277 = empty_strided_cuda((1, 96, 1, 1), (96, 1, 1, 1), torch.float32)
        # Source Nodes: [x_44], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_52.run(buf271, buf272, buf273, primals_344, primals_345, buf274, buf275, buf277, primals_344, primals_345, 96, 3, grid=grid(96), stream=stream0)
        del primals_344
        del primals_345
        buf278 = reinterpret_tensor(buf292, (32, 96, 35, 35), (352800, 1225, 35, 1), 156800)  # alias
        buf1094 = empty_strided_cuda((32, 96, 35, 35), (117600, 1, 3360, 96), torch.bool)
        # Source Nodes: [branch3x3dbl_5, x_44], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_62.run(buf267, buf274, buf275, primals_53, primals_54, buf278, buf1094, 3072, 1225, grid=grid(3072, 1225), stream=stream0)
        del primals_54
        buf279 = reinterpret_tensor(buf205, (32, 256, 35, 35), (313600, 1, 8960, 256), 0); del buf205  # reuse
        # Source Nodes: [branch_pool_2], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_63.run(buf206, buf279, 10035200, grid=grid(10035200), stream=stream0)
        # Source Nodes: [x_45], Original ATen: [aten.convolution]
        buf280 = extern_kernels.convolution(buf279, primals_55, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf280, (32, 64, 35, 35), (78400, 1, 2240, 64))
        buf281 = buf246; del buf246  # reuse
        buf282 = buf245; del buf245  # reuse
        buf283 = buf244; del buf244  # reuse
        # Source Nodes: [x_46], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_41.run(buf280, buf281, buf282, buf283, 19648, 128, grid=grid(19648), stream=stream0)
        buf284 = buf249; del buf249  # reuse
        buf285 = buf248; del buf248  # reuse
        buf286 = buf247; del buf247  # reuse
        # Source Nodes: [x_46], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf281, buf282, buf283, buf284, buf285, buf286, 192, 103, grid=grid(192), stream=stream0)
        buf287 = buf251; del buf251  # reuse
        buf288 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        buf290 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 1, 1), torch.float32)
        # Source Nodes: [x_46], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf284, buf285, buf286, primals_347, primals_348, buf287, buf288, buf290, primals_347, primals_348, 64, 3, grid=grid(64), stream=stream0)
        del primals_347
        del primals_348
        buf291 = reinterpret_tensor(buf292, (32, 64, 35, 35), (352800, 1225, 35, 1), 274400)  # alias
        buf1093 = empty_strided_cuda((32, 64, 35, 35), (78400, 1, 2240, 64), torch.bool)
        # Source Nodes: [branch_pool_3, x_46], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf280, buf287, buf288, primals_56, primals_57, buf291, buf1093, 2048, 1225, grid=grid(2048, 1225), stream=stream0)
        del primals_57
        buf293 = empty_strided_cuda((32, 288, 35, 35), (352800, 1, 10080, 288), torch.float32)
        # Source Nodes: [cat_29], Original ATen: [aten.cat]
        triton_poi_fused_cat_64.run(buf292, buf293, 9216, 1225, grid=grid(9216, 1225), stream=stream0)
        del buf218
        del buf242
        del buf278
        del buf291
        # Source Nodes: [x_48], Original ATen: [aten.convolution]
        buf294 = extern_kernels.convolution(buf293, primals_58, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf294, (32, 64, 35, 35), (78400, 1, 2240, 64))
        buf295 = buf283; del buf283  # reuse
        buf296 = buf282; del buf282  # reuse
        buf297 = buf281; del buf281  # reuse
        # Source Nodes: [x_49], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_41.run(buf294, buf295, buf296, buf297, 19648, 128, grid=grid(19648), stream=stream0)
        buf298 = buf286; del buf286  # reuse
        buf299 = buf285; del buf285  # reuse
        buf300 = buf284; del buf284  # reuse
        # Source Nodes: [x_49], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf295, buf296, buf297, buf298, buf299, buf300, 192, 103, grid=grid(192), stream=stream0)
        buf301 = buf288; del buf288  # reuse
        buf302 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        buf304 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 1, 1), torch.float32)
        # Source Nodes: [x_49], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf298, buf299, buf300, primals_350, primals_351, buf301, buf302, buf304, primals_350, primals_351, 64, 3, grid=grid(64), stream=stream0)
        del primals_350
        del primals_351
        buf379 = buf292; del buf292  # reuse
        buf305 = reinterpret_tensor(buf379, (32, 64, 35, 35), (352800, 1225, 35, 1), 0)  # alias
        buf1092 = empty_strided_cuda((32, 64, 35, 35), (78400, 1, 2240, 64), torch.bool)
        # Source Nodes: [branch1x1_2, x_49], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf294, buf301, buf302, primals_59, primals_60, buf305, buf1092, 2048, 1225, grid=grid(2048, 1225), stream=stream0)
        del primals_60
        # Source Nodes: [x_50], Original ATen: [aten.convolution]
        buf306 = extern_kernels.convolution(buf293, primals_61, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf306, (32, 48, 35, 35), (58800, 1, 1680, 48))
        buf307 = buf222; del buf222  # reuse
        buf308 = buf221; del buf221  # reuse
        buf309 = buf220; del buf220  # reuse
        # Source Nodes: [x_51], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_45.run(buf306, buf307, buf308, buf309, 14736, 128, grid=grid(14736), stream=stream0)
        buf310 = buf225; del buf225  # reuse
        buf311 = buf224; del buf224  # reuse
        buf312 = buf223; del buf223  # reuse
        # Source Nodes: [x_51], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_46.run(buf307, buf308, buf309, buf310, buf311, buf312, 144, 103, grid=grid(144), stream=stream0)
        del buf307
        del buf308
        del buf309
        buf313 = buf227; del buf227  # reuse
        buf314 = empty_strided_cuda((1, 48, 1, 1), (48, 1, 48, 48), torch.float32)
        buf316 = empty_strided_cuda((1, 48, 1, 1), (48, 1, 1, 1), torch.float32)
        # Source Nodes: [x_51], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_47.run(buf310, buf311, buf312, primals_353, primals_354, buf313, buf314, buf316, primals_353, primals_354, 48, 3, grid=grid(48), stream=stream0)
        del buf310
        del buf311
        del buf312
        del primals_353
        del primals_354
        buf317 = empty_strided_cuda((32, 48, 35, 35), (58800, 1, 1680, 48), torch.float32)
        # Source Nodes: [branch5x5_4, x_51], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_48.run(buf306, buf313, buf314, primals_62, primals_63, buf317, 1881600, grid=grid(1881600), stream=stream0)
        del buf314
        del primals_63
        # Source Nodes: [x_52], Original ATen: [aten.convolution]
        buf318 = extern_kernels.convolution(buf317, buf10, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf318, (32, 64, 35, 35), (78400, 1, 2240, 64))
        buf319 = buf297; del buf297  # reuse
        buf320 = buf296; del buf296  # reuse
        buf321 = buf295; del buf295  # reuse
        # Source Nodes: [x_53], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_41.run(buf318, buf319, buf320, buf321, 19648, 128, grid=grid(19648), stream=stream0)
        buf322 = buf300; del buf300  # reuse
        buf323 = buf299; del buf299  # reuse
        buf324 = buf298; del buf298  # reuse
        # Source Nodes: [x_53], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf319, buf320, buf321, buf322, buf323, buf324, 192, 103, grid=grid(192), stream=stream0)
        buf325 = buf302; del buf302  # reuse
        buf326 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        buf328 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 1, 1), torch.float32)
        # Source Nodes: [x_53], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf322, buf323, buf324, primals_356, primals_357, buf325, buf326, buf328, primals_356, primals_357, 64, 3, grid=grid(64), stream=stream0)
        del primals_356
        del primals_357
        buf329 = reinterpret_tensor(buf379, (32, 64, 35, 35), (352800, 1225, 35, 1), 78400)  # alias
        buf1091 = empty_strided_cuda((32, 64, 35, 35), (78400, 1, 2240, 64), torch.bool)
        # Source Nodes: [branch5x5_5, x_53], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf318, buf325, buf326, primals_65, primals_66, buf329, buf1091, 2048, 1225, grid=grid(2048, 1225), stream=stream0)
        del primals_66
        # Source Nodes: [x_54], Original ATen: [aten.convolution]
        buf330 = extern_kernels.convolution(buf293, primals_67, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf330, (32, 64, 35, 35), (78400, 1, 2240, 64))
        buf331 = buf321; del buf321  # reuse
        buf332 = buf320; del buf320  # reuse
        buf333 = buf319; del buf319  # reuse
        # Source Nodes: [x_55], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_41.run(buf330, buf331, buf332, buf333, 19648, 128, grid=grid(19648), stream=stream0)
        buf334 = buf324; del buf324  # reuse
        buf335 = buf323; del buf323  # reuse
        buf336 = buf322; del buf322  # reuse
        # Source Nodes: [x_55], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf331, buf332, buf333, buf334, buf335, buf336, 192, 103, grid=grid(192), stream=stream0)
        buf337 = buf326; del buf326  # reuse
        buf338 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        buf340 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 1, 1), torch.float32)
        # Source Nodes: [x_55], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf334, buf335, buf336, primals_359, primals_360, buf337, buf338, buf340, primals_359, primals_360, 64, 3, grid=grid(64), stream=stream0)
        del primals_359
        del primals_360
        buf341 = empty_strided_cuda((32, 64, 35, 35), (78400, 1, 2240, 64), torch.float32)
        # Source Nodes: [branch3x3dbl_6, x_55], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_49.run(buf330, buf337, buf338, primals_68, primals_69, buf341, 2508800, grid=grid(2508800), stream=stream0)
        del primals_69
        # Source Nodes: [x_56], Original ATen: [aten.convolution]
        buf342 = extern_kernels.convolution(buf341, buf11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf342, (32, 96, 35, 35), (117600, 1, 3360, 96))
        buf343 = buf270; del buf270  # reuse
        buf344 = buf269; del buf269  # reuse
        buf345 = buf268; del buf268  # reuse
        # Source Nodes: [x_57], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_50.run(buf342, buf343, buf344, buf345, 29472, 128, grid=grid(29472), stream=stream0)
        buf346 = buf273; del buf273  # reuse
        buf347 = buf272; del buf272  # reuse
        buf348 = buf271; del buf271  # reuse
        # Source Nodes: [x_57], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf343, buf344, buf345, buf346, buf347, buf348, 288, 103, grid=grid(288), stream=stream0)
        buf349 = buf275; del buf275  # reuse
        buf350 = empty_strided_cuda((1, 96, 1, 1), (96, 1, 96, 96), torch.float32)
        buf352 = empty_strided_cuda((1, 96, 1, 1), (96, 1, 1, 1), torch.float32)
        # Source Nodes: [x_57], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_52.run(buf346, buf347, buf348, primals_362, primals_363, buf349, buf350, buf352, primals_362, primals_363, 96, 3, grid=grid(96), stream=stream0)
        del primals_362
        del primals_363
        buf353 = empty_strided_cuda((32, 96, 35, 35), (117600, 1, 3360, 96), torch.float32)
        # Source Nodes: [branch3x3dbl_7, x_57], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_53.run(buf342, buf349, buf350, primals_71, primals_72, buf353, 3763200, grid=grid(3763200), stream=stream0)
        del primals_72
        # Source Nodes: [x_58], Original ATen: [aten.convolution]
        buf354 = extern_kernels.convolution(buf353, buf12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf354, (32, 96, 35, 35), (117600, 1, 3360, 96))
        buf355 = buf345; del buf345  # reuse
        buf356 = buf344; del buf344  # reuse
        buf357 = buf343; del buf343  # reuse
        # Source Nodes: [x_59], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_50.run(buf354, buf355, buf356, buf357, 29472, 128, grid=grid(29472), stream=stream0)
        buf358 = buf348; del buf348  # reuse
        buf359 = buf347; del buf347  # reuse
        buf360 = buf346; del buf346  # reuse
        # Source Nodes: [x_59], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf355, buf356, buf357, buf358, buf359, buf360, 288, 103, grid=grid(288), stream=stream0)
        buf361 = buf350; del buf350  # reuse
        buf362 = empty_strided_cuda((1, 96, 1, 1), (96, 1, 96, 96), torch.float32)
        buf364 = empty_strided_cuda((1, 96, 1, 1), (96, 1, 1, 1), torch.float32)
        # Source Nodes: [x_59], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_52.run(buf358, buf359, buf360, primals_365, primals_366, buf361, buf362, buf364, primals_365, primals_366, 96, 3, grid=grid(96), stream=stream0)
        del primals_365
        del primals_366
        buf365 = reinterpret_tensor(buf379, (32, 96, 35, 35), (352800, 1225, 35, 1), 156800)  # alias
        buf1090 = empty_strided_cuda((32, 96, 35, 35), (117600, 1, 3360, 96), torch.bool)
        # Source Nodes: [branch3x3dbl_8, x_59], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_62.run(buf354, buf361, buf362, primals_74, primals_75, buf365, buf1090, 3072, 1225, grid=grid(3072, 1225), stream=stream0)
        del primals_75
        buf366 = empty_strided_cuda((32, 288, 35, 35), (352800, 1, 10080, 288), torch.float32)
        # Source Nodes: [branch_pool_4], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_65.run(buf293, buf366, 11289600, grid=grid(11289600), stream=stream0)
        # Source Nodes: [x_60], Original ATen: [aten.convolution]
        buf367 = extern_kernels.convolution(buf366, primals_76, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf367, (32, 64, 35, 35), (78400, 1, 2240, 64))
        buf368 = buf333; del buf333  # reuse
        buf369 = buf332; del buf332  # reuse
        buf370 = buf331; del buf331  # reuse
        # Source Nodes: [x_61], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_41.run(buf367, buf368, buf369, buf370, 19648, 128, grid=grid(19648), stream=stream0)
        buf371 = buf336; del buf336  # reuse
        buf372 = buf335; del buf335  # reuse
        buf373 = buf334; del buf334  # reuse
        # Source Nodes: [x_61], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf368, buf369, buf370, buf371, buf372, buf373, 192, 103, grid=grid(192), stream=stream0)
        buf374 = buf338; del buf338  # reuse
        buf375 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        buf377 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 1, 1), torch.float32)
        # Source Nodes: [x_61], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf371, buf372, buf373, primals_368, primals_369, buf374, buf375, buf377, primals_368, primals_369, 64, 3, grid=grid(64), stream=stream0)
        del primals_368
        del primals_369
        buf378 = reinterpret_tensor(buf379, (32, 64, 35, 35), (352800, 1225, 35, 1), 274400)  # alias
        buf1089 = empty_strided_cuda((32, 64, 35, 35), (78400, 1, 2240, 64), torch.bool)
        # Source Nodes: [branch_pool_5, x_61], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_61.run(buf367, buf374, buf375, primals_77, primals_78, buf378, buf1089, 2048, 1225, grid=grid(2048, 1225), stream=stream0)
        del primals_78
        buf380 = empty_strided_cuda((32, 288, 35, 35), (352800, 1, 10080, 288), torch.float32)
        # Source Nodes: [cat_28], Original ATen: [aten.cat]
        triton_poi_fused_cat_64.run(buf379, buf380, 9216, 1225, grid=grid(9216, 1225), stream=stream0)
        del buf305
        del buf329
        del buf365
        del buf378
        del buf379
        # Source Nodes: [x_63], Original ATen: [aten.convolution]
        buf381 = extern_kernels.convolution(buf380, buf13, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf381, (32, 384, 17, 17), (110976, 1, 6528, 384))
        buf382 = empty_strided_cuda((1, 384, 1, 1, 73), (28032, 1, 28032, 28032, 384), torch.float32)
        buf383 = empty_strided_cuda((1, 384, 1, 1, 73), (28032, 1, 28032, 28032, 384), torch.float32)
        buf384 = empty_strided_cuda((1, 384, 1, 1, 73), (28032, 1, 28032, 28032, 384), torch.float32)
        # Source Nodes: [x_64], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_66.run(buf381, buf382, buf383, buf384, 28032, 127, grid=grid(28032), stream=stream0)
        buf385 = empty_strided_cuda((1, 384, 1, 1), (384, 1, 384, 384), torch.float32)
        buf386 = empty_strided_cuda((1, 384, 1, 1), (384, 1, 384, 384), torch.float32)
        buf388 = empty_strided_cuda((1, 384, 1, 1), (384, 1, 1, 1), torch.float32)
        # Source Nodes: [x_64], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf382, buf383, buf384, primals_371, primals_372, buf385, buf386, buf388, primals_371, primals_372, 384, 73, grid=grid(384), stream=stream0)
        del buf382
        del buf383
        del buf384
        del primals_371
        del primals_372
        buf425 = empty_strided_cuda((32, 768, 17, 17), (221952, 289, 17, 1), torch.float32)
        buf389 = reinterpret_tensor(buf425, (32, 384, 17, 17), (221952, 289, 17, 1), 0)  # alias
        buf1088 = empty_strided_cuda((32, 384, 17, 17), (110976, 1, 6528, 384), torch.bool)
        # Source Nodes: [branch3x3, x_64], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_68.run(buf381, buf385, buf386, primals_80, primals_81, buf389, buf1088, 12288, 289, grid=grid(12288, 289), stream=stream0)
        del primals_81
        # Source Nodes: [x_65], Original ATen: [aten.convolution]
        buf390 = extern_kernels.convolution(buf380, primals_82, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf390, (32, 64, 35, 35), (78400, 1, 2240, 64))
        buf391 = buf370; del buf370  # reuse
        buf392 = buf369; del buf369  # reuse
        buf393 = buf368; del buf368  # reuse
        # Source Nodes: [x_66], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_41.run(buf390, buf391, buf392, buf393, 19648, 128, grid=grid(19648), stream=stream0)
        buf394 = buf373; del buf373  # reuse
        buf395 = buf372; del buf372  # reuse
        buf396 = buf371; del buf371  # reuse
        # Source Nodes: [x_66], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf391, buf392, buf393, buf394, buf395, buf396, 192, 103, grid=grid(192), stream=stream0)
        del buf391
        del buf392
        del buf393
        buf397 = buf375; del buf375  # reuse
        buf398 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        buf400 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 1, 1), torch.float32)
        # Source Nodes: [x_66], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf394, buf395, buf396, primals_374, primals_375, buf397, buf398, buf400, primals_374, primals_375, 64, 3, grid=grid(64), stream=stream0)
        del primals_374
        del primals_375
        buf401 = empty_strided_cuda((32, 64, 35, 35), (78400, 1, 2240, 64), torch.float32)
        # Source Nodes: [branch3x3dbl_9, x_66], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_49.run(buf390, buf397, buf398, primals_83, primals_84, buf401, 2508800, grid=grid(2508800), stream=stream0)
        del buf398
        del primals_84
        # Source Nodes: [x_67], Original ATen: [aten.convolution]
        buf402 = extern_kernels.convolution(buf401, buf14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf402, (32, 96, 35, 35), (117600, 1, 3360, 96))
        buf403 = buf357; del buf357  # reuse
        buf404 = buf356; del buf356  # reuse
        buf405 = buf355; del buf355  # reuse
        # Source Nodes: [x_68], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_50.run(buf402, buf403, buf404, buf405, 29472, 128, grid=grid(29472), stream=stream0)
        buf406 = buf360; del buf360  # reuse
        buf407 = buf359; del buf359  # reuse
        buf408 = buf358; del buf358  # reuse
        # Source Nodes: [x_68], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf403, buf404, buf405, buf406, buf407, buf408, 288, 103, grid=grid(288), stream=stream0)
        del buf403
        del buf404
        del buf405
        buf409 = buf362; del buf362  # reuse
        buf410 = empty_strided_cuda((1, 96, 1, 1), (96, 1, 96, 96), torch.float32)
        buf412 = empty_strided_cuda((1, 96, 1, 1), (96, 1, 1, 1), torch.float32)
        # Source Nodes: [x_68], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_52.run(buf406, buf407, buf408, primals_377, primals_378, buf409, buf410, buf412, primals_377, primals_378, 96, 3, grid=grid(96), stream=stream0)
        del buf406
        del buf407
        del buf408
        del primals_377
        del primals_378
        buf413 = empty_strided_cuda((32, 96, 35, 35), (117600, 1, 3360, 96), torch.float32)
        # Source Nodes: [branch3x3dbl_10, x_68], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_53.run(buf402, buf409, buf410, primals_86, primals_87, buf413, 3763200, grid=grid(3763200), stream=stream0)
        del primals_87
        # Source Nodes: [x_69], Original ATen: [aten.convolution]
        buf414 = extern_kernels.convolution(buf413, buf15, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf414, (32, 96, 17, 17), (27744, 1, 1632, 96))
        buf415 = empty_strided_cuda((1, 96, 1, 1, 73), (7008, 1, 7008, 7008, 96), torch.float32)
        buf416 = empty_strided_cuda((1, 96, 1, 1, 73), (7008, 1, 7008, 7008, 96), torch.float32)
        buf417 = empty_strided_cuda((1, 96, 1, 1, 73), (7008, 1, 7008, 7008, 96), torch.float32)
        # Source Nodes: [x_70], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_69.run(buf414, buf415, buf416, buf417, 7008, 127, grid=grid(7008), stream=stream0)
        buf418 = buf410; del buf410  # reuse
        buf419 = empty_strided_cuda((1, 96, 1, 1), (96, 1, 96, 96), torch.float32)
        buf421 = empty_strided_cuda((1, 96, 1, 1), (96, 1, 1, 1), torch.float32)
        # Source Nodes: [x_70], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_70.run(buf415, buf416, buf417, primals_380, primals_381, buf418, buf419, buf421, primals_380, primals_381, 96, 73, grid=grid(96), stream=stream0)
        del buf415
        del buf416
        del buf417
        del primals_380
        del primals_381
        buf422 = reinterpret_tensor(buf425, (32, 96, 17, 17), (221952, 289, 17, 1), 110976)  # alias
        buf1087 = empty_strided_cuda((32, 96, 17, 17), (27744, 1, 1632, 96), torch.bool)
        # Source Nodes: [branch3x3dbl_11, x_70], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_71.run(buf414, buf418, buf419, primals_89, primals_90, buf422, buf1087, 3072, 289, grid=grid(3072, 289), stream=stream0)
        del buf419
        del primals_90
        buf423 = reinterpret_tensor(buf425, (32, 288, 17, 17), (221952, 289, 17, 1), 138720)  # alias
        # Source Nodes: [branch_pool_6], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_72.run(buf380, buf423, 9216, 289, grid=grid(9216, 289), stream=stream0)
        buf424 = empty_strided_cuda((32, 288, 17, 17), (83232, 1, 4896, 288), torch.int64)
        # Source Nodes: [branch_pool_6], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_73.run(buf380, buf424, 2663424, grid=grid(2663424), stream=stream0)
        buf426 = empty_strided_cuda((32, 768, 17, 17), (221952, 1, 13056, 768), torch.float32)
        # Source Nodes: [cat_27], Original ATen: [aten.cat]
        triton_poi_fused_cat_74.run(buf425, buf426, 24576, 289, grid=grid(24576, 289), stream=stream0)
        del buf389
        del buf422
        del buf423
        # Source Nodes: [x_72], Original ATen: [aten.convolution]
        buf427 = extern_kernels.convolution(buf426, primals_91, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf427, (32, 192, 17, 17), (55488, 1, 3264, 192))
        buf428 = empty_strided_cuda((1, 192, 1, 1, 73), (14016, 1, 14016, 14016, 192), torch.float32)
        buf429 = empty_strided_cuda((1, 192, 1, 1, 73), (14016, 1, 14016, 14016, 192), torch.float32)
        buf430 = empty_strided_cuda((1, 192, 1, 1, 73), (14016, 1, 14016, 14016, 192), torch.float32)
        # Source Nodes: [x_73], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf427, buf428, buf429, buf430, 14016, 127, grid=grid(14016), stream=stream0)
        buf431 = reinterpret_tensor(buf396, (1, 192, 1, 1), (192, 1, 192, 192), 0); del buf396  # reuse
        buf432 = reinterpret_tensor(buf395, (1, 192, 1, 1), (192, 1, 192, 192), 0); del buf395  # reuse
        buf434 = reinterpret_tensor(buf394, (1, 192, 1, 1), (192, 1, 1, 1), 0); del buf394  # reuse
        # Source Nodes: [x_73], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_76.run(buf428, buf429, buf430, primals_383, primals_384, buf431, buf432, buf434, primals_383, primals_384, 192, 73, grid=grid(192), stream=stream0)
        del primals_383
        del primals_384
        buf518 = buf425; del buf425  # reuse
        buf435 = reinterpret_tensor(buf518, (32, 192, 17, 17), (221952, 289, 17, 1), 0)  # alias
        buf1086 = empty_strided_cuda((32, 192, 17, 17), (55488, 1, 3264, 192), torch.bool)
        # Source Nodes: [branch1x1_3, x_73], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_77.run(buf427, buf431, buf432, primals_92, primals_93, buf435, buf1086, 6144, 289, grid=grid(6144, 289), stream=stream0)
        del primals_93
        # Source Nodes: [x_74], Original ATen: [aten.convolution]
        buf436 = extern_kernels.convolution(buf426, primals_94, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf436, (32, 128, 17, 17), (36992, 1, 2176, 128))
        buf437 = empty_strided_cuda((1, 128, 1, 1, 73), (9344, 1, 9344, 9344, 128), torch.float32)
        buf438 = empty_strided_cuda((1, 128, 1, 1, 73), (9344, 1, 9344, 9344, 128), torch.float32)
        buf439 = empty_strided_cuda((1, 128, 1, 1, 73), (9344, 1, 9344, 9344, 128), torch.float32)
        # Source Nodes: [x_75], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_78.run(buf436, buf437, buf438, buf439, 9344, 127, grid=grid(9344), stream=stream0)
        buf440 = empty_strided_cuda((1, 128, 1, 1), (128, 1, 128, 128), torch.float32)
        buf441 = empty_strided_cuda((1, 128, 1, 1), (128, 1, 128, 128), torch.float32)
        buf443 = empty_strided_cuda((1, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        # Source Nodes: [x_75], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_79.run(buf437, buf438, buf439, primals_386, primals_387, buf440, buf441, buf443, primals_386, primals_387, 128, 73, grid=grid(128), stream=stream0)
        del primals_386
        del primals_387
        buf444 = empty_strided_cuda((32, 128, 17, 17), (36992, 1, 2176, 128), torch.float32)
        # Source Nodes: [branch7x7, x_75], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_80.run(buf436, buf440, buf441, primals_95, primals_96, buf444, 1183744, grid=grid(1183744), stream=stream0)
        del primals_96
        # Source Nodes: [x_76], Original ATen: [aten.convolution]
        buf445 = extern_kernels.convolution(buf444, buf16, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf445, (32, 128, 17, 17), (36992, 1, 2176, 128))
        buf446 = buf439; del buf439  # reuse
        buf447 = buf438; del buf438  # reuse
        buf448 = buf437; del buf437  # reuse
        # Source Nodes: [x_77], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_78.run(buf445, buf446, buf447, buf448, 9344, 127, grid=grid(9344), stream=stream0)
        buf449 = buf441; del buf441  # reuse
        buf450 = empty_strided_cuda((1, 128, 1, 1), (128, 1, 128, 128), torch.float32)
        buf452 = empty_strided_cuda((1, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        # Source Nodes: [x_77], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_79.run(buf446, buf447, buf448, primals_389, primals_390, buf449, buf450, buf452, primals_389, primals_390, 128, 73, grid=grid(128), stream=stream0)
        del primals_389
        del primals_390
        buf453 = empty_strided_cuda((32, 128, 17, 17), (36992, 1, 2176, 128), torch.float32)
        # Source Nodes: [branch7x7_1, x_77], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_80.run(buf445, buf449, buf450, primals_98, primals_99, buf453, 1183744, grid=grid(1183744), stream=stream0)
        del primals_99
        # Source Nodes: [x_78], Original ATen: [aten.convolution]
        buf454 = extern_kernels.convolution(buf453, buf17, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf454, (32, 192, 17, 17), (55488, 1, 3264, 192))
        buf455 = buf430; del buf430  # reuse
        buf456 = buf429; del buf429  # reuse
        buf457 = buf428; del buf428  # reuse
        # Source Nodes: [x_79], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf454, buf455, buf456, buf457, 14016, 127, grid=grid(14016), stream=stream0)
        buf458 = buf432; del buf432  # reuse
        buf459 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 192, 192), torch.float32)
        buf461 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 1, 1), torch.float32)
        # Source Nodes: [x_79], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_76.run(buf455, buf456, buf457, primals_392, primals_393, buf458, buf459, buf461, primals_392, primals_393, 192, 73, grid=grid(192), stream=stream0)
        del primals_392
        del primals_393
        buf462 = reinterpret_tensor(buf518, (32, 192, 17, 17), (221952, 289, 17, 1), 55488)  # alias
        buf1085 = empty_strided_cuda((32, 192, 17, 17), (55488, 1, 3264, 192), torch.bool)
        # Source Nodes: [branch7x7_2, x_79], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_77.run(buf454, buf458, buf459, primals_101, primals_102, buf462, buf1085, 6144, 289, grid=grid(6144, 289), stream=stream0)
        del primals_102
        # Source Nodes: [x_80], Original ATen: [aten.convolution]
        buf463 = extern_kernels.convolution(buf426, primals_103, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf463, (32, 128, 17, 17), (36992, 1, 2176, 128))
        buf464 = buf448; del buf448  # reuse
        buf465 = buf447; del buf447  # reuse
        buf466 = buf446; del buf446  # reuse
        # Source Nodes: [x_81], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_78.run(buf463, buf464, buf465, buf466, 9344, 127, grid=grid(9344), stream=stream0)
        buf467 = buf450; del buf450  # reuse
        buf468 = empty_strided_cuda((1, 128, 1, 1), (128, 1, 128, 128), torch.float32)
        buf470 = empty_strided_cuda((1, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        # Source Nodes: [x_81], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_79.run(buf464, buf465, buf466, primals_395, primals_396, buf467, buf468, buf470, primals_395, primals_396, 128, 73, grid=grid(128), stream=stream0)
        del primals_395
        del primals_396
        buf471 = empty_strided_cuda((32, 128, 17, 17), (36992, 1, 2176, 128), torch.float32)
        # Source Nodes: [branch7x7dbl, x_81], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_80.run(buf463, buf467, buf468, primals_104, primals_105, buf471, 1183744, grid=grid(1183744), stream=stream0)
        del primals_105
        # Source Nodes: [x_82], Original ATen: [aten.convolution]
        buf472 = extern_kernels.convolution(buf471, buf18, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf472, (32, 128, 17, 17), (36992, 1, 2176, 128))
        buf473 = buf466; del buf466  # reuse
        buf474 = buf465; del buf465  # reuse
        buf475 = buf464; del buf464  # reuse
        # Source Nodes: [x_83], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_78.run(buf472, buf473, buf474, buf475, 9344, 127, grid=grid(9344), stream=stream0)
        buf476 = buf468; del buf468  # reuse
        buf477 = empty_strided_cuda((1, 128, 1, 1), (128, 1, 128, 128), torch.float32)
        buf479 = empty_strided_cuda((1, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        # Source Nodes: [x_83], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_79.run(buf473, buf474, buf475, primals_398, primals_399, buf476, buf477, buf479, primals_398, primals_399, 128, 73, grid=grid(128), stream=stream0)
        del primals_398
        del primals_399
        buf480 = empty_strided_cuda((32, 128, 17, 17), (36992, 1, 2176, 128), torch.float32)
        # Source Nodes: [branch7x7dbl_1, x_83], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_80.run(buf472, buf476, buf477, primals_107, primals_108, buf480, 1183744, grid=grid(1183744), stream=stream0)
        del primals_108
        # Source Nodes: [x_84], Original ATen: [aten.convolution]
        buf481 = extern_kernels.convolution(buf480, buf19, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf481, (32, 128, 17, 17), (36992, 1, 2176, 128))
        buf482 = buf475; del buf475  # reuse
        buf483 = buf474; del buf474  # reuse
        buf484 = buf473; del buf473  # reuse
        # Source Nodes: [x_85], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_78.run(buf481, buf482, buf483, buf484, 9344, 127, grid=grid(9344), stream=stream0)
        buf485 = buf477; del buf477  # reuse
        buf486 = empty_strided_cuda((1, 128, 1, 1), (128, 1, 128, 128), torch.float32)
        buf488 = empty_strided_cuda((1, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        # Source Nodes: [x_85], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_79.run(buf482, buf483, buf484, primals_401, primals_402, buf485, buf486, buf488, primals_401, primals_402, 128, 73, grid=grid(128), stream=stream0)
        del primals_401
        del primals_402
        buf489 = empty_strided_cuda((32, 128, 17, 17), (36992, 1, 2176, 128), torch.float32)
        # Source Nodes: [branch7x7dbl_2, x_85], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_80.run(buf481, buf485, buf486, primals_110, primals_111, buf489, 1183744, grid=grid(1183744), stream=stream0)
        del primals_111
        # Source Nodes: [x_86], Original ATen: [aten.convolution]
        buf490 = extern_kernels.convolution(buf489, buf20, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf490, (32, 128, 17, 17), (36992, 1, 2176, 128))
        buf491 = buf484; del buf484  # reuse
        buf492 = buf483; del buf483  # reuse
        buf493 = buf482; del buf482  # reuse
        # Source Nodes: [x_87], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_78.run(buf490, buf491, buf492, buf493, 9344, 127, grid=grid(9344), stream=stream0)
        buf494 = buf486; del buf486  # reuse
        buf495 = empty_strided_cuda((1, 128, 1, 1), (128, 1, 128, 128), torch.float32)
        buf497 = empty_strided_cuda((1, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        # Source Nodes: [x_87], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_79.run(buf491, buf492, buf493, primals_404, primals_405, buf494, buf495, buf497, primals_404, primals_405, 128, 73, grid=grid(128), stream=stream0)
        del buf491
        del buf492
        del buf493
        del primals_404
        del primals_405
        buf498 = empty_strided_cuda((32, 128, 17, 17), (36992, 1, 2176, 128), torch.float32)
        # Source Nodes: [branch7x7dbl_3, x_87], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_80.run(buf490, buf494, buf495, primals_113, primals_114, buf498, 1183744, grid=grid(1183744), stream=stream0)
        del primals_114
        # Source Nodes: [x_88], Original ATen: [aten.convolution]
        buf499 = extern_kernels.convolution(buf498, buf21, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf499, (32, 192, 17, 17), (55488, 1, 3264, 192))
        buf500 = buf457; del buf457  # reuse
        buf501 = buf456; del buf456  # reuse
        buf502 = buf455; del buf455  # reuse
        # Source Nodes: [x_89], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf499, buf500, buf501, buf502, 14016, 127, grid=grid(14016), stream=stream0)
        buf503 = buf459; del buf459  # reuse
        buf504 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 192, 192), torch.float32)
        buf506 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 1, 1), torch.float32)
        # Source Nodes: [x_89], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_76.run(buf500, buf501, buf502, primals_407, primals_408, buf503, buf504, buf506, primals_407, primals_408, 192, 73, grid=grid(192), stream=stream0)
        del primals_407
        del primals_408
        buf507 = reinterpret_tensor(buf518, (32, 192, 17, 17), (221952, 289, 17, 1), 110976)  # alias
        buf1084 = empty_strided_cuda((32, 192, 17, 17), (55488, 1, 3264, 192), torch.bool)
        # Source Nodes: [branch7x7dbl_4, x_89], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_77.run(buf499, buf503, buf504, primals_116, primals_117, buf507, buf1084, 6144, 289, grid=grid(6144, 289), stream=stream0)
        del primals_117
        buf508 = empty_strided_cuda((32, 768, 17, 17), (221952, 1, 13056, 768), torch.float32)
        # Source Nodes: [branch_pool_7], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_81.run(buf426, buf508, 7102464, grid=grid(7102464), stream=stream0)
        # Source Nodes: [x_90], Original ATen: [aten.convolution]
        buf509 = extern_kernels.convolution(buf508, primals_118, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf509, (32, 192, 17, 17), (55488, 1, 3264, 192))
        buf510 = buf502; del buf502  # reuse
        buf511 = buf501; del buf501  # reuse
        buf512 = buf500; del buf500  # reuse
        # Source Nodes: [x_91], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf509, buf510, buf511, buf512, 14016, 127, grid=grid(14016), stream=stream0)
        buf513 = buf504; del buf504  # reuse
        buf514 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 192, 192), torch.float32)
        buf516 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 1, 1), torch.float32)
        # Source Nodes: [x_91], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_76.run(buf510, buf511, buf512, primals_410, primals_411, buf513, buf514, buf516, primals_410, primals_411, 192, 73, grid=grid(192), stream=stream0)
        del primals_410
        del primals_411
        buf517 = reinterpret_tensor(buf518, (32, 192, 17, 17), (221952, 289, 17, 1), 166464)  # alias
        buf1083 = empty_strided_cuda((32, 192, 17, 17), (55488, 1, 3264, 192), torch.bool)
        # Source Nodes: [branch_pool_8, x_91], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_77.run(buf509, buf513, buf514, primals_119, primals_120, buf517, buf1083, 6144, 289, grid=grid(6144, 289), stream=stream0)
        del primals_120
        buf519 = empty_strided_cuda((32, 768, 17, 17), (221952, 1, 13056, 768), torch.float32)
        # Source Nodes: [cat_26], Original ATen: [aten.cat]
        triton_poi_fused_cat_74.run(buf518, buf519, 24576, 289, grid=grid(24576, 289), stream=stream0)
        del buf435
        del buf462
        del buf507
        del buf517
        # Source Nodes: [x_93], Original ATen: [aten.convolution]
        buf520 = extern_kernels.convolution(buf519, primals_121, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf520, (32, 192, 17, 17), (55488, 1, 3264, 192))
        buf521 = buf512; del buf512  # reuse
        buf522 = buf511; del buf511  # reuse
        buf523 = buf510; del buf510  # reuse
        # Source Nodes: [x_94], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf520, buf521, buf522, buf523, 14016, 127, grid=grid(14016), stream=stream0)
        buf524 = buf514; del buf514  # reuse
        buf525 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 192, 192), torch.float32)
        buf527 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 1, 1), torch.float32)
        # Source Nodes: [x_94], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_76.run(buf521, buf522, buf523, primals_413, primals_414, buf524, buf525, buf527, primals_413, primals_414, 192, 73, grid=grid(192), stream=stream0)
        del primals_413
        del primals_414
        buf611 = buf518; del buf518  # reuse
        buf528 = reinterpret_tensor(buf611, (32, 192, 17, 17), (221952, 289, 17, 1), 0)  # alias
        buf1082 = empty_strided_cuda((32, 192, 17, 17), (55488, 1, 3264, 192), torch.bool)
        # Source Nodes: [branch1x1_4, x_94], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_77.run(buf520, buf524, buf525, primals_122, primals_123, buf528, buf1082, 6144, 289, grid=grid(6144, 289), stream=stream0)
        del primals_123
        # Source Nodes: [x_95], Original ATen: [aten.convolution]
        buf529 = extern_kernels.convolution(buf519, primals_124, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf529, (32, 160, 17, 17), (46240, 1, 2720, 160))
        buf530 = empty_strided_cuda((1, 160, 1, 1, 73), (11680, 1, 11680, 11680, 160), torch.float32)
        buf531 = empty_strided_cuda((1, 160, 1, 1, 73), (11680, 1, 11680, 11680, 160), torch.float32)
        buf532 = empty_strided_cuda((1, 160, 1, 1, 73), (11680, 1, 11680, 11680, 160), torch.float32)
        # Source Nodes: [x_96], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_82.run(buf529, buf530, buf531, buf532, 11680, 127, grid=grid(11680), stream=stream0)
        buf533 = empty_strided_cuda((1, 160, 1, 1), (160, 1, 160, 160), torch.float32)
        buf534 = empty_strided_cuda((1, 160, 1, 1), (160, 1, 160, 160), torch.float32)
        buf536 = empty_strided_cuda((1, 160, 1, 1), (160, 1, 1, 1), torch.float32)
        # Source Nodes: [x_96], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_83.run(buf530, buf531, buf532, primals_416, primals_417, buf533, buf534, buf536, primals_416, primals_417, 160, 73, grid=grid(160), stream=stream0)
        del primals_416
        del primals_417
        buf537 = empty_strided_cuda((32, 160, 17, 17), (46240, 1, 2720, 160), torch.float32)
        # Source Nodes: [branch7x7_3, x_96], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_84.run(buf529, buf533, buf534, primals_125, primals_126, buf537, 1479680, grid=grid(1479680), stream=stream0)
        del primals_126
        # Source Nodes: [x_97], Original ATen: [aten.convolution]
        buf538 = extern_kernels.convolution(buf537, buf22, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf538, (32, 160, 17, 17), (46240, 1, 2720, 160))
        buf539 = buf532; del buf532  # reuse
        buf540 = buf531; del buf531  # reuse
        buf541 = buf530; del buf530  # reuse
        # Source Nodes: [x_98], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_82.run(buf538, buf539, buf540, buf541, 11680, 127, grid=grid(11680), stream=stream0)
        buf542 = buf534; del buf534  # reuse
        buf543 = empty_strided_cuda((1, 160, 1, 1), (160, 1, 160, 160), torch.float32)
        buf545 = empty_strided_cuda((1, 160, 1, 1), (160, 1, 1, 1), torch.float32)
        # Source Nodes: [x_98], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_83.run(buf539, buf540, buf541, primals_419, primals_420, buf542, buf543, buf545, primals_419, primals_420, 160, 73, grid=grid(160), stream=stream0)
        del primals_419
        del primals_420
        buf546 = empty_strided_cuda((32, 160, 17, 17), (46240, 1, 2720, 160), torch.float32)
        # Source Nodes: [branch7x7_4, x_98], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_84.run(buf538, buf542, buf543, primals_128, primals_129, buf546, 1479680, grid=grid(1479680), stream=stream0)
        del primals_129
        # Source Nodes: [x_99], Original ATen: [aten.convolution]
        buf547 = extern_kernels.convolution(buf546, buf23, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf547, (32, 192, 17, 17), (55488, 1, 3264, 192))
        buf548 = buf523; del buf523  # reuse
        buf549 = buf522; del buf522  # reuse
        buf550 = buf521; del buf521  # reuse
        # Source Nodes: [x_100], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf547, buf548, buf549, buf550, 14016, 127, grid=grid(14016), stream=stream0)
        buf551 = buf525; del buf525  # reuse
        buf552 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 192, 192), torch.float32)
        buf554 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 1, 1), torch.float32)
        # Source Nodes: [x_100], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_76.run(buf548, buf549, buf550, primals_422, primals_423, buf551, buf552, buf554, primals_422, primals_423, 192, 73, grid=grid(192), stream=stream0)
        del primals_422
        del primals_423
        buf555 = reinterpret_tensor(buf611, (32, 192, 17, 17), (221952, 289, 17, 1), 55488)  # alias
        buf1081 = empty_strided_cuda((32, 192, 17, 17), (55488, 1, 3264, 192), torch.bool)
        # Source Nodes: [branch7x7_5, x_100], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_77.run(buf547, buf551, buf552, primals_131, primals_132, buf555, buf1081, 6144, 289, grid=grid(6144, 289), stream=stream0)
        del primals_132
        # Source Nodes: [x_101], Original ATen: [aten.convolution]
        buf556 = extern_kernels.convolution(buf519, primals_133, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf556, (32, 160, 17, 17), (46240, 1, 2720, 160))
        buf557 = buf541; del buf541  # reuse
        buf558 = buf540; del buf540  # reuse
        buf559 = buf539; del buf539  # reuse
        # Source Nodes: [x_102], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_82.run(buf556, buf557, buf558, buf559, 11680, 127, grid=grid(11680), stream=stream0)
        buf560 = buf543; del buf543  # reuse
        buf561 = empty_strided_cuda((1, 160, 1, 1), (160, 1, 160, 160), torch.float32)
        buf563 = empty_strided_cuda((1, 160, 1, 1), (160, 1, 1, 1), torch.float32)
        # Source Nodes: [x_102], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_83.run(buf557, buf558, buf559, primals_425, primals_426, buf560, buf561, buf563, primals_425, primals_426, 160, 73, grid=grid(160), stream=stream0)
        del primals_425
        del primals_426
        buf564 = empty_strided_cuda((32, 160, 17, 17), (46240, 1, 2720, 160), torch.float32)
        # Source Nodes: [branch7x7dbl_5, x_102], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_84.run(buf556, buf560, buf561, primals_134, primals_135, buf564, 1479680, grid=grid(1479680), stream=stream0)
        del primals_135
        # Source Nodes: [x_103], Original ATen: [aten.convolution]
        buf565 = extern_kernels.convolution(buf564, buf24, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf565, (32, 160, 17, 17), (46240, 1, 2720, 160))
        buf566 = buf559; del buf559  # reuse
        buf567 = buf558; del buf558  # reuse
        buf568 = buf557; del buf557  # reuse
        # Source Nodes: [x_104], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_82.run(buf565, buf566, buf567, buf568, 11680, 127, grid=grid(11680), stream=stream0)
        buf569 = buf561; del buf561  # reuse
        buf570 = empty_strided_cuda((1, 160, 1, 1), (160, 1, 160, 160), torch.float32)
        buf572 = empty_strided_cuda((1, 160, 1, 1), (160, 1, 1, 1), torch.float32)
        # Source Nodes: [x_104], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_83.run(buf566, buf567, buf568, primals_428, primals_429, buf569, buf570, buf572, primals_428, primals_429, 160, 73, grid=grid(160), stream=stream0)
        del primals_428
        del primals_429
        buf573 = empty_strided_cuda((32, 160, 17, 17), (46240, 1, 2720, 160), torch.float32)
        # Source Nodes: [branch7x7dbl_6, x_104], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_84.run(buf565, buf569, buf570, primals_137, primals_138, buf573, 1479680, grid=grid(1479680), stream=stream0)
        del primals_138
        # Source Nodes: [x_105], Original ATen: [aten.convolution]
        buf574 = extern_kernels.convolution(buf573, buf25, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf574, (32, 160, 17, 17), (46240, 1, 2720, 160))
        buf575 = buf568; del buf568  # reuse
        buf576 = buf567; del buf567  # reuse
        buf577 = buf566; del buf566  # reuse
        # Source Nodes: [x_106], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_82.run(buf574, buf575, buf576, buf577, 11680, 127, grid=grid(11680), stream=stream0)
        buf578 = buf570; del buf570  # reuse
        buf579 = empty_strided_cuda((1, 160, 1, 1), (160, 1, 160, 160), torch.float32)
        buf581 = empty_strided_cuda((1, 160, 1, 1), (160, 1, 1, 1), torch.float32)
        # Source Nodes: [x_106], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_83.run(buf575, buf576, buf577, primals_431, primals_432, buf578, buf579, buf581, primals_431, primals_432, 160, 73, grid=grid(160), stream=stream0)
        del primals_431
        del primals_432
        buf582 = empty_strided_cuda((32, 160, 17, 17), (46240, 1, 2720, 160), torch.float32)
        # Source Nodes: [branch7x7dbl_7, x_106], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_84.run(buf574, buf578, buf579, primals_140, primals_141, buf582, 1479680, grid=grid(1479680), stream=stream0)
        del primals_141
        # Source Nodes: [x_107], Original ATen: [aten.convolution]
        buf583 = extern_kernels.convolution(buf582, buf26, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf583, (32, 160, 17, 17), (46240, 1, 2720, 160))
        buf584 = buf577; del buf577  # reuse
        buf585 = buf576; del buf576  # reuse
        buf586 = buf575; del buf575  # reuse
        # Source Nodes: [x_108], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_82.run(buf583, buf584, buf585, buf586, 11680, 127, grid=grid(11680), stream=stream0)
        buf587 = buf579; del buf579  # reuse
        buf588 = empty_strided_cuda((1, 160, 1, 1), (160, 1, 160, 160), torch.float32)
        buf590 = empty_strided_cuda((1, 160, 1, 1), (160, 1, 1, 1), torch.float32)
        # Source Nodes: [x_108], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_83.run(buf584, buf585, buf586, primals_434, primals_435, buf587, buf588, buf590, primals_434, primals_435, 160, 73, grid=grid(160), stream=stream0)
        del primals_434
        del primals_435
        buf591 = empty_strided_cuda((32, 160, 17, 17), (46240, 1, 2720, 160), torch.float32)
        # Source Nodes: [branch7x7dbl_8, x_108], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_84.run(buf583, buf587, buf588, primals_143, primals_144, buf591, 1479680, grid=grid(1479680), stream=stream0)
        del primals_144
        # Source Nodes: [x_109], Original ATen: [aten.convolution]
        buf592 = extern_kernels.convolution(buf591, buf27, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf592, (32, 192, 17, 17), (55488, 1, 3264, 192))
        buf593 = buf550; del buf550  # reuse
        buf594 = buf549; del buf549  # reuse
        buf595 = buf548; del buf548  # reuse
        # Source Nodes: [x_110], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf592, buf593, buf594, buf595, 14016, 127, grid=grid(14016), stream=stream0)
        buf596 = buf552; del buf552  # reuse
        buf597 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 192, 192), torch.float32)
        buf599 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 1, 1), torch.float32)
        # Source Nodes: [x_110], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_76.run(buf593, buf594, buf595, primals_437, primals_438, buf596, buf597, buf599, primals_437, primals_438, 192, 73, grid=grid(192), stream=stream0)
        del primals_437
        del primals_438
        buf600 = reinterpret_tensor(buf611, (32, 192, 17, 17), (221952, 289, 17, 1), 110976)  # alias
        buf1080 = empty_strided_cuda((32, 192, 17, 17), (55488, 1, 3264, 192), torch.bool)
        # Source Nodes: [branch7x7dbl_9, x_110], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_77.run(buf592, buf596, buf597, primals_146, primals_147, buf600, buf1080, 6144, 289, grid=grid(6144, 289), stream=stream0)
        del primals_147
        buf601 = empty_strided_cuda((32, 768, 17, 17), (221952, 1, 13056, 768), torch.float32)
        # Source Nodes: [branch_pool_9], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_81.run(buf519, buf601, 7102464, grid=grid(7102464), stream=stream0)
        # Source Nodes: [x_111], Original ATen: [aten.convolution]
        buf602 = extern_kernels.convolution(buf601, primals_148, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf602, (32, 192, 17, 17), (55488, 1, 3264, 192))
        buf603 = buf595; del buf595  # reuse
        buf604 = buf594; del buf594  # reuse
        buf605 = buf593; del buf593  # reuse
        # Source Nodes: [x_112], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf602, buf603, buf604, buf605, 14016, 127, grid=grid(14016), stream=stream0)
        buf606 = buf597; del buf597  # reuse
        buf607 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 192, 192), torch.float32)
        buf609 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 1, 1), torch.float32)
        # Source Nodes: [x_112], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_76.run(buf603, buf604, buf605, primals_440, primals_441, buf606, buf607, buf609, primals_440, primals_441, 192, 73, grid=grid(192), stream=stream0)
        del primals_440
        del primals_441
        buf610 = reinterpret_tensor(buf611, (32, 192, 17, 17), (221952, 289, 17, 1), 166464)  # alias
        buf1079 = empty_strided_cuda((32, 192, 17, 17), (55488, 1, 3264, 192), torch.bool)
        # Source Nodes: [branch_pool_10, x_112], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_77.run(buf602, buf606, buf607, primals_149, primals_150, buf610, buf1079, 6144, 289, grid=grid(6144, 289), stream=stream0)
        del primals_150
        buf612 = empty_strided_cuda((32, 768, 17, 17), (221952, 1, 13056, 768), torch.float32)
        # Source Nodes: [cat_25], Original ATen: [aten.cat]
        triton_poi_fused_cat_74.run(buf611, buf612, 24576, 289, grid=grid(24576, 289), stream=stream0)
        del buf528
        del buf555
        del buf600
        del buf610
        # Source Nodes: [x_114], Original ATen: [aten.convolution]
        buf613 = extern_kernels.convolution(buf612, primals_151, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf613, (32, 192, 17, 17), (55488, 1, 3264, 192))
        buf614 = buf605; del buf605  # reuse
        buf615 = buf604; del buf604  # reuse
        buf616 = buf603; del buf603  # reuse
        # Source Nodes: [x_115], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf613, buf614, buf615, buf616, 14016, 127, grid=grid(14016), stream=stream0)
        buf617 = buf607; del buf607  # reuse
        buf618 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 192, 192), torch.float32)
        buf620 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 1, 1), torch.float32)
        # Source Nodes: [x_115], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_76.run(buf614, buf615, buf616, primals_443, primals_444, buf617, buf618, buf620, primals_443, primals_444, 192, 73, grid=grid(192), stream=stream0)
        del primals_443
        del primals_444
        buf704 = buf611; del buf611  # reuse
        buf621 = reinterpret_tensor(buf704, (32, 192, 17, 17), (221952, 289, 17, 1), 0)  # alias
        buf1078 = empty_strided_cuda((32, 192, 17, 17), (55488, 1, 3264, 192), torch.bool)
        # Source Nodes: [branch1x1_5, x_115], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_77.run(buf613, buf617, buf618, primals_152, primals_153, buf621, buf1078, 6144, 289, grid=grid(6144, 289), stream=stream0)
        del primals_153
        # Source Nodes: [x_116], Original ATen: [aten.convolution]
        buf622 = extern_kernels.convolution(buf612, primals_154, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf622, (32, 160, 17, 17), (46240, 1, 2720, 160))
        buf623 = buf586; del buf586  # reuse
        buf624 = buf585; del buf585  # reuse
        buf625 = buf584; del buf584  # reuse
        # Source Nodes: [x_117], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_82.run(buf622, buf623, buf624, buf625, 11680, 127, grid=grid(11680), stream=stream0)
        buf626 = buf588; del buf588  # reuse
        buf627 = empty_strided_cuda((1, 160, 1, 1), (160, 1, 160, 160), torch.float32)
        buf629 = empty_strided_cuda((1, 160, 1, 1), (160, 1, 1, 1), torch.float32)
        # Source Nodes: [x_117], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_83.run(buf623, buf624, buf625, primals_446, primals_447, buf626, buf627, buf629, primals_446, primals_447, 160, 73, grid=grid(160), stream=stream0)
        del primals_446
        del primals_447
        buf630 = empty_strided_cuda((32, 160, 17, 17), (46240, 1, 2720, 160), torch.float32)
        # Source Nodes: [branch7x7_6, x_117], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_84.run(buf622, buf626, buf627, primals_155, primals_156, buf630, 1479680, grid=grid(1479680), stream=stream0)
        del primals_156
        # Source Nodes: [x_118], Original ATen: [aten.convolution]
        buf631 = extern_kernels.convolution(buf630, buf28, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf631, (32, 160, 17, 17), (46240, 1, 2720, 160))
        buf632 = buf625; del buf625  # reuse
        buf633 = buf624; del buf624  # reuse
        buf634 = buf623; del buf623  # reuse
        # Source Nodes: [x_119], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_82.run(buf631, buf632, buf633, buf634, 11680, 127, grid=grid(11680), stream=stream0)
        buf635 = buf627; del buf627  # reuse
        buf636 = empty_strided_cuda((1, 160, 1, 1), (160, 1, 160, 160), torch.float32)
        buf638 = empty_strided_cuda((1, 160, 1, 1), (160, 1, 1, 1), torch.float32)
        # Source Nodes: [x_119], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_83.run(buf632, buf633, buf634, primals_449, primals_450, buf635, buf636, buf638, primals_449, primals_450, 160, 73, grid=grid(160), stream=stream0)
        del primals_449
        del primals_450
        buf639 = empty_strided_cuda((32, 160, 17, 17), (46240, 1, 2720, 160), torch.float32)
        # Source Nodes: [branch7x7_7, x_119], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_84.run(buf631, buf635, buf636, primals_158, primals_159, buf639, 1479680, grid=grid(1479680), stream=stream0)
        del primals_159
        # Source Nodes: [x_120], Original ATen: [aten.convolution]
        buf640 = extern_kernels.convolution(buf639, buf29, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf640, (32, 192, 17, 17), (55488, 1, 3264, 192))
        buf641 = buf616; del buf616  # reuse
        buf642 = buf615; del buf615  # reuse
        buf643 = buf614; del buf614  # reuse
        # Source Nodes: [x_121], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf640, buf641, buf642, buf643, 14016, 127, grid=grid(14016), stream=stream0)
        buf644 = buf618; del buf618  # reuse
        buf645 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 192, 192), torch.float32)
        buf647 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 1, 1), torch.float32)
        # Source Nodes: [x_121], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_76.run(buf641, buf642, buf643, primals_452, primals_453, buf644, buf645, buf647, primals_452, primals_453, 192, 73, grid=grid(192), stream=stream0)
        del primals_452
        del primals_453
        buf648 = reinterpret_tensor(buf704, (32, 192, 17, 17), (221952, 289, 17, 1), 55488)  # alias
        buf1077 = empty_strided_cuda((32, 192, 17, 17), (55488, 1, 3264, 192), torch.bool)
        # Source Nodes: [branch7x7_8, x_121], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_77.run(buf640, buf644, buf645, primals_161, primals_162, buf648, buf1077, 6144, 289, grid=grid(6144, 289), stream=stream0)
        del primals_162
        # Source Nodes: [x_122], Original ATen: [aten.convolution]
        buf649 = extern_kernels.convolution(buf612, primals_163, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf649, (32, 160, 17, 17), (46240, 1, 2720, 160))
        buf650 = buf634; del buf634  # reuse
        buf651 = buf633; del buf633  # reuse
        buf652 = buf632; del buf632  # reuse
        # Source Nodes: [x_123], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_82.run(buf649, buf650, buf651, buf652, 11680, 127, grid=grid(11680), stream=stream0)
        buf653 = buf636; del buf636  # reuse
        buf654 = empty_strided_cuda((1, 160, 1, 1), (160, 1, 160, 160), torch.float32)
        buf656 = empty_strided_cuda((1, 160, 1, 1), (160, 1, 1, 1), torch.float32)
        # Source Nodes: [x_123], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_83.run(buf650, buf651, buf652, primals_455, primals_456, buf653, buf654, buf656, primals_455, primals_456, 160, 73, grid=grid(160), stream=stream0)
        del primals_455
        del primals_456
        buf657 = empty_strided_cuda((32, 160, 17, 17), (46240, 1, 2720, 160), torch.float32)
        # Source Nodes: [branch7x7dbl_10, x_123], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_84.run(buf649, buf653, buf654, primals_164, primals_165, buf657, 1479680, grid=grid(1479680), stream=stream0)
        del primals_165
        # Source Nodes: [x_124], Original ATen: [aten.convolution]
        buf658 = extern_kernels.convolution(buf657, buf30, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf658, (32, 160, 17, 17), (46240, 1, 2720, 160))
        buf659 = buf652; del buf652  # reuse
        buf660 = buf651; del buf651  # reuse
        buf661 = buf650; del buf650  # reuse
        # Source Nodes: [x_125], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_82.run(buf658, buf659, buf660, buf661, 11680, 127, grid=grid(11680), stream=stream0)
        buf662 = buf654; del buf654  # reuse
        buf663 = empty_strided_cuda((1, 160, 1, 1), (160, 1, 160, 160), torch.float32)
        buf665 = empty_strided_cuda((1, 160, 1, 1), (160, 1, 1, 1), torch.float32)
        # Source Nodes: [x_125], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_83.run(buf659, buf660, buf661, primals_458, primals_459, buf662, buf663, buf665, primals_458, primals_459, 160, 73, grid=grid(160), stream=stream0)
        del primals_458
        del primals_459
        buf666 = empty_strided_cuda((32, 160, 17, 17), (46240, 1, 2720, 160), torch.float32)
        # Source Nodes: [branch7x7dbl_11, x_125], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_84.run(buf658, buf662, buf663, primals_167, primals_168, buf666, 1479680, grid=grid(1479680), stream=stream0)
        del primals_168
        # Source Nodes: [x_126], Original ATen: [aten.convolution]
        buf667 = extern_kernels.convolution(buf666, buf31, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf667, (32, 160, 17, 17), (46240, 1, 2720, 160))
        buf668 = buf661; del buf661  # reuse
        buf669 = buf660; del buf660  # reuse
        buf670 = buf659; del buf659  # reuse
        # Source Nodes: [x_127], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_82.run(buf667, buf668, buf669, buf670, 11680, 127, grid=grid(11680), stream=stream0)
        buf671 = buf663; del buf663  # reuse
        buf672 = empty_strided_cuda((1, 160, 1, 1), (160, 1, 160, 160), torch.float32)
        buf674 = empty_strided_cuda((1, 160, 1, 1), (160, 1, 1, 1), torch.float32)
        # Source Nodes: [x_127], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_83.run(buf668, buf669, buf670, primals_461, primals_462, buf671, buf672, buf674, primals_461, primals_462, 160, 73, grid=grid(160), stream=stream0)
        del primals_461
        del primals_462
        buf675 = empty_strided_cuda((32, 160, 17, 17), (46240, 1, 2720, 160), torch.float32)
        # Source Nodes: [branch7x7dbl_12, x_127], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_84.run(buf667, buf671, buf672, primals_170, primals_171, buf675, 1479680, grid=grid(1479680), stream=stream0)
        del primals_171
        # Source Nodes: [x_128], Original ATen: [aten.convolution]
        buf676 = extern_kernels.convolution(buf675, buf32, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf676, (32, 160, 17, 17), (46240, 1, 2720, 160))
        buf677 = buf670; del buf670  # reuse
        buf678 = buf669; del buf669  # reuse
        buf679 = buf668; del buf668  # reuse
        # Source Nodes: [x_129], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_82.run(buf676, buf677, buf678, buf679, 11680, 127, grid=grid(11680), stream=stream0)
        buf680 = buf672; del buf672  # reuse
        buf681 = empty_strided_cuda((1, 160, 1, 1), (160, 1, 160, 160), torch.float32)
        buf683 = empty_strided_cuda((1, 160, 1, 1), (160, 1, 1, 1), torch.float32)
        # Source Nodes: [x_129], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_83.run(buf677, buf678, buf679, primals_464, primals_465, buf680, buf681, buf683, primals_464, primals_465, 160, 73, grid=grid(160), stream=stream0)
        del buf677
        del buf678
        del buf679
        del primals_464
        del primals_465
        buf684 = empty_strided_cuda((32, 160, 17, 17), (46240, 1, 2720, 160), torch.float32)
        # Source Nodes: [branch7x7dbl_13, x_129], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_84.run(buf676, buf680, buf681, primals_173, primals_174, buf684, 1479680, grid=grid(1479680), stream=stream0)
        del buf681
        del primals_174
        # Source Nodes: [x_130], Original ATen: [aten.convolution]
        buf685 = extern_kernels.convolution(buf684, buf33, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf685, (32, 192, 17, 17), (55488, 1, 3264, 192))
        buf686 = buf643; del buf643  # reuse
        buf687 = buf642; del buf642  # reuse
        buf688 = buf641; del buf641  # reuse
        # Source Nodes: [x_131], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf685, buf686, buf687, buf688, 14016, 127, grid=grid(14016), stream=stream0)
        buf689 = buf645; del buf645  # reuse
        buf690 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 192, 192), torch.float32)
        buf692 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 1, 1), torch.float32)
        # Source Nodes: [x_131], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_76.run(buf686, buf687, buf688, primals_467, primals_468, buf689, buf690, buf692, primals_467, primals_468, 192, 73, grid=grid(192), stream=stream0)
        del primals_467
        del primals_468
        buf693 = reinterpret_tensor(buf704, (32, 192, 17, 17), (221952, 289, 17, 1), 110976)  # alias
        buf1076 = empty_strided_cuda((32, 192, 17, 17), (55488, 1, 3264, 192), torch.bool)
        # Source Nodes: [branch7x7dbl_14, x_131], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_77.run(buf685, buf689, buf690, primals_176, primals_177, buf693, buf1076, 6144, 289, grid=grid(6144, 289), stream=stream0)
        del primals_177
        buf694 = empty_strided_cuda((32, 768, 17, 17), (221952, 1, 13056, 768), torch.float32)
        # Source Nodes: [branch_pool_11], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_81.run(buf612, buf694, 7102464, grid=grid(7102464), stream=stream0)
        # Source Nodes: [x_132], Original ATen: [aten.convolution]
        buf695 = extern_kernels.convolution(buf694, primals_178, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf695, (32, 192, 17, 17), (55488, 1, 3264, 192))
        buf696 = buf688; del buf688  # reuse
        buf697 = buf687; del buf687  # reuse
        buf698 = buf686; del buf686  # reuse
        # Source Nodes: [x_133], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf695, buf696, buf697, buf698, 14016, 127, grid=grid(14016), stream=stream0)
        buf699 = buf690; del buf690  # reuse
        buf700 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 192, 192), torch.float32)
        buf702 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 1, 1), torch.float32)
        # Source Nodes: [x_133], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_76.run(buf696, buf697, buf698, primals_470, primals_471, buf699, buf700, buf702, primals_470, primals_471, 192, 73, grid=grid(192), stream=stream0)
        del primals_470
        del primals_471
        buf703 = reinterpret_tensor(buf704, (32, 192, 17, 17), (221952, 289, 17, 1), 166464)  # alias
        buf1075 = empty_strided_cuda((32, 192, 17, 17), (55488, 1, 3264, 192), torch.bool)
        # Source Nodes: [branch_pool_12, x_133], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_77.run(buf695, buf699, buf700, primals_179, primals_180, buf703, buf1075, 6144, 289, grid=grid(6144, 289), stream=stream0)
        del primals_180
        buf705 = empty_strided_cuda((32, 768, 17, 17), (221952, 1, 13056, 768), torch.float32)
        # Source Nodes: [cat_24], Original ATen: [aten.cat]
        triton_poi_fused_cat_74.run(buf704, buf705, 24576, 289, grid=grid(24576, 289), stream=stream0)
        del buf621
        del buf648
        del buf693
        del buf703
        # Source Nodes: [x_135], Original ATen: [aten.convolution]
        buf706 = extern_kernels.convolution(buf705, primals_181, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf706, (32, 192, 17, 17), (55488, 1, 3264, 192))
        buf707 = buf698; del buf698  # reuse
        buf708 = buf697; del buf697  # reuse
        buf709 = buf696; del buf696  # reuse
        # Source Nodes: [x_136], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf706, buf707, buf708, buf709, 14016, 127, grid=grid(14016), stream=stream0)
        buf710 = buf700; del buf700  # reuse
        buf711 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 192, 192), torch.float32)
        buf713 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 1, 1), torch.float32)
        # Source Nodes: [x_136], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_76.run(buf707, buf708, buf709, primals_473, primals_474, buf710, buf711, buf713, primals_473, primals_474, 192, 73, grid=grid(192), stream=stream0)
        del primals_473
        del primals_474
        buf797 = buf704; del buf704  # reuse
        buf714 = reinterpret_tensor(buf797, (32, 192, 17, 17), (221952, 289, 17, 1), 0)  # alias
        buf1074 = empty_strided_cuda((32, 192, 17, 17), (55488, 1, 3264, 192), torch.bool)
        # Source Nodes: [branch1x1_6, x_136], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_77.run(buf706, buf710, buf711, primals_182, primals_183, buf714, buf1074, 6144, 289, grid=grid(6144, 289), stream=stream0)
        del primals_183
        # Source Nodes: [x_137], Original ATen: [aten.convolution]
        buf715 = extern_kernels.convolution(buf705, primals_184, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf715, (32, 192, 17, 17), (55488, 1, 3264, 192))
        buf716 = buf709; del buf709  # reuse
        buf717 = buf708; del buf708  # reuse
        buf718 = buf707; del buf707  # reuse
        # Source Nodes: [x_138], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf715, buf716, buf717, buf718, 14016, 127, grid=grid(14016), stream=stream0)
        buf719 = buf711; del buf711  # reuse
        buf720 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 192, 192), torch.float32)
        buf722 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 1, 1), torch.float32)
        # Source Nodes: [x_138], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_76.run(buf716, buf717, buf718, primals_476, primals_477, buf719, buf720, buf722, primals_476, primals_477, 192, 73, grid=grid(192), stream=stream0)
        del primals_476
        del primals_477
        buf723 = empty_strided_cuda((32, 192, 17, 17), (55488, 1, 3264, 192), torch.float32)
        # Source Nodes: [branch7x7_9, x_138], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_85.run(buf715, buf719, buf720, primals_185, primals_186, buf723, 1775616, grid=grid(1775616), stream=stream0)
        del primals_186
        # Source Nodes: [x_139], Original ATen: [aten.convolution]
        buf724 = extern_kernels.convolution(buf723, buf34, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf724, (32, 192, 17, 17), (55488, 1, 3264, 192))
        buf725 = buf718; del buf718  # reuse
        buf726 = buf717; del buf717  # reuse
        buf727 = buf716; del buf716  # reuse
        # Source Nodes: [x_140], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf724, buf725, buf726, buf727, 14016, 127, grid=grid(14016), stream=stream0)
        buf728 = buf720; del buf720  # reuse
        buf729 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 192, 192), torch.float32)
        buf731 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 1, 1), torch.float32)
        # Source Nodes: [x_140], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_76.run(buf725, buf726, buf727, primals_479, primals_480, buf728, buf729, buf731, primals_479, primals_480, 192, 73, grid=grid(192), stream=stream0)
        del primals_479
        del primals_480
        buf732 = empty_strided_cuda((32, 192, 17, 17), (55488, 1, 3264, 192), torch.float32)
        # Source Nodes: [branch7x7_10, x_140], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_85.run(buf724, buf728, buf729, primals_188, primals_189, buf732, 1775616, grid=grid(1775616), stream=stream0)
        del primals_189
        # Source Nodes: [x_141], Original ATen: [aten.convolution]
        buf733 = extern_kernels.convolution(buf732, buf35, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf733, (32, 192, 17, 17), (55488, 1, 3264, 192))
        buf734 = buf727; del buf727  # reuse
        buf735 = buf726; del buf726  # reuse
        buf736 = buf725; del buf725  # reuse
        # Source Nodes: [x_142], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf733, buf734, buf735, buf736, 14016, 127, grid=grid(14016), stream=stream0)
        buf737 = buf729; del buf729  # reuse
        buf738 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 192, 192), torch.float32)
        buf740 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 1, 1), torch.float32)
        # Source Nodes: [x_142], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_76.run(buf734, buf735, buf736, primals_482, primals_483, buf737, buf738, buf740, primals_482, primals_483, 192, 73, grid=grid(192), stream=stream0)
        del primals_482
        del primals_483
        buf741 = reinterpret_tensor(buf797, (32, 192, 17, 17), (221952, 289, 17, 1), 55488)  # alias
        buf1073 = empty_strided_cuda((32, 192, 17, 17), (55488, 1, 3264, 192), torch.bool)
        # Source Nodes: [branch7x7_11, x_142], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_77.run(buf733, buf737, buf738, primals_191, primals_192, buf741, buf1073, 6144, 289, grid=grid(6144, 289), stream=stream0)
        del primals_192
        # Source Nodes: [x_143], Original ATen: [aten.convolution]
        buf742 = extern_kernels.convolution(buf705, primals_193, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf742, (32, 192, 17, 17), (55488, 1, 3264, 192))
        buf743 = buf736; del buf736  # reuse
        buf744 = buf735; del buf735  # reuse
        buf745 = buf734; del buf734  # reuse
        # Source Nodes: [x_144], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf742, buf743, buf744, buf745, 14016, 127, grid=grid(14016), stream=stream0)
        buf746 = buf738; del buf738  # reuse
        buf747 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 192, 192), torch.float32)
        buf749 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 1, 1), torch.float32)
        # Source Nodes: [x_144], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_76.run(buf743, buf744, buf745, primals_485, primals_486, buf746, buf747, buf749, primals_485, primals_486, 192, 73, grid=grid(192), stream=stream0)
        del primals_485
        del primals_486
        buf750 = empty_strided_cuda((32, 192, 17, 17), (55488, 1, 3264, 192), torch.float32)
        # Source Nodes: [branch7x7dbl_15, x_144], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_85.run(buf742, buf746, buf747, primals_194, primals_195, buf750, 1775616, grid=grid(1775616), stream=stream0)
        del primals_195
        # Source Nodes: [x_145], Original ATen: [aten.convolution]
        buf751 = extern_kernels.convolution(buf750, buf36, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf751, (32, 192, 17, 17), (55488, 1, 3264, 192))
        buf752 = buf745; del buf745  # reuse
        buf753 = buf744; del buf744  # reuse
        buf754 = buf743; del buf743  # reuse
        # Source Nodes: [x_146], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf751, buf752, buf753, buf754, 14016, 127, grid=grid(14016), stream=stream0)
        buf755 = buf747; del buf747  # reuse
        buf756 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 192, 192), torch.float32)
        buf758 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 1, 1), torch.float32)
        # Source Nodes: [x_146], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_76.run(buf752, buf753, buf754, primals_488, primals_489, buf755, buf756, buf758, primals_488, primals_489, 192, 73, grid=grid(192), stream=stream0)
        del primals_488
        del primals_489
        buf759 = empty_strided_cuda((32, 192, 17, 17), (55488, 1, 3264, 192), torch.float32)
        # Source Nodes: [branch7x7dbl_16, x_146], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_85.run(buf751, buf755, buf756, primals_197, primals_198, buf759, 1775616, grid=grid(1775616), stream=stream0)
        del primals_198
        # Source Nodes: [x_147], Original ATen: [aten.convolution]
        buf760 = extern_kernels.convolution(buf759, buf37, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf760, (32, 192, 17, 17), (55488, 1, 3264, 192))
        buf761 = buf754; del buf754  # reuse
        buf762 = buf753; del buf753  # reuse
        buf763 = buf752; del buf752  # reuse
        # Source Nodes: [x_148], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf760, buf761, buf762, buf763, 14016, 127, grid=grid(14016), stream=stream0)
        buf764 = buf756; del buf756  # reuse
        buf765 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 192, 192), torch.float32)
        buf767 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 1, 1), torch.float32)
        # Source Nodes: [x_148], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_76.run(buf761, buf762, buf763, primals_491, primals_492, buf764, buf765, buf767, primals_491, primals_492, 192, 73, grid=grid(192), stream=stream0)
        del primals_491
        del primals_492
        buf768 = empty_strided_cuda((32, 192, 17, 17), (55488, 1, 3264, 192), torch.float32)
        # Source Nodes: [branch7x7dbl_17, x_148], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_85.run(buf760, buf764, buf765, primals_200, primals_201, buf768, 1775616, grid=grid(1775616), stream=stream0)
        del primals_201
        # Source Nodes: [x_149], Original ATen: [aten.convolution]
        buf769 = extern_kernels.convolution(buf768, buf38, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf769, (32, 192, 17, 17), (55488, 1, 3264, 192))
        buf770 = buf763; del buf763  # reuse
        buf771 = buf762; del buf762  # reuse
        buf772 = buf761; del buf761  # reuse
        # Source Nodes: [x_150], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf769, buf770, buf771, buf772, 14016, 127, grid=grid(14016), stream=stream0)
        buf773 = buf765; del buf765  # reuse
        buf774 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 192, 192), torch.float32)
        buf776 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 1, 1), torch.float32)
        # Source Nodes: [x_150], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_76.run(buf770, buf771, buf772, primals_494, primals_495, buf773, buf774, buf776, primals_494, primals_495, 192, 73, grid=grid(192), stream=stream0)
        del primals_494
        del primals_495
        buf777 = empty_strided_cuda((32, 192, 17, 17), (55488, 1, 3264, 192), torch.float32)
        # Source Nodes: [branch7x7dbl_18, x_150], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_85.run(buf769, buf773, buf774, primals_203, primals_204, buf777, 1775616, grid=grid(1775616), stream=stream0)
        del primals_204
        # Source Nodes: [x_151], Original ATen: [aten.convolution]
        buf778 = extern_kernels.convolution(buf777, buf39, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf778, (32, 192, 17, 17), (55488, 1, 3264, 192))
        buf779 = buf772; del buf772  # reuse
        buf780 = buf771; del buf771  # reuse
        buf781 = buf770; del buf770  # reuse
        # Source Nodes: [x_152], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf778, buf779, buf780, buf781, 14016, 127, grid=grid(14016), stream=stream0)
        buf782 = buf774; del buf774  # reuse
        buf783 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 192, 192), torch.float32)
        buf785 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 1, 1), torch.float32)
        # Source Nodes: [x_152], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_76.run(buf779, buf780, buf781, primals_497, primals_498, buf782, buf783, buf785, primals_497, primals_498, 192, 73, grid=grid(192), stream=stream0)
        del primals_497
        del primals_498
        buf786 = reinterpret_tensor(buf797, (32, 192, 17, 17), (221952, 289, 17, 1), 110976)  # alias
        buf1072 = empty_strided_cuda((32, 192, 17, 17), (55488, 1, 3264, 192), torch.bool)
        # Source Nodes: [branch7x7dbl_19, x_152], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_77.run(buf778, buf782, buf783, primals_206, primals_207, buf786, buf1072, 6144, 289, grid=grid(6144, 289), stream=stream0)
        del primals_207
        buf787 = empty_strided_cuda((32, 768, 17, 17), (221952, 1, 13056, 768), torch.float32)
        # Source Nodes: [branch_pool_13], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_81.run(buf705, buf787, 7102464, grid=grid(7102464), stream=stream0)
        # Source Nodes: [x_153], Original ATen: [aten.convolution]
        buf788 = extern_kernels.convolution(buf787, primals_208, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf788, (32, 192, 17, 17), (55488, 1, 3264, 192))
        buf789 = buf781; del buf781  # reuse
        buf790 = buf780; del buf780  # reuse
        buf791 = buf779; del buf779  # reuse
        # Source Nodes: [x_154], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf788, buf789, buf790, buf791, 14016, 127, grid=grid(14016), stream=stream0)
        buf792 = buf783; del buf783  # reuse
        buf793 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 192, 192), torch.float32)
        buf795 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 1, 1), torch.float32)
        # Source Nodes: [x_154], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_76.run(buf789, buf790, buf791, primals_500, primals_501, buf792, buf793, buf795, primals_500, primals_501, 192, 73, grid=grid(192), stream=stream0)
        del primals_500
        del primals_501
        buf796 = reinterpret_tensor(buf797, (32, 192, 17, 17), (221952, 289, 17, 1), 166464)  # alias
        buf1071 = empty_strided_cuda((32, 192, 17, 17), (55488, 1, 3264, 192), torch.bool)
        # Source Nodes: [branch_pool_14, x_154], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_77.run(buf788, buf792, buf793, primals_209, primals_210, buf796, buf1071, 6144, 289, grid=grid(6144, 289), stream=stream0)
        del primals_210
        buf798 = empty_strided_cuda((32, 768, 17, 17), (221952, 1, 13056, 768), torch.float32)
        # Source Nodes: [cat_23], Original ATen: [aten.cat]
        triton_poi_fused_cat_74.run(buf797, buf798, 24576, 289, grid=grid(24576, 289), stream=stream0)
        del buf714
        del buf741
        del buf786
        del buf796
        del buf797
        buf799 = empty_strided_cuda((32, 768, 5, 5), (19200, 1, 3840, 768), torch.float32)
        # Source Nodes: [x_156], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_86.run(buf798, buf799, 614400, grid=grid(614400), stream=stream0)
        # Source Nodes: [x_157], Original ATen: [aten.convolution]
        buf800 = extern_kernels.convolution(buf799, primals_211, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf800, (32, 128, 5, 5), (3200, 1, 640, 128))
        buf801 = empty_strided_cuda((1, 128, 1, 1, 7), (896, 1, 896, 896, 128), torch.float32)
        buf802 = empty_strided_cuda((1, 128, 1, 1, 7), (896, 1, 896, 896, 128), torch.float32)
        buf803 = empty_strided_cuda((1, 128, 1, 1, 7), (896, 1, 896, 896, 128), torch.float32)
        # Source Nodes: [x_158], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_87.run(buf800, buf801, buf802, buf803, 896, 115, grid=grid(896), stream=stream0)
        buf804 = buf495; del buf495  # reuse
        buf805 = empty_strided_cuda((1, 128, 1, 1), (128, 1, 128, 128), torch.float32)
        buf807 = empty_strided_cuda((1, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        # Source Nodes: [x_158], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_88.run(buf801, buf802, buf803, primals_503, primals_504, buf804, buf805, buf807, primals_503, primals_504, 128, 7, grid=grid(128), stream=stream0)
        del buf801
        del buf802
        del buf803
        del primals_503
        del primals_504
        buf808 = empty_strided_cuda((32, 128, 5, 5), (3200, 1, 640, 128), torch.float32)
        # Source Nodes: [x_158, x_159], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_89.run(buf800, buf804, buf805, primals_212, primals_213, buf808, 102400, grid=grid(102400), stream=stream0)
        del buf805
        del primals_213
        # Source Nodes: [x_160], Original ATen: [aten.convolution]
        buf809 = extern_kernels.convolution(buf808, buf40, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf809, (32, 768, 1, 1), (768, 1, 768, 768))
        buf810 = reinterpret_tensor(buf112, (1, 768, 1, 1), (768, 1, 768, 768), 0); del buf112  # reuse
        buf811 = reinterpret_tensor(buf111, (1, 768, 1, 1), (768, 1, 768, 768), 0); del buf111  # reuse
        buf813 = reinterpret_tensor(buf110, (1, 768, 1, 1), (768, 1, 1, 1), 0); del buf110  # reuse
        # Source Nodes: [x_161], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_90.run(buf809, primals_506, primals_507, buf810, buf811, buf813, primals_506, primals_507, 768, 32, grid=grid(768), stream=stream0)
        del primals_506
        del primals_507
        buf815 = empty_strided_cuda((32, 768, 1, 1), (768, 1, 1, 1), torch.float32)
        buf1070 = empty_strided_cuda((32, 768, 1, 1), (768, 1, 768, 768), torch.bool)
        # Source Nodes: [x_161, x_162, x_163], Original ATen: [aten._native_batch_norm_legit_functional, aten.mean, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_mean_relu_threshold_backward_91.run(buf809, buf810, buf811, primals_215, primals_216, buf815, buf1070, 24576, grid=grid(24576), stream=stream0)
        del buf811
        del primals_216
        buf816 = empty_strided_cuda((32, 1000), (1000, 1), torch.float32)
        # Source Nodes: [aux], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_218, reinterpret_tensor(buf815, (32, 768), (768, 1), 0), reinterpret_tensor(primals_217, (768, 1000), (1, 768), 0), alpha=1, beta=1, out=buf816)
        del primals_218
        # Source Nodes: [x_166], Original ATen: [aten.convolution]
        buf817 = extern_kernels.convolution(buf798, primals_219, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf817, (32, 192, 17, 17), (55488, 1, 3264, 192))
        buf818 = buf791; del buf791  # reuse
        buf819 = buf790; del buf790  # reuse
        buf820 = buf789; del buf789  # reuse
        # Source Nodes: [x_167], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf817, buf818, buf819, buf820, 14016, 127, grid=grid(14016), stream=stream0)
        buf821 = buf793; del buf793  # reuse
        buf822 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 192, 192), torch.float32)
        buf824 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 1, 1), torch.float32)
        # Source Nodes: [x_167], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_76.run(buf818, buf819, buf820, primals_509, primals_510, buf821, buf822, buf824, primals_509, primals_510, 192, 73, grid=grid(192), stream=stream0)
        del primals_509
        del primals_510
        buf825 = empty_strided_cuda((32, 192, 17, 17), (55488, 1, 3264, 192), torch.float32)
        # Source Nodes: [branch3x3_1, x_167], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_85.run(buf817, buf821, buf822, primals_220, primals_221, buf825, 1775616, grid=grid(1775616), stream=stream0)
        del primals_221
        # Source Nodes: [x_168], Original ATen: [aten.convolution]
        buf826 = extern_kernels.convolution(buf825, buf41, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf826, (32, 320, 8, 8), (20480, 1, 2560, 320))
        buf827 = empty_strided_cuda((1, 320, 1, 1, 16), (5120, 1, 5120, 5120, 320), torch.float32)
        buf828 = empty_strided_cuda((1, 320, 1, 1, 16), (5120, 1, 5120, 5120, 320), torch.float32)
        buf829 = empty_strided_cuda((1, 320, 1, 1, 16), (5120, 1, 5120, 5120, 320), torch.float32)
        # Source Nodes: [x_169], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_92.run(buf826, buf827, buf828, buf829, 5120, 128, grid=grid(5120), stream=stream0)
        buf830 = empty_strided_cuda((1, 320, 1, 1), (320, 1, 320, 320), torch.float32)
        buf831 = empty_strided_cuda((1, 320, 1, 1), (320, 1, 320, 320), torch.float32)
        buf833 = empty_strided_cuda((1, 320, 1, 1), (320, 1, 1, 1), torch.float32)
        # Source Nodes: [x_169], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_93.run(buf827, buf828, buf829, primals_512, primals_513, buf830, buf831, buf833, primals_512, primals_513, 320, 16, grid=grid(320), stream=stream0)
        del primals_512
        del primals_513
        buf873 = empty_strided_cuda((32, 1280, 8, 8), (81920, 64, 8, 1), torch.float32)
        buf834 = reinterpret_tensor(buf873, (32, 320, 8, 8), (81920, 64, 8, 1), 0)  # alias
        buf1069 = empty_strided_cuda((32, 320, 8, 8), (20480, 1, 2560, 320), torch.bool)
        # Source Nodes: [branch3x3_2, x_169], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_94.run(buf826, buf830, buf831, primals_223, primals_224, buf834, buf1069, 10240, 64, grid=grid(10240, 64), stream=stream0)
        del primals_224
        # Source Nodes: [x_170], Original ATen: [aten.convolution]
        buf835 = extern_kernels.convolution(buf798, primals_225, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf835, (32, 192, 17, 17), (55488, 1, 3264, 192))
        buf836 = buf820; del buf820  # reuse
        buf837 = buf819; del buf819  # reuse
        buf838 = buf818; del buf818  # reuse
        # Source Nodes: [x_171], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf835, buf836, buf837, buf838, 14016, 127, grid=grid(14016), stream=stream0)
        buf839 = buf822; del buf822  # reuse
        buf840 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 192, 192), torch.float32)
        buf842 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 1, 1), torch.float32)
        # Source Nodes: [x_171], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_76.run(buf836, buf837, buf838, primals_515, primals_516, buf839, buf840, buf842, primals_515, primals_516, 192, 73, grid=grid(192), stream=stream0)
        del primals_515
        del primals_516
        buf843 = empty_strided_cuda((32, 192, 17, 17), (55488, 1, 3264, 192), torch.float32)
        # Source Nodes: [branch7x7x3, x_171], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_85.run(buf835, buf839, buf840, primals_226, primals_227, buf843, 1775616, grid=grid(1775616), stream=stream0)
        del primals_227
        # Source Nodes: [x_172], Original ATen: [aten.convolution]
        buf844 = extern_kernels.convolution(buf843, buf42, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf844, (32, 192, 17, 17), (55488, 1, 3264, 192))
        buf845 = buf838; del buf838  # reuse
        buf846 = buf837; del buf837  # reuse
        buf847 = buf836; del buf836  # reuse
        # Source Nodes: [x_173], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf844, buf845, buf846, buf847, 14016, 127, grid=grid(14016), stream=stream0)
        buf848 = buf840; del buf840  # reuse
        buf849 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 192, 192), torch.float32)
        buf851 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 1, 1), torch.float32)
        # Source Nodes: [x_173], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_76.run(buf845, buf846, buf847, primals_518, primals_519, buf848, buf849, buf851, primals_518, primals_519, 192, 73, grid=grid(192), stream=stream0)
        del primals_518
        del primals_519
        buf852 = empty_strided_cuda((32, 192, 17, 17), (55488, 1, 3264, 192), torch.float32)
        # Source Nodes: [branch7x7x3_1, x_173], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_85.run(buf844, buf848, buf849, primals_229, primals_230, buf852, 1775616, grid=grid(1775616), stream=stream0)
        del primals_230
        # Source Nodes: [x_174], Original ATen: [aten.convolution]
        buf853 = extern_kernels.convolution(buf852, buf43, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf853, (32, 192, 17, 17), (55488, 1, 3264, 192))
        buf854 = buf847; del buf847  # reuse
        buf855 = buf846; del buf846  # reuse
        buf856 = buf845; del buf845  # reuse
        # Source Nodes: [x_175], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf853, buf854, buf855, buf856, 14016, 127, grid=grid(14016), stream=stream0)
        buf857 = buf849; del buf849  # reuse
        buf858 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 192, 192), torch.float32)
        buf860 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 1, 1), torch.float32)
        # Source Nodes: [x_175], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_76.run(buf854, buf855, buf856, primals_521, primals_522, buf857, buf858, buf860, primals_521, primals_522, 192, 73, grid=grid(192), stream=stream0)
        del buf854
        del buf855
        del buf856
        del primals_521
        del primals_522
        buf861 = empty_strided_cuda((32, 192, 17, 17), (55488, 1, 3264, 192), torch.float32)
        # Source Nodes: [branch7x7x3_2, x_175], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_85.run(buf853, buf857, buf858, primals_232, primals_233, buf861, 1775616, grid=grid(1775616), stream=stream0)
        del primals_233
        # Source Nodes: [x_176], Original ATen: [aten.convolution]
        buf862 = extern_kernels.convolution(buf861, buf44, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf862, (32, 192, 8, 8), (12288, 1, 1536, 192))
        buf863 = empty_strided_cuda((1, 192, 1, 1, 16), (3072, 1, 3072, 3072, 192), torch.float32)
        buf864 = empty_strided_cuda((1, 192, 1, 1, 16), (3072, 1, 3072, 3072, 192), torch.float32)
        buf865 = empty_strided_cuda((1, 192, 1, 1, 16), (3072, 1, 3072, 3072, 192), torch.float32)
        # Source Nodes: [x_177], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_95.run(buf862, buf863, buf864, buf865, 3072, 128, grid=grid(3072), stream=stream0)
        buf866 = buf858; del buf858  # reuse
        buf867 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 192, 192), torch.float32)
        buf869 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 1, 1), torch.float32)
        # Source Nodes: [x_177], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_96.run(buf863, buf864, buf865, primals_524, primals_525, buf866, buf867, buf869, primals_524, primals_525, 192, 16, grid=grid(192), stream=stream0)
        del primals_524
        del primals_525
        buf870 = reinterpret_tensor(buf873, (32, 192, 8, 8), (81920, 64, 8, 1), 20480)  # alias
        buf1068 = empty_strided_cuda((32, 192, 8, 8), (12288, 1, 1536, 192), torch.bool)
        # Source Nodes: [branch7x7x3_3, x_177], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_97.run(buf862, buf866, buf867, primals_235, primals_236, buf870, buf1068, 6144, 64, grid=grid(6144, 64), stream=stream0)
        del primals_236
        buf871 = reinterpret_tensor(buf873, (32, 768, 8, 8), (81920, 64, 8, 1), 32768)  # alias
        # Source Nodes: [branch_pool_15], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_98.run(buf798, buf871, 24576, 64, grid=grid(24576, 64), stream=stream0)
        buf872 = empty_strided_cuda((32, 768, 8, 8), (49152, 1, 6144, 768), torch.int64)
        # Source Nodes: [branch_pool_15], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_99.run(buf798, buf872, 1572864, grid=grid(1572864), stream=stream0)
        buf874 = empty_strided_cuda((32, 1280, 8, 8), (81920, 1, 10240, 1280), torch.float32)
        # Source Nodes: [cat_22], Original ATen: [aten.cat]
        triton_poi_fused_cat_100.run(buf873, buf874, 40960, 64, grid=grid(40960, 64), stream=stream0)
        del buf834
        del buf870
        del buf871
        # Source Nodes: [x_179], Original ATen: [aten.convolution]
        buf875 = extern_kernels.convolution(buf874, primals_237, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf875, (32, 320, 8, 8), (20480, 1, 2560, 320))
        buf876 = buf829; del buf829  # reuse
        buf877 = buf828; del buf828  # reuse
        buf878 = buf827; del buf827  # reuse
        # Source Nodes: [x_180], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_92.run(buf875, buf876, buf877, buf878, 5120, 128, grid=grid(5120), stream=stream0)
        buf879 = buf831; del buf831  # reuse
        buf880 = empty_strided_cuda((1, 320, 1, 1), (320, 1, 320, 320), torch.float32)
        buf882 = empty_strided_cuda((1, 320, 1, 1), (320, 1, 1, 1), torch.float32)
        # Source Nodes: [x_180], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_93.run(buf876, buf877, buf878, primals_527, primals_528, buf879, buf880, buf882, primals_527, primals_528, 320, 16, grid=grid(320), stream=stream0)
        del primals_527
        del primals_528
        buf961 = empty_strided_cuda((32, 2048, 8, 8), (131072, 64, 8, 1), torch.float32)
        buf883 = reinterpret_tensor(buf961, (32, 320, 8, 8), (131072, 64, 8, 1), 0)  # alias
        buf1067 = empty_strided_cuda((32, 320, 8, 8), (20480, 1, 2560, 320), torch.bool)
        # Source Nodes: [branch1x1_7, x_180], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_101.run(buf875, buf879, buf880, primals_238, primals_239, buf883, buf1067, 10240, 64, grid=grid(10240, 64), stream=stream0)
        del primals_239
        # Source Nodes: [x_181], Original ATen: [aten.convolution]
        buf884 = extern_kernels.convolution(buf874, primals_240, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf884, (32, 384, 8, 8), (24576, 1, 3072, 384))
        buf885 = empty_strided_cuda((1, 384, 1, 1, 16), (6144, 1, 6144, 6144, 384), torch.float32)
        buf886 = empty_strided_cuda((1, 384, 1, 1, 16), (6144, 1, 6144, 6144, 384), torch.float32)
        buf887 = empty_strided_cuda((1, 384, 1, 1, 16), (6144, 1, 6144, 6144, 384), torch.float32)
        # Source Nodes: [x_182], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_102.run(buf884, buf885, buf886, buf887, 6144, 128, grid=grid(6144), stream=stream0)
        buf888 = buf386; del buf386  # reuse
        buf889 = empty_strided_cuda((1, 384, 1, 1), (384, 1, 384, 384), torch.float32)
        buf891 = empty_strided_cuda((1, 384, 1, 1), (384, 1, 1, 1), torch.float32)
        # Source Nodes: [x_182], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_103.run(buf885, buf886, buf887, primals_530, primals_531, buf888, buf889, buf891, primals_530, primals_531, 384, 16, grid=grid(384), stream=stream0)
        del primals_530
        del primals_531
        buf892 = empty_strided_cuda((32, 384, 8, 8), (24576, 1, 3072, 384), torch.float32)
        # Source Nodes: [branch3x3_3, x_182], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_104.run(buf884, buf888, buf889, primals_241, primals_242, buf892, 786432, grid=grid(786432), stream=stream0)
        del primals_242
        # Source Nodes: [x_183], Original ATen: [aten.convolution]
        buf893 = extern_kernels.convolution(buf892, buf45, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf893, (32, 384, 8, 8), (24576, 1, 3072, 384))
        buf894 = buf887; del buf887  # reuse
        buf895 = buf886; del buf886  # reuse
        buf896 = buf885; del buf885  # reuse
        # Source Nodes: [x_184], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_102.run(buf893, buf894, buf895, buf896, 6144, 128, grid=grid(6144), stream=stream0)
        buf897 = buf889; del buf889  # reuse
        buf898 = empty_strided_cuda((1, 384, 1, 1), (384, 1, 384, 384), torch.float32)
        buf900 = empty_strided_cuda((1, 384, 1, 1), (384, 1, 1, 1), torch.float32)
        # Source Nodes: [x_184], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_103.run(buf894, buf895, buf896, primals_533, primals_534, buf897, buf898, buf900, primals_533, primals_534, 384, 16, grid=grid(384), stream=stream0)
        del primals_533
        del primals_534
        buf911 = empty_strided_cuda((32, 768, 8, 8), (49152, 64, 8, 1), torch.float32)
        buf901 = reinterpret_tensor(buf911, (32, 384, 8, 8), (49152, 64, 8, 1), 0)  # alias
        buf1066 = empty_strided_cuda((32, 384, 8, 8), (24576, 1, 3072, 384), torch.bool)
        # Source Nodes: [relu_80, x_184], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_105.run(buf893, buf897, buf898, primals_244, primals_245, buf901, buf1066, 12288, 64, grid=grid(12288, 64), stream=stream0)
        del primals_245
        # Source Nodes: [x_185], Original ATen: [aten.convolution]
        buf902 = extern_kernels.convolution(buf892, buf46, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf902, (32, 384, 8, 8), (24576, 1, 3072, 384))
        buf903 = buf896; del buf896  # reuse
        buf904 = buf895; del buf895  # reuse
        buf905 = buf894; del buf894  # reuse
        # Source Nodes: [x_186], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_102.run(buf902, buf903, buf904, buf905, 6144, 128, grid=grid(6144), stream=stream0)
        buf906 = buf898; del buf898  # reuse
        buf907 = empty_strided_cuda((1, 384, 1, 1), (384, 1, 384, 384), torch.float32)
        buf909 = empty_strided_cuda((1, 384, 1, 1), (384, 1, 1, 1), torch.float32)
        # Source Nodes: [x_186], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_103.run(buf903, buf904, buf905, primals_536, primals_537, buf906, buf907, buf909, primals_536, primals_537, 384, 16, grid=grid(384), stream=stream0)
        del primals_536
        del primals_537
        buf910 = reinterpret_tensor(buf911, (32, 384, 8, 8), (49152, 64, 8, 1), 24576)  # alias
        buf1065 = empty_strided_cuda((32, 384, 8, 8), (24576, 1, 3072, 384), torch.bool)
        # Source Nodes: [relu_81, x_186], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_105.run(buf902, buf906, buf907, primals_247, primals_248, buf910, buf1065, 12288, 64, grid=grid(12288, 64), stream=stream0)
        del primals_248
        del buf901
        del buf910
        # Source Nodes: [x_187], Original ATen: [aten.convolution]
        buf912 = extern_kernels.convolution(buf874, primals_249, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf912, (32, 448, 8, 8), (28672, 1, 3584, 448))
        buf913 = empty_strided_cuda((1, 448, 1, 1, 16), (7168, 1, 7168, 7168, 448), torch.float32)
        buf914 = empty_strided_cuda((1, 448, 1, 1, 16), (7168, 1, 7168, 7168, 448), torch.float32)
        buf915 = empty_strided_cuda((1, 448, 1, 1, 16), (7168, 1, 7168, 7168, 448), torch.float32)
        # Source Nodes: [x_188], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_106.run(buf912, buf913, buf914, buf915, 7168, 128, grid=grid(7168), stream=stream0)
        buf916 = reinterpret_tensor(buf86, (1, 448, 1, 1), (448, 1, 448, 448), 0); del buf86  # reuse
        buf917 = reinterpret_tensor(buf85, (1, 448, 1, 1), (448, 1, 448, 448), 0); del buf85  # reuse
        buf919 = reinterpret_tensor(buf84, (1, 448, 1, 1), (448, 1, 1, 1), 0); del buf84  # reuse
        # Source Nodes: [x_188], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_107.run(buf913, buf914, buf915, primals_539, primals_540, buf916, buf917, buf919, primals_539, primals_540, 448, 16, grid=grid(448), stream=stream0)
        del primals_539
        del primals_540
        buf920 = empty_strided_cuda((32, 448, 8, 8), (28672, 1, 3584, 448), torch.float32)
        # Source Nodes: [branch3x3dbl_12, x_188], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_108.run(buf912, buf916, buf917, primals_250, primals_251, buf920, 917504, grid=grid(917504), stream=stream0)
        del primals_251
        # Source Nodes: [x_189], Original ATen: [aten.convolution]
        buf921 = extern_kernels.convolution(buf920, buf47, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf921, (32, 384, 8, 8), (24576, 1, 3072, 384))
        buf922 = buf905; del buf905  # reuse
        buf923 = buf904; del buf904  # reuse
        buf924 = buf903; del buf903  # reuse
        # Source Nodes: [x_190], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_102.run(buf921, buf922, buf923, buf924, 6144, 128, grid=grid(6144), stream=stream0)
        buf925 = buf907; del buf907  # reuse
        buf926 = empty_strided_cuda((1, 384, 1, 1), (384, 1, 384, 384), torch.float32)
        buf928 = empty_strided_cuda((1, 384, 1, 1), (384, 1, 1, 1), torch.float32)
        # Source Nodes: [x_190], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_103.run(buf922, buf923, buf924, primals_542, primals_543, buf925, buf926, buf928, primals_542, primals_543, 384, 16, grid=grid(384), stream=stream0)
        del primals_542
        del primals_543
        buf929 = empty_strided_cuda((32, 384, 8, 8), (24576, 1, 3072, 384), torch.float32)
        # Source Nodes: [branch3x3dbl_13, x_190], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_104.run(buf921, buf925, buf926, primals_253, primals_254, buf929, 786432, grid=grid(786432), stream=stream0)
        del primals_254
        # Source Nodes: [x_191], Original ATen: [aten.convolution]
        buf930 = extern_kernels.convolution(buf929, buf48, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf930, (32, 384, 8, 8), (24576, 1, 3072, 384))
        buf931 = buf924; del buf924  # reuse
        buf932 = buf923; del buf923  # reuse
        buf933 = buf922; del buf922  # reuse
        # Source Nodes: [x_192], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_102.run(buf930, buf931, buf932, buf933, 6144, 128, grid=grid(6144), stream=stream0)
        buf934 = buf926; del buf926  # reuse
        buf935 = empty_strided_cuda((1, 384, 1, 1), (384, 1, 384, 384), torch.float32)
        buf937 = empty_strided_cuda((1, 384, 1, 1), (384, 1, 1, 1), torch.float32)
        # Source Nodes: [x_192], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_103.run(buf931, buf932, buf933, primals_545, primals_546, buf934, buf935, buf937, primals_545, primals_546, 384, 16, grid=grid(384), stream=stream0)
        del primals_545
        del primals_546
        buf948 = empty_strided_cuda((32, 768, 8, 8), (49152, 64, 8, 1), torch.float32)
        buf938 = reinterpret_tensor(buf948, (32, 384, 8, 8), (49152, 64, 8, 1), 0)  # alias
        buf1064 = empty_strided_cuda((32, 384, 8, 8), (24576, 1, 3072, 384), torch.bool)
        # Source Nodes: [relu_84, x_192], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_105.run(buf930, buf934, buf935, primals_256, primals_257, buf938, buf1064, 12288, 64, grid=grid(12288, 64), stream=stream0)
        del primals_257
        # Source Nodes: [x_193], Original ATen: [aten.convolution]
        buf939 = extern_kernels.convolution(buf929, buf49, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf939, (32, 384, 8, 8), (24576, 1, 3072, 384))
        buf940 = buf933; del buf933  # reuse
        buf941 = buf932; del buf932  # reuse
        buf942 = buf931; del buf931  # reuse
        # Source Nodes: [x_194], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_102.run(buf939, buf940, buf941, buf942, 6144, 128, grid=grid(6144), stream=stream0)
        buf943 = buf935; del buf935  # reuse
        buf944 = empty_strided_cuda((1, 384, 1, 1), (384, 1, 384, 384), torch.float32)
        buf946 = empty_strided_cuda((1, 384, 1, 1), (384, 1, 1, 1), torch.float32)
        # Source Nodes: [x_194], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_103.run(buf940, buf941, buf942, primals_548, primals_549, buf943, buf944, buf946, primals_548, primals_549, 384, 16, grid=grid(384), stream=stream0)
        del primals_548
        del primals_549
        buf947 = reinterpret_tensor(buf948, (32, 384, 8, 8), (49152, 64, 8, 1), 24576)  # alias
        buf1063 = empty_strided_cuda((32, 384, 8, 8), (24576, 1, 3072, 384), torch.bool)
        # Source Nodes: [relu_85, x_194], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_105.run(buf939, buf943, buf944, primals_259, primals_260, buf947, buf1063, 12288, 64, grid=grid(12288, 64), stream=stream0)
        del primals_260
        buf949 = reinterpret_tensor(buf873, (32, 1280, 8, 8), (81920, 1, 10240, 1280), 0); del buf873  # reuse
        # Source Nodes: [branch_pool_16], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_109.run(buf874, buf949, 2621440, grid=grid(2621440), stream=stream0)
        del buf938
        del buf947
        # Source Nodes: [x_195], Original ATen: [aten.convolution]
        buf950 = extern_kernels.convolution(buf949, primals_261, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf950, (32, 192, 8, 8), (12288, 1, 1536, 192))
        buf951 = buf865; del buf865  # reuse
        buf952 = buf864; del buf864  # reuse
        buf953 = buf863; del buf863  # reuse
        # Source Nodes: [x_196], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_95.run(buf950, buf951, buf952, buf953, 3072, 128, grid=grid(3072), stream=stream0)
        buf954 = buf867; del buf867  # reuse
        buf955 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 192, 192), torch.float32)
        buf957 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 1, 1), torch.float32)
        # Source Nodes: [x_196], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_96.run(buf951, buf952, buf953, primals_551, primals_552, buf954, buf955, buf957, primals_551, primals_552, 192, 16, grid=grid(192), stream=stream0)
        del primals_551
        del primals_552
        buf958 = reinterpret_tensor(buf961, (32, 192, 8, 8), (131072, 64, 8, 1), 118784)  # alias
        buf1062 = empty_strided_cuda((32, 192, 8, 8), (12288, 1, 1536, 192), torch.bool)
        # Source Nodes: [branch_pool_17, x_196], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_110.run(buf950, buf954, buf955, primals_262, primals_263, buf958, buf1062, 6144, 64, grid=grid(6144, 64), stream=stream0)
        del primals_263
        buf959 = reinterpret_tensor(buf961, (32, 768, 8, 8), (131072, 64, 8, 1), 20480)  # alias
        # Source Nodes: [cat_19], Original ATen: [aten.cat]
        triton_poi_fused_cat_111.run(buf911, buf959, 1572864, grid=grid(1572864), stream=stream0)
        buf960 = reinterpret_tensor(buf961, (32, 768, 8, 8), (131072, 64, 8, 1), 69632)  # alias
        # Source Nodes: [cat_19], Original ATen: [aten.cat]
        triton_poi_fused_cat_111.run(buf948, buf960, 1572864, grid=grid(1572864), stream=stream0)
        buf962 = empty_strided_cuda((32, 2048, 8, 8), (131072, 1, 16384, 2048), torch.float32)
        # Source Nodes: [cat_19], Original ATen: [aten.cat]
        triton_poi_fused_cat_112.run(buf961, buf962, 65536, 64, grid=grid(65536, 64), stream=stream0)
        del buf883
        del buf958
        del buf959
        del buf960
        # Source Nodes: [x_198], Original ATen: [aten.convolution]
        buf963 = extern_kernels.convolution(buf962, primals_264, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf963, (32, 320, 8, 8), (20480, 1, 2560, 320))
        buf964 = buf878; del buf878  # reuse
        buf965 = buf877; del buf877  # reuse
        buf966 = buf876; del buf876  # reuse
        # Source Nodes: [x_199], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_92.run(buf963, buf964, buf965, buf966, 5120, 128, grid=grid(5120), stream=stream0)
        buf967 = buf880; del buf880  # reuse
        buf968 = empty_strided_cuda((1, 320, 1, 1), (320, 1, 320, 320), torch.float32)
        buf970 = empty_strided_cuda((1, 320, 1, 1), (320, 1, 1, 1), torch.float32)
        # Source Nodes: [x_199], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_93.run(buf964, buf965, buf966, primals_554, primals_555, buf967, buf968, buf970, primals_554, primals_555, 320, 16, grid=grid(320), stream=stream0)
        del buf964
        del buf965
        del buf966
        del primals_554
        del primals_555
        buf1049 = buf961; del buf961  # reuse
        buf971 = reinterpret_tensor(buf1049, (32, 320, 8, 8), (131072, 64, 8, 1), 0)  # alias
        buf1061 = empty_strided_cuda((32, 320, 8, 8), (20480, 1, 2560, 320), torch.bool)
        # Source Nodes: [branch1x1_8, x_199], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_101.run(buf963, buf967, buf968, primals_265, primals_266, buf971, buf1061, 10240, 64, grid=grid(10240, 64), stream=stream0)
        del buf968
        del primals_266
        # Source Nodes: [x_200], Original ATen: [aten.convolution]
        buf972 = extern_kernels.convolution(buf962, primals_267, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf972, (32, 384, 8, 8), (24576, 1, 3072, 384))
        buf973 = buf942; del buf942  # reuse
        buf974 = buf941; del buf941  # reuse
        buf975 = buf940; del buf940  # reuse
        # Source Nodes: [x_201], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_102.run(buf972, buf973, buf974, buf975, 6144, 128, grid=grid(6144), stream=stream0)
        buf976 = buf944; del buf944  # reuse
        buf977 = empty_strided_cuda((1, 384, 1, 1), (384, 1, 384, 384), torch.float32)
        buf979 = empty_strided_cuda((1, 384, 1, 1), (384, 1, 1, 1), torch.float32)
        # Source Nodes: [x_201], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_103.run(buf973, buf974, buf975, primals_557, primals_558, buf976, buf977, buf979, primals_557, primals_558, 384, 16, grid=grid(384), stream=stream0)
        del primals_557
        del primals_558
        buf980 = empty_strided_cuda((32, 384, 8, 8), (24576, 1, 3072, 384), torch.float32)
        # Source Nodes: [branch3x3_5, x_201], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_104.run(buf972, buf976, buf977, primals_268, primals_269, buf980, 786432, grid=grid(786432), stream=stream0)
        del primals_269
        # Source Nodes: [x_202], Original ATen: [aten.convolution]
        buf981 = extern_kernels.convolution(buf980, buf50, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf981, (32, 384, 8, 8), (24576, 1, 3072, 384))
        buf982 = buf975; del buf975  # reuse
        buf983 = buf974; del buf974  # reuse
        buf984 = buf973; del buf973  # reuse
        # Source Nodes: [x_203], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_102.run(buf981, buf982, buf983, buf984, 6144, 128, grid=grid(6144), stream=stream0)
        buf985 = buf977; del buf977  # reuse
        buf986 = empty_strided_cuda((1, 384, 1, 1), (384, 1, 384, 384), torch.float32)
        buf988 = empty_strided_cuda((1, 384, 1, 1), (384, 1, 1, 1), torch.float32)
        # Source Nodes: [x_203], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_103.run(buf982, buf983, buf984, primals_560, primals_561, buf985, buf986, buf988, primals_560, primals_561, 384, 16, grid=grid(384), stream=stream0)
        del primals_560
        del primals_561
        buf999 = buf948; del buf948  # reuse
        buf989 = reinterpret_tensor(buf999, (32, 384, 8, 8), (49152, 64, 8, 1), 0)  # alias
        buf1060 = empty_strided_cuda((32, 384, 8, 8), (24576, 1, 3072, 384), torch.bool)
        # Source Nodes: [relu_89, x_203], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_105.run(buf981, buf985, buf986, primals_271, primals_272, buf989, buf1060, 12288, 64, grid=grid(12288, 64), stream=stream0)
        del primals_272
        # Source Nodes: [x_204], Original ATen: [aten.convolution]
        buf990 = extern_kernels.convolution(buf980, buf51, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf990, (32, 384, 8, 8), (24576, 1, 3072, 384))
        buf991 = buf984; del buf984  # reuse
        buf992 = buf983; del buf983  # reuse
        buf993 = buf982; del buf982  # reuse
        # Source Nodes: [x_205], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_102.run(buf990, buf991, buf992, buf993, 6144, 128, grid=grid(6144), stream=stream0)
        buf994 = buf986; del buf986  # reuse
        buf995 = empty_strided_cuda((1, 384, 1, 1), (384, 1, 384, 384), torch.float32)
        buf997 = empty_strided_cuda((1, 384, 1, 1), (384, 1, 1, 1), torch.float32)
        # Source Nodes: [x_205], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_103.run(buf991, buf992, buf993, primals_563, primals_564, buf994, buf995, buf997, primals_563, primals_564, 384, 16, grid=grid(384), stream=stream0)
        del primals_563
        del primals_564
        buf998 = reinterpret_tensor(buf999, (32, 384, 8, 8), (49152, 64, 8, 1), 24576)  # alias
        buf1059 = empty_strided_cuda((32, 384, 8, 8), (24576, 1, 3072, 384), torch.bool)
        # Source Nodes: [relu_90, x_205], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_105.run(buf990, buf994, buf995, primals_274, primals_275, buf998, buf1059, 12288, 64, grid=grid(12288, 64), stream=stream0)
        del primals_275
        del buf989
        del buf998
        # Source Nodes: [x_206], Original ATen: [aten.convolution]
        buf1000 = extern_kernels.convolution(buf962, primals_276, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1000, (32, 448, 8, 8), (28672, 1, 3584, 448))
        buf1001 = buf915; del buf915  # reuse
        buf1002 = buf914; del buf914  # reuse
        buf1003 = buf913; del buf913  # reuse
        # Source Nodes: [x_207], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_106.run(buf1000, buf1001, buf1002, buf1003, 7168, 128, grid=grid(7168), stream=stream0)
        buf1004 = buf917; del buf917  # reuse
        buf1005 = empty_strided_cuda((1, 448, 1, 1), (448, 1, 448, 448), torch.float32)
        buf1007 = empty_strided_cuda((1, 448, 1, 1), (448, 1, 1, 1), torch.float32)
        # Source Nodes: [x_207], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_107.run(buf1001, buf1002, buf1003, primals_566, primals_567, buf1004, buf1005, buf1007, primals_566, primals_567, 448, 16, grid=grid(448), stream=stream0)
        del buf1001
        del buf1002
        del buf1003
        del primals_566
        del primals_567
        buf1008 = empty_strided_cuda((32, 448, 8, 8), (28672, 1, 3584, 448), torch.float32)
        # Source Nodes: [branch3x3dbl_15, x_207], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_108.run(buf1000, buf1004, buf1005, primals_277, primals_278, buf1008, 917504, grid=grid(917504), stream=stream0)
        del buf1005
        del primals_278
        # Source Nodes: [x_208], Original ATen: [aten.convolution]
        buf1009 = extern_kernels.convolution(buf1008, buf52, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1009, (32, 384, 8, 8), (24576, 1, 3072, 384))
        buf1010 = buf993; del buf993  # reuse
        buf1011 = buf992; del buf992  # reuse
        buf1012 = buf991; del buf991  # reuse
        # Source Nodes: [x_209], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_102.run(buf1009, buf1010, buf1011, buf1012, 6144, 128, grid=grid(6144), stream=stream0)
        buf1013 = buf995; del buf995  # reuse
        buf1014 = empty_strided_cuda((1, 384, 1, 1), (384, 1, 384, 384), torch.float32)
        buf1016 = empty_strided_cuda((1, 384, 1, 1), (384, 1, 1, 1), torch.float32)
        # Source Nodes: [x_209], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_103.run(buf1010, buf1011, buf1012, primals_569, primals_570, buf1013, buf1014, buf1016, primals_569, primals_570, 384, 16, grid=grid(384), stream=stream0)
        del primals_569
        del primals_570
        buf1017 = empty_strided_cuda((32, 384, 8, 8), (24576, 1, 3072, 384), torch.float32)
        # Source Nodes: [branch3x3dbl_16, x_209], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_104.run(buf1009, buf1013, buf1014, primals_280, primals_281, buf1017, 786432, grid=grid(786432), stream=stream0)
        del primals_281
        # Source Nodes: [x_210], Original ATen: [aten.convolution]
        buf1018 = extern_kernels.convolution(buf1017, buf53, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1018, (32, 384, 8, 8), (24576, 1, 3072, 384))
        buf1019 = buf1012; del buf1012  # reuse
        buf1020 = buf1011; del buf1011  # reuse
        buf1021 = buf1010; del buf1010  # reuse
        # Source Nodes: [x_211], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_102.run(buf1018, buf1019, buf1020, buf1021, 6144, 128, grid=grid(6144), stream=stream0)
        buf1022 = buf1014; del buf1014  # reuse
        buf1023 = empty_strided_cuda((1, 384, 1, 1), (384, 1, 384, 384), torch.float32)
        buf1025 = empty_strided_cuda((1, 384, 1, 1), (384, 1, 1, 1), torch.float32)
        # Source Nodes: [x_211], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_103.run(buf1019, buf1020, buf1021, primals_572, primals_573, buf1022, buf1023, buf1025, primals_572, primals_573, 384, 16, grid=grid(384), stream=stream0)
        del primals_572
        del primals_573
        buf1036 = buf911; del buf911  # reuse
        buf1026 = reinterpret_tensor(buf1036, (32, 384, 8, 8), (49152, 64, 8, 1), 0)  # alias
        buf1058 = empty_strided_cuda((32, 384, 8, 8), (24576, 1, 3072, 384), torch.bool)
        # Source Nodes: [relu_93, x_211], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_105.run(buf1018, buf1022, buf1023, primals_283, primals_284, buf1026, buf1058, 12288, 64, grid=grid(12288, 64), stream=stream0)
        del primals_284
        # Source Nodes: [x_212], Original ATen: [aten.convolution]
        buf1027 = extern_kernels.convolution(buf1017, buf54, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1027, (32, 384, 8, 8), (24576, 1, 3072, 384))
        buf1028 = buf1021; del buf1021  # reuse
        buf1029 = buf1020; del buf1020  # reuse
        buf1030 = buf1019; del buf1019  # reuse
        # Source Nodes: [x_213], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_102.run(buf1027, buf1028, buf1029, buf1030, 6144, 128, grid=grid(6144), stream=stream0)
        buf1031 = buf1023; del buf1023  # reuse
        buf1032 = empty_strided_cuda((1, 384, 1, 1), (384, 1, 384, 384), torch.float32)
        buf1034 = empty_strided_cuda((1, 384, 1, 1), (384, 1, 1, 1), torch.float32)
        # Source Nodes: [x_213], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_103.run(buf1028, buf1029, buf1030, primals_575, primals_576, buf1031, buf1032, buf1034, primals_575, primals_576, 384, 16, grid=grid(384), stream=stream0)
        del buf1028
        del buf1029
        del buf1030
        del primals_575
        del primals_576
        buf1035 = reinterpret_tensor(buf1036, (32, 384, 8, 8), (49152, 64, 8, 1), 24576)  # alias
        buf1057 = empty_strided_cuda((32, 384, 8, 8), (24576, 1, 3072, 384), torch.bool)
        # Source Nodes: [relu_94, x_213], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_105.run(buf1027, buf1031, buf1032, primals_286, primals_287, buf1035, buf1057, 12288, 64, grid=grid(12288, 64), stream=stream0)
        del buf1032
        del primals_287
        buf1037 = empty_strided_cuda((32, 2048, 8, 8), (131072, 1, 16384, 2048), torch.float32)
        # Source Nodes: [branch_pool_18], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_113.run(buf962, buf1037, 4194304, grid=grid(4194304), stream=stream0)
        del buf1026
        del buf1035
        # Source Nodes: [x_214], Original ATen: [aten.convolution]
        buf1038 = extern_kernels.convolution(buf1037, primals_288, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1038, (32, 192, 8, 8), (12288, 1, 1536, 192))
        buf1039 = buf953; del buf953  # reuse
        buf1040 = buf952; del buf952  # reuse
        buf1041 = buf951; del buf951  # reuse
        # Source Nodes: [x_215], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_95.run(buf1038, buf1039, buf1040, buf1041, 3072, 128, grid=grid(3072), stream=stream0)
        buf1042 = buf955; del buf955  # reuse
        buf1043 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 192, 192), torch.float32)
        buf1045 = empty_strided_cuda((1, 192, 1, 1), (192, 1, 1, 1), torch.float32)
        # Source Nodes: [x_215], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_96.run(buf1039, buf1040, buf1041, primals_578, primals_579, buf1042, buf1043, buf1045, primals_578, primals_579, 192, 16, grid=grid(192), stream=stream0)
        del buf1039
        del buf1040
        del buf1041
        del primals_578
        del primals_579
        buf1046 = reinterpret_tensor(buf1049, (32, 192, 8, 8), (131072, 64, 8, 1), 118784)  # alias
        buf1056 = empty_strided_cuda((32, 192, 8, 8), (12288, 1, 1536, 192), torch.bool)
        # Source Nodes: [branch_pool_19, x_215], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_110.run(buf1038, buf1042, buf1043, primals_289, primals_290, buf1046, buf1056, 6144, 64, grid=grid(6144, 64), stream=stream0)
        del buf1043
        del primals_290
        buf1047 = reinterpret_tensor(buf1049, (32, 768, 8, 8), (131072, 64, 8, 1), 20480)  # alias
        # Source Nodes: [cat_16], Original ATen: [aten.cat]
        triton_poi_fused_cat_111.run(buf999, buf1047, 1572864, grid=grid(1572864), stream=stream0)
        del buf999
        buf1048 = reinterpret_tensor(buf1049, (32, 768, 8, 8), (131072, 64, 8, 1), 69632)  # alias
        # Source Nodes: [cat_16], Original ATen: [aten.cat]
        triton_poi_fused_cat_111.run(buf1036, buf1048, 1572864, grid=grid(1572864), stream=stream0)
        del buf1036
        del buf1046
        del buf1047
        del buf1048
        del buf971
        buf1051 = empty_strided_cuda((1, ), (1, ), torch.int64)
        # Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf1051)
        buf1050 = empty_strided_cuda((32, 2048, 1, 1), (2048, 1, 65536, 65536), torch.float32)
        buf1053 = empty_strided_cuda((32, 2048, 1, 1), (2048, 1, 1, 1), torch.bool)
        buf1054 = reinterpret_tensor(buf1050, (32, 2048, 1, 1), (2048, 1, 1, 1), 0); del buf1050  # reuse
        # Source Nodes: [x_217, x_218], Original ATen: [aten.mean, aten.native_dropout]
        triton_per_fused_mean_native_dropout_114.run(buf1054, buf1049, buf1051, buf1053, 0, 65536, 64, grid=grid(65536), stream=stream0)
        del buf1049
        del buf1051
        buf1055 = empty_strided_cuda((32, 1000), (1000, 1), torch.float32)
        # Source Nodes: [x_221], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_292, reinterpret_tensor(buf1054, (32, 2048), (2048, 1), 0), reinterpret_tensor(primals_291, (2048, 1000), (1, 2048), 0), alpha=1, beta=1, out=buf1055)
        del primals_292
        # Source Nodes: [x_2], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_295, primals_295, 1, grid=grid(1), stream=stream0)
        del primals_295
        # Source Nodes: [x_5], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_298, primals_298, 1, grid=grid(1), stream=stream0)
        del primals_298
        # Source Nodes: [x_8], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_301, primals_301, 1, grid=grid(1), stream=stream0)
        del primals_301
        # Source Nodes: [x_12], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_304, primals_304, 1, grid=grid(1), stream=stream0)
        del primals_304
        # Source Nodes: [x_15], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_307, primals_307, 1, grid=grid(1), stream=stream0)
        del primals_307
        # Source Nodes: [x_19], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_310, primals_310, 1, grid=grid(1), stream=stream0)
        del primals_310
        # Source Nodes: [x_21], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_313, primals_313, 1, grid=grid(1), stream=stream0)
        del primals_313
        # Source Nodes: [x_23], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_316, primals_316, 1, grid=grid(1), stream=stream0)
        del primals_316
        # Source Nodes: [x_25], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_319, primals_319, 1, grid=grid(1), stream=stream0)
        del primals_319
        # Source Nodes: [x_27], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_322, primals_322, 1, grid=grid(1), stream=stream0)
        del primals_322
        # Source Nodes: [x_29], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_325, primals_325, 1, grid=grid(1), stream=stream0)
        del primals_325
        # Source Nodes: [x_31], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_328, primals_328, 1, grid=grid(1), stream=stream0)
        del primals_328
        # Source Nodes: [x_34], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_331, primals_331, 1, grid=grid(1), stream=stream0)
        del primals_331
        # Source Nodes: [x_36], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_334, primals_334, 1, grid=grid(1), stream=stream0)
        del primals_334
        # Source Nodes: [x_38], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_337, primals_337, 1, grid=grid(1), stream=stream0)
        del primals_337
        # Source Nodes: [x_40], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_340, primals_340, 1, grid=grid(1), stream=stream0)
        del primals_340
        # Source Nodes: [x_42], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_343, primals_343, 1, grid=grid(1), stream=stream0)
        del primals_343
        # Source Nodes: [x_44], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_346, primals_346, 1, grid=grid(1), stream=stream0)
        del primals_346
        # Source Nodes: [x_46], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_349, primals_349, 1, grid=grid(1), stream=stream0)
        del primals_349
        # Source Nodes: [x_49], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_352, primals_352, 1, grid=grid(1), stream=stream0)
        del primals_352
        # Source Nodes: [x_51], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_355, primals_355, 1, grid=grid(1), stream=stream0)
        del primals_355
        # Source Nodes: [x_53], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_358, primals_358, 1, grid=grid(1), stream=stream0)
        del primals_358
        # Source Nodes: [x_55], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_361, primals_361, 1, grid=grid(1), stream=stream0)
        del primals_361
        # Source Nodes: [x_57], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_364, primals_364, 1, grid=grid(1), stream=stream0)
        del primals_364
        # Source Nodes: [x_59], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_367, primals_367, 1, grid=grid(1), stream=stream0)
        del primals_367
        # Source Nodes: [x_61], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_370, primals_370, 1, grid=grid(1), stream=stream0)
        del primals_370
        # Source Nodes: [x_64], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_373, primals_373, 1, grid=grid(1), stream=stream0)
        del primals_373
        # Source Nodes: [x_66], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_376, primals_376, 1, grid=grid(1), stream=stream0)
        del primals_376
        # Source Nodes: [x_68], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_379, primals_379, 1, grid=grid(1), stream=stream0)
        del primals_379
        # Source Nodes: [x_70], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_382, primals_382, 1, grid=grid(1), stream=stream0)
        del primals_382
        # Source Nodes: [x_73], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_385, primals_385, 1, grid=grid(1), stream=stream0)
        del primals_385
        # Source Nodes: [x_75], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_388, primals_388, 1, grid=grid(1), stream=stream0)
        del primals_388
        # Source Nodes: [x_77], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_391, primals_391, 1, grid=grid(1), stream=stream0)
        del primals_391
        # Source Nodes: [x_79], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_394, primals_394, 1, grid=grid(1), stream=stream0)
        del primals_394
        # Source Nodes: [x_81], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_397, primals_397, 1, grid=grid(1), stream=stream0)
        del primals_397
        # Source Nodes: [x_83], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_400, primals_400, 1, grid=grid(1), stream=stream0)
        del primals_400
        # Source Nodes: [x_85], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_403, primals_403, 1, grid=grid(1), stream=stream0)
        del primals_403
        # Source Nodes: [x_87], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_406, primals_406, 1, grid=grid(1), stream=stream0)
        del primals_406
        # Source Nodes: [x_89], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_409, primals_409, 1, grid=grid(1), stream=stream0)
        del primals_409
        # Source Nodes: [x_91], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_412, primals_412, 1, grid=grid(1), stream=stream0)
        del primals_412
        # Source Nodes: [x_94], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_415, primals_415, 1, grid=grid(1), stream=stream0)
        del primals_415
        # Source Nodes: [x_96], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_418, primals_418, 1, grid=grid(1), stream=stream0)
        del primals_418
        # Source Nodes: [x_98], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_421, primals_421, 1, grid=grid(1), stream=stream0)
        del primals_421
        # Source Nodes: [x_100], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_424, primals_424, 1, grid=grid(1), stream=stream0)
        del primals_424
        # Source Nodes: [x_102], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_427, primals_427, 1, grid=grid(1), stream=stream0)
        del primals_427
        # Source Nodes: [x_104], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_430, primals_430, 1, grid=grid(1), stream=stream0)
        del primals_430
        # Source Nodes: [x_106], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_433, primals_433, 1, grid=grid(1), stream=stream0)
        del primals_433
        # Source Nodes: [x_108], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_436, primals_436, 1, grid=grid(1), stream=stream0)
        del primals_436
        # Source Nodes: [x_110], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_439, primals_439, 1, grid=grid(1), stream=stream0)
        del primals_439
        # Source Nodes: [x_112], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_442, primals_442, 1, grid=grid(1), stream=stream0)
        del primals_442
        # Source Nodes: [x_115], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_445, primals_445, 1, grid=grid(1), stream=stream0)
        del primals_445
        # Source Nodes: [x_117], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_448, primals_448, 1, grid=grid(1), stream=stream0)
        del primals_448
        # Source Nodes: [x_119], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_451, primals_451, 1, grid=grid(1), stream=stream0)
        del primals_451
        # Source Nodes: [x_121], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_454, primals_454, 1, grid=grid(1), stream=stream0)
        del primals_454
        # Source Nodes: [x_123], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_457, primals_457, 1, grid=grid(1), stream=stream0)
        del primals_457
        # Source Nodes: [x_125], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_460, primals_460, 1, grid=grid(1), stream=stream0)
        del primals_460
        # Source Nodes: [x_127], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_463, primals_463, 1, grid=grid(1), stream=stream0)
        del primals_463
        # Source Nodes: [x_129], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_466, primals_466, 1, grid=grid(1), stream=stream0)
        del primals_466
        # Source Nodes: [x_131], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_469, primals_469, 1, grid=grid(1), stream=stream0)
        del primals_469
        # Source Nodes: [x_133], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_472, primals_472, 1, grid=grid(1), stream=stream0)
        del primals_472
        # Source Nodes: [x_136], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_475, primals_475, 1, grid=grid(1), stream=stream0)
        del primals_475
        # Source Nodes: [x_138], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_478, primals_478, 1, grid=grid(1), stream=stream0)
        del primals_478
        # Source Nodes: [x_140], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_481, primals_481, 1, grid=grid(1), stream=stream0)
        del primals_481
        # Source Nodes: [x_142], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_484, primals_484, 1, grid=grid(1), stream=stream0)
        del primals_484
        # Source Nodes: [x_144], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_487, primals_487, 1, grid=grid(1), stream=stream0)
        del primals_487
        # Source Nodes: [x_146], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_490, primals_490, 1, grid=grid(1), stream=stream0)
        del primals_490
        # Source Nodes: [x_148], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_493, primals_493, 1, grid=grid(1), stream=stream0)
        del primals_493
        # Source Nodes: [x_150], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_496, primals_496, 1, grid=grid(1), stream=stream0)
        del primals_496
        # Source Nodes: [x_152], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_499, primals_499, 1, grid=grid(1), stream=stream0)
        del primals_499
        # Source Nodes: [x_154], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_502, primals_502, 1, grid=grid(1), stream=stream0)
        del primals_502
        # Source Nodes: [x_158], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_505, primals_505, 1, grid=grid(1), stream=stream0)
        del primals_505
        # Source Nodes: [x_161], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_508, primals_508, 1, grid=grid(1), stream=stream0)
        del primals_508
        # Source Nodes: [x_167], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_511, primals_511, 1, grid=grid(1), stream=stream0)
        del primals_511
        # Source Nodes: [x_169], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_514, primals_514, 1, grid=grid(1), stream=stream0)
        del primals_514
        # Source Nodes: [x_171], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_517, primals_517, 1, grid=grid(1), stream=stream0)
        del primals_517
        # Source Nodes: [x_173], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_520, primals_520, 1, grid=grid(1), stream=stream0)
        del primals_520
        # Source Nodes: [x_175], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_523, primals_523, 1, grid=grid(1), stream=stream0)
        del primals_523
        # Source Nodes: [x_177], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_526, primals_526, 1, grid=grid(1), stream=stream0)
        del primals_526
        # Source Nodes: [x_180], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_529, primals_529, 1, grid=grid(1), stream=stream0)
        del primals_529
        # Source Nodes: [x_182], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_532, primals_532, 1, grid=grid(1), stream=stream0)
        del primals_532
        # Source Nodes: [x_184], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_535, primals_535, 1, grid=grid(1), stream=stream0)
        del primals_535
        # Source Nodes: [x_186], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_538, primals_538, 1, grid=grid(1), stream=stream0)
        del primals_538
        # Source Nodes: [x_188], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_541, primals_541, 1, grid=grid(1), stream=stream0)
        del primals_541
        # Source Nodes: [x_190], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_544, primals_544, 1, grid=grid(1), stream=stream0)
        del primals_544
        # Source Nodes: [x_192], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_547, primals_547, 1, grid=grid(1), stream=stream0)
        del primals_547
        # Source Nodes: [x_194], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_550, primals_550, 1, grid=grid(1), stream=stream0)
        del primals_550
        # Source Nodes: [x_196], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_553, primals_553, 1, grid=grid(1), stream=stream0)
        del primals_553
        # Source Nodes: [x_199], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_556, primals_556, 1, grid=grid(1), stream=stream0)
        del primals_556
        # Source Nodes: [x_201], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_559, primals_559, 1, grid=grid(1), stream=stream0)
        del primals_559
        # Source Nodes: [x_203], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_562, primals_562, 1, grid=grid(1), stream=stream0)
        del primals_562
        # Source Nodes: [x_205], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_565, primals_565, 1, grid=grid(1), stream=stream0)
        del primals_565
        # Source Nodes: [x_207], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_568, primals_568, 1, grid=grid(1), stream=stream0)
        del primals_568
        # Source Nodes: [x_209], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_571, primals_571, 1, grid=grid(1), stream=stream0)
        del primals_571
        # Source Nodes: [x_211], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_574, primals_574, 1, grid=grid(1), stream=stream0)
        del primals_574
        # Source Nodes: [x_213], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_577, primals_577, 1, grid=grid(1), stream=stream0)
        del primals_577
        # Source Nodes: [x_215], Original ATen: [aten.add]
        triton_poi_fused_add_115.run(primals_580, primals_580, 1, grid=grid(1), stream=stream0)
        del primals_580
    return (buf1055, buf816, buf0, primals_2, buf1, primals_5, buf2, primals_8, primals_10, primals_11, buf3, primals_14, primals_16, primals_17, primals_19, primals_20, buf4, primals_23, primals_25, primals_26, buf5, primals_29, buf6, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, buf7, primals_44, primals_46, primals_47, buf8, primals_50, buf9, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, buf10, primals_65, primals_67, primals_68, buf11, primals_71, buf12, primals_74, primals_76, primals_77, buf13, primals_80, primals_82, primals_83, buf14, primals_86, buf15, primals_89, primals_91, primals_92, primals_94, primals_95, buf16, primals_98, buf17, primals_101, primals_103, primals_104, buf18, primals_107, buf19, primals_110, buf20, primals_113, buf21, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, buf22, primals_128, buf23, primals_131, primals_133, primals_134, buf24, primals_137, buf25, primals_140, buf26, primals_143, buf27, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, buf28, primals_158, buf29, primals_161, primals_163, primals_164, buf30, primals_167, buf31, primals_170, buf32, primals_173, buf33, primals_176, primals_178, primals_179, primals_181, primals_182, primals_184, primals_185, buf34, primals_188, buf35, primals_191, primals_193, primals_194, buf36, primals_197, buf37, primals_200, buf38, primals_203, buf39, primals_206, primals_208, primals_209, primals_211, primals_212, buf40, primals_215, primals_219, primals_220, buf41, primals_223, primals_225, primals_226, buf42, primals_229, buf43, primals_232, buf44, primals_235, primals_237, primals_238, primals_240, primals_241, buf45, primals_244, buf46, primals_247, primals_249, primals_250, buf47, primals_253, buf48, primals_256, buf49, primals_259, primals_261, primals_262, primals_264, primals_265, primals_267, primals_268, buf50, primals_271, buf51, primals_274, primals_276, primals_277, buf52, primals_280, buf53, primals_283, buf54, primals_286, primals_288, primals_289, buf55, buf56, reinterpret_tensor(buf66, (32, ), (1, ), 0), buf67, buf68, reinterpret_tensor(buf78, (32, ), (1, ), 0), buf79, buf80, reinterpret_tensor(buf90, (64, ), (1, ), 0), buf91, buf92, buf93, buf94, reinterpret_tensor(buf104, (80, ), (1, ), 0), buf105, buf106, reinterpret_tensor(buf116, (192, ), (1, ), 0), buf117, buf118, buf119, buf120, reinterpret_tensor(buf130, (64, ), (1, ), 0), buf132, reinterpret_tensor(buf142, (48, ), (1, ), 0), buf143, buf144, reinterpret_tensor(buf154, (64, ), (1, ), 0), buf156, reinterpret_tensor(buf166, (64, ), (1, ), 0), buf167, buf168, reinterpret_tensor(buf178, (96, ), (1, ), 0), buf179, buf180, reinterpret_tensor(buf190, (96, ), (1, ), 0), buf192, buf193, reinterpret_tensor(buf203, (32, ), (1, ), 0), buf206, buf207, reinterpret_tensor(buf217, (64, ), (1, ), 0), buf219, reinterpret_tensor(buf229, (48, ), (1, ), 0), buf230, buf231, reinterpret_tensor(buf241, (64, ), (1, ), 0), buf243, reinterpret_tensor(buf253, (64, ), (1, ), 0), buf254, buf255, reinterpret_tensor(buf265, (96, ), (1, ), 0), buf266, buf267, reinterpret_tensor(buf277, (96, ), (1, ), 0), buf279, buf280, reinterpret_tensor(buf290, (64, ), (1, ), 0), buf293, buf294, reinterpret_tensor(buf304, (64, ), (1, ), 0), buf306, reinterpret_tensor(buf316, (48, ), (1, ), 0), buf317, buf318, reinterpret_tensor(buf328, (64, ), (1, ), 0), buf330, reinterpret_tensor(buf340, (64, ), (1, ), 0), buf341, buf342, reinterpret_tensor(buf352, (96, ), (1, ), 0), buf353, buf354, reinterpret_tensor(buf364, (96, ), (1, ), 0), buf366, buf367, reinterpret_tensor(buf377, (64, ), (1, ), 0), buf380, buf381, reinterpret_tensor(buf388, (384, ), (1, ), 0), buf390, reinterpret_tensor(buf400, (64, ), (1, ), 0), buf401, buf402, reinterpret_tensor(buf412, (96, ), (1, ), 0), buf413, buf414, reinterpret_tensor(buf421, (96, ), (1, ), 0), buf424, buf426, buf427, reinterpret_tensor(buf434, (192, ), (1, ), 0), buf436, reinterpret_tensor(buf443, (128, ), (1, ), 0), buf444, buf445, reinterpret_tensor(buf452, (128, ), (1, ), 0), buf453, buf454, reinterpret_tensor(buf461, (192, ), (1, ), 0), buf463, reinterpret_tensor(buf470, (128, ), (1, ), 0), buf471, buf472, reinterpret_tensor(buf479, (128, ), (1, ), 0), buf480, buf481, reinterpret_tensor(buf488, (128, ), (1, ), 0), buf489, buf490, reinterpret_tensor(buf497, (128, ), (1, ), 0), buf498, buf499, reinterpret_tensor(buf506, (192, ), (1, ), 0), buf508, buf509, reinterpret_tensor(buf516, (192, ), (1, ), 0), buf519, buf520, reinterpret_tensor(buf527, (192, ), (1, ), 0), buf529, reinterpret_tensor(buf536, (160, ), (1, ), 0), buf537, buf538, reinterpret_tensor(buf545, (160, ), (1, ), 0), buf546, buf547, reinterpret_tensor(buf554, (192, ), (1, ), 0), buf556, reinterpret_tensor(buf563, (160, ), (1, ), 0), buf564, buf565, reinterpret_tensor(buf572, (160, ), (1, ), 0), buf573, buf574, reinterpret_tensor(buf581, (160, ), (1, ), 0), buf582, buf583, reinterpret_tensor(buf590, (160, ), (1, ), 0), buf591, buf592, reinterpret_tensor(buf599, (192, ), (1, ), 0), buf601, buf602, reinterpret_tensor(buf609, (192, ), (1, ), 0), buf612, buf613, reinterpret_tensor(buf620, (192, ), (1, ), 0), buf622, reinterpret_tensor(buf629, (160, ), (1, ), 0), buf630, buf631, reinterpret_tensor(buf638, (160, ), (1, ), 0), buf639, buf640, reinterpret_tensor(buf647, (192, ), (1, ), 0), buf649, reinterpret_tensor(buf656, (160, ), (1, ), 0), buf657, buf658, reinterpret_tensor(buf665, (160, ), (1, ), 0), buf666, buf667, reinterpret_tensor(buf674, (160, ), (1, ), 0), buf675, buf676, reinterpret_tensor(buf683, (160, ), (1, ), 0), buf684, buf685, reinterpret_tensor(buf692, (192, ), (1, ), 0), buf694, buf695, reinterpret_tensor(buf702, (192, ), (1, ), 0), buf705, buf706, reinterpret_tensor(buf713, (192, ), (1, ), 0), buf715, reinterpret_tensor(buf722, (192, ), (1, ), 0), buf723, buf724, reinterpret_tensor(buf731, (192, ), (1, ), 0), buf732, buf733, reinterpret_tensor(buf740, (192, ), (1, ), 0), buf742, reinterpret_tensor(buf749, (192, ), (1, ), 0), buf750, buf751, reinterpret_tensor(buf758, (192, ), (1, ), 0), buf759, buf760, reinterpret_tensor(buf767, (192, ), (1, ), 0), buf768, buf769, reinterpret_tensor(buf776, (192, ), (1, ), 0), buf777, buf778, reinterpret_tensor(buf785, (192, ), (1, ), 0), buf787, buf788, reinterpret_tensor(buf795, (192, ), (1, ), 0), buf798, buf799, buf800, reinterpret_tensor(buf807, (128, ), (1, ), 0), buf808, buf809, reinterpret_tensor(buf813, (768, ), (1, ), 0), reinterpret_tensor(buf815, (32, 768), (768, 1), 0), buf817, reinterpret_tensor(buf824, (192, ), (1, ), 0), buf825, buf826, reinterpret_tensor(buf833, (320, ), (1, ), 0), buf835, reinterpret_tensor(buf842, (192, ), (1, ), 0), buf843, buf844, reinterpret_tensor(buf851, (192, ), (1, ), 0), buf852, buf853, reinterpret_tensor(buf860, (192, ), (1, ), 0), buf861, buf862, reinterpret_tensor(buf869, (192, ), (1, ), 0), buf872, buf874, buf875, reinterpret_tensor(buf882, (320, ), (1, ), 0), buf884, reinterpret_tensor(buf891, (384, ), (1, ), 0), buf892, buf893, reinterpret_tensor(buf900, (384, ), (1, ), 0), buf902, reinterpret_tensor(buf909, (384, ), (1, ), 0), buf912, reinterpret_tensor(buf919, (448, ), (1, ), 0), buf920, buf921, reinterpret_tensor(buf928, (384, ), (1, ), 0), buf929, buf930, reinterpret_tensor(buf937, (384, ), (1, ), 0), buf939, reinterpret_tensor(buf946, (384, ), (1, ), 0), buf949, buf950, reinterpret_tensor(buf957, (192, ), (1, ), 0), buf962, buf963, reinterpret_tensor(buf970, (320, ), (1, ), 0), buf972, reinterpret_tensor(buf979, (384, ), (1, ), 0), buf980, buf981, reinterpret_tensor(buf988, (384, ), (1, ), 0), buf990, reinterpret_tensor(buf997, (384, ), (1, ), 0), buf1000, reinterpret_tensor(buf1007, (448, ), (1, ), 0), buf1008, buf1009, reinterpret_tensor(buf1016, (384, ), (1, ), 0), buf1017, buf1018, reinterpret_tensor(buf1025, (384, ), (1, ), 0), buf1027, reinterpret_tensor(buf1034, (384, ), (1, ), 0), buf1037, buf1038, reinterpret_tensor(buf1045, (192, ), (1, ), 0), buf1053, reinterpret_tensor(buf1054, (32, 2048), (2048, 1), 0), reinterpret_tensor(primals_291, (1000, 2048), (2048, 1), 0), buf1056, reinterpret_tensor(buf1042, (1, 192, 1, 1), (192, 1, 1, 1), 0), buf1057, reinterpret_tensor(buf1031, (1, 384, 1, 1), (384, 1, 1, 1), 0), buf1058, reinterpret_tensor(buf1022, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf1013, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf1004, (1, 448, 1, 1), (448, 1, 1, 1), 0), buf1059, reinterpret_tensor(buf994, (1, 384, 1, 1), (384, 1, 1, 1), 0), buf1060, reinterpret_tensor(buf985, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf976, (1, 384, 1, 1), (384, 1, 1, 1), 0), buf1061, reinterpret_tensor(buf967, (1, 320, 1, 1), (320, 1, 1, 1), 0), buf1062, reinterpret_tensor(buf954, (1, 192, 1, 1), (192, 1, 1, 1), 0), buf1063, reinterpret_tensor(buf943, (1, 384, 1, 1), (384, 1, 1, 1), 0), buf1064, reinterpret_tensor(buf934, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf925, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf916, (1, 448, 1, 1), (448, 1, 1, 1), 0), buf1065, reinterpret_tensor(buf906, (1, 384, 1, 1), (384, 1, 1, 1), 0), buf1066, reinterpret_tensor(buf897, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf888, (1, 384, 1, 1), (384, 1, 1, 1), 0), buf1067, reinterpret_tensor(buf879, (1, 320, 1, 1), (320, 1, 1, 1), 0), buf1068, reinterpret_tensor(buf866, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf857, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf848, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf839, (1, 192, 1, 1), (192, 1, 1, 1), 0), buf1069, reinterpret_tensor(buf830, (1, 320, 1, 1), (320, 1, 1, 1), 0), reinterpret_tensor(buf821, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(primals_217, (1000, 768), (768, 1), 0), buf1070, reinterpret_tensor(buf810, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf804, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf1071, reinterpret_tensor(buf792, (1, 192, 1, 1), (192, 1, 1, 1), 0), buf1072, reinterpret_tensor(buf782, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf773, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf764, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf755, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf746, (1, 192, 1, 1), (192, 1, 1, 1), 0), buf1073, reinterpret_tensor(buf737, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf728, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf719, (1, 192, 1, 1), (192, 1, 1, 1), 0), buf1074, reinterpret_tensor(buf710, (1, 192, 1, 1), (192, 1, 1, 1), 0), buf1075, reinterpret_tensor(buf699, (1, 192, 1, 1), (192, 1, 1, 1), 0), buf1076, reinterpret_tensor(buf689, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf680, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf671, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf662, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf653, (1, 160, 1, 1), (160, 1, 1, 1), 0), buf1077, reinterpret_tensor(buf644, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf635, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf626, (1, 160, 1, 1), (160, 1, 1, 1), 0), buf1078, reinterpret_tensor(buf617, (1, 192, 1, 1), (192, 1, 1, 1), 0), buf1079, reinterpret_tensor(buf606, (1, 192, 1, 1), (192, 1, 1, 1), 0), buf1080, reinterpret_tensor(buf596, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf587, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf578, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf569, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf560, (1, 160, 1, 1), (160, 1, 1, 1), 0), buf1081, reinterpret_tensor(buf551, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf542, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf533, (1, 160, 1, 1), (160, 1, 1, 1), 0), buf1082, reinterpret_tensor(buf524, (1, 192, 1, 1), (192, 1, 1, 1), 0), buf1083, reinterpret_tensor(buf513, (1, 192, 1, 1), (192, 1, 1, 1), 0), buf1084, reinterpret_tensor(buf503, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf494, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf485, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf476, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf467, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf1085, reinterpret_tensor(buf458, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf449, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf440, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf1086, reinterpret_tensor(buf431, (1, 192, 1, 1), (192, 1, 1, 1), 0), buf1087, reinterpret_tensor(buf418, (1, 96, 1, 1), (96, 1, 1, 1), 0), reinterpret_tensor(buf409, (1, 96, 1, 1), (96, 1, 1, 1), 0), reinterpret_tensor(buf397, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf1088, reinterpret_tensor(buf385, (1, 384, 1, 1), (384, 1, 1, 1), 0), buf1089, reinterpret_tensor(buf374, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf1090, reinterpret_tensor(buf361, (1, 96, 1, 1), (96, 1, 1, 1), 0), reinterpret_tensor(buf349, (1, 96, 1, 1), (96, 1, 1, 1), 0), reinterpret_tensor(buf337, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf1091, reinterpret_tensor(buf325, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf313, (1, 48, 1, 1), (48, 1, 1, 1), 0), buf1092, reinterpret_tensor(buf301, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf1093, reinterpret_tensor(buf287, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf1094, reinterpret_tensor(buf274, (1, 96, 1, 1), (96, 1, 1, 1), 0), reinterpret_tensor(buf262, (1, 96, 1, 1), (96, 1, 1, 1), 0), reinterpret_tensor(buf250, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf1095, reinterpret_tensor(buf238, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf226, (1, 48, 1, 1), (48, 1, 1, 1), 0), buf1096, reinterpret_tensor(buf214, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf1097, reinterpret_tensor(buf200, (1, 32, 1, 1), (32, 1, 1, 1), 0), buf1098, reinterpret_tensor(buf187, (1, 96, 1, 1), (96, 1, 1, 1), 0), reinterpret_tensor(buf175, (1, 96, 1, 1), (96, 1, 1, 1), 0), reinterpret_tensor(buf163, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf1099, reinterpret_tensor(buf151, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf139, (1, 48, 1, 1), (48, 1, 1, 1), 0), buf1100, reinterpret_tensor(buf127, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf113, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf101, (1, 80, 1, 1), (80, 1, 1, 1), 0), reinterpret_tensor(buf87, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf75, (1, 32, 1, 1), (32, 1, 1, 1), 0), reinterpret_tensor(buf63, (1, 32, 1, 1), (32, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((80, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((192, 80, 3, 3), (720, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((64, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((48, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((64, 48, 5, 5), (1200, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((64, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((32, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((48, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((64, 48, 5, 5), (1200, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((64, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((48, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((64, 48, 5, 5), (1200, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((64, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((64, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((384, 288, 3, 3), (2592, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((64, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((128, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((128, 128, 1, 7), (896, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((192, 128, 7, 1), (896, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((128, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((128, 128, 7, 1), (896, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((128, 128, 1, 7), (896, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((128, 128, 7, 1), (896, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((192, 128, 1, 7), (896, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((160, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((160, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((192, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((160, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((160, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((160, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((160, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((192, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((160, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((160, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((192, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((160, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((160, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((160, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((160, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((192, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((192, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((192, 192, 7, 1), (1344, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((192, 192, 7, 1), (1344, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((192, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((192, 192, 7, 1), (1344, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((192, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((128, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((768, 128, 5, 5), (3200, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((320, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((192, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((192, 192, 7, 1), (1344, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((320, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((384, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((384, 384, 1, 3), (1152, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((384, 384, 3, 1), (1152, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((448, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((384, 448, 3, 3), (4032, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((384, 384, 1, 3), (1152, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((384, 384, 3, 1), (1152, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((192, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((320, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((384, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((384, 384, 1, 3), (1152, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((384, 384, 3, 1), (1152, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((448, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((384, 448, 3, 3), (4032, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((384, 384, 1, 3), (1152, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((384, 384, 3, 1), (1152, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((192, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_296 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_299 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_302 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_305 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_308 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_311 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_314 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_317 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_320 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_323 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_326 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_329 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_332 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_335 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_338 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_341 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_344 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_347 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_350 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_353 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_356 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_359 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_362 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_365 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_368 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_371 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_374 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_377 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_380 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_383 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_386 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_389 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_392 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_395 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_398 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_401 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_404 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_407 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_410 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_413 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_416 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_419 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_420 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_422 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_423 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_425 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_426 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_428 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_429 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_431 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_432 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_434 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_435 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_437 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_438 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_440 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_441 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_443 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_444 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_446 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_447 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_449 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_450 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_452 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_453 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_455 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_456 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_458 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_459 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_460 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_461 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_462 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_463 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_464 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_465 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_466 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_467 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_468 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_470 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_471 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_472 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_473 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_474 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_475 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_476 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_477 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_478 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_479 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_480 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_481 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_482 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_483 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_484 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_485 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_486 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_487 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_488 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_489 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_490 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_491 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_492 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_493 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_494 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_495 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_496 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_497 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_498 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_499 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_500 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_501 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_502 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_503 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_504 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_505 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_506 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_507 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_508 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_509 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_510 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_511 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_512 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_513 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_514 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_515 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_516 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_517 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_518 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_519 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_520 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_521 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_522 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_523 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_524 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_525 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_526 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_527 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_528 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_529 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_530 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_531 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_532 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_533 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_534 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_535 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_536 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_537 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_538 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_539 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_540 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_541 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_542 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_543 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_544 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_545 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_546 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_547 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_548 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_549 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_550 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_551 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_552 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_553 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_554 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_555 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_556 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_557 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_558 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_559 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_560 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_561 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_562 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_563 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_564 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_565 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_566 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_567 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_568 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_569 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_570 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_571 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_572 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_573 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_574 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_575 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_576 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_577 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_578 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_579 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_580 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_581 = rand_strided((32, 3, 299, 299), (268203, 89401, 299, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
