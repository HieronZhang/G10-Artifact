
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

torch._inductor.config.trace.enabled = True




isolate_fails_code_str = None



# torch version: 2.2.2
# torch cuda version: 12.1
# torch git version: 39901f229520a5256505ec24782f716ee7ddc843


# CUDA Info: 
# nvcc not found
# GPU Hardware Info: 
# NVIDIA A100-PCIE-40GB : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, primals_4, primals_14, primals_20, primals_30, primals_36, primals_46, primals_52, primals_62, primals_68, primals_78, primals_84, primals_94, primals_100, primals_110, primals_116, primals_126, primals_132, primals_142, primals_148, primals_158, primals_164, primals_174, primals_180, primals_190, primals_196, primals_202, expand, slice_6, mul_1, gt, view, gt_1, view_16, gt_2, mul_9, view_18, addmm_4, view_20, gt_3, mul_16, view_22, gt_4, view_38, gt_5, mul_22, view_40, addmm_10, view_42, gt_6, mul_29, view_44, gt_7, view_60, gt_8, mul_35, view_62, addmm_16, view_64, gt_9, mul_42, view_66, gt_10, view_82, gt_11, mul_48, view_84, addmm_22, view_86, gt_12, mul_55, view_88, gt_13, view_104, gt_14, mul_61, view_106, addmm_28, view_108, gt_15, mul_68, view_110, gt_16, view_126, gt_17, mul_74, view_128, addmm_34, view_130, gt_18, mul_81, view_132, gt_19, view_148, gt_20, mul_87, view_150, addmm_40, view_152, gt_21, mul_94, view_154, gt_22, view_170, gt_23, mul_100, view_172, addmm_46, view_174, gt_24, mul_107, view_176, gt_25, view_192, gt_26, mul_113, view_194, addmm_52, view_196, gt_27, mul_120, view_198, gt_28, view_214, gt_29, mul_126, view_216, addmm_58, view_218, gt_30, mul_133, view_220, gt_31, view_236, gt_32, mul_139, view_238, addmm_64, view_240, gt_33, mul_146, view_242, gt_34, view_258, gt_35, mul_152, view_260, addmm_70, view_262, gt_36, mul_159, select, tanh, permute_133, div_24, permute_137, permute_141, div_25, permute_145, permute_150, permute_151, alias_14, permute_152, permute_153, permute_157, permute_162, permute_166, div_27, permute_170, permute_174, div_28, permute_178, permute_183, permute_184, alias_15, permute_185, permute_186, permute_190, permute_195, permute_199, div_30, permute_203, permute_207, div_31, permute_211, permute_216, permute_217, alias_16, permute_218, permute_219, permute_223, permute_228, permute_232, div_33, permute_236, permute_240, div_34, permute_244, permute_249, permute_250, alias_17, permute_251, permute_252, permute_256, permute_261, permute_265, div_36, permute_269, permute_273, div_37, permute_277, permute_282, permute_283, alias_18, permute_284, permute_285, permute_289, permute_294, permute_298, div_39, permute_302, permute_306, div_40, permute_310, permute_315, permute_316, alias_19, permute_317, permute_318, permute_322, permute_327, permute_331, div_42, permute_335, permute_339, div_43, permute_343, permute_348, permute_349, alias_20, permute_350, permute_351, permute_355, permute_360, permute_364, div_45, permute_368, permute_372, div_46, permute_376, permute_381, permute_382, alias_21, permute_383, permute_384, permute_388, permute_393, permute_397, div_48, permute_401, permute_405, div_49, permute_409, permute_414, permute_415, alias_22, permute_416, permute_417, permute_421, permute_426, permute_430, div_51, permute_434, permute_438, div_52, permute_442, permute_447, permute_448, alias_23, permute_449, permute_450, permute_454, permute_459, permute_463, div_54, permute_467, permute_471, div_55, permute_475, permute_480, permute_481, alias_24, permute_482, permute_483, permute_487, permute_492, permute_496, div_57, permute_500, permute_504, div_58, permute_508, permute_513, permute_514, alias_25, permute_515, permute_516, permute_520, permute_525, permute_529, div_60, tangents_1, tangents_2):
        view_19 = torch.ops.aten.view.default(addmm_4, [3, 5, 3072]);  addmm_4 = None
        mul_12 = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476)
        erf = torch.ops.aten.erf.default(mul_12);  mul_12 = None
        add_8 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        view_41 = torch.ops.aten.view.default(addmm_10, [3, 5, 3072]);  addmm_10 = None
        mul_25 = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476)
        erf_1 = torch.ops.aten.erf.default(mul_25);  mul_25 = None
        add_16 = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        view_63 = torch.ops.aten.view.default(addmm_16, [3, 5, 3072]);  addmm_16 = None
        mul_38 = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476)
        erf_2 = torch.ops.aten.erf.default(mul_38);  mul_38 = None
        add_24 = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        view_85 = torch.ops.aten.view.default(addmm_22, [3, 5, 3072]);  addmm_22 = None
        mul_51 = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476)
        erf_3 = torch.ops.aten.erf.default(mul_51);  mul_51 = None
        add_32 = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        view_107 = torch.ops.aten.view.default(addmm_28, [3, 5, 3072]);  addmm_28 = None
        mul_64 = torch.ops.aten.mul.Tensor(view_107, 0.7071067811865476)
        erf_4 = torch.ops.aten.erf.default(mul_64);  mul_64 = None
        add_40 = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        view_129 = torch.ops.aten.view.default(addmm_34, [3, 5, 3072]);  addmm_34 = None
        mul_77 = torch.ops.aten.mul.Tensor(view_129, 0.7071067811865476)
        erf_5 = torch.ops.aten.erf.default(mul_77);  mul_77 = None
        add_48 = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        view_151 = torch.ops.aten.view.default(addmm_40, [3, 5, 3072]);  addmm_40 = None
        mul_90 = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476)
        erf_6 = torch.ops.aten.erf.default(mul_90);  mul_90 = None
        add_56 = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        view_173 = torch.ops.aten.view.default(addmm_46, [3, 5, 3072]);  addmm_46 = None
        mul_103 = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476)
        erf_7 = torch.ops.aten.erf.default(mul_103);  mul_103 = None
        add_64 = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        view_195 = torch.ops.aten.view.default(addmm_52, [3, 5, 3072]);  addmm_52 = None
        mul_116 = torch.ops.aten.mul.Tensor(view_195, 0.7071067811865476)
        erf_8 = torch.ops.aten.erf.default(mul_116);  mul_116 = None
        add_72 = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        view_217 = torch.ops.aten.view.default(addmm_58, [3, 5, 3072]);  addmm_58 = None
        mul_129 = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476)
        erf_9 = torch.ops.aten.erf.default(mul_129);  mul_129 = None
        add_80 = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        view_239 = torch.ops.aten.view.default(addmm_64, [3, 5, 3072]);  addmm_64 = None
        mul_142 = torch.ops.aten.mul.Tensor(view_239, 0.7071067811865476)
        erf_10 = torch.ops.aten.erf.default(mul_142);  mul_142 = None
        add_88 = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        view_261 = torch.ops.aten.view.default(addmm_70, [3, 5, 3072]);  addmm_70 = None
        mul_155 = torch.ops.aten.mul.Tensor(view_261, 0.7071067811865476)
        erf_11 = torch.ops.aten.erf.default(mul_155);  mul_155 = None
        add_96 = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        alias_12 = torch.ops.aten.alias.default(tanh);  tanh = None
        alias_13 = torch.ops.aten.alias.default(alias_12);  alias_12 = None
        mul_161 = torch.ops.aten.mul.Tensor(alias_13, alias_13);  alias_13 = None
        sub_38 = torch.ops.aten.sub.Tensor(1, mul_161);  mul_161 = None
        mul_162 = torch.ops.aten.mul.Tensor(tangents_2, sub_38);  tangents_2 = sub_38 = None
        mm = torch.ops.aten.mm.default(mul_162, permute_133);  permute_133 = None
        permute_134 = torch.ops.aten.permute.default(mul_162, [1, 0])
        mm_1 = torch.ops.aten.mm.default(permute_134, select);  permute_134 = select = None
        permute_135 = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
        sum_13 = torch.ops.aten.sum.dim_IntList(mul_162, [0], True);  mul_162 = None
        view_264 = torch.ops.aten.view.default(sum_13, [768]);  sum_13 = None
        permute_136 = torch.ops.aten.permute.default(permute_135, [1, 0]);  permute_135 = None
        full_default = torch.ops.aten.full.default([3, 5, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter = torch.ops.aten.select_scatter.default(full_default, mm, 1, 0);  mm = None
        slice_scatter = torch.ops.aten.slice_scatter.default(full_default, select_scatter, 0, 0, 9223372036854775807);  full_default = select_scatter = None
        add_100 = torch.ops.aten.add.Tensor(tangents_1, slice_scatter);  tangents_1 = slice_scatter = None
        mul_164 = torch.ops.aten.mul.Tensor(add_100, primals_196);  primals_196 = None
        mul_165 = torch.ops.aten.mul.Tensor(mul_164, 768)
        sum_14 = torch.ops.aten.sum.dim_IntList(mul_164, [2], True)
        mul_166 = torch.ops.aten.mul.Tensor(mul_164, mul_159);  mul_164 = None
        sum_15 = torch.ops.aten.sum.dim_IntList(mul_166, [2], True);  mul_166 = None
        mul_167 = torch.ops.aten.mul.Tensor(mul_159, sum_15);  sum_15 = None
        sub_40 = torch.ops.aten.sub.Tensor(mul_165, sum_14);  mul_165 = sum_14 = None
        sub_41 = torch.ops.aten.sub.Tensor(sub_40, mul_167);  sub_40 = mul_167 = None
        mul_168 = torch.ops.aten.mul.Tensor(div_24, sub_41);  div_24 = sub_41 = None
        mul_169 = torch.ops.aten.mul.Tensor(add_100, mul_159);  mul_159 = None
        sum_16 = torch.ops.aten.sum.dim_IntList(mul_169, [0, 1]);  mul_169 = None
        sum_17 = torch.ops.aten.sum.dim_IntList(add_100, [0, 1]);  add_100 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(gt_36, torch.float32);  gt_36 = None
        mul_170 = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_171 = torch.ops.aten.mul.Tensor(mul_168, mul_170);  mul_170 = None
        clone_48 = torch.ops.aten.clone.default(mul_171, memory_format = torch.contiguous_format);  mul_171 = None
        view_265 = torch.ops.aten.view.default(clone_48, [15, 768]);  clone_48 = None
        mm_2 = torch.ops.aten.mm.default(view_265, permute_137);  permute_137 = None
        permute_138 = torch.ops.aten.permute.default(view_265, [1, 0])
        mm_3 = torch.ops.aten.mm.default(permute_138, view_262);  permute_138 = view_262 = None
        permute_139 = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
        sum_18 = torch.ops.aten.sum.dim_IntList(view_265, [0], True);  view_265 = None
        view_266 = torch.ops.aten.view.default(sum_18, [768]);  sum_18 = None
        permute_140 = torch.ops.aten.permute.default(permute_139, [1, 0]);  permute_139 = None
        view_267 = torch.ops.aten.view.default(mm_2, [3, 5, 3072]);  mm_2 = None
        mul_173 = torch.ops.aten.mul.Tensor(add_96, 0.5);  add_96 = None
        mul_174 = torch.ops.aten.mul.Tensor(view_261, view_261)
        mul_175 = torch.ops.aten.mul.Tensor(mul_174, -0.5);  mul_174 = None
        exp_12 = torch.ops.aten.exp.default(mul_175);  mul_175 = None
        mul_176 = torch.ops.aten.mul.Tensor(exp_12, 0.3989422804014327);  exp_12 = None
        mul_177 = torch.ops.aten.mul.Tensor(view_261, mul_176);  view_261 = mul_176 = None
        add_102 = torch.ops.aten.add.Tensor(mul_173, mul_177);  mul_173 = mul_177 = None
        mul_178 = torch.ops.aten.mul.Tensor(view_267, add_102);  view_267 = add_102 = None
        view_268 = torch.ops.aten.view.default(mul_178, [15, 3072]);  mul_178 = None
        mm_4 = torch.ops.aten.mm.default(view_268, permute_141);  permute_141 = None
        permute_142 = torch.ops.aten.permute.default(view_268, [1, 0])
        mm_5 = torch.ops.aten.mm.default(permute_142, view_260);  permute_142 = view_260 = None
        permute_143 = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
        sum_19 = torch.ops.aten.sum.dim_IntList(view_268, [0], True);  view_268 = None
        view_269 = torch.ops.aten.view.default(sum_19, [3072]);  sum_19 = None
        permute_144 = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
        view_270 = torch.ops.aten.view.default(mm_4, [3, 5, 768]);  mm_4 = None
        add_103 = torch.ops.aten.add.Tensor(mul_168, view_270);  mul_168 = view_270 = None
        mul_180 = torch.ops.aten.mul.Tensor(add_103, primals_190);  primals_190 = None
        mul_181 = torch.ops.aten.mul.Tensor(mul_180, 768)
        sum_20 = torch.ops.aten.sum.dim_IntList(mul_180, [2], True)
        mul_182 = torch.ops.aten.mul.Tensor(mul_180, mul_152);  mul_180 = None
        sum_21 = torch.ops.aten.sum.dim_IntList(mul_182, [2], True);  mul_182 = None
        mul_183 = torch.ops.aten.mul.Tensor(mul_152, sum_21);  sum_21 = None
        sub_43 = torch.ops.aten.sub.Tensor(mul_181, sum_20);  mul_181 = sum_20 = None
        sub_44 = torch.ops.aten.sub.Tensor(sub_43, mul_183);  sub_43 = mul_183 = None
        mul_184 = torch.ops.aten.mul.Tensor(div_25, sub_44);  div_25 = sub_44 = None
        mul_185 = torch.ops.aten.mul.Tensor(add_103, mul_152);  mul_152 = None
        sum_22 = torch.ops.aten.sum.dim_IntList(mul_185, [0, 1]);  mul_185 = None
        sum_23 = torch.ops.aten.sum.dim_IntList(add_103, [0, 1]);  add_103 = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(gt_35, torch.float32);  gt_35 = None
        mul_186 = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
        mul_187 = torch.ops.aten.mul.Tensor(mul_184, mul_186);  mul_186 = None
        clone_49 = torch.ops.aten.clone.default(mul_187, memory_format = torch.contiguous_format);  mul_187 = None
        view_271 = torch.ops.aten.view.default(clone_49, [15, 768]);  clone_49 = None
        mm_6 = torch.ops.aten.mm.default(view_271, permute_145);  permute_145 = None
        permute_146 = torch.ops.aten.permute.default(view_271, [1, 0])
        mm_7 = torch.ops.aten.mm.default(permute_146, view_258);  permute_146 = view_258 = None
        permute_147 = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
        sum_24 = torch.ops.aten.sum.dim_IntList(view_271, [0], True);  view_271 = None
        view_272 = torch.ops.aten.view.default(sum_24, [768]);  sum_24 = None
        permute_148 = torch.ops.aten.permute.default(permute_147, [1, 0]);  permute_147 = None
        view_273 = torch.ops.aten.view.default(mm_6, [3, 5, 768]);  mm_6 = None
        view_274 = torch.ops.aten.view.default(view_273, [3, 5, 12, 64]);  view_273 = None
        permute_149 = torch.ops.aten.permute.default(view_274, [0, 2, 1, 3]);  view_274 = None
        clone_50 = torch.ops.aten.clone.default(permute_149, memory_format = torch.contiguous_format);  permute_149 = None
        view_275 = torch.ops.aten.view.default(clone_50, [36, 5, 64]);  clone_50 = None
        bmm_24 = torch.ops.aten.bmm.default(permute_150, view_275);  permute_150 = None
        bmm_25 = torch.ops.aten.bmm.default(view_275, permute_151);  view_275 = permute_151 = None
        view_276 = torch.ops.aten.view.default(bmm_24, [3, 12, 5, 64]);  bmm_24 = None
        view_277 = torch.ops.aten.view.default(bmm_25, [3, 12, 5, 5]);  bmm_25 = None
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(gt_34, torch.float32);  gt_34 = None
        mul_188 = torch.ops.aten.mul.Tensor(convert_element_type_3, 1.1111111111111112);  convert_element_type_3 = None
        mul_189 = torch.ops.aten.mul.Tensor(view_277, mul_188);  view_277 = mul_188 = None
        clone_51 = torch.ops.aten.clone.default(mul_189, memory_format = torch.contiguous_format);  mul_189 = None
        mul_190 = torch.ops.aten.mul.Tensor(clone_51, alias_14);  clone_51 = None
        sum_25 = torch.ops.aten.sum.dim_IntList(mul_190, [-1], True)
        mul_191 = torch.ops.aten.mul.Tensor(alias_14, sum_25);  alias_14 = sum_25 = None
        sub_45 = torch.ops.aten.sub.Tensor(mul_190, mul_191);  mul_190 = mul_191 = None
        div_26 = torch.ops.aten.div.Tensor(sub_45, 8.0);  sub_45 = None
        view_278 = torch.ops.aten.view.default(div_26, [36, 5, 5]);  div_26 = None
        bmm_26 = torch.ops.aten.bmm.default(permute_152, view_278);  permute_152 = None
        bmm_27 = torch.ops.aten.bmm.default(view_278, permute_153);  view_278 = permute_153 = None
        view_279 = torch.ops.aten.view.default(bmm_26, [3, 12, 64, 5]);  bmm_26 = None
        view_280 = torch.ops.aten.view.default(bmm_27, [3, 12, 5, 64]);  bmm_27 = None
        permute_154 = torch.ops.aten.permute.default(view_279, [0, 1, 3, 2]);  view_279 = None
        permute_155 = torch.ops.aten.permute.default(view_280, [0, 2, 1, 3]);  view_280 = None
        clone_52 = torch.ops.aten.clone.default(permute_155, memory_format = torch.contiguous_format);  permute_155 = None
        view_281 = torch.ops.aten.view.default(clone_52, [3, 5, 768]);  clone_52 = None
        permute_156 = torch.ops.aten.permute.default(view_276, [0, 2, 1, 3]);  view_276 = None
        clone_53 = torch.ops.aten.clone.default(permute_156, memory_format = torch.contiguous_format);  permute_156 = None
        view_282 = torch.ops.aten.view.default(clone_53, [3, 5, 768]);  clone_53 = None
        view_283 = torch.ops.aten.view.default(view_282, [15, 768]);  view_282 = None
        mm_8 = torch.ops.aten.mm.default(view_283, permute_157);  permute_157 = None
        permute_158 = torch.ops.aten.permute.default(view_283, [1, 0])
        mm_9 = torch.ops.aten.mm.default(permute_158, view_242);  permute_158 = None
        permute_159 = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
        sum_26 = torch.ops.aten.sum.dim_IntList(view_283, [0], True);  view_283 = None
        view_284 = torch.ops.aten.view.default(sum_26, [768]);  sum_26 = None
        permute_160 = torch.ops.aten.permute.default(permute_159, [1, 0]);  permute_159 = None
        view_285 = torch.ops.aten.view.default(mm_8, [3, 5, 768]);  mm_8 = None
        add_104 = torch.ops.aten.add.Tensor(mul_184, view_285);  mul_184 = view_285 = None
        permute_161 = torch.ops.aten.permute.default(permute_154, [0, 2, 1, 3]);  permute_154 = None
        view_286 = torch.ops.aten.view.default(permute_161, [3, 5, 768]);  permute_161 = None
        clone_54 = torch.ops.aten.clone.default(view_286, memory_format = torch.contiguous_format);  view_286 = None
        view_287 = torch.ops.aten.view.default(clone_54, [15, 768]);  clone_54 = None
        mm_10 = torch.ops.aten.mm.default(view_287, permute_162);  permute_162 = None
        permute_163 = torch.ops.aten.permute.default(view_287, [1, 0])
        mm_11 = torch.ops.aten.mm.default(permute_163, view_242);  permute_163 = None
        permute_164 = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
        sum_27 = torch.ops.aten.sum.dim_IntList(view_287, [0], True);  view_287 = None
        view_288 = torch.ops.aten.view.default(sum_27, [768]);  sum_27 = None
        permute_165 = torch.ops.aten.permute.default(permute_164, [1, 0]);  permute_164 = None
        view_289 = torch.ops.aten.view.default(mm_10, [3, 5, 768]);  mm_10 = None
        add_105 = torch.ops.aten.add.Tensor(add_104, view_289);  add_104 = view_289 = None
        view_290 = torch.ops.aten.view.default(view_281, [15, 768]);  view_281 = None
        mm_12 = torch.ops.aten.mm.default(view_290, permute_166);  permute_166 = None
        permute_167 = torch.ops.aten.permute.default(view_290, [1, 0])
        mm_13 = torch.ops.aten.mm.default(permute_167, view_242);  permute_167 = view_242 = None
        permute_168 = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
        sum_28 = torch.ops.aten.sum.dim_IntList(view_290, [0], True);  view_290 = None
        view_291 = torch.ops.aten.view.default(sum_28, [768]);  sum_28 = None
        permute_169 = torch.ops.aten.permute.default(permute_168, [1, 0]);  permute_168 = None
        view_292 = torch.ops.aten.view.default(mm_12, [3, 5, 768]);  mm_12 = None
        add_106 = torch.ops.aten.add.Tensor(add_105, view_292);  add_105 = view_292 = None
        mul_193 = torch.ops.aten.mul.Tensor(add_106, primals_180);  primals_180 = None
        mul_194 = torch.ops.aten.mul.Tensor(mul_193, 768)
        sum_29 = torch.ops.aten.sum.dim_IntList(mul_193, [2], True)
        mul_195 = torch.ops.aten.mul.Tensor(mul_193, mul_146);  mul_193 = None
        sum_30 = torch.ops.aten.sum.dim_IntList(mul_195, [2], True);  mul_195 = None
        mul_196 = torch.ops.aten.mul.Tensor(mul_146, sum_30);  sum_30 = None
        sub_47 = torch.ops.aten.sub.Tensor(mul_194, sum_29);  mul_194 = sum_29 = None
        sub_48 = torch.ops.aten.sub.Tensor(sub_47, mul_196);  sub_47 = mul_196 = None
        mul_197 = torch.ops.aten.mul.Tensor(div_27, sub_48);  div_27 = sub_48 = None
        mul_198 = torch.ops.aten.mul.Tensor(add_106, mul_146);  mul_146 = None
        sum_31 = torch.ops.aten.sum.dim_IntList(mul_198, [0, 1]);  mul_198 = None
        sum_32 = torch.ops.aten.sum.dim_IntList(add_106, [0, 1]);  add_106 = None
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(gt_33, torch.float32);  gt_33 = None
        mul_199 = torch.ops.aten.mul.Tensor(convert_element_type_4, 1.1111111111111112);  convert_element_type_4 = None
        mul_200 = torch.ops.aten.mul.Tensor(mul_197, mul_199);  mul_199 = None
        clone_55 = torch.ops.aten.clone.default(mul_200, memory_format = torch.contiguous_format);  mul_200 = None
        view_293 = torch.ops.aten.view.default(clone_55, [15, 768]);  clone_55 = None
        mm_14 = torch.ops.aten.mm.default(view_293, permute_170);  permute_170 = None
        permute_171 = torch.ops.aten.permute.default(view_293, [1, 0])
        mm_15 = torch.ops.aten.mm.default(permute_171, view_240);  permute_171 = view_240 = None
        permute_172 = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
        sum_33 = torch.ops.aten.sum.dim_IntList(view_293, [0], True);  view_293 = None
        view_294 = torch.ops.aten.view.default(sum_33, [768]);  sum_33 = None
        permute_173 = torch.ops.aten.permute.default(permute_172, [1, 0]);  permute_172 = None
        view_295 = torch.ops.aten.view.default(mm_14, [3, 5, 3072]);  mm_14 = None
        mul_202 = torch.ops.aten.mul.Tensor(add_88, 0.5);  add_88 = None
        mul_203 = torch.ops.aten.mul.Tensor(view_239, view_239)
        mul_204 = torch.ops.aten.mul.Tensor(mul_203, -0.5);  mul_203 = None
        exp_13 = torch.ops.aten.exp.default(mul_204);  mul_204 = None
        mul_205 = torch.ops.aten.mul.Tensor(exp_13, 0.3989422804014327);  exp_13 = None
        mul_206 = torch.ops.aten.mul.Tensor(view_239, mul_205);  view_239 = mul_205 = None
        add_108 = torch.ops.aten.add.Tensor(mul_202, mul_206);  mul_202 = mul_206 = None
        mul_207 = torch.ops.aten.mul.Tensor(view_295, add_108);  view_295 = add_108 = None
        view_296 = torch.ops.aten.view.default(mul_207, [15, 3072]);  mul_207 = None
        mm_16 = torch.ops.aten.mm.default(view_296, permute_174);  permute_174 = None
        permute_175 = torch.ops.aten.permute.default(view_296, [1, 0])
        mm_17 = torch.ops.aten.mm.default(permute_175, view_238);  permute_175 = view_238 = None
        permute_176 = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
        sum_34 = torch.ops.aten.sum.dim_IntList(view_296, [0], True);  view_296 = None
        view_297 = torch.ops.aten.view.default(sum_34, [3072]);  sum_34 = None
        permute_177 = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
        view_298 = torch.ops.aten.view.default(mm_16, [3, 5, 768]);  mm_16 = None
        add_109 = torch.ops.aten.add.Tensor(mul_197, view_298);  mul_197 = view_298 = None
        mul_209 = torch.ops.aten.mul.Tensor(add_109, primals_174);  primals_174 = None
        mul_210 = torch.ops.aten.mul.Tensor(mul_209, 768)
        sum_35 = torch.ops.aten.sum.dim_IntList(mul_209, [2], True)
        mul_211 = torch.ops.aten.mul.Tensor(mul_209, mul_139);  mul_209 = None
        sum_36 = torch.ops.aten.sum.dim_IntList(mul_211, [2], True);  mul_211 = None
        mul_212 = torch.ops.aten.mul.Tensor(mul_139, sum_36);  sum_36 = None
        sub_50 = torch.ops.aten.sub.Tensor(mul_210, sum_35);  mul_210 = sum_35 = None
        sub_51 = torch.ops.aten.sub.Tensor(sub_50, mul_212);  sub_50 = mul_212 = None
        mul_213 = torch.ops.aten.mul.Tensor(div_28, sub_51);  div_28 = sub_51 = None
        mul_214 = torch.ops.aten.mul.Tensor(add_109, mul_139);  mul_139 = None
        sum_37 = torch.ops.aten.sum.dim_IntList(mul_214, [0, 1]);  mul_214 = None
        sum_38 = torch.ops.aten.sum.dim_IntList(add_109, [0, 1]);  add_109 = None
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(gt_32, torch.float32);  gt_32 = None
        mul_215 = torch.ops.aten.mul.Tensor(convert_element_type_5, 1.1111111111111112);  convert_element_type_5 = None
        mul_216 = torch.ops.aten.mul.Tensor(mul_213, mul_215);  mul_215 = None
        clone_56 = torch.ops.aten.clone.default(mul_216, memory_format = torch.contiguous_format);  mul_216 = None
        view_299 = torch.ops.aten.view.default(clone_56, [15, 768]);  clone_56 = None
        mm_18 = torch.ops.aten.mm.default(view_299, permute_178);  permute_178 = None
        permute_179 = torch.ops.aten.permute.default(view_299, [1, 0])
        mm_19 = torch.ops.aten.mm.default(permute_179, view_236);  permute_179 = view_236 = None
        permute_180 = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
        sum_39 = torch.ops.aten.sum.dim_IntList(view_299, [0], True);  view_299 = None
        view_300 = torch.ops.aten.view.default(sum_39, [768]);  sum_39 = None
        permute_181 = torch.ops.aten.permute.default(permute_180, [1, 0]);  permute_180 = None
        view_301 = torch.ops.aten.view.default(mm_18, [3, 5, 768]);  mm_18 = None
        view_302 = torch.ops.aten.view.default(view_301, [3, 5, 12, 64]);  view_301 = None
        permute_182 = torch.ops.aten.permute.default(view_302, [0, 2, 1, 3]);  view_302 = None
        clone_57 = torch.ops.aten.clone.default(permute_182, memory_format = torch.contiguous_format);  permute_182 = None
        view_303 = torch.ops.aten.view.default(clone_57, [36, 5, 64]);  clone_57 = None
        bmm_28 = torch.ops.aten.bmm.default(permute_183, view_303);  permute_183 = None
        bmm_29 = torch.ops.aten.bmm.default(view_303, permute_184);  view_303 = permute_184 = None
        view_304 = torch.ops.aten.view.default(bmm_28, [3, 12, 5, 64]);  bmm_28 = None
        view_305 = torch.ops.aten.view.default(bmm_29, [3, 12, 5, 5]);  bmm_29 = None
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(gt_31, torch.float32);  gt_31 = None
        mul_217 = torch.ops.aten.mul.Tensor(convert_element_type_6, 1.1111111111111112);  convert_element_type_6 = None
        mul_218 = torch.ops.aten.mul.Tensor(view_305, mul_217);  view_305 = mul_217 = None
        clone_58 = torch.ops.aten.clone.default(mul_218, memory_format = torch.contiguous_format);  mul_218 = None
        mul_219 = torch.ops.aten.mul.Tensor(clone_58, alias_15);  clone_58 = None
        sum_40 = torch.ops.aten.sum.dim_IntList(mul_219, [-1], True)
        mul_220 = torch.ops.aten.mul.Tensor(alias_15, sum_40);  alias_15 = sum_40 = None
        sub_52 = torch.ops.aten.sub.Tensor(mul_219, mul_220);  mul_219 = mul_220 = None
        div_29 = torch.ops.aten.div.Tensor(sub_52, 8.0);  sub_52 = None
        view_306 = torch.ops.aten.view.default(div_29, [36, 5, 5]);  div_29 = None
        bmm_30 = torch.ops.aten.bmm.default(permute_185, view_306);  permute_185 = None
        bmm_31 = torch.ops.aten.bmm.default(view_306, permute_186);  view_306 = permute_186 = None
        view_307 = torch.ops.aten.view.default(bmm_30, [3, 12, 64, 5]);  bmm_30 = None
        view_308 = torch.ops.aten.view.default(bmm_31, [3, 12, 5, 64]);  bmm_31 = None
        permute_187 = torch.ops.aten.permute.default(view_307, [0, 1, 3, 2]);  view_307 = None
        permute_188 = torch.ops.aten.permute.default(view_308, [0, 2, 1, 3]);  view_308 = None
        clone_59 = torch.ops.aten.clone.default(permute_188, memory_format = torch.contiguous_format);  permute_188 = None
        view_309 = torch.ops.aten.view.default(clone_59, [3, 5, 768]);  clone_59 = None
        permute_189 = torch.ops.aten.permute.default(view_304, [0, 2, 1, 3]);  view_304 = None
        clone_60 = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format);  permute_189 = None
        view_310 = torch.ops.aten.view.default(clone_60, [3, 5, 768]);  clone_60 = None
        view_311 = torch.ops.aten.view.default(view_310, [15, 768]);  view_310 = None
        mm_20 = torch.ops.aten.mm.default(view_311, permute_190);  permute_190 = None
        permute_191 = torch.ops.aten.permute.default(view_311, [1, 0])
        mm_21 = torch.ops.aten.mm.default(permute_191, view_220);  permute_191 = None
        permute_192 = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
        sum_41 = torch.ops.aten.sum.dim_IntList(view_311, [0], True);  view_311 = None
        view_312 = torch.ops.aten.view.default(sum_41, [768]);  sum_41 = None
        permute_193 = torch.ops.aten.permute.default(permute_192, [1, 0]);  permute_192 = None
        view_313 = torch.ops.aten.view.default(mm_20, [3, 5, 768]);  mm_20 = None
        add_110 = torch.ops.aten.add.Tensor(mul_213, view_313);  mul_213 = view_313 = None
        permute_194 = torch.ops.aten.permute.default(permute_187, [0, 2, 1, 3]);  permute_187 = None
        view_314 = torch.ops.aten.view.default(permute_194, [3, 5, 768]);  permute_194 = None
        clone_61 = torch.ops.aten.clone.default(view_314, memory_format = torch.contiguous_format);  view_314 = None
        view_315 = torch.ops.aten.view.default(clone_61, [15, 768]);  clone_61 = None
        mm_22 = torch.ops.aten.mm.default(view_315, permute_195);  permute_195 = None
        permute_196 = torch.ops.aten.permute.default(view_315, [1, 0])
        mm_23 = torch.ops.aten.mm.default(permute_196, view_220);  permute_196 = None
        permute_197 = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
        sum_42 = torch.ops.aten.sum.dim_IntList(view_315, [0], True);  view_315 = None
        view_316 = torch.ops.aten.view.default(sum_42, [768]);  sum_42 = None
        permute_198 = torch.ops.aten.permute.default(permute_197, [1, 0]);  permute_197 = None
        view_317 = torch.ops.aten.view.default(mm_22, [3, 5, 768]);  mm_22 = None
        add_111 = torch.ops.aten.add.Tensor(add_110, view_317);  add_110 = view_317 = None
        view_318 = torch.ops.aten.view.default(view_309, [15, 768]);  view_309 = None
        mm_24 = torch.ops.aten.mm.default(view_318, permute_199);  permute_199 = None
        permute_200 = torch.ops.aten.permute.default(view_318, [1, 0])
        mm_25 = torch.ops.aten.mm.default(permute_200, view_220);  permute_200 = view_220 = None
        permute_201 = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
        sum_43 = torch.ops.aten.sum.dim_IntList(view_318, [0], True);  view_318 = None
        view_319 = torch.ops.aten.view.default(sum_43, [768]);  sum_43 = None
        permute_202 = torch.ops.aten.permute.default(permute_201, [1, 0]);  permute_201 = None
        view_320 = torch.ops.aten.view.default(mm_24, [3, 5, 768]);  mm_24 = None
        add_112 = torch.ops.aten.add.Tensor(add_111, view_320);  add_111 = view_320 = None
        mul_222 = torch.ops.aten.mul.Tensor(add_112, primals_164);  primals_164 = None
        mul_223 = torch.ops.aten.mul.Tensor(mul_222, 768)
        sum_44 = torch.ops.aten.sum.dim_IntList(mul_222, [2], True)
        mul_224 = torch.ops.aten.mul.Tensor(mul_222, mul_133);  mul_222 = None
        sum_45 = torch.ops.aten.sum.dim_IntList(mul_224, [2], True);  mul_224 = None
        mul_225 = torch.ops.aten.mul.Tensor(mul_133, sum_45);  sum_45 = None
        sub_54 = torch.ops.aten.sub.Tensor(mul_223, sum_44);  mul_223 = sum_44 = None
        sub_55 = torch.ops.aten.sub.Tensor(sub_54, mul_225);  sub_54 = mul_225 = None
        mul_226 = torch.ops.aten.mul.Tensor(div_30, sub_55);  div_30 = sub_55 = None
        mul_227 = torch.ops.aten.mul.Tensor(add_112, mul_133);  mul_133 = None
        sum_46 = torch.ops.aten.sum.dim_IntList(mul_227, [0, 1]);  mul_227 = None
        sum_47 = torch.ops.aten.sum.dim_IntList(add_112, [0, 1]);  add_112 = None
        convert_element_type_7 = torch.ops.prims.convert_element_type.default(gt_30, torch.float32);  gt_30 = None
        mul_228 = torch.ops.aten.mul.Tensor(convert_element_type_7, 1.1111111111111112);  convert_element_type_7 = None
        mul_229 = torch.ops.aten.mul.Tensor(mul_226, mul_228);  mul_228 = None
        clone_62 = torch.ops.aten.clone.default(mul_229, memory_format = torch.contiguous_format);  mul_229 = None
        view_321 = torch.ops.aten.view.default(clone_62, [15, 768]);  clone_62 = None
        mm_26 = torch.ops.aten.mm.default(view_321, permute_203);  permute_203 = None
        permute_204 = torch.ops.aten.permute.default(view_321, [1, 0])
        mm_27 = torch.ops.aten.mm.default(permute_204, view_218);  permute_204 = view_218 = None
        permute_205 = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
        sum_48 = torch.ops.aten.sum.dim_IntList(view_321, [0], True);  view_321 = None
        view_322 = torch.ops.aten.view.default(sum_48, [768]);  sum_48 = None
        permute_206 = torch.ops.aten.permute.default(permute_205, [1, 0]);  permute_205 = None
        view_323 = torch.ops.aten.view.default(mm_26, [3, 5, 3072]);  mm_26 = None
        mul_231 = torch.ops.aten.mul.Tensor(add_80, 0.5);  add_80 = None
        mul_232 = torch.ops.aten.mul.Tensor(view_217, view_217)
        mul_233 = torch.ops.aten.mul.Tensor(mul_232, -0.5);  mul_232 = None
        exp_14 = torch.ops.aten.exp.default(mul_233);  mul_233 = None
        mul_234 = torch.ops.aten.mul.Tensor(exp_14, 0.3989422804014327);  exp_14 = None
        mul_235 = torch.ops.aten.mul.Tensor(view_217, mul_234);  view_217 = mul_234 = None
        add_114 = torch.ops.aten.add.Tensor(mul_231, mul_235);  mul_231 = mul_235 = None
        mul_236 = torch.ops.aten.mul.Tensor(view_323, add_114);  view_323 = add_114 = None
        view_324 = torch.ops.aten.view.default(mul_236, [15, 3072]);  mul_236 = None
        mm_28 = torch.ops.aten.mm.default(view_324, permute_207);  permute_207 = None
        permute_208 = torch.ops.aten.permute.default(view_324, [1, 0])
        mm_29 = torch.ops.aten.mm.default(permute_208, view_216);  permute_208 = view_216 = None
        permute_209 = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
        sum_49 = torch.ops.aten.sum.dim_IntList(view_324, [0], True);  view_324 = None
        view_325 = torch.ops.aten.view.default(sum_49, [3072]);  sum_49 = None
        permute_210 = torch.ops.aten.permute.default(permute_209, [1, 0]);  permute_209 = None
        view_326 = torch.ops.aten.view.default(mm_28, [3, 5, 768]);  mm_28 = None
        add_115 = torch.ops.aten.add.Tensor(mul_226, view_326);  mul_226 = view_326 = None
        mul_238 = torch.ops.aten.mul.Tensor(add_115, primals_158);  primals_158 = None
        mul_239 = torch.ops.aten.mul.Tensor(mul_238, 768)
        sum_50 = torch.ops.aten.sum.dim_IntList(mul_238, [2], True)
        mul_240 = torch.ops.aten.mul.Tensor(mul_238, mul_126);  mul_238 = None
        sum_51 = torch.ops.aten.sum.dim_IntList(mul_240, [2], True);  mul_240 = None
        mul_241 = torch.ops.aten.mul.Tensor(mul_126, sum_51);  sum_51 = None
        sub_57 = torch.ops.aten.sub.Tensor(mul_239, sum_50);  mul_239 = sum_50 = None
        sub_58 = torch.ops.aten.sub.Tensor(sub_57, mul_241);  sub_57 = mul_241 = None
        mul_242 = torch.ops.aten.mul.Tensor(div_31, sub_58);  div_31 = sub_58 = None
        mul_243 = torch.ops.aten.mul.Tensor(add_115, mul_126);  mul_126 = None
        sum_52 = torch.ops.aten.sum.dim_IntList(mul_243, [0, 1]);  mul_243 = None
        sum_53 = torch.ops.aten.sum.dim_IntList(add_115, [0, 1]);  add_115 = None
        convert_element_type_8 = torch.ops.prims.convert_element_type.default(gt_29, torch.float32);  gt_29 = None
        mul_244 = torch.ops.aten.mul.Tensor(convert_element_type_8, 1.1111111111111112);  convert_element_type_8 = None
        mul_245 = torch.ops.aten.mul.Tensor(mul_242, mul_244);  mul_244 = None
        clone_63 = torch.ops.aten.clone.default(mul_245, memory_format = torch.contiguous_format);  mul_245 = None
        view_327 = torch.ops.aten.view.default(clone_63, [15, 768]);  clone_63 = None
        mm_30 = torch.ops.aten.mm.default(view_327, permute_211);  permute_211 = None
        permute_212 = torch.ops.aten.permute.default(view_327, [1, 0])
        mm_31 = torch.ops.aten.mm.default(permute_212, view_214);  permute_212 = view_214 = None
        permute_213 = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
        sum_54 = torch.ops.aten.sum.dim_IntList(view_327, [0], True);  view_327 = None
        view_328 = torch.ops.aten.view.default(sum_54, [768]);  sum_54 = None
        permute_214 = torch.ops.aten.permute.default(permute_213, [1, 0]);  permute_213 = None
        view_329 = torch.ops.aten.view.default(mm_30, [3, 5, 768]);  mm_30 = None
        view_330 = torch.ops.aten.view.default(view_329, [3, 5, 12, 64]);  view_329 = None
        permute_215 = torch.ops.aten.permute.default(view_330, [0, 2, 1, 3]);  view_330 = None
        clone_64 = torch.ops.aten.clone.default(permute_215, memory_format = torch.contiguous_format);  permute_215 = None
        view_331 = torch.ops.aten.view.default(clone_64, [36, 5, 64]);  clone_64 = None
        bmm_32 = torch.ops.aten.bmm.default(permute_216, view_331);  permute_216 = None
        bmm_33 = torch.ops.aten.bmm.default(view_331, permute_217);  view_331 = permute_217 = None
        view_332 = torch.ops.aten.view.default(bmm_32, [3, 12, 5, 64]);  bmm_32 = None
        view_333 = torch.ops.aten.view.default(bmm_33, [3, 12, 5, 5]);  bmm_33 = None
        convert_element_type_9 = torch.ops.prims.convert_element_type.default(gt_28, torch.float32);  gt_28 = None
        mul_246 = torch.ops.aten.mul.Tensor(convert_element_type_9, 1.1111111111111112);  convert_element_type_9 = None
        mul_247 = torch.ops.aten.mul.Tensor(view_333, mul_246);  view_333 = mul_246 = None
        clone_65 = torch.ops.aten.clone.default(mul_247, memory_format = torch.contiguous_format);  mul_247 = None
        mul_248 = torch.ops.aten.mul.Tensor(clone_65, alias_16);  clone_65 = None
        sum_55 = torch.ops.aten.sum.dim_IntList(mul_248, [-1], True)
        mul_249 = torch.ops.aten.mul.Tensor(alias_16, sum_55);  alias_16 = sum_55 = None
        sub_59 = torch.ops.aten.sub.Tensor(mul_248, mul_249);  mul_248 = mul_249 = None
        div_32 = torch.ops.aten.div.Tensor(sub_59, 8.0);  sub_59 = None
        view_334 = torch.ops.aten.view.default(div_32, [36, 5, 5]);  div_32 = None
        bmm_34 = torch.ops.aten.bmm.default(permute_218, view_334);  permute_218 = None
        bmm_35 = torch.ops.aten.bmm.default(view_334, permute_219);  view_334 = permute_219 = None
        view_335 = torch.ops.aten.view.default(bmm_34, [3, 12, 64, 5]);  bmm_34 = None
        view_336 = torch.ops.aten.view.default(bmm_35, [3, 12, 5, 64]);  bmm_35 = None
        permute_220 = torch.ops.aten.permute.default(view_335, [0, 1, 3, 2]);  view_335 = None
        permute_221 = torch.ops.aten.permute.default(view_336, [0, 2, 1, 3]);  view_336 = None
        clone_66 = torch.ops.aten.clone.default(permute_221, memory_format = torch.contiguous_format);  permute_221 = None
        view_337 = torch.ops.aten.view.default(clone_66, [3, 5, 768]);  clone_66 = None
        permute_222 = torch.ops.aten.permute.default(view_332, [0, 2, 1, 3]);  view_332 = None
        clone_67 = torch.ops.aten.clone.default(permute_222, memory_format = torch.contiguous_format);  permute_222 = None
        view_338 = torch.ops.aten.view.default(clone_67, [3, 5, 768]);  clone_67 = None
        view_339 = torch.ops.aten.view.default(view_338, [15, 768]);  view_338 = None
        mm_32 = torch.ops.aten.mm.default(view_339, permute_223);  permute_223 = None
        permute_224 = torch.ops.aten.permute.default(view_339, [1, 0])
        mm_33 = torch.ops.aten.mm.default(permute_224, view_198);  permute_224 = None
        permute_225 = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
        sum_56 = torch.ops.aten.sum.dim_IntList(view_339, [0], True);  view_339 = None
        view_340 = torch.ops.aten.view.default(sum_56, [768]);  sum_56 = None
        permute_226 = torch.ops.aten.permute.default(permute_225, [1, 0]);  permute_225 = None
        view_341 = torch.ops.aten.view.default(mm_32, [3, 5, 768]);  mm_32 = None
        add_116 = torch.ops.aten.add.Tensor(mul_242, view_341);  mul_242 = view_341 = None
        permute_227 = torch.ops.aten.permute.default(permute_220, [0, 2, 1, 3]);  permute_220 = None
        view_342 = torch.ops.aten.view.default(permute_227, [3, 5, 768]);  permute_227 = None
        clone_68 = torch.ops.aten.clone.default(view_342, memory_format = torch.contiguous_format);  view_342 = None
        view_343 = torch.ops.aten.view.default(clone_68, [15, 768]);  clone_68 = None
        mm_34 = torch.ops.aten.mm.default(view_343, permute_228);  permute_228 = None
        permute_229 = torch.ops.aten.permute.default(view_343, [1, 0])
        mm_35 = torch.ops.aten.mm.default(permute_229, view_198);  permute_229 = None
        permute_230 = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
        sum_57 = torch.ops.aten.sum.dim_IntList(view_343, [0], True);  view_343 = None
        view_344 = torch.ops.aten.view.default(sum_57, [768]);  sum_57 = None
        permute_231 = torch.ops.aten.permute.default(permute_230, [1, 0]);  permute_230 = None
        view_345 = torch.ops.aten.view.default(mm_34, [3, 5, 768]);  mm_34 = None
        add_117 = torch.ops.aten.add.Tensor(add_116, view_345);  add_116 = view_345 = None
        view_346 = torch.ops.aten.view.default(view_337, [15, 768]);  view_337 = None
        mm_36 = torch.ops.aten.mm.default(view_346, permute_232);  permute_232 = None
        permute_233 = torch.ops.aten.permute.default(view_346, [1, 0])
        mm_37 = torch.ops.aten.mm.default(permute_233, view_198);  permute_233 = view_198 = None
        permute_234 = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
        sum_58 = torch.ops.aten.sum.dim_IntList(view_346, [0], True);  view_346 = None
        view_347 = torch.ops.aten.view.default(sum_58, [768]);  sum_58 = None
        permute_235 = torch.ops.aten.permute.default(permute_234, [1, 0]);  permute_234 = None
        view_348 = torch.ops.aten.view.default(mm_36, [3, 5, 768]);  mm_36 = None
        add_118 = torch.ops.aten.add.Tensor(add_117, view_348);  add_117 = view_348 = None
        mul_251 = torch.ops.aten.mul.Tensor(add_118, primals_148);  primals_148 = None
        mul_252 = torch.ops.aten.mul.Tensor(mul_251, 768)
        sum_59 = torch.ops.aten.sum.dim_IntList(mul_251, [2], True)
        mul_253 = torch.ops.aten.mul.Tensor(mul_251, mul_120);  mul_251 = None
        sum_60 = torch.ops.aten.sum.dim_IntList(mul_253, [2], True);  mul_253 = None
        mul_254 = torch.ops.aten.mul.Tensor(mul_120, sum_60);  sum_60 = None
        sub_61 = torch.ops.aten.sub.Tensor(mul_252, sum_59);  mul_252 = sum_59 = None
        sub_62 = torch.ops.aten.sub.Tensor(sub_61, mul_254);  sub_61 = mul_254 = None
        mul_255 = torch.ops.aten.mul.Tensor(div_33, sub_62);  div_33 = sub_62 = None
        mul_256 = torch.ops.aten.mul.Tensor(add_118, mul_120);  mul_120 = None
        sum_61 = torch.ops.aten.sum.dim_IntList(mul_256, [0, 1]);  mul_256 = None
        sum_62 = torch.ops.aten.sum.dim_IntList(add_118, [0, 1]);  add_118 = None
        convert_element_type_10 = torch.ops.prims.convert_element_type.default(gt_27, torch.float32);  gt_27 = None
        mul_257 = torch.ops.aten.mul.Tensor(convert_element_type_10, 1.1111111111111112);  convert_element_type_10 = None
        mul_258 = torch.ops.aten.mul.Tensor(mul_255, mul_257);  mul_257 = None
        clone_69 = torch.ops.aten.clone.default(mul_258, memory_format = torch.contiguous_format);  mul_258 = None
        view_349 = torch.ops.aten.view.default(clone_69, [15, 768]);  clone_69 = None
        mm_38 = torch.ops.aten.mm.default(view_349, permute_236);  permute_236 = None
        permute_237 = torch.ops.aten.permute.default(view_349, [1, 0])
        mm_39 = torch.ops.aten.mm.default(permute_237, view_196);  permute_237 = view_196 = None
        permute_238 = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
        sum_63 = torch.ops.aten.sum.dim_IntList(view_349, [0], True);  view_349 = None
        view_350 = torch.ops.aten.view.default(sum_63, [768]);  sum_63 = None
        permute_239 = torch.ops.aten.permute.default(permute_238, [1, 0]);  permute_238 = None
        view_351 = torch.ops.aten.view.default(mm_38, [3, 5, 3072]);  mm_38 = None
        mul_260 = torch.ops.aten.mul.Tensor(add_72, 0.5);  add_72 = None
        mul_261 = torch.ops.aten.mul.Tensor(view_195, view_195)
        mul_262 = torch.ops.aten.mul.Tensor(mul_261, -0.5);  mul_261 = None
        exp_15 = torch.ops.aten.exp.default(mul_262);  mul_262 = None
        mul_263 = torch.ops.aten.mul.Tensor(exp_15, 0.3989422804014327);  exp_15 = None
        mul_264 = torch.ops.aten.mul.Tensor(view_195, mul_263);  view_195 = mul_263 = None
        add_120 = torch.ops.aten.add.Tensor(mul_260, mul_264);  mul_260 = mul_264 = None
        mul_265 = torch.ops.aten.mul.Tensor(view_351, add_120);  view_351 = add_120 = None
        view_352 = torch.ops.aten.view.default(mul_265, [15, 3072]);  mul_265 = None
        mm_40 = torch.ops.aten.mm.default(view_352, permute_240);  permute_240 = None
        permute_241 = torch.ops.aten.permute.default(view_352, [1, 0])
        mm_41 = torch.ops.aten.mm.default(permute_241, view_194);  permute_241 = view_194 = None
        permute_242 = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
        sum_64 = torch.ops.aten.sum.dim_IntList(view_352, [0], True);  view_352 = None
        view_353 = torch.ops.aten.view.default(sum_64, [3072]);  sum_64 = None
        permute_243 = torch.ops.aten.permute.default(permute_242, [1, 0]);  permute_242 = None
        view_354 = torch.ops.aten.view.default(mm_40, [3, 5, 768]);  mm_40 = None
        add_121 = torch.ops.aten.add.Tensor(mul_255, view_354);  mul_255 = view_354 = None
        mul_267 = torch.ops.aten.mul.Tensor(add_121, primals_142);  primals_142 = None
        mul_268 = torch.ops.aten.mul.Tensor(mul_267, 768)
        sum_65 = torch.ops.aten.sum.dim_IntList(mul_267, [2], True)
        mul_269 = torch.ops.aten.mul.Tensor(mul_267, mul_113);  mul_267 = None
        sum_66 = torch.ops.aten.sum.dim_IntList(mul_269, [2], True);  mul_269 = None
        mul_270 = torch.ops.aten.mul.Tensor(mul_113, sum_66);  sum_66 = None
        sub_64 = torch.ops.aten.sub.Tensor(mul_268, sum_65);  mul_268 = sum_65 = None
        sub_65 = torch.ops.aten.sub.Tensor(sub_64, mul_270);  sub_64 = mul_270 = None
        mul_271 = torch.ops.aten.mul.Tensor(div_34, sub_65);  div_34 = sub_65 = None
        mul_272 = torch.ops.aten.mul.Tensor(add_121, mul_113);  mul_113 = None
        sum_67 = torch.ops.aten.sum.dim_IntList(mul_272, [0, 1]);  mul_272 = None
        sum_68 = torch.ops.aten.sum.dim_IntList(add_121, [0, 1]);  add_121 = None
        convert_element_type_11 = torch.ops.prims.convert_element_type.default(gt_26, torch.float32);  gt_26 = None
        mul_273 = torch.ops.aten.mul.Tensor(convert_element_type_11, 1.1111111111111112);  convert_element_type_11 = None
        mul_274 = torch.ops.aten.mul.Tensor(mul_271, mul_273);  mul_273 = None
        clone_70 = torch.ops.aten.clone.default(mul_274, memory_format = torch.contiguous_format);  mul_274 = None
        view_355 = torch.ops.aten.view.default(clone_70, [15, 768]);  clone_70 = None
        mm_42 = torch.ops.aten.mm.default(view_355, permute_244);  permute_244 = None
        permute_245 = torch.ops.aten.permute.default(view_355, [1, 0])
        mm_43 = torch.ops.aten.mm.default(permute_245, view_192);  permute_245 = view_192 = None
        permute_246 = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
        sum_69 = torch.ops.aten.sum.dim_IntList(view_355, [0], True);  view_355 = None
        view_356 = torch.ops.aten.view.default(sum_69, [768]);  sum_69 = None
        permute_247 = torch.ops.aten.permute.default(permute_246, [1, 0]);  permute_246 = None
        view_357 = torch.ops.aten.view.default(mm_42, [3, 5, 768]);  mm_42 = None
        view_358 = torch.ops.aten.view.default(view_357, [3, 5, 12, 64]);  view_357 = None
        permute_248 = torch.ops.aten.permute.default(view_358, [0, 2, 1, 3]);  view_358 = None
        clone_71 = torch.ops.aten.clone.default(permute_248, memory_format = torch.contiguous_format);  permute_248 = None
        view_359 = torch.ops.aten.view.default(clone_71, [36, 5, 64]);  clone_71 = None
        bmm_36 = torch.ops.aten.bmm.default(permute_249, view_359);  permute_249 = None
        bmm_37 = torch.ops.aten.bmm.default(view_359, permute_250);  view_359 = permute_250 = None
        view_360 = torch.ops.aten.view.default(bmm_36, [3, 12, 5, 64]);  bmm_36 = None
        view_361 = torch.ops.aten.view.default(bmm_37, [3, 12, 5, 5]);  bmm_37 = None
        convert_element_type_12 = torch.ops.prims.convert_element_type.default(gt_25, torch.float32);  gt_25 = None
        mul_275 = torch.ops.aten.mul.Tensor(convert_element_type_12, 1.1111111111111112);  convert_element_type_12 = None
        mul_276 = torch.ops.aten.mul.Tensor(view_361, mul_275);  view_361 = mul_275 = None
        clone_72 = torch.ops.aten.clone.default(mul_276, memory_format = torch.contiguous_format);  mul_276 = None
        mul_277 = torch.ops.aten.mul.Tensor(clone_72, alias_17);  clone_72 = None
        sum_70 = torch.ops.aten.sum.dim_IntList(mul_277, [-1], True)
        mul_278 = torch.ops.aten.mul.Tensor(alias_17, sum_70);  alias_17 = sum_70 = None
        sub_66 = torch.ops.aten.sub.Tensor(mul_277, mul_278);  mul_277 = mul_278 = None
        div_35 = torch.ops.aten.div.Tensor(sub_66, 8.0);  sub_66 = None
        view_362 = torch.ops.aten.view.default(div_35, [36, 5, 5]);  div_35 = None
        bmm_38 = torch.ops.aten.bmm.default(permute_251, view_362);  permute_251 = None
        bmm_39 = torch.ops.aten.bmm.default(view_362, permute_252);  view_362 = permute_252 = None
        view_363 = torch.ops.aten.view.default(bmm_38, [3, 12, 64, 5]);  bmm_38 = None
        view_364 = torch.ops.aten.view.default(bmm_39, [3, 12, 5, 64]);  bmm_39 = None
        permute_253 = torch.ops.aten.permute.default(view_363, [0, 1, 3, 2]);  view_363 = None
        permute_254 = torch.ops.aten.permute.default(view_364, [0, 2, 1, 3]);  view_364 = None
        clone_73 = torch.ops.aten.clone.default(permute_254, memory_format = torch.contiguous_format);  permute_254 = None
        view_365 = torch.ops.aten.view.default(clone_73, [3, 5, 768]);  clone_73 = None
        permute_255 = torch.ops.aten.permute.default(view_360, [0, 2, 1, 3]);  view_360 = None
        clone_74 = torch.ops.aten.clone.default(permute_255, memory_format = torch.contiguous_format);  permute_255 = None
        view_366 = torch.ops.aten.view.default(clone_74, [3, 5, 768]);  clone_74 = None
        view_367 = torch.ops.aten.view.default(view_366, [15, 768]);  view_366 = None
        mm_44 = torch.ops.aten.mm.default(view_367, permute_256);  permute_256 = None
        permute_257 = torch.ops.aten.permute.default(view_367, [1, 0])
        mm_45 = torch.ops.aten.mm.default(permute_257, view_176);  permute_257 = None
        permute_258 = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
        sum_71 = torch.ops.aten.sum.dim_IntList(view_367, [0], True);  view_367 = None
        view_368 = torch.ops.aten.view.default(sum_71, [768]);  sum_71 = None
        permute_259 = torch.ops.aten.permute.default(permute_258, [1, 0]);  permute_258 = None
        view_369 = torch.ops.aten.view.default(mm_44, [3, 5, 768]);  mm_44 = None
        add_122 = torch.ops.aten.add.Tensor(mul_271, view_369);  mul_271 = view_369 = None
        permute_260 = torch.ops.aten.permute.default(permute_253, [0, 2, 1, 3]);  permute_253 = None
        view_370 = torch.ops.aten.view.default(permute_260, [3, 5, 768]);  permute_260 = None
        clone_75 = torch.ops.aten.clone.default(view_370, memory_format = torch.contiguous_format);  view_370 = None
        view_371 = torch.ops.aten.view.default(clone_75, [15, 768]);  clone_75 = None
        mm_46 = torch.ops.aten.mm.default(view_371, permute_261);  permute_261 = None
        permute_262 = torch.ops.aten.permute.default(view_371, [1, 0])
        mm_47 = torch.ops.aten.mm.default(permute_262, view_176);  permute_262 = None
        permute_263 = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
        sum_72 = torch.ops.aten.sum.dim_IntList(view_371, [0], True);  view_371 = None
        view_372 = torch.ops.aten.view.default(sum_72, [768]);  sum_72 = None
        permute_264 = torch.ops.aten.permute.default(permute_263, [1, 0]);  permute_263 = None
        view_373 = torch.ops.aten.view.default(mm_46, [3, 5, 768]);  mm_46 = None
        add_123 = torch.ops.aten.add.Tensor(add_122, view_373);  add_122 = view_373 = None
        view_374 = torch.ops.aten.view.default(view_365, [15, 768]);  view_365 = None
        mm_48 = torch.ops.aten.mm.default(view_374, permute_265);  permute_265 = None
        permute_266 = torch.ops.aten.permute.default(view_374, [1, 0])
        mm_49 = torch.ops.aten.mm.default(permute_266, view_176);  permute_266 = view_176 = None
        permute_267 = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
        sum_73 = torch.ops.aten.sum.dim_IntList(view_374, [0], True);  view_374 = None
        view_375 = torch.ops.aten.view.default(sum_73, [768]);  sum_73 = None
        permute_268 = torch.ops.aten.permute.default(permute_267, [1, 0]);  permute_267 = None
        view_376 = torch.ops.aten.view.default(mm_48, [3, 5, 768]);  mm_48 = None
        add_124 = torch.ops.aten.add.Tensor(add_123, view_376);  add_123 = view_376 = None
        mul_280 = torch.ops.aten.mul.Tensor(add_124, primals_132);  primals_132 = None
        mul_281 = torch.ops.aten.mul.Tensor(mul_280, 768)
        sum_74 = torch.ops.aten.sum.dim_IntList(mul_280, [2], True)
        mul_282 = torch.ops.aten.mul.Tensor(mul_280, mul_107);  mul_280 = None
        sum_75 = torch.ops.aten.sum.dim_IntList(mul_282, [2], True);  mul_282 = None
        mul_283 = torch.ops.aten.mul.Tensor(mul_107, sum_75);  sum_75 = None
        sub_68 = torch.ops.aten.sub.Tensor(mul_281, sum_74);  mul_281 = sum_74 = None
        sub_69 = torch.ops.aten.sub.Tensor(sub_68, mul_283);  sub_68 = mul_283 = None
        mul_284 = torch.ops.aten.mul.Tensor(div_36, sub_69);  div_36 = sub_69 = None
        mul_285 = torch.ops.aten.mul.Tensor(add_124, mul_107);  mul_107 = None
        sum_76 = torch.ops.aten.sum.dim_IntList(mul_285, [0, 1]);  mul_285 = None
        sum_77 = torch.ops.aten.sum.dim_IntList(add_124, [0, 1]);  add_124 = None
        convert_element_type_13 = torch.ops.prims.convert_element_type.default(gt_24, torch.float32);  gt_24 = None
        mul_286 = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
        mul_287 = torch.ops.aten.mul.Tensor(mul_284, mul_286);  mul_286 = None
        clone_76 = torch.ops.aten.clone.default(mul_287, memory_format = torch.contiguous_format);  mul_287 = None
        view_377 = torch.ops.aten.view.default(clone_76, [15, 768]);  clone_76 = None
        mm_50 = torch.ops.aten.mm.default(view_377, permute_269);  permute_269 = None
        permute_270 = torch.ops.aten.permute.default(view_377, [1, 0])
        mm_51 = torch.ops.aten.mm.default(permute_270, view_174);  permute_270 = view_174 = None
        permute_271 = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
        sum_78 = torch.ops.aten.sum.dim_IntList(view_377, [0], True);  view_377 = None
        view_378 = torch.ops.aten.view.default(sum_78, [768]);  sum_78 = None
        permute_272 = torch.ops.aten.permute.default(permute_271, [1, 0]);  permute_271 = None
        view_379 = torch.ops.aten.view.default(mm_50, [3, 5, 3072]);  mm_50 = None
        mul_289 = torch.ops.aten.mul.Tensor(add_64, 0.5);  add_64 = None
        mul_290 = torch.ops.aten.mul.Tensor(view_173, view_173)
        mul_291 = torch.ops.aten.mul.Tensor(mul_290, -0.5);  mul_290 = None
        exp_16 = torch.ops.aten.exp.default(mul_291);  mul_291 = None
        mul_292 = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
        mul_293 = torch.ops.aten.mul.Tensor(view_173, mul_292);  view_173 = mul_292 = None
        add_126 = torch.ops.aten.add.Tensor(mul_289, mul_293);  mul_289 = mul_293 = None
        mul_294 = torch.ops.aten.mul.Tensor(view_379, add_126);  view_379 = add_126 = None
        view_380 = torch.ops.aten.view.default(mul_294, [15, 3072]);  mul_294 = None
        mm_52 = torch.ops.aten.mm.default(view_380, permute_273);  permute_273 = None
        permute_274 = torch.ops.aten.permute.default(view_380, [1, 0])
        mm_53 = torch.ops.aten.mm.default(permute_274, view_172);  permute_274 = view_172 = None
        permute_275 = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
        sum_79 = torch.ops.aten.sum.dim_IntList(view_380, [0], True);  view_380 = None
        view_381 = torch.ops.aten.view.default(sum_79, [3072]);  sum_79 = None
        permute_276 = torch.ops.aten.permute.default(permute_275, [1, 0]);  permute_275 = None
        view_382 = torch.ops.aten.view.default(mm_52, [3, 5, 768]);  mm_52 = None
        add_127 = torch.ops.aten.add.Tensor(mul_284, view_382);  mul_284 = view_382 = None
        mul_296 = torch.ops.aten.mul.Tensor(add_127, primals_126);  primals_126 = None
        mul_297 = torch.ops.aten.mul.Tensor(mul_296, 768)
        sum_80 = torch.ops.aten.sum.dim_IntList(mul_296, [2], True)
        mul_298 = torch.ops.aten.mul.Tensor(mul_296, mul_100);  mul_296 = None
        sum_81 = torch.ops.aten.sum.dim_IntList(mul_298, [2], True);  mul_298 = None
        mul_299 = torch.ops.aten.mul.Tensor(mul_100, sum_81);  sum_81 = None
        sub_71 = torch.ops.aten.sub.Tensor(mul_297, sum_80);  mul_297 = sum_80 = None
        sub_72 = torch.ops.aten.sub.Tensor(sub_71, mul_299);  sub_71 = mul_299 = None
        mul_300 = torch.ops.aten.mul.Tensor(div_37, sub_72);  div_37 = sub_72 = None
        mul_301 = torch.ops.aten.mul.Tensor(add_127, mul_100);  mul_100 = None
        sum_82 = torch.ops.aten.sum.dim_IntList(mul_301, [0, 1]);  mul_301 = None
        sum_83 = torch.ops.aten.sum.dim_IntList(add_127, [0, 1]);  add_127 = None
        convert_element_type_14 = torch.ops.prims.convert_element_type.default(gt_23, torch.float32);  gt_23 = None
        mul_302 = torch.ops.aten.mul.Tensor(convert_element_type_14, 1.1111111111111112);  convert_element_type_14 = None
        mul_303 = torch.ops.aten.mul.Tensor(mul_300, mul_302);  mul_302 = None
        clone_77 = torch.ops.aten.clone.default(mul_303, memory_format = torch.contiguous_format);  mul_303 = None
        view_383 = torch.ops.aten.view.default(clone_77, [15, 768]);  clone_77 = None
        mm_54 = torch.ops.aten.mm.default(view_383, permute_277);  permute_277 = None
        permute_278 = torch.ops.aten.permute.default(view_383, [1, 0])
        mm_55 = torch.ops.aten.mm.default(permute_278, view_170);  permute_278 = view_170 = None
        permute_279 = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
        sum_84 = torch.ops.aten.sum.dim_IntList(view_383, [0], True);  view_383 = None
        view_384 = torch.ops.aten.view.default(sum_84, [768]);  sum_84 = None
        permute_280 = torch.ops.aten.permute.default(permute_279, [1, 0]);  permute_279 = None
        view_385 = torch.ops.aten.view.default(mm_54, [3, 5, 768]);  mm_54 = None
        view_386 = torch.ops.aten.view.default(view_385, [3, 5, 12, 64]);  view_385 = None
        permute_281 = torch.ops.aten.permute.default(view_386, [0, 2, 1, 3]);  view_386 = None
        clone_78 = torch.ops.aten.clone.default(permute_281, memory_format = torch.contiguous_format);  permute_281 = None
        view_387 = torch.ops.aten.view.default(clone_78, [36, 5, 64]);  clone_78 = None
        bmm_40 = torch.ops.aten.bmm.default(permute_282, view_387);  permute_282 = None
        bmm_41 = torch.ops.aten.bmm.default(view_387, permute_283);  view_387 = permute_283 = None
        view_388 = torch.ops.aten.view.default(bmm_40, [3, 12, 5, 64]);  bmm_40 = None
        view_389 = torch.ops.aten.view.default(bmm_41, [3, 12, 5, 5]);  bmm_41 = None
        convert_element_type_15 = torch.ops.prims.convert_element_type.default(gt_22, torch.float32);  gt_22 = None
        mul_304 = torch.ops.aten.mul.Tensor(convert_element_type_15, 1.1111111111111112);  convert_element_type_15 = None
        mul_305 = torch.ops.aten.mul.Tensor(view_389, mul_304);  view_389 = mul_304 = None
        clone_79 = torch.ops.aten.clone.default(mul_305, memory_format = torch.contiguous_format);  mul_305 = None
        mul_306 = torch.ops.aten.mul.Tensor(clone_79, alias_18);  clone_79 = None
        sum_85 = torch.ops.aten.sum.dim_IntList(mul_306, [-1], True)
        mul_307 = torch.ops.aten.mul.Tensor(alias_18, sum_85);  alias_18 = sum_85 = None
        sub_73 = torch.ops.aten.sub.Tensor(mul_306, mul_307);  mul_306 = mul_307 = None
        div_38 = torch.ops.aten.div.Tensor(sub_73, 8.0);  sub_73 = None
        view_390 = torch.ops.aten.view.default(div_38, [36, 5, 5]);  div_38 = None
        bmm_42 = torch.ops.aten.bmm.default(permute_284, view_390);  permute_284 = None
        bmm_43 = torch.ops.aten.bmm.default(view_390, permute_285);  view_390 = permute_285 = None
        view_391 = torch.ops.aten.view.default(bmm_42, [3, 12, 64, 5]);  bmm_42 = None
        view_392 = torch.ops.aten.view.default(bmm_43, [3, 12, 5, 64]);  bmm_43 = None
        permute_286 = torch.ops.aten.permute.default(view_391, [0, 1, 3, 2]);  view_391 = None
        permute_287 = torch.ops.aten.permute.default(view_392, [0, 2, 1, 3]);  view_392 = None
        clone_80 = torch.ops.aten.clone.default(permute_287, memory_format = torch.contiguous_format);  permute_287 = None
        view_393 = torch.ops.aten.view.default(clone_80, [3, 5, 768]);  clone_80 = None
        permute_288 = torch.ops.aten.permute.default(view_388, [0, 2, 1, 3]);  view_388 = None
        clone_81 = torch.ops.aten.clone.default(permute_288, memory_format = torch.contiguous_format);  permute_288 = None
        view_394 = torch.ops.aten.view.default(clone_81, [3, 5, 768]);  clone_81 = None
        view_395 = torch.ops.aten.view.default(view_394, [15, 768]);  view_394 = None
        mm_56 = torch.ops.aten.mm.default(view_395, permute_289);  permute_289 = None
        permute_290 = torch.ops.aten.permute.default(view_395, [1, 0])
        mm_57 = torch.ops.aten.mm.default(permute_290, view_154);  permute_290 = None
        permute_291 = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
        sum_86 = torch.ops.aten.sum.dim_IntList(view_395, [0], True);  view_395 = None
        view_396 = torch.ops.aten.view.default(sum_86, [768]);  sum_86 = None
        permute_292 = torch.ops.aten.permute.default(permute_291, [1, 0]);  permute_291 = None
        view_397 = torch.ops.aten.view.default(mm_56, [3, 5, 768]);  mm_56 = None
        add_128 = torch.ops.aten.add.Tensor(mul_300, view_397);  mul_300 = view_397 = None
        permute_293 = torch.ops.aten.permute.default(permute_286, [0, 2, 1, 3]);  permute_286 = None
        view_398 = torch.ops.aten.view.default(permute_293, [3, 5, 768]);  permute_293 = None
        clone_82 = torch.ops.aten.clone.default(view_398, memory_format = torch.contiguous_format);  view_398 = None
        view_399 = torch.ops.aten.view.default(clone_82, [15, 768]);  clone_82 = None
        mm_58 = torch.ops.aten.mm.default(view_399, permute_294);  permute_294 = None
        permute_295 = torch.ops.aten.permute.default(view_399, [1, 0])
        mm_59 = torch.ops.aten.mm.default(permute_295, view_154);  permute_295 = None
        permute_296 = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
        sum_87 = torch.ops.aten.sum.dim_IntList(view_399, [0], True);  view_399 = None
        view_400 = torch.ops.aten.view.default(sum_87, [768]);  sum_87 = None
        permute_297 = torch.ops.aten.permute.default(permute_296, [1, 0]);  permute_296 = None
        view_401 = torch.ops.aten.view.default(mm_58, [3, 5, 768]);  mm_58 = None
        add_129 = torch.ops.aten.add.Tensor(add_128, view_401);  add_128 = view_401 = None
        view_402 = torch.ops.aten.view.default(view_393, [15, 768]);  view_393 = None
        mm_60 = torch.ops.aten.mm.default(view_402, permute_298);  permute_298 = None
        permute_299 = torch.ops.aten.permute.default(view_402, [1, 0])
        mm_61 = torch.ops.aten.mm.default(permute_299, view_154);  permute_299 = view_154 = None
        permute_300 = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
        sum_88 = torch.ops.aten.sum.dim_IntList(view_402, [0], True);  view_402 = None
        view_403 = torch.ops.aten.view.default(sum_88, [768]);  sum_88 = None
        permute_301 = torch.ops.aten.permute.default(permute_300, [1, 0]);  permute_300 = None
        view_404 = torch.ops.aten.view.default(mm_60, [3, 5, 768]);  mm_60 = None
        add_130 = torch.ops.aten.add.Tensor(add_129, view_404);  add_129 = view_404 = None
        mul_309 = torch.ops.aten.mul.Tensor(add_130, primals_116);  primals_116 = None
        mul_310 = torch.ops.aten.mul.Tensor(mul_309, 768)
        sum_89 = torch.ops.aten.sum.dim_IntList(mul_309, [2], True)
        mul_311 = torch.ops.aten.mul.Tensor(mul_309, mul_94);  mul_309 = None
        sum_90 = torch.ops.aten.sum.dim_IntList(mul_311, [2], True);  mul_311 = None
        mul_312 = torch.ops.aten.mul.Tensor(mul_94, sum_90);  sum_90 = None
        sub_75 = torch.ops.aten.sub.Tensor(mul_310, sum_89);  mul_310 = sum_89 = None
        sub_76 = torch.ops.aten.sub.Tensor(sub_75, mul_312);  sub_75 = mul_312 = None
        mul_313 = torch.ops.aten.mul.Tensor(div_39, sub_76);  div_39 = sub_76 = None
        mul_314 = torch.ops.aten.mul.Tensor(add_130, mul_94);  mul_94 = None
        sum_91 = torch.ops.aten.sum.dim_IntList(mul_314, [0, 1]);  mul_314 = None
        sum_92 = torch.ops.aten.sum.dim_IntList(add_130, [0, 1]);  add_130 = None
        convert_element_type_16 = torch.ops.prims.convert_element_type.default(gt_21, torch.float32);  gt_21 = None
        mul_315 = torch.ops.aten.mul.Tensor(convert_element_type_16, 1.1111111111111112);  convert_element_type_16 = None
        mul_316 = torch.ops.aten.mul.Tensor(mul_313, mul_315);  mul_315 = None
        clone_83 = torch.ops.aten.clone.default(mul_316, memory_format = torch.contiguous_format);  mul_316 = None
        view_405 = torch.ops.aten.view.default(clone_83, [15, 768]);  clone_83 = None
        mm_62 = torch.ops.aten.mm.default(view_405, permute_302);  permute_302 = None
        permute_303 = torch.ops.aten.permute.default(view_405, [1, 0])
        mm_63 = torch.ops.aten.mm.default(permute_303, view_152);  permute_303 = view_152 = None
        permute_304 = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
        sum_93 = torch.ops.aten.sum.dim_IntList(view_405, [0], True);  view_405 = None
        view_406 = torch.ops.aten.view.default(sum_93, [768]);  sum_93 = None
        permute_305 = torch.ops.aten.permute.default(permute_304, [1, 0]);  permute_304 = None
        view_407 = torch.ops.aten.view.default(mm_62, [3, 5, 3072]);  mm_62 = None
        mul_318 = torch.ops.aten.mul.Tensor(add_56, 0.5);  add_56 = None
        mul_319 = torch.ops.aten.mul.Tensor(view_151, view_151)
        mul_320 = torch.ops.aten.mul.Tensor(mul_319, -0.5);  mul_319 = None
        exp_17 = torch.ops.aten.exp.default(mul_320);  mul_320 = None
        mul_321 = torch.ops.aten.mul.Tensor(exp_17, 0.3989422804014327);  exp_17 = None
        mul_322 = torch.ops.aten.mul.Tensor(view_151, mul_321);  view_151 = mul_321 = None
        add_132 = torch.ops.aten.add.Tensor(mul_318, mul_322);  mul_318 = mul_322 = None
        mul_323 = torch.ops.aten.mul.Tensor(view_407, add_132);  view_407 = add_132 = None
        view_408 = torch.ops.aten.view.default(mul_323, [15, 3072]);  mul_323 = None
        mm_64 = torch.ops.aten.mm.default(view_408, permute_306);  permute_306 = None
        permute_307 = torch.ops.aten.permute.default(view_408, [1, 0])
        mm_65 = torch.ops.aten.mm.default(permute_307, view_150);  permute_307 = view_150 = None
        permute_308 = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
        sum_94 = torch.ops.aten.sum.dim_IntList(view_408, [0], True);  view_408 = None
        view_409 = torch.ops.aten.view.default(sum_94, [3072]);  sum_94 = None
        permute_309 = torch.ops.aten.permute.default(permute_308, [1, 0]);  permute_308 = None
        view_410 = torch.ops.aten.view.default(mm_64, [3, 5, 768]);  mm_64 = None
        add_133 = torch.ops.aten.add.Tensor(mul_313, view_410);  mul_313 = view_410 = None
        mul_325 = torch.ops.aten.mul.Tensor(add_133, primals_110);  primals_110 = None
        mul_326 = torch.ops.aten.mul.Tensor(mul_325, 768)
        sum_95 = torch.ops.aten.sum.dim_IntList(mul_325, [2], True)
        mul_327 = torch.ops.aten.mul.Tensor(mul_325, mul_87);  mul_325 = None
        sum_96 = torch.ops.aten.sum.dim_IntList(mul_327, [2], True);  mul_327 = None
        mul_328 = torch.ops.aten.mul.Tensor(mul_87, sum_96);  sum_96 = None
        sub_78 = torch.ops.aten.sub.Tensor(mul_326, sum_95);  mul_326 = sum_95 = None
        sub_79 = torch.ops.aten.sub.Tensor(sub_78, mul_328);  sub_78 = mul_328 = None
        mul_329 = torch.ops.aten.mul.Tensor(div_40, sub_79);  div_40 = sub_79 = None
        mul_330 = torch.ops.aten.mul.Tensor(add_133, mul_87);  mul_87 = None
        sum_97 = torch.ops.aten.sum.dim_IntList(mul_330, [0, 1]);  mul_330 = None
        sum_98 = torch.ops.aten.sum.dim_IntList(add_133, [0, 1]);  add_133 = None
        convert_element_type_17 = torch.ops.prims.convert_element_type.default(gt_20, torch.float32);  gt_20 = None
        mul_331 = torch.ops.aten.mul.Tensor(convert_element_type_17, 1.1111111111111112);  convert_element_type_17 = None
        mul_332 = torch.ops.aten.mul.Tensor(mul_329, mul_331);  mul_331 = None
        clone_84 = torch.ops.aten.clone.default(mul_332, memory_format = torch.contiguous_format);  mul_332 = None
        view_411 = torch.ops.aten.view.default(clone_84, [15, 768]);  clone_84 = None
        mm_66 = torch.ops.aten.mm.default(view_411, permute_310);  permute_310 = None
        permute_311 = torch.ops.aten.permute.default(view_411, [1, 0])
        mm_67 = torch.ops.aten.mm.default(permute_311, view_148);  permute_311 = view_148 = None
        permute_312 = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
        sum_99 = torch.ops.aten.sum.dim_IntList(view_411, [0], True);  view_411 = None
        view_412 = torch.ops.aten.view.default(sum_99, [768]);  sum_99 = None
        permute_313 = torch.ops.aten.permute.default(permute_312, [1, 0]);  permute_312 = None
        view_413 = torch.ops.aten.view.default(mm_66, [3, 5, 768]);  mm_66 = None
        view_414 = torch.ops.aten.view.default(view_413, [3, 5, 12, 64]);  view_413 = None
        permute_314 = torch.ops.aten.permute.default(view_414, [0, 2, 1, 3]);  view_414 = None
        clone_85 = torch.ops.aten.clone.default(permute_314, memory_format = torch.contiguous_format);  permute_314 = None
        view_415 = torch.ops.aten.view.default(clone_85, [36, 5, 64]);  clone_85 = None
        bmm_44 = torch.ops.aten.bmm.default(permute_315, view_415);  permute_315 = None
        bmm_45 = torch.ops.aten.bmm.default(view_415, permute_316);  view_415 = permute_316 = None
        view_416 = torch.ops.aten.view.default(bmm_44, [3, 12, 5, 64]);  bmm_44 = None
        view_417 = torch.ops.aten.view.default(bmm_45, [3, 12, 5, 5]);  bmm_45 = None
        convert_element_type_18 = torch.ops.prims.convert_element_type.default(gt_19, torch.float32);  gt_19 = None
        mul_333 = torch.ops.aten.mul.Tensor(convert_element_type_18, 1.1111111111111112);  convert_element_type_18 = None
        mul_334 = torch.ops.aten.mul.Tensor(view_417, mul_333);  view_417 = mul_333 = None
        clone_86 = torch.ops.aten.clone.default(mul_334, memory_format = torch.contiguous_format);  mul_334 = None
        mul_335 = torch.ops.aten.mul.Tensor(clone_86, alias_19);  clone_86 = None
        sum_100 = torch.ops.aten.sum.dim_IntList(mul_335, [-1], True)
        mul_336 = torch.ops.aten.mul.Tensor(alias_19, sum_100);  alias_19 = sum_100 = None
        sub_80 = torch.ops.aten.sub.Tensor(mul_335, mul_336);  mul_335 = mul_336 = None
        div_41 = torch.ops.aten.div.Tensor(sub_80, 8.0);  sub_80 = None
        view_418 = torch.ops.aten.view.default(div_41, [36, 5, 5]);  div_41 = None
        bmm_46 = torch.ops.aten.bmm.default(permute_317, view_418);  permute_317 = None
        bmm_47 = torch.ops.aten.bmm.default(view_418, permute_318);  view_418 = permute_318 = None
        view_419 = torch.ops.aten.view.default(bmm_46, [3, 12, 64, 5]);  bmm_46 = None
        view_420 = torch.ops.aten.view.default(bmm_47, [3, 12, 5, 64]);  bmm_47 = None
        permute_319 = torch.ops.aten.permute.default(view_419, [0, 1, 3, 2]);  view_419 = None
        permute_320 = torch.ops.aten.permute.default(view_420, [0, 2, 1, 3]);  view_420 = None
        clone_87 = torch.ops.aten.clone.default(permute_320, memory_format = torch.contiguous_format);  permute_320 = None
        view_421 = torch.ops.aten.view.default(clone_87, [3, 5, 768]);  clone_87 = None
        permute_321 = torch.ops.aten.permute.default(view_416, [0, 2, 1, 3]);  view_416 = None
        clone_88 = torch.ops.aten.clone.default(permute_321, memory_format = torch.contiguous_format);  permute_321 = None
        view_422 = torch.ops.aten.view.default(clone_88, [3, 5, 768]);  clone_88 = None
        view_423 = torch.ops.aten.view.default(view_422, [15, 768]);  view_422 = None
        mm_68 = torch.ops.aten.mm.default(view_423, permute_322);  permute_322 = None
        permute_323 = torch.ops.aten.permute.default(view_423, [1, 0])
        mm_69 = torch.ops.aten.mm.default(permute_323, view_132);  permute_323 = None
        permute_324 = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
        sum_101 = torch.ops.aten.sum.dim_IntList(view_423, [0], True);  view_423 = None
        view_424 = torch.ops.aten.view.default(sum_101, [768]);  sum_101 = None
        permute_325 = torch.ops.aten.permute.default(permute_324, [1, 0]);  permute_324 = None
        view_425 = torch.ops.aten.view.default(mm_68, [3, 5, 768]);  mm_68 = None
        add_134 = torch.ops.aten.add.Tensor(mul_329, view_425);  mul_329 = view_425 = None
        permute_326 = torch.ops.aten.permute.default(permute_319, [0, 2, 1, 3]);  permute_319 = None
        view_426 = torch.ops.aten.view.default(permute_326, [3, 5, 768]);  permute_326 = None
        clone_89 = torch.ops.aten.clone.default(view_426, memory_format = torch.contiguous_format);  view_426 = None
        view_427 = torch.ops.aten.view.default(clone_89, [15, 768]);  clone_89 = None
        mm_70 = torch.ops.aten.mm.default(view_427, permute_327);  permute_327 = None
        permute_328 = torch.ops.aten.permute.default(view_427, [1, 0])
        mm_71 = torch.ops.aten.mm.default(permute_328, view_132);  permute_328 = None
        permute_329 = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
        sum_102 = torch.ops.aten.sum.dim_IntList(view_427, [0], True);  view_427 = None
        view_428 = torch.ops.aten.view.default(sum_102, [768]);  sum_102 = None
        permute_330 = torch.ops.aten.permute.default(permute_329, [1, 0]);  permute_329 = None
        view_429 = torch.ops.aten.view.default(mm_70, [3, 5, 768]);  mm_70 = None
        add_135 = torch.ops.aten.add.Tensor(add_134, view_429);  add_134 = view_429 = None
        view_430 = torch.ops.aten.view.default(view_421, [15, 768]);  view_421 = None
        mm_72 = torch.ops.aten.mm.default(view_430, permute_331);  permute_331 = None
        permute_332 = torch.ops.aten.permute.default(view_430, [1, 0])
        mm_73 = torch.ops.aten.mm.default(permute_332, view_132);  permute_332 = view_132 = None
        permute_333 = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
        sum_103 = torch.ops.aten.sum.dim_IntList(view_430, [0], True);  view_430 = None
        view_431 = torch.ops.aten.view.default(sum_103, [768]);  sum_103 = None
        permute_334 = torch.ops.aten.permute.default(permute_333, [1, 0]);  permute_333 = None
        view_432 = torch.ops.aten.view.default(mm_72, [3, 5, 768]);  mm_72 = None
        add_136 = torch.ops.aten.add.Tensor(add_135, view_432);  add_135 = view_432 = None
        mul_338 = torch.ops.aten.mul.Tensor(add_136, primals_100);  primals_100 = None
        mul_339 = torch.ops.aten.mul.Tensor(mul_338, 768)
        sum_104 = torch.ops.aten.sum.dim_IntList(mul_338, [2], True)
        mul_340 = torch.ops.aten.mul.Tensor(mul_338, mul_81);  mul_338 = None
        sum_105 = torch.ops.aten.sum.dim_IntList(mul_340, [2], True);  mul_340 = None
        mul_341 = torch.ops.aten.mul.Tensor(mul_81, sum_105);  sum_105 = None
        sub_82 = torch.ops.aten.sub.Tensor(mul_339, sum_104);  mul_339 = sum_104 = None
        sub_83 = torch.ops.aten.sub.Tensor(sub_82, mul_341);  sub_82 = mul_341 = None
        mul_342 = torch.ops.aten.mul.Tensor(div_42, sub_83);  div_42 = sub_83 = None
        mul_343 = torch.ops.aten.mul.Tensor(add_136, mul_81);  mul_81 = None
        sum_106 = torch.ops.aten.sum.dim_IntList(mul_343, [0, 1]);  mul_343 = None
        sum_107 = torch.ops.aten.sum.dim_IntList(add_136, [0, 1]);  add_136 = None
        convert_element_type_19 = torch.ops.prims.convert_element_type.default(gt_18, torch.float32);  gt_18 = None
        mul_344 = torch.ops.aten.mul.Tensor(convert_element_type_19, 1.1111111111111112);  convert_element_type_19 = None
        mul_345 = torch.ops.aten.mul.Tensor(mul_342, mul_344);  mul_344 = None
        clone_90 = torch.ops.aten.clone.default(mul_345, memory_format = torch.contiguous_format);  mul_345 = None
        view_433 = torch.ops.aten.view.default(clone_90, [15, 768]);  clone_90 = None
        mm_74 = torch.ops.aten.mm.default(view_433, permute_335);  permute_335 = None
        permute_336 = torch.ops.aten.permute.default(view_433, [1, 0])
        mm_75 = torch.ops.aten.mm.default(permute_336, view_130);  permute_336 = view_130 = None
        permute_337 = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
        sum_108 = torch.ops.aten.sum.dim_IntList(view_433, [0], True);  view_433 = None
        view_434 = torch.ops.aten.view.default(sum_108, [768]);  sum_108 = None
        permute_338 = torch.ops.aten.permute.default(permute_337, [1, 0]);  permute_337 = None
        view_435 = torch.ops.aten.view.default(mm_74, [3, 5, 3072]);  mm_74 = None
        mul_347 = torch.ops.aten.mul.Tensor(add_48, 0.5);  add_48 = None
        mul_348 = torch.ops.aten.mul.Tensor(view_129, view_129)
        mul_349 = torch.ops.aten.mul.Tensor(mul_348, -0.5);  mul_348 = None
        exp_18 = torch.ops.aten.exp.default(mul_349);  mul_349 = None
        mul_350 = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
        mul_351 = torch.ops.aten.mul.Tensor(view_129, mul_350);  view_129 = mul_350 = None
        add_138 = torch.ops.aten.add.Tensor(mul_347, mul_351);  mul_347 = mul_351 = None
        mul_352 = torch.ops.aten.mul.Tensor(view_435, add_138);  view_435 = add_138 = None
        view_436 = torch.ops.aten.view.default(mul_352, [15, 3072]);  mul_352 = None
        mm_76 = torch.ops.aten.mm.default(view_436, permute_339);  permute_339 = None
        permute_340 = torch.ops.aten.permute.default(view_436, [1, 0])
        mm_77 = torch.ops.aten.mm.default(permute_340, view_128);  permute_340 = view_128 = None
        permute_341 = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
        sum_109 = torch.ops.aten.sum.dim_IntList(view_436, [0], True);  view_436 = None
        view_437 = torch.ops.aten.view.default(sum_109, [3072]);  sum_109 = None
        permute_342 = torch.ops.aten.permute.default(permute_341, [1, 0]);  permute_341 = None
        view_438 = torch.ops.aten.view.default(mm_76, [3, 5, 768]);  mm_76 = None
        add_139 = torch.ops.aten.add.Tensor(mul_342, view_438);  mul_342 = view_438 = None
        mul_354 = torch.ops.aten.mul.Tensor(add_139, primals_94);  primals_94 = None
        mul_355 = torch.ops.aten.mul.Tensor(mul_354, 768)
        sum_110 = torch.ops.aten.sum.dim_IntList(mul_354, [2], True)
        mul_356 = torch.ops.aten.mul.Tensor(mul_354, mul_74);  mul_354 = None
        sum_111 = torch.ops.aten.sum.dim_IntList(mul_356, [2], True);  mul_356 = None
        mul_357 = torch.ops.aten.mul.Tensor(mul_74, sum_111);  sum_111 = None
        sub_85 = torch.ops.aten.sub.Tensor(mul_355, sum_110);  mul_355 = sum_110 = None
        sub_86 = torch.ops.aten.sub.Tensor(sub_85, mul_357);  sub_85 = mul_357 = None
        mul_358 = torch.ops.aten.mul.Tensor(div_43, sub_86);  div_43 = sub_86 = None
        mul_359 = torch.ops.aten.mul.Tensor(add_139, mul_74);  mul_74 = None
        sum_112 = torch.ops.aten.sum.dim_IntList(mul_359, [0, 1]);  mul_359 = None
        sum_113 = torch.ops.aten.sum.dim_IntList(add_139, [0, 1]);  add_139 = None
        convert_element_type_20 = torch.ops.prims.convert_element_type.default(gt_17, torch.float32);  gt_17 = None
        mul_360 = torch.ops.aten.mul.Tensor(convert_element_type_20, 1.1111111111111112);  convert_element_type_20 = None
        mul_361 = torch.ops.aten.mul.Tensor(mul_358, mul_360);  mul_360 = None
        clone_91 = torch.ops.aten.clone.default(mul_361, memory_format = torch.contiguous_format);  mul_361 = None
        view_439 = torch.ops.aten.view.default(clone_91, [15, 768]);  clone_91 = None
        mm_78 = torch.ops.aten.mm.default(view_439, permute_343);  permute_343 = None
        permute_344 = torch.ops.aten.permute.default(view_439, [1, 0])
        mm_79 = torch.ops.aten.mm.default(permute_344, view_126);  permute_344 = view_126 = None
        permute_345 = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
        sum_114 = torch.ops.aten.sum.dim_IntList(view_439, [0], True);  view_439 = None
        view_440 = torch.ops.aten.view.default(sum_114, [768]);  sum_114 = None
        permute_346 = torch.ops.aten.permute.default(permute_345, [1, 0]);  permute_345 = None
        view_441 = torch.ops.aten.view.default(mm_78, [3, 5, 768]);  mm_78 = None
        view_442 = torch.ops.aten.view.default(view_441, [3, 5, 12, 64]);  view_441 = None
        permute_347 = torch.ops.aten.permute.default(view_442, [0, 2, 1, 3]);  view_442 = None
        clone_92 = torch.ops.aten.clone.default(permute_347, memory_format = torch.contiguous_format);  permute_347 = None
        view_443 = torch.ops.aten.view.default(clone_92, [36, 5, 64]);  clone_92 = None
        bmm_48 = torch.ops.aten.bmm.default(permute_348, view_443);  permute_348 = None
        bmm_49 = torch.ops.aten.bmm.default(view_443, permute_349);  view_443 = permute_349 = None
        view_444 = torch.ops.aten.view.default(bmm_48, [3, 12, 5, 64]);  bmm_48 = None
        view_445 = torch.ops.aten.view.default(bmm_49, [3, 12, 5, 5]);  bmm_49 = None
        convert_element_type_21 = torch.ops.prims.convert_element_type.default(gt_16, torch.float32);  gt_16 = None
        mul_362 = torch.ops.aten.mul.Tensor(convert_element_type_21, 1.1111111111111112);  convert_element_type_21 = None
        mul_363 = torch.ops.aten.mul.Tensor(view_445, mul_362);  view_445 = mul_362 = None
        clone_93 = torch.ops.aten.clone.default(mul_363, memory_format = torch.contiguous_format);  mul_363 = None
        mul_364 = torch.ops.aten.mul.Tensor(clone_93, alias_20);  clone_93 = None
        sum_115 = torch.ops.aten.sum.dim_IntList(mul_364, [-1], True)
        mul_365 = torch.ops.aten.mul.Tensor(alias_20, sum_115);  alias_20 = sum_115 = None
        sub_87 = torch.ops.aten.sub.Tensor(mul_364, mul_365);  mul_364 = mul_365 = None
        div_44 = torch.ops.aten.div.Tensor(sub_87, 8.0);  sub_87 = None
        view_446 = torch.ops.aten.view.default(div_44, [36, 5, 5]);  div_44 = None
        bmm_50 = torch.ops.aten.bmm.default(permute_350, view_446);  permute_350 = None
        bmm_51 = torch.ops.aten.bmm.default(view_446, permute_351);  view_446 = permute_351 = None
        view_447 = torch.ops.aten.view.default(bmm_50, [3, 12, 64, 5]);  bmm_50 = None
        view_448 = torch.ops.aten.view.default(bmm_51, [3, 12, 5, 64]);  bmm_51 = None
        permute_352 = torch.ops.aten.permute.default(view_447, [0, 1, 3, 2]);  view_447 = None
        permute_353 = torch.ops.aten.permute.default(view_448, [0, 2, 1, 3]);  view_448 = None
        clone_94 = torch.ops.aten.clone.default(permute_353, memory_format = torch.contiguous_format);  permute_353 = None
        view_449 = torch.ops.aten.view.default(clone_94, [3, 5, 768]);  clone_94 = None
        permute_354 = torch.ops.aten.permute.default(view_444, [0, 2, 1, 3]);  view_444 = None
        clone_95 = torch.ops.aten.clone.default(permute_354, memory_format = torch.contiguous_format);  permute_354 = None
        view_450 = torch.ops.aten.view.default(clone_95, [3, 5, 768]);  clone_95 = None
        view_451 = torch.ops.aten.view.default(view_450, [15, 768]);  view_450 = None
        mm_80 = torch.ops.aten.mm.default(view_451, permute_355);  permute_355 = None
        permute_356 = torch.ops.aten.permute.default(view_451, [1, 0])
        mm_81 = torch.ops.aten.mm.default(permute_356, view_110);  permute_356 = None
        permute_357 = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
        sum_116 = torch.ops.aten.sum.dim_IntList(view_451, [0], True);  view_451 = None
        view_452 = torch.ops.aten.view.default(sum_116, [768]);  sum_116 = None
        permute_358 = torch.ops.aten.permute.default(permute_357, [1, 0]);  permute_357 = None
        view_453 = torch.ops.aten.view.default(mm_80, [3, 5, 768]);  mm_80 = None
        add_140 = torch.ops.aten.add.Tensor(mul_358, view_453);  mul_358 = view_453 = None
        permute_359 = torch.ops.aten.permute.default(permute_352, [0, 2, 1, 3]);  permute_352 = None
        view_454 = torch.ops.aten.view.default(permute_359, [3, 5, 768]);  permute_359 = None
        clone_96 = torch.ops.aten.clone.default(view_454, memory_format = torch.contiguous_format);  view_454 = None
        view_455 = torch.ops.aten.view.default(clone_96, [15, 768]);  clone_96 = None
        mm_82 = torch.ops.aten.mm.default(view_455, permute_360);  permute_360 = None
        permute_361 = torch.ops.aten.permute.default(view_455, [1, 0])
        mm_83 = torch.ops.aten.mm.default(permute_361, view_110);  permute_361 = None
        permute_362 = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
        sum_117 = torch.ops.aten.sum.dim_IntList(view_455, [0], True);  view_455 = None
        view_456 = torch.ops.aten.view.default(sum_117, [768]);  sum_117 = None
        permute_363 = torch.ops.aten.permute.default(permute_362, [1, 0]);  permute_362 = None
        view_457 = torch.ops.aten.view.default(mm_82, [3, 5, 768]);  mm_82 = None
        add_141 = torch.ops.aten.add.Tensor(add_140, view_457);  add_140 = view_457 = None
        view_458 = torch.ops.aten.view.default(view_449, [15, 768]);  view_449 = None
        mm_84 = torch.ops.aten.mm.default(view_458, permute_364);  permute_364 = None
        permute_365 = torch.ops.aten.permute.default(view_458, [1, 0])
        mm_85 = torch.ops.aten.mm.default(permute_365, view_110);  permute_365 = view_110 = None
        permute_366 = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
        sum_118 = torch.ops.aten.sum.dim_IntList(view_458, [0], True);  view_458 = None
        view_459 = torch.ops.aten.view.default(sum_118, [768]);  sum_118 = None
        permute_367 = torch.ops.aten.permute.default(permute_366, [1, 0]);  permute_366 = None
        view_460 = torch.ops.aten.view.default(mm_84, [3, 5, 768]);  mm_84 = None
        add_142 = torch.ops.aten.add.Tensor(add_141, view_460);  add_141 = view_460 = None
        mul_367 = torch.ops.aten.mul.Tensor(add_142, primals_84);  primals_84 = None
        mul_368 = torch.ops.aten.mul.Tensor(mul_367, 768)
        sum_119 = torch.ops.aten.sum.dim_IntList(mul_367, [2], True)
        mul_369 = torch.ops.aten.mul.Tensor(mul_367, mul_68);  mul_367 = None
        sum_120 = torch.ops.aten.sum.dim_IntList(mul_369, [2], True);  mul_369 = None
        mul_370 = torch.ops.aten.mul.Tensor(mul_68, sum_120);  sum_120 = None
        sub_89 = torch.ops.aten.sub.Tensor(mul_368, sum_119);  mul_368 = sum_119 = None
        sub_90 = torch.ops.aten.sub.Tensor(sub_89, mul_370);  sub_89 = mul_370 = None
        mul_371 = torch.ops.aten.mul.Tensor(div_45, sub_90);  div_45 = sub_90 = None
        mul_372 = torch.ops.aten.mul.Tensor(add_142, mul_68);  mul_68 = None
        sum_121 = torch.ops.aten.sum.dim_IntList(mul_372, [0, 1]);  mul_372 = None
        sum_122 = torch.ops.aten.sum.dim_IntList(add_142, [0, 1]);  add_142 = None
        convert_element_type_22 = torch.ops.prims.convert_element_type.default(gt_15, torch.float32);  gt_15 = None
        mul_373 = torch.ops.aten.mul.Tensor(convert_element_type_22, 1.1111111111111112);  convert_element_type_22 = None
        mul_374 = torch.ops.aten.mul.Tensor(mul_371, mul_373);  mul_373 = None
        clone_97 = torch.ops.aten.clone.default(mul_374, memory_format = torch.contiguous_format);  mul_374 = None
        view_461 = torch.ops.aten.view.default(clone_97, [15, 768]);  clone_97 = None
        mm_86 = torch.ops.aten.mm.default(view_461, permute_368);  permute_368 = None
        permute_369 = torch.ops.aten.permute.default(view_461, [1, 0])
        mm_87 = torch.ops.aten.mm.default(permute_369, view_108);  permute_369 = view_108 = None
        permute_370 = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
        sum_123 = torch.ops.aten.sum.dim_IntList(view_461, [0], True);  view_461 = None
        view_462 = torch.ops.aten.view.default(sum_123, [768]);  sum_123 = None
        permute_371 = torch.ops.aten.permute.default(permute_370, [1, 0]);  permute_370 = None
        view_463 = torch.ops.aten.view.default(mm_86, [3, 5, 3072]);  mm_86 = None
        mul_376 = torch.ops.aten.mul.Tensor(add_40, 0.5);  add_40 = None
        mul_377 = torch.ops.aten.mul.Tensor(view_107, view_107)
        mul_378 = torch.ops.aten.mul.Tensor(mul_377, -0.5);  mul_377 = None
        exp_19 = torch.ops.aten.exp.default(mul_378);  mul_378 = None
        mul_379 = torch.ops.aten.mul.Tensor(exp_19, 0.3989422804014327);  exp_19 = None
        mul_380 = torch.ops.aten.mul.Tensor(view_107, mul_379);  view_107 = mul_379 = None
        add_144 = torch.ops.aten.add.Tensor(mul_376, mul_380);  mul_376 = mul_380 = None
        mul_381 = torch.ops.aten.mul.Tensor(view_463, add_144);  view_463 = add_144 = None
        view_464 = torch.ops.aten.view.default(mul_381, [15, 3072]);  mul_381 = None
        mm_88 = torch.ops.aten.mm.default(view_464, permute_372);  permute_372 = None
        permute_373 = torch.ops.aten.permute.default(view_464, [1, 0])
        mm_89 = torch.ops.aten.mm.default(permute_373, view_106);  permute_373 = view_106 = None
        permute_374 = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
        sum_124 = torch.ops.aten.sum.dim_IntList(view_464, [0], True);  view_464 = None
        view_465 = torch.ops.aten.view.default(sum_124, [3072]);  sum_124 = None
        permute_375 = torch.ops.aten.permute.default(permute_374, [1, 0]);  permute_374 = None
        view_466 = torch.ops.aten.view.default(mm_88, [3, 5, 768]);  mm_88 = None
        add_145 = torch.ops.aten.add.Tensor(mul_371, view_466);  mul_371 = view_466 = None
        mul_383 = torch.ops.aten.mul.Tensor(add_145, primals_78);  primals_78 = None
        mul_384 = torch.ops.aten.mul.Tensor(mul_383, 768)
        sum_125 = torch.ops.aten.sum.dim_IntList(mul_383, [2], True)
        mul_385 = torch.ops.aten.mul.Tensor(mul_383, mul_61);  mul_383 = None
        sum_126 = torch.ops.aten.sum.dim_IntList(mul_385, [2], True);  mul_385 = None
        mul_386 = torch.ops.aten.mul.Tensor(mul_61, sum_126);  sum_126 = None
        sub_92 = torch.ops.aten.sub.Tensor(mul_384, sum_125);  mul_384 = sum_125 = None
        sub_93 = torch.ops.aten.sub.Tensor(sub_92, mul_386);  sub_92 = mul_386 = None
        mul_387 = torch.ops.aten.mul.Tensor(div_46, sub_93);  div_46 = sub_93 = None
        mul_388 = torch.ops.aten.mul.Tensor(add_145, mul_61);  mul_61 = None
        sum_127 = torch.ops.aten.sum.dim_IntList(mul_388, [0, 1]);  mul_388 = None
        sum_128 = torch.ops.aten.sum.dim_IntList(add_145, [0, 1]);  add_145 = None
        convert_element_type_23 = torch.ops.prims.convert_element_type.default(gt_14, torch.float32);  gt_14 = None
        mul_389 = torch.ops.aten.mul.Tensor(convert_element_type_23, 1.1111111111111112);  convert_element_type_23 = None
        mul_390 = torch.ops.aten.mul.Tensor(mul_387, mul_389);  mul_389 = None
        clone_98 = torch.ops.aten.clone.default(mul_390, memory_format = torch.contiguous_format);  mul_390 = None
        view_467 = torch.ops.aten.view.default(clone_98, [15, 768]);  clone_98 = None
        mm_90 = torch.ops.aten.mm.default(view_467, permute_376);  permute_376 = None
        permute_377 = torch.ops.aten.permute.default(view_467, [1, 0])
        mm_91 = torch.ops.aten.mm.default(permute_377, view_104);  permute_377 = view_104 = None
        permute_378 = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
        sum_129 = torch.ops.aten.sum.dim_IntList(view_467, [0], True);  view_467 = None
        view_468 = torch.ops.aten.view.default(sum_129, [768]);  sum_129 = None
        permute_379 = torch.ops.aten.permute.default(permute_378, [1, 0]);  permute_378 = None
        view_469 = torch.ops.aten.view.default(mm_90, [3, 5, 768]);  mm_90 = None
        view_470 = torch.ops.aten.view.default(view_469, [3, 5, 12, 64]);  view_469 = None
        permute_380 = torch.ops.aten.permute.default(view_470, [0, 2, 1, 3]);  view_470 = None
        clone_99 = torch.ops.aten.clone.default(permute_380, memory_format = torch.contiguous_format);  permute_380 = None
        view_471 = torch.ops.aten.view.default(clone_99, [36, 5, 64]);  clone_99 = None
        bmm_52 = torch.ops.aten.bmm.default(permute_381, view_471);  permute_381 = None
        bmm_53 = torch.ops.aten.bmm.default(view_471, permute_382);  view_471 = permute_382 = None
        view_472 = torch.ops.aten.view.default(bmm_52, [3, 12, 5, 64]);  bmm_52 = None
        view_473 = torch.ops.aten.view.default(bmm_53, [3, 12, 5, 5]);  bmm_53 = None
        convert_element_type_24 = torch.ops.prims.convert_element_type.default(gt_13, torch.float32);  gt_13 = None
        mul_391 = torch.ops.aten.mul.Tensor(convert_element_type_24, 1.1111111111111112);  convert_element_type_24 = None
        mul_392 = torch.ops.aten.mul.Tensor(view_473, mul_391);  view_473 = mul_391 = None
        clone_100 = torch.ops.aten.clone.default(mul_392, memory_format = torch.contiguous_format);  mul_392 = None
        mul_393 = torch.ops.aten.mul.Tensor(clone_100, alias_21);  clone_100 = None
        sum_130 = torch.ops.aten.sum.dim_IntList(mul_393, [-1], True)
        mul_394 = torch.ops.aten.mul.Tensor(alias_21, sum_130);  alias_21 = sum_130 = None
        sub_94 = torch.ops.aten.sub.Tensor(mul_393, mul_394);  mul_393 = mul_394 = None
        div_47 = torch.ops.aten.div.Tensor(sub_94, 8.0);  sub_94 = None
        view_474 = torch.ops.aten.view.default(div_47, [36, 5, 5]);  div_47 = None
        bmm_54 = torch.ops.aten.bmm.default(permute_383, view_474);  permute_383 = None
        bmm_55 = torch.ops.aten.bmm.default(view_474, permute_384);  view_474 = permute_384 = None
        view_475 = torch.ops.aten.view.default(bmm_54, [3, 12, 64, 5]);  bmm_54 = None
        view_476 = torch.ops.aten.view.default(bmm_55, [3, 12, 5, 64]);  bmm_55 = None
        permute_385 = torch.ops.aten.permute.default(view_475, [0, 1, 3, 2]);  view_475 = None
        permute_386 = torch.ops.aten.permute.default(view_476, [0, 2, 1, 3]);  view_476 = None
        clone_101 = torch.ops.aten.clone.default(permute_386, memory_format = torch.contiguous_format);  permute_386 = None
        view_477 = torch.ops.aten.view.default(clone_101, [3, 5, 768]);  clone_101 = None
        permute_387 = torch.ops.aten.permute.default(view_472, [0, 2, 1, 3]);  view_472 = None
        clone_102 = torch.ops.aten.clone.default(permute_387, memory_format = torch.contiguous_format);  permute_387 = None
        view_478 = torch.ops.aten.view.default(clone_102, [3, 5, 768]);  clone_102 = None
        view_479 = torch.ops.aten.view.default(view_478, [15, 768]);  view_478 = None
        mm_92 = torch.ops.aten.mm.default(view_479, permute_388);  permute_388 = None
        permute_389 = torch.ops.aten.permute.default(view_479, [1, 0])
        mm_93 = torch.ops.aten.mm.default(permute_389, view_88);  permute_389 = None
        permute_390 = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
        sum_131 = torch.ops.aten.sum.dim_IntList(view_479, [0], True);  view_479 = None
        view_480 = torch.ops.aten.view.default(sum_131, [768]);  sum_131 = None
        permute_391 = torch.ops.aten.permute.default(permute_390, [1, 0]);  permute_390 = None
        view_481 = torch.ops.aten.view.default(mm_92, [3, 5, 768]);  mm_92 = None
        add_146 = torch.ops.aten.add.Tensor(mul_387, view_481);  mul_387 = view_481 = None
        permute_392 = torch.ops.aten.permute.default(permute_385, [0, 2, 1, 3]);  permute_385 = None
        view_482 = torch.ops.aten.view.default(permute_392, [3, 5, 768]);  permute_392 = None
        clone_103 = torch.ops.aten.clone.default(view_482, memory_format = torch.contiguous_format);  view_482 = None
        view_483 = torch.ops.aten.view.default(clone_103, [15, 768]);  clone_103 = None
        mm_94 = torch.ops.aten.mm.default(view_483, permute_393);  permute_393 = None
        permute_394 = torch.ops.aten.permute.default(view_483, [1, 0])
        mm_95 = torch.ops.aten.mm.default(permute_394, view_88);  permute_394 = None
        permute_395 = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
        sum_132 = torch.ops.aten.sum.dim_IntList(view_483, [0], True);  view_483 = None
        view_484 = torch.ops.aten.view.default(sum_132, [768]);  sum_132 = None
        permute_396 = torch.ops.aten.permute.default(permute_395, [1, 0]);  permute_395 = None
        view_485 = torch.ops.aten.view.default(mm_94, [3, 5, 768]);  mm_94 = None
        add_147 = torch.ops.aten.add.Tensor(add_146, view_485);  add_146 = view_485 = None
        view_486 = torch.ops.aten.view.default(view_477, [15, 768]);  view_477 = None
        mm_96 = torch.ops.aten.mm.default(view_486, permute_397);  permute_397 = None
        permute_398 = torch.ops.aten.permute.default(view_486, [1, 0])
        mm_97 = torch.ops.aten.mm.default(permute_398, view_88);  permute_398 = view_88 = None
        permute_399 = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
        sum_133 = torch.ops.aten.sum.dim_IntList(view_486, [0], True);  view_486 = None
        view_487 = torch.ops.aten.view.default(sum_133, [768]);  sum_133 = None
        permute_400 = torch.ops.aten.permute.default(permute_399, [1, 0]);  permute_399 = None
        view_488 = torch.ops.aten.view.default(mm_96, [3, 5, 768]);  mm_96 = None
        add_148 = torch.ops.aten.add.Tensor(add_147, view_488);  add_147 = view_488 = None
        mul_396 = torch.ops.aten.mul.Tensor(add_148, primals_68);  primals_68 = None
        mul_397 = torch.ops.aten.mul.Tensor(mul_396, 768)
        sum_134 = torch.ops.aten.sum.dim_IntList(mul_396, [2], True)
        mul_398 = torch.ops.aten.mul.Tensor(mul_396, mul_55);  mul_396 = None
        sum_135 = torch.ops.aten.sum.dim_IntList(mul_398, [2], True);  mul_398 = None
        mul_399 = torch.ops.aten.mul.Tensor(mul_55, sum_135);  sum_135 = None
        sub_96 = torch.ops.aten.sub.Tensor(mul_397, sum_134);  mul_397 = sum_134 = None
        sub_97 = torch.ops.aten.sub.Tensor(sub_96, mul_399);  sub_96 = mul_399 = None
        mul_400 = torch.ops.aten.mul.Tensor(div_48, sub_97);  div_48 = sub_97 = None
        mul_401 = torch.ops.aten.mul.Tensor(add_148, mul_55);  mul_55 = None
        sum_136 = torch.ops.aten.sum.dim_IntList(mul_401, [0, 1]);  mul_401 = None
        sum_137 = torch.ops.aten.sum.dim_IntList(add_148, [0, 1]);  add_148 = None
        convert_element_type_25 = torch.ops.prims.convert_element_type.default(gt_12, torch.float32);  gt_12 = None
        mul_402 = torch.ops.aten.mul.Tensor(convert_element_type_25, 1.1111111111111112);  convert_element_type_25 = None
        mul_403 = torch.ops.aten.mul.Tensor(mul_400, mul_402);  mul_402 = None
        clone_104 = torch.ops.aten.clone.default(mul_403, memory_format = torch.contiguous_format);  mul_403 = None
        view_489 = torch.ops.aten.view.default(clone_104, [15, 768]);  clone_104 = None
        mm_98 = torch.ops.aten.mm.default(view_489, permute_401);  permute_401 = None
        permute_402 = torch.ops.aten.permute.default(view_489, [1, 0])
        mm_99 = torch.ops.aten.mm.default(permute_402, view_86);  permute_402 = view_86 = None
        permute_403 = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
        sum_138 = torch.ops.aten.sum.dim_IntList(view_489, [0], True);  view_489 = None
        view_490 = torch.ops.aten.view.default(sum_138, [768]);  sum_138 = None
        permute_404 = torch.ops.aten.permute.default(permute_403, [1, 0]);  permute_403 = None
        view_491 = torch.ops.aten.view.default(mm_98, [3, 5, 3072]);  mm_98 = None
        mul_405 = torch.ops.aten.mul.Tensor(add_32, 0.5);  add_32 = None
        mul_406 = torch.ops.aten.mul.Tensor(view_85, view_85)
        mul_407 = torch.ops.aten.mul.Tensor(mul_406, -0.5);  mul_406 = None
        exp_20 = torch.ops.aten.exp.default(mul_407);  mul_407 = None
        mul_408 = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
        mul_409 = torch.ops.aten.mul.Tensor(view_85, mul_408);  view_85 = mul_408 = None
        add_150 = torch.ops.aten.add.Tensor(mul_405, mul_409);  mul_405 = mul_409 = None
        mul_410 = torch.ops.aten.mul.Tensor(view_491, add_150);  view_491 = add_150 = None
        view_492 = torch.ops.aten.view.default(mul_410, [15, 3072]);  mul_410 = None
        mm_100 = torch.ops.aten.mm.default(view_492, permute_405);  permute_405 = None
        permute_406 = torch.ops.aten.permute.default(view_492, [1, 0])
        mm_101 = torch.ops.aten.mm.default(permute_406, view_84);  permute_406 = view_84 = None
        permute_407 = torch.ops.aten.permute.default(mm_101, [1, 0]);  mm_101 = None
        sum_139 = torch.ops.aten.sum.dim_IntList(view_492, [0], True);  view_492 = None
        view_493 = torch.ops.aten.view.default(sum_139, [3072]);  sum_139 = None
        permute_408 = torch.ops.aten.permute.default(permute_407, [1, 0]);  permute_407 = None
        view_494 = torch.ops.aten.view.default(mm_100, [3, 5, 768]);  mm_100 = None
        add_151 = torch.ops.aten.add.Tensor(mul_400, view_494);  mul_400 = view_494 = None
        mul_412 = torch.ops.aten.mul.Tensor(add_151, primals_62);  primals_62 = None
        mul_413 = torch.ops.aten.mul.Tensor(mul_412, 768)
        sum_140 = torch.ops.aten.sum.dim_IntList(mul_412, [2], True)
        mul_414 = torch.ops.aten.mul.Tensor(mul_412, mul_48);  mul_412 = None
        sum_141 = torch.ops.aten.sum.dim_IntList(mul_414, [2], True);  mul_414 = None
        mul_415 = torch.ops.aten.mul.Tensor(mul_48, sum_141);  sum_141 = None
        sub_99 = torch.ops.aten.sub.Tensor(mul_413, sum_140);  mul_413 = sum_140 = None
        sub_100 = torch.ops.aten.sub.Tensor(sub_99, mul_415);  sub_99 = mul_415 = None
        mul_416 = torch.ops.aten.mul.Tensor(div_49, sub_100);  div_49 = sub_100 = None
        mul_417 = torch.ops.aten.mul.Tensor(add_151, mul_48);  mul_48 = None
        sum_142 = torch.ops.aten.sum.dim_IntList(mul_417, [0, 1]);  mul_417 = None
        sum_143 = torch.ops.aten.sum.dim_IntList(add_151, [0, 1]);  add_151 = None
        convert_element_type_26 = torch.ops.prims.convert_element_type.default(gt_11, torch.float32);  gt_11 = None
        mul_418 = torch.ops.aten.mul.Tensor(convert_element_type_26, 1.1111111111111112);  convert_element_type_26 = None
        mul_419 = torch.ops.aten.mul.Tensor(mul_416, mul_418);  mul_418 = None
        clone_105 = torch.ops.aten.clone.default(mul_419, memory_format = torch.contiguous_format);  mul_419 = None
        view_495 = torch.ops.aten.view.default(clone_105, [15, 768]);  clone_105 = None
        mm_102 = torch.ops.aten.mm.default(view_495, permute_409);  permute_409 = None
        permute_410 = torch.ops.aten.permute.default(view_495, [1, 0])
        mm_103 = torch.ops.aten.mm.default(permute_410, view_82);  permute_410 = view_82 = None
        permute_411 = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
        sum_144 = torch.ops.aten.sum.dim_IntList(view_495, [0], True);  view_495 = None
        view_496 = torch.ops.aten.view.default(sum_144, [768]);  sum_144 = None
        permute_412 = torch.ops.aten.permute.default(permute_411, [1, 0]);  permute_411 = None
        view_497 = torch.ops.aten.view.default(mm_102, [3, 5, 768]);  mm_102 = None
        view_498 = torch.ops.aten.view.default(view_497, [3, 5, 12, 64]);  view_497 = None
        permute_413 = torch.ops.aten.permute.default(view_498, [0, 2, 1, 3]);  view_498 = None
        clone_106 = torch.ops.aten.clone.default(permute_413, memory_format = torch.contiguous_format);  permute_413 = None
        view_499 = torch.ops.aten.view.default(clone_106, [36, 5, 64]);  clone_106 = None
        bmm_56 = torch.ops.aten.bmm.default(permute_414, view_499);  permute_414 = None
        bmm_57 = torch.ops.aten.bmm.default(view_499, permute_415);  view_499 = permute_415 = None
        view_500 = torch.ops.aten.view.default(bmm_56, [3, 12, 5, 64]);  bmm_56 = None
        view_501 = torch.ops.aten.view.default(bmm_57, [3, 12, 5, 5]);  bmm_57 = None
        convert_element_type_27 = torch.ops.prims.convert_element_type.default(gt_10, torch.float32);  gt_10 = None
        mul_420 = torch.ops.aten.mul.Tensor(convert_element_type_27, 1.1111111111111112);  convert_element_type_27 = None
        mul_421 = torch.ops.aten.mul.Tensor(view_501, mul_420);  view_501 = mul_420 = None
        clone_107 = torch.ops.aten.clone.default(mul_421, memory_format = torch.contiguous_format);  mul_421 = None
        mul_422 = torch.ops.aten.mul.Tensor(clone_107, alias_22);  clone_107 = None
        sum_145 = torch.ops.aten.sum.dim_IntList(mul_422, [-1], True)
        mul_423 = torch.ops.aten.mul.Tensor(alias_22, sum_145);  alias_22 = sum_145 = None
        sub_101 = torch.ops.aten.sub.Tensor(mul_422, mul_423);  mul_422 = mul_423 = None
        div_50 = torch.ops.aten.div.Tensor(sub_101, 8.0);  sub_101 = None
        view_502 = torch.ops.aten.view.default(div_50, [36, 5, 5]);  div_50 = None
        bmm_58 = torch.ops.aten.bmm.default(permute_416, view_502);  permute_416 = None
        bmm_59 = torch.ops.aten.bmm.default(view_502, permute_417);  view_502 = permute_417 = None
        view_503 = torch.ops.aten.view.default(bmm_58, [3, 12, 64, 5]);  bmm_58 = None
        view_504 = torch.ops.aten.view.default(bmm_59, [3, 12, 5, 64]);  bmm_59 = None
        permute_418 = torch.ops.aten.permute.default(view_503, [0, 1, 3, 2]);  view_503 = None
        permute_419 = torch.ops.aten.permute.default(view_504, [0, 2, 1, 3]);  view_504 = None
        clone_108 = torch.ops.aten.clone.default(permute_419, memory_format = torch.contiguous_format);  permute_419 = None
        view_505 = torch.ops.aten.view.default(clone_108, [3, 5, 768]);  clone_108 = None
        permute_420 = torch.ops.aten.permute.default(view_500, [0, 2, 1, 3]);  view_500 = None
        clone_109 = torch.ops.aten.clone.default(permute_420, memory_format = torch.contiguous_format);  permute_420 = None
        view_506 = torch.ops.aten.view.default(clone_109, [3, 5, 768]);  clone_109 = None
        view_507 = torch.ops.aten.view.default(view_506, [15, 768]);  view_506 = None
        mm_104 = torch.ops.aten.mm.default(view_507, permute_421);  permute_421 = None
        permute_422 = torch.ops.aten.permute.default(view_507, [1, 0])
        mm_105 = torch.ops.aten.mm.default(permute_422, view_66);  permute_422 = None
        permute_423 = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
        sum_146 = torch.ops.aten.sum.dim_IntList(view_507, [0], True);  view_507 = None
        view_508 = torch.ops.aten.view.default(sum_146, [768]);  sum_146 = None
        permute_424 = torch.ops.aten.permute.default(permute_423, [1, 0]);  permute_423 = None
        view_509 = torch.ops.aten.view.default(mm_104, [3, 5, 768]);  mm_104 = None
        add_152 = torch.ops.aten.add.Tensor(mul_416, view_509);  mul_416 = view_509 = None
        permute_425 = torch.ops.aten.permute.default(permute_418, [0, 2, 1, 3]);  permute_418 = None
        view_510 = torch.ops.aten.view.default(permute_425, [3, 5, 768]);  permute_425 = None
        clone_110 = torch.ops.aten.clone.default(view_510, memory_format = torch.contiguous_format);  view_510 = None
        view_511 = torch.ops.aten.view.default(clone_110, [15, 768]);  clone_110 = None
        mm_106 = torch.ops.aten.mm.default(view_511, permute_426);  permute_426 = None
        permute_427 = torch.ops.aten.permute.default(view_511, [1, 0])
        mm_107 = torch.ops.aten.mm.default(permute_427, view_66);  permute_427 = None
        permute_428 = torch.ops.aten.permute.default(mm_107, [1, 0]);  mm_107 = None
        sum_147 = torch.ops.aten.sum.dim_IntList(view_511, [0], True);  view_511 = None
        view_512 = torch.ops.aten.view.default(sum_147, [768]);  sum_147 = None
        permute_429 = torch.ops.aten.permute.default(permute_428, [1, 0]);  permute_428 = None
        view_513 = torch.ops.aten.view.default(mm_106, [3, 5, 768]);  mm_106 = None
        add_153 = torch.ops.aten.add.Tensor(add_152, view_513);  add_152 = view_513 = None
        view_514 = torch.ops.aten.view.default(view_505, [15, 768]);  view_505 = None
        mm_108 = torch.ops.aten.mm.default(view_514, permute_430);  permute_430 = None
        permute_431 = torch.ops.aten.permute.default(view_514, [1, 0])
        mm_109 = torch.ops.aten.mm.default(permute_431, view_66);  permute_431 = view_66 = None
        permute_432 = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
        sum_148 = torch.ops.aten.sum.dim_IntList(view_514, [0], True);  view_514 = None
        view_515 = torch.ops.aten.view.default(sum_148, [768]);  sum_148 = None
        permute_433 = torch.ops.aten.permute.default(permute_432, [1, 0]);  permute_432 = None
        view_516 = torch.ops.aten.view.default(mm_108, [3, 5, 768]);  mm_108 = None
        add_154 = torch.ops.aten.add.Tensor(add_153, view_516);  add_153 = view_516 = None
        mul_425 = torch.ops.aten.mul.Tensor(add_154, primals_52);  primals_52 = None
        mul_426 = torch.ops.aten.mul.Tensor(mul_425, 768)
        sum_149 = torch.ops.aten.sum.dim_IntList(mul_425, [2], True)
        mul_427 = torch.ops.aten.mul.Tensor(mul_425, mul_42);  mul_425 = None
        sum_150 = torch.ops.aten.sum.dim_IntList(mul_427, [2], True);  mul_427 = None
        mul_428 = torch.ops.aten.mul.Tensor(mul_42, sum_150);  sum_150 = None
        sub_103 = torch.ops.aten.sub.Tensor(mul_426, sum_149);  mul_426 = sum_149 = None
        sub_104 = torch.ops.aten.sub.Tensor(sub_103, mul_428);  sub_103 = mul_428 = None
        mul_429 = torch.ops.aten.mul.Tensor(div_51, sub_104);  div_51 = sub_104 = None
        mul_430 = torch.ops.aten.mul.Tensor(add_154, mul_42);  mul_42 = None
        sum_151 = torch.ops.aten.sum.dim_IntList(mul_430, [0, 1]);  mul_430 = None
        sum_152 = torch.ops.aten.sum.dim_IntList(add_154, [0, 1]);  add_154 = None
        convert_element_type_28 = torch.ops.prims.convert_element_type.default(gt_9, torch.float32);  gt_9 = None
        mul_431 = torch.ops.aten.mul.Tensor(convert_element_type_28, 1.1111111111111112);  convert_element_type_28 = None
        mul_432 = torch.ops.aten.mul.Tensor(mul_429, mul_431);  mul_431 = None
        clone_111 = torch.ops.aten.clone.default(mul_432, memory_format = torch.contiguous_format);  mul_432 = None
        view_517 = torch.ops.aten.view.default(clone_111, [15, 768]);  clone_111 = None
        mm_110 = torch.ops.aten.mm.default(view_517, permute_434);  permute_434 = None
        permute_435 = torch.ops.aten.permute.default(view_517, [1, 0])
        mm_111 = torch.ops.aten.mm.default(permute_435, view_64);  permute_435 = view_64 = None
        permute_436 = torch.ops.aten.permute.default(mm_111, [1, 0]);  mm_111 = None
        sum_153 = torch.ops.aten.sum.dim_IntList(view_517, [0], True);  view_517 = None
        view_518 = torch.ops.aten.view.default(sum_153, [768]);  sum_153 = None
        permute_437 = torch.ops.aten.permute.default(permute_436, [1, 0]);  permute_436 = None
        view_519 = torch.ops.aten.view.default(mm_110, [3, 5, 3072]);  mm_110 = None
        mul_434 = torch.ops.aten.mul.Tensor(add_24, 0.5);  add_24 = None
        mul_435 = torch.ops.aten.mul.Tensor(view_63, view_63)
        mul_436 = torch.ops.aten.mul.Tensor(mul_435, -0.5);  mul_435 = None
        exp_21 = torch.ops.aten.exp.default(mul_436);  mul_436 = None
        mul_437 = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
        mul_438 = torch.ops.aten.mul.Tensor(view_63, mul_437);  view_63 = mul_437 = None
        add_156 = torch.ops.aten.add.Tensor(mul_434, mul_438);  mul_434 = mul_438 = None
        mul_439 = torch.ops.aten.mul.Tensor(view_519, add_156);  view_519 = add_156 = None
        view_520 = torch.ops.aten.view.default(mul_439, [15, 3072]);  mul_439 = None
        mm_112 = torch.ops.aten.mm.default(view_520, permute_438);  permute_438 = None
        permute_439 = torch.ops.aten.permute.default(view_520, [1, 0])
        mm_113 = torch.ops.aten.mm.default(permute_439, view_62);  permute_439 = view_62 = None
        permute_440 = torch.ops.aten.permute.default(mm_113, [1, 0]);  mm_113 = None
        sum_154 = torch.ops.aten.sum.dim_IntList(view_520, [0], True);  view_520 = None
        view_521 = torch.ops.aten.view.default(sum_154, [3072]);  sum_154 = None
        permute_441 = torch.ops.aten.permute.default(permute_440, [1, 0]);  permute_440 = None
        view_522 = torch.ops.aten.view.default(mm_112, [3, 5, 768]);  mm_112 = None
        add_157 = torch.ops.aten.add.Tensor(mul_429, view_522);  mul_429 = view_522 = None
        mul_441 = torch.ops.aten.mul.Tensor(add_157, primals_46);  primals_46 = None
        mul_442 = torch.ops.aten.mul.Tensor(mul_441, 768)
        sum_155 = torch.ops.aten.sum.dim_IntList(mul_441, [2], True)
        mul_443 = torch.ops.aten.mul.Tensor(mul_441, mul_35);  mul_441 = None
        sum_156 = torch.ops.aten.sum.dim_IntList(mul_443, [2], True);  mul_443 = None
        mul_444 = torch.ops.aten.mul.Tensor(mul_35, sum_156);  sum_156 = None
        sub_106 = torch.ops.aten.sub.Tensor(mul_442, sum_155);  mul_442 = sum_155 = None
        sub_107 = torch.ops.aten.sub.Tensor(sub_106, mul_444);  sub_106 = mul_444 = None
        mul_445 = torch.ops.aten.mul.Tensor(div_52, sub_107);  div_52 = sub_107 = None
        mul_446 = torch.ops.aten.mul.Tensor(add_157, mul_35);  mul_35 = None
        sum_157 = torch.ops.aten.sum.dim_IntList(mul_446, [0, 1]);  mul_446 = None
        sum_158 = torch.ops.aten.sum.dim_IntList(add_157, [0, 1]);  add_157 = None
        convert_element_type_29 = torch.ops.prims.convert_element_type.default(gt_8, torch.float32);  gt_8 = None
        mul_447 = torch.ops.aten.mul.Tensor(convert_element_type_29, 1.1111111111111112);  convert_element_type_29 = None
        mul_448 = torch.ops.aten.mul.Tensor(mul_445, mul_447);  mul_447 = None
        clone_112 = torch.ops.aten.clone.default(mul_448, memory_format = torch.contiguous_format);  mul_448 = None
        view_523 = torch.ops.aten.view.default(clone_112, [15, 768]);  clone_112 = None
        mm_114 = torch.ops.aten.mm.default(view_523, permute_442);  permute_442 = None
        permute_443 = torch.ops.aten.permute.default(view_523, [1, 0])
        mm_115 = torch.ops.aten.mm.default(permute_443, view_60);  permute_443 = view_60 = None
        permute_444 = torch.ops.aten.permute.default(mm_115, [1, 0]);  mm_115 = None
        sum_159 = torch.ops.aten.sum.dim_IntList(view_523, [0], True);  view_523 = None
        view_524 = torch.ops.aten.view.default(sum_159, [768]);  sum_159 = None
        permute_445 = torch.ops.aten.permute.default(permute_444, [1, 0]);  permute_444 = None
        view_525 = torch.ops.aten.view.default(mm_114, [3, 5, 768]);  mm_114 = None
        view_526 = torch.ops.aten.view.default(view_525, [3, 5, 12, 64]);  view_525 = None
        permute_446 = torch.ops.aten.permute.default(view_526, [0, 2, 1, 3]);  view_526 = None
        clone_113 = torch.ops.aten.clone.default(permute_446, memory_format = torch.contiguous_format);  permute_446 = None
        view_527 = torch.ops.aten.view.default(clone_113, [36, 5, 64]);  clone_113 = None
        bmm_60 = torch.ops.aten.bmm.default(permute_447, view_527);  permute_447 = None
        bmm_61 = torch.ops.aten.bmm.default(view_527, permute_448);  view_527 = permute_448 = None
        view_528 = torch.ops.aten.view.default(bmm_60, [3, 12, 5, 64]);  bmm_60 = None
        view_529 = torch.ops.aten.view.default(bmm_61, [3, 12, 5, 5]);  bmm_61 = None
        convert_element_type_30 = torch.ops.prims.convert_element_type.default(gt_7, torch.float32);  gt_7 = None
        mul_449 = torch.ops.aten.mul.Tensor(convert_element_type_30, 1.1111111111111112);  convert_element_type_30 = None
        mul_450 = torch.ops.aten.mul.Tensor(view_529, mul_449);  view_529 = mul_449 = None
        clone_114 = torch.ops.aten.clone.default(mul_450, memory_format = torch.contiguous_format);  mul_450 = None
        mul_451 = torch.ops.aten.mul.Tensor(clone_114, alias_23);  clone_114 = None
        sum_160 = torch.ops.aten.sum.dim_IntList(mul_451, [-1], True)
        mul_452 = torch.ops.aten.mul.Tensor(alias_23, sum_160);  alias_23 = sum_160 = None
        sub_108 = torch.ops.aten.sub.Tensor(mul_451, mul_452);  mul_451 = mul_452 = None
        div_53 = torch.ops.aten.div.Tensor(sub_108, 8.0);  sub_108 = None
        view_530 = torch.ops.aten.view.default(div_53, [36, 5, 5]);  div_53 = None
        bmm_62 = torch.ops.aten.bmm.default(permute_449, view_530);  permute_449 = None
        bmm_63 = torch.ops.aten.bmm.default(view_530, permute_450);  view_530 = permute_450 = None
        view_531 = torch.ops.aten.view.default(bmm_62, [3, 12, 64, 5]);  bmm_62 = None
        view_532 = torch.ops.aten.view.default(bmm_63, [3, 12, 5, 64]);  bmm_63 = None
        permute_451 = torch.ops.aten.permute.default(view_531, [0, 1, 3, 2]);  view_531 = None
        permute_452 = torch.ops.aten.permute.default(view_532, [0, 2, 1, 3]);  view_532 = None
        clone_115 = torch.ops.aten.clone.default(permute_452, memory_format = torch.contiguous_format);  permute_452 = None
        view_533 = torch.ops.aten.view.default(clone_115, [3, 5, 768]);  clone_115 = None
        permute_453 = torch.ops.aten.permute.default(view_528, [0, 2, 1, 3]);  view_528 = None
        clone_116 = torch.ops.aten.clone.default(permute_453, memory_format = torch.contiguous_format);  permute_453 = None
        view_534 = torch.ops.aten.view.default(clone_116, [3, 5, 768]);  clone_116 = None
        view_535 = torch.ops.aten.view.default(view_534, [15, 768]);  view_534 = None
        mm_116 = torch.ops.aten.mm.default(view_535, permute_454);  permute_454 = None
        permute_455 = torch.ops.aten.permute.default(view_535, [1, 0])
        mm_117 = torch.ops.aten.mm.default(permute_455, view_44);  permute_455 = None
        permute_456 = torch.ops.aten.permute.default(mm_117, [1, 0]);  mm_117 = None
        sum_161 = torch.ops.aten.sum.dim_IntList(view_535, [0], True);  view_535 = None
        view_536 = torch.ops.aten.view.default(sum_161, [768]);  sum_161 = None
        permute_457 = torch.ops.aten.permute.default(permute_456, [1, 0]);  permute_456 = None
        view_537 = torch.ops.aten.view.default(mm_116, [3, 5, 768]);  mm_116 = None
        add_158 = torch.ops.aten.add.Tensor(mul_445, view_537);  mul_445 = view_537 = None
        permute_458 = torch.ops.aten.permute.default(permute_451, [0, 2, 1, 3]);  permute_451 = None
        view_538 = torch.ops.aten.view.default(permute_458, [3, 5, 768]);  permute_458 = None
        clone_117 = torch.ops.aten.clone.default(view_538, memory_format = torch.contiguous_format);  view_538 = None
        view_539 = torch.ops.aten.view.default(clone_117, [15, 768]);  clone_117 = None
        mm_118 = torch.ops.aten.mm.default(view_539, permute_459);  permute_459 = None
        permute_460 = torch.ops.aten.permute.default(view_539, [1, 0])
        mm_119 = torch.ops.aten.mm.default(permute_460, view_44);  permute_460 = None
        permute_461 = torch.ops.aten.permute.default(mm_119, [1, 0]);  mm_119 = None
        sum_162 = torch.ops.aten.sum.dim_IntList(view_539, [0], True);  view_539 = None
        view_540 = torch.ops.aten.view.default(sum_162, [768]);  sum_162 = None
        permute_462 = torch.ops.aten.permute.default(permute_461, [1, 0]);  permute_461 = None
        view_541 = torch.ops.aten.view.default(mm_118, [3, 5, 768]);  mm_118 = None
        add_159 = torch.ops.aten.add.Tensor(add_158, view_541);  add_158 = view_541 = None
        view_542 = torch.ops.aten.view.default(view_533, [15, 768]);  view_533 = None
        mm_120 = torch.ops.aten.mm.default(view_542, permute_463);  permute_463 = None
        permute_464 = torch.ops.aten.permute.default(view_542, [1, 0])
        mm_121 = torch.ops.aten.mm.default(permute_464, view_44);  permute_464 = view_44 = None
        permute_465 = torch.ops.aten.permute.default(mm_121, [1, 0]);  mm_121 = None
        sum_163 = torch.ops.aten.sum.dim_IntList(view_542, [0], True);  view_542 = None
        view_543 = torch.ops.aten.view.default(sum_163, [768]);  sum_163 = None
        permute_466 = torch.ops.aten.permute.default(permute_465, [1, 0]);  permute_465 = None
        view_544 = torch.ops.aten.view.default(mm_120, [3, 5, 768]);  mm_120 = None
        add_160 = torch.ops.aten.add.Tensor(add_159, view_544);  add_159 = view_544 = None
        mul_454 = torch.ops.aten.mul.Tensor(add_160, primals_36);  primals_36 = None
        mul_455 = torch.ops.aten.mul.Tensor(mul_454, 768)
        sum_164 = torch.ops.aten.sum.dim_IntList(mul_454, [2], True)
        mul_456 = torch.ops.aten.mul.Tensor(mul_454, mul_29);  mul_454 = None
        sum_165 = torch.ops.aten.sum.dim_IntList(mul_456, [2], True);  mul_456 = None
        mul_457 = torch.ops.aten.mul.Tensor(mul_29, sum_165);  sum_165 = None
        sub_110 = torch.ops.aten.sub.Tensor(mul_455, sum_164);  mul_455 = sum_164 = None
        sub_111 = torch.ops.aten.sub.Tensor(sub_110, mul_457);  sub_110 = mul_457 = None
        mul_458 = torch.ops.aten.mul.Tensor(div_54, sub_111);  div_54 = sub_111 = None
        mul_459 = torch.ops.aten.mul.Tensor(add_160, mul_29);  mul_29 = None
        sum_166 = torch.ops.aten.sum.dim_IntList(mul_459, [0, 1]);  mul_459 = None
        sum_167 = torch.ops.aten.sum.dim_IntList(add_160, [0, 1]);  add_160 = None
        convert_element_type_31 = torch.ops.prims.convert_element_type.default(gt_6, torch.float32);  gt_6 = None
        mul_460 = torch.ops.aten.mul.Tensor(convert_element_type_31, 1.1111111111111112);  convert_element_type_31 = None
        mul_461 = torch.ops.aten.mul.Tensor(mul_458, mul_460);  mul_460 = None
        clone_118 = torch.ops.aten.clone.default(mul_461, memory_format = torch.contiguous_format);  mul_461 = None
        view_545 = torch.ops.aten.view.default(clone_118, [15, 768]);  clone_118 = None
        mm_122 = torch.ops.aten.mm.default(view_545, permute_467);  permute_467 = None
        permute_468 = torch.ops.aten.permute.default(view_545, [1, 0])
        mm_123 = torch.ops.aten.mm.default(permute_468, view_42);  permute_468 = view_42 = None
        permute_469 = torch.ops.aten.permute.default(mm_123, [1, 0]);  mm_123 = None
        sum_168 = torch.ops.aten.sum.dim_IntList(view_545, [0], True);  view_545 = None
        view_546 = torch.ops.aten.view.default(sum_168, [768]);  sum_168 = None
        permute_470 = torch.ops.aten.permute.default(permute_469, [1, 0]);  permute_469 = None
        view_547 = torch.ops.aten.view.default(mm_122, [3, 5, 3072]);  mm_122 = None
        mul_463 = torch.ops.aten.mul.Tensor(add_16, 0.5);  add_16 = None
        mul_464 = torch.ops.aten.mul.Tensor(view_41, view_41)
        mul_465 = torch.ops.aten.mul.Tensor(mul_464, -0.5);  mul_464 = None
        exp_22 = torch.ops.aten.exp.default(mul_465);  mul_465 = None
        mul_466 = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
        mul_467 = torch.ops.aten.mul.Tensor(view_41, mul_466);  view_41 = mul_466 = None
        add_162 = torch.ops.aten.add.Tensor(mul_463, mul_467);  mul_463 = mul_467 = None
        mul_468 = torch.ops.aten.mul.Tensor(view_547, add_162);  view_547 = add_162 = None
        view_548 = torch.ops.aten.view.default(mul_468, [15, 3072]);  mul_468 = None
        mm_124 = torch.ops.aten.mm.default(view_548, permute_471);  permute_471 = None
        permute_472 = torch.ops.aten.permute.default(view_548, [1, 0])
        mm_125 = torch.ops.aten.mm.default(permute_472, view_40);  permute_472 = view_40 = None
        permute_473 = torch.ops.aten.permute.default(mm_125, [1, 0]);  mm_125 = None
        sum_169 = torch.ops.aten.sum.dim_IntList(view_548, [0], True);  view_548 = None
        view_549 = torch.ops.aten.view.default(sum_169, [3072]);  sum_169 = None
        permute_474 = torch.ops.aten.permute.default(permute_473, [1, 0]);  permute_473 = None
        view_550 = torch.ops.aten.view.default(mm_124, [3, 5, 768]);  mm_124 = None
        add_163 = torch.ops.aten.add.Tensor(mul_458, view_550);  mul_458 = view_550 = None
        mul_470 = torch.ops.aten.mul.Tensor(add_163, primals_30);  primals_30 = None
        mul_471 = torch.ops.aten.mul.Tensor(mul_470, 768)
        sum_170 = torch.ops.aten.sum.dim_IntList(mul_470, [2], True)
        mul_472 = torch.ops.aten.mul.Tensor(mul_470, mul_22);  mul_470 = None
        sum_171 = torch.ops.aten.sum.dim_IntList(mul_472, [2], True);  mul_472 = None
        mul_473 = torch.ops.aten.mul.Tensor(mul_22, sum_171);  sum_171 = None
        sub_113 = torch.ops.aten.sub.Tensor(mul_471, sum_170);  mul_471 = sum_170 = None
        sub_114 = torch.ops.aten.sub.Tensor(sub_113, mul_473);  sub_113 = mul_473 = None
        mul_474 = torch.ops.aten.mul.Tensor(div_55, sub_114);  div_55 = sub_114 = None
        mul_475 = torch.ops.aten.mul.Tensor(add_163, mul_22);  mul_22 = None
        sum_172 = torch.ops.aten.sum.dim_IntList(mul_475, [0, 1]);  mul_475 = None
        sum_173 = torch.ops.aten.sum.dim_IntList(add_163, [0, 1]);  add_163 = None
        convert_element_type_32 = torch.ops.prims.convert_element_type.default(gt_5, torch.float32);  gt_5 = None
        mul_476 = torch.ops.aten.mul.Tensor(convert_element_type_32, 1.1111111111111112);  convert_element_type_32 = None
        mul_477 = torch.ops.aten.mul.Tensor(mul_474, mul_476);  mul_476 = None
        clone_119 = torch.ops.aten.clone.default(mul_477, memory_format = torch.contiguous_format);  mul_477 = None
        view_551 = torch.ops.aten.view.default(clone_119, [15, 768]);  clone_119 = None
        mm_126 = torch.ops.aten.mm.default(view_551, permute_475);  permute_475 = None
        permute_476 = torch.ops.aten.permute.default(view_551, [1, 0])
        mm_127 = torch.ops.aten.mm.default(permute_476, view_38);  permute_476 = view_38 = None
        permute_477 = torch.ops.aten.permute.default(mm_127, [1, 0]);  mm_127 = None
        sum_174 = torch.ops.aten.sum.dim_IntList(view_551, [0], True);  view_551 = None
        view_552 = torch.ops.aten.view.default(sum_174, [768]);  sum_174 = None
        permute_478 = torch.ops.aten.permute.default(permute_477, [1, 0]);  permute_477 = None
        view_553 = torch.ops.aten.view.default(mm_126, [3, 5, 768]);  mm_126 = None
        view_554 = torch.ops.aten.view.default(view_553, [3, 5, 12, 64]);  view_553 = None
        permute_479 = torch.ops.aten.permute.default(view_554, [0, 2, 1, 3]);  view_554 = None
        clone_120 = torch.ops.aten.clone.default(permute_479, memory_format = torch.contiguous_format);  permute_479 = None
        view_555 = torch.ops.aten.view.default(clone_120, [36, 5, 64]);  clone_120 = None
        bmm_64 = torch.ops.aten.bmm.default(permute_480, view_555);  permute_480 = None
        bmm_65 = torch.ops.aten.bmm.default(view_555, permute_481);  view_555 = permute_481 = None
        view_556 = torch.ops.aten.view.default(bmm_64, [3, 12, 5, 64]);  bmm_64 = None
        view_557 = torch.ops.aten.view.default(bmm_65, [3, 12, 5, 5]);  bmm_65 = None
        convert_element_type_33 = torch.ops.prims.convert_element_type.default(gt_4, torch.float32);  gt_4 = None
        mul_478 = torch.ops.aten.mul.Tensor(convert_element_type_33, 1.1111111111111112);  convert_element_type_33 = None
        mul_479 = torch.ops.aten.mul.Tensor(view_557, mul_478);  view_557 = mul_478 = None
        clone_121 = torch.ops.aten.clone.default(mul_479, memory_format = torch.contiguous_format);  mul_479 = None
        mul_480 = torch.ops.aten.mul.Tensor(clone_121, alias_24);  clone_121 = None
        sum_175 = torch.ops.aten.sum.dim_IntList(mul_480, [-1], True)
        mul_481 = torch.ops.aten.mul.Tensor(alias_24, sum_175);  alias_24 = sum_175 = None
        sub_115 = torch.ops.aten.sub.Tensor(mul_480, mul_481);  mul_480 = mul_481 = None
        div_56 = torch.ops.aten.div.Tensor(sub_115, 8.0);  sub_115 = None
        view_558 = torch.ops.aten.view.default(div_56, [36, 5, 5]);  div_56 = None
        bmm_66 = torch.ops.aten.bmm.default(permute_482, view_558);  permute_482 = None
        bmm_67 = torch.ops.aten.bmm.default(view_558, permute_483);  view_558 = permute_483 = None
        view_559 = torch.ops.aten.view.default(bmm_66, [3, 12, 64, 5]);  bmm_66 = None
        view_560 = torch.ops.aten.view.default(bmm_67, [3, 12, 5, 64]);  bmm_67 = None
        permute_484 = torch.ops.aten.permute.default(view_559, [0, 1, 3, 2]);  view_559 = None
        permute_485 = torch.ops.aten.permute.default(view_560, [0, 2, 1, 3]);  view_560 = None
        clone_122 = torch.ops.aten.clone.default(permute_485, memory_format = torch.contiguous_format);  permute_485 = None
        view_561 = torch.ops.aten.view.default(clone_122, [3, 5, 768]);  clone_122 = None
        permute_486 = torch.ops.aten.permute.default(view_556, [0, 2, 1, 3]);  view_556 = None
        clone_123 = torch.ops.aten.clone.default(permute_486, memory_format = torch.contiguous_format);  permute_486 = None
        view_562 = torch.ops.aten.view.default(clone_123, [3, 5, 768]);  clone_123 = None
        view_563 = torch.ops.aten.view.default(view_562, [15, 768]);  view_562 = None
        mm_128 = torch.ops.aten.mm.default(view_563, permute_487);  permute_487 = None
        permute_488 = torch.ops.aten.permute.default(view_563, [1, 0])
        mm_129 = torch.ops.aten.mm.default(permute_488, view_22);  permute_488 = None
        permute_489 = torch.ops.aten.permute.default(mm_129, [1, 0]);  mm_129 = None
        sum_176 = torch.ops.aten.sum.dim_IntList(view_563, [0], True);  view_563 = None
        view_564 = torch.ops.aten.view.default(sum_176, [768]);  sum_176 = None
        permute_490 = torch.ops.aten.permute.default(permute_489, [1, 0]);  permute_489 = None
        view_565 = torch.ops.aten.view.default(mm_128, [3, 5, 768]);  mm_128 = None
        add_164 = torch.ops.aten.add.Tensor(mul_474, view_565);  mul_474 = view_565 = None
        permute_491 = torch.ops.aten.permute.default(permute_484, [0, 2, 1, 3]);  permute_484 = None
        view_566 = torch.ops.aten.view.default(permute_491, [3, 5, 768]);  permute_491 = None
        clone_124 = torch.ops.aten.clone.default(view_566, memory_format = torch.contiguous_format);  view_566 = None
        view_567 = torch.ops.aten.view.default(clone_124, [15, 768]);  clone_124 = None
        mm_130 = torch.ops.aten.mm.default(view_567, permute_492);  permute_492 = None
        permute_493 = torch.ops.aten.permute.default(view_567, [1, 0])
        mm_131 = torch.ops.aten.mm.default(permute_493, view_22);  permute_493 = None
        permute_494 = torch.ops.aten.permute.default(mm_131, [1, 0]);  mm_131 = None
        sum_177 = torch.ops.aten.sum.dim_IntList(view_567, [0], True);  view_567 = None
        view_568 = torch.ops.aten.view.default(sum_177, [768]);  sum_177 = None
        permute_495 = torch.ops.aten.permute.default(permute_494, [1, 0]);  permute_494 = None
        view_569 = torch.ops.aten.view.default(mm_130, [3, 5, 768]);  mm_130 = None
        add_165 = torch.ops.aten.add.Tensor(add_164, view_569);  add_164 = view_569 = None
        view_570 = torch.ops.aten.view.default(view_561, [15, 768]);  view_561 = None
        mm_132 = torch.ops.aten.mm.default(view_570, permute_496);  permute_496 = None
        permute_497 = torch.ops.aten.permute.default(view_570, [1, 0])
        mm_133 = torch.ops.aten.mm.default(permute_497, view_22);  permute_497 = view_22 = None
        permute_498 = torch.ops.aten.permute.default(mm_133, [1, 0]);  mm_133 = None
        sum_178 = torch.ops.aten.sum.dim_IntList(view_570, [0], True);  view_570 = None
        view_571 = torch.ops.aten.view.default(sum_178, [768]);  sum_178 = None
        permute_499 = torch.ops.aten.permute.default(permute_498, [1, 0]);  permute_498 = None
        view_572 = torch.ops.aten.view.default(mm_132, [3, 5, 768]);  mm_132 = None
        add_166 = torch.ops.aten.add.Tensor(add_165, view_572);  add_165 = view_572 = None
        mul_483 = torch.ops.aten.mul.Tensor(add_166, primals_20);  primals_20 = None
        mul_484 = torch.ops.aten.mul.Tensor(mul_483, 768)
        sum_179 = torch.ops.aten.sum.dim_IntList(mul_483, [2], True)
        mul_485 = torch.ops.aten.mul.Tensor(mul_483, mul_16);  mul_483 = None
        sum_180 = torch.ops.aten.sum.dim_IntList(mul_485, [2], True);  mul_485 = None
        mul_486 = torch.ops.aten.mul.Tensor(mul_16, sum_180);  sum_180 = None
        sub_117 = torch.ops.aten.sub.Tensor(mul_484, sum_179);  mul_484 = sum_179 = None
        sub_118 = torch.ops.aten.sub.Tensor(sub_117, mul_486);  sub_117 = mul_486 = None
        mul_487 = torch.ops.aten.mul.Tensor(div_57, sub_118);  div_57 = sub_118 = None
        mul_488 = torch.ops.aten.mul.Tensor(add_166, mul_16);  mul_16 = None
        sum_181 = torch.ops.aten.sum.dim_IntList(mul_488, [0, 1]);  mul_488 = None
        sum_182 = torch.ops.aten.sum.dim_IntList(add_166, [0, 1]);  add_166 = None
        convert_element_type_34 = torch.ops.prims.convert_element_type.default(gt_3, torch.float32);  gt_3 = None
        mul_489 = torch.ops.aten.mul.Tensor(convert_element_type_34, 1.1111111111111112);  convert_element_type_34 = None
        mul_490 = torch.ops.aten.mul.Tensor(mul_487, mul_489);  mul_489 = None
        clone_125 = torch.ops.aten.clone.default(mul_490, memory_format = torch.contiguous_format);  mul_490 = None
        view_573 = torch.ops.aten.view.default(clone_125, [15, 768]);  clone_125 = None
        mm_134 = torch.ops.aten.mm.default(view_573, permute_500);  permute_500 = None
        permute_501 = torch.ops.aten.permute.default(view_573, [1, 0])
        mm_135 = torch.ops.aten.mm.default(permute_501, view_20);  permute_501 = view_20 = None
        permute_502 = torch.ops.aten.permute.default(mm_135, [1, 0]);  mm_135 = None
        sum_183 = torch.ops.aten.sum.dim_IntList(view_573, [0], True);  view_573 = None
        view_574 = torch.ops.aten.view.default(sum_183, [768]);  sum_183 = None
        permute_503 = torch.ops.aten.permute.default(permute_502, [1, 0]);  permute_502 = None
        view_575 = torch.ops.aten.view.default(mm_134, [3, 5, 3072]);  mm_134 = None
        mul_492 = torch.ops.aten.mul.Tensor(add_8, 0.5);  add_8 = None
        mul_493 = torch.ops.aten.mul.Tensor(view_19, view_19)
        mul_494 = torch.ops.aten.mul.Tensor(mul_493, -0.5);  mul_493 = None
        exp_23 = torch.ops.aten.exp.default(mul_494);  mul_494 = None
        mul_495 = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
        mul_496 = torch.ops.aten.mul.Tensor(view_19, mul_495);  view_19 = mul_495 = None
        add_168 = torch.ops.aten.add.Tensor(mul_492, mul_496);  mul_492 = mul_496 = None
        mul_497 = torch.ops.aten.mul.Tensor(view_575, add_168);  view_575 = add_168 = None
        view_576 = torch.ops.aten.view.default(mul_497, [15, 3072]);  mul_497 = None
        mm_136 = torch.ops.aten.mm.default(view_576, permute_504);  permute_504 = None
        permute_505 = torch.ops.aten.permute.default(view_576, [1, 0])
        mm_137 = torch.ops.aten.mm.default(permute_505, view_18);  permute_505 = view_18 = None
        permute_506 = torch.ops.aten.permute.default(mm_137, [1, 0]);  mm_137 = None
        sum_184 = torch.ops.aten.sum.dim_IntList(view_576, [0], True);  view_576 = None
        view_577 = torch.ops.aten.view.default(sum_184, [3072]);  sum_184 = None
        permute_507 = torch.ops.aten.permute.default(permute_506, [1, 0]);  permute_506 = None
        view_578 = torch.ops.aten.view.default(mm_136, [3, 5, 768]);  mm_136 = None
        add_169 = torch.ops.aten.add.Tensor(mul_487, view_578);  mul_487 = view_578 = None
        mul_499 = torch.ops.aten.mul.Tensor(add_169, primals_14);  primals_14 = None
        mul_500 = torch.ops.aten.mul.Tensor(mul_499, 768)
        sum_185 = torch.ops.aten.sum.dim_IntList(mul_499, [2], True)
        mul_501 = torch.ops.aten.mul.Tensor(mul_499, mul_9);  mul_499 = None
        sum_186 = torch.ops.aten.sum.dim_IntList(mul_501, [2], True);  mul_501 = None
        mul_502 = torch.ops.aten.mul.Tensor(mul_9, sum_186);  sum_186 = None
        sub_120 = torch.ops.aten.sub.Tensor(mul_500, sum_185);  mul_500 = sum_185 = None
        sub_121 = torch.ops.aten.sub.Tensor(sub_120, mul_502);  sub_120 = mul_502 = None
        mul_503 = torch.ops.aten.mul.Tensor(div_58, sub_121);  div_58 = sub_121 = None
        mul_504 = torch.ops.aten.mul.Tensor(add_169, mul_9);  mul_9 = None
        sum_187 = torch.ops.aten.sum.dim_IntList(mul_504, [0, 1]);  mul_504 = None
        sum_188 = torch.ops.aten.sum.dim_IntList(add_169, [0, 1]);  add_169 = None
        convert_element_type_35 = torch.ops.prims.convert_element_type.default(gt_2, torch.float32);  gt_2 = None
        mul_505 = torch.ops.aten.mul.Tensor(convert_element_type_35, 1.1111111111111112);  convert_element_type_35 = None
        mul_506 = torch.ops.aten.mul.Tensor(mul_503, mul_505);  mul_505 = None
        clone_126 = torch.ops.aten.clone.default(mul_506, memory_format = torch.contiguous_format);  mul_506 = None
        view_579 = torch.ops.aten.view.default(clone_126, [15, 768]);  clone_126 = None
        mm_138 = torch.ops.aten.mm.default(view_579, permute_508);  permute_508 = None
        permute_509 = torch.ops.aten.permute.default(view_579, [1, 0])
        mm_139 = torch.ops.aten.mm.default(permute_509, view_16);  permute_509 = view_16 = None
        permute_510 = torch.ops.aten.permute.default(mm_139, [1, 0]);  mm_139 = None
        sum_189 = torch.ops.aten.sum.dim_IntList(view_579, [0], True);  view_579 = None
        view_580 = torch.ops.aten.view.default(sum_189, [768]);  sum_189 = None
        permute_511 = torch.ops.aten.permute.default(permute_510, [1, 0]);  permute_510 = None
        view_581 = torch.ops.aten.view.default(mm_138, [3, 5, 768]);  mm_138 = None
        view_582 = torch.ops.aten.view.default(view_581, [3, 5, 12, 64]);  view_581 = None
        permute_512 = torch.ops.aten.permute.default(view_582, [0, 2, 1, 3]);  view_582 = None
        clone_127 = torch.ops.aten.clone.default(permute_512, memory_format = torch.contiguous_format);  permute_512 = None
        view_583 = torch.ops.aten.view.default(clone_127, [36, 5, 64]);  clone_127 = None
        bmm_68 = torch.ops.aten.bmm.default(permute_513, view_583);  permute_513 = None
        bmm_69 = torch.ops.aten.bmm.default(view_583, permute_514);  view_583 = permute_514 = None
        view_584 = torch.ops.aten.view.default(bmm_68, [3, 12, 5, 64]);  bmm_68 = None
        view_585 = torch.ops.aten.view.default(bmm_69, [3, 12, 5, 5]);  bmm_69 = None
        convert_element_type_36 = torch.ops.prims.convert_element_type.default(gt_1, torch.float32);  gt_1 = None
        mul_507 = torch.ops.aten.mul.Tensor(convert_element_type_36, 1.1111111111111112);  convert_element_type_36 = None
        mul_508 = torch.ops.aten.mul.Tensor(view_585, mul_507);  view_585 = mul_507 = None
        clone_128 = torch.ops.aten.clone.default(mul_508, memory_format = torch.contiguous_format);  mul_508 = None
        mul_509 = torch.ops.aten.mul.Tensor(clone_128, alias_25);  clone_128 = None
        sum_190 = torch.ops.aten.sum.dim_IntList(mul_509, [-1], True)
        mul_510 = torch.ops.aten.mul.Tensor(alias_25, sum_190);  alias_25 = sum_190 = None
        sub_122 = torch.ops.aten.sub.Tensor(mul_509, mul_510);  mul_509 = mul_510 = None
        div_59 = torch.ops.aten.div.Tensor(sub_122, 8.0);  sub_122 = None
        view_586 = torch.ops.aten.view.default(div_59, [36, 5, 5]);  div_59 = None
        bmm_70 = torch.ops.aten.bmm.default(permute_515, view_586);  permute_515 = None
        bmm_71 = torch.ops.aten.bmm.default(view_586, permute_516);  view_586 = permute_516 = None
        view_587 = torch.ops.aten.view.default(bmm_70, [3, 12, 64, 5]);  bmm_70 = None
        view_588 = torch.ops.aten.view.default(bmm_71, [3, 12, 5, 64]);  bmm_71 = None
        permute_517 = torch.ops.aten.permute.default(view_587, [0, 1, 3, 2]);  view_587 = None
        permute_518 = torch.ops.aten.permute.default(view_588, [0, 2, 1, 3]);  view_588 = None
        clone_129 = torch.ops.aten.clone.default(permute_518, memory_format = torch.contiguous_format);  permute_518 = None
        view_589 = torch.ops.aten.view.default(clone_129, [3, 5, 768]);  clone_129 = None
        permute_519 = torch.ops.aten.permute.default(view_584, [0, 2, 1, 3]);  view_584 = None
        clone_130 = torch.ops.aten.clone.default(permute_519, memory_format = torch.contiguous_format);  permute_519 = None
        view_590 = torch.ops.aten.view.default(clone_130, [3, 5, 768]);  clone_130 = None
        view_591 = torch.ops.aten.view.default(view_590, [15, 768]);  view_590 = None
        mm_140 = torch.ops.aten.mm.default(view_591, permute_520);  permute_520 = None
        permute_521 = torch.ops.aten.permute.default(view_591, [1, 0])
        mm_141 = torch.ops.aten.mm.default(permute_521, view);  permute_521 = None
        permute_522 = torch.ops.aten.permute.default(mm_141, [1, 0]);  mm_141 = None
        sum_191 = torch.ops.aten.sum.dim_IntList(view_591, [0], True);  view_591 = None
        view_592 = torch.ops.aten.view.default(sum_191, [768]);  sum_191 = None
        permute_523 = torch.ops.aten.permute.default(permute_522, [1, 0]);  permute_522 = None
        view_593 = torch.ops.aten.view.default(mm_140, [3, 5, 768]);  mm_140 = None
        add_170 = torch.ops.aten.add.Tensor(mul_503, view_593);  mul_503 = view_593 = None
        permute_524 = torch.ops.aten.permute.default(permute_517, [0, 2, 1, 3]);  permute_517 = None
        view_594 = torch.ops.aten.view.default(permute_524, [3, 5, 768]);  permute_524 = None
        clone_131 = torch.ops.aten.clone.default(view_594, memory_format = torch.contiguous_format);  view_594 = None
        view_595 = torch.ops.aten.view.default(clone_131, [15, 768]);  clone_131 = None
        mm_142 = torch.ops.aten.mm.default(view_595, permute_525);  permute_525 = None
        permute_526 = torch.ops.aten.permute.default(view_595, [1, 0])
        mm_143 = torch.ops.aten.mm.default(permute_526, view);  permute_526 = None
        permute_527 = torch.ops.aten.permute.default(mm_143, [1, 0]);  mm_143 = None
        sum_192 = torch.ops.aten.sum.dim_IntList(view_595, [0], True);  view_595 = None
        view_596 = torch.ops.aten.view.default(sum_192, [768]);  sum_192 = None
        permute_528 = torch.ops.aten.permute.default(permute_527, [1, 0]);  permute_527 = None
        view_597 = torch.ops.aten.view.default(mm_142, [3, 5, 768]);  mm_142 = None
        add_171 = torch.ops.aten.add.Tensor(add_170, view_597);  add_170 = view_597 = None
        view_598 = torch.ops.aten.view.default(view_589, [15, 768]);  view_589 = None
        mm_144 = torch.ops.aten.mm.default(view_598, permute_529);  permute_529 = None
        permute_530 = torch.ops.aten.permute.default(view_598, [1, 0])
        mm_145 = torch.ops.aten.mm.default(permute_530, view);  permute_530 = view = None
        permute_531 = torch.ops.aten.permute.default(mm_145, [1, 0]);  mm_145 = None
        sum_193 = torch.ops.aten.sum.dim_IntList(view_598, [0], True);  view_598 = None
        view_599 = torch.ops.aten.view.default(sum_193, [768]);  sum_193 = None
        permute_532 = torch.ops.aten.permute.default(permute_531, [1, 0]);  permute_531 = None
        view_600 = torch.ops.aten.view.default(mm_144, [3, 5, 768]);  mm_144 = None
        add_172 = torch.ops.aten.add.Tensor(add_171, view_600);  add_171 = view_600 = None
        convert_element_type_37 = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_511 = torch.ops.aten.mul.Tensor(convert_element_type_37, 1.1111111111111112);  convert_element_type_37 = None
        mul_512 = torch.ops.aten.mul.Tensor(add_172, mul_511);  add_172 = mul_511 = None
        clone_132 = torch.ops.aten.clone.default(mul_512, memory_format = torch.contiguous_format);  mul_512 = None
        mul_514 = torch.ops.aten.mul.Tensor(clone_132, primals_4);  primals_4 = None
        mul_515 = torch.ops.aten.mul.Tensor(mul_514, 768)
        sum_194 = torch.ops.aten.sum.dim_IntList(mul_514, [2], True)
        mul_516 = torch.ops.aten.mul.Tensor(mul_514, mul_1);  mul_514 = None
        sum_195 = torch.ops.aten.sum.dim_IntList(mul_516, [2], True);  mul_516 = None
        mul_517 = torch.ops.aten.mul.Tensor(mul_1, sum_195);  sum_195 = None
        sub_124 = torch.ops.aten.sub.Tensor(mul_515, sum_194);  mul_515 = sum_194 = None
        sub_125 = torch.ops.aten.sub.Tensor(sub_124, mul_517);  sub_124 = mul_517 = None
        mul_518 = torch.ops.aten.mul.Tensor(div_60, sub_125);  div_60 = sub_125 = None
        mul_519 = torch.ops.aten.mul.Tensor(clone_132, mul_1);  mul_1 = None
        sum_196 = torch.ops.aten.sum.dim_IntList(mul_519, [0, 1]);  mul_519 = None
        sum_197 = torch.ops.aten.sum.dim_IntList(clone_132, [0, 1]);  clone_132 = None
        sum_198 = torch.ops.aten.sum.dim_IntList(mul_518, [0], True)
        eq = torch.ops.aten.eq.Scalar(slice_6, -1)
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
        full_default_2 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(unsqueeze_2, full_default_2, sum_198);  unsqueeze_2 = sum_198 = None
        full_default_3 = torch.ops.aten.full.default([512, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _unsafe_index_put = torch.ops.aten._unsafe_index_put.default(full_default_3, [slice_6], where, True);  full_default_3 = slice_6 = where = None
        eq_1 = torch.ops.aten.eq.Scalar(expand, -1)
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
        where_1 = torch.ops.aten.where.self(unsqueeze_3, full_default_2, mul_518);  unsqueeze_3 = None
        full_default_5 = torch.ops.aten.full.default([2, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _unsafe_index_put_1 = torch.ops.aten._unsafe_index_put.default(full_default_5, [expand], where_1, True);  full_default_5 = expand = where_1 = None
        eq_2 = torch.ops.aten.eq.Scalar(primals_202, 0)
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(eq_2, -1);  eq_2 = None
        where_2 = torch.ops.aten.where.self(unsqueeze_4, full_default_2, mul_518);  unsqueeze_4 = full_default_2 = mul_518 = None
        full_default_7 = torch.ops.aten.full.default([30522, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _unsafe_index_put_2 = torch.ops.aten._unsafe_index_put.default(full_default_7, [primals_202], where_2, True);  full_default_7 = primals_202 = where_2 = None
        return [_unsafe_index_put_2, _unsafe_index_put_1, _unsafe_index_put, sum_196, sum_197, permute_532, view_599, permute_528, view_596, permute_523, view_592, permute_511, view_580, sum_187, sum_188, permute_507, view_577, permute_503, view_574, sum_181, sum_182, permute_499, view_571, permute_495, view_568, permute_490, view_564, permute_478, view_552, sum_172, sum_173, permute_474, view_549, permute_470, view_546, sum_166, sum_167, permute_466, view_543, permute_462, view_540, permute_457, view_536, permute_445, view_524, sum_157, sum_158, permute_441, view_521, permute_437, view_518, sum_151, sum_152, permute_433, view_515, permute_429, view_512, permute_424, view_508, permute_412, view_496, sum_142, sum_143, permute_408, view_493, permute_404, view_490, sum_136, sum_137, permute_400, view_487, permute_396, view_484, permute_391, view_480, permute_379, view_468, sum_127, sum_128, permute_375, view_465, permute_371, view_462, sum_121, sum_122, permute_367, view_459, permute_363, view_456, permute_358, view_452, permute_346, view_440, sum_112, sum_113, permute_342, view_437, permute_338, view_434, sum_106, sum_107, permute_334, view_431, permute_330, view_428, permute_325, view_424, permute_313, view_412, sum_97, sum_98, permute_309, view_409, permute_305, view_406, sum_91, sum_92, permute_301, view_403, permute_297, view_400, permute_292, view_396, permute_280, view_384, sum_82, sum_83, permute_276, view_381, permute_272, view_378, sum_76, sum_77, permute_268, view_375, permute_264, view_372, permute_259, view_368, permute_247, view_356, sum_67, sum_68, permute_243, view_353, permute_239, view_350, sum_61, sum_62, permute_235, view_347, permute_231, view_344, permute_226, view_340, permute_214, view_328, sum_52, sum_53, permute_210, view_325, permute_206, view_322, sum_46, sum_47, permute_202, view_319, permute_198, view_316, permute_193, view_312, permute_181, view_300, sum_37, sum_38, permute_177, view_297, permute_173, view_294, sum_31, sum_32, permute_169, view_291, permute_165, view_288, permute_160, view_284, permute_148, view_272, sum_22, sum_23, permute_144, view_269, permute_140, view_266, sum_16, sum_17, permute_136, view_264, None, None, None, None]
        
def load_args(reader):
    buf0 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf0, (768,), is_leaf=True)  # primals_4
    buf1 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf1, (768,), is_leaf=True)  # primals_14
    buf2 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf2, (768,), is_leaf=True)  # primals_20
    buf3 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf3, (768,), is_leaf=True)  # primals_30
    buf4 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf4, (768,), is_leaf=True)  # primals_36
    buf5 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf5, (768,), is_leaf=True)  # primals_46
    buf6 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf6, (768,), is_leaf=True)  # primals_52
    buf7 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf7, (768,), is_leaf=True)  # primals_62
    buf8 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf8, (768,), is_leaf=True)  # primals_68
    buf9 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf9, (768,), is_leaf=True)  # primals_78
    buf10 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf10, (768,), is_leaf=True)  # primals_84
    buf11 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf11, (768,), is_leaf=True)  # primals_94
    buf12 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf12, (768,), is_leaf=True)  # primals_100
    buf13 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf13, (768,), is_leaf=True)  # primals_110
    buf14 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf14, (768,), is_leaf=True)  # primals_116
    buf15 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf15, (768,), is_leaf=True)  # primals_126
    buf16 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf16, (768,), is_leaf=True)  # primals_132
    buf17 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf17, (768,), is_leaf=True)  # primals_142
    buf18 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf18, (768,), is_leaf=True)  # primals_148
    buf19 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf19, (768,), is_leaf=True)  # primals_158
    buf20 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf20, (768,), is_leaf=True)  # primals_164
    buf21 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf21, (768,), is_leaf=True)  # primals_174
    buf22 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf22, (768,), is_leaf=True)  # primals_180
    buf23 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf23, (768,), is_leaf=True)  # primals_190
    buf24 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf24, (768,), is_leaf=True)  # primals_196
    buf25 = reader.storage(None, 120, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf25, (3, 5), dtype=torch.int64, is_leaf=True)  # primals_202
    buf26 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf26, (3, 5), (0, 1), dtype=torch.int64, is_leaf=True)  # expand
    buf27 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf27, (1, 5), (512, 1), dtype=torch.int64, is_leaf=True)  # slice_6
    buf28 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf28, (3, 5, 768), is_leaf=True)  # mul_1
    buf29 = reader.storage(None, 11520, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf29, (3, 5, 768), dtype=torch.bool, is_leaf=True)  # gt
    buf30 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf30, (15, 768), is_leaf=True)  # view
    buf31 = reader.storage(None, 900, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf31, (3, 12, 5, 5), dtype=torch.bool, is_leaf=True)  # gt_1
    buf32 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf32, (15, 768), is_leaf=True)  # view_16
    buf33 = reader.storage(None, 11520, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf33, (3, 5, 768), dtype=torch.bool, is_leaf=True)  # gt_2
    buf34 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf34, (3, 5, 768), is_leaf=True)  # mul_9
    buf35 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf35, (15, 768), is_leaf=True)  # view_18
    buf36 = reader.storage(None, 184320, device=device(type='cuda', index=0))
    reader.tensor(buf36, (15, 3072), is_leaf=True)  # addmm_4
    buf37 = reader.storage(None, 184320, device=device(type='cuda', index=0))
    reader.tensor(buf37, (15, 3072), is_leaf=True)  # view_20
    buf38 = reader.storage(None, 11520, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf38, (3, 5, 768), dtype=torch.bool, is_leaf=True)  # gt_3
    buf39 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf39, (3, 5, 768), is_leaf=True)  # mul_16
    buf40 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf40, (15, 768), is_leaf=True)  # view_22
    buf41 = reader.storage(None, 900, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf41, (3, 12, 5, 5), dtype=torch.bool, is_leaf=True)  # gt_4
    buf42 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf42, (15, 768), is_leaf=True)  # view_38
    buf43 = reader.storage(None, 11520, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf43, (3, 5, 768), dtype=torch.bool, is_leaf=True)  # gt_5
    buf44 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf44, (3, 5, 768), is_leaf=True)  # mul_22
    buf45 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf45, (15, 768), is_leaf=True)  # view_40
    buf46 = reader.storage(None, 184320, device=device(type='cuda', index=0))
    reader.tensor(buf46, (15, 3072), is_leaf=True)  # addmm_10
    buf47 = reader.storage(None, 184320, device=device(type='cuda', index=0))
    reader.tensor(buf47, (15, 3072), is_leaf=True)  # view_42
    buf48 = reader.storage(None, 11520, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf48, (3, 5, 768), dtype=torch.bool, is_leaf=True)  # gt_6
    buf49 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf49, (3, 5, 768), is_leaf=True)  # mul_29
    buf50 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf50, (15, 768), is_leaf=True)  # view_44
    buf51 = reader.storage(None, 900, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf51, (3, 12, 5, 5), dtype=torch.bool, is_leaf=True)  # gt_7
    buf52 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf52, (15, 768), is_leaf=True)  # view_60
    buf53 = reader.storage(None, 11520, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf53, (3, 5, 768), dtype=torch.bool, is_leaf=True)  # gt_8
    buf54 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf54, (3, 5, 768), is_leaf=True)  # mul_35
    buf55 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf55, (15, 768), is_leaf=True)  # view_62
    buf56 = reader.storage(None, 184320, device=device(type='cuda', index=0))
    reader.tensor(buf56, (15, 3072), is_leaf=True)  # addmm_16
    buf57 = reader.storage(None, 184320, device=device(type='cuda', index=0))
    reader.tensor(buf57, (15, 3072), is_leaf=True)  # view_64
    buf58 = reader.storage(None, 11520, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf58, (3, 5, 768), dtype=torch.bool, is_leaf=True)  # gt_9
    buf59 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf59, (3, 5, 768), is_leaf=True)  # mul_42
    buf60 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf60, (15, 768), is_leaf=True)  # view_66
    buf61 = reader.storage(None, 900, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf61, (3, 12, 5, 5), dtype=torch.bool, is_leaf=True)  # gt_10
    buf62 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf62, (15, 768), is_leaf=True)  # view_82
    buf63 = reader.storage(None, 11520, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf63, (3, 5, 768), dtype=torch.bool, is_leaf=True)  # gt_11
    buf64 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf64, (3, 5, 768), is_leaf=True)  # mul_48
    buf65 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf65, (15, 768), is_leaf=True)  # view_84
    buf66 = reader.storage(None, 184320, device=device(type='cuda', index=0))
    reader.tensor(buf66, (15, 3072), is_leaf=True)  # addmm_22
    buf67 = reader.storage(None, 184320, device=device(type='cuda', index=0))
    reader.tensor(buf67, (15, 3072), is_leaf=True)  # view_86
    buf68 = reader.storage(None, 11520, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf68, (3, 5, 768), dtype=torch.bool, is_leaf=True)  # gt_12
    buf69 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf69, (3, 5, 768), is_leaf=True)  # mul_55
    buf70 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf70, (15, 768), is_leaf=True)  # view_88
    buf71 = reader.storage(None, 900, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf71, (3, 12, 5, 5), dtype=torch.bool, is_leaf=True)  # gt_13
    buf72 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf72, (15, 768), is_leaf=True)  # view_104
    buf73 = reader.storage(None, 11520, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf73, (3, 5, 768), dtype=torch.bool, is_leaf=True)  # gt_14
    buf74 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf74, (3, 5, 768), is_leaf=True)  # mul_61
    buf75 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf75, (15, 768), is_leaf=True)  # view_106
    buf76 = reader.storage(None, 184320, device=device(type='cuda', index=0))
    reader.tensor(buf76, (15, 3072), is_leaf=True)  # addmm_28
    buf77 = reader.storage(None, 184320, device=device(type='cuda', index=0))
    reader.tensor(buf77, (15, 3072), is_leaf=True)  # view_108
    buf78 = reader.storage(None, 11520, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf78, (3, 5, 768), dtype=torch.bool, is_leaf=True)  # gt_15
    buf79 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf79, (3, 5, 768), is_leaf=True)  # mul_68
    buf80 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf80, (15, 768), is_leaf=True)  # view_110
    buf81 = reader.storage(None, 900, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf81, (3, 12, 5, 5), dtype=torch.bool, is_leaf=True)  # gt_16
    buf82 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf82, (15, 768), is_leaf=True)  # view_126
    buf83 = reader.storage(None, 11520, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf83, (3, 5, 768), dtype=torch.bool, is_leaf=True)  # gt_17
    buf84 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf84, (3, 5, 768), is_leaf=True)  # mul_74
    buf85 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf85, (15, 768), is_leaf=True)  # view_128
    buf86 = reader.storage(None, 184320, device=device(type='cuda', index=0))
    reader.tensor(buf86, (15, 3072), is_leaf=True)  # addmm_34
    buf87 = reader.storage(None, 184320, device=device(type='cuda', index=0))
    reader.tensor(buf87, (15, 3072), is_leaf=True)  # view_130
    buf88 = reader.storage(None, 11520, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf88, (3, 5, 768), dtype=torch.bool, is_leaf=True)  # gt_18
    buf89 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf89, (3, 5, 768), is_leaf=True)  # mul_81
    buf90 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf90, (15, 768), is_leaf=True)  # view_132
    buf91 = reader.storage(None, 900, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf91, (3, 12, 5, 5), dtype=torch.bool, is_leaf=True)  # gt_19
    buf92 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf92, (15, 768), is_leaf=True)  # view_148
    buf93 = reader.storage(None, 11520, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf93, (3, 5, 768), dtype=torch.bool, is_leaf=True)  # gt_20
    buf94 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf94, (3, 5, 768), is_leaf=True)  # mul_87
    buf95 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf95, (15, 768), is_leaf=True)  # view_150
    buf96 = reader.storage(None, 184320, device=device(type='cuda', index=0))
    reader.tensor(buf96, (15, 3072), is_leaf=True)  # addmm_40
    buf97 = reader.storage(None, 184320, device=device(type='cuda', index=0))
    reader.tensor(buf97, (15, 3072), is_leaf=True)  # view_152
    buf98 = reader.storage(None, 11520, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf98, (3, 5, 768), dtype=torch.bool, is_leaf=True)  # gt_21
    buf99 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf99, (3, 5, 768), is_leaf=True)  # mul_94
    buf100 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf100, (15, 768), is_leaf=True)  # view_154
    buf101 = reader.storage(None, 900, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf101, (3, 12, 5, 5), dtype=torch.bool, is_leaf=True)  # gt_22
    buf102 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf102, (15, 768), is_leaf=True)  # view_170
    buf103 = reader.storage(None, 11520, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf103, (3, 5, 768), dtype=torch.bool, is_leaf=True)  # gt_23
    buf104 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf104, (3, 5, 768), is_leaf=True)  # mul_100
    buf105 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf105, (15, 768), is_leaf=True)  # view_172
    buf106 = reader.storage(None, 184320, device=device(type='cuda', index=0))
    reader.tensor(buf106, (15, 3072), is_leaf=True)  # addmm_46
    buf107 = reader.storage(None, 184320, device=device(type='cuda', index=0))
    reader.tensor(buf107, (15, 3072), is_leaf=True)  # view_174
    buf108 = reader.storage(None, 11520, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf108, (3, 5, 768), dtype=torch.bool, is_leaf=True)  # gt_24
    buf109 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf109, (3, 5, 768), is_leaf=True)  # mul_107
    buf110 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf110, (15, 768), is_leaf=True)  # view_176
    buf111 = reader.storage(None, 900, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf111, (3, 12, 5, 5), dtype=torch.bool, is_leaf=True)  # gt_25
    buf112 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf112, (15, 768), is_leaf=True)  # view_192
    buf113 = reader.storage(None, 11520, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf113, (3, 5, 768), dtype=torch.bool, is_leaf=True)  # gt_26
    buf114 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf114, (3, 5, 768), is_leaf=True)  # mul_113
    buf115 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf115, (15, 768), is_leaf=True)  # view_194
    buf116 = reader.storage(None, 184320, device=device(type='cuda', index=0))
    reader.tensor(buf116, (15, 3072), is_leaf=True)  # addmm_52
    buf117 = reader.storage(None, 184320, device=device(type='cuda', index=0))
    reader.tensor(buf117, (15, 3072), is_leaf=True)  # view_196
    buf118 = reader.storage(None, 11520, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf118, (3, 5, 768), dtype=torch.bool, is_leaf=True)  # gt_27
    buf119 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf119, (3, 5, 768), is_leaf=True)  # mul_120
    buf120 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf120, (15, 768), is_leaf=True)  # view_198
    buf121 = reader.storage(None, 900, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf121, (3, 12, 5, 5), dtype=torch.bool, is_leaf=True)  # gt_28
    buf122 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf122, (15, 768), is_leaf=True)  # view_214
    buf123 = reader.storage(None, 11520, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf123, (3, 5, 768), dtype=torch.bool, is_leaf=True)  # gt_29
    buf124 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf124, (3, 5, 768), is_leaf=True)  # mul_126
    buf125 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf125, (15, 768), is_leaf=True)  # view_216
    buf126 = reader.storage(None, 184320, device=device(type='cuda', index=0))
    reader.tensor(buf126, (15, 3072), is_leaf=True)  # addmm_58
    buf127 = reader.storage(None, 184320, device=device(type='cuda', index=0))
    reader.tensor(buf127, (15, 3072), is_leaf=True)  # view_218
    buf128 = reader.storage(None, 11520, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf128, (3, 5, 768), dtype=torch.bool, is_leaf=True)  # gt_30
    buf129 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf129, (3, 5, 768), is_leaf=True)  # mul_133
    buf130 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf130, (15, 768), is_leaf=True)  # view_220
    buf131 = reader.storage(None, 900, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf131, (3, 12, 5, 5), dtype=torch.bool, is_leaf=True)  # gt_31
    buf132 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf132, (15, 768), is_leaf=True)  # view_236
    buf133 = reader.storage(None, 11520, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf133, (3, 5, 768), dtype=torch.bool, is_leaf=True)  # gt_32
    buf134 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf134, (3, 5, 768), is_leaf=True)  # mul_139
    buf135 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf135, (15, 768), is_leaf=True)  # view_238
    buf136 = reader.storage(None, 184320, device=device(type='cuda', index=0))
    reader.tensor(buf136, (15, 3072), is_leaf=True)  # addmm_64
    buf137 = reader.storage(None, 184320, device=device(type='cuda', index=0))
    reader.tensor(buf137, (15, 3072), is_leaf=True)  # view_240
    buf138 = reader.storage(None, 11520, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf138, (3, 5, 768), dtype=torch.bool, is_leaf=True)  # gt_33
    buf139 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf139, (3, 5, 768), is_leaf=True)  # mul_146
    buf140 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf140, (15, 768), is_leaf=True)  # view_242
    buf141 = reader.storage(None, 900, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf141, (3, 12, 5, 5), dtype=torch.bool, is_leaf=True)  # gt_34
    buf142 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf142, (15, 768), is_leaf=True)  # view_258
    buf143 = reader.storage(None, 11520, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf143, (3, 5, 768), dtype=torch.bool, is_leaf=True)  # gt_35
    buf144 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf144, (3, 5, 768), is_leaf=True)  # mul_152
    buf145 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf145, (15, 768), is_leaf=True)  # view_260
    buf146 = reader.storage(None, 184320, device=device(type='cuda', index=0))
    reader.tensor(buf146, (15, 3072), is_leaf=True)  # addmm_70
    buf147 = reader.storage(None, 184320, device=device(type='cuda', index=0))
    reader.tensor(buf147, (15, 3072), is_leaf=True)  # view_262
    buf148 = reader.storage(None, 11520, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf148, (3, 5, 768), dtype=torch.bool, is_leaf=True)  # gt_36
    buf149 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf149, (3, 5, 768), is_leaf=True)  # mul_159
    buf150 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf150, (3, 768), (3840, 1), is_leaf=True)  # select
    buf151 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf151, (3, 768), is_leaf=True)  # tanh
    buf152 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf152, (768, 768), is_leaf=True)  # permute_133
    buf153 = reader.storage(None, 60, device=device(type='cuda', index=0))
    reader.tensor(buf153, (3, 5, 1), is_leaf=True)  # div_24
    buf154 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf154, (768, 3072), is_leaf=True)  # permute_137
    buf155 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf155, (3072, 768), is_leaf=True)  # permute_141
    buf156 = reader.storage(None, 60, device=device(type='cuda', index=0))
    reader.tensor(buf156, (3, 5, 1), is_leaf=True)  # div_25
    buf157 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf157, (768, 768), is_leaf=True)  # permute_145
    buf158 = reader.storage(None, 3600, device=device(type='cuda', index=0))
    reader.tensor(buf158, (36, 5, 5), (25, 1, 5), is_leaf=True)  # permute_150
    buf159 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf159, (36, 64, 5), (320, 1, 64), is_leaf=True)  # permute_151
    buf160 = reader.storage(None, 3600, device=device(type='cuda', index=0))
    reader.tensor(buf160, (3, 12, 5, 5), is_leaf=True)  # alias_14
    buf161 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf161, (36, 64, 5), (320, 1, 64), is_leaf=True)  # permute_152
    buf162 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf162, (36, 5, 64), (320, 1, 5), is_leaf=True)  # permute_153
    buf163 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf163, (768, 768), is_leaf=True)  # permute_157
    buf164 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf164, (768, 768), is_leaf=True)  # permute_162
    buf165 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf165, (768, 768), is_leaf=True)  # permute_166
    buf166 = reader.storage(None, 60, device=device(type='cuda', index=0))
    reader.tensor(buf166, (3, 5, 1), is_leaf=True)  # div_27
    buf167 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf167, (768, 3072), is_leaf=True)  # permute_170
    buf168 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf168, (3072, 768), is_leaf=True)  # permute_174
    buf169 = reader.storage(None, 60, device=device(type='cuda', index=0))
    reader.tensor(buf169, (3, 5, 1), is_leaf=True)  # div_28
    buf170 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf170, (768, 768), is_leaf=True)  # permute_178
    buf171 = reader.storage(None, 3600, device=device(type='cuda', index=0))
    reader.tensor(buf171, (36, 5, 5), (25, 1, 5), is_leaf=True)  # permute_183
    buf172 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf172, (36, 64, 5), (320, 1, 64), is_leaf=True)  # permute_184
    buf173 = reader.storage(None, 3600, device=device(type='cuda', index=0))
    reader.tensor(buf173, (3, 12, 5, 5), is_leaf=True)  # alias_15
    buf174 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf174, (36, 64, 5), (320, 1, 64), is_leaf=True)  # permute_185
    buf175 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf175, (36, 5, 64), (320, 1, 5), is_leaf=True)  # permute_186
    buf176 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf176, (768, 768), is_leaf=True)  # permute_190
    buf177 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf177, (768, 768), is_leaf=True)  # permute_195
    buf178 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf178, (768, 768), is_leaf=True)  # permute_199
    buf179 = reader.storage(None, 60, device=device(type='cuda', index=0))
    reader.tensor(buf179, (3, 5, 1), is_leaf=True)  # div_30
    buf180 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf180, (768, 3072), is_leaf=True)  # permute_203
    buf181 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf181, (3072, 768), is_leaf=True)  # permute_207
    buf182 = reader.storage(None, 60, device=device(type='cuda', index=0))
    reader.tensor(buf182, (3, 5, 1), is_leaf=True)  # div_31
    buf183 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf183, (768, 768), is_leaf=True)  # permute_211
    buf184 = reader.storage(None, 3600, device=device(type='cuda', index=0))
    reader.tensor(buf184, (36, 5, 5), (25, 1, 5), is_leaf=True)  # permute_216
    buf185 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf185, (36, 64, 5), (320, 1, 64), is_leaf=True)  # permute_217
    buf186 = reader.storage(None, 3600, device=device(type='cuda', index=0))
    reader.tensor(buf186, (3, 12, 5, 5), is_leaf=True)  # alias_16
    buf187 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf187, (36, 64, 5), (320, 1, 64), is_leaf=True)  # permute_218
    buf188 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf188, (36, 5, 64), (320, 1, 5), is_leaf=True)  # permute_219
    buf189 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf189, (768, 768), is_leaf=True)  # permute_223
    buf190 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf190, (768, 768), is_leaf=True)  # permute_228
    buf191 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf191, (768, 768), is_leaf=True)  # permute_232
    buf192 = reader.storage(None, 60, device=device(type='cuda', index=0))
    reader.tensor(buf192, (3, 5, 1), is_leaf=True)  # div_33
    buf193 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf193, (768, 3072), is_leaf=True)  # permute_236
    buf194 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf194, (3072, 768), is_leaf=True)  # permute_240
    buf195 = reader.storage(None, 60, device=device(type='cuda', index=0))
    reader.tensor(buf195, (3, 5, 1), is_leaf=True)  # div_34
    buf196 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf196, (768, 768), is_leaf=True)  # permute_244
    buf197 = reader.storage(None, 3600, device=device(type='cuda', index=0))
    reader.tensor(buf197, (36, 5, 5), (25, 1, 5), is_leaf=True)  # permute_249
    buf198 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf198, (36, 64, 5), (320, 1, 64), is_leaf=True)  # permute_250
    buf199 = reader.storage(None, 3600, device=device(type='cuda', index=0))
    reader.tensor(buf199, (3, 12, 5, 5), is_leaf=True)  # alias_17
    buf200 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf200, (36, 64, 5), (320, 1, 64), is_leaf=True)  # permute_251
    buf201 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf201, (36, 5, 64), (320, 1, 5), is_leaf=True)  # permute_252
    buf202 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf202, (768, 768), is_leaf=True)  # permute_256
    buf203 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf203, (768, 768), is_leaf=True)  # permute_261
    buf204 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf204, (768, 768), is_leaf=True)  # permute_265
    buf205 = reader.storage(None, 60, device=device(type='cuda', index=0))
    reader.tensor(buf205, (3, 5, 1), is_leaf=True)  # div_36
    buf206 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf206, (768, 3072), is_leaf=True)  # permute_269
    buf207 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf207, (3072, 768), is_leaf=True)  # permute_273
    buf208 = reader.storage(None, 60, device=device(type='cuda', index=0))
    reader.tensor(buf208, (3, 5, 1), is_leaf=True)  # div_37
    buf209 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf209, (768, 768), is_leaf=True)  # permute_277
    buf210 = reader.storage(None, 3600, device=device(type='cuda', index=0))
    reader.tensor(buf210, (36, 5, 5), (25, 1, 5), is_leaf=True)  # permute_282
    buf211 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf211, (36, 64, 5), (320, 1, 64), is_leaf=True)  # permute_283
    buf212 = reader.storage(None, 3600, device=device(type='cuda', index=0))
    reader.tensor(buf212, (3, 12, 5, 5), is_leaf=True)  # alias_18
    buf213 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf213, (36, 64, 5), (320, 1, 64), is_leaf=True)  # permute_284
    buf214 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf214, (36, 5, 64), (320, 1, 5), is_leaf=True)  # permute_285
    buf215 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf215, (768, 768), is_leaf=True)  # permute_289
    buf216 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf216, (768, 768), is_leaf=True)  # permute_294
    buf217 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf217, (768, 768), is_leaf=True)  # permute_298
    buf218 = reader.storage(None, 60, device=device(type='cuda', index=0))
    reader.tensor(buf218, (3, 5, 1), is_leaf=True)  # div_39
    buf219 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf219, (768, 3072), is_leaf=True)  # permute_302
    buf220 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf220, (3072, 768), is_leaf=True)  # permute_306
    buf221 = reader.storage(None, 60, device=device(type='cuda', index=0))
    reader.tensor(buf221, (3, 5, 1), is_leaf=True)  # div_40
    buf222 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf222, (768, 768), is_leaf=True)  # permute_310
    buf223 = reader.storage(None, 3600, device=device(type='cuda', index=0))
    reader.tensor(buf223, (36, 5, 5), (25, 1, 5), is_leaf=True)  # permute_315
    buf224 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf224, (36, 64, 5), (320, 1, 64), is_leaf=True)  # permute_316
    buf225 = reader.storage(None, 3600, device=device(type='cuda', index=0))
    reader.tensor(buf225, (3, 12, 5, 5), is_leaf=True)  # alias_19
    buf226 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf226, (36, 64, 5), (320, 1, 64), is_leaf=True)  # permute_317
    buf227 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf227, (36, 5, 64), (320, 1, 5), is_leaf=True)  # permute_318
    buf228 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf228, (768, 768), is_leaf=True)  # permute_322
    buf229 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf229, (768, 768), is_leaf=True)  # permute_327
    buf230 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf230, (768, 768), is_leaf=True)  # permute_331
    buf231 = reader.storage(None, 60, device=device(type='cuda', index=0))
    reader.tensor(buf231, (3, 5, 1), is_leaf=True)  # div_42
    buf232 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf232, (768, 3072), is_leaf=True)  # permute_335
    buf233 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf233, (3072, 768), is_leaf=True)  # permute_339
    buf234 = reader.storage(None, 60, device=device(type='cuda', index=0))
    reader.tensor(buf234, (3, 5, 1), is_leaf=True)  # div_43
    buf235 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf235, (768, 768), is_leaf=True)  # permute_343
    buf236 = reader.storage(None, 3600, device=device(type='cuda', index=0))
    reader.tensor(buf236, (36, 5, 5), (25, 1, 5), is_leaf=True)  # permute_348
    buf237 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf237, (36, 64, 5), (320, 1, 64), is_leaf=True)  # permute_349
    buf238 = reader.storage(None, 3600, device=device(type='cuda', index=0))
    reader.tensor(buf238, (3, 12, 5, 5), is_leaf=True)  # alias_20
    buf239 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf239, (36, 64, 5), (320, 1, 64), is_leaf=True)  # permute_350
    buf240 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf240, (36, 5, 64), (320, 1, 5), is_leaf=True)  # permute_351
    buf241 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf241, (768, 768), is_leaf=True)  # permute_355
    buf242 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf242, (768, 768), is_leaf=True)  # permute_360
    buf243 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf243, (768, 768), is_leaf=True)  # permute_364
    buf244 = reader.storage(None, 60, device=device(type='cuda', index=0))
    reader.tensor(buf244, (3, 5, 1), is_leaf=True)  # div_45
    buf245 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf245, (768, 3072), is_leaf=True)  # permute_368
    buf246 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf246, (3072, 768), is_leaf=True)  # permute_372
    buf247 = reader.storage(None, 60, device=device(type='cuda', index=0))
    reader.tensor(buf247, (3, 5, 1), is_leaf=True)  # div_46
    buf248 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf248, (768, 768), is_leaf=True)  # permute_376
    buf249 = reader.storage(None, 3600, device=device(type='cuda', index=0))
    reader.tensor(buf249, (36, 5, 5), (25, 1, 5), is_leaf=True)  # permute_381
    buf250 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf250, (36, 64, 5), (320, 1, 64), is_leaf=True)  # permute_382
    buf251 = reader.storage(None, 3600, device=device(type='cuda', index=0))
    reader.tensor(buf251, (3, 12, 5, 5), is_leaf=True)  # alias_21
    buf252 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf252, (36, 64, 5), (320, 1, 64), is_leaf=True)  # permute_383
    buf253 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf253, (36, 5, 64), (320, 1, 5), is_leaf=True)  # permute_384
    buf254 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf254, (768, 768), is_leaf=True)  # permute_388
    buf255 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf255, (768, 768), is_leaf=True)  # permute_393
    buf256 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf256, (768, 768), is_leaf=True)  # permute_397
    buf257 = reader.storage(None, 60, device=device(type='cuda', index=0))
    reader.tensor(buf257, (3, 5, 1), is_leaf=True)  # div_48
    buf258 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf258, (768, 3072), is_leaf=True)  # permute_401
    buf259 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf259, (3072, 768), is_leaf=True)  # permute_405
    buf260 = reader.storage(None, 60, device=device(type='cuda', index=0))
    reader.tensor(buf260, (3, 5, 1), is_leaf=True)  # div_49
    buf261 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf261, (768, 768), is_leaf=True)  # permute_409
    buf262 = reader.storage(None, 3600, device=device(type='cuda', index=0))
    reader.tensor(buf262, (36, 5, 5), (25, 1, 5), is_leaf=True)  # permute_414
    buf263 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf263, (36, 64, 5), (320, 1, 64), is_leaf=True)  # permute_415
    buf264 = reader.storage(None, 3600, device=device(type='cuda', index=0))
    reader.tensor(buf264, (3, 12, 5, 5), is_leaf=True)  # alias_22
    buf265 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf265, (36, 64, 5), (320, 1, 64), is_leaf=True)  # permute_416
    buf266 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf266, (36, 5, 64), (320, 1, 5), is_leaf=True)  # permute_417
    buf267 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf267, (768, 768), is_leaf=True)  # permute_421
    buf268 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf268, (768, 768), is_leaf=True)  # permute_426
    buf269 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf269, (768, 768), is_leaf=True)  # permute_430
    buf270 = reader.storage(None, 60, device=device(type='cuda', index=0))
    reader.tensor(buf270, (3, 5, 1), is_leaf=True)  # div_51
    buf271 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf271, (768, 3072), is_leaf=True)  # permute_434
    buf272 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf272, (3072, 768), is_leaf=True)  # permute_438
    buf273 = reader.storage(None, 60, device=device(type='cuda', index=0))
    reader.tensor(buf273, (3, 5, 1), is_leaf=True)  # div_52
    buf274 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf274, (768, 768), is_leaf=True)  # permute_442
    buf275 = reader.storage(None, 3600, device=device(type='cuda', index=0))
    reader.tensor(buf275, (36, 5, 5), (25, 1, 5), is_leaf=True)  # permute_447
    buf276 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf276, (36, 64, 5), (320, 1, 64), is_leaf=True)  # permute_448
    buf277 = reader.storage(None, 3600, device=device(type='cuda', index=0))
    reader.tensor(buf277, (3, 12, 5, 5), is_leaf=True)  # alias_23
    buf278 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf278, (36, 64, 5), (320, 1, 64), is_leaf=True)  # permute_449
    buf279 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf279, (36, 5, 64), (320, 1, 5), is_leaf=True)  # permute_450
    buf280 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf280, (768, 768), is_leaf=True)  # permute_454
    buf281 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf281, (768, 768), is_leaf=True)  # permute_459
    buf282 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf282, (768, 768), is_leaf=True)  # permute_463
    buf283 = reader.storage(None, 60, device=device(type='cuda', index=0))
    reader.tensor(buf283, (3, 5, 1), is_leaf=True)  # div_54
    buf284 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf284, (768, 3072), is_leaf=True)  # permute_467
    buf285 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf285, (3072, 768), is_leaf=True)  # permute_471
    buf286 = reader.storage(None, 60, device=device(type='cuda', index=0))
    reader.tensor(buf286, (3, 5, 1), is_leaf=True)  # div_55
    buf287 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf287, (768, 768), is_leaf=True)  # permute_475
    buf288 = reader.storage(None, 3600, device=device(type='cuda', index=0))
    reader.tensor(buf288, (36, 5, 5), (25, 1, 5), is_leaf=True)  # permute_480
    buf289 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf289, (36, 64, 5), (320, 1, 64), is_leaf=True)  # permute_481
    buf290 = reader.storage(None, 3600, device=device(type='cuda', index=0))
    reader.tensor(buf290, (3, 12, 5, 5), is_leaf=True)  # alias_24
    buf291 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf291, (36, 64, 5), (320, 1, 64), is_leaf=True)  # permute_482
    buf292 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf292, (36, 5, 64), (320, 1, 5), is_leaf=True)  # permute_483
    buf293 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf293, (768, 768), is_leaf=True)  # permute_487
    buf294 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf294, (768, 768), is_leaf=True)  # permute_492
    buf295 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf295, (768, 768), is_leaf=True)  # permute_496
    buf296 = reader.storage(None, 60, device=device(type='cuda', index=0))
    reader.tensor(buf296, (3, 5, 1), is_leaf=True)  # div_57
    buf297 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf297, (768, 3072), is_leaf=True)  # permute_500
    buf298 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf298, (3072, 768), is_leaf=True)  # permute_504
    buf299 = reader.storage(None, 60, device=device(type='cuda', index=0))
    reader.tensor(buf299, (3, 5, 1), is_leaf=True)  # div_58
    buf300 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf300, (768, 768), is_leaf=True)  # permute_508
    buf301 = reader.storage(None, 3600, device=device(type='cuda', index=0))
    reader.tensor(buf301, (36, 5, 5), (25, 1, 5), is_leaf=True)  # permute_513
    buf302 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf302, (36, 64, 5), (320, 1, 64), is_leaf=True)  # permute_514
    buf303 = reader.storage(None, 3600, device=device(type='cuda', index=0))
    reader.tensor(buf303, (3, 12, 5, 5), is_leaf=True)  # alias_25
    buf304 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf304, (36, 64, 5), (320, 1, 64), is_leaf=True)  # permute_515
    buf305 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf305, (36, 5, 64), (320, 1, 5), is_leaf=True)  # permute_516
    buf306 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf306, (768, 768), is_leaf=True)  # permute_520
    buf307 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf307, (768, 768), is_leaf=True)  # permute_525
    buf308 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf308, (768, 768), is_leaf=True)  # permute_529
    buf309 = reader.storage(None, 60, device=device(type='cuda', index=0))
    reader.tensor(buf309, (3, 5, 1), is_leaf=True)  # div_60
    buf310 = reader.storage(None, 46080, device=device(type='cuda', index=0))
    reader.tensor(buf310, (3, 5, 768), is_leaf=True)  # tangents_1
    buf311 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf311, (3, 768), is_leaf=True)  # tangents_2
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
