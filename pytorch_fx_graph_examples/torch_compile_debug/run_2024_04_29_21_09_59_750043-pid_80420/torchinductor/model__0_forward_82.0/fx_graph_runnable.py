
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

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203):
        slice_1 = torch.ops.aten.slice.Tensor(primals_200, 0, 0, 9223372036854775807);  primals_200 = None
        slice_2 = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 5);  slice_1 = None
        expand = torch.ops.aten.expand.default(slice_2, [3, 5]);  slice_2 = None
        slice_3 = torch.ops.aten.slice.Tensor(primals_203, 0, 0, 9223372036854775807);  primals_203 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(slice_3, 1);  slice_3 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        slice_4 = torch.ops.aten.slice.Tensor(unsqueeze_1, 3, 0, 9223372036854775807);  unsqueeze_1 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(slice_4, torch.float32);  slice_4 = None
        sub = torch.ops.aten.sub.Tensor(1.0, convert_element_type);  convert_element_type = None
        mul = torch.ops.aten.mul.Tensor(sub, -3.4028234663852886e+38);  sub = None
        slice_5 = torch.ops.aten.slice.Tensor(primals_201, 0, 0, 9223372036854775807);  primals_201 = None
        slice_6 = torch.ops.aten.slice.Tensor(slice_5, 1, 0, 5);  slice_5 = None
        embedding = torch.ops.aten.embedding.default(primals_1, primals_202, 0);  primals_1 = None
        embedding_1 = torch.ops.aten.embedding.default(primals_2, expand);  primals_2 = None
        add = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        embedding_2 = torch.ops.aten.embedding.default(primals_3, slice_6);  primals_3 = None
        add_1 = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
        var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add_2 = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub_1 = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
        mul_1 = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = None
        mul_2 = torch.ops.aten.mul.Tensor(mul_1, primals_4)
        add_3 = torch.ops.aten.add.Tensor(mul_2, primals_5);  mul_2 = primals_5 = None
        inductor_seeds_default = torch.ops.prims.inductor_seeds.default(37, device(type='cuda', index=0))
        inductor_lookup_seed_default = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_36 = torch.ops.prims.inductor_random.default([3, 5, 768], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt = torch.ops.aten.gt.Scalar(inductor_random_default_36, 0.1);  inductor_random_default_36 = None
        mul_3 = torch.ops.aten.mul.Tensor(gt, add_3);  add_3 = None
        mul_4 = torch.ops.aten.mul.Tensor(mul_3, 1.1111111111111112);  mul_3 = None
        view = torch.ops.aten.view.default(mul_4, [15, 768])
        permute = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
        addmm = torch.ops.aten.addmm.default(primals_7, view, permute);  primals_7 = None
        view_1 = torch.ops.aten.view.default(addmm, [3, 5, 768]);  addmm = None
        permute_1 = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        addmm_1 = torch.ops.aten.addmm.default(primals_9, view, permute_1);  primals_9 = None
        view_3 = torch.ops.aten.view.default(addmm_1, [3, 5, 768]);  addmm_1 = None
        view_4 = torch.ops.aten.view.default(view_3, [3, 5, 12, 64]);  view_3 = None
        permute_2 = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        permute_3 = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
        addmm_2 = torch.ops.aten.addmm.default(primals_11, view, permute_3);  primals_11 = None
        view_6 = torch.ops.aten.view.default(addmm_2, [3, 5, 768]);  addmm_2 = None
        view_7 = torch.ops.aten.view.default(view_6, [3, 5, 12, 64]);  view_6 = None
        permute_4 = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
        view_8 = torch.ops.aten.view.default(view_1, [3, 5, 12, 64]);  view_1 = None
        permute_5 = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        permute_6 = torch.ops.aten.permute.default(permute_2, [0, 1, 3, 2]);  permute_2 = None
        expand_1 = torch.ops.aten.expand.default(permute_5, [3, 12, 5, 64]);  permute_5 = None
        clone = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_9 = torch.ops.aten.view.default(clone, [36, 5, 64]);  clone = None
        expand_2 = torch.ops.aten.expand.default(permute_6, [3, 12, 64, 5]);  permute_6 = None
        clone_1 = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
        view_10 = torch.ops.aten.view.default(clone_1, [36, 64, 5]);  clone_1 = None
        bmm = torch.ops.aten.bmm.default(view_9, view_10)
        view_11 = torch.ops.aten.view.default(bmm, [3, 12, 5, 5]);  bmm = None
        div = torch.ops.aten.div.Tensor(view_11, 8.0);  view_11 = None
        add_4 = torch.ops.aten.add.Tensor(div, mul);  div = None
        amax = torch.ops.aten.amax.default(add_4, [-1], True)
        sub_2 = torch.ops.aten.sub.Tensor(add_4, amax);  add_4 = amax = None
        exp = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_1 = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        alias = torch.ops.aten.alias.default(div_1)
        inductor_lookup_seed_default_1 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1)
        inductor_random_default_35 = torch.ops.prims.inductor_random.default([3, 12, 5, 5], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        gt_1 = torch.ops.aten.gt.Scalar(inductor_random_default_35, 0.1);  inductor_random_default_35 = None
        mul_5 = torch.ops.aten.mul.Tensor(gt_1, div_1);  div_1 = None
        mul_6 = torch.ops.aten.mul.Tensor(mul_5, 1.1111111111111112);  mul_5 = None
        expand_3 = torch.ops.aten.expand.default(mul_6, [3, 12, 5, 5]);  mul_6 = None
        view_12 = torch.ops.aten.view.default(expand_3, [36, 5, 5]);  expand_3 = None
        expand_4 = torch.ops.aten.expand.default(permute_4, [3, 12, 5, 64]);  permute_4 = None
        clone_2 = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_13 = torch.ops.aten.view.default(clone_2, [36, 5, 64]);  clone_2 = None
        bmm_1 = torch.ops.aten.bmm.default(view_12, view_13)
        view_14 = torch.ops.aten.view.default(bmm_1, [3, 12, 5, 64]);  bmm_1 = None
        permute_7 = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
        clone_3 = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_15 = torch.ops.aten.view.default(clone_3, [3, 5, 768]);  clone_3 = None
        view_16 = torch.ops.aten.view.default(view_15, [15, 768]);  view_15 = None
        permute_8 = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
        addmm_3 = torch.ops.aten.addmm.default(primals_13, view_16, permute_8);  primals_13 = None
        view_17 = torch.ops.aten.view.default(addmm_3, [3, 5, 768]);  addmm_3 = None
        inductor_lookup_seed_default_2 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2)
        inductor_random_default_34 = torch.ops.prims.inductor_random.default([3, 5, 768], inductor_lookup_seed_default_2, 'rand');  inductor_lookup_seed_default_2 = None
        gt_2 = torch.ops.aten.gt.Scalar(inductor_random_default_34, 0.1);  inductor_random_default_34 = None
        mul_7 = torch.ops.aten.mul.Tensor(gt_2, view_17);  view_17 = None
        mul_8 = torch.ops.aten.mul.Tensor(mul_7, 1.1111111111111112);  mul_7 = None
        add_5 = torch.ops.aten.add.Tensor(mul_8, mul_4);  mul_8 = mul_4 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_5, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_1[0]
        getitem_3 = var_mean_1[1];  var_mean_1 = None
        add_6 = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_3 = torch.ops.aten.sub.Tensor(add_5, getitem_3);  add_5 = getitem_3 = None
        mul_9 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_1);  sub_3 = None
        mul_10 = torch.ops.aten.mul.Tensor(mul_9, primals_14)
        add_7 = torch.ops.aten.add.Tensor(mul_10, primals_15);  mul_10 = primals_15 = None
        view_18 = torch.ops.aten.view.default(add_7, [15, 768])
        permute_9 = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
        addmm_4 = torch.ops.aten.addmm.default(primals_17, view_18, permute_9);  primals_17 = None
        view_19 = torch.ops.aten.view.default(addmm_4, [3, 5, 3072])
        mul_11 = torch.ops.aten.mul.Tensor(view_19, 0.5)
        mul_12 = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476);  view_19 = None
        erf = torch.ops.aten.erf.default(mul_12);  mul_12 = None
        add_8 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_13 = torch.ops.aten.mul.Tensor(mul_11, add_8);  mul_11 = add_8 = None
        view_20 = torch.ops.aten.view.default(mul_13, [15, 3072]);  mul_13 = None
        permute_10 = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
        addmm_5 = torch.ops.aten.addmm.default(primals_19, view_20, permute_10);  primals_19 = None
        view_21 = torch.ops.aten.view.default(addmm_5, [3, 5, 768]);  addmm_5 = None
        inductor_lookup_seed_default_3 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 3)
        inductor_random_default_33 = torch.ops.prims.inductor_random.default([3, 5, 768], inductor_lookup_seed_default_3, 'rand');  inductor_lookup_seed_default_3 = None
        gt_3 = torch.ops.aten.gt.Scalar(inductor_random_default_33, 0.1);  inductor_random_default_33 = None
        mul_14 = torch.ops.aten.mul.Tensor(gt_3, view_21);  view_21 = None
        mul_15 = torch.ops.aten.mul.Tensor(mul_14, 1.1111111111111112);  mul_14 = None
        add_9 = torch.ops.aten.add.Tensor(mul_15, add_7);  mul_15 = add_7 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_9, [2], correction = 0, keepdim = True)
        getitem_4 = var_mean_2[0]
        getitem_5 = var_mean_2[1];  var_mean_2 = None
        add_10 = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        sub_4 = torch.ops.aten.sub.Tensor(add_9, getitem_5);  add_9 = getitem_5 = None
        mul_16 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = None
        mul_17 = torch.ops.aten.mul.Tensor(mul_16, primals_20)
        add_11 = torch.ops.aten.add.Tensor(mul_17, primals_21);  mul_17 = primals_21 = None
        view_22 = torch.ops.aten.view.default(add_11, [15, 768])
        permute_11 = torch.ops.aten.permute.default(primals_22, [1, 0]);  primals_22 = None
        addmm_6 = torch.ops.aten.addmm.default(primals_23, view_22, permute_11);  primals_23 = None
        view_23 = torch.ops.aten.view.default(addmm_6, [3, 5, 768]);  addmm_6 = None
        permute_12 = torch.ops.aten.permute.default(primals_24, [1, 0]);  primals_24 = None
        addmm_7 = torch.ops.aten.addmm.default(primals_25, view_22, permute_12);  primals_25 = None
        view_25 = torch.ops.aten.view.default(addmm_7, [3, 5, 768]);  addmm_7 = None
        view_26 = torch.ops.aten.view.default(view_25, [3, 5, 12, 64]);  view_25 = None
        permute_13 = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None
        permute_14 = torch.ops.aten.permute.default(primals_26, [1, 0]);  primals_26 = None
        addmm_8 = torch.ops.aten.addmm.default(primals_27, view_22, permute_14);  primals_27 = None
        view_28 = torch.ops.aten.view.default(addmm_8, [3, 5, 768]);  addmm_8 = None
        view_29 = torch.ops.aten.view.default(view_28, [3, 5, 12, 64]);  view_28 = None
        permute_15 = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
        view_30 = torch.ops.aten.view.default(view_23, [3, 5, 12, 64]);  view_23 = None
        permute_16 = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
        permute_17 = torch.ops.aten.permute.default(permute_13, [0, 1, 3, 2]);  permute_13 = None
        expand_5 = torch.ops.aten.expand.default(permute_16, [3, 12, 5, 64]);  permute_16 = None
        clone_4 = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_31 = torch.ops.aten.view.default(clone_4, [36, 5, 64]);  clone_4 = None
        expand_6 = torch.ops.aten.expand.default(permute_17, [3, 12, 64, 5]);  permute_17 = None
        clone_5 = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        view_32 = torch.ops.aten.view.default(clone_5, [36, 64, 5]);  clone_5 = None
        bmm_2 = torch.ops.aten.bmm.default(view_31, view_32)
        view_33 = torch.ops.aten.view.default(bmm_2, [3, 12, 5, 5]);  bmm_2 = None
        div_2 = torch.ops.aten.div.Tensor(view_33, 8.0);  view_33 = None
        add_12 = torch.ops.aten.add.Tensor(div_2, mul);  div_2 = None
        amax_1 = torch.ops.aten.amax.default(add_12, [-1], True)
        sub_5 = torch.ops.aten.sub.Tensor(add_12, amax_1);  add_12 = amax_1 = None
        exp_1 = torch.ops.aten.exp.default(sub_5);  sub_5 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_3 = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        alias_1 = torch.ops.aten.alias.default(div_3)
        inductor_lookup_seed_default_4 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 4)
        inductor_random_default_32 = torch.ops.prims.inductor_random.default([3, 12, 5, 5], inductor_lookup_seed_default_4, 'rand');  inductor_lookup_seed_default_4 = None
        gt_4 = torch.ops.aten.gt.Scalar(inductor_random_default_32, 0.1);  inductor_random_default_32 = None
        mul_18 = torch.ops.aten.mul.Tensor(gt_4, div_3);  div_3 = None
        mul_19 = torch.ops.aten.mul.Tensor(mul_18, 1.1111111111111112);  mul_18 = None
        expand_7 = torch.ops.aten.expand.default(mul_19, [3, 12, 5, 5]);  mul_19 = None
        view_34 = torch.ops.aten.view.default(expand_7, [36, 5, 5]);  expand_7 = None
        expand_8 = torch.ops.aten.expand.default(permute_15, [3, 12, 5, 64]);  permute_15 = None
        clone_6 = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_35 = torch.ops.aten.view.default(clone_6, [36, 5, 64]);  clone_6 = None
        bmm_3 = torch.ops.aten.bmm.default(view_34, view_35)
        view_36 = torch.ops.aten.view.default(bmm_3, [3, 12, 5, 64]);  bmm_3 = None
        permute_18 = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None
        clone_7 = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_37 = torch.ops.aten.view.default(clone_7, [3, 5, 768]);  clone_7 = None
        view_38 = torch.ops.aten.view.default(view_37, [15, 768]);  view_37 = None
        permute_19 = torch.ops.aten.permute.default(primals_28, [1, 0]);  primals_28 = None
        addmm_9 = torch.ops.aten.addmm.default(primals_29, view_38, permute_19);  primals_29 = None
        view_39 = torch.ops.aten.view.default(addmm_9, [3, 5, 768]);  addmm_9 = None
        inductor_lookup_seed_default_5 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 5)
        inductor_random_default_31 = torch.ops.prims.inductor_random.default([3, 5, 768], inductor_lookup_seed_default_5, 'rand');  inductor_lookup_seed_default_5 = None
        gt_5 = torch.ops.aten.gt.Scalar(inductor_random_default_31, 0.1);  inductor_random_default_31 = None
        mul_20 = torch.ops.aten.mul.Tensor(gt_5, view_39);  view_39 = None
        mul_21 = torch.ops.aten.mul.Tensor(mul_20, 1.1111111111111112);  mul_20 = None
        add_13 = torch.ops.aten.add.Tensor(mul_21, add_11);  mul_21 = add_11 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_13, [2], correction = 0, keepdim = True)
        getitem_6 = var_mean_3[0]
        getitem_7 = var_mean_3[1];  var_mean_3 = None
        add_14 = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        sub_6 = torch.ops.aten.sub.Tensor(add_13, getitem_7);  add_13 = getitem_7 = None
        mul_22 = torch.ops.aten.mul.Tensor(sub_6, rsqrt_3);  sub_6 = None
        mul_23 = torch.ops.aten.mul.Tensor(mul_22, primals_30)
        add_15 = torch.ops.aten.add.Tensor(mul_23, primals_31);  mul_23 = primals_31 = None
        view_40 = torch.ops.aten.view.default(add_15, [15, 768])
        permute_20 = torch.ops.aten.permute.default(primals_32, [1, 0]);  primals_32 = None
        addmm_10 = torch.ops.aten.addmm.default(primals_33, view_40, permute_20);  primals_33 = None
        view_41 = torch.ops.aten.view.default(addmm_10, [3, 5, 3072])
        mul_24 = torch.ops.aten.mul.Tensor(view_41, 0.5)
        mul_25 = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476);  view_41 = None
        erf_1 = torch.ops.aten.erf.default(mul_25);  mul_25 = None
        add_16 = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_26 = torch.ops.aten.mul.Tensor(mul_24, add_16);  mul_24 = add_16 = None
        view_42 = torch.ops.aten.view.default(mul_26, [15, 3072]);  mul_26 = None
        permute_21 = torch.ops.aten.permute.default(primals_34, [1, 0]);  primals_34 = None
        addmm_11 = torch.ops.aten.addmm.default(primals_35, view_42, permute_21);  primals_35 = None
        view_43 = torch.ops.aten.view.default(addmm_11, [3, 5, 768]);  addmm_11 = None
        inductor_lookup_seed_default_6 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 6)
        inductor_random_default_30 = torch.ops.prims.inductor_random.default([3, 5, 768], inductor_lookup_seed_default_6, 'rand');  inductor_lookup_seed_default_6 = None
        gt_6 = torch.ops.aten.gt.Scalar(inductor_random_default_30, 0.1);  inductor_random_default_30 = None
        mul_27 = torch.ops.aten.mul.Tensor(gt_6, view_43);  view_43 = None
        mul_28 = torch.ops.aten.mul.Tensor(mul_27, 1.1111111111111112);  mul_27 = None
        add_17 = torch.ops.aten.add.Tensor(mul_28, add_15);  mul_28 = add_15 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
        getitem_8 = var_mean_4[0]
        getitem_9 = var_mean_4[1];  var_mean_4 = None
        add_18 = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_7 = torch.ops.aten.sub.Tensor(add_17, getitem_9);  add_17 = getitem_9 = None
        mul_29 = torch.ops.aten.mul.Tensor(sub_7, rsqrt_4);  sub_7 = None
        mul_30 = torch.ops.aten.mul.Tensor(mul_29, primals_36)
        add_19 = torch.ops.aten.add.Tensor(mul_30, primals_37);  mul_30 = primals_37 = None
        view_44 = torch.ops.aten.view.default(add_19, [15, 768])
        permute_22 = torch.ops.aten.permute.default(primals_38, [1, 0]);  primals_38 = None
        addmm_12 = torch.ops.aten.addmm.default(primals_39, view_44, permute_22);  primals_39 = None
        view_45 = torch.ops.aten.view.default(addmm_12, [3, 5, 768]);  addmm_12 = None
        permute_23 = torch.ops.aten.permute.default(primals_40, [1, 0]);  primals_40 = None
        addmm_13 = torch.ops.aten.addmm.default(primals_41, view_44, permute_23);  primals_41 = None
        view_47 = torch.ops.aten.view.default(addmm_13, [3, 5, 768]);  addmm_13 = None
        view_48 = torch.ops.aten.view.default(view_47, [3, 5, 12, 64]);  view_47 = None
        permute_24 = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None
        permute_25 = torch.ops.aten.permute.default(primals_42, [1, 0]);  primals_42 = None
        addmm_14 = torch.ops.aten.addmm.default(primals_43, view_44, permute_25);  primals_43 = None
        view_50 = torch.ops.aten.view.default(addmm_14, [3, 5, 768]);  addmm_14 = None
        view_51 = torch.ops.aten.view.default(view_50, [3, 5, 12, 64]);  view_50 = None
        permute_26 = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None
        view_52 = torch.ops.aten.view.default(view_45, [3, 5, 12, 64]);  view_45 = None
        permute_27 = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
        permute_28 = torch.ops.aten.permute.default(permute_24, [0, 1, 3, 2]);  permute_24 = None
        expand_9 = torch.ops.aten.expand.default(permute_27, [3, 12, 5, 64]);  permute_27 = None
        clone_8 = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_53 = torch.ops.aten.view.default(clone_8, [36, 5, 64]);  clone_8 = None
        expand_10 = torch.ops.aten.expand.default(permute_28, [3, 12, 64, 5]);  permute_28 = None
        clone_9 = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        view_54 = torch.ops.aten.view.default(clone_9, [36, 64, 5]);  clone_9 = None
        bmm_4 = torch.ops.aten.bmm.default(view_53, view_54)
        view_55 = torch.ops.aten.view.default(bmm_4, [3, 12, 5, 5]);  bmm_4 = None
        div_4 = torch.ops.aten.div.Tensor(view_55, 8.0);  view_55 = None
        add_20 = torch.ops.aten.add.Tensor(div_4, mul);  div_4 = None
        amax_2 = torch.ops.aten.amax.default(add_20, [-1], True)
        sub_8 = torch.ops.aten.sub.Tensor(add_20, amax_2);  add_20 = amax_2 = None
        exp_2 = torch.ops.aten.exp.default(sub_8);  sub_8 = None
        sum_3 = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_5 = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        alias_2 = torch.ops.aten.alias.default(div_5)
        inductor_lookup_seed_default_7 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 7)
        inductor_random_default_29 = torch.ops.prims.inductor_random.default([3, 12, 5, 5], inductor_lookup_seed_default_7, 'rand');  inductor_lookup_seed_default_7 = None
        gt_7 = torch.ops.aten.gt.Scalar(inductor_random_default_29, 0.1);  inductor_random_default_29 = None
        mul_31 = torch.ops.aten.mul.Tensor(gt_7, div_5);  div_5 = None
        mul_32 = torch.ops.aten.mul.Tensor(mul_31, 1.1111111111111112);  mul_31 = None
        expand_11 = torch.ops.aten.expand.default(mul_32, [3, 12, 5, 5]);  mul_32 = None
        view_56 = torch.ops.aten.view.default(expand_11, [36, 5, 5]);  expand_11 = None
        expand_12 = torch.ops.aten.expand.default(permute_26, [3, 12, 5, 64]);  permute_26 = None
        clone_10 = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_57 = torch.ops.aten.view.default(clone_10, [36, 5, 64]);  clone_10 = None
        bmm_5 = torch.ops.aten.bmm.default(view_56, view_57)
        view_58 = torch.ops.aten.view.default(bmm_5, [3, 12, 5, 64]);  bmm_5 = None
        permute_29 = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
        clone_11 = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_59 = torch.ops.aten.view.default(clone_11, [3, 5, 768]);  clone_11 = None
        view_60 = torch.ops.aten.view.default(view_59, [15, 768]);  view_59 = None
        permute_30 = torch.ops.aten.permute.default(primals_44, [1, 0]);  primals_44 = None
        addmm_15 = torch.ops.aten.addmm.default(primals_45, view_60, permute_30);  primals_45 = None
        view_61 = torch.ops.aten.view.default(addmm_15, [3, 5, 768]);  addmm_15 = None
        inductor_lookup_seed_default_8 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 8)
        inductor_random_default_28 = torch.ops.prims.inductor_random.default([3, 5, 768], inductor_lookup_seed_default_8, 'rand');  inductor_lookup_seed_default_8 = None
        gt_8 = torch.ops.aten.gt.Scalar(inductor_random_default_28, 0.1);  inductor_random_default_28 = None
        mul_33 = torch.ops.aten.mul.Tensor(gt_8, view_61);  view_61 = None
        mul_34 = torch.ops.aten.mul.Tensor(mul_33, 1.1111111111111112);  mul_33 = None
        add_21 = torch.ops.aten.add.Tensor(mul_34, add_19);  mul_34 = add_19 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
        getitem_10 = var_mean_5[0]
        getitem_11 = var_mean_5[1];  var_mean_5 = None
        add_22 = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        sub_9 = torch.ops.aten.sub.Tensor(add_21, getitem_11);  add_21 = getitem_11 = None
        mul_35 = torch.ops.aten.mul.Tensor(sub_9, rsqrt_5);  sub_9 = None
        mul_36 = torch.ops.aten.mul.Tensor(mul_35, primals_46)
        add_23 = torch.ops.aten.add.Tensor(mul_36, primals_47);  mul_36 = primals_47 = None
        view_62 = torch.ops.aten.view.default(add_23, [15, 768])
        permute_31 = torch.ops.aten.permute.default(primals_48, [1, 0]);  primals_48 = None
        addmm_16 = torch.ops.aten.addmm.default(primals_49, view_62, permute_31);  primals_49 = None
        view_63 = torch.ops.aten.view.default(addmm_16, [3, 5, 3072])
        mul_37 = torch.ops.aten.mul.Tensor(view_63, 0.5)
        mul_38 = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476);  view_63 = None
        erf_2 = torch.ops.aten.erf.default(mul_38);  mul_38 = None
        add_24 = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_39 = torch.ops.aten.mul.Tensor(mul_37, add_24);  mul_37 = add_24 = None
        view_64 = torch.ops.aten.view.default(mul_39, [15, 3072]);  mul_39 = None
        permute_32 = torch.ops.aten.permute.default(primals_50, [1, 0]);  primals_50 = None
        addmm_17 = torch.ops.aten.addmm.default(primals_51, view_64, permute_32);  primals_51 = None
        view_65 = torch.ops.aten.view.default(addmm_17, [3, 5, 768]);  addmm_17 = None
        inductor_lookup_seed_default_9 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 9)
        inductor_random_default_27 = torch.ops.prims.inductor_random.default([3, 5, 768], inductor_lookup_seed_default_9, 'rand');  inductor_lookup_seed_default_9 = None
        gt_9 = torch.ops.aten.gt.Scalar(inductor_random_default_27, 0.1);  inductor_random_default_27 = None
        mul_40 = torch.ops.aten.mul.Tensor(gt_9, view_65);  view_65 = None
        mul_41 = torch.ops.aten.mul.Tensor(mul_40, 1.1111111111111112);  mul_40 = None
        add_25 = torch.ops.aten.add.Tensor(mul_41, add_23);  mul_41 = add_23 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
        getitem_12 = var_mean_6[0]
        getitem_13 = var_mean_6[1];  var_mean_6 = None
        add_26 = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        sub_10 = torch.ops.aten.sub.Tensor(add_25, getitem_13);  add_25 = getitem_13 = None
        mul_42 = torch.ops.aten.mul.Tensor(sub_10, rsqrt_6);  sub_10 = None
        mul_43 = torch.ops.aten.mul.Tensor(mul_42, primals_52)
        add_27 = torch.ops.aten.add.Tensor(mul_43, primals_53);  mul_43 = primals_53 = None
        view_66 = torch.ops.aten.view.default(add_27, [15, 768])
        permute_33 = torch.ops.aten.permute.default(primals_54, [1, 0]);  primals_54 = None
        addmm_18 = torch.ops.aten.addmm.default(primals_55, view_66, permute_33);  primals_55 = None
        view_67 = torch.ops.aten.view.default(addmm_18, [3, 5, 768]);  addmm_18 = None
        permute_34 = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
        addmm_19 = torch.ops.aten.addmm.default(primals_57, view_66, permute_34);  primals_57 = None
        view_69 = torch.ops.aten.view.default(addmm_19, [3, 5, 768]);  addmm_19 = None
        view_70 = torch.ops.aten.view.default(view_69, [3, 5, 12, 64]);  view_69 = None
        permute_35 = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None
        permute_36 = torch.ops.aten.permute.default(primals_58, [1, 0]);  primals_58 = None
        addmm_20 = torch.ops.aten.addmm.default(primals_59, view_66, permute_36);  primals_59 = None
        view_72 = torch.ops.aten.view.default(addmm_20, [3, 5, 768]);  addmm_20 = None
        view_73 = torch.ops.aten.view.default(view_72, [3, 5, 12, 64]);  view_72 = None
        permute_37 = torch.ops.aten.permute.default(view_73, [0, 2, 1, 3]);  view_73 = None
        view_74 = torch.ops.aten.view.default(view_67, [3, 5, 12, 64]);  view_67 = None
        permute_38 = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
        permute_39 = torch.ops.aten.permute.default(permute_35, [0, 1, 3, 2]);  permute_35 = None
        expand_13 = torch.ops.aten.expand.default(permute_38, [3, 12, 5, 64]);  permute_38 = None
        clone_12 = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_75 = torch.ops.aten.view.default(clone_12, [36, 5, 64]);  clone_12 = None
        expand_14 = torch.ops.aten.expand.default(permute_39, [3, 12, 64, 5]);  permute_39 = None
        clone_13 = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
        view_76 = torch.ops.aten.view.default(clone_13, [36, 64, 5]);  clone_13 = None
        bmm_6 = torch.ops.aten.bmm.default(view_75, view_76)
        view_77 = torch.ops.aten.view.default(bmm_6, [3, 12, 5, 5]);  bmm_6 = None
        div_6 = torch.ops.aten.div.Tensor(view_77, 8.0);  view_77 = None
        add_28 = torch.ops.aten.add.Tensor(div_6, mul);  div_6 = None
        amax_3 = torch.ops.aten.amax.default(add_28, [-1], True)
        sub_11 = torch.ops.aten.sub.Tensor(add_28, amax_3);  add_28 = amax_3 = None
        exp_3 = torch.ops.aten.exp.default(sub_11);  sub_11 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_7 = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        alias_3 = torch.ops.aten.alias.default(div_7)
        inductor_lookup_seed_default_10 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 10)
        inductor_random_default_26 = torch.ops.prims.inductor_random.default([3, 12, 5, 5], inductor_lookup_seed_default_10, 'rand');  inductor_lookup_seed_default_10 = None
        gt_10 = torch.ops.aten.gt.Scalar(inductor_random_default_26, 0.1);  inductor_random_default_26 = None
        mul_44 = torch.ops.aten.mul.Tensor(gt_10, div_7);  div_7 = None
        mul_45 = torch.ops.aten.mul.Tensor(mul_44, 1.1111111111111112);  mul_44 = None
        expand_15 = torch.ops.aten.expand.default(mul_45, [3, 12, 5, 5]);  mul_45 = None
        view_78 = torch.ops.aten.view.default(expand_15, [36, 5, 5]);  expand_15 = None
        expand_16 = torch.ops.aten.expand.default(permute_37, [3, 12, 5, 64]);  permute_37 = None
        clone_14 = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_79 = torch.ops.aten.view.default(clone_14, [36, 5, 64]);  clone_14 = None
        bmm_7 = torch.ops.aten.bmm.default(view_78, view_79)
        view_80 = torch.ops.aten.view.default(bmm_7, [3, 12, 5, 64]);  bmm_7 = None
        permute_40 = torch.ops.aten.permute.default(view_80, [0, 2, 1, 3]);  view_80 = None
        clone_15 = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_81 = torch.ops.aten.view.default(clone_15, [3, 5, 768]);  clone_15 = None
        view_82 = torch.ops.aten.view.default(view_81, [15, 768]);  view_81 = None
        permute_41 = torch.ops.aten.permute.default(primals_60, [1, 0]);  primals_60 = None
        addmm_21 = torch.ops.aten.addmm.default(primals_61, view_82, permute_41);  primals_61 = None
        view_83 = torch.ops.aten.view.default(addmm_21, [3, 5, 768]);  addmm_21 = None
        inductor_lookup_seed_default_11 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 11)
        inductor_random_default_25 = torch.ops.prims.inductor_random.default([3, 5, 768], inductor_lookup_seed_default_11, 'rand');  inductor_lookup_seed_default_11 = None
        gt_11 = torch.ops.aten.gt.Scalar(inductor_random_default_25, 0.1);  inductor_random_default_25 = None
        mul_46 = torch.ops.aten.mul.Tensor(gt_11, view_83);  view_83 = None
        mul_47 = torch.ops.aten.mul.Tensor(mul_46, 1.1111111111111112);  mul_46 = None
        add_29 = torch.ops.aten.add.Tensor(mul_47, add_27);  mul_47 = add_27 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_29, [2], correction = 0, keepdim = True)
        getitem_14 = var_mean_7[0]
        getitem_15 = var_mean_7[1];  var_mean_7 = None
        add_30 = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        sub_12 = torch.ops.aten.sub.Tensor(add_29, getitem_15);  add_29 = getitem_15 = None
        mul_48 = torch.ops.aten.mul.Tensor(sub_12, rsqrt_7);  sub_12 = None
        mul_49 = torch.ops.aten.mul.Tensor(mul_48, primals_62)
        add_31 = torch.ops.aten.add.Tensor(mul_49, primals_63);  mul_49 = primals_63 = None
        view_84 = torch.ops.aten.view.default(add_31, [15, 768])
        permute_42 = torch.ops.aten.permute.default(primals_64, [1, 0]);  primals_64 = None
        addmm_22 = torch.ops.aten.addmm.default(primals_65, view_84, permute_42);  primals_65 = None
        view_85 = torch.ops.aten.view.default(addmm_22, [3, 5, 3072])
        mul_50 = torch.ops.aten.mul.Tensor(view_85, 0.5)
        mul_51 = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476);  view_85 = None
        erf_3 = torch.ops.aten.erf.default(mul_51);  mul_51 = None
        add_32 = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_52 = torch.ops.aten.mul.Tensor(mul_50, add_32);  mul_50 = add_32 = None
        view_86 = torch.ops.aten.view.default(mul_52, [15, 3072]);  mul_52 = None
        permute_43 = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
        addmm_23 = torch.ops.aten.addmm.default(primals_67, view_86, permute_43);  primals_67 = None
        view_87 = torch.ops.aten.view.default(addmm_23, [3, 5, 768]);  addmm_23 = None
        inductor_lookup_seed_default_12 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 12)
        inductor_random_default_24 = torch.ops.prims.inductor_random.default([3, 5, 768], inductor_lookup_seed_default_12, 'rand');  inductor_lookup_seed_default_12 = None
        gt_12 = torch.ops.aten.gt.Scalar(inductor_random_default_24, 0.1);  inductor_random_default_24 = None
        mul_53 = torch.ops.aten.mul.Tensor(gt_12, view_87);  view_87 = None
        mul_54 = torch.ops.aten.mul.Tensor(mul_53, 1.1111111111111112);  mul_53 = None
        add_33 = torch.ops.aten.add.Tensor(mul_54, add_31);  mul_54 = add_31 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_33, [2], correction = 0, keepdim = True)
        getitem_16 = var_mean_8[0]
        getitem_17 = var_mean_8[1];  var_mean_8 = None
        add_34 = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        sub_13 = torch.ops.aten.sub.Tensor(add_33, getitem_17);  add_33 = getitem_17 = None
        mul_55 = torch.ops.aten.mul.Tensor(sub_13, rsqrt_8);  sub_13 = None
        mul_56 = torch.ops.aten.mul.Tensor(mul_55, primals_68)
        add_35 = torch.ops.aten.add.Tensor(mul_56, primals_69);  mul_56 = primals_69 = None
        view_88 = torch.ops.aten.view.default(add_35, [15, 768])
        permute_44 = torch.ops.aten.permute.default(primals_70, [1, 0]);  primals_70 = None
        addmm_24 = torch.ops.aten.addmm.default(primals_71, view_88, permute_44);  primals_71 = None
        view_89 = torch.ops.aten.view.default(addmm_24, [3, 5, 768]);  addmm_24 = None
        permute_45 = torch.ops.aten.permute.default(primals_72, [1, 0]);  primals_72 = None
        addmm_25 = torch.ops.aten.addmm.default(primals_73, view_88, permute_45);  primals_73 = None
        view_91 = torch.ops.aten.view.default(addmm_25, [3, 5, 768]);  addmm_25 = None
        view_92 = torch.ops.aten.view.default(view_91, [3, 5, 12, 64]);  view_91 = None
        permute_46 = torch.ops.aten.permute.default(view_92, [0, 2, 1, 3]);  view_92 = None
        permute_47 = torch.ops.aten.permute.default(primals_74, [1, 0]);  primals_74 = None
        addmm_26 = torch.ops.aten.addmm.default(primals_75, view_88, permute_47);  primals_75 = None
        view_94 = torch.ops.aten.view.default(addmm_26, [3, 5, 768]);  addmm_26 = None
        view_95 = torch.ops.aten.view.default(view_94, [3, 5, 12, 64]);  view_94 = None
        permute_48 = torch.ops.aten.permute.default(view_95, [0, 2, 1, 3]);  view_95 = None
        view_96 = torch.ops.aten.view.default(view_89, [3, 5, 12, 64]);  view_89 = None
        permute_49 = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
        permute_50 = torch.ops.aten.permute.default(permute_46, [0, 1, 3, 2]);  permute_46 = None
        expand_17 = torch.ops.aten.expand.default(permute_49, [3, 12, 5, 64]);  permute_49 = None
        clone_16 = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_97 = torch.ops.aten.view.default(clone_16, [36, 5, 64]);  clone_16 = None
        expand_18 = torch.ops.aten.expand.default(permute_50, [3, 12, 64, 5]);  permute_50 = None
        clone_17 = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_98 = torch.ops.aten.view.default(clone_17, [36, 64, 5]);  clone_17 = None
        bmm_8 = torch.ops.aten.bmm.default(view_97, view_98)
        view_99 = torch.ops.aten.view.default(bmm_8, [3, 12, 5, 5]);  bmm_8 = None
        div_8 = torch.ops.aten.div.Tensor(view_99, 8.0);  view_99 = None
        add_36 = torch.ops.aten.add.Tensor(div_8, mul);  div_8 = None
        amax_4 = torch.ops.aten.amax.default(add_36, [-1], True)
        sub_14 = torch.ops.aten.sub.Tensor(add_36, amax_4);  add_36 = amax_4 = None
        exp_4 = torch.ops.aten.exp.default(sub_14);  sub_14 = None
        sum_5 = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_9 = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        alias_4 = torch.ops.aten.alias.default(div_9)
        inductor_lookup_seed_default_13 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 13)
        inductor_random_default_23 = torch.ops.prims.inductor_random.default([3, 12, 5, 5], inductor_lookup_seed_default_13, 'rand');  inductor_lookup_seed_default_13 = None
        gt_13 = torch.ops.aten.gt.Scalar(inductor_random_default_23, 0.1);  inductor_random_default_23 = None
        mul_57 = torch.ops.aten.mul.Tensor(gt_13, div_9);  div_9 = None
        mul_58 = torch.ops.aten.mul.Tensor(mul_57, 1.1111111111111112);  mul_57 = None
        expand_19 = torch.ops.aten.expand.default(mul_58, [3, 12, 5, 5]);  mul_58 = None
        view_100 = torch.ops.aten.view.default(expand_19, [36, 5, 5]);  expand_19 = None
        expand_20 = torch.ops.aten.expand.default(permute_48, [3, 12, 5, 64]);  permute_48 = None
        clone_18 = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_101 = torch.ops.aten.view.default(clone_18, [36, 5, 64]);  clone_18 = None
        bmm_9 = torch.ops.aten.bmm.default(view_100, view_101)
        view_102 = torch.ops.aten.view.default(bmm_9, [3, 12, 5, 64]);  bmm_9 = None
        permute_51 = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
        clone_19 = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
        view_103 = torch.ops.aten.view.default(clone_19, [3, 5, 768]);  clone_19 = None
        view_104 = torch.ops.aten.view.default(view_103, [15, 768]);  view_103 = None
        permute_52 = torch.ops.aten.permute.default(primals_76, [1, 0]);  primals_76 = None
        addmm_27 = torch.ops.aten.addmm.default(primals_77, view_104, permute_52);  primals_77 = None
        view_105 = torch.ops.aten.view.default(addmm_27, [3, 5, 768]);  addmm_27 = None
        inductor_lookup_seed_default_14 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 14)
        inductor_random_default_22 = torch.ops.prims.inductor_random.default([3, 5, 768], inductor_lookup_seed_default_14, 'rand');  inductor_lookup_seed_default_14 = None
        gt_14 = torch.ops.aten.gt.Scalar(inductor_random_default_22, 0.1);  inductor_random_default_22 = None
        mul_59 = torch.ops.aten.mul.Tensor(gt_14, view_105);  view_105 = None
        mul_60 = torch.ops.aten.mul.Tensor(mul_59, 1.1111111111111112);  mul_59 = None
        add_37 = torch.ops.aten.add.Tensor(mul_60, add_35);  mul_60 = add_35 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_37, [2], correction = 0, keepdim = True)
        getitem_18 = var_mean_9[0]
        getitem_19 = var_mean_9[1];  var_mean_9 = None
        add_38 = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        sub_15 = torch.ops.aten.sub.Tensor(add_37, getitem_19);  add_37 = getitem_19 = None
        mul_61 = torch.ops.aten.mul.Tensor(sub_15, rsqrt_9);  sub_15 = None
        mul_62 = torch.ops.aten.mul.Tensor(mul_61, primals_78)
        add_39 = torch.ops.aten.add.Tensor(mul_62, primals_79);  mul_62 = primals_79 = None
        view_106 = torch.ops.aten.view.default(add_39, [15, 768])
        permute_53 = torch.ops.aten.permute.default(primals_80, [1, 0]);  primals_80 = None
        addmm_28 = torch.ops.aten.addmm.default(primals_81, view_106, permute_53);  primals_81 = None
        view_107 = torch.ops.aten.view.default(addmm_28, [3, 5, 3072])
        mul_63 = torch.ops.aten.mul.Tensor(view_107, 0.5)
        mul_64 = torch.ops.aten.mul.Tensor(view_107, 0.7071067811865476);  view_107 = None
        erf_4 = torch.ops.aten.erf.default(mul_64);  mul_64 = None
        add_40 = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_65 = torch.ops.aten.mul.Tensor(mul_63, add_40);  mul_63 = add_40 = None
        view_108 = torch.ops.aten.view.default(mul_65, [15, 3072]);  mul_65 = None
        permute_54 = torch.ops.aten.permute.default(primals_82, [1, 0]);  primals_82 = None
        addmm_29 = torch.ops.aten.addmm.default(primals_83, view_108, permute_54);  primals_83 = None
        view_109 = torch.ops.aten.view.default(addmm_29, [3, 5, 768]);  addmm_29 = None
        inductor_lookup_seed_default_15 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 15)
        inductor_random_default_21 = torch.ops.prims.inductor_random.default([3, 5, 768], inductor_lookup_seed_default_15, 'rand');  inductor_lookup_seed_default_15 = None
        gt_15 = torch.ops.aten.gt.Scalar(inductor_random_default_21, 0.1);  inductor_random_default_21 = None
        mul_66 = torch.ops.aten.mul.Tensor(gt_15, view_109);  view_109 = None
        mul_67 = torch.ops.aten.mul.Tensor(mul_66, 1.1111111111111112);  mul_66 = None
        add_41 = torch.ops.aten.add.Tensor(mul_67, add_39);  mul_67 = add_39 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
        getitem_20 = var_mean_10[0]
        getitem_21 = var_mean_10[1];  var_mean_10 = None
        add_42 = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        sub_16 = torch.ops.aten.sub.Tensor(add_41, getitem_21);  add_41 = getitem_21 = None
        mul_68 = torch.ops.aten.mul.Tensor(sub_16, rsqrt_10);  sub_16 = None
        mul_69 = torch.ops.aten.mul.Tensor(mul_68, primals_84)
        add_43 = torch.ops.aten.add.Tensor(mul_69, primals_85);  mul_69 = primals_85 = None
        view_110 = torch.ops.aten.view.default(add_43, [15, 768])
        permute_55 = torch.ops.aten.permute.default(primals_86, [1, 0]);  primals_86 = None
        addmm_30 = torch.ops.aten.addmm.default(primals_87, view_110, permute_55);  primals_87 = None
        view_111 = torch.ops.aten.view.default(addmm_30, [3, 5, 768]);  addmm_30 = None
        permute_56 = torch.ops.aten.permute.default(primals_88, [1, 0]);  primals_88 = None
        addmm_31 = torch.ops.aten.addmm.default(primals_89, view_110, permute_56);  primals_89 = None
        view_113 = torch.ops.aten.view.default(addmm_31, [3, 5, 768]);  addmm_31 = None
        view_114 = torch.ops.aten.view.default(view_113, [3, 5, 12, 64]);  view_113 = None
        permute_57 = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None
        permute_58 = torch.ops.aten.permute.default(primals_90, [1, 0]);  primals_90 = None
        addmm_32 = torch.ops.aten.addmm.default(primals_91, view_110, permute_58);  primals_91 = None
        view_116 = torch.ops.aten.view.default(addmm_32, [3, 5, 768]);  addmm_32 = None
        view_117 = torch.ops.aten.view.default(view_116, [3, 5, 12, 64]);  view_116 = None
        permute_59 = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None
        view_118 = torch.ops.aten.view.default(view_111, [3, 5, 12, 64]);  view_111 = None
        permute_60 = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None
        permute_61 = torch.ops.aten.permute.default(permute_57, [0, 1, 3, 2]);  permute_57 = None
        expand_21 = torch.ops.aten.expand.default(permute_60, [3, 12, 5, 64]);  permute_60 = None
        clone_20 = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_119 = torch.ops.aten.view.default(clone_20, [36, 5, 64]);  clone_20 = None
        expand_22 = torch.ops.aten.expand.default(permute_61, [3, 12, 64, 5]);  permute_61 = None
        clone_21 = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
        view_120 = torch.ops.aten.view.default(clone_21, [36, 64, 5]);  clone_21 = None
        bmm_10 = torch.ops.aten.bmm.default(view_119, view_120)
        view_121 = torch.ops.aten.view.default(bmm_10, [3, 12, 5, 5]);  bmm_10 = None
        div_10 = torch.ops.aten.div.Tensor(view_121, 8.0);  view_121 = None
        add_44 = torch.ops.aten.add.Tensor(div_10, mul);  div_10 = None
        amax_5 = torch.ops.aten.amax.default(add_44, [-1], True)
        sub_17 = torch.ops.aten.sub.Tensor(add_44, amax_5);  add_44 = amax_5 = None
        exp_5 = torch.ops.aten.exp.default(sub_17);  sub_17 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_11 = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        alias_5 = torch.ops.aten.alias.default(div_11)
        inductor_lookup_seed_default_16 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 16)
        inductor_random_default_20 = torch.ops.prims.inductor_random.default([3, 12, 5, 5], inductor_lookup_seed_default_16, 'rand');  inductor_lookup_seed_default_16 = None
        gt_16 = torch.ops.aten.gt.Scalar(inductor_random_default_20, 0.1);  inductor_random_default_20 = None
        mul_70 = torch.ops.aten.mul.Tensor(gt_16, div_11);  div_11 = None
        mul_71 = torch.ops.aten.mul.Tensor(mul_70, 1.1111111111111112);  mul_70 = None
        expand_23 = torch.ops.aten.expand.default(mul_71, [3, 12, 5, 5]);  mul_71 = None
        view_122 = torch.ops.aten.view.default(expand_23, [36, 5, 5]);  expand_23 = None
        expand_24 = torch.ops.aten.expand.default(permute_59, [3, 12, 5, 64]);  permute_59 = None
        clone_22 = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_123 = torch.ops.aten.view.default(clone_22, [36, 5, 64]);  clone_22 = None
        bmm_11 = torch.ops.aten.bmm.default(view_122, view_123)
        view_124 = torch.ops.aten.view.default(bmm_11, [3, 12, 5, 64]);  bmm_11 = None
        permute_62 = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None
        clone_23 = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_125 = torch.ops.aten.view.default(clone_23, [3, 5, 768]);  clone_23 = None
        view_126 = torch.ops.aten.view.default(view_125, [15, 768]);  view_125 = None
        permute_63 = torch.ops.aten.permute.default(primals_92, [1, 0]);  primals_92 = None
        addmm_33 = torch.ops.aten.addmm.default(primals_93, view_126, permute_63);  primals_93 = None
        view_127 = torch.ops.aten.view.default(addmm_33, [3, 5, 768]);  addmm_33 = None
        inductor_lookup_seed_default_17 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 17)
        inductor_random_default_19 = torch.ops.prims.inductor_random.default([3, 5, 768], inductor_lookup_seed_default_17, 'rand');  inductor_lookup_seed_default_17 = None
        gt_17 = torch.ops.aten.gt.Scalar(inductor_random_default_19, 0.1);  inductor_random_default_19 = None
        mul_72 = torch.ops.aten.mul.Tensor(gt_17, view_127);  view_127 = None
        mul_73 = torch.ops.aten.mul.Tensor(mul_72, 1.1111111111111112);  mul_72 = None
        add_45 = torch.ops.aten.add.Tensor(mul_73, add_43);  mul_73 = add_43 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
        getitem_22 = var_mean_11[0]
        getitem_23 = var_mean_11[1];  var_mean_11 = None
        add_46 = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        sub_18 = torch.ops.aten.sub.Tensor(add_45, getitem_23);  add_45 = getitem_23 = None
        mul_74 = torch.ops.aten.mul.Tensor(sub_18, rsqrt_11);  sub_18 = None
        mul_75 = torch.ops.aten.mul.Tensor(mul_74, primals_94)
        add_47 = torch.ops.aten.add.Tensor(mul_75, primals_95);  mul_75 = primals_95 = None
        view_128 = torch.ops.aten.view.default(add_47, [15, 768])
        permute_64 = torch.ops.aten.permute.default(primals_96, [1, 0]);  primals_96 = None
        addmm_34 = torch.ops.aten.addmm.default(primals_97, view_128, permute_64);  primals_97 = None
        view_129 = torch.ops.aten.view.default(addmm_34, [3, 5, 3072])
        mul_76 = torch.ops.aten.mul.Tensor(view_129, 0.5)
        mul_77 = torch.ops.aten.mul.Tensor(view_129, 0.7071067811865476);  view_129 = None
        erf_5 = torch.ops.aten.erf.default(mul_77);  mul_77 = None
        add_48 = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_78 = torch.ops.aten.mul.Tensor(mul_76, add_48);  mul_76 = add_48 = None
        view_130 = torch.ops.aten.view.default(mul_78, [15, 3072]);  mul_78 = None
        permute_65 = torch.ops.aten.permute.default(primals_98, [1, 0]);  primals_98 = None
        addmm_35 = torch.ops.aten.addmm.default(primals_99, view_130, permute_65);  primals_99 = None
        view_131 = torch.ops.aten.view.default(addmm_35, [3, 5, 768]);  addmm_35 = None
        inductor_lookup_seed_default_18 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 18)
        inductor_random_default_18 = torch.ops.prims.inductor_random.default([3, 5, 768], inductor_lookup_seed_default_18, 'rand');  inductor_lookup_seed_default_18 = None
        gt_18 = torch.ops.aten.gt.Scalar(inductor_random_default_18, 0.1);  inductor_random_default_18 = None
        mul_79 = torch.ops.aten.mul.Tensor(gt_18, view_131);  view_131 = None
        mul_80 = torch.ops.aten.mul.Tensor(mul_79, 1.1111111111111112);  mul_79 = None
        add_49 = torch.ops.aten.add.Tensor(mul_80, add_47);  mul_80 = add_47 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
        getitem_24 = var_mean_12[0]
        getitem_25 = var_mean_12[1];  var_mean_12 = None
        add_50 = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        sub_19 = torch.ops.aten.sub.Tensor(add_49, getitem_25);  add_49 = getitem_25 = None
        mul_81 = torch.ops.aten.mul.Tensor(sub_19, rsqrt_12);  sub_19 = None
        mul_82 = torch.ops.aten.mul.Tensor(mul_81, primals_100)
        add_51 = torch.ops.aten.add.Tensor(mul_82, primals_101);  mul_82 = primals_101 = None
        view_132 = torch.ops.aten.view.default(add_51, [15, 768])
        permute_66 = torch.ops.aten.permute.default(primals_102, [1, 0]);  primals_102 = None
        addmm_36 = torch.ops.aten.addmm.default(primals_103, view_132, permute_66);  primals_103 = None
        view_133 = torch.ops.aten.view.default(addmm_36, [3, 5, 768]);  addmm_36 = None
        permute_67 = torch.ops.aten.permute.default(primals_104, [1, 0]);  primals_104 = None
        addmm_37 = torch.ops.aten.addmm.default(primals_105, view_132, permute_67);  primals_105 = None
        view_135 = torch.ops.aten.view.default(addmm_37, [3, 5, 768]);  addmm_37 = None
        view_136 = torch.ops.aten.view.default(view_135, [3, 5, 12, 64]);  view_135 = None
        permute_68 = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
        permute_69 = torch.ops.aten.permute.default(primals_106, [1, 0]);  primals_106 = None
        addmm_38 = torch.ops.aten.addmm.default(primals_107, view_132, permute_69);  primals_107 = None
        view_138 = torch.ops.aten.view.default(addmm_38, [3, 5, 768]);  addmm_38 = None
        view_139 = torch.ops.aten.view.default(view_138, [3, 5, 12, 64]);  view_138 = None
        permute_70 = torch.ops.aten.permute.default(view_139, [0, 2, 1, 3]);  view_139 = None
        view_140 = torch.ops.aten.view.default(view_133, [3, 5, 12, 64]);  view_133 = None
        permute_71 = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None
        permute_72 = torch.ops.aten.permute.default(permute_68, [0, 1, 3, 2]);  permute_68 = None
        expand_25 = torch.ops.aten.expand.default(permute_71, [3, 12, 5, 64]);  permute_71 = None
        clone_24 = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_141 = torch.ops.aten.view.default(clone_24, [36, 5, 64]);  clone_24 = None
        expand_26 = torch.ops.aten.expand.default(permute_72, [3, 12, 64, 5]);  permute_72 = None
        clone_25 = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
        view_142 = torch.ops.aten.view.default(clone_25, [36, 64, 5]);  clone_25 = None
        bmm_12 = torch.ops.aten.bmm.default(view_141, view_142)
        view_143 = torch.ops.aten.view.default(bmm_12, [3, 12, 5, 5]);  bmm_12 = None
        div_12 = torch.ops.aten.div.Tensor(view_143, 8.0);  view_143 = None
        add_52 = torch.ops.aten.add.Tensor(div_12, mul);  div_12 = None
        amax_6 = torch.ops.aten.amax.default(add_52, [-1], True)
        sub_20 = torch.ops.aten.sub.Tensor(add_52, amax_6);  add_52 = amax_6 = None
        exp_6 = torch.ops.aten.exp.default(sub_20);  sub_20 = None
        sum_7 = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_13 = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        alias_6 = torch.ops.aten.alias.default(div_13)
        inductor_lookup_seed_default_19 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 19)
        inductor_random_default_17 = torch.ops.prims.inductor_random.default([3, 12, 5, 5], inductor_lookup_seed_default_19, 'rand');  inductor_lookup_seed_default_19 = None
        gt_19 = torch.ops.aten.gt.Scalar(inductor_random_default_17, 0.1);  inductor_random_default_17 = None
        mul_83 = torch.ops.aten.mul.Tensor(gt_19, div_13);  div_13 = None
        mul_84 = torch.ops.aten.mul.Tensor(mul_83, 1.1111111111111112);  mul_83 = None
        expand_27 = torch.ops.aten.expand.default(mul_84, [3, 12, 5, 5]);  mul_84 = None
        view_144 = torch.ops.aten.view.default(expand_27, [36, 5, 5]);  expand_27 = None
        expand_28 = torch.ops.aten.expand.default(permute_70, [3, 12, 5, 64]);  permute_70 = None
        clone_26 = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_145 = torch.ops.aten.view.default(clone_26, [36, 5, 64]);  clone_26 = None
        bmm_13 = torch.ops.aten.bmm.default(view_144, view_145)
        view_146 = torch.ops.aten.view.default(bmm_13, [3, 12, 5, 64]);  bmm_13 = None
        permute_73 = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
        clone_27 = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
        view_147 = torch.ops.aten.view.default(clone_27, [3, 5, 768]);  clone_27 = None
        view_148 = torch.ops.aten.view.default(view_147, [15, 768]);  view_147 = None
        permute_74 = torch.ops.aten.permute.default(primals_108, [1, 0]);  primals_108 = None
        addmm_39 = torch.ops.aten.addmm.default(primals_109, view_148, permute_74);  primals_109 = None
        view_149 = torch.ops.aten.view.default(addmm_39, [3, 5, 768]);  addmm_39 = None
        inductor_lookup_seed_default_20 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 20)
        inductor_random_default_16 = torch.ops.prims.inductor_random.default([3, 5, 768], inductor_lookup_seed_default_20, 'rand');  inductor_lookup_seed_default_20 = None
        gt_20 = torch.ops.aten.gt.Scalar(inductor_random_default_16, 0.1);  inductor_random_default_16 = None
        mul_85 = torch.ops.aten.mul.Tensor(gt_20, view_149);  view_149 = None
        mul_86 = torch.ops.aten.mul.Tensor(mul_85, 1.1111111111111112);  mul_85 = None
        add_53 = torch.ops.aten.add.Tensor(mul_86, add_51);  mul_86 = add_51 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_53, [2], correction = 0, keepdim = True)
        getitem_26 = var_mean_13[0]
        getitem_27 = var_mean_13[1];  var_mean_13 = None
        add_54 = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        sub_21 = torch.ops.aten.sub.Tensor(add_53, getitem_27);  add_53 = getitem_27 = None
        mul_87 = torch.ops.aten.mul.Tensor(sub_21, rsqrt_13);  sub_21 = None
        mul_88 = torch.ops.aten.mul.Tensor(mul_87, primals_110)
        add_55 = torch.ops.aten.add.Tensor(mul_88, primals_111);  mul_88 = primals_111 = None
        view_150 = torch.ops.aten.view.default(add_55, [15, 768])
        permute_75 = torch.ops.aten.permute.default(primals_112, [1, 0]);  primals_112 = None
        addmm_40 = torch.ops.aten.addmm.default(primals_113, view_150, permute_75);  primals_113 = None
        view_151 = torch.ops.aten.view.default(addmm_40, [3, 5, 3072])
        mul_89 = torch.ops.aten.mul.Tensor(view_151, 0.5)
        mul_90 = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476);  view_151 = None
        erf_6 = torch.ops.aten.erf.default(mul_90);  mul_90 = None
        add_56 = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_91 = torch.ops.aten.mul.Tensor(mul_89, add_56);  mul_89 = add_56 = None
        view_152 = torch.ops.aten.view.default(mul_91, [15, 3072]);  mul_91 = None
        permute_76 = torch.ops.aten.permute.default(primals_114, [1, 0]);  primals_114 = None
        addmm_41 = torch.ops.aten.addmm.default(primals_115, view_152, permute_76);  primals_115 = None
        view_153 = torch.ops.aten.view.default(addmm_41, [3, 5, 768]);  addmm_41 = None
        inductor_lookup_seed_default_21 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 21)
        inductor_random_default_15 = torch.ops.prims.inductor_random.default([3, 5, 768], inductor_lookup_seed_default_21, 'rand');  inductor_lookup_seed_default_21 = None
        gt_21 = torch.ops.aten.gt.Scalar(inductor_random_default_15, 0.1);  inductor_random_default_15 = None
        mul_92 = torch.ops.aten.mul.Tensor(gt_21, view_153);  view_153 = None
        mul_93 = torch.ops.aten.mul.Tensor(mul_92, 1.1111111111111112);  mul_92 = None
        add_57 = torch.ops.aten.add.Tensor(mul_93, add_55);  mul_93 = add_55 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
        getitem_28 = var_mean_14[0]
        getitem_29 = var_mean_14[1];  var_mean_14 = None
        add_58 = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
        rsqrt_14 = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        sub_22 = torch.ops.aten.sub.Tensor(add_57, getitem_29);  add_57 = getitem_29 = None
        mul_94 = torch.ops.aten.mul.Tensor(sub_22, rsqrt_14);  sub_22 = None
        mul_95 = torch.ops.aten.mul.Tensor(mul_94, primals_116)
        add_59 = torch.ops.aten.add.Tensor(mul_95, primals_117);  mul_95 = primals_117 = None
        view_154 = torch.ops.aten.view.default(add_59, [15, 768])
        permute_77 = torch.ops.aten.permute.default(primals_118, [1, 0]);  primals_118 = None
        addmm_42 = torch.ops.aten.addmm.default(primals_119, view_154, permute_77);  primals_119 = None
        view_155 = torch.ops.aten.view.default(addmm_42, [3, 5, 768]);  addmm_42 = None
        permute_78 = torch.ops.aten.permute.default(primals_120, [1, 0]);  primals_120 = None
        addmm_43 = torch.ops.aten.addmm.default(primals_121, view_154, permute_78);  primals_121 = None
        view_157 = torch.ops.aten.view.default(addmm_43, [3, 5, 768]);  addmm_43 = None
        view_158 = torch.ops.aten.view.default(view_157, [3, 5, 12, 64]);  view_157 = None
        permute_79 = torch.ops.aten.permute.default(view_158, [0, 2, 1, 3]);  view_158 = None
        permute_80 = torch.ops.aten.permute.default(primals_122, [1, 0]);  primals_122 = None
        addmm_44 = torch.ops.aten.addmm.default(primals_123, view_154, permute_80);  primals_123 = None
        view_160 = torch.ops.aten.view.default(addmm_44, [3, 5, 768]);  addmm_44 = None
        view_161 = torch.ops.aten.view.default(view_160, [3, 5, 12, 64]);  view_160 = None
        permute_81 = torch.ops.aten.permute.default(view_161, [0, 2, 1, 3]);  view_161 = None
        view_162 = torch.ops.aten.view.default(view_155, [3, 5, 12, 64]);  view_155 = None
        permute_82 = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
        permute_83 = torch.ops.aten.permute.default(permute_79, [0, 1, 3, 2]);  permute_79 = None
        expand_29 = torch.ops.aten.expand.default(permute_82, [3, 12, 5, 64]);  permute_82 = None
        clone_28 = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        view_163 = torch.ops.aten.view.default(clone_28, [36, 5, 64]);  clone_28 = None
        expand_30 = torch.ops.aten.expand.default(permute_83, [3, 12, 64, 5]);  permute_83 = None
        clone_29 = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
        view_164 = torch.ops.aten.view.default(clone_29, [36, 64, 5]);  clone_29 = None
        bmm_14 = torch.ops.aten.bmm.default(view_163, view_164)
        view_165 = torch.ops.aten.view.default(bmm_14, [3, 12, 5, 5]);  bmm_14 = None
        div_14 = torch.ops.aten.div.Tensor(view_165, 8.0);  view_165 = None
        add_60 = torch.ops.aten.add.Tensor(div_14, mul);  div_14 = None
        amax_7 = torch.ops.aten.amax.default(add_60, [-1], True)
        sub_23 = torch.ops.aten.sub.Tensor(add_60, amax_7);  add_60 = amax_7 = None
        exp_7 = torch.ops.aten.exp.default(sub_23);  sub_23 = None
        sum_8 = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_15 = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        alias_7 = torch.ops.aten.alias.default(div_15)
        inductor_lookup_seed_default_22 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 22)
        inductor_random_default_14 = torch.ops.prims.inductor_random.default([3, 12, 5, 5], inductor_lookup_seed_default_22, 'rand');  inductor_lookup_seed_default_22 = None
        gt_22 = torch.ops.aten.gt.Scalar(inductor_random_default_14, 0.1);  inductor_random_default_14 = None
        mul_96 = torch.ops.aten.mul.Tensor(gt_22, div_15);  div_15 = None
        mul_97 = torch.ops.aten.mul.Tensor(mul_96, 1.1111111111111112);  mul_96 = None
        expand_31 = torch.ops.aten.expand.default(mul_97, [3, 12, 5, 5]);  mul_97 = None
        view_166 = torch.ops.aten.view.default(expand_31, [36, 5, 5]);  expand_31 = None
        expand_32 = torch.ops.aten.expand.default(permute_81, [3, 12, 5, 64]);  permute_81 = None
        clone_30 = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        view_167 = torch.ops.aten.view.default(clone_30, [36, 5, 64]);  clone_30 = None
        bmm_15 = torch.ops.aten.bmm.default(view_166, view_167)
        view_168 = torch.ops.aten.view.default(bmm_15, [3, 12, 5, 64]);  bmm_15 = None
        permute_84 = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
        clone_31 = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
        view_169 = torch.ops.aten.view.default(clone_31, [3, 5, 768]);  clone_31 = None
        view_170 = torch.ops.aten.view.default(view_169, [15, 768]);  view_169 = None
        permute_85 = torch.ops.aten.permute.default(primals_124, [1, 0]);  primals_124 = None
        addmm_45 = torch.ops.aten.addmm.default(primals_125, view_170, permute_85);  primals_125 = None
        view_171 = torch.ops.aten.view.default(addmm_45, [3, 5, 768]);  addmm_45 = None
        inductor_lookup_seed_default_23 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 23)
        inductor_random_default_13 = torch.ops.prims.inductor_random.default([3, 5, 768], inductor_lookup_seed_default_23, 'rand');  inductor_lookup_seed_default_23 = None
        gt_23 = torch.ops.aten.gt.Scalar(inductor_random_default_13, 0.1);  inductor_random_default_13 = None
        mul_98 = torch.ops.aten.mul.Tensor(gt_23, view_171);  view_171 = None
        mul_99 = torch.ops.aten.mul.Tensor(mul_98, 1.1111111111111112);  mul_98 = None
        add_61 = torch.ops.aten.add.Tensor(mul_99, add_59);  mul_99 = add_59 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_61, [2], correction = 0, keepdim = True)
        getitem_30 = var_mean_15[0]
        getitem_31 = var_mean_15[1];  var_mean_15 = None
        add_62 = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_15 = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        sub_24 = torch.ops.aten.sub.Tensor(add_61, getitem_31);  add_61 = getitem_31 = None
        mul_100 = torch.ops.aten.mul.Tensor(sub_24, rsqrt_15);  sub_24 = None
        mul_101 = torch.ops.aten.mul.Tensor(mul_100, primals_126)
        add_63 = torch.ops.aten.add.Tensor(mul_101, primals_127);  mul_101 = primals_127 = None
        view_172 = torch.ops.aten.view.default(add_63, [15, 768])
        permute_86 = torch.ops.aten.permute.default(primals_128, [1, 0]);  primals_128 = None
        addmm_46 = torch.ops.aten.addmm.default(primals_129, view_172, permute_86);  primals_129 = None
        view_173 = torch.ops.aten.view.default(addmm_46, [3, 5, 3072])
        mul_102 = torch.ops.aten.mul.Tensor(view_173, 0.5)
        mul_103 = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476);  view_173 = None
        erf_7 = torch.ops.aten.erf.default(mul_103);  mul_103 = None
        add_64 = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_104 = torch.ops.aten.mul.Tensor(mul_102, add_64);  mul_102 = add_64 = None
        view_174 = torch.ops.aten.view.default(mul_104, [15, 3072]);  mul_104 = None
        permute_87 = torch.ops.aten.permute.default(primals_130, [1, 0]);  primals_130 = None
        addmm_47 = torch.ops.aten.addmm.default(primals_131, view_174, permute_87);  primals_131 = None
        view_175 = torch.ops.aten.view.default(addmm_47, [3, 5, 768]);  addmm_47 = None
        inductor_lookup_seed_default_24 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 24)
        inductor_random_default_12 = torch.ops.prims.inductor_random.default([3, 5, 768], inductor_lookup_seed_default_24, 'rand');  inductor_lookup_seed_default_24 = None
        gt_24 = torch.ops.aten.gt.Scalar(inductor_random_default_12, 0.1);  inductor_random_default_12 = None
        mul_105 = torch.ops.aten.mul.Tensor(gt_24, view_175);  view_175 = None
        mul_106 = torch.ops.aten.mul.Tensor(mul_105, 1.1111111111111112);  mul_105 = None
        add_65 = torch.ops.aten.add.Tensor(mul_106, add_63);  mul_106 = add_63 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_65, [2], correction = 0, keepdim = True)
        getitem_32 = var_mean_16[0]
        getitem_33 = var_mean_16[1];  var_mean_16 = None
        add_66 = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_16 = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        sub_25 = torch.ops.aten.sub.Tensor(add_65, getitem_33);  add_65 = getitem_33 = None
        mul_107 = torch.ops.aten.mul.Tensor(sub_25, rsqrt_16);  sub_25 = None
        mul_108 = torch.ops.aten.mul.Tensor(mul_107, primals_132)
        add_67 = torch.ops.aten.add.Tensor(mul_108, primals_133);  mul_108 = primals_133 = None
        view_176 = torch.ops.aten.view.default(add_67, [15, 768])
        permute_88 = torch.ops.aten.permute.default(primals_134, [1, 0]);  primals_134 = None
        addmm_48 = torch.ops.aten.addmm.default(primals_135, view_176, permute_88);  primals_135 = None
        view_177 = torch.ops.aten.view.default(addmm_48, [3, 5, 768]);  addmm_48 = None
        permute_89 = torch.ops.aten.permute.default(primals_136, [1, 0]);  primals_136 = None
        addmm_49 = torch.ops.aten.addmm.default(primals_137, view_176, permute_89);  primals_137 = None
        view_179 = torch.ops.aten.view.default(addmm_49, [3, 5, 768]);  addmm_49 = None
        view_180 = torch.ops.aten.view.default(view_179, [3, 5, 12, 64]);  view_179 = None
        permute_90 = torch.ops.aten.permute.default(view_180, [0, 2, 1, 3]);  view_180 = None
        permute_91 = torch.ops.aten.permute.default(primals_138, [1, 0]);  primals_138 = None
        addmm_50 = torch.ops.aten.addmm.default(primals_139, view_176, permute_91);  primals_139 = None
        view_182 = torch.ops.aten.view.default(addmm_50, [3, 5, 768]);  addmm_50 = None
        view_183 = torch.ops.aten.view.default(view_182, [3, 5, 12, 64]);  view_182 = None
        permute_92 = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None
        view_184 = torch.ops.aten.view.default(view_177, [3, 5, 12, 64]);  view_177 = None
        permute_93 = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
        permute_94 = torch.ops.aten.permute.default(permute_90, [0, 1, 3, 2]);  permute_90 = None
        expand_33 = torch.ops.aten.expand.default(permute_93, [3, 12, 5, 64]);  permute_93 = None
        clone_32 = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_185 = torch.ops.aten.view.default(clone_32, [36, 5, 64]);  clone_32 = None
        expand_34 = torch.ops.aten.expand.default(permute_94, [3, 12, 64, 5]);  permute_94 = None
        clone_33 = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
        view_186 = torch.ops.aten.view.default(clone_33, [36, 64, 5]);  clone_33 = None
        bmm_16 = torch.ops.aten.bmm.default(view_185, view_186)
        view_187 = torch.ops.aten.view.default(bmm_16, [3, 12, 5, 5]);  bmm_16 = None
        div_16 = torch.ops.aten.div.Tensor(view_187, 8.0);  view_187 = None
        add_68 = torch.ops.aten.add.Tensor(div_16, mul);  div_16 = None
        amax_8 = torch.ops.aten.amax.default(add_68, [-1], True)
        sub_26 = torch.ops.aten.sub.Tensor(add_68, amax_8);  add_68 = amax_8 = None
        exp_8 = torch.ops.aten.exp.default(sub_26);  sub_26 = None
        sum_9 = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_17 = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        alias_8 = torch.ops.aten.alias.default(div_17)
        inductor_lookup_seed_default_25 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 25)
        inductor_random_default_11 = torch.ops.prims.inductor_random.default([3, 12, 5, 5], inductor_lookup_seed_default_25, 'rand');  inductor_lookup_seed_default_25 = None
        gt_25 = torch.ops.aten.gt.Scalar(inductor_random_default_11, 0.1);  inductor_random_default_11 = None
        mul_109 = torch.ops.aten.mul.Tensor(gt_25, div_17);  div_17 = None
        mul_110 = torch.ops.aten.mul.Tensor(mul_109, 1.1111111111111112);  mul_109 = None
        expand_35 = torch.ops.aten.expand.default(mul_110, [3, 12, 5, 5]);  mul_110 = None
        view_188 = torch.ops.aten.view.default(expand_35, [36, 5, 5]);  expand_35 = None
        expand_36 = torch.ops.aten.expand.default(permute_92, [3, 12, 5, 64]);  permute_92 = None
        clone_34 = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_189 = torch.ops.aten.view.default(clone_34, [36, 5, 64]);  clone_34 = None
        bmm_17 = torch.ops.aten.bmm.default(view_188, view_189)
        view_190 = torch.ops.aten.view.default(bmm_17, [3, 12, 5, 64]);  bmm_17 = None
        permute_95 = torch.ops.aten.permute.default(view_190, [0, 2, 1, 3]);  view_190 = None
        clone_35 = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        view_191 = torch.ops.aten.view.default(clone_35, [3, 5, 768]);  clone_35 = None
        view_192 = torch.ops.aten.view.default(view_191, [15, 768]);  view_191 = None
        permute_96 = torch.ops.aten.permute.default(primals_140, [1, 0]);  primals_140 = None
        addmm_51 = torch.ops.aten.addmm.default(primals_141, view_192, permute_96);  primals_141 = None
        view_193 = torch.ops.aten.view.default(addmm_51, [3, 5, 768]);  addmm_51 = None
        inductor_lookup_seed_default_26 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 26)
        inductor_random_default_10 = torch.ops.prims.inductor_random.default([3, 5, 768], inductor_lookup_seed_default_26, 'rand');  inductor_lookup_seed_default_26 = None
        gt_26 = torch.ops.aten.gt.Scalar(inductor_random_default_10, 0.1);  inductor_random_default_10 = None
        mul_111 = torch.ops.aten.mul.Tensor(gt_26, view_193);  view_193 = None
        mul_112 = torch.ops.aten.mul.Tensor(mul_111, 1.1111111111111112);  mul_111 = None
        add_69 = torch.ops.aten.add.Tensor(mul_112, add_67);  mul_112 = add_67 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_69, [2], correction = 0, keepdim = True)
        getitem_34 = var_mean_17[0]
        getitem_35 = var_mean_17[1];  var_mean_17 = None
        add_70 = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
        rsqrt_17 = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
        sub_27 = torch.ops.aten.sub.Tensor(add_69, getitem_35);  add_69 = getitem_35 = None
        mul_113 = torch.ops.aten.mul.Tensor(sub_27, rsqrt_17);  sub_27 = None
        mul_114 = torch.ops.aten.mul.Tensor(mul_113, primals_142)
        add_71 = torch.ops.aten.add.Tensor(mul_114, primals_143);  mul_114 = primals_143 = None
        view_194 = torch.ops.aten.view.default(add_71, [15, 768])
        permute_97 = torch.ops.aten.permute.default(primals_144, [1, 0]);  primals_144 = None
        addmm_52 = torch.ops.aten.addmm.default(primals_145, view_194, permute_97);  primals_145 = None
        view_195 = torch.ops.aten.view.default(addmm_52, [3, 5, 3072])
        mul_115 = torch.ops.aten.mul.Tensor(view_195, 0.5)
        mul_116 = torch.ops.aten.mul.Tensor(view_195, 0.7071067811865476);  view_195 = None
        erf_8 = torch.ops.aten.erf.default(mul_116);  mul_116 = None
        add_72 = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_117 = torch.ops.aten.mul.Tensor(mul_115, add_72);  mul_115 = add_72 = None
        view_196 = torch.ops.aten.view.default(mul_117, [15, 3072]);  mul_117 = None
        permute_98 = torch.ops.aten.permute.default(primals_146, [1, 0]);  primals_146 = None
        addmm_53 = torch.ops.aten.addmm.default(primals_147, view_196, permute_98);  primals_147 = None
        view_197 = torch.ops.aten.view.default(addmm_53, [3, 5, 768]);  addmm_53 = None
        inductor_lookup_seed_default_27 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 27)
        inductor_random_default_9 = torch.ops.prims.inductor_random.default([3, 5, 768], inductor_lookup_seed_default_27, 'rand');  inductor_lookup_seed_default_27 = None
        gt_27 = torch.ops.aten.gt.Scalar(inductor_random_default_9, 0.1);  inductor_random_default_9 = None
        mul_118 = torch.ops.aten.mul.Tensor(gt_27, view_197);  view_197 = None
        mul_119 = torch.ops.aten.mul.Tensor(mul_118, 1.1111111111111112);  mul_118 = None
        add_73 = torch.ops.aten.add.Tensor(mul_119, add_71);  mul_119 = add_71 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
        getitem_36 = var_mean_18[0]
        getitem_37 = var_mean_18[1];  var_mean_18 = None
        add_74 = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
        rsqrt_18 = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        sub_28 = torch.ops.aten.sub.Tensor(add_73, getitem_37);  add_73 = getitem_37 = None
        mul_120 = torch.ops.aten.mul.Tensor(sub_28, rsqrt_18);  sub_28 = None
        mul_121 = torch.ops.aten.mul.Tensor(mul_120, primals_148)
        add_75 = torch.ops.aten.add.Tensor(mul_121, primals_149);  mul_121 = primals_149 = None
        view_198 = torch.ops.aten.view.default(add_75, [15, 768])
        permute_99 = torch.ops.aten.permute.default(primals_150, [1, 0]);  primals_150 = None
        addmm_54 = torch.ops.aten.addmm.default(primals_151, view_198, permute_99);  primals_151 = None
        view_199 = torch.ops.aten.view.default(addmm_54, [3, 5, 768]);  addmm_54 = None
        permute_100 = torch.ops.aten.permute.default(primals_152, [1, 0]);  primals_152 = None
        addmm_55 = torch.ops.aten.addmm.default(primals_153, view_198, permute_100);  primals_153 = None
        view_201 = torch.ops.aten.view.default(addmm_55, [3, 5, 768]);  addmm_55 = None
        view_202 = torch.ops.aten.view.default(view_201, [3, 5, 12, 64]);  view_201 = None
        permute_101 = torch.ops.aten.permute.default(view_202, [0, 2, 1, 3]);  view_202 = None
        permute_102 = torch.ops.aten.permute.default(primals_154, [1, 0]);  primals_154 = None
        addmm_56 = torch.ops.aten.addmm.default(primals_155, view_198, permute_102);  primals_155 = None
        view_204 = torch.ops.aten.view.default(addmm_56, [3, 5, 768]);  addmm_56 = None
        view_205 = torch.ops.aten.view.default(view_204, [3, 5, 12, 64]);  view_204 = None
        permute_103 = torch.ops.aten.permute.default(view_205, [0, 2, 1, 3]);  view_205 = None
        view_206 = torch.ops.aten.view.default(view_199, [3, 5, 12, 64]);  view_199 = None
        permute_104 = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
        permute_105 = torch.ops.aten.permute.default(permute_101, [0, 1, 3, 2]);  permute_101 = None
        expand_37 = torch.ops.aten.expand.default(permute_104, [3, 12, 5, 64]);  permute_104 = None
        clone_36 = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        view_207 = torch.ops.aten.view.default(clone_36, [36, 5, 64]);  clone_36 = None
        expand_38 = torch.ops.aten.expand.default(permute_105, [3, 12, 64, 5]);  permute_105 = None
        clone_37 = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
        view_208 = torch.ops.aten.view.default(clone_37, [36, 64, 5]);  clone_37 = None
        bmm_18 = torch.ops.aten.bmm.default(view_207, view_208)
        view_209 = torch.ops.aten.view.default(bmm_18, [3, 12, 5, 5]);  bmm_18 = None
        div_18 = torch.ops.aten.div.Tensor(view_209, 8.0);  view_209 = None
        add_76 = torch.ops.aten.add.Tensor(div_18, mul);  div_18 = None
        amax_9 = torch.ops.aten.amax.default(add_76, [-1], True)
        sub_29 = torch.ops.aten.sub.Tensor(add_76, amax_9);  add_76 = amax_9 = None
        exp_9 = torch.ops.aten.exp.default(sub_29);  sub_29 = None
        sum_10 = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_19 = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        alias_9 = torch.ops.aten.alias.default(div_19)
        inductor_lookup_seed_default_28 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 28)
        inductor_random_default_8 = torch.ops.prims.inductor_random.default([3, 12, 5, 5], inductor_lookup_seed_default_28, 'rand');  inductor_lookup_seed_default_28 = None
        gt_28 = torch.ops.aten.gt.Scalar(inductor_random_default_8, 0.1);  inductor_random_default_8 = None
        mul_122 = torch.ops.aten.mul.Tensor(gt_28, div_19);  div_19 = None
        mul_123 = torch.ops.aten.mul.Tensor(mul_122, 1.1111111111111112);  mul_122 = None
        expand_39 = torch.ops.aten.expand.default(mul_123, [3, 12, 5, 5]);  mul_123 = None
        view_210 = torch.ops.aten.view.default(expand_39, [36, 5, 5]);  expand_39 = None
        expand_40 = torch.ops.aten.expand.default(permute_103, [3, 12, 5, 64]);  permute_103 = None
        clone_38 = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        view_211 = torch.ops.aten.view.default(clone_38, [36, 5, 64]);  clone_38 = None
        bmm_19 = torch.ops.aten.bmm.default(view_210, view_211)
        view_212 = torch.ops.aten.view.default(bmm_19, [3, 12, 5, 64]);  bmm_19 = None
        permute_106 = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None
        clone_39 = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
        view_213 = torch.ops.aten.view.default(clone_39, [3, 5, 768]);  clone_39 = None
        view_214 = torch.ops.aten.view.default(view_213, [15, 768]);  view_213 = None
        permute_107 = torch.ops.aten.permute.default(primals_156, [1, 0]);  primals_156 = None
        addmm_57 = torch.ops.aten.addmm.default(primals_157, view_214, permute_107);  primals_157 = None
        view_215 = torch.ops.aten.view.default(addmm_57, [3, 5, 768]);  addmm_57 = None
        inductor_lookup_seed_default_29 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 29)
        inductor_random_default_7 = torch.ops.prims.inductor_random.default([3, 5, 768], inductor_lookup_seed_default_29, 'rand');  inductor_lookup_seed_default_29 = None
        gt_29 = torch.ops.aten.gt.Scalar(inductor_random_default_7, 0.1);  inductor_random_default_7 = None
        mul_124 = torch.ops.aten.mul.Tensor(gt_29, view_215);  view_215 = None
        mul_125 = torch.ops.aten.mul.Tensor(mul_124, 1.1111111111111112);  mul_124 = None
        add_77 = torch.ops.aten.add.Tensor(mul_125, add_75);  mul_125 = add_75 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_77, [2], correction = 0, keepdim = True)
        getitem_38 = var_mean_19[0]
        getitem_39 = var_mean_19[1];  var_mean_19 = None
        add_78 = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_19 = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        sub_30 = torch.ops.aten.sub.Tensor(add_77, getitem_39);  add_77 = getitem_39 = None
        mul_126 = torch.ops.aten.mul.Tensor(sub_30, rsqrt_19);  sub_30 = None
        mul_127 = torch.ops.aten.mul.Tensor(mul_126, primals_158)
        add_79 = torch.ops.aten.add.Tensor(mul_127, primals_159);  mul_127 = primals_159 = None
        view_216 = torch.ops.aten.view.default(add_79, [15, 768])
        permute_108 = torch.ops.aten.permute.default(primals_160, [1, 0]);  primals_160 = None
        addmm_58 = torch.ops.aten.addmm.default(primals_161, view_216, permute_108);  primals_161 = None
        view_217 = torch.ops.aten.view.default(addmm_58, [3, 5, 3072])
        mul_128 = torch.ops.aten.mul.Tensor(view_217, 0.5)
        mul_129 = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476);  view_217 = None
        erf_9 = torch.ops.aten.erf.default(mul_129);  mul_129 = None
        add_80 = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_130 = torch.ops.aten.mul.Tensor(mul_128, add_80);  mul_128 = add_80 = None
        view_218 = torch.ops.aten.view.default(mul_130, [15, 3072]);  mul_130 = None
        permute_109 = torch.ops.aten.permute.default(primals_162, [1, 0]);  primals_162 = None
        addmm_59 = torch.ops.aten.addmm.default(primals_163, view_218, permute_109);  primals_163 = None
        view_219 = torch.ops.aten.view.default(addmm_59, [3, 5, 768]);  addmm_59 = None
        inductor_lookup_seed_default_30 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 30)
        inductor_random_default_6 = torch.ops.prims.inductor_random.default([3, 5, 768], inductor_lookup_seed_default_30, 'rand');  inductor_lookup_seed_default_30 = None
        gt_30 = torch.ops.aten.gt.Scalar(inductor_random_default_6, 0.1);  inductor_random_default_6 = None
        mul_131 = torch.ops.aten.mul.Tensor(gt_30, view_219);  view_219 = None
        mul_132 = torch.ops.aten.mul.Tensor(mul_131, 1.1111111111111112);  mul_131 = None
        add_81 = torch.ops.aten.add.Tensor(mul_132, add_79);  mul_132 = add_79 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
        getitem_40 = var_mean_20[0]
        getitem_41 = var_mean_20[1];  var_mean_20 = None
        add_82 = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_20 = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        sub_31 = torch.ops.aten.sub.Tensor(add_81, getitem_41);  add_81 = getitem_41 = None
        mul_133 = torch.ops.aten.mul.Tensor(sub_31, rsqrt_20);  sub_31 = None
        mul_134 = torch.ops.aten.mul.Tensor(mul_133, primals_164)
        add_83 = torch.ops.aten.add.Tensor(mul_134, primals_165);  mul_134 = primals_165 = None
        view_220 = torch.ops.aten.view.default(add_83, [15, 768])
        permute_110 = torch.ops.aten.permute.default(primals_166, [1, 0]);  primals_166 = None
        addmm_60 = torch.ops.aten.addmm.default(primals_167, view_220, permute_110);  primals_167 = None
        view_221 = torch.ops.aten.view.default(addmm_60, [3, 5, 768]);  addmm_60 = None
        permute_111 = torch.ops.aten.permute.default(primals_168, [1, 0]);  primals_168 = None
        addmm_61 = torch.ops.aten.addmm.default(primals_169, view_220, permute_111);  primals_169 = None
        view_223 = torch.ops.aten.view.default(addmm_61, [3, 5, 768]);  addmm_61 = None
        view_224 = torch.ops.aten.view.default(view_223, [3, 5, 12, 64]);  view_223 = None
        permute_112 = torch.ops.aten.permute.default(view_224, [0, 2, 1, 3]);  view_224 = None
        permute_113 = torch.ops.aten.permute.default(primals_170, [1, 0]);  primals_170 = None
        addmm_62 = torch.ops.aten.addmm.default(primals_171, view_220, permute_113);  primals_171 = None
        view_226 = torch.ops.aten.view.default(addmm_62, [3, 5, 768]);  addmm_62 = None
        view_227 = torch.ops.aten.view.default(view_226, [3, 5, 12, 64]);  view_226 = None
        permute_114 = torch.ops.aten.permute.default(view_227, [0, 2, 1, 3]);  view_227 = None
        view_228 = torch.ops.aten.view.default(view_221, [3, 5, 12, 64]);  view_221 = None
        permute_115 = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None
        permute_116 = torch.ops.aten.permute.default(permute_112, [0, 1, 3, 2]);  permute_112 = None
        expand_41 = torch.ops.aten.expand.default(permute_115, [3, 12, 5, 64]);  permute_115 = None
        clone_40 = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        view_229 = torch.ops.aten.view.default(clone_40, [36, 5, 64]);  clone_40 = None
        expand_42 = torch.ops.aten.expand.default(permute_116, [3, 12, 64, 5]);  permute_116 = None
        clone_41 = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        view_230 = torch.ops.aten.view.default(clone_41, [36, 64, 5]);  clone_41 = None
        bmm_20 = torch.ops.aten.bmm.default(view_229, view_230)
        view_231 = torch.ops.aten.view.default(bmm_20, [3, 12, 5, 5]);  bmm_20 = None
        div_20 = torch.ops.aten.div.Tensor(view_231, 8.0);  view_231 = None
        add_84 = torch.ops.aten.add.Tensor(div_20, mul);  div_20 = None
        amax_10 = torch.ops.aten.amax.default(add_84, [-1], True)
        sub_32 = torch.ops.aten.sub.Tensor(add_84, amax_10);  add_84 = amax_10 = None
        exp_10 = torch.ops.aten.exp.default(sub_32);  sub_32 = None
        sum_11 = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_21 = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        alias_10 = torch.ops.aten.alias.default(div_21)
        inductor_lookup_seed_default_31 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 31)
        inductor_random_default_5 = torch.ops.prims.inductor_random.default([3, 12, 5, 5], inductor_lookup_seed_default_31, 'rand');  inductor_lookup_seed_default_31 = None
        gt_31 = torch.ops.aten.gt.Scalar(inductor_random_default_5, 0.1);  inductor_random_default_5 = None
        mul_135 = torch.ops.aten.mul.Tensor(gt_31, div_21);  div_21 = None
        mul_136 = torch.ops.aten.mul.Tensor(mul_135, 1.1111111111111112);  mul_135 = None
        expand_43 = torch.ops.aten.expand.default(mul_136, [3, 12, 5, 5]);  mul_136 = None
        view_232 = torch.ops.aten.view.default(expand_43, [36, 5, 5]);  expand_43 = None
        expand_44 = torch.ops.aten.expand.default(permute_114, [3, 12, 5, 64]);  permute_114 = None
        clone_42 = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_233 = torch.ops.aten.view.default(clone_42, [36, 5, 64]);  clone_42 = None
        bmm_21 = torch.ops.aten.bmm.default(view_232, view_233)
        view_234 = torch.ops.aten.view.default(bmm_21, [3, 12, 5, 64]);  bmm_21 = None
        permute_117 = torch.ops.aten.permute.default(view_234, [0, 2, 1, 3]);  view_234 = None
        clone_43 = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
        view_235 = torch.ops.aten.view.default(clone_43, [3, 5, 768]);  clone_43 = None
        view_236 = torch.ops.aten.view.default(view_235, [15, 768]);  view_235 = None
        permute_118 = torch.ops.aten.permute.default(primals_172, [1, 0]);  primals_172 = None
        addmm_63 = torch.ops.aten.addmm.default(primals_173, view_236, permute_118);  primals_173 = None
        view_237 = torch.ops.aten.view.default(addmm_63, [3, 5, 768]);  addmm_63 = None
        inductor_lookup_seed_default_32 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 32)
        inductor_random_default_4 = torch.ops.prims.inductor_random.default([3, 5, 768], inductor_lookup_seed_default_32, 'rand');  inductor_lookup_seed_default_32 = None
        gt_32 = torch.ops.aten.gt.Scalar(inductor_random_default_4, 0.1);  inductor_random_default_4 = None
        mul_137 = torch.ops.aten.mul.Tensor(gt_32, view_237);  view_237 = None
        mul_138 = torch.ops.aten.mul.Tensor(mul_137, 1.1111111111111112);  mul_137 = None
        add_85 = torch.ops.aten.add.Tensor(mul_138, add_83);  mul_138 = add_83 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_85, [2], correction = 0, keepdim = True)
        getitem_42 = var_mean_21[0]
        getitem_43 = var_mean_21[1];  var_mean_21 = None
        add_86 = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
        rsqrt_21 = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        sub_33 = torch.ops.aten.sub.Tensor(add_85, getitem_43);  add_85 = getitem_43 = None
        mul_139 = torch.ops.aten.mul.Tensor(sub_33, rsqrt_21);  sub_33 = None
        mul_140 = torch.ops.aten.mul.Tensor(mul_139, primals_174)
        add_87 = torch.ops.aten.add.Tensor(mul_140, primals_175);  mul_140 = primals_175 = None
        view_238 = torch.ops.aten.view.default(add_87, [15, 768])
        permute_119 = torch.ops.aten.permute.default(primals_176, [1, 0]);  primals_176 = None
        addmm_64 = torch.ops.aten.addmm.default(primals_177, view_238, permute_119);  primals_177 = None
        view_239 = torch.ops.aten.view.default(addmm_64, [3, 5, 3072])
        mul_141 = torch.ops.aten.mul.Tensor(view_239, 0.5)
        mul_142 = torch.ops.aten.mul.Tensor(view_239, 0.7071067811865476);  view_239 = None
        erf_10 = torch.ops.aten.erf.default(mul_142);  mul_142 = None
        add_88 = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_143 = torch.ops.aten.mul.Tensor(mul_141, add_88);  mul_141 = add_88 = None
        view_240 = torch.ops.aten.view.default(mul_143, [15, 3072]);  mul_143 = None
        permute_120 = torch.ops.aten.permute.default(primals_178, [1, 0]);  primals_178 = None
        addmm_65 = torch.ops.aten.addmm.default(primals_179, view_240, permute_120);  primals_179 = None
        view_241 = torch.ops.aten.view.default(addmm_65, [3, 5, 768]);  addmm_65 = None
        inductor_lookup_seed_default_33 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 33)
        inductor_random_default_3 = torch.ops.prims.inductor_random.default([3, 5, 768], inductor_lookup_seed_default_33, 'rand');  inductor_lookup_seed_default_33 = None
        gt_33 = torch.ops.aten.gt.Scalar(inductor_random_default_3, 0.1);  inductor_random_default_3 = None
        mul_144 = torch.ops.aten.mul.Tensor(gt_33, view_241);  view_241 = None
        mul_145 = torch.ops.aten.mul.Tensor(mul_144, 1.1111111111111112);  mul_144 = None
        add_89 = torch.ops.aten.add.Tensor(mul_145, add_87);  mul_145 = add_87 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_89, [2], correction = 0, keepdim = True)
        getitem_44 = var_mean_22[0]
        getitem_45 = var_mean_22[1];  var_mean_22 = None
        add_90 = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
        rsqrt_22 = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        sub_34 = torch.ops.aten.sub.Tensor(add_89, getitem_45);  add_89 = getitem_45 = None
        mul_146 = torch.ops.aten.mul.Tensor(sub_34, rsqrt_22);  sub_34 = None
        mul_147 = torch.ops.aten.mul.Tensor(mul_146, primals_180)
        add_91 = torch.ops.aten.add.Tensor(mul_147, primals_181);  mul_147 = primals_181 = None
        view_242 = torch.ops.aten.view.default(add_91, [15, 768])
        permute_121 = torch.ops.aten.permute.default(primals_182, [1, 0]);  primals_182 = None
        addmm_66 = torch.ops.aten.addmm.default(primals_183, view_242, permute_121);  primals_183 = None
        view_243 = torch.ops.aten.view.default(addmm_66, [3, 5, 768]);  addmm_66 = None
        permute_122 = torch.ops.aten.permute.default(primals_184, [1, 0]);  primals_184 = None
        addmm_67 = torch.ops.aten.addmm.default(primals_185, view_242, permute_122);  primals_185 = None
        view_245 = torch.ops.aten.view.default(addmm_67, [3, 5, 768]);  addmm_67 = None
        view_246 = torch.ops.aten.view.default(view_245, [3, 5, 12, 64]);  view_245 = None
        permute_123 = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None
        permute_124 = torch.ops.aten.permute.default(primals_186, [1, 0]);  primals_186 = None
        addmm_68 = torch.ops.aten.addmm.default(primals_187, view_242, permute_124);  primals_187 = None
        view_248 = torch.ops.aten.view.default(addmm_68, [3, 5, 768]);  addmm_68 = None
        view_249 = torch.ops.aten.view.default(view_248, [3, 5, 12, 64]);  view_248 = None
        permute_125 = torch.ops.aten.permute.default(view_249, [0, 2, 1, 3]);  view_249 = None
        view_250 = torch.ops.aten.view.default(view_243, [3, 5, 12, 64]);  view_243 = None
        permute_126 = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None
        permute_127 = torch.ops.aten.permute.default(permute_123, [0, 1, 3, 2]);  permute_123 = None
        expand_45 = torch.ops.aten.expand.default(permute_126, [3, 12, 5, 64]);  permute_126 = None
        clone_44 = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
        view_251 = torch.ops.aten.view.default(clone_44, [36, 5, 64]);  clone_44 = None
        expand_46 = torch.ops.aten.expand.default(permute_127, [3, 12, 64, 5]);  permute_127 = None
        clone_45 = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
        view_252 = torch.ops.aten.view.default(clone_45, [36, 64, 5]);  clone_45 = None
        bmm_22 = torch.ops.aten.bmm.default(view_251, view_252)
        view_253 = torch.ops.aten.view.default(bmm_22, [3, 12, 5, 5]);  bmm_22 = None
        div_22 = torch.ops.aten.div.Tensor(view_253, 8.0);  view_253 = None
        add_92 = torch.ops.aten.add.Tensor(div_22, mul);  div_22 = mul = None
        amax_11 = torch.ops.aten.amax.default(add_92, [-1], True)
        sub_35 = torch.ops.aten.sub.Tensor(add_92, amax_11);  add_92 = amax_11 = None
        exp_11 = torch.ops.aten.exp.default(sub_35);  sub_35 = None
        sum_12 = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_23 = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        alias_11 = torch.ops.aten.alias.default(div_23)
        inductor_lookup_seed_default_34 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 34)
        inductor_random_default_2 = torch.ops.prims.inductor_random.default([3, 12, 5, 5], inductor_lookup_seed_default_34, 'rand');  inductor_lookup_seed_default_34 = None
        gt_34 = torch.ops.aten.gt.Scalar(inductor_random_default_2, 0.1);  inductor_random_default_2 = None
        mul_148 = torch.ops.aten.mul.Tensor(gt_34, div_23);  div_23 = None
        mul_149 = torch.ops.aten.mul.Tensor(mul_148, 1.1111111111111112);  mul_148 = None
        expand_47 = torch.ops.aten.expand.default(mul_149, [3, 12, 5, 5]);  mul_149 = None
        view_254 = torch.ops.aten.view.default(expand_47, [36, 5, 5]);  expand_47 = None
        expand_48 = torch.ops.aten.expand.default(permute_125, [3, 12, 5, 64]);  permute_125 = None
        clone_46 = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        view_255 = torch.ops.aten.view.default(clone_46, [36, 5, 64]);  clone_46 = None
        bmm_23 = torch.ops.aten.bmm.default(view_254, view_255)
        view_256 = torch.ops.aten.view.default(bmm_23, [3, 12, 5, 64]);  bmm_23 = None
        permute_128 = torch.ops.aten.permute.default(view_256, [0, 2, 1, 3]);  view_256 = None
        clone_47 = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
        view_257 = torch.ops.aten.view.default(clone_47, [3, 5, 768]);  clone_47 = None
        view_258 = torch.ops.aten.view.default(view_257, [15, 768]);  view_257 = None
        permute_129 = torch.ops.aten.permute.default(primals_188, [1, 0]);  primals_188 = None
        addmm_69 = torch.ops.aten.addmm.default(primals_189, view_258, permute_129);  primals_189 = None
        view_259 = torch.ops.aten.view.default(addmm_69, [3, 5, 768]);  addmm_69 = None
        inductor_lookup_seed_default_35 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 35)
        inductor_random_default_1 = torch.ops.prims.inductor_random.default([3, 5, 768], inductor_lookup_seed_default_35, 'rand');  inductor_lookup_seed_default_35 = None
        gt_35 = torch.ops.aten.gt.Scalar(inductor_random_default_1, 0.1);  inductor_random_default_1 = None
        mul_150 = torch.ops.aten.mul.Tensor(gt_35, view_259);  view_259 = None
        mul_151 = torch.ops.aten.mul.Tensor(mul_150, 1.1111111111111112);  mul_150 = None
        add_93 = torch.ops.aten.add.Tensor(mul_151, add_91);  mul_151 = add_91 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_93, [2], correction = 0, keepdim = True)
        getitem_46 = var_mean_23[0]
        getitem_47 = var_mean_23[1];  var_mean_23 = None
        add_94 = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_23 = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
        sub_36 = torch.ops.aten.sub.Tensor(add_93, getitem_47);  add_93 = getitem_47 = None
        mul_152 = torch.ops.aten.mul.Tensor(sub_36, rsqrt_23);  sub_36 = None
        mul_153 = torch.ops.aten.mul.Tensor(mul_152, primals_190)
        add_95 = torch.ops.aten.add.Tensor(mul_153, primals_191);  mul_153 = primals_191 = None
        view_260 = torch.ops.aten.view.default(add_95, [15, 768])
        permute_130 = torch.ops.aten.permute.default(primals_192, [1, 0]);  primals_192 = None
        addmm_70 = torch.ops.aten.addmm.default(primals_193, view_260, permute_130);  primals_193 = None
        view_261 = torch.ops.aten.view.default(addmm_70, [3, 5, 3072])
        mul_154 = torch.ops.aten.mul.Tensor(view_261, 0.5)
        mul_155 = torch.ops.aten.mul.Tensor(view_261, 0.7071067811865476);  view_261 = None
        erf_11 = torch.ops.aten.erf.default(mul_155);  mul_155 = None
        add_96 = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_156 = torch.ops.aten.mul.Tensor(mul_154, add_96);  mul_154 = add_96 = None
        view_262 = torch.ops.aten.view.default(mul_156, [15, 3072]);  mul_156 = None
        permute_131 = torch.ops.aten.permute.default(primals_194, [1, 0]);  primals_194 = None
        addmm_71 = torch.ops.aten.addmm.default(primals_195, view_262, permute_131);  primals_195 = None
        view_263 = torch.ops.aten.view.default(addmm_71, [3, 5, 768]);  addmm_71 = None
        inductor_lookup_seed_default_36 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 36);  inductor_seeds_default = None
        inductor_random_default = torch.ops.prims.inductor_random.default([3, 5, 768], inductor_lookup_seed_default_36, 'rand');  inductor_lookup_seed_default_36 = None
        gt_36 = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_157 = torch.ops.aten.mul.Tensor(gt_36, view_263);  view_263 = None
        mul_158 = torch.ops.aten.mul.Tensor(mul_157, 1.1111111111111112);  mul_157 = None
        add_97 = torch.ops.aten.add.Tensor(mul_158, add_95);  mul_158 = add_95 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_97, [2], correction = 0, keepdim = True)
        getitem_48 = var_mean_24[0]
        getitem_49 = var_mean_24[1];  var_mean_24 = None
        add_98 = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_24 = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        sub_37 = torch.ops.aten.sub.Tensor(add_97, getitem_49);  add_97 = getitem_49 = None
        mul_159 = torch.ops.aten.mul.Tensor(sub_37, rsqrt_24);  sub_37 = None
        mul_160 = torch.ops.aten.mul.Tensor(mul_159, primals_196)
        add_99 = torch.ops.aten.add.Tensor(mul_160, primals_197);  mul_160 = primals_197 = None
        slice_7 = torch.ops.aten.slice.Tensor(add_99, 0, 0, 9223372036854775807)
        select = torch.ops.aten.select.int(slice_7, 1, 0);  slice_7 = None
        permute_132 = torch.ops.aten.permute.default(primals_198, [1, 0]);  primals_198 = None
        addmm_72 = torch.ops.aten.addmm.default(primals_199, select, permute_132);  primals_199 = None
        tanh = torch.ops.aten.tanh.default(addmm_72);  addmm_72 = None
        permute_133 = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
        div_24 = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None
        permute_137 = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
        permute_141 = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
        div_25 = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
        permute_145 = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
        permute_150 = torch.ops.aten.permute.default(view_254, [0, 2, 1]);  view_254 = None
        permute_151 = torch.ops.aten.permute.default(view_255, [0, 2, 1]);  view_255 = None
        alias_14 = torch.ops.aten.alias.default(alias_11);  alias_11 = None
        permute_152 = torch.ops.aten.permute.default(view_251, [0, 2, 1]);  view_251 = None
        permute_153 = torch.ops.aten.permute.default(view_252, [0, 2, 1]);  view_252 = None
        permute_157 = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None
        permute_162 = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None
        permute_166 = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
        div_27 = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None
        permute_170 = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
        permute_174 = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
        div_28 = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None
        permute_178 = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
        permute_183 = torch.ops.aten.permute.default(view_232, [0, 2, 1]);  view_232 = None
        permute_184 = torch.ops.aten.permute.default(view_233, [0, 2, 1]);  view_233 = None
        alias_15 = torch.ops.aten.alias.default(alias_10);  alias_10 = None
        permute_185 = torch.ops.aten.permute.default(view_229, [0, 2, 1]);  view_229 = None
        permute_186 = torch.ops.aten.permute.default(view_230, [0, 2, 1]);  view_230 = None
        permute_190 = torch.ops.aten.permute.default(permute_113, [1, 0]);  permute_113 = None
        permute_195 = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
        permute_199 = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
        div_30 = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None
        permute_203 = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
        permute_207 = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
        div_31 = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
        permute_211 = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
        permute_216 = torch.ops.aten.permute.default(view_210, [0, 2, 1]);  view_210 = None
        permute_217 = torch.ops.aten.permute.default(view_211, [0, 2, 1]);  view_211 = None
        alias_16 = torch.ops.aten.alias.default(alias_9);  alias_9 = None
        permute_218 = torch.ops.aten.permute.default(view_207, [0, 2, 1]);  view_207 = None
        permute_219 = torch.ops.aten.permute.default(view_208, [0, 2, 1]);  view_208 = None
        permute_223 = torch.ops.aten.permute.default(permute_102, [1, 0]);  permute_102 = None
        permute_228 = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None
        permute_232 = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
        div_33 = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None
        permute_236 = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
        permute_240 = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
        div_34 = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None
        permute_244 = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
        permute_249 = torch.ops.aten.permute.default(view_188, [0, 2, 1]);  view_188 = None
        permute_250 = torch.ops.aten.permute.default(view_189, [0, 2, 1]);  view_189 = None
        alias_17 = torch.ops.aten.alias.default(alias_8);  alias_8 = None
        permute_251 = torch.ops.aten.permute.default(view_185, [0, 2, 1]);  view_185 = None
        permute_252 = torch.ops.aten.permute.default(view_186, [0, 2, 1]);  view_186 = None
        permute_256 = torch.ops.aten.permute.default(permute_91, [1, 0]);  permute_91 = None
        permute_261 = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
        permute_265 = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
        div_36 = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None
        permute_269 = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
        permute_273 = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
        div_37 = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
        permute_277 = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
        permute_282 = torch.ops.aten.permute.default(view_166, [0, 2, 1]);  view_166 = None
        permute_283 = torch.ops.aten.permute.default(view_167, [0, 2, 1]);  view_167 = None
        alias_18 = torch.ops.aten.alias.default(alias_7);  alias_7 = None
        permute_284 = torch.ops.aten.permute.default(view_163, [0, 2, 1]);  view_163 = None
        permute_285 = torch.ops.aten.permute.default(view_164, [0, 2, 1]);  view_164 = None
        permute_289 = torch.ops.aten.permute.default(permute_80, [1, 0]);  permute_80 = None
        permute_294 = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
        permute_298 = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
        div_39 = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None
        permute_302 = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
        permute_306 = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
        div_40 = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
        permute_310 = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
        permute_315 = torch.ops.aten.permute.default(view_144, [0, 2, 1]);  view_144 = None
        permute_316 = torch.ops.aten.permute.default(view_145, [0, 2, 1]);  view_145 = None
        alias_19 = torch.ops.aten.alias.default(alias_6);  alias_6 = None
        permute_317 = torch.ops.aten.permute.default(view_141, [0, 2, 1]);  view_141 = None
        permute_318 = torch.ops.aten.permute.default(view_142, [0, 2, 1]);  view_142 = None
        permute_322 = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
        permute_327 = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
        permute_331 = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
        div_42 = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
        permute_335 = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
        permute_339 = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
        div_43 = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
        permute_343 = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
        permute_348 = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None
        permute_349 = torch.ops.aten.permute.default(view_123, [0, 2, 1]);  view_123 = None
        alias_20 = torch.ops.aten.alias.default(alias_5);  alias_5 = None
        permute_350 = torch.ops.aten.permute.default(view_119, [0, 2, 1]);  view_119 = None
        permute_351 = torch.ops.aten.permute.default(view_120, [0, 2, 1]);  view_120 = None
        permute_355 = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
        permute_360 = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
        permute_364 = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
        div_45 = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
        permute_368 = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
        permute_372 = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
        div_46 = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
        permute_376 = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
        permute_381 = torch.ops.aten.permute.default(view_100, [0, 2, 1]);  view_100 = None
        permute_382 = torch.ops.aten.permute.default(view_101, [0, 2, 1]);  view_101 = None
        alias_21 = torch.ops.aten.alias.default(alias_4);  alias_4 = None
        permute_383 = torch.ops.aten.permute.default(view_97, [0, 2, 1]);  view_97 = None
        permute_384 = torch.ops.aten.permute.default(view_98, [0, 2, 1]);  view_98 = None
        permute_388 = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
        permute_393 = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
        permute_397 = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
        div_48 = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
        permute_401 = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
        permute_405 = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
        div_49 = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
        permute_409 = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
        permute_414 = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None
        permute_415 = torch.ops.aten.permute.default(view_79, [0, 2, 1]);  view_79 = None
        alias_22 = torch.ops.aten.alias.default(alias_3);  alias_3 = None
        permute_416 = torch.ops.aten.permute.default(view_75, [0, 2, 1]);  view_75 = None
        permute_417 = torch.ops.aten.permute.default(view_76, [0, 2, 1]);  view_76 = None
        permute_421 = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
        permute_426 = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
        permute_430 = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
        div_51 = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
        permute_434 = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
        permute_438 = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
        div_52 = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
        permute_442 = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
        permute_447 = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None
        permute_448 = torch.ops.aten.permute.default(view_57, [0, 2, 1]);  view_57 = None
        alias_23 = torch.ops.aten.alias.default(alias_2);  alias_2 = None
        permute_449 = torch.ops.aten.permute.default(view_53, [0, 2, 1]);  view_53 = None
        permute_450 = torch.ops.aten.permute.default(view_54, [0, 2, 1]);  view_54 = None
        permute_454 = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
        permute_459 = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
        permute_463 = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        div_54 = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
        permute_467 = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        permute_471 = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
        div_55 = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
        permute_475 = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
        permute_480 = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None
        permute_481 = torch.ops.aten.permute.default(view_35, [0, 2, 1]);  view_35 = None
        alias_24 = torch.ops.aten.alias.default(alias_1);  alias_1 = None
        permute_482 = torch.ops.aten.permute.default(view_31, [0, 2, 1]);  view_31 = None
        permute_483 = torch.ops.aten.permute.default(view_32, [0, 2, 1]);  view_32 = None
        permute_487 = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
        permute_492 = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        permute_496 = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        div_57 = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
        permute_500 = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        permute_504 = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        div_58 = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
        permute_508 = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        permute_513 = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
        permute_514 = torch.ops.aten.permute.default(view_13, [0, 2, 1]);  view_13 = None
        alias_25 = torch.ops.aten.alias.default(alias);  alias = None
        permute_515 = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
        permute_516 = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
        permute_520 = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        permute_525 = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        permute_529 = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        div_60 = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        return [add_99, tanh, primals_4, primals_14, primals_20, primals_30, primals_36, primals_46, primals_52, primals_62, primals_68, primals_78, primals_84, primals_94, primals_100, primals_110, primals_116, primals_126, primals_132, primals_142, primals_148, primals_158, primals_164, primals_174, primals_180, primals_190, primals_196, primals_202, expand, slice_6, mul_1, gt, view, gt_1, view_16, gt_2, mul_9, view_18, addmm_4, view_20, gt_3, mul_16, view_22, gt_4, view_38, gt_5, mul_22, view_40, addmm_10, view_42, gt_6, mul_29, view_44, gt_7, view_60, gt_8, mul_35, view_62, addmm_16, view_64, gt_9, mul_42, view_66, gt_10, view_82, gt_11, mul_48, view_84, addmm_22, view_86, gt_12, mul_55, view_88, gt_13, view_104, gt_14, mul_61, view_106, addmm_28, view_108, gt_15, mul_68, view_110, gt_16, view_126, gt_17, mul_74, view_128, addmm_34, view_130, gt_18, mul_81, view_132, gt_19, view_148, gt_20, mul_87, view_150, addmm_40, view_152, gt_21, mul_94, view_154, gt_22, view_170, gt_23, mul_100, view_172, addmm_46, view_174, gt_24, mul_107, view_176, gt_25, view_192, gt_26, mul_113, view_194, addmm_52, view_196, gt_27, mul_120, view_198, gt_28, view_214, gt_29, mul_126, view_216, addmm_58, view_218, gt_30, mul_133, view_220, gt_31, view_236, gt_32, mul_139, view_238, addmm_64, view_240, gt_33, mul_146, view_242, gt_34, view_258, gt_35, mul_152, view_260, addmm_70, view_262, gt_36, mul_159, select, tanh, permute_133, div_24, permute_137, permute_141, div_25, permute_145, permute_150, permute_151, alias_14, permute_152, permute_153, permute_157, permute_162, permute_166, div_27, permute_170, permute_174, div_28, permute_178, permute_183, permute_184, alias_15, permute_185, permute_186, permute_190, permute_195, permute_199, div_30, permute_203, permute_207, div_31, permute_211, permute_216, permute_217, alias_16, permute_218, permute_219, permute_223, permute_228, permute_232, div_33, permute_236, permute_240, div_34, permute_244, permute_249, permute_250, alias_17, permute_251, permute_252, permute_256, permute_261, permute_265, div_36, permute_269, permute_273, div_37, permute_277, permute_282, permute_283, alias_18, permute_284, permute_285, permute_289, permute_294, permute_298, div_39, permute_302, permute_306, div_40, permute_310, permute_315, permute_316, alias_19, permute_317, permute_318, permute_322, permute_327, permute_331, div_42, permute_335, permute_339, div_43, permute_343, permute_348, permute_349, alias_20, permute_350, permute_351, permute_355, permute_360, permute_364, div_45, permute_368, permute_372, div_46, permute_376, permute_381, permute_382, alias_21, permute_383, permute_384, permute_388, permute_393, permute_397, div_48, permute_401, permute_405, div_49, permute_409, permute_414, permute_415, alias_22, permute_416, permute_417, permute_421, permute_426, permute_430, div_51, permute_434, permute_438, div_52, permute_442, permute_447, permute_448, alias_23, permute_449, permute_450, permute_454, permute_459, permute_463, div_54, permute_467, permute_471, div_55, permute_475, permute_480, permute_481, alias_24, permute_482, permute_483, permute_487, permute_492, permute_496, div_57, permute_500, permute_504, div_58, permute_508, permute_513, permute_514, alias_25, permute_515, permute_516, permute_520, permute_525, permute_529, div_60]
        
def load_args(reader):
    buf0 = reader.storage(None, 93763584, device=device(type='cuda', index=0))
    reader.tensor(buf0, (30522, 768), requires_grad=True, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf1, (2, 768), requires_grad=True, is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf2, (512, 768), requires_grad=True, is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf3, (768,), requires_grad=True, is_leaf=True)  # primals_4
    buf4 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf4, (768,), requires_grad=True, is_leaf=True)  # primals_5
    buf5 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf5, (768, 768), requires_grad=True, is_leaf=True)  # primals_6
    buf6 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf6, (768,), requires_grad=True, is_leaf=True)  # primals_7
    buf7 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf7, (768, 768), requires_grad=True, is_leaf=True)  # primals_8
    buf8 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf8, (768,), requires_grad=True, is_leaf=True)  # primals_9
    buf9 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf9, (768, 768), requires_grad=True, is_leaf=True)  # primals_10
    buf10 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf10, (768,), requires_grad=True, is_leaf=True)  # primals_11
    buf11 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf11, (768, 768), requires_grad=True, is_leaf=True)  # primals_12
    buf12 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf12, (768,), requires_grad=True, is_leaf=True)  # primals_13
    buf13 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf13, (768,), requires_grad=True, is_leaf=True)  # primals_14
    buf14 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf14, (768,), requires_grad=True, is_leaf=True)  # primals_15
    buf15 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf15, (3072, 768), requires_grad=True, is_leaf=True)  # primals_16
    buf16 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf16, (3072,), requires_grad=True, is_leaf=True)  # primals_17
    buf17 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf17, (768, 3072), requires_grad=True, is_leaf=True)  # primals_18
    buf18 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf18, (768,), requires_grad=True, is_leaf=True)  # primals_19
    buf19 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf19, (768,), requires_grad=True, is_leaf=True)  # primals_20
    buf20 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf20, (768,), requires_grad=True, is_leaf=True)  # primals_21
    buf21 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf21, (768, 768), requires_grad=True, is_leaf=True)  # primals_22
    buf22 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf22, (768,), requires_grad=True, is_leaf=True)  # primals_23
    buf23 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf23, (768, 768), requires_grad=True, is_leaf=True)  # primals_24
    buf24 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf24, (768,), requires_grad=True, is_leaf=True)  # primals_25
    buf25 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf25, (768, 768), requires_grad=True, is_leaf=True)  # primals_26
    buf26 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf26, (768,), requires_grad=True, is_leaf=True)  # primals_27
    buf27 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf27, (768, 768), requires_grad=True, is_leaf=True)  # primals_28
    buf28 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf28, (768,), requires_grad=True, is_leaf=True)  # primals_29
    buf29 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf29, (768,), requires_grad=True, is_leaf=True)  # primals_30
    buf30 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf30, (768,), requires_grad=True, is_leaf=True)  # primals_31
    buf31 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf31, (3072, 768), requires_grad=True, is_leaf=True)  # primals_32
    buf32 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf32, (3072,), requires_grad=True, is_leaf=True)  # primals_33
    buf33 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf33, (768, 3072), requires_grad=True, is_leaf=True)  # primals_34
    buf34 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf34, (768,), requires_grad=True, is_leaf=True)  # primals_35
    buf35 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf35, (768,), requires_grad=True, is_leaf=True)  # primals_36
    buf36 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf36, (768,), requires_grad=True, is_leaf=True)  # primals_37
    buf37 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf37, (768, 768), requires_grad=True, is_leaf=True)  # primals_38
    buf38 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf38, (768,), requires_grad=True, is_leaf=True)  # primals_39
    buf39 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf39, (768, 768), requires_grad=True, is_leaf=True)  # primals_40
    buf40 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf40, (768,), requires_grad=True, is_leaf=True)  # primals_41
    buf41 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf41, (768, 768), requires_grad=True, is_leaf=True)  # primals_42
    buf42 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf42, (768,), requires_grad=True, is_leaf=True)  # primals_43
    buf43 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf43, (768, 768), requires_grad=True, is_leaf=True)  # primals_44
    buf44 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf44, (768,), requires_grad=True, is_leaf=True)  # primals_45
    buf45 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf45, (768,), requires_grad=True, is_leaf=True)  # primals_46
    buf46 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf46, (768,), requires_grad=True, is_leaf=True)  # primals_47
    buf47 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf47, (3072, 768), requires_grad=True, is_leaf=True)  # primals_48
    buf48 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf48, (3072,), requires_grad=True, is_leaf=True)  # primals_49
    buf49 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf49, (768, 3072), requires_grad=True, is_leaf=True)  # primals_50
    buf50 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf50, (768,), requires_grad=True, is_leaf=True)  # primals_51
    buf51 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf51, (768,), requires_grad=True, is_leaf=True)  # primals_52
    buf52 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf52, (768,), requires_grad=True, is_leaf=True)  # primals_53
    buf53 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf53, (768, 768), requires_grad=True, is_leaf=True)  # primals_54
    buf54 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf54, (768,), requires_grad=True, is_leaf=True)  # primals_55
    buf55 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf55, (768, 768), requires_grad=True, is_leaf=True)  # primals_56
    buf56 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf56, (768,), requires_grad=True, is_leaf=True)  # primals_57
    buf57 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf57, (768, 768), requires_grad=True, is_leaf=True)  # primals_58
    buf58 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf58, (768,), requires_grad=True, is_leaf=True)  # primals_59
    buf59 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf59, (768, 768), requires_grad=True, is_leaf=True)  # primals_60
    buf60 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf60, (768,), requires_grad=True, is_leaf=True)  # primals_61
    buf61 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf61, (768,), requires_grad=True, is_leaf=True)  # primals_62
    buf62 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf62, (768,), requires_grad=True, is_leaf=True)  # primals_63
    buf63 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf63, (3072, 768), requires_grad=True, is_leaf=True)  # primals_64
    buf64 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf64, (3072,), requires_grad=True, is_leaf=True)  # primals_65
    buf65 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf65, (768, 3072), requires_grad=True, is_leaf=True)  # primals_66
    buf66 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf66, (768,), requires_grad=True, is_leaf=True)  # primals_67
    buf67 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf67, (768,), requires_grad=True, is_leaf=True)  # primals_68
    buf68 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf68, (768,), requires_grad=True, is_leaf=True)  # primals_69
    buf69 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf69, (768, 768), requires_grad=True, is_leaf=True)  # primals_70
    buf70 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf70, (768,), requires_grad=True, is_leaf=True)  # primals_71
    buf71 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf71, (768, 768), requires_grad=True, is_leaf=True)  # primals_72
    buf72 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf72, (768,), requires_grad=True, is_leaf=True)  # primals_73
    buf73 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf73, (768, 768), requires_grad=True, is_leaf=True)  # primals_74
    buf74 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf74, (768,), requires_grad=True, is_leaf=True)  # primals_75
    buf75 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf75, (768, 768), requires_grad=True, is_leaf=True)  # primals_76
    buf76 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf76, (768,), requires_grad=True, is_leaf=True)  # primals_77
    buf77 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf77, (768,), requires_grad=True, is_leaf=True)  # primals_78
    buf78 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf78, (768,), requires_grad=True, is_leaf=True)  # primals_79
    buf79 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf79, (3072, 768), requires_grad=True, is_leaf=True)  # primals_80
    buf80 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf80, (3072,), requires_grad=True, is_leaf=True)  # primals_81
    buf81 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf81, (768, 3072), requires_grad=True, is_leaf=True)  # primals_82
    buf82 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf82, (768,), requires_grad=True, is_leaf=True)  # primals_83
    buf83 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf83, (768,), requires_grad=True, is_leaf=True)  # primals_84
    buf84 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf84, (768,), requires_grad=True, is_leaf=True)  # primals_85
    buf85 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf85, (768, 768), requires_grad=True, is_leaf=True)  # primals_86
    buf86 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf86, (768,), requires_grad=True, is_leaf=True)  # primals_87
    buf87 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf87, (768, 768), requires_grad=True, is_leaf=True)  # primals_88
    buf88 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf88, (768,), requires_grad=True, is_leaf=True)  # primals_89
    buf89 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf89, (768, 768), requires_grad=True, is_leaf=True)  # primals_90
    buf90 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf90, (768,), requires_grad=True, is_leaf=True)  # primals_91
    buf91 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf91, (768, 768), requires_grad=True, is_leaf=True)  # primals_92
    buf92 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf92, (768,), requires_grad=True, is_leaf=True)  # primals_93
    buf93 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf93, (768,), requires_grad=True, is_leaf=True)  # primals_94
    buf94 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf94, (768,), requires_grad=True, is_leaf=True)  # primals_95
    buf95 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf95, (3072, 768), requires_grad=True, is_leaf=True)  # primals_96
    buf96 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf96, (3072,), requires_grad=True, is_leaf=True)  # primals_97
    buf97 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf97, (768, 3072), requires_grad=True, is_leaf=True)  # primals_98
    buf98 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf98, (768,), requires_grad=True, is_leaf=True)  # primals_99
    buf99 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf99, (768,), requires_grad=True, is_leaf=True)  # primals_100
    buf100 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf100, (768,), requires_grad=True, is_leaf=True)  # primals_101
    buf101 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf101, (768, 768), requires_grad=True, is_leaf=True)  # primals_102
    buf102 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf102, (768,), requires_grad=True, is_leaf=True)  # primals_103
    buf103 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf103, (768, 768), requires_grad=True, is_leaf=True)  # primals_104
    buf104 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf104, (768,), requires_grad=True, is_leaf=True)  # primals_105
    buf105 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf105, (768, 768), requires_grad=True, is_leaf=True)  # primals_106
    buf106 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf106, (768,), requires_grad=True, is_leaf=True)  # primals_107
    buf107 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf107, (768, 768), requires_grad=True, is_leaf=True)  # primals_108
    buf108 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf108, (768,), requires_grad=True, is_leaf=True)  # primals_109
    buf109 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf109, (768,), requires_grad=True, is_leaf=True)  # primals_110
    buf110 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf110, (768,), requires_grad=True, is_leaf=True)  # primals_111
    buf111 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf111, (3072, 768), requires_grad=True, is_leaf=True)  # primals_112
    buf112 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf112, (3072,), requires_grad=True, is_leaf=True)  # primals_113
    buf113 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf113, (768, 3072), requires_grad=True, is_leaf=True)  # primals_114
    buf114 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf114, (768,), requires_grad=True, is_leaf=True)  # primals_115
    buf115 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf115, (768,), requires_grad=True, is_leaf=True)  # primals_116
    buf116 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf116, (768,), requires_grad=True, is_leaf=True)  # primals_117
    buf117 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf117, (768, 768), requires_grad=True, is_leaf=True)  # primals_118
    buf118 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf118, (768,), requires_grad=True, is_leaf=True)  # primals_119
    buf119 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf119, (768, 768), requires_grad=True, is_leaf=True)  # primals_120
    buf120 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf120, (768,), requires_grad=True, is_leaf=True)  # primals_121
    buf121 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf121, (768, 768), requires_grad=True, is_leaf=True)  # primals_122
    buf122 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf122, (768,), requires_grad=True, is_leaf=True)  # primals_123
    buf123 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf123, (768, 768), requires_grad=True, is_leaf=True)  # primals_124
    buf124 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf124, (768,), requires_grad=True, is_leaf=True)  # primals_125
    buf125 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf125, (768,), requires_grad=True, is_leaf=True)  # primals_126
    buf126 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf126, (768,), requires_grad=True, is_leaf=True)  # primals_127
    buf127 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf127, (3072, 768), requires_grad=True, is_leaf=True)  # primals_128
    buf128 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf128, (3072,), requires_grad=True, is_leaf=True)  # primals_129
    buf129 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf129, (768, 3072), requires_grad=True, is_leaf=True)  # primals_130
    buf130 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf130, (768,), requires_grad=True, is_leaf=True)  # primals_131
    buf131 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf131, (768,), requires_grad=True, is_leaf=True)  # primals_132
    buf132 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf132, (768,), requires_grad=True, is_leaf=True)  # primals_133
    buf133 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf133, (768, 768), requires_grad=True, is_leaf=True)  # primals_134
    buf134 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf134, (768,), requires_grad=True, is_leaf=True)  # primals_135
    buf135 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf135, (768, 768), requires_grad=True, is_leaf=True)  # primals_136
    buf136 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf136, (768,), requires_grad=True, is_leaf=True)  # primals_137
    buf137 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf137, (768, 768), requires_grad=True, is_leaf=True)  # primals_138
    buf138 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf138, (768,), requires_grad=True, is_leaf=True)  # primals_139
    buf139 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf139, (768, 768), requires_grad=True, is_leaf=True)  # primals_140
    buf140 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf140, (768,), requires_grad=True, is_leaf=True)  # primals_141
    buf141 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf141, (768,), requires_grad=True, is_leaf=True)  # primals_142
    buf142 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf142, (768,), requires_grad=True, is_leaf=True)  # primals_143
    buf143 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf143, (3072, 768), requires_grad=True, is_leaf=True)  # primals_144
    buf144 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf144, (3072,), requires_grad=True, is_leaf=True)  # primals_145
    buf145 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf145, (768, 3072), requires_grad=True, is_leaf=True)  # primals_146
    buf146 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf146, (768,), requires_grad=True, is_leaf=True)  # primals_147
    buf147 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf147, (768,), requires_grad=True, is_leaf=True)  # primals_148
    buf148 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf148, (768,), requires_grad=True, is_leaf=True)  # primals_149
    buf149 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf149, (768, 768), requires_grad=True, is_leaf=True)  # primals_150
    buf150 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf150, (768,), requires_grad=True, is_leaf=True)  # primals_151
    buf151 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf151, (768, 768), requires_grad=True, is_leaf=True)  # primals_152
    buf152 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf152, (768,), requires_grad=True, is_leaf=True)  # primals_153
    buf153 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf153, (768, 768), requires_grad=True, is_leaf=True)  # primals_154
    buf154 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf154, (768,), requires_grad=True, is_leaf=True)  # primals_155
    buf155 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf155, (768, 768), requires_grad=True, is_leaf=True)  # primals_156
    buf156 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf156, (768,), requires_grad=True, is_leaf=True)  # primals_157
    buf157 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf157, (768,), requires_grad=True, is_leaf=True)  # primals_158
    buf158 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf158, (768,), requires_grad=True, is_leaf=True)  # primals_159
    buf159 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf159, (3072, 768), requires_grad=True, is_leaf=True)  # primals_160
    buf160 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf160, (3072,), requires_grad=True, is_leaf=True)  # primals_161
    buf161 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf161, (768, 3072), requires_grad=True, is_leaf=True)  # primals_162
    buf162 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf162, (768,), requires_grad=True, is_leaf=True)  # primals_163
    buf163 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf163, (768,), requires_grad=True, is_leaf=True)  # primals_164
    buf164 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf164, (768,), requires_grad=True, is_leaf=True)  # primals_165
    buf165 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf165, (768, 768), requires_grad=True, is_leaf=True)  # primals_166
    buf166 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf166, (768,), requires_grad=True, is_leaf=True)  # primals_167
    buf167 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf167, (768, 768), requires_grad=True, is_leaf=True)  # primals_168
    buf168 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf168, (768,), requires_grad=True, is_leaf=True)  # primals_169
    buf169 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf169, (768, 768), requires_grad=True, is_leaf=True)  # primals_170
    buf170 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf170, (768,), requires_grad=True, is_leaf=True)  # primals_171
    buf171 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf171, (768, 768), requires_grad=True, is_leaf=True)  # primals_172
    buf172 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf172, (768,), requires_grad=True, is_leaf=True)  # primals_173
    buf173 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf173, (768,), requires_grad=True, is_leaf=True)  # primals_174
    buf174 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf174, (768,), requires_grad=True, is_leaf=True)  # primals_175
    buf175 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf175, (3072, 768), requires_grad=True, is_leaf=True)  # primals_176
    buf176 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf176, (3072,), requires_grad=True, is_leaf=True)  # primals_177
    buf177 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf177, (768, 3072), requires_grad=True, is_leaf=True)  # primals_178
    buf178 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf178, (768,), requires_grad=True, is_leaf=True)  # primals_179
    buf179 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf179, (768,), requires_grad=True, is_leaf=True)  # primals_180
    buf180 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf180, (768,), requires_grad=True, is_leaf=True)  # primals_181
    buf181 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf181, (768, 768), requires_grad=True, is_leaf=True)  # primals_182
    buf182 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf182, (768,), requires_grad=True, is_leaf=True)  # primals_183
    buf183 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf183, (768, 768), requires_grad=True, is_leaf=True)  # primals_184
    buf184 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf184, (768,), requires_grad=True, is_leaf=True)  # primals_185
    buf185 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf185, (768, 768), requires_grad=True, is_leaf=True)  # primals_186
    buf186 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf186, (768,), requires_grad=True, is_leaf=True)  # primals_187
    buf187 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf187, (768, 768), requires_grad=True, is_leaf=True)  # primals_188
    buf188 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf188, (768,), requires_grad=True, is_leaf=True)  # primals_189
    buf189 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf189, (768,), requires_grad=True, is_leaf=True)  # primals_190
    buf190 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf190, (768,), requires_grad=True, is_leaf=True)  # primals_191
    buf191 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf191, (3072, 768), requires_grad=True, is_leaf=True)  # primals_192
    buf192 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf192, (3072,), requires_grad=True, is_leaf=True)  # primals_193
    buf193 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf193, (768, 3072), requires_grad=True, is_leaf=True)  # primals_194
    buf194 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf194, (768,), requires_grad=True, is_leaf=True)  # primals_195
    buf195 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf195, (768,), requires_grad=True, is_leaf=True)  # primals_196
    buf196 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf196, (768,), requires_grad=True, is_leaf=True)  # primals_197
    buf197 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf197, (768, 768), requires_grad=True, is_leaf=True)  # primals_198
    buf198 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf198, (768,), requires_grad=True, is_leaf=True)  # primals_199
    buf199 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf199, (1, 512), dtype=torch.int64, is_leaf=True)  # primals_200
    buf200 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf200, (1, 512), dtype=torch.int64, is_leaf=True)  # primals_201
    buf201 = reader.storage(None, 120, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf201, (3, 5), dtype=torch.int64, is_leaf=True)  # primals_202
    buf202 = reader.storage(None, 120, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf202, (3, 5), dtype=torch.int64, is_leaf=True)  # primals_203
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
