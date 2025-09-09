
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



# torch version: 2.3.0+cu121
# torch cuda version: 12.1
# torch git version: 97ff6cfd9c86c5c09d7ce775ab64ec5c99230f5d


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2023 NVIDIA Corporation 
# Built on Tue_Jun_13_19:16:58_PDT_2023 
# Cuda compilation tools, release 12.2, V12.2.91 
# Build cuda_12.2.r12.2/compiler.32965470_0 

# GPU Hardware Info: 
# NVIDIA A100-PCIE-40GB : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_160, primals_161, primals_163, primals_164, primals_166, primals_167, primals_169, primals_170, primals_172, primals_173, primals_175, primals_176, primals_178, primals_179, primals_181, primals_182, primals_184, primals_185, primals_187, primals_188, primals_190, primals_191, primals_193, primals_194, primals_196, primals_197, primals_199, primals_200, primals_202, primals_203, primals_205, primals_206, primals_208, primals_209, primals_211, primals_212, primals_214, primals_215, primals_219, primals_220, primals_222, primals_223, primals_225, primals_226, primals_228, primals_229, primals_231, primals_232, primals_234, primals_235, primals_237, primals_238, primals_240, primals_241, primals_243, primals_244, primals_246, primals_247, primals_249, primals_250, primals_252, primals_253, primals_255, primals_256, primals_258, primals_259, primals_261, primals_262, primals_264, primals_265, primals_267, primals_268, primals_270, primals_271, primals_273, primals_274, primals_276, primals_277, primals_279, primals_280, primals_282, primals_283, primals_285, primals_286, primals_288, primals_289, cat, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, getitem_6, getitem_7, convolution_3, squeeze_10, relu_3, convolution_4, squeeze_13, relu_4, getitem_12, getitem_13, convolution_5, squeeze_16, convolution_6, squeeze_19, relu_6, convolution_7, squeeze_22, convolution_8, squeeze_25, relu_8, convolution_9, squeeze_28, relu_9, convolution_10, squeeze_31, avg_pool2d, convolution_11, squeeze_34, cat_1, convolution_12, squeeze_37, convolution_13, squeeze_40, relu_13, convolution_14, squeeze_43, convolution_15, squeeze_46, relu_15, convolution_16, squeeze_49, relu_16, convolution_17, squeeze_52, avg_pool2d_1, convolution_18, squeeze_55, cat_2, convolution_19, squeeze_58, convolution_20, squeeze_61, relu_20, convolution_21, squeeze_64, convolution_22, squeeze_67, relu_22, convolution_23, squeeze_70, relu_23, convolution_24, squeeze_73, avg_pool2d_2, convolution_25, squeeze_76, cat_3, convolution_26, squeeze_79, convolution_27, squeeze_82, relu_27, convolution_28, squeeze_85, relu_28, convolution_29, squeeze_88, getitem_65, cat_4, convolution_30, squeeze_91, convolution_31, squeeze_94, relu_31, convolution_32, squeeze_97, relu_32, convolution_33, squeeze_100, convolution_34, squeeze_103, relu_34, convolution_35, squeeze_106, relu_35, convolution_36, squeeze_109, relu_36, convolution_37, squeeze_112, relu_37, convolution_38, squeeze_115, avg_pool2d_3, convolution_39, squeeze_118, cat_5, convolution_40, squeeze_121, convolution_41, squeeze_124, relu_41, convolution_42, squeeze_127, relu_42, convolution_43, squeeze_130, convolution_44, squeeze_133, relu_44, convolution_45, squeeze_136, relu_45, convolution_46, squeeze_139, relu_46, convolution_47, squeeze_142, relu_47, convolution_48, squeeze_145, avg_pool2d_4, convolution_49, squeeze_148, cat_6, convolution_50, squeeze_151, convolution_51, squeeze_154, relu_51, convolution_52, squeeze_157, relu_52, convolution_53, squeeze_160, convolution_54, squeeze_163, relu_54, convolution_55, squeeze_166, relu_55, convolution_56, squeeze_169, relu_56, convolution_57, squeeze_172, relu_57, convolution_58, squeeze_175, avg_pool2d_5, convolution_59, squeeze_178, cat_7, convolution_60, squeeze_181, convolution_61, squeeze_184, relu_61, convolution_62, squeeze_187, relu_62, convolution_63, squeeze_190, convolution_64, squeeze_193, relu_64, convolution_65, squeeze_196, relu_65, convolution_66, squeeze_199, relu_66, convolution_67, squeeze_202, relu_67, convolution_68, squeeze_205, avg_pool2d_6, convolution_69, squeeze_208, cat_8, avg_pool2d_7, convolution_70, squeeze_211, relu_70, convolution_71, squeeze_214, view, convolution_72, squeeze_217, relu_72, convolution_73, squeeze_220, convolution_74, squeeze_223, relu_74, convolution_75, squeeze_226, relu_75, convolution_76, squeeze_229, relu_76, convolution_77, squeeze_232, getitem_163, cat_9, convolution_78, squeeze_235, convolution_79, squeeze_238, relu_79, convolution_80, squeeze_241, convolution_81, squeeze_244, convolution_82, squeeze_247, relu_82, convolution_83, squeeze_250, relu_83, convolution_84, squeeze_253, convolution_85, squeeze_256, avg_pool2d_8, convolution_86, squeeze_259, cat_12, convolution_87, squeeze_262, convolution_88, squeeze_265, relu_88, convolution_89, squeeze_268, convolution_90, squeeze_271, convolution_91, squeeze_274, relu_91, convolution_92, squeeze_277, relu_92, convolution_93, squeeze_280, convolution_94, squeeze_283, avg_pool2d_9, convolution_95, squeeze_286, gt, view_1, permute_2, le, unsqueeze_389, le_1, unsqueeze_401, le_2, unsqueeze_413, unsqueeze_425, unsqueeze_437, le_5, unsqueeze_449, le_6, unsqueeze_461, unsqueeze_473, le_8, unsqueeze_485, le_9, unsqueeze_497, le_10, unsqueeze_509, le_11, unsqueeze_521, unsqueeze_533, unsqueeze_545, le_14, unsqueeze_557, le_15, unsqueeze_569, unsqueeze_581, le_17, unsqueeze_593, le_18, unsqueeze_605, unsqueeze_617, unsqueeze_629, unsqueeze_641, le_22, unsqueeze_653, unsqueeze_665, permute_6, le_24, unsqueeze_677, unsqueeze_689, le_26, unsqueeze_701, le_27, unsqueeze_713, unsqueeze_725, unsqueeze_737, unsqueeze_749, unsqueeze_761, le_32, unsqueeze_773, unsqueeze_785, unsqueeze_797, le_35, unsqueeze_809, le_36, unsqueeze_821, le_37, unsqueeze_833, unsqueeze_845, unsqueeze_857, unsqueeze_869, unsqueeze_881, le_42, unsqueeze_893, unsqueeze_905, unsqueeze_917, le_45, unsqueeze_929, le_46, unsqueeze_941, le_47, unsqueeze_953, unsqueeze_965, unsqueeze_977, unsqueeze_989, unsqueeze_1001, le_52, unsqueeze_1013, unsqueeze_1025, unsqueeze_1037, le_55, unsqueeze_1049, le_56, unsqueeze_1061, le_57, unsqueeze_1073, unsqueeze_1085, unsqueeze_1097, unsqueeze_1109, unsqueeze_1121, le_62, unsqueeze_1133, unsqueeze_1145, unsqueeze_1157, le_65, unsqueeze_1169, le_66, unsqueeze_1181, unsqueeze_1193, unsqueeze_1205, le_69, unsqueeze_1217, le_70, unsqueeze_1229, le_71, unsqueeze_1241, unsqueeze_1253, unsqueeze_1265, le_74, unsqueeze_1277, unsqueeze_1289, le_76, unsqueeze_1301, le_77, unsqueeze_1313, le_78, unsqueeze_1325, unsqueeze_1337, unsqueeze_1349, le_81, unsqueeze_1361, unsqueeze_1373, le_83, unsqueeze_1385, le_84, unsqueeze_1397, le_85, unsqueeze_1409, unsqueeze_1421, unsqueeze_1433, le_88, unsqueeze_1445, unsqueeze_1457, le_90, unsqueeze_1469, unsqueeze_1481, unsqueeze_1493, unsqueeze_1505, unsqueeze_1517, unsqueeze_1529, tangents_1, tangents_2):
        mm = torch.ops.aten.mm.default(tangents_1, permute_2);  permute_2 = None
        permute_3 = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_1 = torch.ops.aten.mm.default(permute_3, view_1);  permute_3 = view_1 = None
        permute_4 = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        view_2 = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
        permute_5 = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        view_3 = torch.ops.aten.view.default(mm, [32, 2048, 1, 1]);  mm = None
        convert_element_type = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_677 = torch.ops.aten.mul.Tensor(convert_element_type, 2.0);  convert_element_type = None
        mul_678 = torch.ops.aten.mul.Tensor(view_3, mul_677);  view_3 = mul_677 = None
        clone = torch.ops.aten.clone.default(mul_678, memory_format = torch.contiguous_format);  mul_678 = None
        expand = torch.ops.aten.expand.default(clone, [32, 2048, 8, 8]);  clone = None
        div = torch.ops.aten.div.Scalar(expand, 64);  expand = None
        slice_4 = torch.ops.aten.slice.Tensor(div, 1, 0, 320)
        slice_5 = torch.ops.aten.slice.Tensor(div, 1, 320, 1088)
        slice_6 = torch.ops.aten.slice.Tensor(div, 1, 1088, 1856)
        slice_7 = torch.ops.aten.slice.Tensor(div, 1, 1856, 2048);  div = None
        full_default = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(le, full_default, slice_7);  le = slice_7 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
        sub_96 = torch.ops.aten.sub.Tensor(convolution_95, unsqueeze_389);  convolution_95 = unsqueeze_389 = None
        mul_679 = torch.ops.aten.mul.Tensor(where, sub_96)
        sum_3 = torch.ops.aten.sum.dim_IntList(mul_679, [0, 2, 3]);  mul_679 = None
        mul_680 = torch.ops.aten.mul.Tensor(sum_2, 0.00048828125)
        unsqueeze_390 = torch.ops.aten.unsqueeze.default(mul_680, 0);  mul_680 = None
        unsqueeze_391 = torch.ops.aten.unsqueeze.default(unsqueeze_390, 2);  unsqueeze_390 = None
        unsqueeze_392 = torch.ops.aten.unsqueeze.default(unsqueeze_391, 3);  unsqueeze_391 = None
        mul_681 = torch.ops.aten.mul.Tensor(sum_3, 0.00048828125)
        mul_682 = torch.ops.aten.mul.Tensor(squeeze_286, squeeze_286)
        mul_683 = torch.ops.aten.mul.Tensor(mul_681, mul_682);  mul_681 = mul_682 = None
        unsqueeze_393 = torch.ops.aten.unsqueeze.default(mul_683, 0);  mul_683 = None
        unsqueeze_394 = torch.ops.aten.unsqueeze.default(unsqueeze_393, 2);  unsqueeze_393 = None
        unsqueeze_395 = torch.ops.aten.unsqueeze.default(unsqueeze_394, 3);  unsqueeze_394 = None
        mul_684 = torch.ops.aten.mul.Tensor(squeeze_286, primals_289);  primals_289 = None
        unsqueeze_396 = torch.ops.aten.unsqueeze.default(mul_684, 0);  mul_684 = None
        unsqueeze_397 = torch.ops.aten.unsqueeze.default(unsqueeze_396, 2);  unsqueeze_396 = None
        unsqueeze_398 = torch.ops.aten.unsqueeze.default(unsqueeze_397, 3);  unsqueeze_397 = None
        mul_685 = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_395);  sub_96 = unsqueeze_395 = None
        sub_98 = torch.ops.aten.sub.Tensor(where, mul_685);  where = mul_685 = None
        sub_99 = torch.ops.aten.sub.Tensor(sub_98, unsqueeze_392);  sub_98 = unsqueeze_392 = None
        mul_686 = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_398);  sub_99 = unsqueeze_398 = None
        mul_687 = torch.ops.aten.mul.Tensor(sum_3, squeeze_286);  sum_3 = squeeze_286 = None
        convolution_backward = torch.ops.aten.convolution_backward.default(mul_686, avg_pool2d_9, primals_288, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_686 = avg_pool2d_9 = primals_288 = None
        getitem_200 = convolution_backward[0]
        getitem_201 = convolution_backward[1];  convolution_backward = None
        avg_pool2d_backward = torch.ops.aten.avg_pool2d_backward.default(getitem_200, cat_12, [3, 3], [1, 1], [1, 1], False, True, None);  getitem_200 = None
        slice_8 = torch.ops.aten.slice.Tensor(slice_6, 1, 0, 384)
        slice_9 = torch.ops.aten.slice.Tensor(slice_6, 1, 384, 768);  slice_6 = None
        where_1 = torch.ops.aten.where.self(le_1, full_default, slice_9);  le_1 = slice_9 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
        sub_100 = torch.ops.aten.sub.Tensor(convolution_94, unsqueeze_401);  convolution_94 = unsqueeze_401 = None
        mul_688 = torch.ops.aten.mul.Tensor(where_1, sub_100)
        sum_5 = torch.ops.aten.sum.dim_IntList(mul_688, [0, 2, 3]);  mul_688 = None
        mul_689 = torch.ops.aten.mul.Tensor(sum_4, 0.00048828125)
        unsqueeze_402 = torch.ops.aten.unsqueeze.default(mul_689, 0);  mul_689 = None
        unsqueeze_403 = torch.ops.aten.unsqueeze.default(unsqueeze_402, 2);  unsqueeze_402 = None
        unsqueeze_404 = torch.ops.aten.unsqueeze.default(unsqueeze_403, 3);  unsqueeze_403 = None
        mul_690 = torch.ops.aten.mul.Tensor(sum_5, 0.00048828125)
        mul_691 = torch.ops.aten.mul.Tensor(squeeze_283, squeeze_283)
        mul_692 = torch.ops.aten.mul.Tensor(mul_690, mul_691);  mul_690 = mul_691 = None
        unsqueeze_405 = torch.ops.aten.unsqueeze.default(mul_692, 0);  mul_692 = None
        unsqueeze_406 = torch.ops.aten.unsqueeze.default(unsqueeze_405, 2);  unsqueeze_405 = None
        unsqueeze_407 = torch.ops.aten.unsqueeze.default(unsqueeze_406, 3);  unsqueeze_406 = None
        mul_693 = torch.ops.aten.mul.Tensor(squeeze_283, primals_286);  primals_286 = None
        unsqueeze_408 = torch.ops.aten.unsqueeze.default(mul_693, 0);  mul_693 = None
        unsqueeze_409 = torch.ops.aten.unsqueeze.default(unsqueeze_408, 2);  unsqueeze_408 = None
        unsqueeze_410 = torch.ops.aten.unsqueeze.default(unsqueeze_409, 3);  unsqueeze_409 = None
        mul_694 = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_407);  sub_100 = unsqueeze_407 = None
        sub_102 = torch.ops.aten.sub.Tensor(where_1, mul_694);  where_1 = mul_694 = None
        sub_103 = torch.ops.aten.sub.Tensor(sub_102, unsqueeze_404);  sub_102 = unsqueeze_404 = None
        mul_695 = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_410);  sub_103 = unsqueeze_410 = None
        mul_696 = torch.ops.aten.mul.Tensor(sum_5, squeeze_283);  sum_5 = squeeze_283 = None
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_695, relu_92, primals_285, [0], [1, 1], [1, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_695 = primals_285 = None
        getitem_203 = convolution_backward_1[0]
        getitem_204 = convolution_backward_1[1];  convolution_backward_1 = None
        where_2 = torch.ops.aten.where.self(le_2, full_default, slice_8);  le_2 = slice_8 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
        sub_104 = torch.ops.aten.sub.Tensor(convolution_93, unsqueeze_413);  convolution_93 = unsqueeze_413 = None
        mul_697 = torch.ops.aten.mul.Tensor(where_2, sub_104)
        sum_7 = torch.ops.aten.sum.dim_IntList(mul_697, [0, 2, 3]);  mul_697 = None
        mul_698 = torch.ops.aten.mul.Tensor(sum_6, 0.00048828125)
        unsqueeze_414 = torch.ops.aten.unsqueeze.default(mul_698, 0);  mul_698 = None
        unsqueeze_415 = torch.ops.aten.unsqueeze.default(unsqueeze_414, 2);  unsqueeze_414 = None
        unsqueeze_416 = torch.ops.aten.unsqueeze.default(unsqueeze_415, 3);  unsqueeze_415 = None
        mul_699 = torch.ops.aten.mul.Tensor(sum_7, 0.00048828125)
        mul_700 = torch.ops.aten.mul.Tensor(squeeze_280, squeeze_280)
        mul_701 = torch.ops.aten.mul.Tensor(mul_699, mul_700);  mul_699 = mul_700 = None
        unsqueeze_417 = torch.ops.aten.unsqueeze.default(mul_701, 0);  mul_701 = None
        unsqueeze_418 = torch.ops.aten.unsqueeze.default(unsqueeze_417, 2);  unsqueeze_417 = None
        unsqueeze_419 = torch.ops.aten.unsqueeze.default(unsqueeze_418, 3);  unsqueeze_418 = None
        mul_702 = torch.ops.aten.mul.Tensor(squeeze_280, primals_283);  primals_283 = None
        unsqueeze_420 = torch.ops.aten.unsqueeze.default(mul_702, 0);  mul_702 = None
        unsqueeze_421 = torch.ops.aten.unsqueeze.default(unsqueeze_420, 2);  unsqueeze_420 = None
        unsqueeze_422 = torch.ops.aten.unsqueeze.default(unsqueeze_421, 3);  unsqueeze_421 = None
        mul_703 = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_419);  sub_104 = unsqueeze_419 = None
        sub_106 = torch.ops.aten.sub.Tensor(where_2, mul_703);  where_2 = mul_703 = None
        sub_107 = torch.ops.aten.sub.Tensor(sub_106, unsqueeze_416);  sub_106 = unsqueeze_416 = None
        mul_704 = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_422);  sub_107 = unsqueeze_422 = None
        mul_705 = torch.ops.aten.mul.Tensor(sum_7, squeeze_280);  sum_7 = squeeze_280 = None
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_704, relu_92, primals_282, [0], [1, 1], [0, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_704 = primals_282 = None
        getitem_206 = convolution_backward_2[0]
        getitem_207 = convolution_backward_2[1];  convolution_backward_2 = None
        add_483 = torch.ops.aten.add.Tensor(getitem_203, getitem_206);  getitem_203 = getitem_206 = None
        alias_206 = torch.ops.aten.alias.default(relu_92);  relu_92 = None
        alias_207 = torch.ops.aten.alias.default(alias_206);  alias_206 = None
        le_3 = torch.ops.aten.le.Scalar(alias_207, 0);  alias_207 = None
        where_3 = torch.ops.aten.where.self(le_3, full_default, add_483);  le_3 = add_483 = None
        sum_8 = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
        sub_108 = torch.ops.aten.sub.Tensor(convolution_92, unsqueeze_425);  convolution_92 = unsqueeze_425 = None
        mul_706 = torch.ops.aten.mul.Tensor(where_3, sub_108)
        sum_9 = torch.ops.aten.sum.dim_IntList(mul_706, [0, 2, 3]);  mul_706 = None
        mul_707 = torch.ops.aten.mul.Tensor(sum_8, 0.00048828125)
        unsqueeze_426 = torch.ops.aten.unsqueeze.default(mul_707, 0);  mul_707 = None
        unsqueeze_427 = torch.ops.aten.unsqueeze.default(unsqueeze_426, 2);  unsqueeze_426 = None
        unsqueeze_428 = torch.ops.aten.unsqueeze.default(unsqueeze_427, 3);  unsqueeze_427 = None
        mul_708 = torch.ops.aten.mul.Tensor(sum_9, 0.00048828125)
        mul_709 = torch.ops.aten.mul.Tensor(squeeze_277, squeeze_277)
        mul_710 = torch.ops.aten.mul.Tensor(mul_708, mul_709);  mul_708 = mul_709 = None
        unsqueeze_429 = torch.ops.aten.unsqueeze.default(mul_710, 0);  mul_710 = None
        unsqueeze_430 = torch.ops.aten.unsqueeze.default(unsqueeze_429, 2);  unsqueeze_429 = None
        unsqueeze_431 = torch.ops.aten.unsqueeze.default(unsqueeze_430, 3);  unsqueeze_430 = None
        mul_711 = torch.ops.aten.mul.Tensor(squeeze_277, primals_280);  primals_280 = None
        unsqueeze_432 = torch.ops.aten.unsqueeze.default(mul_711, 0);  mul_711 = None
        unsqueeze_433 = torch.ops.aten.unsqueeze.default(unsqueeze_432, 2);  unsqueeze_432 = None
        unsqueeze_434 = torch.ops.aten.unsqueeze.default(unsqueeze_433, 3);  unsqueeze_433 = None
        mul_712 = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_431);  sub_108 = unsqueeze_431 = None
        sub_110 = torch.ops.aten.sub.Tensor(where_3, mul_712);  where_3 = mul_712 = None
        sub_111 = torch.ops.aten.sub.Tensor(sub_110, unsqueeze_428);  sub_110 = unsqueeze_428 = None
        mul_713 = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_434);  sub_111 = unsqueeze_434 = None
        mul_714 = torch.ops.aten.mul.Tensor(sum_9, squeeze_277);  sum_9 = squeeze_277 = None
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_713, relu_91, primals_279, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_713 = primals_279 = None
        getitem_209 = convolution_backward_3[0]
        getitem_210 = convolution_backward_3[1];  convolution_backward_3 = None
        alias_210 = torch.ops.aten.alias.default(relu_91);  relu_91 = None
        alias_211 = torch.ops.aten.alias.default(alias_210);  alias_210 = None
        le_4 = torch.ops.aten.le.Scalar(alias_211, 0);  alias_211 = None
        where_4 = torch.ops.aten.where.self(le_4, full_default, getitem_209);  le_4 = getitem_209 = None
        sum_10 = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
        sub_112 = torch.ops.aten.sub.Tensor(convolution_91, unsqueeze_437);  convolution_91 = unsqueeze_437 = None
        mul_715 = torch.ops.aten.mul.Tensor(where_4, sub_112)
        sum_11 = torch.ops.aten.sum.dim_IntList(mul_715, [0, 2, 3]);  mul_715 = None
        mul_716 = torch.ops.aten.mul.Tensor(sum_10, 0.00048828125)
        unsqueeze_438 = torch.ops.aten.unsqueeze.default(mul_716, 0);  mul_716 = None
        unsqueeze_439 = torch.ops.aten.unsqueeze.default(unsqueeze_438, 2);  unsqueeze_438 = None
        unsqueeze_440 = torch.ops.aten.unsqueeze.default(unsqueeze_439, 3);  unsqueeze_439 = None
        mul_717 = torch.ops.aten.mul.Tensor(sum_11, 0.00048828125)
        mul_718 = torch.ops.aten.mul.Tensor(squeeze_274, squeeze_274)
        mul_719 = torch.ops.aten.mul.Tensor(mul_717, mul_718);  mul_717 = mul_718 = None
        unsqueeze_441 = torch.ops.aten.unsqueeze.default(mul_719, 0);  mul_719 = None
        unsqueeze_442 = torch.ops.aten.unsqueeze.default(unsqueeze_441, 2);  unsqueeze_441 = None
        unsqueeze_443 = torch.ops.aten.unsqueeze.default(unsqueeze_442, 3);  unsqueeze_442 = None
        mul_720 = torch.ops.aten.mul.Tensor(squeeze_274, primals_277);  primals_277 = None
        unsqueeze_444 = torch.ops.aten.unsqueeze.default(mul_720, 0);  mul_720 = None
        unsqueeze_445 = torch.ops.aten.unsqueeze.default(unsqueeze_444, 2);  unsqueeze_444 = None
        unsqueeze_446 = torch.ops.aten.unsqueeze.default(unsqueeze_445, 3);  unsqueeze_445 = None
        mul_721 = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_443);  sub_112 = unsqueeze_443 = None
        sub_114 = torch.ops.aten.sub.Tensor(where_4, mul_721);  where_4 = mul_721 = None
        sub_115 = torch.ops.aten.sub.Tensor(sub_114, unsqueeze_440);  sub_114 = unsqueeze_440 = None
        mul_722 = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_446);  sub_115 = unsqueeze_446 = None
        mul_723 = torch.ops.aten.mul.Tensor(sum_11, squeeze_274);  sum_11 = squeeze_274 = None
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_722, cat_12, primals_276, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_722 = primals_276 = None
        getitem_212 = convolution_backward_4[0]
        getitem_213 = convolution_backward_4[1];  convolution_backward_4 = None
        add_484 = torch.ops.aten.add.Tensor(avg_pool2d_backward, getitem_212);  avg_pool2d_backward = getitem_212 = None
        slice_10 = torch.ops.aten.slice.Tensor(slice_5, 1, 0, 384)
        slice_11 = torch.ops.aten.slice.Tensor(slice_5, 1, 384, 768);  slice_5 = None
        where_5 = torch.ops.aten.where.self(le_5, full_default, slice_11);  le_5 = slice_11 = None
        sum_12 = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
        sub_116 = torch.ops.aten.sub.Tensor(convolution_90, unsqueeze_449);  convolution_90 = unsqueeze_449 = None
        mul_724 = torch.ops.aten.mul.Tensor(where_5, sub_116)
        sum_13 = torch.ops.aten.sum.dim_IntList(mul_724, [0, 2, 3]);  mul_724 = None
        mul_725 = torch.ops.aten.mul.Tensor(sum_12, 0.00048828125)
        unsqueeze_450 = torch.ops.aten.unsqueeze.default(mul_725, 0);  mul_725 = None
        unsqueeze_451 = torch.ops.aten.unsqueeze.default(unsqueeze_450, 2);  unsqueeze_450 = None
        unsqueeze_452 = torch.ops.aten.unsqueeze.default(unsqueeze_451, 3);  unsqueeze_451 = None
        mul_726 = torch.ops.aten.mul.Tensor(sum_13, 0.00048828125)
        mul_727 = torch.ops.aten.mul.Tensor(squeeze_271, squeeze_271)
        mul_728 = torch.ops.aten.mul.Tensor(mul_726, mul_727);  mul_726 = mul_727 = None
        unsqueeze_453 = torch.ops.aten.unsqueeze.default(mul_728, 0);  mul_728 = None
        unsqueeze_454 = torch.ops.aten.unsqueeze.default(unsqueeze_453, 2);  unsqueeze_453 = None
        unsqueeze_455 = torch.ops.aten.unsqueeze.default(unsqueeze_454, 3);  unsqueeze_454 = None
        mul_729 = torch.ops.aten.mul.Tensor(squeeze_271, primals_274);  primals_274 = None
        unsqueeze_456 = torch.ops.aten.unsqueeze.default(mul_729, 0);  mul_729 = None
        unsqueeze_457 = torch.ops.aten.unsqueeze.default(unsqueeze_456, 2);  unsqueeze_456 = None
        unsqueeze_458 = torch.ops.aten.unsqueeze.default(unsqueeze_457, 3);  unsqueeze_457 = None
        mul_730 = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_455);  sub_116 = unsqueeze_455 = None
        sub_118 = torch.ops.aten.sub.Tensor(where_5, mul_730);  where_5 = mul_730 = None
        sub_119 = torch.ops.aten.sub.Tensor(sub_118, unsqueeze_452);  sub_118 = unsqueeze_452 = None
        mul_731 = torch.ops.aten.mul.Tensor(sub_119, unsqueeze_458);  sub_119 = unsqueeze_458 = None
        mul_732 = torch.ops.aten.mul.Tensor(sum_13, squeeze_271);  sum_13 = squeeze_271 = None
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_731, relu_88, primals_273, [0], [1, 1], [1, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_731 = primals_273 = None
        getitem_215 = convolution_backward_5[0]
        getitem_216 = convolution_backward_5[1];  convolution_backward_5 = None
        where_6 = torch.ops.aten.where.self(le_6, full_default, slice_10);  le_6 = slice_10 = None
        sum_14 = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
        sub_120 = torch.ops.aten.sub.Tensor(convolution_89, unsqueeze_461);  convolution_89 = unsqueeze_461 = None
        mul_733 = torch.ops.aten.mul.Tensor(where_6, sub_120)
        sum_15 = torch.ops.aten.sum.dim_IntList(mul_733, [0, 2, 3]);  mul_733 = None
        mul_734 = torch.ops.aten.mul.Tensor(sum_14, 0.00048828125)
        unsqueeze_462 = torch.ops.aten.unsqueeze.default(mul_734, 0);  mul_734 = None
        unsqueeze_463 = torch.ops.aten.unsqueeze.default(unsqueeze_462, 2);  unsqueeze_462 = None
        unsqueeze_464 = torch.ops.aten.unsqueeze.default(unsqueeze_463, 3);  unsqueeze_463 = None
        mul_735 = torch.ops.aten.mul.Tensor(sum_15, 0.00048828125)
        mul_736 = torch.ops.aten.mul.Tensor(squeeze_268, squeeze_268)
        mul_737 = torch.ops.aten.mul.Tensor(mul_735, mul_736);  mul_735 = mul_736 = None
        unsqueeze_465 = torch.ops.aten.unsqueeze.default(mul_737, 0);  mul_737 = None
        unsqueeze_466 = torch.ops.aten.unsqueeze.default(unsqueeze_465, 2);  unsqueeze_465 = None
        unsqueeze_467 = torch.ops.aten.unsqueeze.default(unsqueeze_466, 3);  unsqueeze_466 = None
        mul_738 = torch.ops.aten.mul.Tensor(squeeze_268, primals_271);  primals_271 = None
        unsqueeze_468 = torch.ops.aten.unsqueeze.default(mul_738, 0);  mul_738 = None
        unsqueeze_469 = torch.ops.aten.unsqueeze.default(unsqueeze_468, 2);  unsqueeze_468 = None
        unsqueeze_470 = torch.ops.aten.unsqueeze.default(unsqueeze_469, 3);  unsqueeze_469 = None
        mul_739 = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_467);  sub_120 = unsqueeze_467 = None
        sub_122 = torch.ops.aten.sub.Tensor(where_6, mul_739);  where_6 = mul_739 = None
        sub_123 = torch.ops.aten.sub.Tensor(sub_122, unsqueeze_464);  sub_122 = unsqueeze_464 = None
        mul_740 = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_470);  sub_123 = unsqueeze_470 = None
        mul_741 = torch.ops.aten.mul.Tensor(sum_15, squeeze_268);  sum_15 = squeeze_268 = None
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_740, relu_88, primals_270, [0], [1, 1], [0, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_740 = primals_270 = None
        getitem_218 = convolution_backward_6[0]
        getitem_219 = convolution_backward_6[1];  convolution_backward_6 = None
        add_485 = torch.ops.aten.add.Tensor(getitem_215, getitem_218);  getitem_215 = getitem_218 = None
        alias_222 = torch.ops.aten.alias.default(relu_88);  relu_88 = None
        alias_223 = torch.ops.aten.alias.default(alias_222);  alias_222 = None
        le_7 = torch.ops.aten.le.Scalar(alias_223, 0);  alias_223 = None
        where_7 = torch.ops.aten.where.self(le_7, full_default, add_485);  le_7 = add_485 = None
        sum_16 = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
        sub_124 = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_473);  convolution_88 = unsqueeze_473 = None
        mul_742 = torch.ops.aten.mul.Tensor(where_7, sub_124)
        sum_17 = torch.ops.aten.sum.dim_IntList(mul_742, [0, 2, 3]);  mul_742 = None
        mul_743 = torch.ops.aten.mul.Tensor(sum_16, 0.00048828125)
        unsqueeze_474 = torch.ops.aten.unsqueeze.default(mul_743, 0);  mul_743 = None
        unsqueeze_475 = torch.ops.aten.unsqueeze.default(unsqueeze_474, 2);  unsqueeze_474 = None
        unsqueeze_476 = torch.ops.aten.unsqueeze.default(unsqueeze_475, 3);  unsqueeze_475 = None
        mul_744 = torch.ops.aten.mul.Tensor(sum_17, 0.00048828125)
        mul_745 = torch.ops.aten.mul.Tensor(squeeze_265, squeeze_265)
        mul_746 = torch.ops.aten.mul.Tensor(mul_744, mul_745);  mul_744 = mul_745 = None
        unsqueeze_477 = torch.ops.aten.unsqueeze.default(mul_746, 0);  mul_746 = None
        unsqueeze_478 = torch.ops.aten.unsqueeze.default(unsqueeze_477, 2);  unsqueeze_477 = None
        unsqueeze_479 = torch.ops.aten.unsqueeze.default(unsqueeze_478, 3);  unsqueeze_478 = None
        mul_747 = torch.ops.aten.mul.Tensor(squeeze_265, primals_268);  primals_268 = None
        unsqueeze_480 = torch.ops.aten.unsqueeze.default(mul_747, 0);  mul_747 = None
        unsqueeze_481 = torch.ops.aten.unsqueeze.default(unsqueeze_480, 2);  unsqueeze_480 = None
        unsqueeze_482 = torch.ops.aten.unsqueeze.default(unsqueeze_481, 3);  unsqueeze_481 = None
        mul_748 = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_479);  sub_124 = unsqueeze_479 = None
        sub_126 = torch.ops.aten.sub.Tensor(where_7, mul_748);  where_7 = mul_748 = None
        sub_127 = torch.ops.aten.sub.Tensor(sub_126, unsqueeze_476);  sub_126 = unsqueeze_476 = None
        mul_749 = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_482);  sub_127 = unsqueeze_482 = None
        mul_750 = torch.ops.aten.mul.Tensor(sum_17, squeeze_265);  sum_17 = squeeze_265 = None
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_749, cat_12, primals_267, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_749 = primals_267 = None
        getitem_221 = convolution_backward_7[0]
        getitem_222 = convolution_backward_7[1];  convolution_backward_7 = None
        add_486 = torch.ops.aten.add.Tensor(add_484, getitem_221);  add_484 = getitem_221 = None
        where_8 = torch.ops.aten.where.self(le_8, full_default, slice_4);  le_8 = slice_4 = None
        sum_18 = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
        sub_128 = torch.ops.aten.sub.Tensor(convolution_87, unsqueeze_485);  convolution_87 = unsqueeze_485 = None
        mul_751 = torch.ops.aten.mul.Tensor(where_8, sub_128)
        sum_19 = torch.ops.aten.sum.dim_IntList(mul_751, [0, 2, 3]);  mul_751 = None
        mul_752 = torch.ops.aten.mul.Tensor(sum_18, 0.00048828125)
        unsqueeze_486 = torch.ops.aten.unsqueeze.default(mul_752, 0);  mul_752 = None
        unsqueeze_487 = torch.ops.aten.unsqueeze.default(unsqueeze_486, 2);  unsqueeze_486 = None
        unsqueeze_488 = torch.ops.aten.unsqueeze.default(unsqueeze_487, 3);  unsqueeze_487 = None
        mul_753 = torch.ops.aten.mul.Tensor(sum_19, 0.00048828125)
        mul_754 = torch.ops.aten.mul.Tensor(squeeze_262, squeeze_262)
        mul_755 = torch.ops.aten.mul.Tensor(mul_753, mul_754);  mul_753 = mul_754 = None
        unsqueeze_489 = torch.ops.aten.unsqueeze.default(mul_755, 0);  mul_755 = None
        unsqueeze_490 = torch.ops.aten.unsqueeze.default(unsqueeze_489, 2);  unsqueeze_489 = None
        unsqueeze_491 = torch.ops.aten.unsqueeze.default(unsqueeze_490, 3);  unsqueeze_490 = None
        mul_756 = torch.ops.aten.mul.Tensor(squeeze_262, primals_265);  primals_265 = None
        unsqueeze_492 = torch.ops.aten.unsqueeze.default(mul_756, 0);  mul_756 = None
        unsqueeze_493 = torch.ops.aten.unsqueeze.default(unsqueeze_492, 2);  unsqueeze_492 = None
        unsqueeze_494 = torch.ops.aten.unsqueeze.default(unsqueeze_493, 3);  unsqueeze_493 = None
        mul_757 = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_491);  sub_128 = unsqueeze_491 = None
        sub_130 = torch.ops.aten.sub.Tensor(where_8, mul_757);  where_8 = mul_757 = None
        sub_131 = torch.ops.aten.sub.Tensor(sub_130, unsqueeze_488);  sub_130 = unsqueeze_488 = None
        mul_758 = torch.ops.aten.mul.Tensor(sub_131, unsqueeze_494);  sub_131 = unsqueeze_494 = None
        mul_759 = torch.ops.aten.mul.Tensor(sum_19, squeeze_262);  sum_19 = squeeze_262 = None
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_758, cat_12, primals_264, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_758 = cat_12 = primals_264 = None
        getitem_224 = convolution_backward_8[0]
        getitem_225 = convolution_backward_8[1];  convolution_backward_8 = None
        add_487 = torch.ops.aten.add.Tensor(add_486, getitem_224);  add_486 = getitem_224 = None
        slice_12 = torch.ops.aten.slice.Tensor(add_487, 1, 0, 320)
        slice_13 = torch.ops.aten.slice.Tensor(add_487, 1, 320, 1088)
        slice_14 = torch.ops.aten.slice.Tensor(add_487, 1, 1088, 1856)
        slice_15 = torch.ops.aten.slice.Tensor(add_487, 1, 1856, 2048);  add_487 = None
        where_9 = torch.ops.aten.where.self(le_9, full_default, slice_15);  le_9 = slice_15 = None
        sum_20 = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
        sub_132 = torch.ops.aten.sub.Tensor(convolution_86, unsqueeze_497);  convolution_86 = unsqueeze_497 = None
        mul_760 = torch.ops.aten.mul.Tensor(where_9, sub_132)
        sum_21 = torch.ops.aten.sum.dim_IntList(mul_760, [0, 2, 3]);  mul_760 = None
        mul_761 = torch.ops.aten.mul.Tensor(sum_20, 0.00048828125)
        unsqueeze_498 = torch.ops.aten.unsqueeze.default(mul_761, 0);  mul_761 = None
        unsqueeze_499 = torch.ops.aten.unsqueeze.default(unsqueeze_498, 2);  unsqueeze_498 = None
        unsqueeze_500 = torch.ops.aten.unsqueeze.default(unsqueeze_499, 3);  unsqueeze_499 = None
        mul_762 = torch.ops.aten.mul.Tensor(sum_21, 0.00048828125)
        mul_763 = torch.ops.aten.mul.Tensor(squeeze_259, squeeze_259)
        mul_764 = torch.ops.aten.mul.Tensor(mul_762, mul_763);  mul_762 = mul_763 = None
        unsqueeze_501 = torch.ops.aten.unsqueeze.default(mul_764, 0);  mul_764 = None
        unsqueeze_502 = torch.ops.aten.unsqueeze.default(unsqueeze_501, 2);  unsqueeze_501 = None
        unsqueeze_503 = torch.ops.aten.unsqueeze.default(unsqueeze_502, 3);  unsqueeze_502 = None
        mul_765 = torch.ops.aten.mul.Tensor(squeeze_259, primals_262);  primals_262 = None
        unsqueeze_504 = torch.ops.aten.unsqueeze.default(mul_765, 0);  mul_765 = None
        unsqueeze_505 = torch.ops.aten.unsqueeze.default(unsqueeze_504, 2);  unsqueeze_504 = None
        unsqueeze_506 = torch.ops.aten.unsqueeze.default(unsqueeze_505, 3);  unsqueeze_505 = None
        mul_766 = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_503);  sub_132 = unsqueeze_503 = None
        sub_134 = torch.ops.aten.sub.Tensor(where_9, mul_766);  where_9 = mul_766 = None
        sub_135 = torch.ops.aten.sub.Tensor(sub_134, unsqueeze_500);  sub_134 = unsqueeze_500 = None
        mul_767 = torch.ops.aten.mul.Tensor(sub_135, unsqueeze_506);  sub_135 = unsqueeze_506 = None
        mul_768 = torch.ops.aten.mul.Tensor(sum_21, squeeze_259);  sum_21 = squeeze_259 = None
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_767, avg_pool2d_8, primals_261, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_767 = avg_pool2d_8 = primals_261 = None
        getitem_227 = convolution_backward_9[0]
        getitem_228 = convolution_backward_9[1];  convolution_backward_9 = None
        avg_pool2d_backward_1 = torch.ops.aten.avg_pool2d_backward.default(getitem_227, cat_9, [3, 3], [1, 1], [1, 1], False, True, None);  getitem_227 = None
        slice_16 = torch.ops.aten.slice.Tensor(slice_14, 1, 0, 384)
        slice_17 = torch.ops.aten.slice.Tensor(slice_14, 1, 384, 768);  slice_14 = None
        where_10 = torch.ops.aten.where.self(le_10, full_default, slice_17);  le_10 = slice_17 = None
        sum_22 = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
        sub_136 = torch.ops.aten.sub.Tensor(convolution_85, unsqueeze_509);  convolution_85 = unsqueeze_509 = None
        mul_769 = torch.ops.aten.mul.Tensor(where_10, sub_136)
        sum_23 = torch.ops.aten.sum.dim_IntList(mul_769, [0, 2, 3]);  mul_769 = None
        mul_770 = torch.ops.aten.mul.Tensor(sum_22, 0.00048828125)
        unsqueeze_510 = torch.ops.aten.unsqueeze.default(mul_770, 0);  mul_770 = None
        unsqueeze_511 = torch.ops.aten.unsqueeze.default(unsqueeze_510, 2);  unsqueeze_510 = None
        unsqueeze_512 = torch.ops.aten.unsqueeze.default(unsqueeze_511, 3);  unsqueeze_511 = None
        mul_771 = torch.ops.aten.mul.Tensor(sum_23, 0.00048828125)
        mul_772 = torch.ops.aten.mul.Tensor(squeeze_256, squeeze_256)
        mul_773 = torch.ops.aten.mul.Tensor(mul_771, mul_772);  mul_771 = mul_772 = None
        unsqueeze_513 = torch.ops.aten.unsqueeze.default(mul_773, 0);  mul_773 = None
        unsqueeze_514 = torch.ops.aten.unsqueeze.default(unsqueeze_513, 2);  unsqueeze_513 = None
        unsqueeze_515 = torch.ops.aten.unsqueeze.default(unsqueeze_514, 3);  unsqueeze_514 = None
        mul_774 = torch.ops.aten.mul.Tensor(squeeze_256, primals_259);  primals_259 = None
        unsqueeze_516 = torch.ops.aten.unsqueeze.default(mul_774, 0);  mul_774 = None
        unsqueeze_517 = torch.ops.aten.unsqueeze.default(unsqueeze_516, 2);  unsqueeze_516 = None
        unsqueeze_518 = torch.ops.aten.unsqueeze.default(unsqueeze_517, 3);  unsqueeze_517 = None
        mul_775 = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_515);  sub_136 = unsqueeze_515 = None
        sub_138 = torch.ops.aten.sub.Tensor(where_10, mul_775);  where_10 = mul_775 = None
        sub_139 = torch.ops.aten.sub.Tensor(sub_138, unsqueeze_512);  sub_138 = unsqueeze_512 = None
        mul_776 = torch.ops.aten.mul.Tensor(sub_139, unsqueeze_518);  sub_139 = unsqueeze_518 = None
        mul_777 = torch.ops.aten.mul.Tensor(sum_23, squeeze_256);  sum_23 = squeeze_256 = None
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_776, relu_83, primals_258, [0], [1, 1], [1, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_776 = primals_258 = None
        getitem_230 = convolution_backward_10[0]
        getitem_231 = convolution_backward_10[1];  convolution_backward_10 = None
        where_11 = torch.ops.aten.where.self(le_11, full_default, slice_16);  le_11 = slice_16 = None
        sum_24 = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
        sub_140 = torch.ops.aten.sub.Tensor(convolution_84, unsqueeze_521);  convolution_84 = unsqueeze_521 = None
        mul_778 = torch.ops.aten.mul.Tensor(where_11, sub_140)
        sum_25 = torch.ops.aten.sum.dim_IntList(mul_778, [0, 2, 3]);  mul_778 = None
        mul_779 = torch.ops.aten.mul.Tensor(sum_24, 0.00048828125)
        unsqueeze_522 = torch.ops.aten.unsqueeze.default(mul_779, 0);  mul_779 = None
        unsqueeze_523 = torch.ops.aten.unsqueeze.default(unsqueeze_522, 2);  unsqueeze_522 = None
        unsqueeze_524 = torch.ops.aten.unsqueeze.default(unsqueeze_523, 3);  unsqueeze_523 = None
        mul_780 = torch.ops.aten.mul.Tensor(sum_25, 0.00048828125)
        mul_781 = torch.ops.aten.mul.Tensor(squeeze_253, squeeze_253)
        mul_782 = torch.ops.aten.mul.Tensor(mul_780, mul_781);  mul_780 = mul_781 = None
        unsqueeze_525 = torch.ops.aten.unsqueeze.default(mul_782, 0);  mul_782 = None
        unsqueeze_526 = torch.ops.aten.unsqueeze.default(unsqueeze_525, 2);  unsqueeze_525 = None
        unsqueeze_527 = torch.ops.aten.unsqueeze.default(unsqueeze_526, 3);  unsqueeze_526 = None
        mul_783 = torch.ops.aten.mul.Tensor(squeeze_253, primals_256);  primals_256 = None
        unsqueeze_528 = torch.ops.aten.unsqueeze.default(mul_783, 0);  mul_783 = None
        unsqueeze_529 = torch.ops.aten.unsqueeze.default(unsqueeze_528, 2);  unsqueeze_528 = None
        unsqueeze_530 = torch.ops.aten.unsqueeze.default(unsqueeze_529, 3);  unsqueeze_529 = None
        mul_784 = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_527);  sub_140 = unsqueeze_527 = None
        sub_142 = torch.ops.aten.sub.Tensor(where_11, mul_784);  where_11 = mul_784 = None
        sub_143 = torch.ops.aten.sub.Tensor(sub_142, unsqueeze_524);  sub_142 = unsqueeze_524 = None
        mul_785 = torch.ops.aten.mul.Tensor(sub_143, unsqueeze_530);  sub_143 = unsqueeze_530 = None
        mul_786 = torch.ops.aten.mul.Tensor(sum_25, squeeze_253);  sum_25 = squeeze_253 = None
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_785, relu_83, primals_255, [0], [1, 1], [0, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_785 = primals_255 = None
        getitem_233 = convolution_backward_11[0]
        getitem_234 = convolution_backward_11[1];  convolution_backward_11 = None
        add_488 = torch.ops.aten.add.Tensor(getitem_230, getitem_233);  getitem_230 = getitem_233 = None
        alias_242 = torch.ops.aten.alias.default(relu_83);  relu_83 = None
        alias_243 = torch.ops.aten.alias.default(alias_242);  alias_242 = None
        le_12 = torch.ops.aten.le.Scalar(alias_243, 0);  alias_243 = None
        where_12 = torch.ops.aten.where.self(le_12, full_default, add_488);  le_12 = add_488 = None
        sum_26 = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
        sub_144 = torch.ops.aten.sub.Tensor(convolution_83, unsqueeze_533);  convolution_83 = unsqueeze_533 = None
        mul_787 = torch.ops.aten.mul.Tensor(where_12, sub_144)
        sum_27 = torch.ops.aten.sum.dim_IntList(mul_787, [0, 2, 3]);  mul_787 = None
        mul_788 = torch.ops.aten.mul.Tensor(sum_26, 0.00048828125)
        unsqueeze_534 = torch.ops.aten.unsqueeze.default(mul_788, 0);  mul_788 = None
        unsqueeze_535 = torch.ops.aten.unsqueeze.default(unsqueeze_534, 2);  unsqueeze_534 = None
        unsqueeze_536 = torch.ops.aten.unsqueeze.default(unsqueeze_535, 3);  unsqueeze_535 = None
        mul_789 = torch.ops.aten.mul.Tensor(sum_27, 0.00048828125)
        mul_790 = torch.ops.aten.mul.Tensor(squeeze_250, squeeze_250)
        mul_791 = torch.ops.aten.mul.Tensor(mul_789, mul_790);  mul_789 = mul_790 = None
        unsqueeze_537 = torch.ops.aten.unsqueeze.default(mul_791, 0);  mul_791 = None
        unsqueeze_538 = torch.ops.aten.unsqueeze.default(unsqueeze_537, 2);  unsqueeze_537 = None
        unsqueeze_539 = torch.ops.aten.unsqueeze.default(unsqueeze_538, 3);  unsqueeze_538 = None
        mul_792 = torch.ops.aten.mul.Tensor(squeeze_250, primals_253);  primals_253 = None
        unsqueeze_540 = torch.ops.aten.unsqueeze.default(mul_792, 0);  mul_792 = None
        unsqueeze_541 = torch.ops.aten.unsqueeze.default(unsqueeze_540, 2);  unsqueeze_540 = None
        unsqueeze_542 = torch.ops.aten.unsqueeze.default(unsqueeze_541, 3);  unsqueeze_541 = None
        mul_793 = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_539);  sub_144 = unsqueeze_539 = None
        sub_146 = torch.ops.aten.sub.Tensor(where_12, mul_793);  where_12 = mul_793 = None
        sub_147 = torch.ops.aten.sub.Tensor(sub_146, unsqueeze_536);  sub_146 = unsqueeze_536 = None
        mul_794 = torch.ops.aten.mul.Tensor(sub_147, unsqueeze_542);  sub_147 = unsqueeze_542 = None
        mul_795 = torch.ops.aten.mul.Tensor(sum_27, squeeze_250);  sum_27 = squeeze_250 = None
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_794, relu_82, primals_252, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_794 = primals_252 = None
        getitem_236 = convolution_backward_12[0]
        getitem_237 = convolution_backward_12[1];  convolution_backward_12 = None
        alias_246 = torch.ops.aten.alias.default(relu_82);  relu_82 = None
        alias_247 = torch.ops.aten.alias.default(alias_246);  alias_246 = None
        le_13 = torch.ops.aten.le.Scalar(alias_247, 0);  alias_247 = None
        where_13 = torch.ops.aten.where.self(le_13, full_default, getitem_236);  le_13 = getitem_236 = None
        sum_28 = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
        sub_148 = torch.ops.aten.sub.Tensor(convolution_82, unsqueeze_545);  convolution_82 = unsqueeze_545 = None
        mul_796 = torch.ops.aten.mul.Tensor(where_13, sub_148)
        sum_29 = torch.ops.aten.sum.dim_IntList(mul_796, [0, 2, 3]);  mul_796 = None
        mul_797 = torch.ops.aten.mul.Tensor(sum_28, 0.00048828125)
        unsqueeze_546 = torch.ops.aten.unsqueeze.default(mul_797, 0);  mul_797 = None
        unsqueeze_547 = torch.ops.aten.unsqueeze.default(unsqueeze_546, 2);  unsqueeze_546 = None
        unsqueeze_548 = torch.ops.aten.unsqueeze.default(unsqueeze_547, 3);  unsqueeze_547 = None
        mul_798 = torch.ops.aten.mul.Tensor(sum_29, 0.00048828125)
        mul_799 = torch.ops.aten.mul.Tensor(squeeze_247, squeeze_247)
        mul_800 = torch.ops.aten.mul.Tensor(mul_798, mul_799);  mul_798 = mul_799 = None
        unsqueeze_549 = torch.ops.aten.unsqueeze.default(mul_800, 0);  mul_800 = None
        unsqueeze_550 = torch.ops.aten.unsqueeze.default(unsqueeze_549, 2);  unsqueeze_549 = None
        unsqueeze_551 = torch.ops.aten.unsqueeze.default(unsqueeze_550, 3);  unsqueeze_550 = None
        mul_801 = torch.ops.aten.mul.Tensor(squeeze_247, primals_250);  primals_250 = None
        unsqueeze_552 = torch.ops.aten.unsqueeze.default(mul_801, 0);  mul_801 = None
        unsqueeze_553 = torch.ops.aten.unsqueeze.default(unsqueeze_552, 2);  unsqueeze_552 = None
        unsqueeze_554 = torch.ops.aten.unsqueeze.default(unsqueeze_553, 3);  unsqueeze_553 = None
        mul_802 = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_551);  sub_148 = unsqueeze_551 = None
        sub_150 = torch.ops.aten.sub.Tensor(where_13, mul_802);  where_13 = mul_802 = None
        sub_151 = torch.ops.aten.sub.Tensor(sub_150, unsqueeze_548);  sub_150 = unsqueeze_548 = None
        mul_803 = torch.ops.aten.mul.Tensor(sub_151, unsqueeze_554);  sub_151 = unsqueeze_554 = None
        mul_804 = torch.ops.aten.mul.Tensor(sum_29, squeeze_247);  sum_29 = squeeze_247 = None
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_803, cat_9, primals_249, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_803 = primals_249 = None
        getitem_239 = convolution_backward_13[0]
        getitem_240 = convolution_backward_13[1];  convolution_backward_13 = None
        add_489 = torch.ops.aten.add.Tensor(avg_pool2d_backward_1, getitem_239);  avg_pool2d_backward_1 = getitem_239 = None
        slice_18 = torch.ops.aten.slice.Tensor(slice_13, 1, 0, 384)
        slice_19 = torch.ops.aten.slice.Tensor(slice_13, 1, 384, 768);  slice_13 = None
        where_14 = torch.ops.aten.where.self(le_14, full_default, slice_19);  le_14 = slice_19 = None
        sum_30 = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
        sub_152 = torch.ops.aten.sub.Tensor(convolution_81, unsqueeze_557);  convolution_81 = unsqueeze_557 = None
        mul_805 = torch.ops.aten.mul.Tensor(where_14, sub_152)
        sum_31 = torch.ops.aten.sum.dim_IntList(mul_805, [0, 2, 3]);  mul_805 = None
        mul_806 = torch.ops.aten.mul.Tensor(sum_30, 0.00048828125)
        unsqueeze_558 = torch.ops.aten.unsqueeze.default(mul_806, 0);  mul_806 = None
        unsqueeze_559 = torch.ops.aten.unsqueeze.default(unsqueeze_558, 2);  unsqueeze_558 = None
        unsqueeze_560 = torch.ops.aten.unsqueeze.default(unsqueeze_559, 3);  unsqueeze_559 = None
        mul_807 = torch.ops.aten.mul.Tensor(sum_31, 0.00048828125)
        mul_808 = torch.ops.aten.mul.Tensor(squeeze_244, squeeze_244)
        mul_809 = torch.ops.aten.mul.Tensor(mul_807, mul_808);  mul_807 = mul_808 = None
        unsqueeze_561 = torch.ops.aten.unsqueeze.default(mul_809, 0);  mul_809 = None
        unsqueeze_562 = torch.ops.aten.unsqueeze.default(unsqueeze_561, 2);  unsqueeze_561 = None
        unsqueeze_563 = torch.ops.aten.unsqueeze.default(unsqueeze_562, 3);  unsqueeze_562 = None
        mul_810 = torch.ops.aten.mul.Tensor(squeeze_244, primals_247);  primals_247 = None
        unsqueeze_564 = torch.ops.aten.unsqueeze.default(mul_810, 0);  mul_810 = None
        unsqueeze_565 = torch.ops.aten.unsqueeze.default(unsqueeze_564, 2);  unsqueeze_564 = None
        unsqueeze_566 = torch.ops.aten.unsqueeze.default(unsqueeze_565, 3);  unsqueeze_565 = None
        mul_811 = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_563);  sub_152 = unsqueeze_563 = None
        sub_154 = torch.ops.aten.sub.Tensor(where_14, mul_811);  where_14 = mul_811 = None
        sub_155 = torch.ops.aten.sub.Tensor(sub_154, unsqueeze_560);  sub_154 = unsqueeze_560 = None
        mul_812 = torch.ops.aten.mul.Tensor(sub_155, unsqueeze_566);  sub_155 = unsqueeze_566 = None
        mul_813 = torch.ops.aten.mul.Tensor(sum_31, squeeze_244);  sum_31 = squeeze_244 = None
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_812, relu_79, primals_246, [0], [1, 1], [1, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_812 = primals_246 = None
        getitem_242 = convolution_backward_14[0]
        getitem_243 = convolution_backward_14[1];  convolution_backward_14 = None
        where_15 = torch.ops.aten.where.self(le_15, full_default, slice_18);  le_15 = slice_18 = None
        sum_32 = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
        sub_156 = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_569);  convolution_80 = unsqueeze_569 = None
        mul_814 = torch.ops.aten.mul.Tensor(where_15, sub_156)
        sum_33 = torch.ops.aten.sum.dim_IntList(mul_814, [0, 2, 3]);  mul_814 = None
        mul_815 = torch.ops.aten.mul.Tensor(sum_32, 0.00048828125)
        unsqueeze_570 = torch.ops.aten.unsqueeze.default(mul_815, 0);  mul_815 = None
        unsqueeze_571 = torch.ops.aten.unsqueeze.default(unsqueeze_570, 2);  unsqueeze_570 = None
        unsqueeze_572 = torch.ops.aten.unsqueeze.default(unsqueeze_571, 3);  unsqueeze_571 = None
        mul_816 = torch.ops.aten.mul.Tensor(sum_33, 0.00048828125)
        mul_817 = torch.ops.aten.mul.Tensor(squeeze_241, squeeze_241)
        mul_818 = torch.ops.aten.mul.Tensor(mul_816, mul_817);  mul_816 = mul_817 = None
        unsqueeze_573 = torch.ops.aten.unsqueeze.default(mul_818, 0);  mul_818 = None
        unsqueeze_574 = torch.ops.aten.unsqueeze.default(unsqueeze_573, 2);  unsqueeze_573 = None
        unsqueeze_575 = torch.ops.aten.unsqueeze.default(unsqueeze_574, 3);  unsqueeze_574 = None
        mul_819 = torch.ops.aten.mul.Tensor(squeeze_241, primals_244);  primals_244 = None
        unsqueeze_576 = torch.ops.aten.unsqueeze.default(mul_819, 0);  mul_819 = None
        unsqueeze_577 = torch.ops.aten.unsqueeze.default(unsqueeze_576, 2);  unsqueeze_576 = None
        unsqueeze_578 = torch.ops.aten.unsqueeze.default(unsqueeze_577, 3);  unsqueeze_577 = None
        mul_820 = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_575);  sub_156 = unsqueeze_575 = None
        sub_158 = torch.ops.aten.sub.Tensor(where_15, mul_820);  where_15 = mul_820 = None
        sub_159 = torch.ops.aten.sub.Tensor(sub_158, unsqueeze_572);  sub_158 = unsqueeze_572 = None
        mul_821 = torch.ops.aten.mul.Tensor(sub_159, unsqueeze_578);  sub_159 = unsqueeze_578 = None
        mul_822 = torch.ops.aten.mul.Tensor(sum_33, squeeze_241);  sum_33 = squeeze_241 = None
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_821, relu_79, primals_243, [0], [1, 1], [0, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_821 = primals_243 = None
        getitem_245 = convolution_backward_15[0]
        getitem_246 = convolution_backward_15[1];  convolution_backward_15 = None
        add_490 = torch.ops.aten.add.Tensor(getitem_242, getitem_245);  getitem_242 = getitem_245 = None
        alias_258 = torch.ops.aten.alias.default(relu_79);  relu_79 = None
        alias_259 = torch.ops.aten.alias.default(alias_258);  alias_258 = None
        le_16 = torch.ops.aten.le.Scalar(alias_259, 0);  alias_259 = None
        where_16 = torch.ops.aten.where.self(le_16, full_default, add_490);  le_16 = add_490 = None
        sum_34 = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
        sub_160 = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_581);  convolution_79 = unsqueeze_581 = None
        mul_823 = torch.ops.aten.mul.Tensor(where_16, sub_160)
        sum_35 = torch.ops.aten.sum.dim_IntList(mul_823, [0, 2, 3]);  mul_823 = None
        mul_824 = torch.ops.aten.mul.Tensor(sum_34, 0.00048828125)
        unsqueeze_582 = torch.ops.aten.unsqueeze.default(mul_824, 0);  mul_824 = None
        unsqueeze_583 = torch.ops.aten.unsqueeze.default(unsqueeze_582, 2);  unsqueeze_582 = None
        unsqueeze_584 = torch.ops.aten.unsqueeze.default(unsqueeze_583, 3);  unsqueeze_583 = None
        mul_825 = torch.ops.aten.mul.Tensor(sum_35, 0.00048828125)
        mul_826 = torch.ops.aten.mul.Tensor(squeeze_238, squeeze_238)
        mul_827 = torch.ops.aten.mul.Tensor(mul_825, mul_826);  mul_825 = mul_826 = None
        unsqueeze_585 = torch.ops.aten.unsqueeze.default(mul_827, 0);  mul_827 = None
        unsqueeze_586 = torch.ops.aten.unsqueeze.default(unsqueeze_585, 2);  unsqueeze_585 = None
        unsqueeze_587 = torch.ops.aten.unsqueeze.default(unsqueeze_586, 3);  unsqueeze_586 = None
        mul_828 = torch.ops.aten.mul.Tensor(squeeze_238, primals_241);  primals_241 = None
        unsqueeze_588 = torch.ops.aten.unsqueeze.default(mul_828, 0);  mul_828 = None
        unsqueeze_589 = torch.ops.aten.unsqueeze.default(unsqueeze_588, 2);  unsqueeze_588 = None
        unsqueeze_590 = torch.ops.aten.unsqueeze.default(unsqueeze_589, 3);  unsqueeze_589 = None
        mul_829 = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_587);  sub_160 = unsqueeze_587 = None
        sub_162 = torch.ops.aten.sub.Tensor(where_16, mul_829);  where_16 = mul_829 = None
        sub_163 = torch.ops.aten.sub.Tensor(sub_162, unsqueeze_584);  sub_162 = unsqueeze_584 = None
        mul_830 = torch.ops.aten.mul.Tensor(sub_163, unsqueeze_590);  sub_163 = unsqueeze_590 = None
        mul_831 = torch.ops.aten.mul.Tensor(sum_35, squeeze_238);  sum_35 = squeeze_238 = None
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_830, cat_9, primals_240, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_830 = primals_240 = None
        getitem_248 = convolution_backward_16[0]
        getitem_249 = convolution_backward_16[1];  convolution_backward_16 = None
        add_491 = torch.ops.aten.add.Tensor(add_489, getitem_248);  add_489 = getitem_248 = None
        where_17 = torch.ops.aten.where.self(le_17, full_default, slice_12);  le_17 = slice_12 = None
        sum_36 = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
        sub_164 = torch.ops.aten.sub.Tensor(convolution_78, unsqueeze_593);  convolution_78 = unsqueeze_593 = None
        mul_832 = torch.ops.aten.mul.Tensor(where_17, sub_164)
        sum_37 = torch.ops.aten.sum.dim_IntList(mul_832, [0, 2, 3]);  mul_832 = None
        mul_833 = torch.ops.aten.mul.Tensor(sum_36, 0.00048828125)
        unsqueeze_594 = torch.ops.aten.unsqueeze.default(mul_833, 0);  mul_833 = None
        unsqueeze_595 = torch.ops.aten.unsqueeze.default(unsqueeze_594, 2);  unsqueeze_594 = None
        unsqueeze_596 = torch.ops.aten.unsqueeze.default(unsqueeze_595, 3);  unsqueeze_595 = None
        mul_834 = torch.ops.aten.mul.Tensor(sum_37, 0.00048828125)
        mul_835 = torch.ops.aten.mul.Tensor(squeeze_235, squeeze_235)
        mul_836 = torch.ops.aten.mul.Tensor(mul_834, mul_835);  mul_834 = mul_835 = None
        unsqueeze_597 = torch.ops.aten.unsqueeze.default(mul_836, 0);  mul_836 = None
        unsqueeze_598 = torch.ops.aten.unsqueeze.default(unsqueeze_597, 2);  unsqueeze_597 = None
        unsqueeze_599 = torch.ops.aten.unsqueeze.default(unsqueeze_598, 3);  unsqueeze_598 = None
        mul_837 = torch.ops.aten.mul.Tensor(squeeze_235, primals_238);  primals_238 = None
        unsqueeze_600 = torch.ops.aten.unsqueeze.default(mul_837, 0);  mul_837 = None
        unsqueeze_601 = torch.ops.aten.unsqueeze.default(unsqueeze_600, 2);  unsqueeze_600 = None
        unsqueeze_602 = torch.ops.aten.unsqueeze.default(unsqueeze_601, 3);  unsqueeze_601 = None
        mul_838 = torch.ops.aten.mul.Tensor(sub_164, unsqueeze_599);  sub_164 = unsqueeze_599 = None
        sub_166 = torch.ops.aten.sub.Tensor(where_17, mul_838);  where_17 = mul_838 = None
        sub_167 = torch.ops.aten.sub.Tensor(sub_166, unsqueeze_596);  sub_166 = unsqueeze_596 = None
        mul_839 = torch.ops.aten.mul.Tensor(sub_167, unsqueeze_602);  sub_167 = unsqueeze_602 = None
        mul_840 = torch.ops.aten.mul.Tensor(sum_37, squeeze_235);  sum_37 = squeeze_235 = None
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_839, cat_9, primals_237, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_839 = cat_9 = primals_237 = None
        getitem_251 = convolution_backward_17[0]
        getitem_252 = convolution_backward_17[1];  convolution_backward_17 = None
        add_492 = torch.ops.aten.add.Tensor(add_491, getitem_251);  add_491 = getitem_251 = None
        slice_20 = torch.ops.aten.slice.Tensor(add_492, 1, 0, 320)
        slice_21 = torch.ops.aten.slice.Tensor(add_492, 1, 320, 512)
        slice_22 = torch.ops.aten.slice.Tensor(add_492, 1, 512, 1280);  add_492 = None
        max_pool2d_with_indices_backward = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_22, cat_8, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_163);  slice_22 = getitem_163 = None
        where_18 = torch.ops.aten.where.self(le_18, full_default, slice_21);  le_18 = slice_21 = None
        sum_38 = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
        sub_168 = torch.ops.aten.sub.Tensor(convolution_77, unsqueeze_605);  convolution_77 = unsqueeze_605 = None
        mul_841 = torch.ops.aten.mul.Tensor(where_18, sub_168)
        sum_39 = torch.ops.aten.sum.dim_IntList(mul_841, [0, 2, 3]);  mul_841 = None
        mul_842 = torch.ops.aten.mul.Tensor(sum_38, 0.00048828125)
        unsqueeze_606 = torch.ops.aten.unsqueeze.default(mul_842, 0);  mul_842 = None
        unsqueeze_607 = torch.ops.aten.unsqueeze.default(unsqueeze_606, 2);  unsqueeze_606 = None
        unsqueeze_608 = torch.ops.aten.unsqueeze.default(unsqueeze_607, 3);  unsqueeze_607 = None
        mul_843 = torch.ops.aten.mul.Tensor(sum_39, 0.00048828125)
        mul_844 = torch.ops.aten.mul.Tensor(squeeze_232, squeeze_232)
        mul_845 = torch.ops.aten.mul.Tensor(mul_843, mul_844);  mul_843 = mul_844 = None
        unsqueeze_609 = torch.ops.aten.unsqueeze.default(mul_845, 0);  mul_845 = None
        unsqueeze_610 = torch.ops.aten.unsqueeze.default(unsqueeze_609, 2);  unsqueeze_609 = None
        unsqueeze_611 = torch.ops.aten.unsqueeze.default(unsqueeze_610, 3);  unsqueeze_610 = None
        mul_846 = torch.ops.aten.mul.Tensor(squeeze_232, primals_235);  primals_235 = None
        unsqueeze_612 = torch.ops.aten.unsqueeze.default(mul_846, 0);  mul_846 = None
        unsqueeze_613 = torch.ops.aten.unsqueeze.default(unsqueeze_612, 2);  unsqueeze_612 = None
        unsqueeze_614 = torch.ops.aten.unsqueeze.default(unsqueeze_613, 3);  unsqueeze_613 = None
        mul_847 = torch.ops.aten.mul.Tensor(sub_168, unsqueeze_611);  sub_168 = unsqueeze_611 = None
        sub_170 = torch.ops.aten.sub.Tensor(where_18, mul_847);  where_18 = mul_847 = None
        sub_171 = torch.ops.aten.sub.Tensor(sub_170, unsqueeze_608);  sub_170 = unsqueeze_608 = None
        mul_848 = torch.ops.aten.mul.Tensor(sub_171, unsqueeze_614);  sub_171 = unsqueeze_614 = None
        mul_849 = torch.ops.aten.mul.Tensor(sum_39, squeeze_232);  sum_39 = squeeze_232 = None
        convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_848, relu_76, primals_234, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_848 = primals_234 = None
        getitem_254 = convolution_backward_18[0]
        getitem_255 = convolution_backward_18[1];  convolution_backward_18 = None
        alias_270 = torch.ops.aten.alias.default(relu_76);  relu_76 = None
        alias_271 = torch.ops.aten.alias.default(alias_270);  alias_270 = None
        le_19 = torch.ops.aten.le.Scalar(alias_271, 0);  alias_271 = None
        where_19 = torch.ops.aten.where.self(le_19, full_default, getitem_254);  le_19 = getitem_254 = None
        sum_40 = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
        sub_172 = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_617);  convolution_76 = unsqueeze_617 = None
        mul_850 = torch.ops.aten.mul.Tensor(where_19, sub_172)
        sum_41 = torch.ops.aten.sum.dim_IntList(mul_850, [0, 2, 3]);  mul_850 = None
        mul_851 = torch.ops.aten.mul.Tensor(sum_40, 0.00010813148788927336)
        unsqueeze_618 = torch.ops.aten.unsqueeze.default(mul_851, 0);  mul_851 = None
        unsqueeze_619 = torch.ops.aten.unsqueeze.default(unsqueeze_618, 2);  unsqueeze_618 = None
        unsqueeze_620 = torch.ops.aten.unsqueeze.default(unsqueeze_619, 3);  unsqueeze_619 = None
        mul_852 = torch.ops.aten.mul.Tensor(sum_41, 0.00010813148788927336)
        mul_853 = torch.ops.aten.mul.Tensor(squeeze_229, squeeze_229)
        mul_854 = torch.ops.aten.mul.Tensor(mul_852, mul_853);  mul_852 = mul_853 = None
        unsqueeze_621 = torch.ops.aten.unsqueeze.default(mul_854, 0);  mul_854 = None
        unsqueeze_622 = torch.ops.aten.unsqueeze.default(unsqueeze_621, 2);  unsqueeze_621 = None
        unsqueeze_623 = torch.ops.aten.unsqueeze.default(unsqueeze_622, 3);  unsqueeze_622 = None
        mul_855 = torch.ops.aten.mul.Tensor(squeeze_229, primals_232);  primals_232 = None
        unsqueeze_624 = torch.ops.aten.unsqueeze.default(mul_855, 0);  mul_855 = None
        unsqueeze_625 = torch.ops.aten.unsqueeze.default(unsqueeze_624, 2);  unsqueeze_624 = None
        unsqueeze_626 = torch.ops.aten.unsqueeze.default(unsqueeze_625, 3);  unsqueeze_625 = None
        mul_856 = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_623);  sub_172 = unsqueeze_623 = None
        sub_174 = torch.ops.aten.sub.Tensor(where_19, mul_856);  where_19 = mul_856 = None
        sub_175 = torch.ops.aten.sub.Tensor(sub_174, unsqueeze_620);  sub_174 = unsqueeze_620 = None
        mul_857 = torch.ops.aten.mul.Tensor(sub_175, unsqueeze_626);  sub_175 = unsqueeze_626 = None
        mul_858 = torch.ops.aten.mul.Tensor(sum_41, squeeze_229);  sum_41 = squeeze_229 = None
        convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_857, relu_75, primals_231, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_857 = primals_231 = None
        getitem_257 = convolution_backward_19[0]
        getitem_258 = convolution_backward_19[1];  convolution_backward_19 = None
        alias_274 = torch.ops.aten.alias.default(relu_75);  relu_75 = None
        alias_275 = torch.ops.aten.alias.default(alias_274);  alias_274 = None
        le_20 = torch.ops.aten.le.Scalar(alias_275, 0);  alias_275 = None
        where_20 = torch.ops.aten.where.self(le_20, full_default, getitem_257);  le_20 = getitem_257 = None
        sum_42 = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
        sub_176 = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_629);  convolution_75 = unsqueeze_629 = None
        mul_859 = torch.ops.aten.mul.Tensor(where_20, sub_176)
        sum_43 = torch.ops.aten.sum.dim_IntList(mul_859, [0, 2, 3]);  mul_859 = None
        mul_860 = torch.ops.aten.mul.Tensor(sum_42, 0.00010813148788927336)
        unsqueeze_630 = torch.ops.aten.unsqueeze.default(mul_860, 0);  mul_860 = None
        unsqueeze_631 = torch.ops.aten.unsqueeze.default(unsqueeze_630, 2);  unsqueeze_630 = None
        unsqueeze_632 = torch.ops.aten.unsqueeze.default(unsqueeze_631, 3);  unsqueeze_631 = None
        mul_861 = torch.ops.aten.mul.Tensor(sum_43, 0.00010813148788927336)
        mul_862 = torch.ops.aten.mul.Tensor(squeeze_226, squeeze_226)
        mul_863 = torch.ops.aten.mul.Tensor(mul_861, mul_862);  mul_861 = mul_862 = None
        unsqueeze_633 = torch.ops.aten.unsqueeze.default(mul_863, 0);  mul_863 = None
        unsqueeze_634 = torch.ops.aten.unsqueeze.default(unsqueeze_633, 2);  unsqueeze_633 = None
        unsqueeze_635 = torch.ops.aten.unsqueeze.default(unsqueeze_634, 3);  unsqueeze_634 = None
        mul_864 = torch.ops.aten.mul.Tensor(squeeze_226, primals_229);  primals_229 = None
        unsqueeze_636 = torch.ops.aten.unsqueeze.default(mul_864, 0);  mul_864 = None
        unsqueeze_637 = torch.ops.aten.unsqueeze.default(unsqueeze_636, 2);  unsqueeze_636 = None
        unsqueeze_638 = torch.ops.aten.unsqueeze.default(unsqueeze_637, 3);  unsqueeze_637 = None
        mul_865 = torch.ops.aten.mul.Tensor(sub_176, unsqueeze_635);  sub_176 = unsqueeze_635 = None
        sub_178 = torch.ops.aten.sub.Tensor(where_20, mul_865);  where_20 = mul_865 = None
        sub_179 = torch.ops.aten.sub.Tensor(sub_178, unsqueeze_632);  sub_178 = unsqueeze_632 = None
        mul_866 = torch.ops.aten.mul.Tensor(sub_179, unsqueeze_638);  sub_179 = unsqueeze_638 = None
        mul_867 = torch.ops.aten.mul.Tensor(sum_43, squeeze_226);  sum_43 = squeeze_226 = None
        convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_866, relu_74, primals_228, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_866 = primals_228 = None
        getitem_260 = convolution_backward_20[0]
        getitem_261 = convolution_backward_20[1];  convolution_backward_20 = None
        alias_278 = torch.ops.aten.alias.default(relu_74);  relu_74 = None
        alias_279 = torch.ops.aten.alias.default(alias_278);  alias_278 = None
        le_21 = torch.ops.aten.le.Scalar(alias_279, 0);  alias_279 = None
        where_21 = torch.ops.aten.where.self(le_21, full_default, getitem_260);  le_21 = getitem_260 = None
        sum_44 = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
        sub_180 = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_641);  convolution_74 = unsqueeze_641 = None
        mul_868 = torch.ops.aten.mul.Tensor(where_21, sub_180)
        sum_45 = torch.ops.aten.sum.dim_IntList(mul_868, [0, 2, 3]);  mul_868 = None
        mul_869 = torch.ops.aten.mul.Tensor(sum_44, 0.00010813148788927336)
        unsqueeze_642 = torch.ops.aten.unsqueeze.default(mul_869, 0);  mul_869 = None
        unsqueeze_643 = torch.ops.aten.unsqueeze.default(unsqueeze_642, 2);  unsqueeze_642 = None
        unsqueeze_644 = torch.ops.aten.unsqueeze.default(unsqueeze_643, 3);  unsqueeze_643 = None
        mul_870 = torch.ops.aten.mul.Tensor(sum_45, 0.00010813148788927336)
        mul_871 = torch.ops.aten.mul.Tensor(squeeze_223, squeeze_223)
        mul_872 = torch.ops.aten.mul.Tensor(mul_870, mul_871);  mul_870 = mul_871 = None
        unsqueeze_645 = torch.ops.aten.unsqueeze.default(mul_872, 0);  mul_872 = None
        unsqueeze_646 = torch.ops.aten.unsqueeze.default(unsqueeze_645, 2);  unsqueeze_645 = None
        unsqueeze_647 = torch.ops.aten.unsqueeze.default(unsqueeze_646, 3);  unsqueeze_646 = None
        mul_873 = torch.ops.aten.mul.Tensor(squeeze_223, primals_226);  primals_226 = None
        unsqueeze_648 = torch.ops.aten.unsqueeze.default(mul_873, 0);  mul_873 = None
        unsqueeze_649 = torch.ops.aten.unsqueeze.default(unsqueeze_648, 2);  unsqueeze_648 = None
        unsqueeze_650 = torch.ops.aten.unsqueeze.default(unsqueeze_649, 3);  unsqueeze_649 = None
        mul_874 = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_647);  sub_180 = unsqueeze_647 = None
        sub_182 = torch.ops.aten.sub.Tensor(where_21, mul_874);  where_21 = mul_874 = None
        sub_183 = torch.ops.aten.sub.Tensor(sub_182, unsqueeze_644);  sub_182 = unsqueeze_644 = None
        mul_875 = torch.ops.aten.mul.Tensor(sub_183, unsqueeze_650);  sub_183 = unsqueeze_650 = None
        mul_876 = torch.ops.aten.mul.Tensor(sum_45, squeeze_223);  sum_45 = squeeze_223 = None
        convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_875, cat_8, primals_225, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_875 = primals_225 = None
        getitem_263 = convolution_backward_21[0]
        getitem_264 = convolution_backward_21[1];  convolution_backward_21 = None
        add_493 = torch.ops.aten.add.Tensor(max_pool2d_with_indices_backward, getitem_263);  max_pool2d_with_indices_backward = getitem_263 = None
        where_22 = torch.ops.aten.where.self(le_22, full_default, slice_20);  le_22 = slice_20 = None
        sum_46 = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
        sub_184 = torch.ops.aten.sub.Tensor(convolution_73, unsqueeze_653);  convolution_73 = unsqueeze_653 = None
        mul_877 = torch.ops.aten.mul.Tensor(where_22, sub_184)
        sum_47 = torch.ops.aten.sum.dim_IntList(mul_877, [0, 2, 3]);  mul_877 = None
        mul_878 = torch.ops.aten.mul.Tensor(sum_46, 0.00048828125)
        unsqueeze_654 = torch.ops.aten.unsqueeze.default(mul_878, 0);  mul_878 = None
        unsqueeze_655 = torch.ops.aten.unsqueeze.default(unsqueeze_654, 2);  unsqueeze_654 = None
        unsqueeze_656 = torch.ops.aten.unsqueeze.default(unsqueeze_655, 3);  unsqueeze_655 = None
        mul_879 = torch.ops.aten.mul.Tensor(sum_47, 0.00048828125)
        mul_880 = torch.ops.aten.mul.Tensor(squeeze_220, squeeze_220)
        mul_881 = torch.ops.aten.mul.Tensor(mul_879, mul_880);  mul_879 = mul_880 = None
        unsqueeze_657 = torch.ops.aten.unsqueeze.default(mul_881, 0);  mul_881 = None
        unsqueeze_658 = torch.ops.aten.unsqueeze.default(unsqueeze_657, 2);  unsqueeze_657 = None
        unsqueeze_659 = torch.ops.aten.unsqueeze.default(unsqueeze_658, 3);  unsqueeze_658 = None
        mul_882 = torch.ops.aten.mul.Tensor(squeeze_220, primals_223);  primals_223 = None
        unsqueeze_660 = torch.ops.aten.unsqueeze.default(mul_882, 0);  mul_882 = None
        unsqueeze_661 = torch.ops.aten.unsqueeze.default(unsqueeze_660, 2);  unsqueeze_660 = None
        unsqueeze_662 = torch.ops.aten.unsqueeze.default(unsqueeze_661, 3);  unsqueeze_661 = None
        mul_883 = torch.ops.aten.mul.Tensor(sub_184, unsqueeze_659);  sub_184 = unsqueeze_659 = None
        sub_186 = torch.ops.aten.sub.Tensor(where_22, mul_883);  where_22 = mul_883 = None
        sub_187 = torch.ops.aten.sub.Tensor(sub_186, unsqueeze_656);  sub_186 = unsqueeze_656 = None
        mul_884 = torch.ops.aten.mul.Tensor(sub_187, unsqueeze_662);  sub_187 = unsqueeze_662 = None
        mul_885 = torch.ops.aten.mul.Tensor(sum_47, squeeze_220);  sum_47 = squeeze_220 = None
        convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_884, relu_72, primals_222, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_884 = primals_222 = None
        getitem_266 = convolution_backward_22[0]
        getitem_267 = convolution_backward_22[1];  convolution_backward_22 = None
        alias_286 = torch.ops.aten.alias.default(relu_72);  relu_72 = None
        alias_287 = torch.ops.aten.alias.default(alias_286);  alias_286 = None
        le_23 = torch.ops.aten.le.Scalar(alias_287, 0);  alias_287 = None
        where_23 = torch.ops.aten.where.self(le_23, full_default, getitem_266);  le_23 = getitem_266 = None
        sum_48 = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
        sub_188 = torch.ops.aten.sub.Tensor(convolution_72, unsqueeze_665);  convolution_72 = unsqueeze_665 = None
        mul_886 = torch.ops.aten.mul.Tensor(where_23, sub_188)
        sum_49 = torch.ops.aten.sum.dim_IntList(mul_886, [0, 2, 3]);  mul_886 = None
        mul_887 = torch.ops.aten.mul.Tensor(sum_48, 0.00010813148788927336)
        unsqueeze_666 = torch.ops.aten.unsqueeze.default(mul_887, 0);  mul_887 = None
        unsqueeze_667 = torch.ops.aten.unsqueeze.default(unsqueeze_666, 2);  unsqueeze_666 = None
        unsqueeze_668 = torch.ops.aten.unsqueeze.default(unsqueeze_667, 3);  unsqueeze_667 = None
        mul_888 = torch.ops.aten.mul.Tensor(sum_49, 0.00010813148788927336)
        mul_889 = torch.ops.aten.mul.Tensor(squeeze_217, squeeze_217)
        mul_890 = torch.ops.aten.mul.Tensor(mul_888, mul_889);  mul_888 = mul_889 = None
        unsqueeze_669 = torch.ops.aten.unsqueeze.default(mul_890, 0);  mul_890 = None
        unsqueeze_670 = torch.ops.aten.unsqueeze.default(unsqueeze_669, 2);  unsqueeze_669 = None
        unsqueeze_671 = torch.ops.aten.unsqueeze.default(unsqueeze_670, 3);  unsqueeze_670 = None
        mul_891 = torch.ops.aten.mul.Tensor(squeeze_217, primals_220);  primals_220 = None
        unsqueeze_672 = torch.ops.aten.unsqueeze.default(mul_891, 0);  mul_891 = None
        unsqueeze_673 = torch.ops.aten.unsqueeze.default(unsqueeze_672, 2);  unsqueeze_672 = None
        unsqueeze_674 = torch.ops.aten.unsqueeze.default(unsqueeze_673, 3);  unsqueeze_673 = None
        mul_892 = torch.ops.aten.mul.Tensor(sub_188, unsqueeze_671);  sub_188 = unsqueeze_671 = None
        sub_190 = torch.ops.aten.sub.Tensor(where_23, mul_892);  where_23 = mul_892 = None
        sub_191 = torch.ops.aten.sub.Tensor(sub_190, unsqueeze_668);  sub_190 = unsqueeze_668 = None
        mul_893 = torch.ops.aten.mul.Tensor(sub_191, unsqueeze_674);  sub_191 = unsqueeze_674 = None
        mul_894 = torch.ops.aten.mul.Tensor(sum_49, squeeze_217);  sum_49 = squeeze_217 = None
        convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_893, cat_8, primals_219, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_893 = primals_219 = None
        getitem_269 = convolution_backward_23[0]
        getitem_270 = convolution_backward_23[1];  convolution_backward_23 = None
        add_494 = torch.ops.aten.add.Tensor(add_493, getitem_269);  add_493 = getitem_269 = None
        mm_2 = torch.ops.aten.mm.default(tangents_2, permute_6);  permute_6 = None
        permute_7 = torch.ops.aten.permute.default(tangents_2, [1, 0])
        mm_3 = torch.ops.aten.mm.default(permute_7, view);  permute_7 = view = None
        permute_8 = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
        sum_50 = torch.ops.aten.sum.dim_IntList(tangents_2, [0], True);  tangents_2 = None
        view_4 = torch.ops.aten.view.default(sum_50, [1000]);  sum_50 = None
        permute_9 = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        view_5 = torch.ops.aten.view.default(mm_2, [32, 768, 1, 1]);  mm_2 = None
        expand_1 = torch.ops.aten.expand.default(view_5, [32, 768, 1, 1]);  view_5 = None
        div_1 = torch.ops.aten.div.Scalar(expand_1, 1);  expand_1 = None
        where_24 = torch.ops.aten.where.self(le_24, full_default, div_1);  le_24 = div_1 = None
        sum_51 = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
        sub_192 = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_677);  convolution_71 = unsqueeze_677 = None
        mul_895 = torch.ops.aten.mul.Tensor(where_24, sub_192)
        sum_52 = torch.ops.aten.sum.dim_IntList(mul_895, [0, 2, 3]);  mul_895 = None
        mul_896 = torch.ops.aten.mul.Tensor(sum_51, 0.03125)
        unsqueeze_678 = torch.ops.aten.unsqueeze.default(mul_896, 0);  mul_896 = None
        unsqueeze_679 = torch.ops.aten.unsqueeze.default(unsqueeze_678, 2);  unsqueeze_678 = None
        unsqueeze_680 = torch.ops.aten.unsqueeze.default(unsqueeze_679, 3);  unsqueeze_679 = None
        mul_897 = torch.ops.aten.mul.Tensor(sum_52, 0.03125)
        mul_898 = torch.ops.aten.mul.Tensor(squeeze_214, squeeze_214)
        mul_899 = torch.ops.aten.mul.Tensor(mul_897, mul_898);  mul_897 = mul_898 = None
        unsqueeze_681 = torch.ops.aten.unsqueeze.default(mul_899, 0);  mul_899 = None
        unsqueeze_682 = torch.ops.aten.unsqueeze.default(unsqueeze_681, 2);  unsqueeze_681 = None
        unsqueeze_683 = torch.ops.aten.unsqueeze.default(unsqueeze_682, 3);  unsqueeze_682 = None
        mul_900 = torch.ops.aten.mul.Tensor(squeeze_214, primals_215);  primals_215 = None
        unsqueeze_684 = torch.ops.aten.unsqueeze.default(mul_900, 0);  mul_900 = None
        unsqueeze_685 = torch.ops.aten.unsqueeze.default(unsqueeze_684, 2);  unsqueeze_684 = None
        unsqueeze_686 = torch.ops.aten.unsqueeze.default(unsqueeze_685, 3);  unsqueeze_685 = None
        mul_901 = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_683);  sub_192 = unsqueeze_683 = None
        sub_194 = torch.ops.aten.sub.Tensor(where_24, mul_901);  where_24 = mul_901 = None
        sub_195 = torch.ops.aten.sub.Tensor(sub_194, unsqueeze_680);  sub_194 = unsqueeze_680 = None
        mul_902 = torch.ops.aten.mul.Tensor(sub_195, unsqueeze_686);  sub_195 = unsqueeze_686 = None
        mul_903 = torch.ops.aten.mul.Tensor(sum_52, squeeze_214);  sum_52 = squeeze_214 = None
        convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_902, relu_70, primals_214, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_902 = primals_214 = None
        getitem_272 = convolution_backward_24[0]
        getitem_273 = convolution_backward_24[1];  convolution_backward_24 = None
        alias_294 = torch.ops.aten.alias.default(relu_70);  relu_70 = None
        alias_295 = torch.ops.aten.alias.default(alias_294);  alias_294 = None
        le_25 = torch.ops.aten.le.Scalar(alias_295, 0);  alias_295 = None
        where_25 = torch.ops.aten.where.self(le_25, full_default, getitem_272);  le_25 = getitem_272 = None
        sum_53 = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
        sub_196 = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_689);  convolution_70 = unsqueeze_689 = None
        mul_904 = torch.ops.aten.mul.Tensor(where_25, sub_196)
        sum_54 = torch.ops.aten.sum.dim_IntList(mul_904, [0, 2, 3]);  mul_904 = None
        mul_905 = torch.ops.aten.mul.Tensor(sum_53, 0.00125)
        unsqueeze_690 = torch.ops.aten.unsqueeze.default(mul_905, 0);  mul_905 = None
        unsqueeze_691 = torch.ops.aten.unsqueeze.default(unsqueeze_690, 2);  unsqueeze_690 = None
        unsqueeze_692 = torch.ops.aten.unsqueeze.default(unsqueeze_691, 3);  unsqueeze_691 = None
        mul_906 = torch.ops.aten.mul.Tensor(sum_54, 0.00125)
        mul_907 = torch.ops.aten.mul.Tensor(squeeze_211, squeeze_211)
        mul_908 = torch.ops.aten.mul.Tensor(mul_906, mul_907);  mul_906 = mul_907 = None
        unsqueeze_693 = torch.ops.aten.unsqueeze.default(mul_908, 0);  mul_908 = None
        unsqueeze_694 = torch.ops.aten.unsqueeze.default(unsqueeze_693, 2);  unsqueeze_693 = None
        unsqueeze_695 = torch.ops.aten.unsqueeze.default(unsqueeze_694, 3);  unsqueeze_694 = None
        mul_909 = torch.ops.aten.mul.Tensor(squeeze_211, primals_212);  primals_212 = None
        unsqueeze_696 = torch.ops.aten.unsqueeze.default(mul_909, 0);  mul_909 = None
        unsqueeze_697 = torch.ops.aten.unsqueeze.default(unsqueeze_696, 2);  unsqueeze_696 = None
        unsqueeze_698 = torch.ops.aten.unsqueeze.default(unsqueeze_697, 3);  unsqueeze_697 = None
        mul_910 = torch.ops.aten.mul.Tensor(sub_196, unsqueeze_695);  sub_196 = unsqueeze_695 = None
        sub_198 = torch.ops.aten.sub.Tensor(where_25, mul_910);  where_25 = mul_910 = None
        sub_199 = torch.ops.aten.sub.Tensor(sub_198, unsqueeze_692);  sub_198 = unsqueeze_692 = None
        mul_911 = torch.ops.aten.mul.Tensor(sub_199, unsqueeze_698);  sub_199 = unsqueeze_698 = None
        mul_912 = torch.ops.aten.mul.Tensor(sum_54, squeeze_211);  sum_54 = squeeze_211 = None
        convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_911, avg_pool2d_7, primals_211, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_911 = avg_pool2d_7 = primals_211 = None
        getitem_275 = convolution_backward_25[0]
        getitem_276 = convolution_backward_25[1];  convolution_backward_25 = None
        avg_pool2d_backward_2 = torch.ops.aten.avg_pool2d_backward.default(getitem_275, cat_8, [5, 5], [3, 3], [0, 0], False, True, None);  getitem_275 = cat_8 = None
        add_495 = torch.ops.aten.add.Tensor(add_494, avg_pool2d_backward_2);  add_494 = avg_pool2d_backward_2 = None
        slice_23 = torch.ops.aten.slice.Tensor(add_495, 1, 0, 192)
        slice_24 = torch.ops.aten.slice.Tensor(add_495, 1, 192, 384)
        slice_25 = torch.ops.aten.slice.Tensor(add_495, 1, 384, 576)
        slice_26 = torch.ops.aten.slice.Tensor(add_495, 1, 576, 768);  add_495 = None
        where_26 = torch.ops.aten.where.self(le_26, full_default, slice_26);  le_26 = slice_26 = None
        sum_55 = torch.ops.aten.sum.dim_IntList(where_26, [0, 2, 3])
        sub_200 = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_701);  convolution_69 = unsqueeze_701 = None
        mul_913 = torch.ops.aten.mul.Tensor(where_26, sub_200)
        sum_56 = torch.ops.aten.sum.dim_IntList(mul_913, [0, 2, 3]);  mul_913 = None
        mul_914 = torch.ops.aten.mul.Tensor(sum_55, 0.00010813148788927336)
        unsqueeze_702 = torch.ops.aten.unsqueeze.default(mul_914, 0);  mul_914 = None
        unsqueeze_703 = torch.ops.aten.unsqueeze.default(unsqueeze_702, 2);  unsqueeze_702 = None
        unsqueeze_704 = torch.ops.aten.unsqueeze.default(unsqueeze_703, 3);  unsqueeze_703 = None
        mul_915 = torch.ops.aten.mul.Tensor(sum_56, 0.00010813148788927336)
        mul_916 = torch.ops.aten.mul.Tensor(squeeze_208, squeeze_208)
        mul_917 = torch.ops.aten.mul.Tensor(mul_915, mul_916);  mul_915 = mul_916 = None
        unsqueeze_705 = torch.ops.aten.unsqueeze.default(mul_917, 0);  mul_917 = None
        unsqueeze_706 = torch.ops.aten.unsqueeze.default(unsqueeze_705, 2);  unsqueeze_705 = None
        unsqueeze_707 = torch.ops.aten.unsqueeze.default(unsqueeze_706, 3);  unsqueeze_706 = None
        mul_918 = torch.ops.aten.mul.Tensor(squeeze_208, primals_209);  primals_209 = None
        unsqueeze_708 = torch.ops.aten.unsqueeze.default(mul_918, 0);  mul_918 = None
        unsqueeze_709 = torch.ops.aten.unsqueeze.default(unsqueeze_708, 2);  unsqueeze_708 = None
        unsqueeze_710 = torch.ops.aten.unsqueeze.default(unsqueeze_709, 3);  unsqueeze_709 = None
        mul_919 = torch.ops.aten.mul.Tensor(sub_200, unsqueeze_707);  sub_200 = unsqueeze_707 = None
        sub_202 = torch.ops.aten.sub.Tensor(where_26, mul_919);  where_26 = mul_919 = None
        sub_203 = torch.ops.aten.sub.Tensor(sub_202, unsqueeze_704);  sub_202 = unsqueeze_704 = None
        mul_920 = torch.ops.aten.mul.Tensor(sub_203, unsqueeze_710);  sub_203 = unsqueeze_710 = None
        mul_921 = torch.ops.aten.mul.Tensor(sum_56, squeeze_208);  sum_56 = squeeze_208 = None
        convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_920, avg_pool2d_6, primals_208, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_920 = avg_pool2d_6 = primals_208 = None
        getitem_278 = convolution_backward_26[0]
        getitem_279 = convolution_backward_26[1];  convolution_backward_26 = None
        avg_pool2d_backward_3 = torch.ops.aten.avg_pool2d_backward.default(getitem_278, cat_7, [3, 3], [1, 1], [1, 1], False, True, None);  getitem_278 = None
        where_27 = torch.ops.aten.where.self(le_27, full_default, slice_25);  le_27 = slice_25 = None
        sum_57 = torch.ops.aten.sum.dim_IntList(where_27, [0, 2, 3])
        sub_204 = torch.ops.aten.sub.Tensor(convolution_68, unsqueeze_713);  convolution_68 = unsqueeze_713 = None
        mul_922 = torch.ops.aten.mul.Tensor(where_27, sub_204)
        sum_58 = torch.ops.aten.sum.dim_IntList(mul_922, [0, 2, 3]);  mul_922 = None
        mul_923 = torch.ops.aten.mul.Tensor(sum_57, 0.00010813148788927336)
        unsqueeze_714 = torch.ops.aten.unsqueeze.default(mul_923, 0);  mul_923 = None
        unsqueeze_715 = torch.ops.aten.unsqueeze.default(unsqueeze_714, 2);  unsqueeze_714 = None
        unsqueeze_716 = torch.ops.aten.unsqueeze.default(unsqueeze_715, 3);  unsqueeze_715 = None
        mul_924 = torch.ops.aten.mul.Tensor(sum_58, 0.00010813148788927336)
        mul_925 = torch.ops.aten.mul.Tensor(squeeze_205, squeeze_205)
        mul_926 = torch.ops.aten.mul.Tensor(mul_924, mul_925);  mul_924 = mul_925 = None
        unsqueeze_717 = torch.ops.aten.unsqueeze.default(mul_926, 0);  mul_926 = None
        unsqueeze_718 = torch.ops.aten.unsqueeze.default(unsqueeze_717, 2);  unsqueeze_717 = None
        unsqueeze_719 = torch.ops.aten.unsqueeze.default(unsqueeze_718, 3);  unsqueeze_718 = None
        mul_927 = torch.ops.aten.mul.Tensor(squeeze_205, primals_206);  primals_206 = None
        unsqueeze_720 = torch.ops.aten.unsqueeze.default(mul_927, 0);  mul_927 = None
        unsqueeze_721 = torch.ops.aten.unsqueeze.default(unsqueeze_720, 2);  unsqueeze_720 = None
        unsqueeze_722 = torch.ops.aten.unsqueeze.default(unsqueeze_721, 3);  unsqueeze_721 = None
        mul_928 = torch.ops.aten.mul.Tensor(sub_204, unsqueeze_719);  sub_204 = unsqueeze_719 = None
        sub_206 = torch.ops.aten.sub.Tensor(where_27, mul_928);  where_27 = mul_928 = None
        sub_207 = torch.ops.aten.sub.Tensor(sub_206, unsqueeze_716);  sub_206 = unsqueeze_716 = None
        mul_929 = torch.ops.aten.mul.Tensor(sub_207, unsqueeze_722);  sub_207 = unsqueeze_722 = None
        mul_930 = torch.ops.aten.mul.Tensor(sum_58, squeeze_205);  sum_58 = squeeze_205 = None
        convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_929, relu_67, primals_205, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_929 = primals_205 = None
        getitem_281 = convolution_backward_27[0]
        getitem_282 = convolution_backward_27[1];  convolution_backward_27 = None
        alias_306 = torch.ops.aten.alias.default(relu_67);  relu_67 = None
        alias_307 = torch.ops.aten.alias.default(alias_306);  alias_306 = None
        le_28 = torch.ops.aten.le.Scalar(alias_307, 0);  alias_307 = None
        where_28 = torch.ops.aten.where.self(le_28, full_default, getitem_281);  le_28 = getitem_281 = None
        sum_59 = torch.ops.aten.sum.dim_IntList(where_28, [0, 2, 3])
        sub_208 = torch.ops.aten.sub.Tensor(convolution_67, unsqueeze_725);  convolution_67 = unsqueeze_725 = None
        mul_931 = torch.ops.aten.mul.Tensor(where_28, sub_208)
        sum_60 = torch.ops.aten.sum.dim_IntList(mul_931, [0, 2, 3]);  mul_931 = None
        mul_932 = torch.ops.aten.mul.Tensor(sum_59, 0.00010813148788927336)
        unsqueeze_726 = torch.ops.aten.unsqueeze.default(mul_932, 0);  mul_932 = None
        unsqueeze_727 = torch.ops.aten.unsqueeze.default(unsqueeze_726, 2);  unsqueeze_726 = None
        unsqueeze_728 = torch.ops.aten.unsqueeze.default(unsqueeze_727, 3);  unsqueeze_727 = None
        mul_933 = torch.ops.aten.mul.Tensor(sum_60, 0.00010813148788927336)
        mul_934 = torch.ops.aten.mul.Tensor(squeeze_202, squeeze_202)
        mul_935 = torch.ops.aten.mul.Tensor(mul_933, mul_934);  mul_933 = mul_934 = None
        unsqueeze_729 = torch.ops.aten.unsqueeze.default(mul_935, 0);  mul_935 = None
        unsqueeze_730 = torch.ops.aten.unsqueeze.default(unsqueeze_729, 2);  unsqueeze_729 = None
        unsqueeze_731 = torch.ops.aten.unsqueeze.default(unsqueeze_730, 3);  unsqueeze_730 = None
        mul_936 = torch.ops.aten.mul.Tensor(squeeze_202, primals_203);  primals_203 = None
        unsqueeze_732 = torch.ops.aten.unsqueeze.default(mul_936, 0);  mul_936 = None
        unsqueeze_733 = torch.ops.aten.unsqueeze.default(unsqueeze_732, 2);  unsqueeze_732 = None
        unsqueeze_734 = torch.ops.aten.unsqueeze.default(unsqueeze_733, 3);  unsqueeze_733 = None
        mul_937 = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_731);  sub_208 = unsqueeze_731 = None
        sub_210 = torch.ops.aten.sub.Tensor(where_28, mul_937);  where_28 = mul_937 = None
        sub_211 = torch.ops.aten.sub.Tensor(sub_210, unsqueeze_728);  sub_210 = unsqueeze_728 = None
        mul_938 = torch.ops.aten.mul.Tensor(sub_211, unsqueeze_734);  sub_211 = unsqueeze_734 = None
        mul_939 = torch.ops.aten.mul.Tensor(sum_60, squeeze_202);  sum_60 = squeeze_202 = None
        convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_938, relu_66, primals_202, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_938 = primals_202 = None
        getitem_284 = convolution_backward_28[0]
        getitem_285 = convolution_backward_28[1];  convolution_backward_28 = None
        alias_310 = torch.ops.aten.alias.default(relu_66);  relu_66 = None
        alias_311 = torch.ops.aten.alias.default(alias_310);  alias_310 = None
        le_29 = torch.ops.aten.le.Scalar(alias_311, 0);  alias_311 = None
        where_29 = torch.ops.aten.where.self(le_29, full_default, getitem_284);  le_29 = getitem_284 = None
        sum_61 = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
        sub_212 = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_737);  convolution_66 = unsqueeze_737 = None
        mul_940 = torch.ops.aten.mul.Tensor(where_29, sub_212)
        sum_62 = torch.ops.aten.sum.dim_IntList(mul_940, [0, 2, 3]);  mul_940 = None
        mul_941 = torch.ops.aten.mul.Tensor(sum_61, 0.00010813148788927336)
        unsqueeze_738 = torch.ops.aten.unsqueeze.default(mul_941, 0);  mul_941 = None
        unsqueeze_739 = torch.ops.aten.unsqueeze.default(unsqueeze_738, 2);  unsqueeze_738 = None
        unsqueeze_740 = torch.ops.aten.unsqueeze.default(unsqueeze_739, 3);  unsqueeze_739 = None
        mul_942 = torch.ops.aten.mul.Tensor(sum_62, 0.00010813148788927336)
        mul_943 = torch.ops.aten.mul.Tensor(squeeze_199, squeeze_199)
        mul_944 = torch.ops.aten.mul.Tensor(mul_942, mul_943);  mul_942 = mul_943 = None
        unsqueeze_741 = torch.ops.aten.unsqueeze.default(mul_944, 0);  mul_944 = None
        unsqueeze_742 = torch.ops.aten.unsqueeze.default(unsqueeze_741, 2);  unsqueeze_741 = None
        unsqueeze_743 = torch.ops.aten.unsqueeze.default(unsqueeze_742, 3);  unsqueeze_742 = None
        mul_945 = torch.ops.aten.mul.Tensor(squeeze_199, primals_200);  primals_200 = None
        unsqueeze_744 = torch.ops.aten.unsqueeze.default(mul_945, 0);  mul_945 = None
        unsqueeze_745 = torch.ops.aten.unsqueeze.default(unsqueeze_744, 2);  unsqueeze_744 = None
        unsqueeze_746 = torch.ops.aten.unsqueeze.default(unsqueeze_745, 3);  unsqueeze_745 = None
        mul_946 = torch.ops.aten.mul.Tensor(sub_212, unsqueeze_743);  sub_212 = unsqueeze_743 = None
        sub_214 = torch.ops.aten.sub.Tensor(where_29, mul_946);  where_29 = mul_946 = None
        sub_215 = torch.ops.aten.sub.Tensor(sub_214, unsqueeze_740);  sub_214 = unsqueeze_740 = None
        mul_947 = torch.ops.aten.mul.Tensor(sub_215, unsqueeze_746);  sub_215 = unsqueeze_746 = None
        mul_948 = torch.ops.aten.mul.Tensor(sum_62, squeeze_199);  sum_62 = squeeze_199 = None
        convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_947, relu_65, primals_199, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_947 = primals_199 = None
        getitem_287 = convolution_backward_29[0]
        getitem_288 = convolution_backward_29[1];  convolution_backward_29 = None
        alias_314 = torch.ops.aten.alias.default(relu_65);  relu_65 = None
        alias_315 = torch.ops.aten.alias.default(alias_314);  alias_314 = None
        le_30 = torch.ops.aten.le.Scalar(alias_315, 0);  alias_315 = None
        where_30 = torch.ops.aten.where.self(le_30, full_default, getitem_287);  le_30 = getitem_287 = None
        sum_63 = torch.ops.aten.sum.dim_IntList(where_30, [0, 2, 3])
        sub_216 = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_749);  convolution_65 = unsqueeze_749 = None
        mul_949 = torch.ops.aten.mul.Tensor(where_30, sub_216)
        sum_64 = torch.ops.aten.sum.dim_IntList(mul_949, [0, 2, 3]);  mul_949 = None
        mul_950 = torch.ops.aten.mul.Tensor(sum_63, 0.00010813148788927336)
        unsqueeze_750 = torch.ops.aten.unsqueeze.default(mul_950, 0);  mul_950 = None
        unsqueeze_751 = torch.ops.aten.unsqueeze.default(unsqueeze_750, 2);  unsqueeze_750 = None
        unsqueeze_752 = torch.ops.aten.unsqueeze.default(unsqueeze_751, 3);  unsqueeze_751 = None
        mul_951 = torch.ops.aten.mul.Tensor(sum_64, 0.00010813148788927336)
        mul_952 = torch.ops.aten.mul.Tensor(squeeze_196, squeeze_196)
        mul_953 = torch.ops.aten.mul.Tensor(mul_951, mul_952);  mul_951 = mul_952 = None
        unsqueeze_753 = torch.ops.aten.unsqueeze.default(mul_953, 0);  mul_953 = None
        unsqueeze_754 = torch.ops.aten.unsqueeze.default(unsqueeze_753, 2);  unsqueeze_753 = None
        unsqueeze_755 = torch.ops.aten.unsqueeze.default(unsqueeze_754, 3);  unsqueeze_754 = None
        mul_954 = torch.ops.aten.mul.Tensor(squeeze_196, primals_197);  primals_197 = None
        unsqueeze_756 = torch.ops.aten.unsqueeze.default(mul_954, 0);  mul_954 = None
        unsqueeze_757 = torch.ops.aten.unsqueeze.default(unsqueeze_756, 2);  unsqueeze_756 = None
        unsqueeze_758 = torch.ops.aten.unsqueeze.default(unsqueeze_757, 3);  unsqueeze_757 = None
        mul_955 = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_755);  sub_216 = unsqueeze_755 = None
        sub_218 = torch.ops.aten.sub.Tensor(where_30, mul_955);  where_30 = mul_955 = None
        sub_219 = torch.ops.aten.sub.Tensor(sub_218, unsqueeze_752);  sub_218 = unsqueeze_752 = None
        mul_956 = torch.ops.aten.mul.Tensor(sub_219, unsqueeze_758);  sub_219 = unsqueeze_758 = None
        mul_957 = torch.ops.aten.mul.Tensor(sum_64, squeeze_196);  sum_64 = squeeze_196 = None
        convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_956, relu_64, primals_196, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_956 = primals_196 = None
        getitem_290 = convolution_backward_30[0]
        getitem_291 = convolution_backward_30[1];  convolution_backward_30 = None
        alias_318 = torch.ops.aten.alias.default(relu_64);  relu_64 = None
        alias_319 = torch.ops.aten.alias.default(alias_318);  alias_318 = None
        le_31 = torch.ops.aten.le.Scalar(alias_319, 0);  alias_319 = None
        where_31 = torch.ops.aten.where.self(le_31, full_default, getitem_290);  le_31 = getitem_290 = None
        sum_65 = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
        sub_220 = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_761);  convolution_64 = unsqueeze_761 = None
        mul_958 = torch.ops.aten.mul.Tensor(where_31, sub_220)
        sum_66 = torch.ops.aten.sum.dim_IntList(mul_958, [0, 2, 3]);  mul_958 = None
        mul_959 = torch.ops.aten.mul.Tensor(sum_65, 0.00010813148788927336)
        unsqueeze_762 = torch.ops.aten.unsqueeze.default(mul_959, 0);  mul_959 = None
        unsqueeze_763 = torch.ops.aten.unsqueeze.default(unsqueeze_762, 2);  unsqueeze_762 = None
        unsqueeze_764 = torch.ops.aten.unsqueeze.default(unsqueeze_763, 3);  unsqueeze_763 = None
        mul_960 = torch.ops.aten.mul.Tensor(sum_66, 0.00010813148788927336)
        mul_961 = torch.ops.aten.mul.Tensor(squeeze_193, squeeze_193)
        mul_962 = torch.ops.aten.mul.Tensor(mul_960, mul_961);  mul_960 = mul_961 = None
        unsqueeze_765 = torch.ops.aten.unsqueeze.default(mul_962, 0);  mul_962 = None
        unsqueeze_766 = torch.ops.aten.unsqueeze.default(unsqueeze_765, 2);  unsqueeze_765 = None
        unsqueeze_767 = torch.ops.aten.unsqueeze.default(unsqueeze_766, 3);  unsqueeze_766 = None
        mul_963 = torch.ops.aten.mul.Tensor(squeeze_193, primals_194);  primals_194 = None
        unsqueeze_768 = torch.ops.aten.unsqueeze.default(mul_963, 0);  mul_963 = None
        unsqueeze_769 = torch.ops.aten.unsqueeze.default(unsqueeze_768, 2);  unsqueeze_768 = None
        unsqueeze_770 = torch.ops.aten.unsqueeze.default(unsqueeze_769, 3);  unsqueeze_769 = None
        mul_964 = torch.ops.aten.mul.Tensor(sub_220, unsqueeze_767);  sub_220 = unsqueeze_767 = None
        sub_222 = torch.ops.aten.sub.Tensor(where_31, mul_964);  where_31 = mul_964 = None
        sub_223 = torch.ops.aten.sub.Tensor(sub_222, unsqueeze_764);  sub_222 = unsqueeze_764 = None
        mul_965 = torch.ops.aten.mul.Tensor(sub_223, unsqueeze_770);  sub_223 = unsqueeze_770 = None
        mul_966 = torch.ops.aten.mul.Tensor(sum_66, squeeze_193);  sum_66 = squeeze_193 = None
        convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_965, cat_7, primals_193, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_965 = primals_193 = None
        getitem_293 = convolution_backward_31[0]
        getitem_294 = convolution_backward_31[1];  convolution_backward_31 = None
        add_496 = torch.ops.aten.add.Tensor(avg_pool2d_backward_3, getitem_293);  avg_pool2d_backward_3 = getitem_293 = None
        where_32 = torch.ops.aten.where.self(le_32, full_default, slice_24);  le_32 = slice_24 = None
        sum_67 = torch.ops.aten.sum.dim_IntList(where_32, [0, 2, 3])
        sub_224 = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_773);  convolution_63 = unsqueeze_773 = None
        mul_967 = torch.ops.aten.mul.Tensor(where_32, sub_224)
        sum_68 = torch.ops.aten.sum.dim_IntList(mul_967, [0, 2, 3]);  mul_967 = None
        mul_968 = torch.ops.aten.mul.Tensor(sum_67, 0.00010813148788927336)
        unsqueeze_774 = torch.ops.aten.unsqueeze.default(mul_968, 0);  mul_968 = None
        unsqueeze_775 = torch.ops.aten.unsqueeze.default(unsqueeze_774, 2);  unsqueeze_774 = None
        unsqueeze_776 = torch.ops.aten.unsqueeze.default(unsqueeze_775, 3);  unsqueeze_775 = None
        mul_969 = torch.ops.aten.mul.Tensor(sum_68, 0.00010813148788927336)
        mul_970 = torch.ops.aten.mul.Tensor(squeeze_190, squeeze_190)
        mul_971 = torch.ops.aten.mul.Tensor(mul_969, mul_970);  mul_969 = mul_970 = None
        unsqueeze_777 = torch.ops.aten.unsqueeze.default(mul_971, 0);  mul_971 = None
        unsqueeze_778 = torch.ops.aten.unsqueeze.default(unsqueeze_777, 2);  unsqueeze_777 = None
        unsqueeze_779 = torch.ops.aten.unsqueeze.default(unsqueeze_778, 3);  unsqueeze_778 = None
        mul_972 = torch.ops.aten.mul.Tensor(squeeze_190, primals_191);  primals_191 = None
        unsqueeze_780 = torch.ops.aten.unsqueeze.default(mul_972, 0);  mul_972 = None
        unsqueeze_781 = torch.ops.aten.unsqueeze.default(unsqueeze_780, 2);  unsqueeze_780 = None
        unsqueeze_782 = torch.ops.aten.unsqueeze.default(unsqueeze_781, 3);  unsqueeze_781 = None
        mul_973 = torch.ops.aten.mul.Tensor(sub_224, unsqueeze_779);  sub_224 = unsqueeze_779 = None
        sub_226 = torch.ops.aten.sub.Tensor(where_32, mul_973);  where_32 = mul_973 = None
        sub_227 = torch.ops.aten.sub.Tensor(sub_226, unsqueeze_776);  sub_226 = unsqueeze_776 = None
        mul_974 = torch.ops.aten.mul.Tensor(sub_227, unsqueeze_782);  sub_227 = unsqueeze_782 = None
        mul_975 = torch.ops.aten.mul.Tensor(sum_68, squeeze_190);  sum_68 = squeeze_190 = None
        convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_974, relu_62, primals_190, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_974 = primals_190 = None
        getitem_296 = convolution_backward_32[0]
        getitem_297 = convolution_backward_32[1];  convolution_backward_32 = None
        alias_326 = torch.ops.aten.alias.default(relu_62);  relu_62 = None
        alias_327 = torch.ops.aten.alias.default(alias_326);  alias_326 = None
        le_33 = torch.ops.aten.le.Scalar(alias_327, 0);  alias_327 = None
        where_33 = torch.ops.aten.where.self(le_33, full_default, getitem_296);  le_33 = getitem_296 = None
        sum_69 = torch.ops.aten.sum.dim_IntList(where_33, [0, 2, 3])
        sub_228 = torch.ops.aten.sub.Tensor(convolution_62, unsqueeze_785);  convolution_62 = unsqueeze_785 = None
        mul_976 = torch.ops.aten.mul.Tensor(where_33, sub_228)
        sum_70 = torch.ops.aten.sum.dim_IntList(mul_976, [0, 2, 3]);  mul_976 = None
        mul_977 = torch.ops.aten.mul.Tensor(sum_69, 0.00010813148788927336)
        unsqueeze_786 = torch.ops.aten.unsqueeze.default(mul_977, 0);  mul_977 = None
        unsqueeze_787 = torch.ops.aten.unsqueeze.default(unsqueeze_786, 2);  unsqueeze_786 = None
        unsqueeze_788 = torch.ops.aten.unsqueeze.default(unsqueeze_787, 3);  unsqueeze_787 = None
        mul_978 = torch.ops.aten.mul.Tensor(sum_70, 0.00010813148788927336)
        mul_979 = torch.ops.aten.mul.Tensor(squeeze_187, squeeze_187)
        mul_980 = torch.ops.aten.mul.Tensor(mul_978, mul_979);  mul_978 = mul_979 = None
        unsqueeze_789 = torch.ops.aten.unsqueeze.default(mul_980, 0);  mul_980 = None
        unsqueeze_790 = torch.ops.aten.unsqueeze.default(unsqueeze_789, 2);  unsqueeze_789 = None
        unsqueeze_791 = torch.ops.aten.unsqueeze.default(unsqueeze_790, 3);  unsqueeze_790 = None
        mul_981 = torch.ops.aten.mul.Tensor(squeeze_187, primals_188);  primals_188 = None
        unsqueeze_792 = torch.ops.aten.unsqueeze.default(mul_981, 0);  mul_981 = None
        unsqueeze_793 = torch.ops.aten.unsqueeze.default(unsqueeze_792, 2);  unsqueeze_792 = None
        unsqueeze_794 = torch.ops.aten.unsqueeze.default(unsqueeze_793, 3);  unsqueeze_793 = None
        mul_982 = torch.ops.aten.mul.Tensor(sub_228, unsqueeze_791);  sub_228 = unsqueeze_791 = None
        sub_230 = torch.ops.aten.sub.Tensor(where_33, mul_982);  where_33 = mul_982 = None
        sub_231 = torch.ops.aten.sub.Tensor(sub_230, unsqueeze_788);  sub_230 = unsqueeze_788 = None
        mul_983 = torch.ops.aten.mul.Tensor(sub_231, unsqueeze_794);  sub_231 = unsqueeze_794 = None
        mul_984 = torch.ops.aten.mul.Tensor(sum_70, squeeze_187);  sum_70 = squeeze_187 = None
        convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_983, relu_61, primals_187, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_983 = primals_187 = None
        getitem_299 = convolution_backward_33[0]
        getitem_300 = convolution_backward_33[1];  convolution_backward_33 = None
        alias_330 = torch.ops.aten.alias.default(relu_61);  relu_61 = None
        alias_331 = torch.ops.aten.alias.default(alias_330);  alias_330 = None
        le_34 = torch.ops.aten.le.Scalar(alias_331, 0);  alias_331 = None
        where_34 = torch.ops.aten.where.self(le_34, full_default, getitem_299);  le_34 = getitem_299 = None
        sum_71 = torch.ops.aten.sum.dim_IntList(where_34, [0, 2, 3])
        sub_232 = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_797);  convolution_61 = unsqueeze_797 = None
        mul_985 = torch.ops.aten.mul.Tensor(where_34, sub_232)
        sum_72 = torch.ops.aten.sum.dim_IntList(mul_985, [0, 2, 3]);  mul_985 = None
        mul_986 = torch.ops.aten.mul.Tensor(sum_71, 0.00010813148788927336)
        unsqueeze_798 = torch.ops.aten.unsqueeze.default(mul_986, 0);  mul_986 = None
        unsqueeze_799 = torch.ops.aten.unsqueeze.default(unsqueeze_798, 2);  unsqueeze_798 = None
        unsqueeze_800 = torch.ops.aten.unsqueeze.default(unsqueeze_799, 3);  unsqueeze_799 = None
        mul_987 = torch.ops.aten.mul.Tensor(sum_72, 0.00010813148788927336)
        mul_988 = torch.ops.aten.mul.Tensor(squeeze_184, squeeze_184)
        mul_989 = torch.ops.aten.mul.Tensor(mul_987, mul_988);  mul_987 = mul_988 = None
        unsqueeze_801 = torch.ops.aten.unsqueeze.default(mul_989, 0);  mul_989 = None
        unsqueeze_802 = torch.ops.aten.unsqueeze.default(unsqueeze_801, 2);  unsqueeze_801 = None
        unsqueeze_803 = torch.ops.aten.unsqueeze.default(unsqueeze_802, 3);  unsqueeze_802 = None
        mul_990 = torch.ops.aten.mul.Tensor(squeeze_184, primals_185);  primals_185 = None
        unsqueeze_804 = torch.ops.aten.unsqueeze.default(mul_990, 0);  mul_990 = None
        unsqueeze_805 = torch.ops.aten.unsqueeze.default(unsqueeze_804, 2);  unsqueeze_804 = None
        unsqueeze_806 = torch.ops.aten.unsqueeze.default(unsqueeze_805, 3);  unsqueeze_805 = None
        mul_991 = torch.ops.aten.mul.Tensor(sub_232, unsqueeze_803);  sub_232 = unsqueeze_803 = None
        sub_234 = torch.ops.aten.sub.Tensor(where_34, mul_991);  where_34 = mul_991 = None
        sub_235 = torch.ops.aten.sub.Tensor(sub_234, unsqueeze_800);  sub_234 = unsqueeze_800 = None
        mul_992 = torch.ops.aten.mul.Tensor(sub_235, unsqueeze_806);  sub_235 = unsqueeze_806 = None
        mul_993 = torch.ops.aten.mul.Tensor(sum_72, squeeze_184);  sum_72 = squeeze_184 = None
        convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_992, cat_7, primals_184, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_992 = primals_184 = None
        getitem_302 = convolution_backward_34[0]
        getitem_303 = convolution_backward_34[1];  convolution_backward_34 = None
        add_497 = torch.ops.aten.add.Tensor(add_496, getitem_302);  add_496 = getitem_302 = None
        where_35 = torch.ops.aten.where.self(le_35, full_default, slice_23);  le_35 = slice_23 = None
        sum_73 = torch.ops.aten.sum.dim_IntList(where_35, [0, 2, 3])
        sub_236 = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_809);  convolution_60 = unsqueeze_809 = None
        mul_994 = torch.ops.aten.mul.Tensor(where_35, sub_236)
        sum_74 = torch.ops.aten.sum.dim_IntList(mul_994, [0, 2, 3]);  mul_994 = None
        mul_995 = torch.ops.aten.mul.Tensor(sum_73, 0.00010813148788927336)
        unsqueeze_810 = torch.ops.aten.unsqueeze.default(mul_995, 0);  mul_995 = None
        unsqueeze_811 = torch.ops.aten.unsqueeze.default(unsqueeze_810, 2);  unsqueeze_810 = None
        unsqueeze_812 = torch.ops.aten.unsqueeze.default(unsqueeze_811, 3);  unsqueeze_811 = None
        mul_996 = torch.ops.aten.mul.Tensor(sum_74, 0.00010813148788927336)
        mul_997 = torch.ops.aten.mul.Tensor(squeeze_181, squeeze_181)
        mul_998 = torch.ops.aten.mul.Tensor(mul_996, mul_997);  mul_996 = mul_997 = None
        unsqueeze_813 = torch.ops.aten.unsqueeze.default(mul_998, 0);  mul_998 = None
        unsqueeze_814 = torch.ops.aten.unsqueeze.default(unsqueeze_813, 2);  unsqueeze_813 = None
        unsqueeze_815 = torch.ops.aten.unsqueeze.default(unsqueeze_814, 3);  unsqueeze_814 = None
        mul_999 = torch.ops.aten.mul.Tensor(squeeze_181, primals_182);  primals_182 = None
        unsqueeze_816 = torch.ops.aten.unsqueeze.default(mul_999, 0);  mul_999 = None
        unsqueeze_817 = torch.ops.aten.unsqueeze.default(unsqueeze_816, 2);  unsqueeze_816 = None
        unsqueeze_818 = torch.ops.aten.unsqueeze.default(unsqueeze_817, 3);  unsqueeze_817 = None
        mul_1000 = torch.ops.aten.mul.Tensor(sub_236, unsqueeze_815);  sub_236 = unsqueeze_815 = None
        sub_238 = torch.ops.aten.sub.Tensor(where_35, mul_1000);  where_35 = mul_1000 = None
        sub_239 = torch.ops.aten.sub.Tensor(sub_238, unsqueeze_812);  sub_238 = unsqueeze_812 = None
        mul_1001 = torch.ops.aten.mul.Tensor(sub_239, unsqueeze_818);  sub_239 = unsqueeze_818 = None
        mul_1002 = torch.ops.aten.mul.Tensor(sum_74, squeeze_181);  sum_74 = squeeze_181 = None
        convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_1001, cat_7, primals_181, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1001 = cat_7 = primals_181 = None
        getitem_305 = convolution_backward_35[0]
        getitem_306 = convolution_backward_35[1];  convolution_backward_35 = None
        add_498 = torch.ops.aten.add.Tensor(add_497, getitem_305);  add_497 = getitem_305 = None
        slice_27 = torch.ops.aten.slice.Tensor(add_498, 1, 0, 192)
        slice_28 = torch.ops.aten.slice.Tensor(add_498, 1, 192, 384)
        slice_29 = torch.ops.aten.slice.Tensor(add_498, 1, 384, 576)
        slice_30 = torch.ops.aten.slice.Tensor(add_498, 1, 576, 768);  add_498 = None
        where_36 = torch.ops.aten.where.self(le_36, full_default, slice_30);  le_36 = slice_30 = None
        sum_75 = torch.ops.aten.sum.dim_IntList(where_36, [0, 2, 3])
        sub_240 = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_821);  convolution_59 = unsqueeze_821 = None
        mul_1003 = torch.ops.aten.mul.Tensor(where_36, sub_240)
        sum_76 = torch.ops.aten.sum.dim_IntList(mul_1003, [0, 2, 3]);  mul_1003 = None
        mul_1004 = torch.ops.aten.mul.Tensor(sum_75, 0.00010813148788927336)
        unsqueeze_822 = torch.ops.aten.unsqueeze.default(mul_1004, 0);  mul_1004 = None
        unsqueeze_823 = torch.ops.aten.unsqueeze.default(unsqueeze_822, 2);  unsqueeze_822 = None
        unsqueeze_824 = torch.ops.aten.unsqueeze.default(unsqueeze_823, 3);  unsqueeze_823 = None
        mul_1005 = torch.ops.aten.mul.Tensor(sum_76, 0.00010813148788927336)
        mul_1006 = torch.ops.aten.mul.Tensor(squeeze_178, squeeze_178)
        mul_1007 = torch.ops.aten.mul.Tensor(mul_1005, mul_1006);  mul_1005 = mul_1006 = None
        unsqueeze_825 = torch.ops.aten.unsqueeze.default(mul_1007, 0);  mul_1007 = None
        unsqueeze_826 = torch.ops.aten.unsqueeze.default(unsqueeze_825, 2);  unsqueeze_825 = None
        unsqueeze_827 = torch.ops.aten.unsqueeze.default(unsqueeze_826, 3);  unsqueeze_826 = None
        mul_1008 = torch.ops.aten.mul.Tensor(squeeze_178, primals_179);  primals_179 = None
        unsqueeze_828 = torch.ops.aten.unsqueeze.default(mul_1008, 0);  mul_1008 = None
        unsqueeze_829 = torch.ops.aten.unsqueeze.default(unsqueeze_828, 2);  unsqueeze_828 = None
        unsqueeze_830 = torch.ops.aten.unsqueeze.default(unsqueeze_829, 3);  unsqueeze_829 = None
        mul_1009 = torch.ops.aten.mul.Tensor(sub_240, unsqueeze_827);  sub_240 = unsqueeze_827 = None
        sub_242 = torch.ops.aten.sub.Tensor(where_36, mul_1009);  where_36 = mul_1009 = None
        sub_243 = torch.ops.aten.sub.Tensor(sub_242, unsqueeze_824);  sub_242 = unsqueeze_824 = None
        mul_1010 = torch.ops.aten.mul.Tensor(sub_243, unsqueeze_830);  sub_243 = unsqueeze_830 = None
        mul_1011 = torch.ops.aten.mul.Tensor(sum_76, squeeze_178);  sum_76 = squeeze_178 = None
        convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_1010, avg_pool2d_5, primals_178, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1010 = avg_pool2d_5 = primals_178 = None
        getitem_308 = convolution_backward_36[0]
        getitem_309 = convolution_backward_36[1];  convolution_backward_36 = None
        avg_pool2d_backward_4 = torch.ops.aten.avg_pool2d_backward.default(getitem_308, cat_6, [3, 3], [1, 1], [1, 1], False, True, None);  getitem_308 = None
        where_37 = torch.ops.aten.where.self(le_37, full_default, slice_29);  le_37 = slice_29 = None
        sum_77 = torch.ops.aten.sum.dim_IntList(where_37, [0, 2, 3])
        sub_244 = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_833);  convolution_58 = unsqueeze_833 = None
        mul_1012 = torch.ops.aten.mul.Tensor(where_37, sub_244)
        sum_78 = torch.ops.aten.sum.dim_IntList(mul_1012, [0, 2, 3]);  mul_1012 = None
        mul_1013 = torch.ops.aten.mul.Tensor(sum_77, 0.00010813148788927336)
        unsqueeze_834 = torch.ops.aten.unsqueeze.default(mul_1013, 0);  mul_1013 = None
        unsqueeze_835 = torch.ops.aten.unsqueeze.default(unsqueeze_834, 2);  unsqueeze_834 = None
        unsqueeze_836 = torch.ops.aten.unsqueeze.default(unsqueeze_835, 3);  unsqueeze_835 = None
        mul_1014 = torch.ops.aten.mul.Tensor(sum_78, 0.00010813148788927336)
        mul_1015 = torch.ops.aten.mul.Tensor(squeeze_175, squeeze_175)
        mul_1016 = torch.ops.aten.mul.Tensor(mul_1014, mul_1015);  mul_1014 = mul_1015 = None
        unsqueeze_837 = torch.ops.aten.unsqueeze.default(mul_1016, 0);  mul_1016 = None
        unsqueeze_838 = torch.ops.aten.unsqueeze.default(unsqueeze_837, 2);  unsqueeze_837 = None
        unsqueeze_839 = torch.ops.aten.unsqueeze.default(unsqueeze_838, 3);  unsqueeze_838 = None
        mul_1017 = torch.ops.aten.mul.Tensor(squeeze_175, primals_176);  primals_176 = None
        unsqueeze_840 = torch.ops.aten.unsqueeze.default(mul_1017, 0);  mul_1017 = None
        unsqueeze_841 = torch.ops.aten.unsqueeze.default(unsqueeze_840, 2);  unsqueeze_840 = None
        unsqueeze_842 = torch.ops.aten.unsqueeze.default(unsqueeze_841, 3);  unsqueeze_841 = None
        mul_1018 = torch.ops.aten.mul.Tensor(sub_244, unsqueeze_839);  sub_244 = unsqueeze_839 = None
        sub_246 = torch.ops.aten.sub.Tensor(where_37, mul_1018);  where_37 = mul_1018 = None
        sub_247 = torch.ops.aten.sub.Tensor(sub_246, unsqueeze_836);  sub_246 = unsqueeze_836 = None
        mul_1019 = torch.ops.aten.mul.Tensor(sub_247, unsqueeze_842);  sub_247 = unsqueeze_842 = None
        mul_1020 = torch.ops.aten.mul.Tensor(sum_78, squeeze_175);  sum_78 = squeeze_175 = None
        convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_1019, relu_57, primals_175, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1019 = primals_175 = None
        getitem_311 = convolution_backward_37[0]
        getitem_312 = convolution_backward_37[1];  convolution_backward_37 = None
        alias_346 = torch.ops.aten.alias.default(relu_57);  relu_57 = None
        alias_347 = torch.ops.aten.alias.default(alias_346);  alias_346 = None
        le_38 = torch.ops.aten.le.Scalar(alias_347, 0);  alias_347 = None
        where_38 = torch.ops.aten.where.self(le_38, full_default, getitem_311);  le_38 = getitem_311 = None
        sum_79 = torch.ops.aten.sum.dim_IntList(where_38, [0, 2, 3])
        sub_248 = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_845);  convolution_57 = unsqueeze_845 = None
        mul_1021 = torch.ops.aten.mul.Tensor(where_38, sub_248)
        sum_80 = torch.ops.aten.sum.dim_IntList(mul_1021, [0, 2, 3]);  mul_1021 = None
        mul_1022 = torch.ops.aten.mul.Tensor(sum_79, 0.00010813148788927336)
        unsqueeze_846 = torch.ops.aten.unsqueeze.default(mul_1022, 0);  mul_1022 = None
        unsqueeze_847 = torch.ops.aten.unsqueeze.default(unsqueeze_846, 2);  unsqueeze_846 = None
        unsqueeze_848 = torch.ops.aten.unsqueeze.default(unsqueeze_847, 3);  unsqueeze_847 = None
        mul_1023 = torch.ops.aten.mul.Tensor(sum_80, 0.00010813148788927336)
        mul_1024 = torch.ops.aten.mul.Tensor(squeeze_172, squeeze_172)
        mul_1025 = torch.ops.aten.mul.Tensor(mul_1023, mul_1024);  mul_1023 = mul_1024 = None
        unsqueeze_849 = torch.ops.aten.unsqueeze.default(mul_1025, 0);  mul_1025 = None
        unsqueeze_850 = torch.ops.aten.unsqueeze.default(unsqueeze_849, 2);  unsqueeze_849 = None
        unsqueeze_851 = torch.ops.aten.unsqueeze.default(unsqueeze_850, 3);  unsqueeze_850 = None
        mul_1026 = torch.ops.aten.mul.Tensor(squeeze_172, primals_173);  primals_173 = None
        unsqueeze_852 = torch.ops.aten.unsqueeze.default(mul_1026, 0);  mul_1026 = None
        unsqueeze_853 = torch.ops.aten.unsqueeze.default(unsqueeze_852, 2);  unsqueeze_852 = None
        unsqueeze_854 = torch.ops.aten.unsqueeze.default(unsqueeze_853, 3);  unsqueeze_853 = None
        mul_1027 = torch.ops.aten.mul.Tensor(sub_248, unsqueeze_851);  sub_248 = unsqueeze_851 = None
        sub_250 = torch.ops.aten.sub.Tensor(where_38, mul_1027);  where_38 = mul_1027 = None
        sub_251 = torch.ops.aten.sub.Tensor(sub_250, unsqueeze_848);  sub_250 = unsqueeze_848 = None
        mul_1028 = torch.ops.aten.mul.Tensor(sub_251, unsqueeze_854);  sub_251 = unsqueeze_854 = None
        mul_1029 = torch.ops.aten.mul.Tensor(sum_80, squeeze_172);  sum_80 = squeeze_172 = None
        convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_1028, relu_56, primals_172, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1028 = primals_172 = None
        getitem_314 = convolution_backward_38[0]
        getitem_315 = convolution_backward_38[1];  convolution_backward_38 = None
        alias_350 = torch.ops.aten.alias.default(relu_56);  relu_56 = None
        alias_351 = torch.ops.aten.alias.default(alias_350);  alias_350 = None
        le_39 = torch.ops.aten.le.Scalar(alias_351, 0);  alias_351 = None
        where_39 = torch.ops.aten.where.self(le_39, full_default, getitem_314);  le_39 = getitem_314 = None
        sum_81 = torch.ops.aten.sum.dim_IntList(where_39, [0, 2, 3])
        sub_252 = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_857);  convolution_56 = unsqueeze_857 = None
        mul_1030 = torch.ops.aten.mul.Tensor(where_39, sub_252)
        sum_82 = torch.ops.aten.sum.dim_IntList(mul_1030, [0, 2, 3]);  mul_1030 = None
        mul_1031 = torch.ops.aten.mul.Tensor(sum_81, 0.00010813148788927336)
        unsqueeze_858 = torch.ops.aten.unsqueeze.default(mul_1031, 0);  mul_1031 = None
        unsqueeze_859 = torch.ops.aten.unsqueeze.default(unsqueeze_858, 2);  unsqueeze_858 = None
        unsqueeze_860 = torch.ops.aten.unsqueeze.default(unsqueeze_859, 3);  unsqueeze_859 = None
        mul_1032 = torch.ops.aten.mul.Tensor(sum_82, 0.00010813148788927336)
        mul_1033 = torch.ops.aten.mul.Tensor(squeeze_169, squeeze_169)
        mul_1034 = torch.ops.aten.mul.Tensor(mul_1032, mul_1033);  mul_1032 = mul_1033 = None
        unsqueeze_861 = torch.ops.aten.unsqueeze.default(mul_1034, 0);  mul_1034 = None
        unsqueeze_862 = torch.ops.aten.unsqueeze.default(unsqueeze_861, 2);  unsqueeze_861 = None
        unsqueeze_863 = torch.ops.aten.unsqueeze.default(unsqueeze_862, 3);  unsqueeze_862 = None
        mul_1035 = torch.ops.aten.mul.Tensor(squeeze_169, primals_170);  primals_170 = None
        unsqueeze_864 = torch.ops.aten.unsqueeze.default(mul_1035, 0);  mul_1035 = None
        unsqueeze_865 = torch.ops.aten.unsqueeze.default(unsqueeze_864, 2);  unsqueeze_864 = None
        unsqueeze_866 = torch.ops.aten.unsqueeze.default(unsqueeze_865, 3);  unsqueeze_865 = None
        mul_1036 = torch.ops.aten.mul.Tensor(sub_252, unsqueeze_863);  sub_252 = unsqueeze_863 = None
        sub_254 = torch.ops.aten.sub.Tensor(where_39, mul_1036);  where_39 = mul_1036 = None
        sub_255 = torch.ops.aten.sub.Tensor(sub_254, unsqueeze_860);  sub_254 = unsqueeze_860 = None
        mul_1037 = torch.ops.aten.mul.Tensor(sub_255, unsqueeze_866);  sub_255 = unsqueeze_866 = None
        mul_1038 = torch.ops.aten.mul.Tensor(sum_82, squeeze_169);  sum_82 = squeeze_169 = None
        convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_1037, relu_55, primals_169, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1037 = primals_169 = None
        getitem_317 = convolution_backward_39[0]
        getitem_318 = convolution_backward_39[1];  convolution_backward_39 = None
        alias_354 = torch.ops.aten.alias.default(relu_55);  relu_55 = None
        alias_355 = torch.ops.aten.alias.default(alias_354);  alias_354 = None
        le_40 = torch.ops.aten.le.Scalar(alias_355, 0);  alias_355 = None
        where_40 = torch.ops.aten.where.self(le_40, full_default, getitem_317);  le_40 = getitem_317 = None
        sum_83 = torch.ops.aten.sum.dim_IntList(where_40, [0, 2, 3])
        sub_256 = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_869);  convolution_55 = unsqueeze_869 = None
        mul_1039 = torch.ops.aten.mul.Tensor(where_40, sub_256)
        sum_84 = torch.ops.aten.sum.dim_IntList(mul_1039, [0, 2, 3]);  mul_1039 = None
        mul_1040 = torch.ops.aten.mul.Tensor(sum_83, 0.00010813148788927336)
        unsqueeze_870 = torch.ops.aten.unsqueeze.default(mul_1040, 0);  mul_1040 = None
        unsqueeze_871 = torch.ops.aten.unsqueeze.default(unsqueeze_870, 2);  unsqueeze_870 = None
        unsqueeze_872 = torch.ops.aten.unsqueeze.default(unsqueeze_871, 3);  unsqueeze_871 = None
        mul_1041 = torch.ops.aten.mul.Tensor(sum_84, 0.00010813148788927336)
        mul_1042 = torch.ops.aten.mul.Tensor(squeeze_166, squeeze_166)
        mul_1043 = torch.ops.aten.mul.Tensor(mul_1041, mul_1042);  mul_1041 = mul_1042 = None
        unsqueeze_873 = torch.ops.aten.unsqueeze.default(mul_1043, 0);  mul_1043 = None
        unsqueeze_874 = torch.ops.aten.unsqueeze.default(unsqueeze_873, 2);  unsqueeze_873 = None
        unsqueeze_875 = torch.ops.aten.unsqueeze.default(unsqueeze_874, 3);  unsqueeze_874 = None
        mul_1044 = torch.ops.aten.mul.Tensor(squeeze_166, primals_167);  primals_167 = None
        unsqueeze_876 = torch.ops.aten.unsqueeze.default(mul_1044, 0);  mul_1044 = None
        unsqueeze_877 = torch.ops.aten.unsqueeze.default(unsqueeze_876, 2);  unsqueeze_876 = None
        unsqueeze_878 = torch.ops.aten.unsqueeze.default(unsqueeze_877, 3);  unsqueeze_877 = None
        mul_1045 = torch.ops.aten.mul.Tensor(sub_256, unsqueeze_875);  sub_256 = unsqueeze_875 = None
        sub_258 = torch.ops.aten.sub.Tensor(where_40, mul_1045);  where_40 = mul_1045 = None
        sub_259 = torch.ops.aten.sub.Tensor(sub_258, unsqueeze_872);  sub_258 = unsqueeze_872 = None
        mul_1046 = torch.ops.aten.mul.Tensor(sub_259, unsqueeze_878);  sub_259 = unsqueeze_878 = None
        mul_1047 = torch.ops.aten.mul.Tensor(sum_84, squeeze_166);  sum_84 = squeeze_166 = None
        convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_1046, relu_54, primals_166, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1046 = primals_166 = None
        getitem_320 = convolution_backward_40[0]
        getitem_321 = convolution_backward_40[1];  convolution_backward_40 = None
        alias_358 = torch.ops.aten.alias.default(relu_54);  relu_54 = None
        alias_359 = torch.ops.aten.alias.default(alias_358);  alias_358 = None
        le_41 = torch.ops.aten.le.Scalar(alias_359, 0);  alias_359 = None
        where_41 = torch.ops.aten.where.self(le_41, full_default, getitem_320);  le_41 = getitem_320 = None
        sum_85 = torch.ops.aten.sum.dim_IntList(where_41, [0, 2, 3])
        sub_260 = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_881);  convolution_54 = unsqueeze_881 = None
        mul_1048 = torch.ops.aten.mul.Tensor(where_41, sub_260)
        sum_86 = torch.ops.aten.sum.dim_IntList(mul_1048, [0, 2, 3]);  mul_1048 = None
        mul_1049 = torch.ops.aten.mul.Tensor(sum_85, 0.00010813148788927336)
        unsqueeze_882 = torch.ops.aten.unsqueeze.default(mul_1049, 0);  mul_1049 = None
        unsqueeze_883 = torch.ops.aten.unsqueeze.default(unsqueeze_882, 2);  unsqueeze_882 = None
        unsqueeze_884 = torch.ops.aten.unsqueeze.default(unsqueeze_883, 3);  unsqueeze_883 = None
        mul_1050 = torch.ops.aten.mul.Tensor(sum_86, 0.00010813148788927336)
        mul_1051 = torch.ops.aten.mul.Tensor(squeeze_163, squeeze_163)
        mul_1052 = torch.ops.aten.mul.Tensor(mul_1050, mul_1051);  mul_1050 = mul_1051 = None
        unsqueeze_885 = torch.ops.aten.unsqueeze.default(mul_1052, 0);  mul_1052 = None
        unsqueeze_886 = torch.ops.aten.unsqueeze.default(unsqueeze_885, 2);  unsqueeze_885 = None
        unsqueeze_887 = torch.ops.aten.unsqueeze.default(unsqueeze_886, 3);  unsqueeze_886 = None
        mul_1053 = torch.ops.aten.mul.Tensor(squeeze_163, primals_164);  primals_164 = None
        unsqueeze_888 = torch.ops.aten.unsqueeze.default(mul_1053, 0);  mul_1053 = None
        unsqueeze_889 = torch.ops.aten.unsqueeze.default(unsqueeze_888, 2);  unsqueeze_888 = None
        unsqueeze_890 = torch.ops.aten.unsqueeze.default(unsqueeze_889, 3);  unsqueeze_889 = None
        mul_1054 = torch.ops.aten.mul.Tensor(sub_260, unsqueeze_887);  sub_260 = unsqueeze_887 = None
        sub_262 = torch.ops.aten.sub.Tensor(where_41, mul_1054);  where_41 = mul_1054 = None
        sub_263 = torch.ops.aten.sub.Tensor(sub_262, unsqueeze_884);  sub_262 = unsqueeze_884 = None
        mul_1055 = torch.ops.aten.mul.Tensor(sub_263, unsqueeze_890);  sub_263 = unsqueeze_890 = None
        mul_1056 = torch.ops.aten.mul.Tensor(sum_86, squeeze_163);  sum_86 = squeeze_163 = None
        convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_1055, cat_6, primals_163, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1055 = primals_163 = None
        getitem_323 = convolution_backward_41[0]
        getitem_324 = convolution_backward_41[1];  convolution_backward_41 = None
        add_499 = torch.ops.aten.add.Tensor(avg_pool2d_backward_4, getitem_323);  avg_pool2d_backward_4 = getitem_323 = None
        where_42 = torch.ops.aten.where.self(le_42, full_default, slice_28);  le_42 = slice_28 = None
        sum_87 = torch.ops.aten.sum.dim_IntList(where_42, [0, 2, 3])
        sub_264 = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_893);  convolution_53 = unsqueeze_893 = None
        mul_1057 = torch.ops.aten.mul.Tensor(where_42, sub_264)
        sum_88 = torch.ops.aten.sum.dim_IntList(mul_1057, [0, 2, 3]);  mul_1057 = None
        mul_1058 = torch.ops.aten.mul.Tensor(sum_87, 0.00010813148788927336)
        unsqueeze_894 = torch.ops.aten.unsqueeze.default(mul_1058, 0);  mul_1058 = None
        unsqueeze_895 = torch.ops.aten.unsqueeze.default(unsqueeze_894, 2);  unsqueeze_894 = None
        unsqueeze_896 = torch.ops.aten.unsqueeze.default(unsqueeze_895, 3);  unsqueeze_895 = None
        mul_1059 = torch.ops.aten.mul.Tensor(sum_88, 0.00010813148788927336)
        mul_1060 = torch.ops.aten.mul.Tensor(squeeze_160, squeeze_160)
        mul_1061 = torch.ops.aten.mul.Tensor(mul_1059, mul_1060);  mul_1059 = mul_1060 = None
        unsqueeze_897 = torch.ops.aten.unsqueeze.default(mul_1061, 0);  mul_1061 = None
        unsqueeze_898 = torch.ops.aten.unsqueeze.default(unsqueeze_897, 2);  unsqueeze_897 = None
        unsqueeze_899 = torch.ops.aten.unsqueeze.default(unsqueeze_898, 3);  unsqueeze_898 = None
        mul_1062 = torch.ops.aten.mul.Tensor(squeeze_160, primals_161);  primals_161 = None
        unsqueeze_900 = torch.ops.aten.unsqueeze.default(mul_1062, 0);  mul_1062 = None
        unsqueeze_901 = torch.ops.aten.unsqueeze.default(unsqueeze_900, 2);  unsqueeze_900 = None
        unsqueeze_902 = torch.ops.aten.unsqueeze.default(unsqueeze_901, 3);  unsqueeze_901 = None
        mul_1063 = torch.ops.aten.mul.Tensor(sub_264, unsqueeze_899);  sub_264 = unsqueeze_899 = None
        sub_266 = torch.ops.aten.sub.Tensor(where_42, mul_1063);  where_42 = mul_1063 = None
        sub_267 = torch.ops.aten.sub.Tensor(sub_266, unsqueeze_896);  sub_266 = unsqueeze_896 = None
        mul_1064 = torch.ops.aten.mul.Tensor(sub_267, unsqueeze_902);  sub_267 = unsqueeze_902 = None
        mul_1065 = torch.ops.aten.mul.Tensor(sum_88, squeeze_160);  sum_88 = squeeze_160 = None
        convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_1064, relu_52, primals_160, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1064 = primals_160 = None
        getitem_326 = convolution_backward_42[0]
        getitem_327 = convolution_backward_42[1];  convolution_backward_42 = None
        alias_366 = torch.ops.aten.alias.default(relu_52);  relu_52 = None
        alias_367 = torch.ops.aten.alias.default(alias_366);  alias_366 = None
        le_43 = torch.ops.aten.le.Scalar(alias_367, 0);  alias_367 = None
        where_43 = torch.ops.aten.where.self(le_43, full_default, getitem_326);  le_43 = getitem_326 = None
        sum_89 = torch.ops.aten.sum.dim_IntList(where_43, [0, 2, 3])
        sub_268 = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_905);  convolution_52 = unsqueeze_905 = None
        mul_1066 = torch.ops.aten.mul.Tensor(where_43, sub_268)
        sum_90 = torch.ops.aten.sum.dim_IntList(mul_1066, [0, 2, 3]);  mul_1066 = None
        mul_1067 = torch.ops.aten.mul.Tensor(sum_89, 0.00010813148788927336)
        unsqueeze_906 = torch.ops.aten.unsqueeze.default(mul_1067, 0);  mul_1067 = None
        unsqueeze_907 = torch.ops.aten.unsqueeze.default(unsqueeze_906, 2);  unsqueeze_906 = None
        unsqueeze_908 = torch.ops.aten.unsqueeze.default(unsqueeze_907, 3);  unsqueeze_907 = None
        mul_1068 = torch.ops.aten.mul.Tensor(sum_90, 0.00010813148788927336)
        mul_1069 = torch.ops.aten.mul.Tensor(squeeze_157, squeeze_157)
        mul_1070 = torch.ops.aten.mul.Tensor(mul_1068, mul_1069);  mul_1068 = mul_1069 = None
        unsqueeze_909 = torch.ops.aten.unsqueeze.default(mul_1070, 0);  mul_1070 = None
        unsqueeze_910 = torch.ops.aten.unsqueeze.default(unsqueeze_909, 2);  unsqueeze_909 = None
        unsqueeze_911 = torch.ops.aten.unsqueeze.default(unsqueeze_910, 3);  unsqueeze_910 = None
        mul_1071 = torch.ops.aten.mul.Tensor(squeeze_157, primals_158);  primals_158 = None
        unsqueeze_912 = torch.ops.aten.unsqueeze.default(mul_1071, 0);  mul_1071 = None
        unsqueeze_913 = torch.ops.aten.unsqueeze.default(unsqueeze_912, 2);  unsqueeze_912 = None
        unsqueeze_914 = torch.ops.aten.unsqueeze.default(unsqueeze_913, 3);  unsqueeze_913 = None
        mul_1072 = torch.ops.aten.mul.Tensor(sub_268, unsqueeze_911);  sub_268 = unsqueeze_911 = None
        sub_270 = torch.ops.aten.sub.Tensor(where_43, mul_1072);  where_43 = mul_1072 = None
        sub_271 = torch.ops.aten.sub.Tensor(sub_270, unsqueeze_908);  sub_270 = unsqueeze_908 = None
        mul_1073 = torch.ops.aten.mul.Tensor(sub_271, unsqueeze_914);  sub_271 = unsqueeze_914 = None
        mul_1074 = torch.ops.aten.mul.Tensor(sum_90, squeeze_157);  sum_90 = squeeze_157 = None
        convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_1073, relu_51, primals_157, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1073 = primals_157 = None
        getitem_329 = convolution_backward_43[0]
        getitem_330 = convolution_backward_43[1];  convolution_backward_43 = None
        alias_370 = torch.ops.aten.alias.default(relu_51);  relu_51 = None
        alias_371 = torch.ops.aten.alias.default(alias_370);  alias_370 = None
        le_44 = torch.ops.aten.le.Scalar(alias_371, 0);  alias_371 = None
        where_44 = torch.ops.aten.where.self(le_44, full_default, getitem_329);  le_44 = getitem_329 = None
        sum_91 = torch.ops.aten.sum.dim_IntList(where_44, [0, 2, 3])
        sub_272 = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_917);  convolution_51 = unsqueeze_917 = None
        mul_1075 = torch.ops.aten.mul.Tensor(where_44, sub_272)
        sum_92 = torch.ops.aten.sum.dim_IntList(mul_1075, [0, 2, 3]);  mul_1075 = None
        mul_1076 = torch.ops.aten.mul.Tensor(sum_91, 0.00010813148788927336)
        unsqueeze_918 = torch.ops.aten.unsqueeze.default(mul_1076, 0);  mul_1076 = None
        unsqueeze_919 = torch.ops.aten.unsqueeze.default(unsqueeze_918, 2);  unsqueeze_918 = None
        unsqueeze_920 = torch.ops.aten.unsqueeze.default(unsqueeze_919, 3);  unsqueeze_919 = None
        mul_1077 = torch.ops.aten.mul.Tensor(sum_92, 0.00010813148788927336)
        mul_1078 = torch.ops.aten.mul.Tensor(squeeze_154, squeeze_154)
        mul_1079 = torch.ops.aten.mul.Tensor(mul_1077, mul_1078);  mul_1077 = mul_1078 = None
        unsqueeze_921 = torch.ops.aten.unsqueeze.default(mul_1079, 0);  mul_1079 = None
        unsqueeze_922 = torch.ops.aten.unsqueeze.default(unsqueeze_921, 2);  unsqueeze_921 = None
        unsqueeze_923 = torch.ops.aten.unsqueeze.default(unsqueeze_922, 3);  unsqueeze_922 = None
        mul_1080 = torch.ops.aten.mul.Tensor(squeeze_154, primals_155);  primals_155 = None
        unsqueeze_924 = torch.ops.aten.unsqueeze.default(mul_1080, 0);  mul_1080 = None
        unsqueeze_925 = torch.ops.aten.unsqueeze.default(unsqueeze_924, 2);  unsqueeze_924 = None
        unsqueeze_926 = torch.ops.aten.unsqueeze.default(unsqueeze_925, 3);  unsqueeze_925 = None
        mul_1081 = torch.ops.aten.mul.Tensor(sub_272, unsqueeze_923);  sub_272 = unsqueeze_923 = None
        sub_274 = torch.ops.aten.sub.Tensor(where_44, mul_1081);  where_44 = mul_1081 = None
        sub_275 = torch.ops.aten.sub.Tensor(sub_274, unsqueeze_920);  sub_274 = unsqueeze_920 = None
        mul_1082 = torch.ops.aten.mul.Tensor(sub_275, unsqueeze_926);  sub_275 = unsqueeze_926 = None
        mul_1083 = torch.ops.aten.mul.Tensor(sum_92, squeeze_154);  sum_92 = squeeze_154 = None
        convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_1082, cat_6, primals_154, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1082 = primals_154 = None
        getitem_332 = convolution_backward_44[0]
        getitem_333 = convolution_backward_44[1];  convolution_backward_44 = None
        add_500 = torch.ops.aten.add.Tensor(add_499, getitem_332);  add_499 = getitem_332 = None
        where_45 = torch.ops.aten.where.self(le_45, full_default, slice_27);  le_45 = slice_27 = None
        sum_93 = torch.ops.aten.sum.dim_IntList(where_45, [0, 2, 3])
        sub_276 = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_929);  convolution_50 = unsqueeze_929 = None
        mul_1084 = torch.ops.aten.mul.Tensor(where_45, sub_276)
        sum_94 = torch.ops.aten.sum.dim_IntList(mul_1084, [0, 2, 3]);  mul_1084 = None
        mul_1085 = torch.ops.aten.mul.Tensor(sum_93, 0.00010813148788927336)
        unsqueeze_930 = torch.ops.aten.unsqueeze.default(mul_1085, 0);  mul_1085 = None
        unsqueeze_931 = torch.ops.aten.unsqueeze.default(unsqueeze_930, 2);  unsqueeze_930 = None
        unsqueeze_932 = torch.ops.aten.unsqueeze.default(unsqueeze_931, 3);  unsqueeze_931 = None
        mul_1086 = torch.ops.aten.mul.Tensor(sum_94, 0.00010813148788927336)
        mul_1087 = torch.ops.aten.mul.Tensor(squeeze_151, squeeze_151)
        mul_1088 = torch.ops.aten.mul.Tensor(mul_1086, mul_1087);  mul_1086 = mul_1087 = None
        unsqueeze_933 = torch.ops.aten.unsqueeze.default(mul_1088, 0);  mul_1088 = None
        unsqueeze_934 = torch.ops.aten.unsqueeze.default(unsqueeze_933, 2);  unsqueeze_933 = None
        unsqueeze_935 = torch.ops.aten.unsqueeze.default(unsqueeze_934, 3);  unsqueeze_934 = None
        mul_1089 = torch.ops.aten.mul.Tensor(squeeze_151, primals_152);  primals_152 = None
        unsqueeze_936 = torch.ops.aten.unsqueeze.default(mul_1089, 0);  mul_1089 = None
        unsqueeze_937 = torch.ops.aten.unsqueeze.default(unsqueeze_936, 2);  unsqueeze_936 = None
        unsqueeze_938 = torch.ops.aten.unsqueeze.default(unsqueeze_937, 3);  unsqueeze_937 = None
        mul_1090 = torch.ops.aten.mul.Tensor(sub_276, unsqueeze_935);  sub_276 = unsqueeze_935 = None
        sub_278 = torch.ops.aten.sub.Tensor(where_45, mul_1090);  where_45 = mul_1090 = None
        sub_279 = torch.ops.aten.sub.Tensor(sub_278, unsqueeze_932);  sub_278 = unsqueeze_932 = None
        mul_1091 = torch.ops.aten.mul.Tensor(sub_279, unsqueeze_938);  sub_279 = unsqueeze_938 = None
        mul_1092 = torch.ops.aten.mul.Tensor(sum_94, squeeze_151);  sum_94 = squeeze_151 = None
        convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_1091, cat_6, primals_151, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1091 = cat_6 = primals_151 = None
        getitem_335 = convolution_backward_45[0]
        getitem_336 = convolution_backward_45[1];  convolution_backward_45 = None
        add_501 = torch.ops.aten.add.Tensor(add_500, getitem_335);  add_500 = getitem_335 = None
        slice_31 = torch.ops.aten.slice.Tensor(add_501, 1, 0, 192)
        slice_32 = torch.ops.aten.slice.Tensor(add_501, 1, 192, 384)
        slice_33 = torch.ops.aten.slice.Tensor(add_501, 1, 384, 576)
        slice_34 = torch.ops.aten.slice.Tensor(add_501, 1, 576, 768);  add_501 = None
        where_46 = torch.ops.aten.where.self(le_46, full_default, slice_34);  le_46 = slice_34 = None
        sum_95 = torch.ops.aten.sum.dim_IntList(where_46, [0, 2, 3])
        sub_280 = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_941);  convolution_49 = unsqueeze_941 = None
        mul_1093 = torch.ops.aten.mul.Tensor(where_46, sub_280)
        sum_96 = torch.ops.aten.sum.dim_IntList(mul_1093, [0, 2, 3]);  mul_1093 = None
        mul_1094 = torch.ops.aten.mul.Tensor(sum_95, 0.00010813148788927336)
        unsqueeze_942 = torch.ops.aten.unsqueeze.default(mul_1094, 0);  mul_1094 = None
        unsqueeze_943 = torch.ops.aten.unsqueeze.default(unsqueeze_942, 2);  unsqueeze_942 = None
        unsqueeze_944 = torch.ops.aten.unsqueeze.default(unsqueeze_943, 3);  unsqueeze_943 = None
        mul_1095 = torch.ops.aten.mul.Tensor(sum_96, 0.00010813148788927336)
        mul_1096 = torch.ops.aten.mul.Tensor(squeeze_148, squeeze_148)
        mul_1097 = torch.ops.aten.mul.Tensor(mul_1095, mul_1096);  mul_1095 = mul_1096 = None
        unsqueeze_945 = torch.ops.aten.unsqueeze.default(mul_1097, 0);  mul_1097 = None
        unsqueeze_946 = torch.ops.aten.unsqueeze.default(unsqueeze_945, 2);  unsqueeze_945 = None
        unsqueeze_947 = torch.ops.aten.unsqueeze.default(unsqueeze_946, 3);  unsqueeze_946 = None
        mul_1098 = torch.ops.aten.mul.Tensor(squeeze_148, primals_149);  primals_149 = None
        unsqueeze_948 = torch.ops.aten.unsqueeze.default(mul_1098, 0);  mul_1098 = None
        unsqueeze_949 = torch.ops.aten.unsqueeze.default(unsqueeze_948, 2);  unsqueeze_948 = None
        unsqueeze_950 = torch.ops.aten.unsqueeze.default(unsqueeze_949, 3);  unsqueeze_949 = None
        mul_1099 = torch.ops.aten.mul.Tensor(sub_280, unsqueeze_947);  sub_280 = unsqueeze_947 = None
        sub_282 = torch.ops.aten.sub.Tensor(where_46, mul_1099);  where_46 = mul_1099 = None
        sub_283 = torch.ops.aten.sub.Tensor(sub_282, unsqueeze_944);  sub_282 = unsqueeze_944 = None
        mul_1100 = torch.ops.aten.mul.Tensor(sub_283, unsqueeze_950);  sub_283 = unsqueeze_950 = None
        mul_1101 = torch.ops.aten.mul.Tensor(sum_96, squeeze_148);  sum_96 = squeeze_148 = None
        convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_1100, avg_pool2d_4, primals_148, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1100 = avg_pool2d_4 = primals_148 = None
        getitem_338 = convolution_backward_46[0]
        getitem_339 = convolution_backward_46[1];  convolution_backward_46 = None
        avg_pool2d_backward_5 = torch.ops.aten.avg_pool2d_backward.default(getitem_338, cat_5, [3, 3], [1, 1], [1, 1], False, True, None);  getitem_338 = None
        where_47 = torch.ops.aten.where.self(le_47, full_default, slice_33);  le_47 = slice_33 = None
        sum_97 = torch.ops.aten.sum.dim_IntList(where_47, [0, 2, 3])
        sub_284 = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_953);  convolution_48 = unsqueeze_953 = None
        mul_1102 = torch.ops.aten.mul.Tensor(where_47, sub_284)
        sum_98 = torch.ops.aten.sum.dim_IntList(mul_1102, [0, 2, 3]);  mul_1102 = None
        mul_1103 = torch.ops.aten.mul.Tensor(sum_97, 0.00010813148788927336)
        unsqueeze_954 = torch.ops.aten.unsqueeze.default(mul_1103, 0);  mul_1103 = None
        unsqueeze_955 = torch.ops.aten.unsqueeze.default(unsqueeze_954, 2);  unsqueeze_954 = None
        unsqueeze_956 = torch.ops.aten.unsqueeze.default(unsqueeze_955, 3);  unsqueeze_955 = None
        mul_1104 = torch.ops.aten.mul.Tensor(sum_98, 0.00010813148788927336)
        mul_1105 = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
        mul_1106 = torch.ops.aten.mul.Tensor(mul_1104, mul_1105);  mul_1104 = mul_1105 = None
        unsqueeze_957 = torch.ops.aten.unsqueeze.default(mul_1106, 0);  mul_1106 = None
        unsqueeze_958 = torch.ops.aten.unsqueeze.default(unsqueeze_957, 2);  unsqueeze_957 = None
        unsqueeze_959 = torch.ops.aten.unsqueeze.default(unsqueeze_958, 3);  unsqueeze_958 = None
        mul_1107 = torch.ops.aten.mul.Tensor(squeeze_145, primals_146);  primals_146 = None
        unsqueeze_960 = torch.ops.aten.unsqueeze.default(mul_1107, 0);  mul_1107 = None
        unsqueeze_961 = torch.ops.aten.unsqueeze.default(unsqueeze_960, 2);  unsqueeze_960 = None
        unsqueeze_962 = torch.ops.aten.unsqueeze.default(unsqueeze_961, 3);  unsqueeze_961 = None
        mul_1108 = torch.ops.aten.mul.Tensor(sub_284, unsqueeze_959);  sub_284 = unsqueeze_959 = None
        sub_286 = torch.ops.aten.sub.Tensor(where_47, mul_1108);  where_47 = mul_1108 = None
        sub_287 = torch.ops.aten.sub.Tensor(sub_286, unsqueeze_956);  sub_286 = unsqueeze_956 = None
        mul_1109 = torch.ops.aten.mul.Tensor(sub_287, unsqueeze_962);  sub_287 = unsqueeze_962 = None
        mul_1110 = torch.ops.aten.mul.Tensor(sum_98, squeeze_145);  sum_98 = squeeze_145 = None
        convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_1109, relu_47, primals_145, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1109 = primals_145 = None
        getitem_341 = convolution_backward_47[0]
        getitem_342 = convolution_backward_47[1];  convolution_backward_47 = None
        alias_386 = torch.ops.aten.alias.default(relu_47);  relu_47 = None
        alias_387 = torch.ops.aten.alias.default(alias_386);  alias_386 = None
        le_48 = torch.ops.aten.le.Scalar(alias_387, 0);  alias_387 = None
        where_48 = torch.ops.aten.where.self(le_48, full_default, getitem_341);  le_48 = getitem_341 = None
        sum_99 = torch.ops.aten.sum.dim_IntList(where_48, [0, 2, 3])
        sub_288 = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_965);  convolution_47 = unsqueeze_965 = None
        mul_1111 = torch.ops.aten.mul.Tensor(where_48, sub_288)
        sum_100 = torch.ops.aten.sum.dim_IntList(mul_1111, [0, 2, 3]);  mul_1111 = None
        mul_1112 = torch.ops.aten.mul.Tensor(sum_99, 0.00010813148788927336)
        unsqueeze_966 = torch.ops.aten.unsqueeze.default(mul_1112, 0);  mul_1112 = None
        unsqueeze_967 = torch.ops.aten.unsqueeze.default(unsqueeze_966, 2);  unsqueeze_966 = None
        unsqueeze_968 = torch.ops.aten.unsqueeze.default(unsqueeze_967, 3);  unsqueeze_967 = None
        mul_1113 = torch.ops.aten.mul.Tensor(sum_100, 0.00010813148788927336)
        mul_1114 = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
        mul_1115 = torch.ops.aten.mul.Tensor(mul_1113, mul_1114);  mul_1113 = mul_1114 = None
        unsqueeze_969 = torch.ops.aten.unsqueeze.default(mul_1115, 0);  mul_1115 = None
        unsqueeze_970 = torch.ops.aten.unsqueeze.default(unsqueeze_969, 2);  unsqueeze_969 = None
        unsqueeze_971 = torch.ops.aten.unsqueeze.default(unsqueeze_970, 3);  unsqueeze_970 = None
        mul_1116 = torch.ops.aten.mul.Tensor(squeeze_142, primals_143);  primals_143 = None
        unsqueeze_972 = torch.ops.aten.unsqueeze.default(mul_1116, 0);  mul_1116 = None
        unsqueeze_973 = torch.ops.aten.unsqueeze.default(unsqueeze_972, 2);  unsqueeze_972 = None
        unsqueeze_974 = torch.ops.aten.unsqueeze.default(unsqueeze_973, 3);  unsqueeze_973 = None
        mul_1117 = torch.ops.aten.mul.Tensor(sub_288, unsqueeze_971);  sub_288 = unsqueeze_971 = None
        sub_290 = torch.ops.aten.sub.Tensor(where_48, mul_1117);  where_48 = mul_1117 = None
        sub_291 = torch.ops.aten.sub.Tensor(sub_290, unsqueeze_968);  sub_290 = unsqueeze_968 = None
        mul_1118 = torch.ops.aten.mul.Tensor(sub_291, unsqueeze_974);  sub_291 = unsqueeze_974 = None
        mul_1119 = torch.ops.aten.mul.Tensor(sum_100, squeeze_142);  sum_100 = squeeze_142 = None
        convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_1118, relu_46, primals_142, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1118 = primals_142 = None
        getitem_344 = convolution_backward_48[0]
        getitem_345 = convolution_backward_48[1];  convolution_backward_48 = None
        alias_390 = torch.ops.aten.alias.default(relu_46);  relu_46 = None
        alias_391 = torch.ops.aten.alias.default(alias_390);  alias_390 = None
        le_49 = torch.ops.aten.le.Scalar(alias_391, 0);  alias_391 = None
        where_49 = torch.ops.aten.where.self(le_49, full_default, getitem_344);  le_49 = getitem_344 = None
        sum_101 = torch.ops.aten.sum.dim_IntList(where_49, [0, 2, 3])
        sub_292 = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_977);  convolution_46 = unsqueeze_977 = None
        mul_1120 = torch.ops.aten.mul.Tensor(where_49, sub_292)
        sum_102 = torch.ops.aten.sum.dim_IntList(mul_1120, [0, 2, 3]);  mul_1120 = None
        mul_1121 = torch.ops.aten.mul.Tensor(sum_101, 0.00010813148788927336)
        unsqueeze_978 = torch.ops.aten.unsqueeze.default(mul_1121, 0);  mul_1121 = None
        unsqueeze_979 = torch.ops.aten.unsqueeze.default(unsqueeze_978, 2);  unsqueeze_978 = None
        unsqueeze_980 = torch.ops.aten.unsqueeze.default(unsqueeze_979, 3);  unsqueeze_979 = None
        mul_1122 = torch.ops.aten.mul.Tensor(sum_102, 0.00010813148788927336)
        mul_1123 = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
        mul_1124 = torch.ops.aten.mul.Tensor(mul_1122, mul_1123);  mul_1122 = mul_1123 = None
        unsqueeze_981 = torch.ops.aten.unsqueeze.default(mul_1124, 0);  mul_1124 = None
        unsqueeze_982 = torch.ops.aten.unsqueeze.default(unsqueeze_981, 2);  unsqueeze_981 = None
        unsqueeze_983 = torch.ops.aten.unsqueeze.default(unsqueeze_982, 3);  unsqueeze_982 = None
        mul_1125 = torch.ops.aten.mul.Tensor(squeeze_139, primals_140);  primals_140 = None
        unsqueeze_984 = torch.ops.aten.unsqueeze.default(mul_1125, 0);  mul_1125 = None
        unsqueeze_985 = torch.ops.aten.unsqueeze.default(unsqueeze_984, 2);  unsqueeze_984 = None
        unsqueeze_986 = torch.ops.aten.unsqueeze.default(unsqueeze_985, 3);  unsqueeze_985 = None
        mul_1126 = torch.ops.aten.mul.Tensor(sub_292, unsqueeze_983);  sub_292 = unsqueeze_983 = None
        sub_294 = torch.ops.aten.sub.Tensor(where_49, mul_1126);  where_49 = mul_1126 = None
        sub_295 = torch.ops.aten.sub.Tensor(sub_294, unsqueeze_980);  sub_294 = unsqueeze_980 = None
        mul_1127 = torch.ops.aten.mul.Tensor(sub_295, unsqueeze_986);  sub_295 = unsqueeze_986 = None
        mul_1128 = torch.ops.aten.mul.Tensor(sum_102, squeeze_139);  sum_102 = squeeze_139 = None
        convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_1127, relu_45, primals_139, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1127 = primals_139 = None
        getitem_347 = convolution_backward_49[0]
        getitem_348 = convolution_backward_49[1];  convolution_backward_49 = None
        alias_394 = torch.ops.aten.alias.default(relu_45);  relu_45 = None
        alias_395 = torch.ops.aten.alias.default(alias_394);  alias_394 = None
        le_50 = torch.ops.aten.le.Scalar(alias_395, 0);  alias_395 = None
        where_50 = torch.ops.aten.where.self(le_50, full_default, getitem_347);  le_50 = getitem_347 = None
        sum_103 = torch.ops.aten.sum.dim_IntList(where_50, [0, 2, 3])
        sub_296 = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_989);  convolution_45 = unsqueeze_989 = None
        mul_1129 = torch.ops.aten.mul.Tensor(where_50, sub_296)
        sum_104 = torch.ops.aten.sum.dim_IntList(mul_1129, [0, 2, 3]);  mul_1129 = None
        mul_1130 = torch.ops.aten.mul.Tensor(sum_103, 0.00010813148788927336)
        unsqueeze_990 = torch.ops.aten.unsqueeze.default(mul_1130, 0);  mul_1130 = None
        unsqueeze_991 = torch.ops.aten.unsqueeze.default(unsqueeze_990, 2);  unsqueeze_990 = None
        unsqueeze_992 = torch.ops.aten.unsqueeze.default(unsqueeze_991, 3);  unsqueeze_991 = None
        mul_1131 = torch.ops.aten.mul.Tensor(sum_104, 0.00010813148788927336)
        mul_1132 = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
        mul_1133 = torch.ops.aten.mul.Tensor(mul_1131, mul_1132);  mul_1131 = mul_1132 = None
        unsqueeze_993 = torch.ops.aten.unsqueeze.default(mul_1133, 0);  mul_1133 = None
        unsqueeze_994 = torch.ops.aten.unsqueeze.default(unsqueeze_993, 2);  unsqueeze_993 = None
        unsqueeze_995 = torch.ops.aten.unsqueeze.default(unsqueeze_994, 3);  unsqueeze_994 = None
        mul_1134 = torch.ops.aten.mul.Tensor(squeeze_136, primals_137);  primals_137 = None
        unsqueeze_996 = torch.ops.aten.unsqueeze.default(mul_1134, 0);  mul_1134 = None
        unsqueeze_997 = torch.ops.aten.unsqueeze.default(unsqueeze_996, 2);  unsqueeze_996 = None
        unsqueeze_998 = torch.ops.aten.unsqueeze.default(unsqueeze_997, 3);  unsqueeze_997 = None
        mul_1135 = torch.ops.aten.mul.Tensor(sub_296, unsqueeze_995);  sub_296 = unsqueeze_995 = None
        sub_298 = torch.ops.aten.sub.Tensor(where_50, mul_1135);  where_50 = mul_1135 = None
        sub_299 = torch.ops.aten.sub.Tensor(sub_298, unsqueeze_992);  sub_298 = unsqueeze_992 = None
        mul_1136 = torch.ops.aten.mul.Tensor(sub_299, unsqueeze_998);  sub_299 = unsqueeze_998 = None
        mul_1137 = torch.ops.aten.mul.Tensor(sum_104, squeeze_136);  sum_104 = squeeze_136 = None
        convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_1136, relu_44, primals_136, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1136 = primals_136 = None
        getitem_350 = convolution_backward_50[0]
        getitem_351 = convolution_backward_50[1];  convolution_backward_50 = None
        alias_398 = torch.ops.aten.alias.default(relu_44);  relu_44 = None
        alias_399 = torch.ops.aten.alias.default(alias_398);  alias_398 = None
        le_51 = torch.ops.aten.le.Scalar(alias_399, 0);  alias_399 = None
        where_51 = torch.ops.aten.where.self(le_51, full_default, getitem_350);  le_51 = getitem_350 = None
        sum_105 = torch.ops.aten.sum.dim_IntList(where_51, [0, 2, 3])
        sub_300 = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_1001);  convolution_44 = unsqueeze_1001 = None
        mul_1138 = torch.ops.aten.mul.Tensor(where_51, sub_300)
        sum_106 = torch.ops.aten.sum.dim_IntList(mul_1138, [0, 2, 3]);  mul_1138 = None
        mul_1139 = torch.ops.aten.mul.Tensor(sum_105, 0.00010813148788927336)
        unsqueeze_1002 = torch.ops.aten.unsqueeze.default(mul_1139, 0);  mul_1139 = None
        unsqueeze_1003 = torch.ops.aten.unsqueeze.default(unsqueeze_1002, 2);  unsqueeze_1002 = None
        unsqueeze_1004 = torch.ops.aten.unsqueeze.default(unsqueeze_1003, 3);  unsqueeze_1003 = None
        mul_1140 = torch.ops.aten.mul.Tensor(sum_106, 0.00010813148788927336)
        mul_1141 = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
        mul_1142 = torch.ops.aten.mul.Tensor(mul_1140, mul_1141);  mul_1140 = mul_1141 = None
        unsqueeze_1005 = torch.ops.aten.unsqueeze.default(mul_1142, 0);  mul_1142 = None
        unsqueeze_1006 = torch.ops.aten.unsqueeze.default(unsqueeze_1005, 2);  unsqueeze_1005 = None
        unsqueeze_1007 = torch.ops.aten.unsqueeze.default(unsqueeze_1006, 3);  unsqueeze_1006 = None
        mul_1143 = torch.ops.aten.mul.Tensor(squeeze_133, primals_134);  primals_134 = None
        unsqueeze_1008 = torch.ops.aten.unsqueeze.default(mul_1143, 0);  mul_1143 = None
        unsqueeze_1009 = torch.ops.aten.unsqueeze.default(unsqueeze_1008, 2);  unsqueeze_1008 = None
        unsqueeze_1010 = torch.ops.aten.unsqueeze.default(unsqueeze_1009, 3);  unsqueeze_1009 = None
        mul_1144 = torch.ops.aten.mul.Tensor(sub_300, unsqueeze_1007);  sub_300 = unsqueeze_1007 = None
        sub_302 = torch.ops.aten.sub.Tensor(where_51, mul_1144);  where_51 = mul_1144 = None
        sub_303 = torch.ops.aten.sub.Tensor(sub_302, unsqueeze_1004);  sub_302 = unsqueeze_1004 = None
        mul_1145 = torch.ops.aten.mul.Tensor(sub_303, unsqueeze_1010);  sub_303 = unsqueeze_1010 = None
        mul_1146 = torch.ops.aten.mul.Tensor(sum_106, squeeze_133);  sum_106 = squeeze_133 = None
        convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_1145, cat_5, primals_133, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1145 = primals_133 = None
        getitem_353 = convolution_backward_51[0]
        getitem_354 = convolution_backward_51[1];  convolution_backward_51 = None
        add_502 = torch.ops.aten.add.Tensor(avg_pool2d_backward_5, getitem_353);  avg_pool2d_backward_5 = getitem_353 = None
        where_52 = torch.ops.aten.where.self(le_52, full_default, slice_32);  le_52 = slice_32 = None
        sum_107 = torch.ops.aten.sum.dim_IntList(where_52, [0, 2, 3])
        sub_304 = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_1013);  convolution_43 = unsqueeze_1013 = None
        mul_1147 = torch.ops.aten.mul.Tensor(where_52, sub_304)
        sum_108 = torch.ops.aten.sum.dim_IntList(mul_1147, [0, 2, 3]);  mul_1147 = None
        mul_1148 = torch.ops.aten.mul.Tensor(sum_107, 0.00010813148788927336)
        unsqueeze_1014 = torch.ops.aten.unsqueeze.default(mul_1148, 0);  mul_1148 = None
        unsqueeze_1015 = torch.ops.aten.unsqueeze.default(unsqueeze_1014, 2);  unsqueeze_1014 = None
        unsqueeze_1016 = torch.ops.aten.unsqueeze.default(unsqueeze_1015, 3);  unsqueeze_1015 = None
        mul_1149 = torch.ops.aten.mul.Tensor(sum_108, 0.00010813148788927336)
        mul_1150 = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
        mul_1151 = torch.ops.aten.mul.Tensor(mul_1149, mul_1150);  mul_1149 = mul_1150 = None
        unsqueeze_1017 = torch.ops.aten.unsqueeze.default(mul_1151, 0);  mul_1151 = None
        unsqueeze_1018 = torch.ops.aten.unsqueeze.default(unsqueeze_1017, 2);  unsqueeze_1017 = None
        unsqueeze_1019 = torch.ops.aten.unsqueeze.default(unsqueeze_1018, 3);  unsqueeze_1018 = None
        mul_1152 = torch.ops.aten.mul.Tensor(squeeze_130, primals_131);  primals_131 = None
        unsqueeze_1020 = torch.ops.aten.unsqueeze.default(mul_1152, 0);  mul_1152 = None
        unsqueeze_1021 = torch.ops.aten.unsqueeze.default(unsqueeze_1020, 2);  unsqueeze_1020 = None
        unsqueeze_1022 = torch.ops.aten.unsqueeze.default(unsqueeze_1021, 3);  unsqueeze_1021 = None
        mul_1153 = torch.ops.aten.mul.Tensor(sub_304, unsqueeze_1019);  sub_304 = unsqueeze_1019 = None
        sub_306 = torch.ops.aten.sub.Tensor(where_52, mul_1153);  where_52 = mul_1153 = None
        sub_307 = torch.ops.aten.sub.Tensor(sub_306, unsqueeze_1016);  sub_306 = unsqueeze_1016 = None
        mul_1154 = torch.ops.aten.mul.Tensor(sub_307, unsqueeze_1022);  sub_307 = unsqueeze_1022 = None
        mul_1155 = torch.ops.aten.mul.Tensor(sum_108, squeeze_130);  sum_108 = squeeze_130 = None
        convolution_backward_52 = torch.ops.aten.convolution_backward.default(mul_1154, relu_42, primals_130, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1154 = primals_130 = None
        getitem_356 = convolution_backward_52[0]
        getitem_357 = convolution_backward_52[1];  convolution_backward_52 = None
        alias_406 = torch.ops.aten.alias.default(relu_42);  relu_42 = None
        alias_407 = torch.ops.aten.alias.default(alias_406);  alias_406 = None
        le_53 = torch.ops.aten.le.Scalar(alias_407, 0);  alias_407 = None
        where_53 = torch.ops.aten.where.self(le_53, full_default, getitem_356);  le_53 = getitem_356 = None
        sum_109 = torch.ops.aten.sum.dim_IntList(where_53, [0, 2, 3])
        sub_308 = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_1025);  convolution_42 = unsqueeze_1025 = None
        mul_1156 = torch.ops.aten.mul.Tensor(where_53, sub_308)
        sum_110 = torch.ops.aten.sum.dim_IntList(mul_1156, [0, 2, 3]);  mul_1156 = None
        mul_1157 = torch.ops.aten.mul.Tensor(sum_109, 0.00010813148788927336)
        unsqueeze_1026 = torch.ops.aten.unsqueeze.default(mul_1157, 0);  mul_1157 = None
        unsqueeze_1027 = torch.ops.aten.unsqueeze.default(unsqueeze_1026, 2);  unsqueeze_1026 = None
        unsqueeze_1028 = torch.ops.aten.unsqueeze.default(unsqueeze_1027, 3);  unsqueeze_1027 = None
        mul_1158 = torch.ops.aten.mul.Tensor(sum_110, 0.00010813148788927336)
        mul_1159 = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
        mul_1160 = torch.ops.aten.mul.Tensor(mul_1158, mul_1159);  mul_1158 = mul_1159 = None
        unsqueeze_1029 = torch.ops.aten.unsqueeze.default(mul_1160, 0);  mul_1160 = None
        unsqueeze_1030 = torch.ops.aten.unsqueeze.default(unsqueeze_1029, 2);  unsqueeze_1029 = None
        unsqueeze_1031 = torch.ops.aten.unsqueeze.default(unsqueeze_1030, 3);  unsqueeze_1030 = None
        mul_1161 = torch.ops.aten.mul.Tensor(squeeze_127, primals_128);  primals_128 = None
        unsqueeze_1032 = torch.ops.aten.unsqueeze.default(mul_1161, 0);  mul_1161 = None
        unsqueeze_1033 = torch.ops.aten.unsqueeze.default(unsqueeze_1032, 2);  unsqueeze_1032 = None
        unsqueeze_1034 = torch.ops.aten.unsqueeze.default(unsqueeze_1033, 3);  unsqueeze_1033 = None
        mul_1162 = torch.ops.aten.mul.Tensor(sub_308, unsqueeze_1031);  sub_308 = unsqueeze_1031 = None
        sub_310 = torch.ops.aten.sub.Tensor(where_53, mul_1162);  where_53 = mul_1162 = None
        sub_311 = torch.ops.aten.sub.Tensor(sub_310, unsqueeze_1028);  sub_310 = unsqueeze_1028 = None
        mul_1163 = torch.ops.aten.mul.Tensor(sub_311, unsqueeze_1034);  sub_311 = unsqueeze_1034 = None
        mul_1164 = torch.ops.aten.mul.Tensor(sum_110, squeeze_127);  sum_110 = squeeze_127 = None
        convolution_backward_53 = torch.ops.aten.convolution_backward.default(mul_1163, relu_41, primals_127, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1163 = primals_127 = None
        getitem_359 = convolution_backward_53[0]
        getitem_360 = convolution_backward_53[1];  convolution_backward_53 = None
        alias_410 = torch.ops.aten.alias.default(relu_41);  relu_41 = None
        alias_411 = torch.ops.aten.alias.default(alias_410);  alias_410 = None
        le_54 = torch.ops.aten.le.Scalar(alias_411, 0);  alias_411 = None
        where_54 = torch.ops.aten.where.self(le_54, full_default, getitem_359);  le_54 = getitem_359 = None
        sum_111 = torch.ops.aten.sum.dim_IntList(where_54, [0, 2, 3])
        sub_312 = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_1037);  convolution_41 = unsqueeze_1037 = None
        mul_1165 = torch.ops.aten.mul.Tensor(where_54, sub_312)
        sum_112 = torch.ops.aten.sum.dim_IntList(mul_1165, [0, 2, 3]);  mul_1165 = None
        mul_1166 = torch.ops.aten.mul.Tensor(sum_111, 0.00010813148788927336)
        unsqueeze_1038 = torch.ops.aten.unsqueeze.default(mul_1166, 0);  mul_1166 = None
        unsqueeze_1039 = torch.ops.aten.unsqueeze.default(unsqueeze_1038, 2);  unsqueeze_1038 = None
        unsqueeze_1040 = torch.ops.aten.unsqueeze.default(unsqueeze_1039, 3);  unsqueeze_1039 = None
        mul_1167 = torch.ops.aten.mul.Tensor(sum_112, 0.00010813148788927336)
        mul_1168 = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
        mul_1169 = torch.ops.aten.mul.Tensor(mul_1167, mul_1168);  mul_1167 = mul_1168 = None
        unsqueeze_1041 = torch.ops.aten.unsqueeze.default(mul_1169, 0);  mul_1169 = None
        unsqueeze_1042 = torch.ops.aten.unsqueeze.default(unsqueeze_1041, 2);  unsqueeze_1041 = None
        unsqueeze_1043 = torch.ops.aten.unsqueeze.default(unsqueeze_1042, 3);  unsqueeze_1042 = None
        mul_1170 = torch.ops.aten.mul.Tensor(squeeze_124, primals_125);  primals_125 = None
        unsqueeze_1044 = torch.ops.aten.unsqueeze.default(mul_1170, 0);  mul_1170 = None
        unsqueeze_1045 = torch.ops.aten.unsqueeze.default(unsqueeze_1044, 2);  unsqueeze_1044 = None
        unsqueeze_1046 = torch.ops.aten.unsqueeze.default(unsqueeze_1045, 3);  unsqueeze_1045 = None
        mul_1171 = torch.ops.aten.mul.Tensor(sub_312, unsqueeze_1043);  sub_312 = unsqueeze_1043 = None
        sub_314 = torch.ops.aten.sub.Tensor(where_54, mul_1171);  where_54 = mul_1171 = None
        sub_315 = torch.ops.aten.sub.Tensor(sub_314, unsqueeze_1040);  sub_314 = unsqueeze_1040 = None
        mul_1172 = torch.ops.aten.mul.Tensor(sub_315, unsqueeze_1046);  sub_315 = unsqueeze_1046 = None
        mul_1173 = torch.ops.aten.mul.Tensor(sum_112, squeeze_124);  sum_112 = squeeze_124 = None
        convolution_backward_54 = torch.ops.aten.convolution_backward.default(mul_1172, cat_5, primals_124, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1172 = primals_124 = None
        getitem_362 = convolution_backward_54[0]
        getitem_363 = convolution_backward_54[1];  convolution_backward_54 = None
        add_503 = torch.ops.aten.add.Tensor(add_502, getitem_362);  add_502 = getitem_362 = None
        where_55 = torch.ops.aten.where.self(le_55, full_default, slice_31);  le_55 = slice_31 = None
        sum_113 = torch.ops.aten.sum.dim_IntList(where_55, [0, 2, 3])
        sub_316 = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_1049);  convolution_40 = unsqueeze_1049 = None
        mul_1174 = torch.ops.aten.mul.Tensor(where_55, sub_316)
        sum_114 = torch.ops.aten.sum.dim_IntList(mul_1174, [0, 2, 3]);  mul_1174 = None
        mul_1175 = torch.ops.aten.mul.Tensor(sum_113, 0.00010813148788927336)
        unsqueeze_1050 = torch.ops.aten.unsqueeze.default(mul_1175, 0);  mul_1175 = None
        unsqueeze_1051 = torch.ops.aten.unsqueeze.default(unsqueeze_1050, 2);  unsqueeze_1050 = None
        unsqueeze_1052 = torch.ops.aten.unsqueeze.default(unsqueeze_1051, 3);  unsqueeze_1051 = None
        mul_1176 = torch.ops.aten.mul.Tensor(sum_114, 0.00010813148788927336)
        mul_1177 = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
        mul_1178 = torch.ops.aten.mul.Tensor(mul_1176, mul_1177);  mul_1176 = mul_1177 = None
        unsqueeze_1053 = torch.ops.aten.unsqueeze.default(mul_1178, 0);  mul_1178 = None
        unsqueeze_1054 = torch.ops.aten.unsqueeze.default(unsqueeze_1053, 2);  unsqueeze_1053 = None
        unsqueeze_1055 = torch.ops.aten.unsqueeze.default(unsqueeze_1054, 3);  unsqueeze_1054 = None
        mul_1179 = torch.ops.aten.mul.Tensor(squeeze_121, primals_122);  primals_122 = None
        unsqueeze_1056 = torch.ops.aten.unsqueeze.default(mul_1179, 0);  mul_1179 = None
        unsqueeze_1057 = torch.ops.aten.unsqueeze.default(unsqueeze_1056, 2);  unsqueeze_1056 = None
        unsqueeze_1058 = torch.ops.aten.unsqueeze.default(unsqueeze_1057, 3);  unsqueeze_1057 = None
        mul_1180 = torch.ops.aten.mul.Tensor(sub_316, unsqueeze_1055);  sub_316 = unsqueeze_1055 = None
        sub_318 = torch.ops.aten.sub.Tensor(where_55, mul_1180);  where_55 = mul_1180 = None
        sub_319 = torch.ops.aten.sub.Tensor(sub_318, unsqueeze_1052);  sub_318 = unsqueeze_1052 = None
        mul_1181 = torch.ops.aten.mul.Tensor(sub_319, unsqueeze_1058);  sub_319 = unsqueeze_1058 = None
        mul_1182 = torch.ops.aten.mul.Tensor(sum_114, squeeze_121);  sum_114 = squeeze_121 = None
        convolution_backward_55 = torch.ops.aten.convolution_backward.default(mul_1181, cat_5, primals_121, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1181 = cat_5 = primals_121 = None
        getitem_365 = convolution_backward_55[0]
        getitem_366 = convolution_backward_55[1];  convolution_backward_55 = None
        add_504 = torch.ops.aten.add.Tensor(add_503, getitem_365);  add_503 = getitem_365 = None
        slice_35 = torch.ops.aten.slice.Tensor(add_504, 1, 0, 192)
        slice_36 = torch.ops.aten.slice.Tensor(add_504, 1, 192, 384)
        slice_37 = torch.ops.aten.slice.Tensor(add_504, 1, 384, 576)
        slice_38 = torch.ops.aten.slice.Tensor(add_504, 1, 576, 768);  add_504 = None
        where_56 = torch.ops.aten.where.self(le_56, full_default, slice_38);  le_56 = slice_38 = None
        sum_115 = torch.ops.aten.sum.dim_IntList(where_56, [0, 2, 3])
        sub_320 = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_1061);  convolution_39 = unsqueeze_1061 = None
        mul_1183 = torch.ops.aten.mul.Tensor(where_56, sub_320)
        sum_116 = torch.ops.aten.sum.dim_IntList(mul_1183, [0, 2, 3]);  mul_1183 = None
        mul_1184 = torch.ops.aten.mul.Tensor(sum_115, 0.00010813148788927336)
        unsqueeze_1062 = torch.ops.aten.unsqueeze.default(mul_1184, 0);  mul_1184 = None
        unsqueeze_1063 = torch.ops.aten.unsqueeze.default(unsqueeze_1062, 2);  unsqueeze_1062 = None
        unsqueeze_1064 = torch.ops.aten.unsqueeze.default(unsqueeze_1063, 3);  unsqueeze_1063 = None
        mul_1185 = torch.ops.aten.mul.Tensor(sum_116, 0.00010813148788927336)
        mul_1186 = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
        mul_1187 = torch.ops.aten.mul.Tensor(mul_1185, mul_1186);  mul_1185 = mul_1186 = None
        unsqueeze_1065 = torch.ops.aten.unsqueeze.default(mul_1187, 0);  mul_1187 = None
        unsqueeze_1066 = torch.ops.aten.unsqueeze.default(unsqueeze_1065, 2);  unsqueeze_1065 = None
        unsqueeze_1067 = torch.ops.aten.unsqueeze.default(unsqueeze_1066, 3);  unsqueeze_1066 = None
        mul_1188 = torch.ops.aten.mul.Tensor(squeeze_118, primals_119);  primals_119 = None
        unsqueeze_1068 = torch.ops.aten.unsqueeze.default(mul_1188, 0);  mul_1188 = None
        unsqueeze_1069 = torch.ops.aten.unsqueeze.default(unsqueeze_1068, 2);  unsqueeze_1068 = None
        unsqueeze_1070 = torch.ops.aten.unsqueeze.default(unsqueeze_1069, 3);  unsqueeze_1069 = None
        mul_1189 = torch.ops.aten.mul.Tensor(sub_320, unsqueeze_1067);  sub_320 = unsqueeze_1067 = None
        sub_322 = torch.ops.aten.sub.Tensor(where_56, mul_1189);  where_56 = mul_1189 = None
        sub_323 = torch.ops.aten.sub.Tensor(sub_322, unsqueeze_1064);  sub_322 = unsqueeze_1064 = None
        mul_1190 = torch.ops.aten.mul.Tensor(sub_323, unsqueeze_1070);  sub_323 = unsqueeze_1070 = None
        mul_1191 = torch.ops.aten.mul.Tensor(sum_116, squeeze_118);  sum_116 = squeeze_118 = None
        convolution_backward_56 = torch.ops.aten.convolution_backward.default(mul_1190, avg_pool2d_3, primals_118, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1190 = avg_pool2d_3 = primals_118 = None
        getitem_368 = convolution_backward_56[0]
        getitem_369 = convolution_backward_56[1];  convolution_backward_56 = None
        avg_pool2d_backward_6 = torch.ops.aten.avg_pool2d_backward.default(getitem_368, cat_4, [3, 3], [1, 1], [1, 1], False, True, None);  getitem_368 = None
        where_57 = torch.ops.aten.where.self(le_57, full_default, slice_37);  le_57 = slice_37 = None
        sum_117 = torch.ops.aten.sum.dim_IntList(where_57, [0, 2, 3])
        sub_324 = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_1073);  convolution_38 = unsqueeze_1073 = None
        mul_1192 = torch.ops.aten.mul.Tensor(where_57, sub_324)
        sum_118 = torch.ops.aten.sum.dim_IntList(mul_1192, [0, 2, 3]);  mul_1192 = None
        mul_1193 = torch.ops.aten.mul.Tensor(sum_117, 0.00010813148788927336)
        unsqueeze_1074 = torch.ops.aten.unsqueeze.default(mul_1193, 0);  mul_1193 = None
        unsqueeze_1075 = torch.ops.aten.unsqueeze.default(unsqueeze_1074, 2);  unsqueeze_1074 = None
        unsqueeze_1076 = torch.ops.aten.unsqueeze.default(unsqueeze_1075, 3);  unsqueeze_1075 = None
        mul_1194 = torch.ops.aten.mul.Tensor(sum_118, 0.00010813148788927336)
        mul_1195 = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
        mul_1196 = torch.ops.aten.mul.Tensor(mul_1194, mul_1195);  mul_1194 = mul_1195 = None
        unsqueeze_1077 = torch.ops.aten.unsqueeze.default(mul_1196, 0);  mul_1196 = None
        unsqueeze_1078 = torch.ops.aten.unsqueeze.default(unsqueeze_1077, 2);  unsqueeze_1077 = None
        unsqueeze_1079 = torch.ops.aten.unsqueeze.default(unsqueeze_1078, 3);  unsqueeze_1078 = None
        mul_1197 = torch.ops.aten.mul.Tensor(squeeze_115, primals_116);  primals_116 = None
        unsqueeze_1080 = torch.ops.aten.unsqueeze.default(mul_1197, 0);  mul_1197 = None
        unsqueeze_1081 = torch.ops.aten.unsqueeze.default(unsqueeze_1080, 2);  unsqueeze_1080 = None
        unsqueeze_1082 = torch.ops.aten.unsqueeze.default(unsqueeze_1081, 3);  unsqueeze_1081 = None
        mul_1198 = torch.ops.aten.mul.Tensor(sub_324, unsqueeze_1079);  sub_324 = unsqueeze_1079 = None
        sub_326 = torch.ops.aten.sub.Tensor(where_57, mul_1198);  where_57 = mul_1198 = None
        sub_327 = torch.ops.aten.sub.Tensor(sub_326, unsqueeze_1076);  sub_326 = unsqueeze_1076 = None
        mul_1199 = torch.ops.aten.mul.Tensor(sub_327, unsqueeze_1082);  sub_327 = unsqueeze_1082 = None
        mul_1200 = torch.ops.aten.mul.Tensor(sum_118, squeeze_115);  sum_118 = squeeze_115 = None
        convolution_backward_57 = torch.ops.aten.convolution_backward.default(mul_1199, relu_37, primals_115, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1199 = primals_115 = None
        getitem_371 = convolution_backward_57[0]
        getitem_372 = convolution_backward_57[1];  convolution_backward_57 = None
        alias_426 = torch.ops.aten.alias.default(relu_37);  relu_37 = None
        alias_427 = torch.ops.aten.alias.default(alias_426);  alias_426 = None
        le_58 = torch.ops.aten.le.Scalar(alias_427, 0);  alias_427 = None
        where_58 = torch.ops.aten.where.self(le_58, full_default, getitem_371);  le_58 = getitem_371 = None
        sum_119 = torch.ops.aten.sum.dim_IntList(where_58, [0, 2, 3])
        sub_328 = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_1085);  convolution_37 = unsqueeze_1085 = None
        mul_1201 = torch.ops.aten.mul.Tensor(where_58, sub_328)
        sum_120 = torch.ops.aten.sum.dim_IntList(mul_1201, [0, 2, 3]);  mul_1201 = None
        mul_1202 = torch.ops.aten.mul.Tensor(sum_119, 0.00010813148788927336)
        unsqueeze_1086 = torch.ops.aten.unsqueeze.default(mul_1202, 0);  mul_1202 = None
        unsqueeze_1087 = torch.ops.aten.unsqueeze.default(unsqueeze_1086, 2);  unsqueeze_1086 = None
        unsqueeze_1088 = torch.ops.aten.unsqueeze.default(unsqueeze_1087, 3);  unsqueeze_1087 = None
        mul_1203 = torch.ops.aten.mul.Tensor(sum_120, 0.00010813148788927336)
        mul_1204 = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
        mul_1205 = torch.ops.aten.mul.Tensor(mul_1203, mul_1204);  mul_1203 = mul_1204 = None
        unsqueeze_1089 = torch.ops.aten.unsqueeze.default(mul_1205, 0);  mul_1205 = None
        unsqueeze_1090 = torch.ops.aten.unsqueeze.default(unsqueeze_1089, 2);  unsqueeze_1089 = None
        unsqueeze_1091 = torch.ops.aten.unsqueeze.default(unsqueeze_1090, 3);  unsqueeze_1090 = None
        mul_1206 = torch.ops.aten.mul.Tensor(squeeze_112, primals_113);  primals_113 = None
        unsqueeze_1092 = torch.ops.aten.unsqueeze.default(mul_1206, 0);  mul_1206 = None
        unsqueeze_1093 = torch.ops.aten.unsqueeze.default(unsqueeze_1092, 2);  unsqueeze_1092 = None
        unsqueeze_1094 = torch.ops.aten.unsqueeze.default(unsqueeze_1093, 3);  unsqueeze_1093 = None
        mul_1207 = torch.ops.aten.mul.Tensor(sub_328, unsqueeze_1091);  sub_328 = unsqueeze_1091 = None
        sub_330 = torch.ops.aten.sub.Tensor(where_58, mul_1207);  where_58 = mul_1207 = None
        sub_331 = torch.ops.aten.sub.Tensor(sub_330, unsqueeze_1088);  sub_330 = unsqueeze_1088 = None
        mul_1208 = torch.ops.aten.mul.Tensor(sub_331, unsqueeze_1094);  sub_331 = unsqueeze_1094 = None
        mul_1209 = torch.ops.aten.mul.Tensor(sum_120, squeeze_112);  sum_120 = squeeze_112 = None
        convolution_backward_58 = torch.ops.aten.convolution_backward.default(mul_1208, relu_36, primals_112, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1208 = primals_112 = None
        getitem_374 = convolution_backward_58[0]
        getitem_375 = convolution_backward_58[1];  convolution_backward_58 = None
        alias_430 = torch.ops.aten.alias.default(relu_36);  relu_36 = None
        alias_431 = torch.ops.aten.alias.default(alias_430);  alias_430 = None
        le_59 = torch.ops.aten.le.Scalar(alias_431, 0);  alias_431 = None
        where_59 = torch.ops.aten.where.self(le_59, full_default, getitem_374);  le_59 = getitem_374 = None
        sum_121 = torch.ops.aten.sum.dim_IntList(where_59, [0, 2, 3])
        sub_332 = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_1097);  convolution_36 = unsqueeze_1097 = None
        mul_1210 = torch.ops.aten.mul.Tensor(where_59, sub_332)
        sum_122 = torch.ops.aten.sum.dim_IntList(mul_1210, [0, 2, 3]);  mul_1210 = None
        mul_1211 = torch.ops.aten.mul.Tensor(sum_121, 0.00010813148788927336)
        unsqueeze_1098 = torch.ops.aten.unsqueeze.default(mul_1211, 0);  mul_1211 = None
        unsqueeze_1099 = torch.ops.aten.unsqueeze.default(unsqueeze_1098, 2);  unsqueeze_1098 = None
        unsqueeze_1100 = torch.ops.aten.unsqueeze.default(unsqueeze_1099, 3);  unsqueeze_1099 = None
        mul_1212 = torch.ops.aten.mul.Tensor(sum_122, 0.00010813148788927336)
        mul_1213 = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
        mul_1214 = torch.ops.aten.mul.Tensor(mul_1212, mul_1213);  mul_1212 = mul_1213 = None
        unsqueeze_1101 = torch.ops.aten.unsqueeze.default(mul_1214, 0);  mul_1214 = None
        unsqueeze_1102 = torch.ops.aten.unsqueeze.default(unsqueeze_1101, 2);  unsqueeze_1101 = None
        unsqueeze_1103 = torch.ops.aten.unsqueeze.default(unsqueeze_1102, 3);  unsqueeze_1102 = None
        mul_1215 = torch.ops.aten.mul.Tensor(squeeze_109, primals_110);  primals_110 = None
        unsqueeze_1104 = torch.ops.aten.unsqueeze.default(mul_1215, 0);  mul_1215 = None
        unsqueeze_1105 = torch.ops.aten.unsqueeze.default(unsqueeze_1104, 2);  unsqueeze_1104 = None
        unsqueeze_1106 = torch.ops.aten.unsqueeze.default(unsqueeze_1105, 3);  unsqueeze_1105 = None
        mul_1216 = torch.ops.aten.mul.Tensor(sub_332, unsqueeze_1103);  sub_332 = unsqueeze_1103 = None
        sub_334 = torch.ops.aten.sub.Tensor(where_59, mul_1216);  where_59 = mul_1216 = None
        sub_335 = torch.ops.aten.sub.Tensor(sub_334, unsqueeze_1100);  sub_334 = unsqueeze_1100 = None
        mul_1217 = torch.ops.aten.mul.Tensor(sub_335, unsqueeze_1106);  sub_335 = unsqueeze_1106 = None
        mul_1218 = torch.ops.aten.mul.Tensor(sum_122, squeeze_109);  sum_122 = squeeze_109 = None
        convolution_backward_59 = torch.ops.aten.convolution_backward.default(mul_1217, relu_35, primals_109, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1217 = primals_109 = None
        getitem_377 = convolution_backward_59[0]
        getitem_378 = convolution_backward_59[1];  convolution_backward_59 = None
        alias_434 = torch.ops.aten.alias.default(relu_35);  relu_35 = None
        alias_435 = torch.ops.aten.alias.default(alias_434);  alias_434 = None
        le_60 = torch.ops.aten.le.Scalar(alias_435, 0);  alias_435 = None
        where_60 = torch.ops.aten.where.self(le_60, full_default, getitem_377);  le_60 = getitem_377 = None
        sum_123 = torch.ops.aten.sum.dim_IntList(where_60, [0, 2, 3])
        sub_336 = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_1109);  convolution_35 = unsqueeze_1109 = None
        mul_1219 = torch.ops.aten.mul.Tensor(where_60, sub_336)
        sum_124 = torch.ops.aten.sum.dim_IntList(mul_1219, [0, 2, 3]);  mul_1219 = None
        mul_1220 = torch.ops.aten.mul.Tensor(sum_123, 0.00010813148788927336)
        unsqueeze_1110 = torch.ops.aten.unsqueeze.default(mul_1220, 0);  mul_1220 = None
        unsqueeze_1111 = torch.ops.aten.unsqueeze.default(unsqueeze_1110, 2);  unsqueeze_1110 = None
        unsqueeze_1112 = torch.ops.aten.unsqueeze.default(unsqueeze_1111, 3);  unsqueeze_1111 = None
        mul_1221 = torch.ops.aten.mul.Tensor(sum_124, 0.00010813148788927336)
        mul_1222 = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
        mul_1223 = torch.ops.aten.mul.Tensor(mul_1221, mul_1222);  mul_1221 = mul_1222 = None
        unsqueeze_1113 = torch.ops.aten.unsqueeze.default(mul_1223, 0);  mul_1223 = None
        unsqueeze_1114 = torch.ops.aten.unsqueeze.default(unsqueeze_1113, 2);  unsqueeze_1113 = None
        unsqueeze_1115 = torch.ops.aten.unsqueeze.default(unsqueeze_1114, 3);  unsqueeze_1114 = None
        mul_1224 = torch.ops.aten.mul.Tensor(squeeze_106, primals_107);  primals_107 = None
        unsqueeze_1116 = torch.ops.aten.unsqueeze.default(mul_1224, 0);  mul_1224 = None
        unsqueeze_1117 = torch.ops.aten.unsqueeze.default(unsqueeze_1116, 2);  unsqueeze_1116 = None
        unsqueeze_1118 = torch.ops.aten.unsqueeze.default(unsqueeze_1117, 3);  unsqueeze_1117 = None
        mul_1225 = torch.ops.aten.mul.Tensor(sub_336, unsqueeze_1115);  sub_336 = unsqueeze_1115 = None
        sub_338 = torch.ops.aten.sub.Tensor(where_60, mul_1225);  where_60 = mul_1225 = None
        sub_339 = torch.ops.aten.sub.Tensor(sub_338, unsqueeze_1112);  sub_338 = unsqueeze_1112 = None
        mul_1226 = torch.ops.aten.mul.Tensor(sub_339, unsqueeze_1118);  sub_339 = unsqueeze_1118 = None
        mul_1227 = torch.ops.aten.mul.Tensor(sum_124, squeeze_106);  sum_124 = squeeze_106 = None
        convolution_backward_60 = torch.ops.aten.convolution_backward.default(mul_1226, relu_34, primals_106, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1226 = primals_106 = None
        getitem_380 = convolution_backward_60[0]
        getitem_381 = convolution_backward_60[1];  convolution_backward_60 = None
        alias_438 = torch.ops.aten.alias.default(relu_34);  relu_34 = None
        alias_439 = torch.ops.aten.alias.default(alias_438);  alias_438 = None
        le_61 = torch.ops.aten.le.Scalar(alias_439, 0);  alias_439 = None
        where_61 = torch.ops.aten.where.self(le_61, full_default, getitem_380);  le_61 = getitem_380 = None
        sum_125 = torch.ops.aten.sum.dim_IntList(where_61, [0, 2, 3])
        sub_340 = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_1121);  convolution_34 = unsqueeze_1121 = None
        mul_1228 = torch.ops.aten.mul.Tensor(where_61, sub_340)
        sum_126 = torch.ops.aten.sum.dim_IntList(mul_1228, [0, 2, 3]);  mul_1228 = None
        mul_1229 = torch.ops.aten.mul.Tensor(sum_125, 0.00010813148788927336)
        unsqueeze_1122 = torch.ops.aten.unsqueeze.default(mul_1229, 0);  mul_1229 = None
        unsqueeze_1123 = torch.ops.aten.unsqueeze.default(unsqueeze_1122, 2);  unsqueeze_1122 = None
        unsqueeze_1124 = torch.ops.aten.unsqueeze.default(unsqueeze_1123, 3);  unsqueeze_1123 = None
        mul_1230 = torch.ops.aten.mul.Tensor(sum_126, 0.00010813148788927336)
        mul_1231 = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
        mul_1232 = torch.ops.aten.mul.Tensor(mul_1230, mul_1231);  mul_1230 = mul_1231 = None
        unsqueeze_1125 = torch.ops.aten.unsqueeze.default(mul_1232, 0);  mul_1232 = None
        unsqueeze_1126 = torch.ops.aten.unsqueeze.default(unsqueeze_1125, 2);  unsqueeze_1125 = None
        unsqueeze_1127 = torch.ops.aten.unsqueeze.default(unsqueeze_1126, 3);  unsqueeze_1126 = None
        mul_1233 = torch.ops.aten.mul.Tensor(squeeze_103, primals_104);  primals_104 = None
        unsqueeze_1128 = torch.ops.aten.unsqueeze.default(mul_1233, 0);  mul_1233 = None
        unsqueeze_1129 = torch.ops.aten.unsqueeze.default(unsqueeze_1128, 2);  unsqueeze_1128 = None
        unsqueeze_1130 = torch.ops.aten.unsqueeze.default(unsqueeze_1129, 3);  unsqueeze_1129 = None
        mul_1234 = torch.ops.aten.mul.Tensor(sub_340, unsqueeze_1127);  sub_340 = unsqueeze_1127 = None
        sub_342 = torch.ops.aten.sub.Tensor(where_61, mul_1234);  where_61 = mul_1234 = None
        sub_343 = torch.ops.aten.sub.Tensor(sub_342, unsqueeze_1124);  sub_342 = unsqueeze_1124 = None
        mul_1235 = torch.ops.aten.mul.Tensor(sub_343, unsqueeze_1130);  sub_343 = unsqueeze_1130 = None
        mul_1236 = torch.ops.aten.mul.Tensor(sum_126, squeeze_103);  sum_126 = squeeze_103 = None
        convolution_backward_61 = torch.ops.aten.convolution_backward.default(mul_1235, cat_4, primals_103, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1235 = primals_103 = None
        getitem_383 = convolution_backward_61[0]
        getitem_384 = convolution_backward_61[1];  convolution_backward_61 = None
        add_505 = torch.ops.aten.add.Tensor(avg_pool2d_backward_6, getitem_383);  avg_pool2d_backward_6 = getitem_383 = None
        where_62 = torch.ops.aten.where.self(le_62, full_default, slice_36);  le_62 = slice_36 = None
        sum_127 = torch.ops.aten.sum.dim_IntList(where_62, [0, 2, 3])
        sub_344 = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_1133);  convolution_33 = unsqueeze_1133 = None
        mul_1237 = torch.ops.aten.mul.Tensor(where_62, sub_344)
        sum_128 = torch.ops.aten.sum.dim_IntList(mul_1237, [0, 2, 3]);  mul_1237 = None
        mul_1238 = torch.ops.aten.mul.Tensor(sum_127, 0.00010813148788927336)
        unsqueeze_1134 = torch.ops.aten.unsqueeze.default(mul_1238, 0);  mul_1238 = None
        unsqueeze_1135 = torch.ops.aten.unsqueeze.default(unsqueeze_1134, 2);  unsqueeze_1134 = None
        unsqueeze_1136 = torch.ops.aten.unsqueeze.default(unsqueeze_1135, 3);  unsqueeze_1135 = None
        mul_1239 = torch.ops.aten.mul.Tensor(sum_128, 0.00010813148788927336)
        mul_1240 = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
        mul_1241 = torch.ops.aten.mul.Tensor(mul_1239, mul_1240);  mul_1239 = mul_1240 = None
        unsqueeze_1137 = torch.ops.aten.unsqueeze.default(mul_1241, 0);  mul_1241 = None
        unsqueeze_1138 = torch.ops.aten.unsqueeze.default(unsqueeze_1137, 2);  unsqueeze_1137 = None
        unsqueeze_1139 = torch.ops.aten.unsqueeze.default(unsqueeze_1138, 3);  unsqueeze_1138 = None
        mul_1242 = torch.ops.aten.mul.Tensor(squeeze_100, primals_101);  primals_101 = None
        unsqueeze_1140 = torch.ops.aten.unsqueeze.default(mul_1242, 0);  mul_1242 = None
        unsqueeze_1141 = torch.ops.aten.unsqueeze.default(unsqueeze_1140, 2);  unsqueeze_1140 = None
        unsqueeze_1142 = torch.ops.aten.unsqueeze.default(unsqueeze_1141, 3);  unsqueeze_1141 = None
        mul_1243 = torch.ops.aten.mul.Tensor(sub_344, unsqueeze_1139);  sub_344 = unsqueeze_1139 = None
        sub_346 = torch.ops.aten.sub.Tensor(where_62, mul_1243);  where_62 = mul_1243 = None
        sub_347 = torch.ops.aten.sub.Tensor(sub_346, unsqueeze_1136);  sub_346 = unsqueeze_1136 = None
        mul_1244 = torch.ops.aten.mul.Tensor(sub_347, unsqueeze_1142);  sub_347 = unsqueeze_1142 = None
        mul_1245 = torch.ops.aten.mul.Tensor(sum_128, squeeze_100);  sum_128 = squeeze_100 = None
        convolution_backward_62 = torch.ops.aten.convolution_backward.default(mul_1244, relu_32, primals_100, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1244 = primals_100 = None
        getitem_386 = convolution_backward_62[0]
        getitem_387 = convolution_backward_62[1];  convolution_backward_62 = None
        alias_446 = torch.ops.aten.alias.default(relu_32);  relu_32 = None
        alias_447 = torch.ops.aten.alias.default(alias_446);  alias_446 = None
        le_63 = torch.ops.aten.le.Scalar(alias_447, 0);  alias_447 = None
        where_63 = torch.ops.aten.where.self(le_63, full_default, getitem_386);  le_63 = getitem_386 = None
        sum_129 = torch.ops.aten.sum.dim_IntList(where_63, [0, 2, 3])
        sub_348 = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_1145);  convolution_32 = unsqueeze_1145 = None
        mul_1246 = torch.ops.aten.mul.Tensor(where_63, sub_348)
        sum_130 = torch.ops.aten.sum.dim_IntList(mul_1246, [0, 2, 3]);  mul_1246 = None
        mul_1247 = torch.ops.aten.mul.Tensor(sum_129, 0.00010813148788927336)
        unsqueeze_1146 = torch.ops.aten.unsqueeze.default(mul_1247, 0);  mul_1247 = None
        unsqueeze_1147 = torch.ops.aten.unsqueeze.default(unsqueeze_1146, 2);  unsqueeze_1146 = None
        unsqueeze_1148 = torch.ops.aten.unsqueeze.default(unsqueeze_1147, 3);  unsqueeze_1147 = None
        mul_1248 = torch.ops.aten.mul.Tensor(sum_130, 0.00010813148788927336)
        mul_1249 = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
        mul_1250 = torch.ops.aten.mul.Tensor(mul_1248, mul_1249);  mul_1248 = mul_1249 = None
        unsqueeze_1149 = torch.ops.aten.unsqueeze.default(mul_1250, 0);  mul_1250 = None
        unsqueeze_1150 = torch.ops.aten.unsqueeze.default(unsqueeze_1149, 2);  unsqueeze_1149 = None
        unsqueeze_1151 = torch.ops.aten.unsqueeze.default(unsqueeze_1150, 3);  unsqueeze_1150 = None
        mul_1251 = torch.ops.aten.mul.Tensor(squeeze_97, primals_98);  primals_98 = None
        unsqueeze_1152 = torch.ops.aten.unsqueeze.default(mul_1251, 0);  mul_1251 = None
        unsqueeze_1153 = torch.ops.aten.unsqueeze.default(unsqueeze_1152, 2);  unsqueeze_1152 = None
        unsqueeze_1154 = torch.ops.aten.unsqueeze.default(unsqueeze_1153, 3);  unsqueeze_1153 = None
        mul_1252 = torch.ops.aten.mul.Tensor(sub_348, unsqueeze_1151);  sub_348 = unsqueeze_1151 = None
        sub_350 = torch.ops.aten.sub.Tensor(where_63, mul_1252);  where_63 = mul_1252 = None
        sub_351 = torch.ops.aten.sub.Tensor(sub_350, unsqueeze_1148);  sub_350 = unsqueeze_1148 = None
        mul_1253 = torch.ops.aten.mul.Tensor(sub_351, unsqueeze_1154);  sub_351 = unsqueeze_1154 = None
        mul_1254 = torch.ops.aten.mul.Tensor(sum_130, squeeze_97);  sum_130 = squeeze_97 = None
        convolution_backward_63 = torch.ops.aten.convolution_backward.default(mul_1253, relu_31, primals_97, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1253 = primals_97 = None
        getitem_389 = convolution_backward_63[0]
        getitem_390 = convolution_backward_63[1];  convolution_backward_63 = None
        alias_450 = torch.ops.aten.alias.default(relu_31);  relu_31 = None
        alias_451 = torch.ops.aten.alias.default(alias_450);  alias_450 = None
        le_64 = torch.ops.aten.le.Scalar(alias_451, 0);  alias_451 = None
        where_64 = torch.ops.aten.where.self(le_64, full_default, getitem_389);  le_64 = getitem_389 = None
        sum_131 = torch.ops.aten.sum.dim_IntList(where_64, [0, 2, 3])
        sub_352 = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_1157);  convolution_31 = unsqueeze_1157 = None
        mul_1255 = torch.ops.aten.mul.Tensor(where_64, sub_352)
        sum_132 = torch.ops.aten.sum.dim_IntList(mul_1255, [0, 2, 3]);  mul_1255 = None
        mul_1256 = torch.ops.aten.mul.Tensor(sum_131, 0.00010813148788927336)
        unsqueeze_1158 = torch.ops.aten.unsqueeze.default(mul_1256, 0);  mul_1256 = None
        unsqueeze_1159 = torch.ops.aten.unsqueeze.default(unsqueeze_1158, 2);  unsqueeze_1158 = None
        unsqueeze_1160 = torch.ops.aten.unsqueeze.default(unsqueeze_1159, 3);  unsqueeze_1159 = None
        mul_1257 = torch.ops.aten.mul.Tensor(sum_132, 0.00010813148788927336)
        mul_1258 = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
        mul_1259 = torch.ops.aten.mul.Tensor(mul_1257, mul_1258);  mul_1257 = mul_1258 = None
        unsqueeze_1161 = torch.ops.aten.unsqueeze.default(mul_1259, 0);  mul_1259 = None
        unsqueeze_1162 = torch.ops.aten.unsqueeze.default(unsqueeze_1161, 2);  unsqueeze_1161 = None
        unsqueeze_1163 = torch.ops.aten.unsqueeze.default(unsqueeze_1162, 3);  unsqueeze_1162 = None
        mul_1260 = torch.ops.aten.mul.Tensor(squeeze_94, primals_95);  primals_95 = None
        unsqueeze_1164 = torch.ops.aten.unsqueeze.default(mul_1260, 0);  mul_1260 = None
        unsqueeze_1165 = torch.ops.aten.unsqueeze.default(unsqueeze_1164, 2);  unsqueeze_1164 = None
        unsqueeze_1166 = torch.ops.aten.unsqueeze.default(unsqueeze_1165, 3);  unsqueeze_1165 = None
        mul_1261 = torch.ops.aten.mul.Tensor(sub_352, unsqueeze_1163);  sub_352 = unsqueeze_1163 = None
        sub_354 = torch.ops.aten.sub.Tensor(where_64, mul_1261);  where_64 = mul_1261 = None
        sub_355 = torch.ops.aten.sub.Tensor(sub_354, unsqueeze_1160);  sub_354 = unsqueeze_1160 = None
        mul_1262 = torch.ops.aten.mul.Tensor(sub_355, unsqueeze_1166);  sub_355 = unsqueeze_1166 = None
        mul_1263 = torch.ops.aten.mul.Tensor(sum_132, squeeze_94);  sum_132 = squeeze_94 = None
        convolution_backward_64 = torch.ops.aten.convolution_backward.default(mul_1262, cat_4, primals_94, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1262 = primals_94 = None
        getitem_392 = convolution_backward_64[0]
        getitem_393 = convolution_backward_64[1];  convolution_backward_64 = None
        add_506 = torch.ops.aten.add.Tensor(add_505, getitem_392);  add_505 = getitem_392 = None
        where_65 = torch.ops.aten.where.self(le_65, full_default, slice_35);  le_65 = slice_35 = None
        sum_133 = torch.ops.aten.sum.dim_IntList(where_65, [0, 2, 3])
        sub_356 = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_1169);  convolution_30 = unsqueeze_1169 = None
        mul_1264 = torch.ops.aten.mul.Tensor(where_65, sub_356)
        sum_134 = torch.ops.aten.sum.dim_IntList(mul_1264, [0, 2, 3]);  mul_1264 = None
        mul_1265 = torch.ops.aten.mul.Tensor(sum_133, 0.00010813148788927336)
        unsqueeze_1170 = torch.ops.aten.unsqueeze.default(mul_1265, 0);  mul_1265 = None
        unsqueeze_1171 = torch.ops.aten.unsqueeze.default(unsqueeze_1170, 2);  unsqueeze_1170 = None
        unsqueeze_1172 = torch.ops.aten.unsqueeze.default(unsqueeze_1171, 3);  unsqueeze_1171 = None
        mul_1266 = torch.ops.aten.mul.Tensor(sum_134, 0.00010813148788927336)
        mul_1267 = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
        mul_1268 = torch.ops.aten.mul.Tensor(mul_1266, mul_1267);  mul_1266 = mul_1267 = None
        unsqueeze_1173 = torch.ops.aten.unsqueeze.default(mul_1268, 0);  mul_1268 = None
        unsqueeze_1174 = torch.ops.aten.unsqueeze.default(unsqueeze_1173, 2);  unsqueeze_1173 = None
        unsqueeze_1175 = torch.ops.aten.unsqueeze.default(unsqueeze_1174, 3);  unsqueeze_1174 = None
        mul_1269 = torch.ops.aten.mul.Tensor(squeeze_91, primals_92);  primals_92 = None
        unsqueeze_1176 = torch.ops.aten.unsqueeze.default(mul_1269, 0);  mul_1269 = None
        unsqueeze_1177 = torch.ops.aten.unsqueeze.default(unsqueeze_1176, 2);  unsqueeze_1176 = None
        unsqueeze_1178 = torch.ops.aten.unsqueeze.default(unsqueeze_1177, 3);  unsqueeze_1177 = None
        mul_1270 = torch.ops.aten.mul.Tensor(sub_356, unsqueeze_1175);  sub_356 = unsqueeze_1175 = None
        sub_358 = torch.ops.aten.sub.Tensor(where_65, mul_1270);  where_65 = mul_1270 = None
        sub_359 = torch.ops.aten.sub.Tensor(sub_358, unsqueeze_1172);  sub_358 = unsqueeze_1172 = None
        mul_1271 = torch.ops.aten.mul.Tensor(sub_359, unsqueeze_1178);  sub_359 = unsqueeze_1178 = None
        mul_1272 = torch.ops.aten.mul.Tensor(sum_134, squeeze_91);  sum_134 = squeeze_91 = None
        convolution_backward_65 = torch.ops.aten.convolution_backward.default(mul_1271, cat_4, primals_91, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1271 = cat_4 = primals_91 = None
        getitem_395 = convolution_backward_65[0]
        getitem_396 = convolution_backward_65[1];  convolution_backward_65 = None
        add_507 = torch.ops.aten.add.Tensor(add_506, getitem_395);  add_506 = getitem_395 = None
        slice_39 = torch.ops.aten.slice.Tensor(add_507, 1, 0, 384)
        slice_40 = torch.ops.aten.slice.Tensor(add_507, 1, 384, 480)
        slice_41 = torch.ops.aten.slice.Tensor(add_507, 1, 480, 768);  add_507 = None
        max_pool2d_with_indices_backward_1 = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_41, cat_3, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_65);  slice_41 = getitem_65 = None
        where_66 = torch.ops.aten.where.self(le_66, full_default, slice_40);  le_66 = slice_40 = None
        sum_135 = torch.ops.aten.sum.dim_IntList(where_66, [0, 2, 3])
        sub_360 = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_1181);  convolution_29 = unsqueeze_1181 = None
        mul_1273 = torch.ops.aten.mul.Tensor(where_66, sub_360)
        sum_136 = torch.ops.aten.sum.dim_IntList(mul_1273, [0, 2, 3]);  mul_1273 = None
        mul_1274 = torch.ops.aten.mul.Tensor(sum_135, 0.00010813148788927336)
        unsqueeze_1182 = torch.ops.aten.unsqueeze.default(mul_1274, 0);  mul_1274 = None
        unsqueeze_1183 = torch.ops.aten.unsqueeze.default(unsqueeze_1182, 2);  unsqueeze_1182 = None
        unsqueeze_1184 = torch.ops.aten.unsqueeze.default(unsqueeze_1183, 3);  unsqueeze_1183 = None
        mul_1275 = torch.ops.aten.mul.Tensor(sum_136, 0.00010813148788927336)
        mul_1276 = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
        mul_1277 = torch.ops.aten.mul.Tensor(mul_1275, mul_1276);  mul_1275 = mul_1276 = None
        unsqueeze_1185 = torch.ops.aten.unsqueeze.default(mul_1277, 0);  mul_1277 = None
        unsqueeze_1186 = torch.ops.aten.unsqueeze.default(unsqueeze_1185, 2);  unsqueeze_1185 = None
        unsqueeze_1187 = torch.ops.aten.unsqueeze.default(unsqueeze_1186, 3);  unsqueeze_1186 = None
        mul_1278 = torch.ops.aten.mul.Tensor(squeeze_88, primals_89);  primals_89 = None
        unsqueeze_1188 = torch.ops.aten.unsqueeze.default(mul_1278, 0);  mul_1278 = None
        unsqueeze_1189 = torch.ops.aten.unsqueeze.default(unsqueeze_1188, 2);  unsqueeze_1188 = None
        unsqueeze_1190 = torch.ops.aten.unsqueeze.default(unsqueeze_1189, 3);  unsqueeze_1189 = None
        mul_1279 = torch.ops.aten.mul.Tensor(sub_360, unsqueeze_1187);  sub_360 = unsqueeze_1187 = None
        sub_362 = torch.ops.aten.sub.Tensor(where_66, mul_1279);  where_66 = mul_1279 = None
        sub_363 = torch.ops.aten.sub.Tensor(sub_362, unsqueeze_1184);  sub_362 = unsqueeze_1184 = None
        mul_1280 = torch.ops.aten.mul.Tensor(sub_363, unsqueeze_1190);  sub_363 = unsqueeze_1190 = None
        mul_1281 = torch.ops.aten.mul.Tensor(sum_136, squeeze_88);  sum_136 = squeeze_88 = None
        convolution_backward_66 = torch.ops.aten.convolution_backward.default(mul_1280, relu_28, primals_88, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1280 = primals_88 = None
        getitem_398 = convolution_backward_66[0]
        getitem_399 = convolution_backward_66[1];  convolution_backward_66 = None
        alias_462 = torch.ops.aten.alias.default(relu_28);  relu_28 = None
        alias_463 = torch.ops.aten.alias.default(alias_462);  alias_462 = None
        le_67 = torch.ops.aten.le.Scalar(alias_463, 0);  alias_463 = None
        where_67 = torch.ops.aten.where.self(le_67, full_default, getitem_398);  le_67 = getitem_398 = None
        sum_137 = torch.ops.aten.sum.dim_IntList(where_67, [0, 2, 3])
        sub_364 = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_1193);  convolution_28 = unsqueeze_1193 = None
        mul_1282 = torch.ops.aten.mul.Tensor(where_67, sub_364)
        sum_138 = torch.ops.aten.sum.dim_IntList(mul_1282, [0, 2, 3]);  mul_1282 = None
        mul_1283 = torch.ops.aten.mul.Tensor(sum_137, 2.5510204081632654e-05)
        unsqueeze_1194 = torch.ops.aten.unsqueeze.default(mul_1283, 0);  mul_1283 = None
        unsqueeze_1195 = torch.ops.aten.unsqueeze.default(unsqueeze_1194, 2);  unsqueeze_1194 = None
        unsqueeze_1196 = torch.ops.aten.unsqueeze.default(unsqueeze_1195, 3);  unsqueeze_1195 = None
        mul_1284 = torch.ops.aten.mul.Tensor(sum_138, 2.5510204081632654e-05)
        mul_1285 = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
        mul_1286 = torch.ops.aten.mul.Tensor(mul_1284, mul_1285);  mul_1284 = mul_1285 = None
        unsqueeze_1197 = torch.ops.aten.unsqueeze.default(mul_1286, 0);  mul_1286 = None
        unsqueeze_1198 = torch.ops.aten.unsqueeze.default(unsqueeze_1197, 2);  unsqueeze_1197 = None
        unsqueeze_1199 = torch.ops.aten.unsqueeze.default(unsqueeze_1198, 3);  unsqueeze_1198 = None
        mul_1287 = torch.ops.aten.mul.Tensor(squeeze_85, primals_86);  primals_86 = None
        unsqueeze_1200 = torch.ops.aten.unsqueeze.default(mul_1287, 0);  mul_1287 = None
        unsqueeze_1201 = torch.ops.aten.unsqueeze.default(unsqueeze_1200, 2);  unsqueeze_1200 = None
        unsqueeze_1202 = torch.ops.aten.unsqueeze.default(unsqueeze_1201, 3);  unsqueeze_1201 = None
        mul_1288 = torch.ops.aten.mul.Tensor(sub_364, unsqueeze_1199);  sub_364 = unsqueeze_1199 = None
        sub_366 = torch.ops.aten.sub.Tensor(where_67, mul_1288);  where_67 = mul_1288 = None
        sub_367 = torch.ops.aten.sub.Tensor(sub_366, unsqueeze_1196);  sub_366 = unsqueeze_1196 = None
        mul_1289 = torch.ops.aten.mul.Tensor(sub_367, unsqueeze_1202);  sub_367 = unsqueeze_1202 = None
        mul_1290 = torch.ops.aten.mul.Tensor(sum_138, squeeze_85);  sum_138 = squeeze_85 = None
        convolution_backward_67 = torch.ops.aten.convolution_backward.default(mul_1289, relu_27, primals_85, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1289 = primals_85 = None
        getitem_401 = convolution_backward_67[0]
        getitem_402 = convolution_backward_67[1];  convolution_backward_67 = None
        alias_466 = torch.ops.aten.alias.default(relu_27);  relu_27 = None
        alias_467 = torch.ops.aten.alias.default(alias_466);  alias_466 = None
        le_68 = torch.ops.aten.le.Scalar(alias_467, 0);  alias_467 = None
        where_68 = torch.ops.aten.where.self(le_68, full_default, getitem_401);  le_68 = getitem_401 = None
        sum_139 = torch.ops.aten.sum.dim_IntList(where_68, [0, 2, 3])
        sub_368 = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_1205);  convolution_27 = unsqueeze_1205 = None
        mul_1291 = torch.ops.aten.mul.Tensor(where_68, sub_368)
        sum_140 = torch.ops.aten.sum.dim_IntList(mul_1291, [0, 2, 3]);  mul_1291 = None
        mul_1292 = torch.ops.aten.mul.Tensor(sum_139, 2.5510204081632654e-05)
        unsqueeze_1206 = torch.ops.aten.unsqueeze.default(mul_1292, 0);  mul_1292 = None
        unsqueeze_1207 = torch.ops.aten.unsqueeze.default(unsqueeze_1206, 2);  unsqueeze_1206 = None
        unsqueeze_1208 = torch.ops.aten.unsqueeze.default(unsqueeze_1207, 3);  unsqueeze_1207 = None
        mul_1293 = torch.ops.aten.mul.Tensor(sum_140, 2.5510204081632654e-05)
        mul_1294 = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
        mul_1295 = torch.ops.aten.mul.Tensor(mul_1293, mul_1294);  mul_1293 = mul_1294 = None
        unsqueeze_1209 = torch.ops.aten.unsqueeze.default(mul_1295, 0);  mul_1295 = None
        unsqueeze_1210 = torch.ops.aten.unsqueeze.default(unsqueeze_1209, 2);  unsqueeze_1209 = None
        unsqueeze_1211 = torch.ops.aten.unsqueeze.default(unsqueeze_1210, 3);  unsqueeze_1210 = None
        mul_1296 = torch.ops.aten.mul.Tensor(squeeze_82, primals_83);  primals_83 = None
        unsqueeze_1212 = torch.ops.aten.unsqueeze.default(mul_1296, 0);  mul_1296 = None
        unsqueeze_1213 = torch.ops.aten.unsqueeze.default(unsqueeze_1212, 2);  unsqueeze_1212 = None
        unsqueeze_1214 = torch.ops.aten.unsqueeze.default(unsqueeze_1213, 3);  unsqueeze_1213 = None
        mul_1297 = torch.ops.aten.mul.Tensor(sub_368, unsqueeze_1211);  sub_368 = unsqueeze_1211 = None
        sub_370 = torch.ops.aten.sub.Tensor(where_68, mul_1297);  where_68 = mul_1297 = None
        sub_371 = torch.ops.aten.sub.Tensor(sub_370, unsqueeze_1208);  sub_370 = unsqueeze_1208 = None
        mul_1298 = torch.ops.aten.mul.Tensor(sub_371, unsqueeze_1214);  sub_371 = unsqueeze_1214 = None
        mul_1299 = torch.ops.aten.mul.Tensor(sum_140, squeeze_82);  sum_140 = squeeze_82 = None
        convolution_backward_68 = torch.ops.aten.convolution_backward.default(mul_1298, cat_3, primals_82, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1298 = primals_82 = None
        getitem_404 = convolution_backward_68[0]
        getitem_405 = convolution_backward_68[1];  convolution_backward_68 = None
        add_508 = torch.ops.aten.add.Tensor(max_pool2d_with_indices_backward_1, getitem_404);  max_pool2d_with_indices_backward_1 = getitem_404 = None
        where_69 = torch.ops.aten.where.self(le_69, full_default, slice_39);  le_69 = slice_39 = None
        sum_141 = torch.ops.aten.sum.dim_IntList(where_69, [0, 2, 3])
        sub_372 = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_1217);  convolution_26 = unsqueeze_1217 = None
        mul_1300 = torch.ops.aten.mul.Tensor(where_69, sub_372)
        sum_142 = torch.ops.aten.sum.dim_IntList(mul_1300, [0, 2, 3]);  mul_1300 = None
        mul_1301 = torch.ops.aten.mul.Tensor(sum_141, 0.00010813148788927336)
        unsqueeze_1218 = torch.ops.aten.unsqueeze.default(mul_1301, 0);  mul_1301 = None
        unsqueeze_1219 = torch.ops.aten.unsqueeze.default(unsqueeze_1218, 2);  unsqueeze_1218 = None
        unsqueeze_1220 = torch.ops.aten.unsqueeze.default(unsqueeze_1219, 3);  unsqueeze_1219 = None
        mul_1302 = torch.ops.aten.mul.Tensor(sum_142, 0.00010813148788927336)
        mul_1303 = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
        mul_1304 = torch.ops.aten.mul.Tensor(mul_1302, mul_1303);  mul_1302 = mul_1303 = None
        unsqueeze_1221 = torch.ops.aten.unsqueeze.default(mul_1304, 0);  mul_1304 = None
        unsqueeze_1222 = torch.ops.aten.unsqueeze.default(unsqueeze_1221, 2);  unsqueeze_1221 = None
        unsqueeze_1223 = torch.ops.aten.unsqueeze.default(unsqueeze_1222, 3);  unsqueeze_1222 = None
        mul_1305 = torch.ops.aten.mul.Tensor(squeeze_79, primals_80);  primals_80 = None
        unsqueeze_1224 = torch.ops.aten.unsqueeze.default(mul_1305, 0);  mul_1305 = None
        unsqueeze_1225 = torch.ops.aten.unsqueeze.default(unsqueeze_1224, 2);  unsqueeze_1224 = None
        unsqueeze_1226 = torch.ops.aten.unsqueeze.default(unsqueeze_1225, 3);  unsqueeze_1225 = None
        mul_1306 = torch.ops.aten.mul.Tensor(sub_372, unsqueeze_1223);  sub_372 = unsqueeze_1223 = None
        sub_374 = torch.ops.aten.sub.Tensor(where_69, mul_1306);  where_69 = mul_1306 = None
        sub_375 = torch.ops.aten.sub.Tensor(sub_374, unsqueeze_1220);  sub_374 = unsqueeze_1220 = None
        mul_1307 = torch.ops.aten.mul.Tensor(sub_375, unsqueeze_1226);  sub_375 = unsqueeze_1226 = None
        mul_1308 = torch.ops.aten.mul.Tensor(sum_142, squeeze_79);  sum_142 = squeeze_79 = None
        convolution_backward_69 = torch.ops.aten.convolution_backward.default(mul_1307, cat_3, primals_79, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1307 = cat_3 = primals_79 = None
        getitem_407 = convolution_backward_69[0]
        getitem_408 = convolution_backward_69[1];  convolution_backward_69 = None
        add_509 = torch.ops.aten.add.Tensor(add_508, getitem_407);  add_508 = getitem_407 = None
        slice_42 = torch.ops.aten.slice.Tensor(add_509, 1, 0, 64)
        slice_43 = torch.ops.aten.slice.Tensor(add_509, 1, 64, 128)
        slice_44 = torch.ops.aten.slice.Tensor(add_509, 1, 128, 224)
        slice_45 = torch.ops.aten.slice.Tensor(add_509, 1, 224, 288);  add_509 = None
        where_70 = torch.ops.aten.where.self(le_70, full_default, slice_45);  le_70 = slice_45 = None
        sum_143 = torch.ops.aten.sum.dim_IntList(where_70, [0, 2, 3])
        sub_376 = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_1229);  convolution_25 = unsqueeze_1229 = None
        mul_1309 = torch.ops.aten.mul.Tensor(where_70, sub_376)
        sum_144 = torch.ops.aten.sum.dim_IntList(mul_1309, [0, 2, 3]);  mul_1309 = None
        mul_1310 = torch.ops.aten.mul.Tensor(sum_143, 2.5510204081632654e-05)
        unsqueeze_1230 = torch.ops.aten.unsqueeze.default(mul_1310, 0);  mul_1310 = None
        unsqueeze_1231 = torch.ops.aten.unsqueeze.default(unsqueeze_1230, 2);  unsqueeze_1230 = None
        unsqueeze_1232 = torch.ops.aten.unsqueeze.default(unsqueeze_1231, 3);  unsqueeze_1231 = None
        mul_1311 = torch.ops.aten.mul.Tensor(sum_144, 2.5510204081632654e-05)
        mul_1312 = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
        mul_1313 = torch.ops.aten.mul.Tensor(mul_1311, mul_1312);  mul_1311 = mul_1312 = None
        unsqueeze_1233 = torch.ops.aten.unsqueeze.default(mul_1313, 0);  mul_1313 = None
        unsqueeze_1234 = torch.ops.aten.unsqueeze.default(unsqueeze_1233, 2);  unsqueeze_1233 = None
        unsqueeze_1235 = torch.ops.aten.unsqueeze.default(unsqueeze_1234, 3);  unsqueeze_1234 = None
        mul_1314 = torch.ops.aten.mul.Tensor(squeeze_76, primals_77);  primals_77 = None
        unsqueeze_1236 = torch.ops.aten.unsqueeze.default(mul_1314, 0);  mul_1314 = None
        unsqueeze_1237 = torch.ops.aten.unsqueeze.default(unsqueeze_1236, 2);  unsqueeze_1236 = None
        unsqueeze_1238 = torch.ops.aten.unsqueeze.default(unsqueeze_1237, 3);  unsqueeze_1237 = None
        mul_1315 = torch.ops.aten.mul.Tensor(sub_376, unsqueeze_1235);  sub_376 = unsqueeze_1235 = None
        sub_378 = torch.ops.aten.sub.Tensor(where_70, mul_1315);  where_70 = mul_1315 = None
        sub_379 = torch.ops.aten.sub.Tensor(sub_378, unsqueeze_1232);  sub_378 = unsqueeze_1232 = None
        mul_1316 = torch.ops.aten.mul.Tensor(sub_379, unsqueeze_1238);  sub_379 = unsqueeze_1238 = None
        mul_1317 = torch.ops.aten.mul.Tensor(sum_144, squeeze_76);  sum_144 = squeeze_76 = None
        convolution_backward_70 = torch.ops.aten.convolution_backward.default(mul_1316, avg_pool2d_2, primals_76, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1316 = avg_pool2d_2 = primals_76 = None
        getitem_410 = convolution_backward_70[0]
        getitem_411 = convolution_backward_70[1];  convolution_backward_70 = None
        avg_pool2d_backward_7 = torch.ops.aten.avg_pool2d_backward.default(getitem_410, cat_2, [3, 3], [1, 1], [1, 1], False, True, None);  getitem_410 = None
        where_71 = torch.ops.aten.where.self(le_71, full_default, slice_44);  le_71 = slice_44 = None
        sum_145 = torch.ops.aten.sum.dim_IntList(where_71, [0, 2, 3])
        sub_380 = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_1241);  convolution_24 = unsqueeze_1241 = None
        mul_1318 = torch.ops.aten.mul.Tensor(where_71, sub_380)
        sum_146 = torch.ops.aten.sum.dim_IntList(mul_1318, [0, 2, 3]);  mul_1318 = None
        mul_1319 = torch.ops.aten.mul.Tensor(sum_145, 2.5510204081632654e-05)
        unsqueeze_1242 = torch.ops.aten.unsqueeze.default(mul_1319, 0);  mul_1319 = None
        unsqueeze_1243 = torch.ops.aten.unsqueeze.default(unsqueeze_1242, 2);  unsqueeze_1242 = None
        unsqueeze_1244 = torch.ops.aten.unsqueeze.default(unsqueeze_1243, 3);  unsqueeze_1243 = None
        mul_1320 = torch.ops.aten.mul.Tensor(sum_146, 2.5510204081632654e-05)
        mul_1321 = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
        mul_1322 = torch.ops.aten.mul.Tensor(mul_1320, mul_1321);  mul_1320 = mul_1321 = None
        unsqueeze_1245 = torch.ops.aten.unsqueeze.default(mul_1322, 0);  mul_1322 = None
        unsqueeze_1246 = torch.ops.aten.unsqueeze.default(unsqueeze_1245, 2);  unsqueeze_1245 = None
        unsqueeze_1247 = torch.ops.aten.unsqueeze.default(unsqueeze_1246, 3);  unsqueeze_1246 = None
        mul_1323 = torch.ops.aten.mul.Tensor(squeeze_73, primals_74);  primals_74 = None
        unsqueeze_1248 = torch.ops.aten.unsqueeze.default(mul_1323, 0);  mul_1323 = None
        unsqueeze_1249 = torch.ops.aten.unsqueeze.default(unsqueeze_1248, 2);  unsqueeze_1248 = None
        unsqueeze_1250 = torch.ops.aten.unsqueeze.default(unsqueeze_1249, 3);  unsqueeze_1249 = None
        mul_1324 = torch.ops.aten.mul.Tensor(sub_380, unsqueeze_1247);  sub_380 = unsqueeze_1247 = None
        sub_382 = torch.ops.aten.sub.Tensor(where_71, mul_1324);  where_71 = mul_1324 = None
        sub_383 = torch.ops.aten.sub.Tensor(sub_382, unsqueeze_1244);  sub_382 = unsqueeze_1244 = None
        mul_1325 = torch.ops.aten.mul.Tensor(sub_383, unsqueeze_1250);  sub_383 = unsqueeze_1250 = None
        mul_1326 = torch.ops.aten.mul.Tensor(sum_146, squeeze_73);  sum_146 = squeeze_73 = None
        convolution_backward_71 = torch.ops.aten.convolution_backward.default(mul_1325, relu_23, primals_73, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1325 = primals_73 = None
        getitem_413 = convolution_backward_71[0]
        getitem_414 = convolution_backward_71[1];  convolution_backward_71 = None
        alias_482 = torch.ops.aten.alias.default(relu_23);  relu_23 = None
        alias_483 = torch.ops.aten.alias.default(alias_482);  alias_482 = None
        le_72 = torch.ops.aten.le.Scalar(alias_483, 0);  alias_483 = None
        where_72 = torch.ops.aten.where.self(le_72, full_default, getitem_413);  le_72 = getitem_413 = None
        sum_147 = torch.ops.aten.sum.dim_IntList(where_72, [0, 2, 3])
        sub_384 = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_1253);  convolution_23 = unsqueeze_1253 = None
        mul_1327 = torch.ops.aten.mul.Tensor(where_72, sub_384)
        sum_148 = torch.ops.aten.sum.dim_IntList(mul_1327, [0, 2, 3]);  mul_1327 = None
        mul_1328 = torch.ops.aten.mul.Tensor(sum_147, 2.5510204081632654e-05)
        unsqueeze_1254 = torch.ops.aten.unsqueeze.default(mul_1328, 0);  mul_1328 = None
        unsqueeze_1255 = torch.ops.aten.unsqueeze.default(unsqueeze_1254, 2);  unsqueeze_1254 = None
        unsqueeze_1256 = torch.ops.aten.unsqueeze.default(unsqueeze_1255, 3);  unsqueeze_1255 = None
        mul_1329 = torch.ops.aten.mul.Tensor(sum_148, 2.5510204081632654e-05)
        mul_1330 = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
        mul_1331 = torch.ops.aten.mul.Tensor(mul_1329, mul_1330);  mul_1329 = mul_1330 = None
        unsqueeze_1257 = torch.ops.aten.unsqueeze.default(mul_1331, 0);  mul_1331 = None
        unsqueeze_1258 = torch.ops.aten.unsqueeze.default(unsqueeze_1257, 2);  unsqueeze_1257 = None
        unsqueeze_1259 = torch.ops.aten.unsqueeze.default(unsqueeze_1258, 3);  unsqueeze_1258 = None
        mul_1332 = torch.ops.aten.mul.Tensor(squeeze_70, primals_71);  primals_71 = None
        unsqueeze_1260 = torch.ops.aten.unsqueeze.default(mul_1332, 0);  mul_1332 = None
        unsqueeze_1261 = torch.ops.aten.unsqueeze.default(unsqueeze_1260, 2);  unsqueeze_1260 = None
        unsqueeze_1262 = torch.ops.aten.unsqueeze.default(unsqueeze_1261, 3);  unsqueeze_1261 = None
        mul_1333 = torch.ops.aten.mul.Tensor(sub_384, unsqueeze_1259);  sub_384 = unsqueeze_1259 = None
        sub_386 = torch.ops.aten.sub.Tensor(where_72, mul_1333);  where_72 = mul_1333 = None
        sub_387 = torch.ops.aten.sub.Tensor(sub_386, unsqueeze_1256);  sub_386 = unsqueeze_1256 = None
        mul_1334 = torch.ops.aten.mul.Tensor(sub_387, unsqueeze_1262);  sub_387 = unsqueeze_1262 = None
        mul_1335 = torch.ops.aten.mul.Tensor(sum_148, squeeze_70);  sum_148 = squeeze_70 = None
        convolution_backward_72 = torch.ops.aten.convolution_backward.default(mul_1334, relu_22, primals_70, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1334 = primals_70 = None
        getitem_416 = convolution_backward_72[0]
        getitem_417 = convolution_backward_72[1];  convolution_backward_72 = None
        alias_486 = torch.ops.aten.alias.default(relu_22);  relu_22 = None
        alias_487 = torch.ops.aten.alias.default(alias_486);  alias_486 = None
        le_73 = torch.ops.aten.le.Scalar(alias_487, 0);  alias_487 = None
        where_73 = torch.ops.aten.where.self(le_73, full_default, getitem_416);  le_73 = getitem_416 = None
        sum_149 = torch.ops.aten.sum.dim_IntList(where_73, [0, 2, 3])
        sub_388 = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_1265);  convolution_22 = unsqueeze_1265 = None
        mul_1336 = torch.ops.aten.mul.Tensor(where_73, sub_388)
        sum_150 = torch.ops.aten.sum.dim_IntList(mul_1336, [0, 2, 3]);  mul_1336 = None
        mul_1337 = torch.ops.aten.mul.Tensor(sum_149, 2.5510204081632654e-05)
        unsqueeze_1266 = torch.ops.aten.unsqueeze.default(mul_1337, 0);  mul_1337 = None
        unsqueeze_1267 = torch.ops.aten.unsqueeze.default(unsqueeze_1266, 2);  unsqueeze_1266 = None
        unsqueeze_1268 = torch.ops.aten.unsqueeze.default(unsqueeze_1267, 3);  unsqueeze_1267 = None
        mul_1338 = torch.ops.aten.mul.Tensor(sum_150, 2.5510204081632654e-05)
        mul_1339 = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
        mul_1340 = torch.ops.aten.mul.Tensor(mul_1338, mul_1339);  mul_1338 = mul_1339 = None
        unsqueeze_1269 = torch.ops.aten.unsqueeze.default(mul_1340, 0);  mul_1340 = None
        unsqueeze_1270 = torch.ops.aten.unsqueeze.default(unsqueeze_1269, 2);  unsqueeze_1269 = None
        unsqueeze_1271 = torch.ops.aten.unsqueeze.default(unsqueeze_1270, 3);  unsqueeze_1270 = None
        mul_1341 = torch.ops.aten.mul.Tensor(squeeze_67, primals_68);  primals_68 = None
        unsqueeze_1272 = torch.ops.aten.unsqueeze.default(mul_1341, 0);  mul_1341 = None
        unsqueeze_1273 = torch.ops.aten.unsqueeze.default(unsqueeze_1272, 2);  unsqueeze_1272 = None
        unsqueeze_1274 = torch.ops.aten.unsqueeze.default(unsqueeze_1273, 3);  unsqueeze_1273 = None
        mul_1342 = torch.ops.aten.mul.Tensor(sub_388, unsqueeze_1271);  sub_388 = unsqueeze_1271 = None
        sub_390 = torch.ops.aten.sub.Tensor(where_73, mul_1342);  where_73 = mul_1342 = None
        sub_391 = torch.ops.aten.sub.Tensor(sub_390, unsqueeze_1268);  sub_390 = unsqueeze_1268 = None
        mul_1343 = torch.ops.aten.mul.Tensor(sub_391, unsqueeze_1274);  sub_391 = unsqueeze_1274 = None
        mul_1344 = torch.ops.aten.mul.Tensor(sum_150, squeeze_67);  sum_150 = squeeze_67 = None
        convolution_backward_73 = torch.ops.aten.convolution_backward.default(mul_1343, cat_2, primals_67, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1343 = primals_67 = None
        getitem_419 = convolution_backward_73[0]
        getitem_420 = convolution_backward_73[1];  convolution_backward_73 = None
        add_510 = torch.ops.aten.add.Tensor(avg_pool2d_backward_7, getitem_419);  avg_pool2d_backward_7 = getitem_419 = None
        where_74 = torch.ops.aten.where.self(le_74, full_default, slice_43);  le_74 = slice_43 = None
        sum_151 = torch.ops.aten.sum.dim_IntList(where_74, [0, 2, 3])
        sub_392 = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_1277);  convolution_21 = unsqueeze_1277 = None
        mul_1345 = torch.ops.aten.mul.Tensor(where_74, sub_392)
        sum_152 = torch.ops.aten.sum.dim_IntList(mul_1345, [0, 2, 3]);  mul_1345 = None
        mul_1346 = torch.ops.aten.mul.Tensor(sum_151, 2.5510204081632654e-05)
        unsqueeze_1278 = torch.ops.aten.unsqueeze.default(mul_1346, 0);  mul_1346 = None
        unsqueeze_1279 = torch.ops.aten.unsqueeze.default(unsqueeze_1278, 2);  unsqueeze_1278 = None
        unsqueeze_1280 = torch.ops.aten.unsqueeze.default(unsqueeze_1279, 3);  unsqueeze_1279 = None
        mul_1347 = torch.ops.aten.mul.Tensor(sum_152, 2.5510204081632654e-05)
        mul_1348 = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
        mul_1349 = torch.ops.aten.mul.Tensor(mul_1347, mul_1348);  mul_1347 = mul_1348 = None
        unsqueeze_1281 = torch.ops.aten.unsqueeze.default(mul_1349, 0);  mul_1349 = None
        unsqueeze_1282 = torch.ops.aten.unsqueeze.default(unsqueeze_1281, 2);  unsqueeze_1281 = None
        unsqueeze_1283 = torch.ops.aten.unsqueeze.default(unsqueeze_1282, 3);  unsqueeze_1282 = None
        mul_1350 = torch.ops.aten.mul.Tensor(squeeze_64, primals_65);  primals_65 = None
        unsqueeze_1284 = torch.ops.aten.unsqueeze.default(mul_1350, 0);  mul_1350 = None
        unsqueeze_1285 = torch.ops.aten.unsqueeze.default(unsqueeze_1284, 2);  unsqueeze_1284 = None
        unsqueeze_1286 = torch.ops.aten.unsqueeze.default(unsqueeze_1285, 3);  unsqueeze_1285 = None
        mul_1351 = torch.ops.aten.mul.Tensor(sub_392, unsqueeze_1283);  sub_392 = unsqueeze_1283 = None
        sub_394 = torch.ops.aten.sub.Tensor(where_74, mul_1351);  where_74 = mul_1351 = None
        sub_395 = torch.ops.aten.sub.Tensor(sub_394, unsqueeze_1280);  sub_394 = unsqueeze_1280 = None
        mul_1352 = torch.ops.aten.mul.Tensor(sub_395, unsqueeze_1286);  sub_395 = unsqueeze_1286 = None
        mul_1353 = torch.ops.aten.mul.Tensor(sum_152, squeeze_64);  sum_152 = squeeze_64 = None
        convolution_backward_74 = torch.ops.aten.convolution_backward.default(mul_1352, relu_20, primals_64, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1352 = primals_64 = None
        getitem_422 = convolution_backward_74[0]
        getitem_423 = convolution_backward_74[1];  convolution_backward_74 = None
        alias_494 = torch.ops.aten.alias.default(relu_20);  relu_20 = None
        alias_495 = torch.ops.aten.alias.default(alias_494);  alias_494 = None
        le_75 = torch.ops.aten.le.Scalar(alias_495, 0);  alias_495 = None
        where_75 = torch.ops.aten.where.self(le_75, full_default, getitem_422);  le_75 = getitem_422 = None
        sum_153 = torch.ops.aten.sum.dim_IntList(where_75, [0, 2, 3])
        sub_396 = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_1289);  convolution_20 = unsqueeze_1289 = None
        mul_1354 = torch.ops.aten.mul.Tensor(where_75, sub_396)
        sum_154 = torch.ops.aten.sum.dim_IntList(mul_1354, [0, 2, 3]);  mul_1354 = None
        mul_1355 = torch.ops.aten.mul.Tensor(sum_153, 2.5510204081632654e-05)
        unsqueeze_1290 = torch.ops.aten.unsqueeze.default(mul_1355, 0);  mul_1355 = None
        unsqueeze_1291 = torch.ops.aten.unsqueeze.default(unsqueeze_1290, 2);  unsqueeze_1290 = None
        unsqueeze_1292 = torch.ops.aten.unsqueeze.default(unsqueeze_1291, 3);  unsqueeze_1291 = None
        mul_1356 = torch.ops.aten.mul.Tensor(sum_154, 2.5510204081632654e-05)
        mul_1357 = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
        mul_1358 = torch.ops.aten.mul.Tensor(mul_1356, mul_1357);  mul_1356 = mul_1357 = None
        unsqueeze_1293 = torch.ops.aten.unsqueeze.default(mul_1358, 0);  mul_1358 = None
        unsqueeze_1294 = torch.ops.aten.unsqueeze.default(unsqueeze_1293, 2);  unsqueeze_1293 = None
        unsqueeze_1295 = torch.ops.aten.unsqueeze.default(unsqueeze_1294, 3);  unsqueeze_1294 = None
        mul_1359 = torch.ops.aten.mul.Tensor(squeeze_61, primals_62);  primals_62 = None
        unsqueeze_1296 = torch.ops.aten.unsqueeze.default(mul_1359, 0);  mul_1359 = None
        unsqueeze_1297 = torch.ops.aten.unsqueeze.default(unsqueeze_1296, 2);  unsqueeze_1296 = None
        unsqueeze_1298 = torch.ops.aten.unsqueeze.default(unsqueeze_1297, 3);  unsqueeze_1297 = None
        mul_1360 = torch.ops.aten.mul.Tensor(sub_396, unsqueeze_1295);  sub_396 = unsqueeze_1295 = None
        sub_398 = torch.ops.aten.sub.Tensor(where_75, mul_1360);  where_75 = mul_1360 = None
        sub_399 = torch.ops.aten.sub.Tensor(sub_398, unsqueeze_1292);  sub_398 = unsqueeze_1292 = None
        mul_1361 = torch.ops.aten.mul.Tensor(sub_399, unsqueeze_1298);  sub_399 = unsqueeze_1298 = None
        mul_1362 = torch.ops.aten.mul.Tensor(sum_154, squeeze_61);  sum_154 = squeeze_61 = None
        convolution_backward_75 = torch.ops.aten.convolution_backward.default(mul_1361, cat_2, primals_61, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1361 = primals_61 = None
        getitem_425 = convolution_backward_75[0]
        getitem_426 = convolution_backward_75[1];  convolution_backward_75 = None
        add_511 = torch.ops.aten.add.Tensor(add_510, getitem_425);  add_510 = getitem_425 = None
        where_76 = torch.ops.aten.where.self(le_76, full_default, slice_42);  le_76 = slice_42 = None
        sum_155 = torch.ops.aten.sum.dim_IntList(where_76, [0, 2, 3])
        sub_400 = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_1301);  convolution_19 = unsqueeze_1301 = None
        mul_1363 = torch.ops.aten.mul.Tensor(where_76, sub_400)
        sum_156 = torch.ops.aten.sum.dim_IntList(mul_1363, [0, 2, 3]);  mul_1363 = None
        mul_1364 = torch.ops.aten.mul.Tensor(sum_155, 2.5510204081632654e-05)
        unsqueeze_1302 = torch.ops.aten.unsqueeze.default(mul_1364, 0);  mul_1364 = None
        unsqueeze_1303 = torch.ops.aten.unsqueeze.default(unsqueeze_1302, 2);  unsqueeze_1302 = None
        unsqueeze_1304 = torch.ops.aten.unsqueeze.default(unsqueeze_1303, 3);  unsqueeze_1303 = None
        mul_1365 = torch.ops.aten.mul.Tensor(sum_156, 2.5510204081632654e-05)
        mul_1366 = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
        mul_1367 = torch.ops.aten.mul.Tensor(mul_1365, mul_1366);  mul_1365 = mul_1366 = None
        unsqueeze_1305 = torch.ops.aten.unsqueeze.default(mul_1367, 0);  mul_1367 = None
        unsqueeze_1306 = torch.ops.aten.unsqueeze.default(unsqueeze_1305, 2);  unsqueeze_1305 = None
        unsqueeze_1307 = torch.ops.aten.unsqueeze.default(unsqueeze_1306, 3);  unsqueeze_1306 = None
        mul_1368 = torch.ops.aten.mul.Tensor(squeeze_58, primals_59);  primals_59 = None
        unsqueeze_1308 = torch.ops.aten.unsqueeze.default(mul_1368, 0);  mul_1368 = None
        unsqueeze_1309 = torch.ops.aten.unsqueeze.default(unsqueeze_1308, 2);  unsqueeze_1308 = None
        unsqueeze_1310 = torch.ops.aten.unsqueeze.default(unsqueeze_1309, 3);  unsqueeze_1309 = None
        mul_1369 = torch.ops.aten.mul.Tensor(sub_400, unsqueeze_1307);  sub_400 = unsqueeze_1307 = None
        sub_402 = torch.ops.aten.sub.Tensor(where_76, mul_1369);  where_76 = mul_1369 = None
        sub_403 = torch.ops.aten.sub.Tensor(sub_402, unsqueeze_1304);  sub_402 = unsqueeze_1304 = None
        mul_1370 = torch.ops.aten.mul.Tensor(sub_403, unsqueeze_1310);  sub_403 = unsqueeze_1310 = None
        mul_1371 = torch.ops.aten.mul.Tensor(sum_156, squeeze_58);  sum_156 = squeeze_58 = None
        convolution_backward_76 = torch.ops.aten.convolution_backward.default(mul_1370, cat_2, primals_58, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1370 = cat_2 = primals_58 = None
        getitem_428 = convolution_backward_76[0]
        getitem_429 = convolution_backward_76[1];  convolution_backward_76 = None
        add_512 = torch.ops.aten.add.Tensor(add_511, getitem_428);  add_511 = getitem_428 = None
        slice_46 = torch.ops.aten.slice.Tensor(add_512, 1, 0, 64)
        slice_47 = torch.ops.aten.slice.Tensor(add_512, 1, 64, 128)
        slice_48 = torch.ops.aten.slice.Tensor(add_512, 1, 128, 224)
        slice_49 = torch.ops.aten.slice.Tensor(add_512, 1, 224, 288);  add_512 = None
        where_77 = torch.ops.aten.where.self(le_77, full_default, slice_49);  le_77 = slice_49 = None
        sum_157 = torch.ops.aten.sum.dim_IntList(where_77, [0, 2, 3])
        sub_404 = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_1313);  convolution_18 = unsqueeze_1313 = None
        mul_1372 = torch.ops.aten.mul.Tensor(where_77, sub_404)
        sum_158 = torch.ops.aten.sum.dim_IntList(mul_1372, [0, 2, 3]);  mul_1372 = None
        mul_1373 = torch.ops.aten.mul.Tensor(sum_157, 2.5510204081632654e-05)
        unsqueeze_1314 = torch.ops.aten.unsqueeze.default(mul_1373, 0);  mul_1373 = None
        unsqueeze_1315 = torch.ops.aten.unsqueeze.default(unsqueeze_1314, 2);  unsqueeze_1314 = None
        unsqueeze_1316 = torch.ops.aten.unsqueeze.default(unsqueeze_1315, 3);  unsqueeze_1315 = None
        mul_1374 = torch.ops.aten.mul.Tensor(sum_158, 2.5510204081632654e-05)
        mul_1375 = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
        mul_1376 = torch.ops.aten.mul.Tensor(mul_1374, mul_1375);  mul_1374 = mul_1375 = None
        unsqueeze_1317 = torch.ops.aten.unsqueeze.default(mul_1376, 0);  mul_1376 = None
        unsqueeze_1318 = torch.ops.aten.unsqueeze.default(unsqueeze_1317, 2);  unsqueeze_1317 = None
        unsqueeze_1319 = torch.ops.aten.unsqueeze.default(unsqueeze_1318, 3);  unsqueeze_1318 = None
        mul_1377 = torch.ops.aten.mul.Tensor(squeeze_55, primals_56);  primals_56 = None
        unsqueeze_1320 = torch.ops.aten.unsqueeze.default(mul_1377, 0);  mul_1377 = None
        unsqueeze_1321 = torch.ops.aten.unsqueeze.default(unsqueeze_1320, 2);  unsqueeze_1320 = None
        unsqueeze_1322 = torch.ops.aten.unsqueeze.default(unsqueeze_1321, 3);  unsqueeze_1321 = None
        mul_1378 = torch.ops.aten.mul.Tensor(sub_404, unsqueeze_1319);  sub_404 = unsqueeze_1319 = None
        sub_406 = torch.ops.aten.sub.Tensor(where_77, mul_1378);  where_77 = mul_1378 = None
        sub_407 = torch.ops.aten.sub.Tensor(sub_406, unsqueeze_1316);  sub_406 = unsqueeze_1316 = None
        mul_1379 = torch.ops.aten.mul.Tensor(sub_407, unsqueeze_1322);  sub_407 = unsqueeze_1322 = None
        mul_1380 = torch.ops.aten.mul.Tensor(sum_158, squeeze_55);  sum_158 = squeeze_55 = None
        convolution_backward_77 = torch.ops.aten.convolution_backward.default(mul_1379, avg_pool2d_1, primals_55, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1379 = avg_pool2d_1 = primals_55 = None
        getitem_431 = convolution_backward_77[0]
        getitem_432 = convolution_backward_77[1];  convolution_backward_77 = None
        avg_pool2d_backward_8 = torch.ops.aten.avg_pool2d_backward.default(getitem_431, cat_1, [3, 3], [1, 1], [1, 1], False, True, None);  getitem_431 = None
        where_78 = torch.ops.aten.where.self(le_78, full_default, slice_48);  le_78 = slice_48 = None
        sum_159 = torch.ops.aten.sum.dim_IntList(where_78, [0, 2, 3])
        sub_408 = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_1325);  convolution_17 = unsqueeze_1325 = None
        mul_1381 = torch.ops.aten.mul.Tensor(where_78, sub_408)
        sum_160 = torch.ops.aten.sum.dim_IntList(mul_1381, [0, 2, 3]);  mul_1381 = None
        mul_1382 = torch.ops.aten.mul.Tensor(sum_159, 2.5510204081632654e-05)
        unsqueeze_1326 = torch.ops.aten.unsqueeze.default(mul_1382, 0);  mul_1382 = None
        unsqueeze_1327 = torch.ops.aten.unsqueeze.default(unsqueeze_1326, 2);  unsqueeze_1326 = None
        unsqueeze_1328 = torch.ops.aten.unsqueeze.default(unsqueeze_1327, 3);  unsqueeze_1327 = None
        mul_1383 = torch.ops.aten.mul.Tensor(sum_160, 2.5510204081632654e-05)
        mul_1384 = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
        mul_1385 = torch.ops.aten.mul.Tensor(mul_1383, mul_1384);  mul_1383 = mul_1384 = None
        unsqueeze_1329 = torch.ops.aten.unsqueeze.default(mul_1385, 0);  mul_1385 = None
        unsqueeze_1330 = torch.ops.aten.unsqueeze.default(unsqueeze_1329, 2);  unsqueeze_1329 = None
        unsqueeze_1331 = torch.ops.aten.unsqueeze.default(unsqueeze_1330, 3);  unsqueeze_1330 = None
        mul_1386 = torch.ops.aten.mul.Tensor(squeeze_52, primals_53);  primals_53 = None
        unsqueeze_1332 = torch.ops.aten.unsqueeze.default(mul_1386, 0);  mul_1386 = None
        unsqueeze_1333 = torch.ops.aten.unsqueeze.default(unsqueeze_1332, 2);  unsqueeze_1332 = None
        unsqueeze_1334 = torch.ops.aten.unsqueeze.default(unsqueeze_1333, 3);  unsqueeze_1333 = None
        mul_1387 = torch.ops.aten.mul.Tensor(sub_408, unsqueeze_1331);  sub_408 = unsqueeze_1331 = None
        sub_410 = torch.ops.aten.sub.Tensor(where_78, mul_1387);  where_78 = mul_1387 = None
        sub_411 = torch.ops.aten.sub.Tensor(sub_410, unsqueeze_1328);  sub_410 = unsqueeze_1328 = None
        mul_1388 = torch.ops.aten.mul.Tensor(sub_411, unsqueeze_1334);  sub_411 = unsqueeze_1334 = None
        mul_1389 = torch.ops.aten.mul.Tensor(sum_160, squeeze_52);  sum_160 = squeeze_52 = None
        convolution_backward_78 = torch.ops.aten.convolution_backward.default(mul_1388, relu_16, primals_52, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1388 = primals_52 = None
        getitem_434 = convolution_backward_78[0]
        getitem_435 = convolution_backward_78[1];  convolution_backward_78 = None
        alias_510 = torch.ops.aten.alias.default(relu_16);  relu_16 = None
        alias_511 = torch.ops.aten.alias.default(alias_510);  alias_510 = None
        le_79 = torch.ops.aten.le.Scalar(alias_511, 0);  alias_511 = None
        where_79 = torch.ops.aten.where.self(le_79, full_default, getitem_434);  le_79 = getitem_434 = None
        sum_161 = torch.ops.aten.sum.dim_IntList(where_79, [0, 2, 3])
        sub_412 = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_1337);  convolution_16 = unsqueeze_1337 = None
        mul_1390 = torch.ops.aten.mul.Tensor(where_79, sub_412)
        sum_162 = torch.ops.aten.sum.dim_IntList(mul_1390, [0, 2, 3]);  mul_1390 = None
        mul_1391 = torch.ops.aten.mul.Tensor(sum_161, 2.5510204081632654e-05)
        unsqueeze_1338 = torch.ops.aten.unsqueeze.default(mul_1391, 0);  mul_1391 = None
        unsqueeze_1339 = torch.ops.aten.unsqueeze.default(unsqueeze_1338, 2);  unsqueeze_1338 = None
        unsqueeze_1340 = torch.ops.aten.unsqueeze.default(unsqueeze_1339, 3);  unsqueeze_1339 = None
        mul_1392 = torch.ops.aten.mul.Tensor(sum_162, 2.5510204081632654e-05)
        mul_1393 = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
        mul_1394 = torch.ops.aten.mul.Tensor(mul_1392, mul_1393);  mul_1392 = mul_1393 = None
        unsqueeze_1341 = torch.ops.aten.unsqueeze.default(mul_1394, 0);  mul_1394 = None
        unsqueeze_1342 = torch.ops.aten.unsqueeze.default(unsqueeze_1341, 2);  unsqueeze_1341 = None
        unsqueeze_1343 = torch.ops.aten.unsqueeze.default(unsqueeze_1342, 3);  unsqueeze_1342 = None
        mul_1395 = torch.ops.aten.mul.Tensor(squeeze_49, primals_50);  primals_50 = None
        unsqueeze_1344 = torch.ops.aten.unsqueeze.default(mul_1395, 0);  mul_1395 = None
        unsqueeze_1345 = torch.ops.aten.unsqueeze.default(unsqueeze_1344, 2);  unsqueeze_1344 = None
        unsqueeze_1346 = torch.ops.aten.unsqueeze.default(unsqueeze_1345, 3);  unsqueeze_1345 = None
        mul_1396 = torch.ops.aten.mul.Tensor(sub_412, unsqueeze_1343);  sub_412 = unsqueeze_1343 = None
        sub_414 = torch.ops.aten.sub.Tensor(where_79, mul_1396);  where_79 = mul_1396 = None
        sub_415 = torch.ops.aten.sub.Tensor(sub_414, unsqueeze_1340);  sub_414 = unsqueeze_1340 = None
        mul_1397 = torch.ops.aten.mul.Tensor(sub_415, unsqueeze_1346);  sub_415 = unsqueeze_1346 = None
        mul_1398 = torch.ops.aten.mul.Tensor(sum_162, squeeze_49);  sum_162 = squeeze_49 = None
        convolution_backward_79 = torch.ops.aten.convolution_backward.default(mul_1397, relu_15, primals_49, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1397 = primals_49 = None
        getitem_437 = convolution_backward_79[0]
        getitem_438 = convolution_backward_79[1];  convolution_backward_79 = None
        alias_514 = torch.ops.aten.alias.default(relu_15);  relu_15 = None
        alias_515 = torch.ops.aten.alias.default(alias_514);  alias_514 = None
        le_80 = torch.ops.aten.le.Scalar(alias_515, 0);  alias_515 = None
        where_80 = torch.ops.aten.where.self(le_80, full_default, getitem_437);  le_80 = getitem_437 = None
        sum_163 = torch.ops.aten.sum.dim_IntList(where_80, [0, 2, 3])
        sub_416 = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_1349);  convolution_15 = unsqueeze_1349 = None
        mul_1399 = torch.ops.aten.mul.Tensor(where_80, sub_416)
        sum_164 = torch.ops.aten.sum.dim_IntList(mul_1399, [0, 2, 3]);  mul_1399 = None
        mul_1400 = torch.ops.aten.mul.Tensor(sum_163, 2.5510204081632654e-05)
        unsqueeze_1350 = torch.ops.aten.unsqueeze.default(mul_1400, 0);  mul_1400 = None
        unsqueeze_1351 = torch.ops.aten.unsqueeze.default(unsqueeze_1350, 2);  unsqueeze_1350 = None
        unsqueeze_1352 = torch.ops.aten.unsqueeze.default(unsqueeze_1351, 3);  unsqueeze_1351 = None
        mul_1401 = torch.ops.aten.mul.Tensor(sum_164, 2.5510204081632654e-05)
        mul_1402 = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
        mul_1403 = torch.ops.aten.mul.Tensor(mul_1401, mul_1402);  mul_1401 = mul_1402 = None
        unsqueeze_1353 = torch.ops.aten.unsqueeze.default(mul_1403, 0);  mul_1403 = None
        unsqueeze_1354 = torch.ops.aten.unsqueeze.default(unsqueeze_1353, 2);  unsqueeze_1353 = None
        unsqueeze_1355 = torch.ops.aten.unsqueeze.default(unsqueeze_1354, 3);  unsqueeze_1354 = None
        mul_1404 = torch.ops.aten.mul.Tensor(squeeze_46, primals_47);  primals_47 = None
        unsqueeze_1356 = torch.ops.aten.unsqueeze.default(mul_1404, 0);  mul_1404 = None
        unsqueeze_1357 = torch.ops.aten.unsqueeze.default(unsqueeze_1356, 2);  unsqueeze_1356 = None
        unsqueeze_1358 = torch.ops.aten.unsqueeze.default(unsqueeze_1357, 3);  unsqueeze_1357 = None
        mul_1405 = torch.ops.aten.mul.Tensor(sub_416, unsqueeze_1355);  sub_416 = unsqueeze_1355 = None
        sub_418 = torch.ops.aten.sub.Tensor(where_80, mul_1405);  where_80 = mul_1405 = None
        sub_419 = torch.ops.aten.sub.Tensor(sub_418, unsqueeze_1352);  sub_418 = unsqueeze_1352 = None
        mul_1406 = torch.ops.aten.mul.Tensor(sub_419, unsqueeze_1358);  sub_419 = unsqueeze_1358 = None
        mul_1407 = torch.ops.aten.mul.Tensor(sum_164, squeeze_46);  sum_164 = squeeze_46 = None
        convolution_backward_80 = torch.ops.aten.convolution_backward.default(mul_1406, cat_1, primals_46, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1406 = primals_46 = None
        getitem_440 = convolution_backward_80[0]
        getitem_441 = convolution_backward_80[1];  convolution_backward_80 = None
        add_513 = torch.ops.aten.add.Tensor(avg_pool2d_backward_8, getitem_440);  avg_pool2d_backward_8 = getitem_440 = None
        where_81 = torch.ops.aten.where.self(le_81, full_default, slice_47);  le_81 = slice_47 = None
        sum_165 = torch.ops.aten.sum.dim_IntList(where_81, [0, 2, 3])
        sub_420 = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_1361);  convolution_14 = unsqueeze_1361 = None
        mul_1408 = torch.ops.aten.mul.Tensor(where_81, sub_420)
        sum_166 = torch.ops.aten.sum.dim_IntList(mul_1408, [0, 2, 3]);  mul_1408 = None
        mul_1409 = torch.ops.aten.mul.Tensor(sum_165, 2.5510204081632654e-05)
        unsqueeze_1362 = torch.ops.aten.unsqueeze.default(mul_1409, 0);  mul_1409 = None
        unsqueeze_1363 = torch.ops.aten.unsqueeze.default(unsqueeze_1362, 2);  unsqueeze_1362 = None
        unsqueeze_1364 = torch.ops.aten.unsqueeze.default(unsqueeze_1363, 3);  unsqueeze_1363 = None
        mul_1410 = torch.ops.aten.mul.Tensor(sum_166, 2.5510204081632654e-05)
        mul_1411 = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
        mul_1412 = torch.ops.aten.mul.Tensor(mul_1410, mul_1411);  mul_1410 = mul_1411 = None
        unsqueeze_1365 = torch.ops.aten.unsqueeze.default(mul_1412, 0);  mul_1412 = None
        unsqueeze_1366 = torch.ops.aten.unsqueeze.default(unsqueeze_1365, 2);  unsqueeze_1365 = None
        unsqueeze_1367 = torch.ops.aten.unsqueeze.default(unsqueeze_1366, 3);  unsqueeze_1366 = None
        mul_1413 = torch.ops.aten.mul.Tensor(squeeze_43, primals_44);  primals_44 = None
        unsqueeze_1368 = torch.ops.aten.unsqueeze.default(mul_1413, 0);  mul_1413 = None
        unsqueeze_1369 = torch.ops.aten.unsqueeze.default(unsqueeze_1368, 2);  unsqueeze_1368 = None
        unsqueeze_1370 = torch.ops.aten.unsqueeze.default(unsqueeze_1369, 3);  unsqueeze_1369 = None
        mul_1414 = torch.ops.aten.mul.Tensor(sub_420, unsqueeze_1367);  sub_420 = unsqueeze_1367 = None
        sub_422 = torch.ops.aten.sub.Tensor(where_81, mul_1414);  where_81 = mul_1414 = None
        sub_423 = torch.ops.aten.sub.Tensor(sub_422, unsqueeze_1364);  sub_422 = unsqueeze_1364 = None
        mul_1415 = torch.ops.aten.mul.Tensor(sub_423, unsqueeze_1370);  sub_423 = unsqueeze_1370 = None
        mul_1416 = torch.ops.aten.mul.Tensor(sum_166, squeeze_43);  sum_166 = squeeze_43 = None
        convolution_backward_81 = torch.ops.aten.convolution_backward.default(mul_1415, relu_13, primals_43, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1415 = primals_43 = None
        getitem_443 = convolution_backward_81[0]
        getitem_444 = convolution_backward_81[1];  convolution_backward_81 = None
        alias_522 = torch.ops.aten.alias.default(relu_13);  relu_13 = None
        alias_523 = torch.ops.aten.alias.default(alias_522);  alias_522 = None
        le_82 = torch.ops.aten.le.Scalar(alias_523, 0);  alias_523 = None
        where_82 = torch.ops.aten.where.self(le_82, full_default, getitem_443);  le_82 = getitem_443 = None
        sum_167 = torch.ops.aten.sum.dim_IntList(where_82, [0, 2, 3])
        sub_424 = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_1373);  convolution_13 = unsqueeze_1373 = None
        mul_1417 = torch.ops.aten.mul.Tensor(where_82, sub_424)
        sum_168 = torch.ops.aten.sum.dim_IntList(mul_1417, [0, 2, 3]);  mul_1417 = None
        mul_1418 = torch.ops.aten.mul.Tensor(sum_167, 2.5510204081632654e-05)
        unsqueeze_1374 = torch.ops.aten.unsqueeze.default(mul_1418, 0);  mul_1418 = None
        unsqueeze_1375 = torch.ops.aten.unsqueeze.default(unsqueeze_1374, 2);  unsqueeze_1374 = None
        unsqueeze_1376 = torch.ops.aten.unsqueeze.default(unsqueeze_1375, 3);  unsqueeze_1375 = None
        mul_1419 = torch.ops.aten.mul.Tensor(sum_168, 2.5510204081632654e-05)
        mul_1420 = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
        mul_1421 = torch.ops.aten.mul.Tensor(mul_1419, mul_1420);  mul_1419 = mul_1420 = None
        unsqueeze_1377 = torch.ops.aten.unsqueeze.default(mul_1421, 0);  mul_1421 = None
        unsqueeze_1378 = torch.ops.aten.unsqueeze.default(unsqueeze_1377, 2);  unsqueeze_1377 = None
        unsqueeze_1379 = torch.ops.aten.unsqueeze.default(unsqueeze_1378, 3);  unsqueeze_1378 = None
        mul_1422 = torch.ops.aten.mul.Tensor(squeeze_40, primals_41);  primals_41 = None
        unsqueeze_1380 = torch.ops.aten.unsqueeze.default(mul_1422, 0);  mul_1422 = None
        unsqueeze_1381 = torch.ops.aten.unsqueeze.default(unsqueeze_1380, 2);  unsqueeze_1380 = None
        unsqueeze_1382 = torch.ops.aten.unsqueeze.default(unsqueeze_1381, 3);  unsqueeze_1381 = None
        mul_1423 = torch.ops.aten.mul.Tensor(sub_424, unsqueeze_1379);  sub_424 = unsqueeze_1379 = None
        sub_426 = torch.ops.aten.sub.Tensor(where_82, mul_1423);  where_82 = mul_1423 = None
        sub_427 = torch.ops.aten.sub.Tensor(sub_426, unsqueeze_1376);  sub_426 = unsqueeze_1376 = None
        mul_1424 = torch.ops.aten.mul.Tensor(sub_427, unsqueeze_1382);  sub_427 = unsqueeze_1382 = None
        mul_1425 = torch.ops.aten.mul.Tensor(sum_168, squeeze_40);  sum_168 = squeeze_40 = None
        convolution_backward_82 = torch.ops.aten.convolution_backward.default(mul_1424, cat_1, primals_40, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1424 = primals_40 = None
        getitem_446 = convolution_backward_82[0]
        getitem_447 = convolution_backward_82[1];  convolution_backward_82 = None
        add_514 = torch.ops.aten.add.Tensor(add_513, getitem_446);  add_513 = getitem_446 = None
        where_83 = torch.ops.aten.where.self(le_83, full_default, slice_46);  le_83 = slice_46 = None
        sum_169 = torch.ops.aten.sum.dim_IntList(where_83, [0, 2, 3])
        sub_428 = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_1385);  convolution_12 = unsqueeze_1385 = None
        mul_1426 = torch.ops.aten.mul.Tensor(where_83, sub_428)
        sum_170 = torch.ops.aten.sum.dim_IntList(mul_1426, [0, 2, 3]);  mul_1426 = None
        mul_1427 = torch.ops.aten.mul.Tensor(sum_169, 2.5510204081632654e-05)
        unsqueeze_1386 = torch.ops.aten.unsqueeze.default(mul_1427, 0);  mul_1427 = None
        unsqueeze_1387 = torch.ops.aten.unsqueeze.default(unsqueeze_1386, 2);  unsqueeze_1386 = None
        unsqueeze_1388 = torch.ops.aten.unsqueeze.default(unsqueeze_1387, 3);  unsqueeze_1387 = None
        mul_1428 = torch.ops.aten.mul.Tensor(sum_170, 2.5510204081632654e-05)
        mul_1429 = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
        mul_1430 = torch.ops.aten.mul.Tensor(mul_1428, mul_1429);  mul_1428 = mul_1429 = None
        unsqueeze_1389 = torch.ops.aten.unsqueeze.default(mul_1430, 0);  mul_1430 = None
        unsqueeze_1390 = torch.ops.aten.unsqueeze.default(unsqueeze_1389, 2);  unsqueeze_1389 = None
        unsqueeze_1391 = torch.ops.aten.unsqueeze.default(unsqueeze_1390, 3);  unsqueeze_1390 = None
        mul_1431 = torch.ops.aten.mul.Tensor(squeeze_37, primals_38);  primals_38 = None
        unsqueeze_1392 = torch.ops.aten.unsqueeze.default(mul_1431, 0);  mul_1431 = None
        unsqueeze_1393 = torch.ops.aten.unsqueeze.default(unsqueeze_1392, 2);  unsqueeze_1392 = None
        unsqueeze_1394 = torch.ops.aten.unsqueeze.default(unsqueeze_1393, 3);  unsqueeze_1393 = None
        mul_1432 = torch.ops.aten.mul.Tensor(sub_428, unsqueeze_1391);  sub_428 = unsqueeze_1391 = None
        sub_430 = torch.ops.aten.sub.Tensor(where_83, mul_1432);  where_83 = mul_1432 = None
        sub_431 = torch.ops.aten.sub.Tensor(sub_430, unsqueeze_1388);  sub_430 = unsqueeze_1388 = None
        mul_1433 = torch.ops.aten.mul.Tensor(sub_431, unsqueeze_1394);  sub_431 = unsqueeze_1394 = None
        mul_1434 = torch.ops.aten.mul.Tensor(sum_170, squeeze_37);  sum_170 = squeeze_37 = None
        convolution_backward_83 = torch.ops.aten.convolution_backward.default(mul_1433, cat_1, primals_37, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1433 = cat_1 = primals_37 = None
        getitem_449 = convolution_backward_83[0]
        getitem_450 = convolution_backward_83[1];  convolution_backward_83 = None
        add_515 = torch.ops.aten.add.Tensor(add_514, getitem_449);  add_514 = getitem_449 = None
        slice_50 = torch.ops.aten.slice.Tensor(add_515, 1, 0, 64)
        slice_51 = torch.ops.aten.slice.Tensor(add_515, 1, 64, 128)
        slice_52 = torch.ops.aten.slice.Tensor(add_515, 1, 128, 224)
        slice_53 = torch.ops.aten.slice.Tensor(add_515, 1, 224, 256);  add_515 = None
        where_84 = torch.ops.aten.where.self(le_84, full_default, slice_53);  le_84 = slice_53 = None
        sum_171 = torch.ops.aten.sum.dim_IntList(where_84, [0, 2, 3])
        sub_432 = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_1397);  convolution_11 = unsqueeze_1397 = None
        mul_1435 = torch.ops.aten.mul.Tensor(where_84, sub_432)
        sum_172 = torch.ops.aten.sum.dim_IntList(mul_1435, [0, 2, 3]);  mul_1435 = None
        mul_1436 = torch.ops.aten.mul.Tensor(sum_171, 2.5510204081632654e-05)
        unsqueeze_1398 = torch.ops.aten.unsqueeze.default(mul_1436, 0);  mul_1436 = None
        unsqueeze_1399 = torch.ops.aten.unsqueeze.default(unsqueeze_1398, 2);  unsqueeze_1398 = None
        unsqueeze_1400 = torch.ops.aten.unsqueeze.default(unsqueeze_1399, 3);  unsqueeze_1399 = None
        mul_1437 = torch.ops.aten.mul.Tensor(sum_172, 2.5510204081632654e-05)
        mul_1438 = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
        mul_1439 = torch.ops.aten.mul.Tensor(mul_1437, mul_1438);  mul_1437 = mul_1438 = None
        unsqueeze_1401 = torch.ops.aten.unsqueeze.default(mul_1439, 0);  mul_1439 = None
        unsqueeze_1402 = torch.ops.aten.unsqueeze.default(unsqueeze_1401, 2);  unsqueeze_1401 = None
        unsqueeze_1403 = torch.ops.aten.unsqueeze.default(unsqueeze_1402, 3);  unsqueeze_1402 = None
        mul_1440 = torch.ops.aten.mul.Tensor(squeeze_34, primals_35);  primals_35 = None
        unsqueeze_1404 = torch.ops.aten.unsqueeze.default(mul_1440, 0);  mul_1440 = None
        unsqueeze_1405 = torch.ops.aten.unsqueeze.default(unsqueeze_1404, 2);  unsqueeze_1404 = None
        unsqueeze_1406 = torch.ops.aten.unsqueeze.default(unsqueeze_1405, 3);  unsqueeze_1405 = None
        mul_1441 = torch.ops.aten.mul.Tensor(sub_432, unsqueeze_1403);  sub_432 = unsqueeze_1403 = None
        sub_434 = torch.ops.aten.sub.Tensor(where_84, mul_1441);  where_84 = mul_1441 = None
        sub_435 = torch.ops.aten.sub.Tensor(sub_434, unsqueeze_1400);  sub_434 = unsqueeze_1400 = None
        mul_1442 = torch.ops.aten.mul.Tensor(sub_435, unsqueeze_1406);  sub_435 = unsqueeze_1406 = None
        mul_1443 = torch.ops.aten.mul.Tensor(sum_172, squeeze_34);  sum_172 = squeeze_34 = None
        convolution_backward_84 = torch.ops.aten.convolution_backward.default(mul_1442, avg_pool2d, primals_34, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1442 = avg_pool2d = primals_34 = None
        getitem_452 = convolution_backward_84[0]
        getitem_453 = convolution_backward_84[1];  convolution_backward_84 = None
        avg_pool2d_backward_9 = torch.ops.aten.avg_pool2d_backward.default(getitem_452, getitem_12, [3, 3], [1, 1], [1, 1], False, True, None);  getitem_452 = None
        where_85 = torch.ops.aten.where.self(le_85, full_default, slice_52);  le_85 = slice_52 = None
        sum_173 = torch.ops.aten.sum.dim_IntList(where_85, [0, 2, 3])
        sub_436 = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_1409);  convolution_10 = unsqueeze_1409 = None
        mul_1444 = torch.ops.aten.mul.Tensor(where_85, sub_436)
        sum_174 = torch.ops.aten.sum.dim_IntList(mul_1444, [0, 2, 3]);  mul_1444 = None
        mul_1445 = torch.ops.aten.mul.Tensor(sum_173, 2.5510204081632654e-05)
        unsqueeze_1410 = torch.ops.aten.unsqueeze.default(mul_1445, 0);  mul_1445 = None
        unsqueeze_1411 = torch.ops.aten.unsqueeze.default(unsqueeze_1410, 2);  unsqueeze_1410 = None
        unsqueeze_1412 = torch.ops.aten.unsqueeze.default(unsqueeze_1411, 3);  unsqueeze_1411 = None
        mul_1446 = torch.ops.aten.mul.Tensor(sum_174, 2.5510204081632654e-05)
        mul_1447 = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
        mul_1448 = torch.ops.aten.mul.Tensor(mul_1446, mul_1447);  mul_1446 = mul_1447 = None
        unsqueeze_1413 = torch.ops.aten.unsqueeze.default(mul_1448, 0);  mul_1448 = None
        unsqueeze_1414 = torch.ops.aten.unsqueeze.default(unsqueeze_1413, 2);  unsqueeze_1413 = None
        unsqueeze_1415 = torch.ops.aten.unsqueeze.default(unsqueeze_1414, 3);  unsqueeze_1414 = None
        mul_1449 = torch.ops.aten.mul.Tensor(squeeze_31, primals_32);  primals_32 = None
        unsqueeze_1416 = torch.ops.aten.unsqueeze.default(mul_1449, 0);  mul_1449 = None
        unsqueeze_1417 = torch.ops.aten.unsqueeze.default(unsqueeze_1416, 2);  unsqueeze_1416 = None
        unsqueeze_1418 = torch.ops.aten.unsqueeze.default(unsqueeze_1417, 3);  unsqueeze_1417 = None
        mul_1450 = torch.ops.aten.mul.Tensor(sub_436, unsqueeze_1415);  sub_436 = unsqueeze_1415 = None
        sub_438 = torch.ops.aten.sub.Tensor(where_85, mul_1450);  where_85 = mul_1450 = None
        sub_439 = torch.ops.aten.sub.Tensor(sub_438, unsqueeze_1412);  sub_438 = unsqueeze_1412 = None
        mul_1451 = torch.ops.aten.mul.Tensor(sub_439, unsqueeze_1418);  sub_439 = unsqueeze_1418 = None
        mul_1452 = torch.ops.aten.mul.Tensor(sum_174, squeeze_31);  sum_174 = squeeze_31 = None
        convolution_backward_85 = torch.ops.aten.convolution_backward.default(mul_1451, relu_9, primals_31, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1451 = primals_31 = None
        getitem_455 = convolution_backward_85[0]
        getitem_456 = convolution_backward_85[1];  convolution_backward_85 = None
        alias_538 = torch.ops.aten.alias.default(relu_9);  relu_9 = None
        alias_539 = torch.ops.aten.alias.default(alias_538);  alias_538 = None
        le_86 = torch.ops.aten.le.Scalar(alias_539, 0);  alias_539 = None
        where_86 = torch.ops.aten.where.self(le_86, full_default, getitem_455);  le_86 = getitem_455 = None
        sum_175 = torch.ops.aten.sum.dim_IntList(where_86, [0, 2, 3])
        sub_440 = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_1421);  convolution_9 = unsqueeze_1421 = None
        mul_1453 = torch.ops.aten.mul.Tensor(where_86, sub_440)
        sum_176 = torch.ops.aten.sum.dim_IntList(mul_1453, [0, 2, 3]);  mul_1453 = None
        mul_1454 = torch.ops.aten.mul.Tensor(sum_175, 2.5510204081632654e-05)
        unsqueeze_1422 = torch.ops.aten.unsqueeze.default(mul_1454, 0);  mul_1454 = None
        unsqueeze_1423 = torch.ops.aten.unsqueeze.default(unsqueeze_1422, 2);  unsqueeze_1422 = None
        unsqueeze_1424 = torch.ops.aten.unsqueeze.default(unsqueeze_1423, 3);  unsqueeze_1423 = None
        mul_1455 = torch.ops.aten.mul.Tensor(sum_176, 2.5510204081632654e-05)
        mul_1456 = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
        mul_1457 = torch.ops.aten.mul.Tensor(mul_1455, mul_1456);  mul_1455 = mul_1456 = None
        unsqueeze_1425 = torch.ops.aten.unsqueeze.default(mul_1457, 0);  mul_1457 = None
        unsqueeze_1426 = torch.ops.aten.unsqueeze.default(unsqueeze_1425, 2);  unsqueeze_1425 = None
        unsqueeze_1427 = torch.ops.aten.unsqueeze.default(unsqueeze_1426, 3);  unsqueeze_1426 = None
        mul_1458 = torch.ops.aten.mul.Tensor(squeeze_28, primals_29);  primals_29 = None
        unsqueeze_1428 = torch.ops.aten.unsqueeze.default(mul_1458, 0);  mul_1458 = None
        unsqueeze_1429 = torch.ops.aten.unsqueeze.default(unsqueeze_1428, 2);  unsqueeze_1428 = None
        unsqueeze_1430 = torch.ops.aten.unsqueeze.default(unsqueeze_1429, 3);  unsqueeze_1429 = None
        mul_1459 = torch.ops.aten.mul.Tensor(sub_440, unsqueeze_1427);  sub_440 = unsqueeze_1427 = None
        sub_442 = torch.ops.aten.sub.Tensor(where_86, mul_1459);  where_86 = mul_1459 = None
        sub_443 = torch.ops.aten.sub.Tensor(sub_442, unsqueeze_1424);  sub_442 = unsqueeze_1424 = None
        mul_1460 = torch.ops.aten.mul.Tensor(sub_443, unsqueeze_1430);  sub_443 = unsqueeze_1430 = None
        mul_1461 = torch.ops.aten.mul.Tensor(sum_176, squeeze_28);  sum_176 = squeeze_28 = None
        convolution_backward_86 = torch.ops.aten.convolution_backward.default(mul_1460, relu_8, primals_28, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1460 = primals_28 = None
        getitem_458 = convolution_backward_86[0]
        getitem_459 = convolution_backward_86[1];  convolution_backward_86 = None
        alias_542 = torch.ops.aten.alias.default(relu_8);  relu_8 = None
        alias_543 = torch.ops.aten.alias.default(alias_542);  alias_542 = None
        le_87 = torch.ops.aten.le.Scalar(alias_543, 0);  alias_543 = None
        where_87 = torch.ops.aten.where.self(le_87, full_default, getitem_458);  le_87 = getitem_458 = None
        sum_177 = torch.ops.aten.sum.dim_IntList(where_87, [0, 2, 3])
        sub_444 = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_1433);  convolution_8 = unsqueeze_1433 = None
        mul_1462 = torch.ops.aten.mul.Tensor(where_87, sub_444)
        sum_178 = torch.ops.aten.sum.dim_IntList(mul_1462, [0, 2, 3]);  mul_1462 = None
        mul_1463 = torch.ops.aten.mul.Tensor(sum_177, 2.5510204081632654e-05)
        unsqueeze_1434 = torch.ops.aten.unsqueeze.default(mul_1463, 0);  mul_1463 = None
        unsqueeze_1435 = torch.ops.aten.unsqueeze.default(unsqueeze_1434, 2);  unsqueeze_1434 = None
        unsqueeze_1436 = torch.ops.aten.unsqueeze.default(unsqueeze_1435, 3);  unsqueeze_1435 = None
        mul_1464 = torch.ops.aten.mul.Tensor(sum_178, 2.5510204081632654e-05)
        mul_1465 = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
        mul_1466 = torch.ops.aten.mul.Tensor(mul_1464, mul_1465);  mul_1464 = mul_1465 = None
        unsqueeze_1437 = torch.ops.aten.unsqueeze.default(mul_1466, 0);  mul_1466 = None
        unsqueeze_1438 = torch.ops.aten.unsqueeze.default(unsqueeze_1437, 2);  unsqueeze_1437 = None
        unsqueeze_1439 = torch.ops.aten.unsqueeze.default(unsqueeze_1438, 3);  unsqueeze_1438 = None
        mul_1467 = torch.ops.aten.mul.Tensor(squeeze_25, primals_26);  primals_26 = None
        unsqueeze_1440 = torch.ops.aten.unsqueeze.default(mul_1467, 0);  mul_1467 = None
        unsqueeze_1441 = torch.ops.aten.unsqueeze.default(unsqueeze_1440, 2);  unsqueeze_1440 = None
        unsqueeze_1442 = torch.ops.aten.unsqueeze.default(unsqueeze_1441, 3);  unsqueeze_1441 = None
        mul_1468 = torch.ops.aten.mul.Tensor(sub_444, unsqueeze_1439);  sub_444 = unsqueeze_1439 = None
        sub_446 = torch.ops.aten.sub.Tensor(where_87, mul_1468);  where_87 = mul_1468 = None
        sub_447 = torch.ops.aten.sub.Tensor(sub_446, unsqueeze_1436);  sub_446 = unsqueeze_1436 = None
        mul_1469 = torch.ops.aten.mul.Tensor(sub_447, unsqueeze_1442);  sub_447 = unsqueeze_1442 = None
        mul_1470 = torch.ops.aten.mul.Tensor(sum_178, squeeze_25);  sum_178 = squeeze_25 = None
        convolution_backward_87 = torch.ops.aten.convolution_backward.default(mul_1469, getitem_12, primals_25, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1469 = primals_25 = None
        getitem_461 = convolution_backward_87[0]
        getitem_462 = convolution_backward_87[1];  convolution_backward_87 = None
        add_516 = torch.ops.aten.add.Tensor(avg_pool2d_backward_9, getitem_461);  avg_pool2d_backward_9 = getitem_461 = None
        where_88 = torch.ops.aten.where.self(le_88, full_default, slice_51);  le_88 = slice_51 = None
        sum_179 = torch.ops.aten.sum.dim_IntList(where_88, [0, 2, 3])
        sub_448 = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_1445);  convolution_7 = unsqueeze_1445 = None
        mul_1471 = torch.ops.aten.mul.Tensor(where_88, sub_448)
        sum_180 = torch.ops.aten.sum.dim_IntList(mul_1471, [0, 2, 3]);  mul_1471 = None
        mul_1472 = torch.ops.aten.mul.Tensor(sum_179, 2.5510204081632654e-05)
        unsqueeze_1446 = torch.ops.aten.unsqueeze.default(mul_1472, 0);  mul_1472 = None
        unsqueeze_1447 = torch.ops.aten.unsqueeze.default(unsqueeze_1446, 2);  unsqueeze_1446 = None
        unsqueeze_1448 = torch.ops.aten.unsqueeze.default(unsqueeze_1447, 3);  unsqueeze_1447 = None
        mul_1473 = torch.ops.aten.mul.Tensor(sum_180, 2.5510204081632654e-05)
        mul_1474 = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
        mul_1475 = torch.ops.aten.mul.Tensor(mul_1473, mul_1474);  mul_1473 = mul_1474 = None
        unsqueeze_1449 = torch.ops.aten.unsqueeze.default(mul_1475, 0);  mul_1475 = None
        unsqueeze_1450 = torch.ops.aten.unsqueeze.default(unsqueeze_1449, 2);  unsqueeze_1449 = None
        unsqueeze_1451 = torch.ops.aten.unsqueeze.default(unsqueeze_1450, 3);  unsqueeze_1450 = None
        mul_1476 = torch.ops.aten.mul.Tensor(squeeze_22, primals_23);  primals_23 = None
        unsqueeze_1452 = torch.ops.aten.unsqueeze.default(mul_1476, 0);  mul_1476 = None
        unsqueeze_1453 = torch.ops.aten.unsqueeze.default(unsqueeze_1452, 2);  unsqueeze_1452 = None
        unsqueeze_1454 = torch.ops.aten.unsqueeze.default(unsqueeze_1453, 3);  unsqueeze_1453 = None
        mul_1477 = torch.ops.aten.mul.Tensor(sub_448, unsqueeze_1451);  sub_448 = unsqueeze_1451 = None
        sub_450 = torch.ops.aten.sub.Tensor(where_88, mul_1477);  where_88 = mul_1477 = None
        sub_451 = torch.ops.aten.sub.Tensor(sub_450, unsqueeze_1448);  sub_450 = unsqueeze_1448 = None
        mul_1478 = torch.ops.aten.mul.Tensor(sub_451, unsqueeze_1454);  sub_451 = unsqueeze_1454 = None
        mul_1479 = torch.ops.aten.mul.Tensor(sum_180, squeeze_22);  sum_180 = squeeze_22 = None
        convolution_backward_88 = torch.ops.aten.convolution_backward.default(mul_1478, relu_6, primals_22, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1478 = primals_22 = None
        getitem_464 = convolution_backward_88[0]
        getitem_465 = convolution_backward_88[1];  convolution_backward_88 = None
        alias_550 = torch.ops.aten.alias.default(relu_6);  relu_6 = None
        alias_551 = torch.ops.aten.alias.default(alias_550);  alias_550 = None
        le_89 = torch.ops.aten.le.Scalar(alias_551, 0);  alias_551 = None
        where_89 = torch.ops.aten.where.self(le_89, full_default, getitem_464);  le_89 = getitem_464 = None
        sum_181 = torch.ops.aten.sum.dim_IntList(where_89, [0, 2, 3])
        sub_452 = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_1457);  convolution_6 = unsqueeze_1457 = None
        mul_1480 = torch.ops.aten.mul.Tensor(where_89, sub_452)
        sum_182 = torch.ops.aten.sum.dim_IntList(mul_1480, [0, 2, 3]);  mul_1480 = None
        mul_1481 = torch.ops.aten.mul.Tensor(sum_181, 2.5510204081632654e-05)
        unsqueeze_1458 = torch.ops.aten.unsqueeze.default(mul_1481, 0);  mul_1481 = None
        unsqueeze_1459 = torch.ops.aten.unsqueeze.default(unsqueeze_1458, 2);  unsqueeze_1458 = None
        unsqueeze_1460 = torch.ops.aten.unsqueeze.default(unsqueeze_1459, 3);  unsqueeze_1459 = None
        mul_1482 = torch.ops.aten.mul.Tensor(sum_182, 2.5510204081632654e-05)
        mul_1483 = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
        mul_1484 = torch.ops.aten.mul.Tensor(mul_1482, mul_1483);  mul_1482 = mul_1483 = None
        unsqueeze_1461 = torch.ops.aten.unsqueeze.default(mul_1484, 0);  mul_1484 = None
        unsqueeze_1462 = torch.ops.aten.unsqueeze.default(unsqueeze_1461, 2);  unsqueeze_1461 = None
        unsqueeze_1463 = torch.ops.aten.unsqueeze.default(unsqueeze_1462, 3);  unsqueeze_1462 = None
        mul_1485 = torch.ops.aten.mul.Tensor(squeeze_19, primals_20);  primals_20 = None
        unsqueeze_1464 = torch.ops.aten.unsqueeze.default(mul_1485, 0);  mul_1485 = None
        unsqueeze_1465 = torch.ops.aten.unsqueeze.default(unsqueeze_1464, 2);  unsqueeze_1464 = None
        unsqueeze_1466 = torch.ops.aten.unsqueeze.default(unsqueeze_1465, 3);  unsqueeze_1465 = None
        mul_1486 = torch.ops.aten.mul.Tensor(sub_452, unsqueeze_1463);  sub_452 = unsqueeze_1463 = None
        sub_454 = torch.ops.aten.sub.Tensor(where_89, mul_1486);  where_89 = mul_1486 = None
        sub_455 = torch.ops.aten.sub.Tensor(sub_454, unsqueeze_1460);  sub_454 = unsqueeze_1460 = None
        mul_1487 = torch.ops.aten.mul.Tensor(sub_455, unsqueeze_1466);  sub_455 = unsqueeze_1466 = None
        mul_1488 = torch.ops.aten.mul.Tensor(sum_182, squeeze_19);  sum_182 = squeeze_19 = None
        convolution_backward_89 = torch.ops.aten.convolution_backward.default(mul_1487, getitem_12, primals_19, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1487 = primals_19 = None
        getitem_467 = convolution_backward_89[0]
        getitem_468 = convolution_backward_89[1];  convolution_backward_89 = None
        add_517 = torch.ops.aten.add.Tensor(add_516, getitem_467);  add_516 = getitem_467 = None
        where_90 = torch.ops.aten.where.self(le_90, full_default, slice_50);  le_90 = slice_50 = None
        sum_183 = torch.ops.aten.sum.dim_IntList(where_90, [0, 2, 3])
        sub_456 = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_1469);  convolution_5 = unsqueeze_1469 = None
        mul_1489 = torch.ops.aten.mul.Tensor(where_90, sub_456)
        sum_184 = torch.ops.aten.sum.dim_IntList(mul_1489, [0, 2, 3]);  mul_1489 = None
        mul_1490 = torch.ops.aten.mul.Tensor(sum_183, 2.5510204081632654e-05)
        unsqueeze_1470 = torch.ops.aten.unsqueeze.default(mul_1490, 0);  mul_1490 = None
        unsqueeze_1471 = torch.ops.aten.unsqueeze.default(unsqueeze_1470, 2);  unsqueeze_1470 = None
        unsqueeze_1472 = torch.ops.aten.unsqueeze.default(unsqueeze_1471, 3);  unsqueeze_1471 = None
        mul_1491 = torch.ops.aten.mul.Tensor(sum_184, 2.5510204081632654e-05)
        mul_1492 = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
        mul_1493 = torch.ops.aten.mul.Tensor(mul_1491, mul_1492);  mul_1491 = mul_1492 = None
        unsqueeze_1473 = torch.ops.aten.unsqueeze.default(mul_1493, 0);  mul_1493 = None
        unsqueeze_1474 = torch.ops.aten.unsqueeze.default(unsqueeze_1473, 2);  unsqueeze_1473 = None
        unsqueeze_1475 = torch.ops.aten.unsqueeze.default(unsqueeze_1474, 3);  unsqueeze_1474 = None
        mul_1494 = torch.ops.aten.mul.Tensor(squeeze_16, primals_17);  primals_17 = None
        unsqueeze_1476 = torch.ops.aten.unsqueeze.default(mul_1494, 0);  mul_1494 = None
        unsqueeze_1477 = torch.ops.aten.unsqueeze.default(unsqueeze_1476, 2);  unsqueeze_1476 = None
        unsqueeze_1478 = torch.ops.aten.unsqueeze.default(unsqueeze_1477, 3);  unsqueeze_1477 = None
        mul_1495 = torch.ops.aten.mul.Tensor(sub_456, unsqueeze_1475);  sub_456 = unsqueeze_1475 = None
        sub_458 = torch.ops.aten.sub.Tensor(where_90, mul_1495);  where_90 = mul_1495 = None
        sub_459 = torch.ops.aten.sub.Tensor(sub_458, unsqueeze_1472);  sub_458 = unsqueeze_1472 = None
        mul_1496 = torch.ops.aten.mul.Tensor(sub_459, unsqueeze_1478);  sub_459 = unsqueeze_1478 = None
        mul_1497 = torch.ops.aten.mul.Tensor(sum_184, squeeze_16);  sum_184 = squeeze_16 = None
        convolution_backward_90 = torch.ops.aten.convolution_backward.default(mul_1496, getitem_12, primals_16, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1496 = getitem_12 = primals_16 = None
        getitem_470 = convolution_backward_90[0]
        getitem_471 = convolution_backward_90[1];  convolution_backward_90 = None
        add_518 = torch.ops.aten.add.Tensor(add_517, getitem_470);  add_517 = getitem_470 = None
        max_pool2d_with_indices_backward_2 = torch.ops.aten.max_pool2d_with_indices_backward.default(add_518, relu_4, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_13);  add_518 = getitem_13 = None
        alias_558 = torch.ops.aten.alias.default(relu_4);  relu_4 = None
        alias_559 = torch.ops.aten.alias.default(alias_558);  alias_558 = None
        le_91 = torch.ops.aten.le.Scalar(alias_559, 0);  alias_559 = None
        where_91 = torch.ops.aten.where.self(le_91, full_default, max_pool2d_with_indices_backward_2);  le_91 = max_pool2d_with_indices_backward_2 = None
        sum_185 = torch.ops.aten.sum.dim_IntList(where_91, [0, 2, 3])
        sub_460 = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_1481);  convolution_4 = unsqueeze_1481 = None
        mul_1498 = torch.ops.aten.mul.Tensor(where_91, sub_460)
        sum_186 = torch.ops.aten.sum.dim_IntList(mul_1498, [0, 2, 3]);  mul_1498 = None
        mul_1499 = torch.ops.aten.mul.Tensor(sum_185, 6.199166831977782e-06)
        unsqueeze_1482 = torch.ops.aten.unsqueeze.default(mul_1499, 0);  mul_1499 = None
        unsqueeze_1483 = torch.ops.aten.unsqueeze.default(unsqueeze_1482, 2);  unsqueeze_1482 = None
        unsqueeze_1484 = torch.ops.aten.unsqueeze.default(unsqueeze_1483, 3);  unsqueeze_1483 = None
        mul_1500 = torch.ops.aten.mul.Tensor(sum_186, 6.199166831977782e-06)
        mul_1501 = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
        mul_1502 = torch.ops.aten.mul.Tensor(mul_1500, mul_1501);  mul_1500 = mul_1501 = None
        unsqueeze_1485 = torch.ops.aten.unsqueeze.default(mul_1502, 0);  mul_1502 = None
        unsqueeze_1486 = torch.ops.aten.unsqueeze.default(unsqueeze_1485, 2);  unsqueeze_1485 = None
        unsqueeze_1487 = torch.ops.aten.unsqueeze.default(unsqueeze_1486, 3);  unsqueeze_1486 = None
        mul_1503 = torch.ops.aten.mul.Tensor(squeeze_13, primals_14);  primals_14 = None
        unsqueeze_1488 = torch.ops.aten.unsqueeze.default(mul_1503, 0);  mul_1503 = None
        unsqueeze_1489 = torch.ops.aten.unsqueeze.default(unsqueeze_1488, 2);  unsqueeze_1488 = None
        unsqueeze_1490 = torch.ops.aten.unsqueeze.default(unsqueeze_1489, 3);  unsqueeze_1489 = None
        mul_1504 = torch.ops.aten.mul.Tensor(sub_460, unsqueeze_1487);  sub_460 = unsqueeze_1487 = None
        sub_462 = torch.ops.aten.sub.Tensor(where_91, mul_1504);  where_91 = mul_1504 = None
        sub_463 = torch.ops.aten.sub.Tensor(sub_462, unsqueeze_1484);  sub_462 = unsqueeze_1484 = None
        mul_1505 = torch.ops.aten.mul.Tensor(sub_463, unsqueeze_1490);  sub_463 = unsqueeze_1490 = None
        mul_1506 = torch.ops.aten.mul.Tensor(sum_186, squeeze_13);  sum_186 = squeeze_13 = None
        convolution_backward_91 = torch.ops.aten.convolution_backward.default(mul_1505, relu_3, primals_13, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1505 = primals_13 = None
        getitem_473 = convolution_backward_91[0]
        getitem_474 = convolution_backward_91[1];  convolution_backward_91 = None
        alias_562 = torch.ops.aten.alias.default(relu_3);  relu_3 = None
        alias_563 = torch.ops.aten.alias.default(alias_562);  alias_562 = None
        le_92 = torch.ops.aten.le.Scalar(alias_563, 0);  alias_563 = None
        where_92 = torch.ops.aten.where.self(le_92, full_default, getitem_473);  le_92 = getitem_473 = None
        sum_187 = torch.ops.aten.sum.dim_IntList(where_92, [0, 2, 3])
        sub_464 = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_1493);  convolution_3 = unsqueeze_1493 = None
        mul_1507 = torch.ops.aten.mul.Tensor(where_92, sub_464)
        sum_188 = torch.ops.aten.sum.dim_IntList(mul_1507, [0, 2, 3]);  mul_1507 = None
        mul_1508 = torch.ops.aten.mul.Tensor(sum_187, 5.864139613435917e-06)
        unsqueeze_1494 = torch.ops.aten.unsqueeze.default(mul_1508, 0);  mul_1508 = None
        unsqueeze_1495 = torch.ops.aten.unsqueeze.default(unsqueeze_1494, 2);  unsqueeze_1494 = None
        unsqueeze_1496 = torch.ops.aten.unsqueeze.default(unsqueeze_1495, 3);  unsqueeze_1495 = None
        mul_1509 = torch.ops.aten.mul.Tensor(sum_188, 5.864139613435917e-06)
        mul_1510 = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
        mul_1511 = torch.ops.aten.mul.Tensor(mul_1509, mul_1510);  mul_1509 = mul_1510 = None
        unsqueeze_1497 = torch.ops.aten.unsqueeze.default(mul_1511, 0);  mul_1511 = None
        unsqueeze_1498 = torch.ops.aten.unsqueeze.default(unsqueeze_1497, 2);  unsqueeze_1497 = None
        unsqueeze_1499 = torch.ops.aten.unsqueeze.default(unsqueeze_1498, 3);  unsqueeze_1498 = None
        mul_1512 = torch.ops.aten.mul.Tensor(squeeze_10, primals_11);  primals_11 = None
        unsqueeze_1500 = torch.ops.aten.unsqueeze.default(mul_1512, 0);  mul_1512 = None
        unsqueeze_1501 = torch.ops.aten.unsqueeze.default(unsqueeze_1500, 2);  unsqueeze_1500 = None
        unsqueeze_1502 = torch.ops.aten.unsqueeze.default(unsqueeze_1501, 3);  unsqueeze_1501 = None
        mul_1513 = torch.ops.aten.mul.Tensor(sub_464, unsqueeze_1499);  sub_464 = unsqueeze_1499 = None
        sub_466 = torch.ops.aten.sub.Tensor(where_92, mul_1513);  where_92 = mul_1513 = None
        sub_467 = torch.ops.aten.sub.Tensor(sub_466, unsqueeze_1496);  sub_466 = unsqueeze_1496 = None
        mul_1514 = torch.ops.aten.mul.Tensor(sub_467, unsqueeze_1502);  sub_467 = unsqueeze_1502 = None
        mul_1515 = torch.ops.aten.mul.Tensor(sum_188, squeeze_10);  sum_188 = squeeze_10 = None
        convolution_backward_92 = torch.ops.aten.convolution_backward.default(mul_1514, getitem_6, primals_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1514 = getitem_6 = primals_10 = None
        getitem_476 = convolution_backward_92[0]
        getitem_477 = convolution_backward_92[1];  convolution_backward_92 = None
        max_pool2d_with_indices_backward_3 = torch.ops.aten.max_pool2d_with_indices_backward.default(getitem_476, relu_2, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_7);  getitem_476 = getitem_7 = None
        alias_566 = torch.ops.aten.alias.default(relu_2);  relu_2 = None
        alias_567 = torch.ops.aten.alias.default(alias_566);  alias_566 = None
        le_93 = torch.ops.aten.le.Scalar(alias_567, 0);  alias_567 = None
        where_93 = torch.ops.aten.where.self(le_93, full_default, max_pool2d_with_indices_backward_3);  le_93 = max_pool2d_with_indices_backward_3 = None
        sum_189 = torch.ops.aten.sum.dim_IntList(where_93, [0, 2, 3])
        sub_468 = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_1505);  convolution_2 = unsqueeze_1505 = None
        mul_1516 = torch.ops.aten.mul.Tensor(where_93, sub_468)
        sum_190 = torch.ops.aten.sum.dim_IntList(mul_1516, [0, 2, 3]);  mul_1516 = None
        mul_1517 = torch.ops.aten.mul.Tensor(sum_189, 1.446156693970105e-06)
        unsqueeze_1506 = torch.ops.aten.unsqueeze.default(mul_1517, 0);  mul_1517 = None
        unsqueeze_1507 = torch.ops.aten.unsqueeze.default(unsqueeze_1506, 2);  unsqueeze_1506 = None
        unsqueeze_1508 = torch.ops.aten.unsqueeze.default(unsqueeze_1507, 3);  unsqueeze_1507 = None
        mul_1518 = torch.ops.aten.mul.Tensor(sum_190, 1.446156693970105e-06)
        mul_1519 = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
        mul_1520 = torch.ops.aten.mul.Tensor(mul_1518, mul_1519);  mul_1518 = mul_1519 = None
        unsqueeze_1509 = torch.ops.aten.unsqueeze.default(mul_1520, 0);  mul_1520 = None
        unsqueeze_1510 = torch.ops.aten.unsqueeze.default(unsqueeze_1509, 2);  unsqueeze_1509 = None
        unsqueeze_1511 = torch.ops.aten.unsqueeze.default(unsqueeze_1510, 3);  unsqueeze_1510 = None
        mul_1521 = torch.ops.aten.mul.Tensor(squeeze_7, primals_8);  primals_8 = None
        unsqueeze_1512 = torch.ops.aten.unsqueeze.default(mul_1521, 0);  mul_1521 = None
        unsqueeze_1513 = torch.ops.aten.unsqueeze.default(unsqueeze_1512, 2);  unsqueeze_1512 = None
        unsqueeze_1514 = torch.ops.aten.unsqueeze.default(unsqueeze_1513, 3);  unsqueeze_1513 = None
        mul_1522 = torch.ops.aten.mul.Tensor(sub_468, unsqueeze_1511);  sub_468 = unsqueeze_1511 = None
        sub_470 = torch.ops.aten.sub.Tensor(where_93, mul_1522);  where_93 = mul_1522 = None
        sub_471 = torch.ops.aten.sub.Tensor(sub_470, unsqueeze_1508);  sub_470 = unsqueeze_1508 = None
        mul_1523 = torch.ops.aten.mul.Tensor(sub_471, unsqueeze_1514);  sub_471 = unsqueeze_1514 = None
        mul_1524 = torch.ops.aten.mul.Tensor(sum_190, squeeze_7);  sum_190 = squeeze_7 = None
        convolution_backward_93 = torch.ops.aten.convolution_backward.default(mul_1523, relu_1, primals_7, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1523 = primals_7 = None
        getitem_479 = convolution_backward_93[0]
        getitem_480 = convolution_backward_93[1];  convolution_backward_93 = None
        alias_570 = torch.ops.aten.alias.default(relu_1);  relu_1 = None
        alias_571 = torch.ops.aten.alias.default(alias_570);  alias_570 = None
        le_94 = torch.ops.aten.le.Scalar(alias_571, 0);  alias_571 = None
        where_94 = torch.ops.aten.where.self(le_94, full_default, getitem_479);  le_94 = getitem_479 = None
        sum_191 = torch.ops.aten.sum.dim_IntList(where_94, [0, 2, 3])
        sub_472 = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_1517);  convolution_1 = unsqueeze_1517 = None
        mul_1525 = torch.ops.aten.mul.Tensor(where_94, sub_472)
        sum_192 = torch.ops.aten.sum.dim_IntList(mul_1525, [0, 2, 3]);  mul_1525 = None
        mul_1526 = torch.ops.aten.mul.Tensor(sum_191, 1.446156693970105e-06)
        unsqueeze_1518 = torch.ops.aten.unsqueeze.default(mul_1526, 0);  mul_1526 = None
        unsqueeze_1519 = torch.ops.aten.unsqueeze.default(unsqueeze_1518, 2);  unsqueeze_1518 = None
        unsqueeze_1520 = torch.ops.aten.unsqueeze.default(unsqueeze_1519, 3);  unsqueeze_1519 = None
        mul_1527 = torch.ops.aten.mul.Tensor(sum_192, 1.446156693970105e-06)
        mul_1528 = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
        mul_1529 = torch.ops.aten.mul.Tensor(mul_1527, mul_1528);  mul_1527 = mul_1528 = None
        unsqueeze_1521 = torch.ops.aten.unsqueeze.default(mul_1529, 0);  mul_1529 = None
        unsqueeze_1522 = torch.ops.aten.unsqueeze.default(unsqueeze_1521, 2);  unsqueeze_1521 = None
        unsqueeze_1523 = torch.ops.aten.unsqueeze.default(unsqueeze_1522, 3);  unsqueeze_1522 = None
        mul_1530 = torch.ops.aten.mul.Tensor(squeeze_4, primals_5);  primals_5 = None
        unsqueeze_1524 = torch.ops.aten.unsqueeze.default(mul_1530, 0);  mul_1530 = None
        unsqueeze_1525 = torch.ops.aten.unsqueeze.default(unsqueeze_1524, 2);  unsqueeze_1524 = None
        unsqueeze_1526 = torch.ops.aten.unsqueeze.default(unsqueeze_1525, 3);  unsqueeze_1525 = None
        mul_1531 = torch.ops.aten.mul.Tensor(sub_472, unsqueeze_1523);  sub_472 = unsqueeze_1523 = None
        sub_474 = torch.ops.aten.sub.Tensor(where_94, mul_1531);  where_94 = mul_1531 = None
        sub_475 = torch.ops.aten.sub.Tensor(sub_474, unsqueeze_1520);  sub_474 = unsqueeze_1520 = None
        mul_1532 = torch.ops.aten.mul.Tensor(sub_475, unsqueeze_1526);  sub_475 = unsqueeze_1526 = None
        mul_1533 = torch.ops.aten.mul.Tensor(sum_192, squeeze_4);  sum_192 = squeeze_4 = None
        convolution_backward_94 = torch.ops.aten.convolution_backward.default(mul_1532, relu, primals_4, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1532 = primals_4 = None
        getitem_482 = convolution_backward_94[0]
        getitem_483 = convolution_backward_94[1];  convolution_backward_94 = None
        alias_574 = torch.ops.aten.alias.default(relu);  relu = None
        alias_575 = torch.ops.aten.alias.default(alias_574);  alias_574 = None
        le_95 = torch.ops.aten.le.Scalar(alias_575, 0);  alias_575 = None
        where_95 = torch.ops.aten.where.self(le_95, full_default, getitem_482);  le_95 = full_default = getitem_482 = None
        sum_193 = torch.ops.aten.sum.dim_IntList(where_95, [0, 2, 3])
        sub_476 = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1529);  convolution = unsqueeze_1529 = None
        mul_1534 = torch.ops.aten.mul.Tensor(where_95, sub_476)
        sum_194 = torch.ops.aten.sum.dim_IntList(mul_1534, [0, 2, 3]);  mul_1534 = None
        mul_1535 = torch.ops.aten.mul.Tensor(sum_193, 1.4075942525111482e-06)
        unsqueeze_1530 = torch.ops.aten.unsqueeze.default(mul_1535, 0);  mul_1535 = None
        unsqueeze_1531 = torch.ops.aten.unsqueeze.default(unsqueeze_1530, 2);  unsqueeze_1530 = None
        unsqueeze_1532 = torch.ops.aten.unsqueeze.default(unsqueeze_1531, 3);  unsqueeze_1531 = None
        mul_1536 = torch.ops.aten.mul.Tensor(sum_194, 1.4075942525111482e-06)
        mul_1537 = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_1538 = torch.ops.aten.mul.Tensor(mul_1536, mul_1537);  mul_1536 = mul_1537 = None
        unsqueeze_1533 = torch.ops.aten.unsqueeze.default(mul_1538, 0);  mul_1538 = None
        unsqueeze_1534 = torch.ops.aten.unsqueeze.default(unsqueeze_1533, 2);  unsqueeze_1533 = None
        unsqueeze_1535 = torch.ops.aten.unsqueeze.default(unsqueeze_1534, 3);  unsqueeze_1534 = None
        mul_1539 = torch.ops.aten.mul.Tensor(squeeze_1, primals_2);  primals_2 = None
        unsqueeze_1536 = torch.ops.aten.unsqueeze.default(mul_1539, 0);  mul_1539 = None
        unsqueeze_1537 = torch.ops.aten.unsqueeze.default(unsqueeze_1536, 2);  unsqueeze_1536 = None
        unsqueeze_1538 = torch.ops.aten.unsqueeze.default(unsqueeze_1537, 3);  unsqueeze_1537 = None
        mul_1540 = torch.ops.aten.mul.Tensor(sub_476, unsqueeze_1535);  sub_476 = unsqueeze_1535 = None
        sub_478 = torch.ops.aten.sub.Tensor(where_95, mul_1540);  where_95 = mul_1540 = None
        sub_479 = torch.ops.aten.sub.Tensor(sub_478, unsqueeze_1532);  sub_478 = unsqueeze_1532 = None
        mul_1541 = torch.ops.aten.mul.Tensor(sub_479, unsqueeze_1538);  sub_479 = unsqueeze_1538 = None
        mul_1542 = torch.ops.aten.mul.Tensor(sum_194, squeeze_1);  sum_194 = squeeze_1 = None
        convolution_backward_95 = torch.ops.aten.convolution_backward.default(mul_1541, cat, primals_1, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_1541 = cat = primals_1 = None
        getitem_486 = convolution_backward_95[1];  convolution_backward_95 = None
        return [getitem_486, mul_1542, sum_193, getitem_483, mul_1533, sum_191, getitem_480, mul_1524, sum_189, getitem_477, mul_1515, sum_187, getitem_474, mul_1506, sum_185, getitem_471, mul_1497, sum_183, getitem_468, mul_1488, sum_181, getitem_465, mul_1479, sum_179, getitem_462, mul_1470, sum_177, getitem_459, mul_1461, sum_175, getitem_456, mul_1452, sum_173, getitem_453, mul_1443, sum_171, getitem_450, mul_1434, sum_169, getitem_447, mul_1425, sum_167, getitem_444, mul_1416, sum_165, getitem_441, mul_1407, sum_163, getitem_438, mul_1398, sum_161, getitem_435, mul_1389, sum_159, getitem_432, mul_1380, sum_157, getitem_429, mul_1371, sum_155, getitem_426, mul_1362, sum_153, getitem_423, mul_1353, sum_151, getitem_420, mul_1344, sum_149, getitem_417, mul_1335, sum_147, getitem_414, mul_1326, sum_145, getitem_411, mul_1317, sum_143, getitem_408, mul_1308, sum_141, getitem_405, mul_1299, sum_139, getitem_402, mul_1290, sum_137, getitem_399, mul_1281, sum_135, getitem_396, mul_1272, sum_133, getitem_393, mul_1263, sum_131, getitem_390, mul_1254, sum_129, getitem_387, mul_1245, sum_127, getitem_384, mul_1236, sum_125, getitem_381, mul_1227, sum_123, getitem_378, mul_1218, sum_121, getitem_375, mul_1209, sum_119, getitem_372, mul_1200, sum_117, getitem_369, mul_1191, sum_115, getitem_366, mul_1182, sum_113, getitem_363, mul_1173, sum_111, getitem_360, mul_1164, sum_109, getitem_357, mul_1155, sum_107, getitem_354, mul_1146, sum_105, getitem_351, mul_1137, sum_103, getitem_348, mul_1128, sum_101, getitem_345, mul_1119, sum_99, getitem_342, mul_1110, sum_97, getitem_339, mul_1101, sum_95, getitem_336, mul_1092, sum_93, getitem_333, mul_1083, sum_91, getitem_330, mul_1074, sum_89, getitem_327, mul_1065, sum_87, getitem_324, mul_1056, sum_85, getitem_321, mul_1047, sum_83, getitem_318, mul_1038, sum_81, getitem_315, mul_1029, sum_79, getitem_312, mul_1020, sum_77, getitem_309, mul_1011, sum_75, getitem_306, mul_1002, sum_73, getitem_303, mul_993, sum_71, getitem_300, mul_984, sum_69, getitem_297, mul_975, sum_67, getitem_294, mul_966, sum_65, getitem_291, mul_957, sum_63, getitem_288, mul_948, sum_61, getitem_285, mul_939, sum_59, getitem_282, mul_930, sum_57, getitem_279, mul_921, sum_55, getitem_276, mul_912, sum_53, getitem_273, mul_903, sum_51, permute_9, view_4, getitem_270, mul_894, sum_48, getitem_267, mul_885, sum_46, getitem_264, mul_876, sum_44, getitem_261, mul_867, sum_42, getitem_258, mul_858, sum_40, getitem_255, mul_849, sum_38, getitem_252, mul_840, sum_36, getitem_249, mul_831, sum_34, getitem_246, mul_822, sum_32, getitem_243, mul_813, sum_30, getitem_240, mul_804, sum_28, getitem_237, mul_795, sum_26, getitem_234, mul_786, sum_24, getitem_231, mul_777, sum_22, getitem_228, mul_768, sum_20, getitem_225, mul_759, sum_18, getitem_222, mul_750, sum_16, getitem_219, mul_741, sum_14, getitem_216, mul_732, sum_12, getitem_213, mul_723, sum_10, getitem_210, mul_714, sum_8, getitem_207, mul_705, sum_6, getitem_204, mul_696, sum_4, getitem_201, mul_687, sum_2, permute_5, view_2, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
        
def load_args(reader):
    buf0 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf0, (32, 3, 3, 3), (27, 1, 9, 3), is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf1, (32,), is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 36864, device=device(type='cuda', index=0))
    reader.tensor(buf2, (32, 32, 3, 3), (288, 1, 96, 32), is_leaf=True)  # primals_4
    buf3 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf3, (32,), is_leaf=True)  # primals_5
    buf4 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf4, (64, 32, 3, 3), (288, 1, 96, 32), is_leaf=True)  # primals_7
    buf5 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf5, (64,), is_leaf=True)  # primals_8
    buf6 = reader.storage(None, 20480, device=device(type='cuda', index=0))
    reader.tensor(buf6, (80, 64, 1, 1), is_leaf=True)  # primals_10
    buf7 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf7, (80,), is_leaf=True)  # primals_11
    buf8 = reader.storage(None, 552960, device=device(type='cuda', index=0))
    reader.tensor(buf8, (192, 80, 3, 3), (720, 1, 240, 80), is_leaf=True)  # primals_13
    buf9 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf9, (192,), is_leaf=True)  # primals_14
    buf10 = reader.storage(None, 49152, device=device(type='cuda', index=0))
    reader.tensor(buf10, (64, 192, 1, 1), is_leaf=True)  # primals_16
    buf11 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf11, (64,), is_leaf=True)  # primals_17
    buf12 = reader.storage(None, 36864, device=device(type='cuda', index=0))
    reader.tensor(buf12, (48, 192, 1, 1), is_leaf=True)  # primals_19
    buf13 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf13, (48,), is_leaf=True)  # primals_20
    buf14 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf14, (64, 48, 5, 5), (1200, 1, 240, 48), is_leaf=True)  # primals_22
    buf15 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf15, (64,), is_leaf=True)  # primals_23
    buf16 = reader.storage(None, 49152, device=device(type='cuda', index=0))
    reader.tensor(buf16, (64, 192, 1, 1), is_leaf=True)  # primals_25
    buf17 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf17, (64,), is_leaf=True)  # primals_26
    buf18 = reader.storage(None, 221184, device=device(type='cuda', index=0))
    reader.tensor(buf18, (96, 64, 3, 3), (576, 1, 192, 64), is_leaf=True)  # primals_28
    buf19 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf19, (96,), is_leaf=True)  # primals_29
    buf20 = reader.storage(None, 331776, device=device(type='cuda', index=0))
    reader.tensor(buf20, (96, 96, 3, 3), (864, 1, 288, 96), is_leaf=True)  # primals_31
    buf21 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf21, (96,), is_leaf=True)  # primals_32
    buf22 = reader.storage(None, 24576, device=device(type='cuda', index=0))
    reader.tensor(buf22, (32, 192, 1, 1), is_leaf=True)  # primals_34
    buf23 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf23, (32,), is_leaf=True)  # primals_35
    buf24 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf24, (64, 256, 1, 1), is_leaf=True)  # primals_37
    buf25 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf25, (64,), is_leaf=True)  # primals_38
    buf26 = reader.storage(None, 49152, device=device(type='cuda', index=0))
    reader.tensor(buf26, (48, 256, 1, 1), is_leaf=True)  # primals_40
    buf27 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf27, (48,), is_leaf=True)  # primals_41
    buf28 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf28, (64, 48, 5, 5), (1200, 1, 240, 48), is_leaf=True)  # primals_43
    buf29 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf29, (64,), is_leaf=True)  # primals_44
    buf30 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf30, (64, 256, 1, 1), is_leaf=True)  # primals_46
    buf31 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf31, (64,), is_leaf=True)  # primals_47
    buf32 = reader.storage(None, 221184, device=device(type='cuda', index=0))
    reader.tensor(buf32, (96, 64, 3, 3), (576, 1, 192, 64), is_leaf=True)  # primals_49
    buf33 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf33, (96,), is_leaf=True)  # primals_50
    buf34 = reader.storage(None, 331776, device=device(type='cuda', index=0))
    reader.tensor(buf34, (96, 96, 3, 3), (864, 1, 288, 96), is_leaf=True)  # primals_52
    buf35 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf35, (96,), is_leaf=True)  # primals_53
    buf36 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf36, (64, 256, 1, 1), is_leaf=True)  # primals_55
    buf37 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf37, (64,), is_leaf=True)  # primals_56
    buf38 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf38, (64, 288, 1, 1), is_leaf=True)  # primals_58
    buf39 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf39, (64,), is_leaf=True)  # primals_59
    buf40 = reader.storage(None, 55296, device=device(type='cuda', index=0))
    reader.tensor(buf40, (48, 288, 1, 1), is_leaf=True)  # primals_61
    buf41 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf41, (48,), is_leaf=True)  # primals_62
    buf42 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf42, (64, 48, 5, 5), (1200, 1, 240, 48), is_leaf=True)  # primals_64
    buf43 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf43, (64,), is_leaf=True)  # primals_65
    buf44 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf44, (64, 288, 1, 1), is_leaf=True)  # primals_67
    buf45 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf45, (64,), is_leaf=True)  # primals_68
    buf46 = reader.storage(None, 221184, device=device(type='cuda', index=0))
    reader.tensor(buf46, (96, 64, 3, 3), (576, 1, 192, 64), is_leaf=True)  # primals_70
    buf47 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf47, (96,), is_leaf=True)  # primals_71
    buf48 = reader.storage(None, 331776, device=device(type='cuda', index=0))
    reader.tensor(buf48, (96, 96, 3, 3), (864, 1, 288, 96), is_leaf=True)  # primals_73
    buf49 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf49, (96,), is_leaf=True)  # primals_74
    buf50 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf50, (64, 288, 1, 1), is_leaf=True)  # primals_76
    buf51 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf51, (64,), is_leaf=True)  # primals_77
    buf52 = reader.storage(None, 3981312, device=device(type='cuda', index=0))
    reader.tensor(buf52, (384, 288, 3, 3), (2592, 1, 864, 288), is_leaf=True)  # primals_79
    buf53 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf53, (384,), is_leaf=True)  # primals_80
    buf54 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf54, (64, 288, 1, 1), is_leaf=True)  # primals_82
    buf55 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf55, (64,), is_leaf=True)  # primals_83
    buf56 = reader.storage(None, 221184, device=device(type='cuda', index=0))
    reader.tensor(buf56, (96, 64, 3, 3), (576, 1, 192, 64), is_leaf=True)  # primals_85
    buf57 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf57, (96,), is_leaf=True)  # primals_86
    buf58 = reader.storage(None, 331776, device=device(type='cuda', index=0))
    reader.tensor(buf58, (96, 96, 3, 3), (864, 1, 288, 96), is_leaf=True)  # primals_88
    buf59 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf59, (96,), is_leaf=True)  # primals_89
    buf60 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf60, (192, 768, 1, 1), is_leaf=True)  # primals_91
    buf61 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf61, (192,), is_leaf=True)  # primals_92
    buf62 = reader.storage(None, 393216, device=device(type='cuda', index=0))
    reader.tensor(buf62, (128, 768, 1, 1), is_leaf=True)  # primals_94
    buf63 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf63, (128,), is_leaf=True)  # primals_95
    buf64 = reader.storage(None, 458752, device=device(type='cuda', index=0))
    reader.tensor(buf64, (128, 128, 1, 7), (896, 1, 896, 128), is_leaf=True)  # primals_97
    buf65 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf65, (128,), is_leaf=True)  # primals_98
    buf66 = reader.storage(None, 688128, device=device(type='cuda', index=0))
    reader.tensor(buf66, (192, 128, 7, 1), (896, 1, 128, 128), is_leaf=True)  # primals_100
    buf67 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf67, (192,), is_leaf=True)  # primals_101
    buf68 = reader.storage(None, 393216, device=device(type='cuda', index=0))
    reader.tensor(buf68, (128, 768, 1, 1), is_leaf=True)  # primals_103
    buf69 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf69, (128,), is_leaf=True)  # primals_104
    buf70 = reader.storage(None, 458752, device=device(type='cuda', index=0))
    reader.tensor(buf70, (128, 128, 7, 1), (896, 1, 128, 128), is_leaf=True)  # primals_106
    buf71 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf71, (128,), is_leaf=True)  # primals_107
    buf72 = reader.storage(None, 458752, device=device(type='cuda', index=0))
    reader.tensor(buf72, (128, 128, 1, 7), (896, 1, 896, 128), is_leaf=True)  # primals_109
    buf73 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf73, (128,), is_leaf=True)  # primals_110
    buf74 = reader.storage(None, 458752, device=device(type='cuda', index=0))
    reader.tensor(buf74, (128, 128, 7, 1), (896, 1, 128, 128), is_leaf=True)  # primals_112
    buf75 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf75, (128,), is_leaf=True)  # primals_113
    buf76 = reader.storage(None, 688128, device=device(type='cuda', index=0))
    reader.tensor(buf76, (192, 128, 1, 7), (896, 1, 896, 128), is_leaf=True)  # primals_115
    buf77 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf77, (192,), is_leaf=True)  # primals_116
    buf78 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf78, (192, 768, 1, 1), is_leaf=True)  # primals_118
    buf79 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf79, (192,), is_leaf=True)  # primals_119
    buf80 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf80, (192, 768, 1, 1), is_leaf=True)  # primals_121
    buf81 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf81, (192,), is_leaf=True)  # primals_122
    buf82 = reader.storage(None, 491520, device=device(type='cuda', index=0))
    reader.tensor(buf82, (160, 768, 1, 1), is_leaf=True)  # primals_124
    buf83 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf83, (160,), is_leaf=True)  # primals_125
    buf84 = reader.storage(None, 716800, device=device(type='cuda', index=0))
    reader.tensor(buf84, (160, 160, 1, 7), (1120, 1, 1120, 160), is_leaf=True)  # primals_127
    buf85 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf85, (160,), is_leaf=True)  # primals_128
    buf86 = reader.storage(None, 860160, device=device(type='cuda', index=0))
    reader.tensor(buf86, (192, 160, 7, 1), (1120, 1, 160, 160), is_leaf=True)  # primals_130
    buf87 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf87, (192,), is_leaf=True)  # primals_131
    buf88 = reader.storage(None, 491520, device=device(type='cuda', index=0))
    reader.tensor(buf88, (160, 768, 1, 1), is_leaf=True)  # primals_133
    buf89 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf89, (160,), is_leaf=True)  # primals_134
    buf90 = reader.storage(None, 716800, device=device(type='cuda', index=0))
    reader.tensor(buf90, (160, 160, 7, 1), (1120, 1, 160, 160), is_leaf=True)  # primals_136
    buf91 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf91, (160,), is_leaf=True)  # primals_137
    buf92 = reader.storage(None, 716800, device=device(type='cuda', index=0))
    reader.tensor(buf92, (160, 160, 1, 7), (1120, 1, 1120, 160), is_leaf=True)  # primals_139
    buf93 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf93, (160,), is_leaf=True)  # primals_140
    buf94 = reader.storage(None, 716800, device=device(type='cuda', index=0))
    reader.tensor(buf94, (160, 160, 7, 1), (1120, 1, 160, 160), is_leaf=True)  # primals_142
    buf95 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf95, (160,), is_leaf=True)  # primals_143
    buf96 = reader.storage(None, 860160, device=device(type='cuda', index=0))
    reader.tensor(buf96, (192, 160, 1, 7), (1120, 1, 1120, 160), is_leaf=True)  # primals_145
    buf97 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf97, (192,), is_leaf=True)  # primals_146
    buf98 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf98, (192, 768, 1, 1), is_leaf=True)  # primals_148
    buf99 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf99, (192,), is_leaf=True)  # primals_149
    buf100 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf100, (192, 768, 1, 1), is_leaf=True)  # primals_151
    buf101 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf101, (192,), is_leaf=True)  # primals_152
    buf102 = reader.storage(None, 491520, device=device(type='cuda', index=0))
    reader.tensor(buf102, (160, 768, 1, 1), is_leaf=True)  # primals_154
    buf103 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf103, (160,), is_leaf=True)  # primals_155
    buf104 = reader.storage(None, 716800, device=device(type='cuda', index=0))
    reader.tensor(buf104, (160, 160, 1, 7), (1120, 1, 1120, 160), is_leaf=True)  # primals_157
    buf105 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf105, (160,), is_leaf=True)  # primals_158
    buf106 = reader.storage(None, 860160, device=device(type='cuda', index=0))
    reader.tensor(buf106, (192, 160, 7, 1), (1120, 1, 160, 160), is_leaf=True)  # primals_160
    buf107 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf107, (192,), is_leaf=True)  # primals_161
    buf108 = reader.storage(None, 491520, device=device(type='cuda', index=0))
    reader.tensor(buf108, (160, 768, 1, 1), is_leaf=True)  # primals_163
    buf109 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf109, (160,), is_leaf=True)  # primals_164
    buf110 = reader.storage(None, 716800, device=device(type='cuda', index=0))
    reader.tensor(buf110, (160, 160, 7, 1), (1120, 1, 160, 160), is_leaf=True)  # primals_166
    buf111 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf111, (160,), is_leaf=True)  # primals_167
    buf112 = reader.storage(None, 716800, device=device(type='cuda', index=0))
    reader.tensor(buf112, (160, 160, 1, 7), (1120, 1, 1120, 160), is_leaf=True)  # primals_169
    buf113 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf113, (160,), is_leaf=True)  # primals_170
    buf114 = reader.storage(None, 716800, device=device(type='cuda', index=0))
    reader.tensor(buf114, (160, 160, 7, 1), (1120, 1, 160, 160), is_leaf=True)  # primals_172
    buf115 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf115, (160,), is_leaf=True)  # primals_173
    buf116 = reader.storage(None, 860160, device=device(type='cuda', index=0))
    reader.tensor(buf116, (192, 160, 1, 7), (1120, 1, 1120, 160), is_leaf=True)  # primals_175
    buf117 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf117, (192,), is_leaf=True)  # primals_176
    buf118 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf118, (192, 768, 1, 1), is_leaf=True)  # primals_178
    buf119 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf119, (192,), is_leaf=True)  # primals_179
    buf120 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf120, (192, 768, 1, 1), is_leaf=True)  # primals_181
    buf121 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf121, (192,), is_leaf=True)  # primals_182
    buf122 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf122, (192, 768, 1, 1), is_leaf=True)  # primals_184
    buf123 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf123, (192,), is_leaf=True)  # primals_185
    buf124 = reader.storage(None, 1032192, device=device(type='cuda', index=0))
    reader.tensor(buf124, (192, 192, 1, 7), (1344, 1, 1344, 192), is_leaf=True)  # primals_187
    buf125 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf125, (192,), is_leaf=True)  # primals_188
    buf126 = reader.storage(None, 1032192, device=device(type='cuda', index=0))
    reader.tensor(buf126, (192, 192, 7, 1), (1344, 1, 192, 192), is_leaf=True)  # primals_190
    buf127 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf127, (192,), is_leaf=True)  # primals_191
    buf128 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf128, (192, 768, 1, 1), is_leaf=True)  # primals_193
    buf129 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf129, (192,), is_leaf=True)  # primals_194
    buf130 = reader.storage(None, 1032192, device=device(type='cuda', index=0))
    reader.tensor(buf130, (192, 192, 7, 1), (1344, 1, 192, 192), is_leaf=True)  # primals_196
    buf131 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf131, (192,), is_leaf=True)  # primals_197
    buf132 = reader.storage(None, 1032192, device=device(type='cuda', index=0))
    reader.tensor(buf132, (192, 192, 1, 7), (1344, 1, 1344, 192), is_leaf=True)  # primals_199
    buf133 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf133, (192,), is_leaf=True)  # primals_200
    buf134 = reader.storage(None, 1032192, device=device(type='cuda', index=0))
    reader.tensor(buf134, (192, 192, 7, 1), (1344, 1, 192, 192), is_leaf=True)  # primals_202
    buf135 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf135, (192,), is_leaf=True)  # primals_203
    buf136 = reader.storage(None, 1032192, device=device(type='cuda', index=0))
    reader.tensor(buf136, (192, 192, 1, 7), (1344, 1, 1344, 192), is_leaf=True)  # primals_205
    buf137 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf137, (192,), is_leaf=True)  # primals_206
    buf138 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf138, (192, 768, 1, 1), is_leaf=True)  # primals_208
    buf139 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf139, (192,), is_leaf=True)  # primals_209
    buf140 = reader.storage(None, 393216, device=device(type='cuda', index=0))
    reader.tensor(buf140, (128, 768, 1, 1), is_leaf=True)  # primals_211
    buf141 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf141, (128,), is_leaf=True)  # primals_212
    buf142 = reader.storage(None, 9830400, device=device(type='cuda', index=0))
    reader.tensor(buf142, (768, 128, 5, 5), (3200, 1, 640, 128), is_leaf=True)  # primals_214
    buf143 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf143, (768,), is_leaf=True)  # primals_215
    buf144 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf144, (192, 768, 1, 1), is_leaf=True)  # primals_219
    buf145 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf145, (192,), is_leaf=True)  # primals_220
    buf146 = reader.storage(None, 2211840, device=device(type='cuda', index=0))
    reader.tensor(buf146, (320, 192, 3, 3), (1728, 1, 576, 192), is_leaf=True)  # primals_222
    buf147 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf147, (320,), is_leaf=True)  # primals_223
    buf148 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf148, (192, 768, 1, 1), is_leaf=True)  # primals_225
    buf149 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf149, (192,), is_leaf=True)  # primals_226
    buf150 = reader.storage(None, 1032192, device=device(type='cuda', index=0))
    reader.tensor(buf150, (192, 192, 1, 7), (1344, 1, 1344, 192), is_leaf=True)  # primals_228
    buf151 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf151, (192,), is_leaf=True)  # primals_229
    buf152 = reader.storage(None, 1032192, device=device(type='cuda', index=0))
    reader.tensor(buf152, (192, 192, 7, 1), (1344, 1, 192, 192), is_leaf=True)  # primals_231
    buf153 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf153, (192,), is_leaf=True)  # primals_232
    buf154 = reader.storage(None, 1327104, device=device(type='cuda', index=0))
    reader.tensor(buf154, (192, 192, 3, 3), (1728, 1, 576, 192), is_leaf=True)  # primals_234
    buf155 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf155, (192,), is_leaf=True)  # primals_235
    buf156 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf156, (320, 1280, 1, 1), is_leaf=True)  # primals_237
    buf157 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf157, (320,), is_leaf=True)  # primals_238
    buf158 = reader.storage(None, 1966080, device=device(type='cuda', index=0))
    reader.tensor(buf158, (384, 1280, 1, 1), is_leaf=True)  # primals_240
    buf159 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf159, (384,), is_leaf=True)  # primals_241
    buf160 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf160, (384, 384, 1, 3), (1152, 1, 1152, 384), is_leaf=True)  # primals_243
    buf161 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf161, (384,), is_leaf=True)  # primals_244
    buf162 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf162, (384, 384, 3, 1), (1152, 1, 384, 384), is_leaf=True)  # primals_246
    buf163 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf163, (384,), is_leaf=True)  # primals_247
    buf164 = reader.storage(None, 2293760, device=device(type='cuda', index=0))
    reader.tensor(buf164, (448, 1280, 1, 1), is_leaf=True)  # primals_249
    buf165 = reader.storage(None, 1792, device=device(type='cuda', index=0))
    reader.tensor(buf165, (448,), is_leaf=True)  # primals_250
    buf166 = reader.storage(None, 6193152, device=device(type='cuda', index=0))
    reader.tensor(buf166, (384, 448, 3, 3), (4032, 1, 1344, 448), is_leaf=True)  # primals_252
    buf167 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf167, (384,), is_leaf=True)  # primals_253
    buf168 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf168, (384, 384, 1, 3), (1152, 1, 1152, 384), is_leaf=True)  # primals_255
    buf169 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf169, (384,), is_leaf=True)  # primals_256
    buf170 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf170, (384, 384, 3, 1), (1152, 1, 384, 384), is_leaf=True)  # primals_258
    buf171 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf171, (384,), is_leaf=True)  # primals_259
    buf172 = reader.storage(None, 983040, device=device(type='cuda', index=0))
    reader.tensor(buf172, (192, 1280, 1, 1), is_leaf=True)  # primals_261
    buf173 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf173, (192,), is_leaf=True)  # primals_262
    buf174 = reader.storage(None, 2621440, device=device(type='cuda', index=0))
    reader.tensor(buf174, (320, 2048, 1, 1), is_leaf=True)  # primals_264
    buf175 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf175, (320,), is_leaf=True)  # primals_265
    buf176 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf176, (384, 2048, 1, 1), is_leaf=True)  # primals_267
    buf177 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf177, (384,), is_leaf=True)  # primals_268
    buf178 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf178, (384, 384, 1, 3), (1152, 1, 1152, 384), is_leaf=True)  # primals_270
    buf179 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf179, (384,), is_leaf=True)  # primals_271
    buf180 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf180, (384, 384, 3, 1), (1152, 1, 384, 384), is_leaf=True)  # primals_273
    buf181 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf181, (384,), is_leaf=True)  # primals_274
    buf182 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf182, (448, 2048, 1, 1), is_leaf=True)  # primals_276
    buf183 = reader.storage(None, 1792, device=device(type='cuda', index=0))
    reader.tensor(buf183, (448,), is_leaf=True)  # primals_277
    buf184 = reader.storage(None, 6193152, device=device(type='cuda', index=0))
    reader.tensor(buf184, (384, 448, 3, 3), (4032, 1, 1344, 448), is_leaf=True)  # primals_279
    buf185 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf185, (384,), is_leaf=True)  # primals_280
    buf186 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf186, (384, 384, 1, 3), (1152, 1, 1152, 384), is_leaf=True)  # primals_282
    buf187 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf187, (384,), is_leaf=True)  # primals_283
    buf188 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf188, (384, 384, 3, 1), (1152, 1, 384, 384), is_leaf=True)  # primals_285
    buf189 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf189, (384,), is_leaf=True)  # primals_286
    buf190 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf190, (192, 2048, 1, 1), is_leaf=True)  # primals_288
    buf191 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf191, (192,), is_leaf=True)  # primals_289
    buf192 = reader.storage(None, 34329984, device=device(type='cuda', index=0))
    reader.tensor(buf192, (32, 3, 299, 299), (268203, 1, 897, 3), is_leaf=True)  # cat
    buf193 = reader.storage(None, 90935296, device=device(type='cuda', index=0))
    reader.tensor(buf193, (32, 32, 149, 149), (710432, 1, 4768, 32), is_leaf=True)  # convolution
    buf194 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf194, (32,), is_leaf=True)  # squeeze_1
    buf195 = reader.storage(None, 90935296, device=device(type='cuda', index=0))
    reader.tensor(buf195, (32, 32, 149, 149), (710432, 1, 4768, 32), is_leaf=True)  # relu
    buf196 = reader.storage(None, 88510464, device=device(type='cuda', index=0))
    reader.tensor(buf196, (32, 32, 147, 147), (691488, 1, 4704, 32), is_leaf=True)  # convolution_1
    buf197 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf197, (32,), is_leaf=True)  # squeeze_4
    buf198 = reader.storage(None, 88510464, device=device(type='cuda', index=0))
    reader.tensor(buf198, (32, 32, 147, 147), (691488, 1, 4704, 32), is_leaf=True)  # relu_1
    buf199 = reader.storage(None, 177020928, device=device(type='cuda', index=0))
    reader.tensor(buf199, (32, 64, 147, 147), (1382976, 1, 9408, 64), is_leaf=True)  # convolution_2
    buf200 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf200, (64,), is_leaf=True)  # squeeze_7
    buf201 = reader.storage(None, 177020928, device=device(type='cuda', index=0))
    reader.tensor(buf201, (32, 64, 147, 147), (1382976, 1, 9408, 64), is_leaf=True)  # relu_2
    buf202 = reader.storage(None, 43655168, device=device(type='cuda', index=0))
    reader.tensor(buf202, (32, 64, 73, 73), (341056, 1, 4672, 64), is_leaf=True)  # getitem_6
    buf203 = reader.storage(None, 87310336, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf203, (32, 64, 73, 73), (341056, 1, 4672, 64), dtype=torch.int64, is_leaf=True)  # getitem_7
    buf204 = reader.storage(None, 54568960, device=device(type='cuda', index=0))
    reader.tensor(buf204, (32, 80, 73, 73), (426320, 1, 5840, 80), is_leaf=True)  # convolution_3
    buf205 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf205, (80,), is_leaf=True)  # squeeze_10
    buf206 = reader.storage(None, 54568960, device=device(type='cuda', index=0))
    reader.tensor(buf206, (32, 80, 73, 73), (426320, 1, 5840, 80), is_leaf=True)  # relu_3
    buf207 = reader.storage(None, 123887616, device=device(type='cuda', index=0))
    reader.tensor(buf207, (32, 192, 71, 71), (967872, 1, 13632, 192), is_leaf=True)  # convolution_4
    buf208 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf208, (192,), is_leaf=True)  # squeeze_13
    buf209 = reader.storage(None, 123887616, device=device(type='cuda', index=0))
    reader.tensor(buf209, (32, 192, 71, 71), (967872, 1, 13632, 192), is_leaf=True)  # relu_4
    buf210 = reader.storage(None, 30105600, device=device(type='cuda', index=0))
    reader.tensor(buf210, (32, 192, 35, 35), (235200, 1, 6720, 192), is_leaf=True)  # getitem_12
    buf211 = reader.storage(None, 60211200, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf211, (32, 192, 35, 35), (235200, 1, 6720, 192), dtype=torch.int64, is_leaf=True)  # getitem_13
    buf212 = reader.storage(None, 10035200, device=device(type='cuda', index=0))
    reader.tensor(buf212, (32, 64, 35, 35), (78400, 1, 2240, 64), is_leaf=True)  # convolution_5
    buf213 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf213, (64,), is_leaf=True)  # squeeze_16
    buf214 = reader.storage(None, 7526400, device=device(type='cuda', index=0))
    reader.tensor(buf214, (32, 48, 35, 35), (58800, 1, 1680, 48), is_leaf=True)  # convolution_6
    buf215 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf215, (48,), is_leaf=True)  # squeeze_19
    buf216 = reader.storage(None, 7526400, device=device(type='cuda', index=0))
    reader.tensor(buf216, (32, 48, 35, 35), (58800, 1, 1680, 48), is_leaf=True)  # relu_6
    buf217 = reader.storage(None, 10035200, device=device(type='cuda', index=0))
    reader.tensor(buf217, (32, 64, 35, 35), (78400, 1, 2240, 64), is_leaf=True)  # convolution_7
    buf218 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf218, (64,), is_leaf=True)  # squeeze_22
    buf219 = reader.storage(None, 10035200, device=device(type='cuda', index=0))
    reader.tensor(buf219, (32, 64, 35, 35), (78400, 1, 2240, 64), is_leaf=True)  # convolution_8
    buf220 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf220, (64,), is_leaf=True)  # squeeze_25
    buf221 = reader.storage(None, 10035200, device=device(type='cuda', index=0))
    reader.tensor(buf221, (32, 64, 35, 35), (78400, 1, 2240, 64), is_leaf=True)  # relu_8
    buf222 = reader.storage(None, 15052800, device=device(type='cuda', index=0))
    reader.tensor(buf222, (32, 96, 35, 35), (117600, 1, 3360, 96), is_leaf=True)  # convolution_9
    buf223 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf223, (96,), is_leaf=True)  # squeeze_28
    buf224 = reader.storage(None, 15052800, device=device(type='cuda', index=0))
    reader.tensor(buf224, (32, 96, 35, 35), (117600, 1, 3360, 96), is_leaf=True)  # relu_9
    buf225 = reader.storage(None, 15052800, device=device(type='cuda', index=0))
    reader.tensor(buf225, (32, 96, 35, 35), (117600, 1, 3360, 96), is_leaf=True)  # convolution_10
    buf226 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf226, (96,), is_leaf=True)  # squeeze_31
    buf227 = reader.storage(None, 30105600, device=device(type='cuda', index=0))
    reader.tensor(buf227, (32, 192, 35, 35), (235200, 1, 6720, 192), is_leaf=True)  # avg_pool2d
    buf228 = reader.storage(None, 5017600, device=device(type='cuda', index=0))
    reader.tensor(buf228, (32, 32, 35, 35), (39200, 1, 1120, 32), is_leaf=True)  # convolution_11
    buf229 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf229, (32,), is_leaf=True)  # squeeze_34
    buf230 = reader.storage(None, 40140800, device=device(type='cuda', index=0))
    reader.tensor(buf230, (32, 256, 35, 35), (313600, 1, 8960, 256), is_leaf=True)  # cat_1
    buf231 = reader.storage(None, 10035200, device=device(type='cuda', index=0))
    reader.tensor(buf231, (32, 64, 35, 35), (78400, 1, 2240, 64), is_leaf=True)  # convolution_12
    buf232 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf232, (64,), is_leaf=True)  # squeeze_37
    buf233 = reader.storage(None, 7526400, device=device(type='cuda', index=0))
    reader.tensor(buf233, (32, 48, 35, 35), (58800, 1, 1680, 48), is_leaf=True)  # convolution_13
    buf234 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf234, (48,), is_leaf=True)  # squeeze_40
    buf235 = reader.storage(None, 7526400, device=device(type='cuda', index=0))
    reader.tensor(buf235, (32, 48, 35, 35), (58800, 1, 1680, 48), is_leaf=True)  # relu_13
    buf236 = reader.storage(None, 10035200, device=device(type='cuda', index=0))
    reader.tensor(buf236, (32, 64, 35, 35), (78400, 1, 2240, 64), is_leaf=True)  # convolution_14
    buf237 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf237, (64,), is_leaf=True)  # squeeze_43
    buf238 = reader.storage(None, 10035200, device=device(type='cuda', index=0))
    reader.tensor(buf238, (32, 64, 35, 35), (78400, 1, 2240, 64), is_leaf=True)  # convolution_15
    buf239 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf239, (64,), is_leaf=True)  # squeeze_46
    buf240 = reader.storage(None, 10035200, device=device(type='cuda', index=0))
    reader.tensor(buf240, (32, 64, 35, 35), (78400, 1, 2240, 64), is_leaf=True)  # relu_15
    buf241 = reader.storage(None, 15052800, device=device(type='cuda', index=0))
    reader.tensor(buf241, (32, 96, 35, 35), (117600, 1, 3360, 96), is_leaf=True)  # convolution_16
    buf242 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf242, (96,), is_leaf=True)  # squeeze_49
    buf243 = reader.storage(None, 15052800, device=device(type='cuda', index=0))
    reader.tensor(buf243, (32, 96, 35, 35), (117600, 1, 3360, 96), is_leaf=True)  # relu_16
    buf244 = reader.storage(None, 15052800, device=device(type='cuda', index=0))
    reader.tensor(buf244, (32, 96, 35, 35), (117600, 1, 3360, 96), is_leaf=True)  # convolution_17
    buf245 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf245, (96,), is_leaf=True)  # squeeze_52
    buf246 = reader.storage(None, 40140800, device=device(type='cuda', index=0))
    reader.tensor(buf246, (32, 256, 35, 35), (313600, 1, 8960, 256), is_leaf=True)  # avg_pool2d_1
    buf247 = reader.storage(None, 10035200, device=device(type='cuda', index=0))
    reader.tensor(buf247, (32, 64, 35, 35), (78400, 1, 2240, 64), is_leaf=True)  # convolution_18
    buf248 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf248, (64,), is_leaf=True)  # squeeze_55
    buf249 = reader.storage(None, 45158400, device=device(type='cuda', index=0))
    reader.tensor(buf249, (32, 288, 35, 35), (352800, 1, 10080, 288), is_leaf=True)  # cat_2
    buf250 = reader.storage(None, 10035200, device=device(type='cuda', index=0))
    reader.tensor(buf250, (32, 64, 35, 35), (78400, 1, 2240, 64), is_leaf=True)  # convolution_19
    buf251 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf251, (64,), is_leaf=True)  # squeeze_58
    buf252 = reader.storage(None, 7526400, device=device(type='cuda', index=0))
    reader.tensor(buf252, (32, 48, 35, 35), (58800, 1, 1680, 48), is_leaf=True)  # convolution_20
    buf253 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf253, (48,), is_leaf=True)  # squeeze_61
    buf254 = reader.storage(None, 7526400, device=device(type='cuda', index=0))
    reader.tensor(buf254, (32, 48, 35, 35), (58800, 1, 1680, 48), is_leaf=True)  # relu_20
    buf255 = reader.storage(None, 10035200, device=device(type='cuda', index=0))
    reader.tensor(buf255, (32, 64, 35, 35), (78400, 1, 2240, 64), is_leaf=True)  # convolution_21
    buf256 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf256, (64,), is_leaf=True)  # squeeze_64
    buf257 = reader.storage(None, 10035200, device=device(type='cuda', index=0))
    reader.tensor(buf257, (32, 64, 35, 35), (78400, 1, 2240, 64), is_leaf=True)  # convolution_22
    buf258 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf258, (64,), is_leaf=True)  # squeeze_67
    buf259 = reader.storage(None, 10035200, device=device(type='cuda', index=0))
    reader.tensor(buf259, (32, 64, 35, 35), (78400, 1, 2240, 64), is_leaf=True)  # relu_22
    buf260 = reader.storage(None, 15052800, device=device(type='cuda', index=0))
    reader.tensor(buf260, (32, 96, 35, 35), (117600, 1, 3360, 96), is_leaf=True)  # convolution_23
    buf261 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf261, (96,), is_leaf=True)  # squeeze_70
    buf262 = reader.storage(None, 15052800, device=device(type='cuda', index=0))
    reader.tensor(buf262, (32, 96, 35, 35), (117600, 1, 3360, 96), is_leaf=True)  # relu_23
    buf263 = reader.storage(None, 15052800, device=device(type='cuda', index=0))
    reader.tensor(buf263, (32, 96, 35, 35), (117600, 1, 3360, 96), is_leaf=True)  # convolution_24
    buf264 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf264, (96,), is_leaf=True)  # squeeze_73
    buf265 = reader.storage(None, 45158400, device=device(type='cuda', index=0))
    reader.tensor(buf265, (32, 288, 35, 35), (352800, 1, 10080, 288), is_leaf=True)  # avg_pool2d_2
    buf266 = reader.storage(None, 10035200, device=device(type='cuda', index=0))
    reader.tensor(buf266, (32, 64, 35, 35), (78400, 1, 2240, 64), is_leaf=True)  # convolution_25
    buf267 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf267, (64,), is_leaf=True)  # squeeze_76
    buf268 = reader.storage(None, 45158400, device=device(type='cuda', index=0))
    reader.tensor(buf268, (32, 288, 35, 35), (352800, 1, 10080, 288), is_leaf=True)  # cat_3
    buf269 = reader.storage(None, 14204928, device=device(type='cuda', index=0))
    reader.tensor(buf269, (32, 384, 17, 17), (110976, 1, 6528, 384), is_leaf=True)  # convolution_26
    buf270 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf270, (384,), is_leaf=True)  # squeeze_79
    buf271 = reader.storage(None, 10035200, device=device(type='cuda', index=0))
    reader.tensor(buf271, (32, 64, 35, 35), (78400, 1, 2240, 64), is_leaf=True)  # convolution_27
    buf272 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf272, (64,), is_leaf=True)  # squeeze_82
    buf273 = reader.storage(None, 10035200, device=device(type='cuda', index=0))
    reader.tensor(buf273, (32, 64, 35, 35), (78400, 1, 2240, 64), is_leaf=True)  # relu_27
    buf274 = reader.storage(None, 15052800, device=device(type='cuda', index=0))
    reader.tensor(buf274, (32, 96, 35, 35), (117600, 1, 3360, 96), is_leaf=True)  # convolution_28
    buf275 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf275, (96,), is_leaf=True)  # squeeze_85
    buf276 = reader.storage(None, 15052800, device=device(type='cuda', index=0))
    reader.tensor(buf276, (32, 96, 35, 35), (117600, 1, 3360, 96), is_leaf=True)  # relu_28
    buf277 = reader.storage(None, 3551232, device=device(type='cuda', index=0))
    reader.tensor(buf277, (32, 96, 17, 17), (27744, 1, 1632, 96), is_leaf=True)  # convolution_29
    buf278 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf278, (96,), is_leaf=True)  # squeeze_88
    buf279 = reader.storage(None, 21307392, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf279, (32, 288, 17, 17), (83232, 1, 4896, 288), dtype=torch.int64, is_leaf=True)  # getitem_65
    buf280 = reader.storage(None, 28409856, device=device(type='cuda', index=0))
    reader.tensor(buf280, (32, 768, 17, 17), (221952, 1, 13056, 768), is_leaf=True)  # cat_4
    buf281 = reader.storage(None, 7102464, device=device(type='cuda', index=0))
    reader.tensor(buf281, (32, 192, 17, 17), (55488, 1, 3264, 192), is_leaf=True)  # convolution_30
    buf282 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf282, (192,), is_leaf=True)  # squeeze_91
    buf283 = reader.storage(None, 4734976, device=device(type='cuda', index=0))
    reader.tensor(buf283, (32, 128, 17, 17), (36992, 1, 2176, 128), is_leaf=True)  # convolution_31
    buf284 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf284, (128,), is_leaf=True)  # squeeze_94
    buf285 = reader.storage(None, 4734976, device=device(type='cuda', index=0))
    reader.tensor(buf285, (32, 128, 17, 17), (36992, 1, 2176, 128), is_leaf=True)  # relu_31
    buf286 = reader.storage(None, 4734976, device=device(type='cuda', index=0))
    reader.tensor(buf286, (32, 128, 17, 17), (36992, 1, 2176, 128), is_leaf=True)  # convolution_32
    buf287 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf287, (128,), is_leaf=True)  # squeeze_97
    buf288 = reader.storage(None, 4734976, device=device(type='cuda', index=0))
    reader.tensor(buf288, (32, 128, 17, 17), (36992, 1, 2176, 128), is_leaf=True)  # relu_32
    buf289 = reader.storage(None, 7102464, device=device(type='cuda', index=0))
    reader.tensor(buf289, (32, 192, 17, 17), (55488, 1, 3264, 192), is_leaf=True)  # convolution_33
    buf290 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf290, (192,), is_leaf=True)  # squeeze_100
    buf291 = reader.storage(None, 4734976, device=device(type='cuda', index=0))
    reader.tensor(buf291, (32, 128, 17, 17), (36992, 1, 2176, 128), is_leaf=True)  # convolution_34
    buf292 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf292, (128,), is_leaf=True)  # squeeze_103
    buf293 = reader.storage(None, 4734976, device=device(type='cuda', index=0))
    reader.tensor(buf293, (32, 128, 17, 17), (36992, 1, 2176, 128), is_leaf=True)  # relu_34
    buf294 = reader.storage(None, 4734976, device=device(type='cuda', index=0))
    reader.tensor(buf294, (32, 128, 17, 17), (36992, 1, 2176, 128), is_leaf=True)  # convolution_35
    buf295 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf295, (128,), is_leaf=True)  # squeeze_106
    buf296 = reader.storage(None, 4734976, device=device(type='cuda', index=0))
    reader.tensor(buf296, (32, 128, 17, 17), (36992, 1, 2176, 128), is_leaf=True)  # relu_35
    buf297 = reader.storage(None, 4734976, device=device(type='cuda', index=0))
    reader.tensor(buf297, (32, 128, 17, 17), (36992, 1, 2176, 128), is_leaf=True)  # convolution_36
    buf298 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf298, (128,), is_leaf=True)  # squeeze_109
    buf299 = reader.storage(None, 4734976, device=device(type='cuda', index=0))
    reader.tensor(buf299, (32, 128, 17, 17), (36992, 1, 2176, 128), is_leaf=True)  # relu_36
    buf300 = reader.storage(None, 4734976, device=device(type='cuda', index=0))
    reader.tensor(buf300, (32, 128, 17, 17), (36992, 1, 2176, 128), is_leaf=True)  # convolution_37
    buf301 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf301, (128,), is_leaf=True)  # squeeze_112
    buf302 = reader.storage(None, 4734976, device=device(type='cuda', index=0))
    reader.tensor(buf302, (32, 128, 17, 17), (36992, 1, 2176, 128), is_leaf=True)  # relu_37
    buf303 = reader.storage(None, 7102464, device=device(type='cuda', index=0))
    reader.tensor(buf303, (32, 192, 17, 17), (55488, 1, 3264, 192), is_leaf=True)  # convolution_38
    buf304 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf304, (192,), is_leaf=True)  # squeeze_115
    buf305 = reader.storage(None, 28409856, device=device(type='cuda', index=0))
    reader.tensor(buf305, (32, 768, 17, 17), (221952, 1, 13056, 768), is_leaf=True)  # avg_pool2d_3
    buf306 = reader.storage(None, 7102464, device=device(type='cuda', index=0))
    reader.tensor(buf306, (32, 192, 17, 17), (55488, 1, 3264, 192), is_leaf=True)  # convolution_39
    buf307 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf307, (192,), is_leaf=True)  # squeeze_118
    buf308 = reader.storage(None, 28409856, device=device(type='cuda', index=0))
    reader.tensor(buf308, (32, 768, 17, 17), (221952, 1, 13056, 768), is_leaf=True)  # cat_5
    buf309 = reader.storage(None, 7102464, device=device(type='cuda', index=0))
    reader.tensor(buf309, (32, 192, 17, 17), (55488, 1, 3264, 192), is_leaf=True)  # convolution_40
    buf310 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf310, (192,), is_leaf=True)  # squeeze_121
    buf311 = reader.storage(None, 5918720, device=device(type='cuda', index=0))
    reader.tensor(buf311, (32, 160, 17, 17), (46240, 1, 2720, 160), is_leaf=True)  # convolution_41
    buf312 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf312, (160,), is_leaf=True)  # squeeze_124
    buf313 = reader.storage(None, 5918720, device=device(type='cuda', index=0))
    reader.tensor(buf313, (32, 160, 17, 17), (46240, 1, 2720, 160), is_leaf=True)  # relu_41
    buf314 = reader.storage(None, 5918720, device=device(type='cuda', index=0))
    reader.tensor(buf314, (32, 160, 17, 17), (46240, 1, 2720, 160), is_leaf=True)  # convolution_42
    buf315 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf315, (160,), is_leaf=True)  # squeeze_127
    buf316 = reader.storage(None, 5918720, device=device(type='cuda', index=0))
    reader.tensor(buf316, (32, 160, 17, 17), (46240, 1, 2720, 160), is_leaf=True)  # relu_42
    buf317 = reader.storage(None, 7102464, device=device(type='cuda', index=0))
    reader.tensor(buf317, (32, 192, 17, 17), (55488, 1, 3264, 192), is_leaf=True)  # convolution_43
    buf318 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf318, (192,), is_leaf=True)  # squeeze_130
    buf319 = reader.storage(None, 5918720, device=device(type='cuda', index=0))
    reader.tensor(buf319, (32, 160, 17, 17), (46240, 1, 2720, 160), is_leaf=True)  # convolution_44
    buf320 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf320, (160,), is_leaf=True)  # squeeze_133
    buf321 = reader.storage(None, 5918720, device=device(type='cuda', index=0))
    reader.tensor(buf321, (32, 160, 17, 17), (46240, 1, 2720, 160), is_leaf=True)  # relu_44
    buf322 = reader.storage(None, 5918720, device=device(type='cuda', index=0))
    reader.tensor(buf322, (32, 160, 17, 17), (46240, 1, 2720, 160), is_leaf=True)  # convolution_45
    buf323 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf323, (160,), is_leaf=True)  # squeeze_136
    buf324 = reader.storage(None, 5918720, device=device(type='cuda', index=0))
    reader.tensor(buf324, (32, 160, 17, 17), (46240, 1, 2720, 160), is_leaf=True)  # relu_45
    buf325 = reader.storage(None, 5918720, device=device(type='cuda', index=0))
    reader.tensor(buf325, (32, 160, 17, 17), (46240, 1, 2720, 160), is_leaf=True)  # convolution_46
    buf326 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf326, (160,), is_leaf=True)  # squeeze_139
    buf327 = reader.storage(None, 5918720, device=device(type='cuda', index=0))
    reader.tensor(buf327, (32, 160, 17, 17), (46240, 1, 2720, 160), is_leaf=True)  # relu_46
    buf328 = reader.storage(None, 5918720, device=device(type='cuda', index=0))
    reader.tensor(buf328, (32, 160, 17, 17), (46240, 1, 2720, 160), is_leaf=True)  # convolution_47
    buf329 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf329, (160,), is_leaf=True)  # squeeze_142
    buf330 = reader.storage(None, 5918720, device=device(type='cuda', index=0))
    reader.tensor(buf330, (32, 160, 17, 17), (46240, 1, 2720, 160), is_leaf=True)  # relu_47
    buf331 = reader.storage(None, 7102464, device=device(type='cuda', index=0))
    reader.tensor(buf331, (32, 192, 17, 17), (55488, 1, 3264, 192), is_leaf=True)  # convolution_48
    buf332 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf332, (192,), is_leaf=True)  # squeeze_145
    buf333 = reader.storage(None, 28409856, device=device(type='cuda', index=0))
    reader.tensor(buf333, (32, 768, 17, 17), (221952, 1, 13056, 768), is_leaf=True)  # avg_pool2d_4
    buf334 = reader.storage(None, 7102464, device=device(type='cuda', index=0))
    reader.tensor(buf334, (32, 192, 17, 17), (55488, 1, 3264, 192), is_leaf=True)  # convolution_49
    buf335 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf335, (192,), is_leaf=True)  # squeeze_148
    buf336 = reader.storage(None, 28409856, device=device(type='cuda', index=0))
    reader.tensor(buf336, (32, 768, 17, 17), (221952, 1, 13056, 768), is_leaf=True)  # cat_6
    buf337 = reader.storage(None, 7102464, device=device(type='cuda', index=0))
    reader.tensor(buf337, (32, 192, 17, 17), (55488, 1, 3264, 192), is_leaf=True)  # convolution_50
    buf338 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf338, (192,), is_leaf=True)  # squeeze_151
    buf339 = reader.storage(None, 5918720, device=device(type='cuda', index=0))
    reader.tensor(buf339, (32, 160, 17, 17), (46240, 1, 2720, 160), is_leaf=True)  # convolution_51
    buf340 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf340, (160,), is_leaf=True)  # squeeze_154
    buf341 = reader.storage(None, 5918720, device=device(type='cuda', index=0))
    reader.tensor(buf341, (32, 160, 17, 17), (46240, 1, 2720, 160), is_leaf=True)  # relu_51
    buf342 = reader.storage(None, 5918720, device=device(type='cuda', index=0))
    reader.tensor(buf342, (32, 160, 17, 17), (46240, 1, 2720, 160), is_leaf=True)  # convolution_52
    buf343 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf343, (160,), is_leaf=True)  # squeeze_157
    buf344 = reader.storage(None, 5918720, device=device(type='cuda', index=0))
    reader.tensor(buf344, (32, 160, 17, 17), (46240, 1, 2720, 160), is_leaf=True)  # relu_52
    buf345 = reader.storage(None, 7102464, device=device(type='cuda', index=0))
    reader.tensor(buf345, (32, 192, 17, 17), (55488, 1, 3264, 192), is_leaf=True)  # convolution_53
    buf346 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf346, (192,), is_leaf=True)  # squeeze_160
    buf347 = reader.storage(None, 5918720, device=device(type='cuda', index=0))
    reader.tensor(buf347, (32, 160, 17, 17), (46240, 1, 2720, 160), is_leaf=True)  # convolution_54
    buf348 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf348, (160,), is_leaf=True)  # squeeze_163
    buf349 = reader.storage(None, 5918720, device=device(type='cuda', index=0))
    reader.tensor(buf349, (32, 160, 17, 17), (46240, 1, 2720, 160), is_leaf=True)  # relu_54
    buf350 = reader.storage(None, 5918720, device=device(type='cuda', index=0))
    reader.tensor(buf350, (32, 160, 17, 17), (46240, 1, 2720, 160), is_leaf=True)  # convolution_55
    buf351 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf351, (160,), is_leaf=True)  # squeeze_166
    buf352 = reader.storage(None, 5918720, device=device(type='cuda', index=0))
    reader.tensor(buf352, (32, 160, 17, 17), (46240, 1, 2720, 160), is_leaf=True)  # relu_55
    buf353 = reader.storage(None, 5918720, device=device(type='cuda', index=0))
    reader.tensor(buf353, (32, 160, 17, 17), (46240, 1, 2720, 160), is_leaf=True)  # convolution_56
    buf354 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf354, (160,), is_leaf=True)  # squeeze_169
    buf355 = reader.storage(None, 5918720, device=device(type='cuda', index=0))
    reader.tensor(buf355, (32, 160, 17, 17), (46240, 1, 2720, 160), is_leaf=True)  # relu_56
    buf356 = reader.storage(None, 5918720, device=device(type='cuda', index=0))
    reader.tensor(buf356, (32, 160, 17, 17), (46240, 1, 2720, 160), is_leaf=True)  # convolution_57
    buf357 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf357, (160,), is_leaf=True)  # squeeze_172
    buf358 = reader.storage(None, 5918720, device=device(type='cuda', index=0))
    reader.tensor(buf358, (32, 160, 17, 17), (46240, 1, 2720, 160), is_leaf=True)  # relu_57
    buf359 = reader.storage(None, 7102464, device=device(type='cuda', index=0))
    reader.tensor(buf359, (32, 192, 17, 17), (55488, 1, 3264, 192), is_leaf=True)  # convolution_58
    buf360 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf360, (192,), is_leaf=True)  # squeeze_175
    buf361 = reader.storage(None, 28409856, device=device(type='cuda', index=0))
    reader.tensor(buf361, (32, 768, 17, 17), (221952, 1, 13056, 768), is_leaf=True)  # avg_pool2d_5
    buf362 = reader.storage(None, 7102464, device=device(type='cuda', index=0))
    reader.tensor(buf362, (32, 192, 17, 17), (55488, 1, 3264, 192), is_leaf=True)  # convolution_59
    buf363 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf363, (192,), is_leaf=True)  # squeeze_178
    buf364 = reader.storage(None, 28409856, device=device(type='cuda', index=0))
    reader.tensor(buf364, (32, 768, 17, 17), (221952, 1, 13056, 768), is_leaf=True)  # cat_7
    buf365 = reader.storage(None, 7102464, device=device(type='cuda', index=0))
    reader.tensor(buf365, (32, 192, 17, 17), (55488, 1, 3264, 192), is_leaf=True)  # convolution_60
    buf366 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf366, (192,), is_leaf=True)  # squeeze_181
    buf367 = reader.storage(None, 7102464, device=device(type='cuda', index=0))
    reader.tensor(buf367, (32, 192, 17, 17), (55488, 1, 3264, 192), is_leaf=True)  # convolution_61
    buf368 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf368, (192,), is_leaf=True)  # squeeze_184
    buf369 = reader.storage(None, 7102464, device=device(type='cuda', index=0))
    reader.tensor(buf369, (32, 192, 17, 17), (55488, 1, 3264, 192), is_leaf=True)  # relu_61
    buf370 = reader.storage(None, 7102464, device=device(type='cuda', index=0))
    reader.tensor(buf370, (32, 192, 17, 17), (55488, 1, 3264, 192), is_leaf=True)  # convolution_62
    buf371 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf371, (192,), is_leaf=True)  # squeeze_187
    buf372 = reader.storage(None, 7102464, device=device(type='cuda', index=0))
    reader.tensor(buf372, (32, 192, 17, 17), (55488, 1, 3264, 192), is_leaf=True)  # relu_62
    buf373 = reader.storage(None, 7102464, device=device(type='cuda', index=0))
    reader.tensor(buf373, (32, 192, 17, 17), (55488, 1, 3264, 192), is_leaf=True)  # convolution_63
    buf374 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf374, (192,), is_leaf=True)  # squeeze_190
    buf375 = reader.storage(None, 7102464, device=device(type='cuda', index=0))
    reader.tensor(buf375, (32, 192, 17, 17), (55488, 1, 3264, 192), is_leaf=True)  # convolution_64
    buf376 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf376, (192,), is_leaf=True)  # squeeze_193
    buf377 = reader.storage(None, 7102464, device=device(type='cuda', index=0))
    reader.tensor(buf377, (32, 192, 17, 17), (55488, 1, 3264, 192), is_leaf=True)  # relu_64
    buf378 = reader.storage(None, 7102464, device=device(type='cuda', index=0))
    reader.tensor(buf378, (32, 192, 17, 17), (55488, 1, 3264, 192), is_leaf=True)  # convolution_65
    buf379 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf379, (192,), is_leaf=True)  # squeeze_196
    buf380 = reader.storage(None, 7102464, device=device(type='cuda', index=0))
    reader.tensor(buf380, (32, 192, 17, 17), (55488, 1, 3264, 192), is_leaf=True)  # relu_65
    buf381 = reader.storage(None, 7102464, device=device(type='cuda', index=0))
    reader.tensor(buf381, (32, 192, 17, 17), (55488, 1, 3264, 192), is_leaf=True)  # convolution_66
    buf382 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf382, (192,), is_leaf=True)  # squeeze_199
    buf383 = reader.storage(None, 7102464, device=device(type='cuda', index=0))
    reader.tensor(buf383, (32, 192, 17, 17), (55488, 1, 3264, 192), is_leaf=True)  # relu_66
    buf384 = reader.storage(None, 7102464, device=device(type='cuda', index=0))
    reader.tensor(buf384, (32, 192, 17, 17), (55488, 1, 3264, 192), is_leaf=True)  # convolution_67
    buf385 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf385, (192,), is_leaf=True)  # squeeze_202
    buf386 = reader.storage(None, 7102464, device=device(type='cuda', index=0))
    reader.tensor(buf386, (32, 192, 17, 17), (55488, 1, 3264, 192), is_leaf=True)  # relu_67
    buf387 = reader.storage(None, 7102464, device=device(type='cuda', index=0))
    reader.tensor(buf387, (32, 192, 17, 17), (55488, 1, 3264, 192), is_leaf=True)  # convolution_68
    buf388 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf388, (192,), is_leaf=True)  # squeeze_205
    buf389 = reader.storage(None, 28409856, device=device(type='cuda', index=0))
    reader.tensor(buf389, (32, 768, 17, 17), (221952, 1, 13056, 768), is_leaf=True)  # avg_pool2d_6
    buf390 = reader.storage(None, 7102464, device=device(type='cuda', index=0))
    reader.tensor(buf390, (32, 192, 17, 17), (55488, 1, 3264, 192), is_leaf=True)  # convolution_69
    buf391 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf391, (192,), is_leaf=True)  # squeeze_208
    buf392 = reader.storage(None, 28409856, device=device(type='cuda', index=0))
    reader.tensor(buf392, (32, 768, 17, 17), (221952, 1, 13056, 768), is_leaf=True)  # cat_8
    buf393 = reader.storage(None, 2457600, device=device(type='cuda', index=0))
    reader.tensor(buf393, (32, 768, 5, 5), (19200, 1, 3840, 768), is_leaf=True)  # avg_pool2d_7
    buf394 = reader.storage(None, 409600, device=device(type='cuda', index=0))
    reader.tensor(buf394, (32, 128, 5, 5), (3200, 1, 640, 128), is_leaf=True)  # convolution_70
    buf395 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf395, (128,), is_leaf=True)  # squeeze_211
    buf396 = reader.storage(None, 409600, device=device(type='cuda', index=0))
    reader.tensor(buf396, (32, 128, 5, 5), (3200, 1, 640, 128), is_leaf=True)  # relu_70
    buf397 = reader.storage(None, 98304, device=device(type='cuda', index=0))
    reader.tensor(buf397, (32, 768, 1, 1), (768, 1, 768, 768), is_leaf=True)  # convolution_71
    buf398 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf398, (768,), is_leaf=True)  # squeeze_214
    buf399 = reader.storage(None, 98304, device=device(type='cuda', index=0))
    reader.tensor(buf399, (32, 768), is_leaf=True)  # view
    buf400 = reader.storage(None, 7102464, device=device(type='cuda', index=0))
    reader.tensor(buf400, (32, 192, 17, 17), (55488, 1, 3264, 192), is_leaf=True)  # convolution_72
    buf401 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf401, (192,), is_leaf=True)  # squeeze_217
    buf402 = reader.storage(None, 7102464, device=device(type='cuda', index=0))
    reader.tensor(buf402, (32, 192, 17, 17), (55488, 1, 3264, 192), is_leaf=True)  # relu_72
    buf403 = reader.storage(None, 2621440, device=device(type='cuda', index=0))
    reader.tensor(buf403, (32, 320, 8, 8), (20480, 1, 2560, 320), is_leaf=True)  # convolution_73
    buf404 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf404, (320,), is_leaf=True)  # squeeze_220
    buf405 = reader.storage(None, 7102464, device=device(type='cuda', index=0))
    reader.tensor(buf405, (32, 192, 17, 17), (55488, 1, 3264, 192), is_leaf=True)  # convolution_74
    buf406 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf406, (192,), is_leaf=True)  # squeeze_223
    buf407 = reader.storage(None, 7102464, device=device(type='cuda', index=0))
    reader.tensor(buf407, (32, 192, 17, 17), (55488, 1, 3264, 192), is_leaf=True)  # relu_74
    buf408 = reader.storage(None, 7102464, device=device(type='cuda', index=0))
    reader.tensor(buf408, (32, 192, 17, 17), (55488, 1, 3264, 192), is_leaf=True)  # convolution_75
    buf409 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf409, (192,), is_leaf=True)  # squeeze_226
    buf410 = reader.storage(None, 7102464, device=device(type='cuda', index=0))
    reader.tensor(buf410, (32, 192, 17, 17), (55488, 1, 3264, 192), is_leaf=True)  # relu_75
    buf411 = reader.storage(None, 7102464, device=device(type='cuda', index=0))
    reader.tensor(buf411, (32, 192, 17, 17), (55488, 1, 3264, 192), is_leaf=True)  # convolution_76
    buf412 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf412, (192,), is_leaf=True)  # squeeze_229
    buf413 = reader.storage(None, 7102464, device=device(type='cuda', index=0))
    reader.tensor(buf413, (32, 192, 17, 17), (55488, 1, 3264, 192), is_leaf=True)  # relu_76
    buf414 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf414, (32, 192, 8, 8), (12288, 1, 1536, 192), is_leaf=True)  # convolution_77
    buf415 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf415, (192,), is_leaf=True)  # squeeze_232
    buf416 = reader.storage(None, 12582912, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf416, (32, 768, 8, 8), (49152, 1, 6144, 768), dtype=torch.int64, is_leaf=True)  # getitem_163
    buf417 = reader.storage(None, 10485760, device=device(type='cuda', index=0))
    reader.tensor(buf417, (32, 1280, 8, 8), (81920, 1, 10240, 1280), is_leaf=True)  # cat_9
    buf418 = reader.storage(None, 2621440, device=device(type='cuda', index=0))
    reader.tensor(buf418, (32, 320, 8, 8), (20480, 1, 2560, 320), is_leaf=True)  # convolution_78
    buf419 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf419, (320,), is_leaf=True)  # squeeze_235
    buf420 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf420, (32, 384, 8, 8), (24576, 1, 3072, 384), is_leaf=True)  # convolution_79
    buf421 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf421, (384,), is_leaf=True)  # squeeze_238
    buf422 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf422, (32, 384, 8, 8), (24576, 1, 3072, 384), is_leaf=True)  # relu_79
    buf423 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf423, (32, 384, 8, 8), (24576, 1, 3072, 384), is_leaf=True)  # convolution_80
    buf424 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf424, (384,), is_leaf=True)  # squeeze_241
    buf425 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf425, (32, 384, 8, 8), (24576, 1, 3072, 384), is_leaf=True)  # convolution_81
    buf426 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf426, (384,), is_leaf=True)  # squeeze_244
    buf427 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf427, (32, 448, 8, 8), (28672, 1, 3584, 448), is_leaf=True)  # convolution_82
    buf428 = reader.storage(None, 1792, device=device(type='cuda', index=0))
    reader.tensor(buf428, (448,), is_leaf=True)  # squeeze_247
    buf429 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf429, (32, 448, 8, 8), (28672, 1, 3584, 448), is_leaf=True)  # relu_82
    buf430 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf430, (32, 384, 8, 8), (24576, 1, 3072, 384), is_leaf=True)  # convolution_83
    buf431 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf431, (384,), is_leaf=True)  # squeeze_250
    buf432 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf432, (32, 384, 8, 8), (24576, 1, 3072, 384), is_leaf=True)  # relu_83
    buf433 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf433, (32, 384, 8, 8), (24576, 1, 3072, 384), is_leaf=True)  # convolution_84
    buf434 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf434, (384,), is_leaf=True)  # squeeze_253
    buf435 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf435, (32, 384, 8, 8), (24576, 1, 3072, 384), is_leaf=True)  # convolution_85
    buf436 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf436, (384,), is_leaf=True)  # squeeze_256
    buf437 = reader.storage(None, 10485760, device=device(type='cuda', index=0))
    reader.tensor(buf437, (32, 1280, 8, 8), (81920, 1, 10240, 1280), is_leaf=True)  # avg_pool2d_8
    buf438 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf438, (32, 192, 8, 8), (12288, 1, 1536, 192), is_leaf=True)  # convolution_86
    buf439 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf439, (192,), is_leaf=True)  # squeeze_259
    buf440 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf440, (32, 2048, 8, 8), (131072, 1, 16384, 2048), is_leaf=True)  # cat_12
    buf441 = reader.storage(None, 2621440, device=device(type='cuda', index=0))
    reader.tensor(buf441, (32, 320, 8, 8), (20480, 1, 2560, 320), is_leaf=True)  # convolution_87
    buf442 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf442, (320,), is_leaf=True)  # squeeze_262
    buf443 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf443, (32, 384, 8, 8), (24576, 1, 3072, 384), is_leaf=True)  # convolution_88
    buf444 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf444, (384,), is_leaf=True)  # squeeze_265
    buf445 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf445, (32, 384, 8, 8), (24576, 1, 3072, 384), is_leaf=True)  # relu_88
    buf446 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf446, (32, 384, 8, 8), (24576, 1, 3072, 384), is_leaf=True)  # convolution_89
    buf447 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf447, (384,), is_leaf=True)  # squeeze_268
    buf448 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf448, (32, 384, 8, 8), (24576, 1, 3072, 384), is_leaf=True)  # convolution_90
    buf449 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf449, (384,), is_leaf=True)  # squeeze_271
    buf450 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf450, (32, 448, 8, 8), (28672, 1, 3584, 448), is_leaf=True)  # convolution_91
    buf451 = reader.storage(None, 1792, device=device(type='cuda', index=0))
    reader.tensor(buf451, (448,), is_leaf=True)  # squeeze_274
    buf452 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf452, (32, 448, 8, 8), (28672, 1, 3584, 448), is_leaf=True)  # relu_91
    buf453 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf453, (32, 384, 8, 8), (24576, 1, 3072, 384), is_leaf=True)  # convolution_92
    buf454 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf454, (384,), is_leaf=True)  # squeeze_277
    buf455 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf455, (32, 384, 8, 8), (24576, 1, 3072, 384), is_leaf=True)  # relu_92
    buf456 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf456, (32, 384, 8, 8), (24576, 1, 3072, 384), is_leaf=True)  # convolution_93
    buf457 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf457, (384,), is_leaf=True)  # squeeze_280
    buf458 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf458, (32, 384, 8, 8), (24576, 1, 3072, 384), is_leaf=True)  # convolution_94
    buf459 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf459, (384,), is_leaf=True)  # squeeze_283
    buf460 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf460, (32, 2048, 8, 8), (131072, 1, 16384, 2048), is_leaf=True)  # avg_pool2d_9
    buf461 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf461, (32, 192, 8, 8), (12288, 1, 1536, 192), is_leaf=True)  # convolution_95
    buf462 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf462, (192,), is_leaf=True)  # squeeze_286
    buf463 = reader.storage(None, 65536, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf463, (32, 2048, 1, 1), dtype=torch.bool, is_leaf=True)  # gt
    buf464 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf464, (32, 2048), is_leaf=True)  # view_1
    buf465 = reader.storage(None, 8192000, device=device(type='cuda', index=0))
    reader.tensor(buf465, (1000, 2048), is_leaf=True)  # permute_2
    buf466 = reader.storage(None, 393216, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf466, (32, 192, 8, 8), (12288, 1, 1536, 192), dtype=torch.bool, is_leaf=True)  # le
    buf467 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf467, (1, 192, 1, 1), is_leaf=True)  # unsqueeze_389
    buf468 = reader.storage(None, 786432, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf468, (32, 384, 8, 8), (24576, 1, 3072, 384), dtype=torch.bool, is_leaf=True)  # le_1
    buf469 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf469, (1, 384, 1, 1), is_leaf=True)  # unsqueeze_401
    buf470 = reader.storage(None, 786432, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf470, (32, 384, 8, 8), (24576, 1, 3072, 384), dtype=torch.bool, is_leaf=True)  # le_2
    buf471 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf471, (1, 384, 1, 1), is_leaf=True)  # unsqueeze_413
    buf472 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf472, (1, 384, 1, 1), is_leaf=True)  # unsqueeze_425
    buf473 = reader.storage(None, 1792, device=device(type='cuda', index=0))
    reader.tensor(buf473, (1, 448, 1, 1), is_leaf=True)  # unsqueeze_437
    buf474 = reader.storage(None, 786432, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf474, (32, 384, 8, 8), (24576, 1, 3072, 384), dtype=torch.bool, is_leaf=True)  # le_5
    buf475 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf475, (1, 384, 1, 1), is_leaf=True)  # unsqueeze_449
    buf476 = reader.storage(None, 786432, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf476, (32, 384, 8, 8), (24576, 1, 3072, 384), dtype=torch.bool, is_leaf=True)  # le_6
    buf477 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf477, (1, 384, 1, 1), is_leaf=True)  # unsqueeze_461
    buf478 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf478, (1, 384, 1, 1), is_leaf=True)  # unsqueeze_473
    buf479 = reader.storage(None, 655360, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf479, (32, 320, 8, 8), (20480, 1, 2560, 320), dtype=torch.bool, is_leaf=True)  # le_8
    buf480 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf480, (1, 320, 1, 1), is_leaf=True)  # unsqueeze_485
    buf481 = reader.storage(None, 393216, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf481, (32, 192, 8, 8), (12288, 1, 1536, 192), dtype=torch.bool, is_leaf=True)  # le_9
    buf482 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf482, (1, 192, 1, 1), is_leaf=True)  # unsqueeze_497
    buf483 = reader.storage(None, 786432, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf483, (32, 384, 8, 8), (24576, 1, 3072, 384), dtype=torch.bool, is_leaf=True)  # le_10
    buf484 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf484, (1, 384, 1, 1), is_leaf=True)  # unsqueeze_509
    buf485 = reader.storage(None, 786432, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf485, (32, 384, 8, 8), (24576, 1, 3072, 384), dtype=torch.bool, is_leaf=True)  # le_11
    buf486 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf486, (1, 384, 1, 1), is_leaf=True)  # unsqueeze_521
    buf487 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf487, (1, 384, 1, 1), is_leaf=True)  # unsqueeze_533
    buf488 = reader.storage(None, 1792, device=device(type='cuda', index=0))
    reader.tensor(buf488, (1, 448, 1, 1), is_leaf=True)  # unsqueeze_545
    buf489 = reader.storage(None, 786432, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf489, (32, 384, 8, 8), (24576, 1, 3072, 384), dtype=torch.bool, is_leaf=True)  # le_14
    buf490 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf490, (1, 384, 1, 1), is_leaf=True)  # unsqueeze_557
    buf491 = reader.storage(None, 786432, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf491, (32, 384, 8, 8), (24576, 1, 3072, 384), dtype=torch.bool, is_leaf=True)  # le_15
    buf492 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf492, (1, 384, 1, 1), is_leaf=True)  # unsqueeze_569
    buf493 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf493, (1, 384, 1, 1), is_leaf=True)  # unsqueeze_581
    buf494 = reader.storage(None, 655360, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf494, (32, 320, 8, 8), (20480, 1, 2560, 320), dtype=torch.bool, is_leaf=True)  # le_17
    buf495 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf495, (1, 320, 1, 1), is_leaf=True)  # unsqueeze_593
    buf496 = reader.storage(None, 393216, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf496, (32, 192, 8, 8), (12288, 1, 1536, 192), dtype=torch.bool, is_leaf=True)  # le_18
    buf497 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf497, (1, 192, 1, 1), is_leaf=True)  # unsqueeze_605
    buf498 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf498, (1, 192, 1, 1), is_leaf=True)  # unsqueeze_617
    buf499 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf499, (1, 192, 1, 1), is_leaf=True)  # unsqueeze_629
    buf500 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf500, (1, 192, 1, 1), is_leaf=True)  # unsqueeze_641
    buf501 = reader.storage(None, 655360, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf501, (32, 320, 8, 8), (20480, 1, 2560, 320), dtype=torch.bool, is_leaf=True)  # le_22
    buf502 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf502, (1, 320, 1, 1), is_leaf=True)  # unsqueeze_653
    buf503 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf503, (1, 192, 1, 1), is_leaf=True)  # unsqueeze_665
    buf504 = reader.storage(None, 3072000, device=device(type='cuda', index=0))
    reader.tensor(buf504, (1000, 768), is_leaf=True)  # permute_6
    buf505 = reader.storage(None, 24576, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf505, (32, 768, 1, 1), (768, 1, 768, 768), dtype=torch.bool, is_leaf=True)  # le_24
    buf506 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf506, (1, 768, 1, 1), is_leaf=True)  # unsqueeze_677
    buf507 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf507, (1, 128, 1, 1), is_leaf=True)  # unsqueeze_689
    buf508 = reader.storage(None, 1775616, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf508, (32, 192, 17, 17), (55488, 1, 3264, 192), dtype=torch.bool, is_leaf=True)  # le_26
    buf509 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf509, (1, 192, 1, 1), is_leaf=True)  # unsqueeze_701
    buf510 = reader.storage(None, 1775616, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf510, (32, 192, 17, 17), (55488, 1, 3264, 192), dtype=torch.bool, is_leaf=True)  # le_27
    buf511 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf511, (1, 192, 1, 1), is_leaf=True)  # unsqueeze_713
    buf512 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf512, (1, 192, 1, 1), is_leaf=True)  # unsqueeze_725
    buf513 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf513, (1, 192, 1, 1), is_leaf=True)  # unsqueeze_737
    buf514 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf514, (1, 192, 1, 1), is_leaf=True)  # unsqueeze_749
    buf515 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf515, (1, 192, 1, 1), is_leaf=True)  # unsqueeze_761
    buf516 = reader.storage(None, 1775616, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf516, (32, 192, 17, 17), (55488, 1, 3264, 192), dtype=torch.bool, is_leaf=True)  # le_32
    buf517 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf517, (1, 192, 1, 1), is_leaf=True)  # unsqueeze_773
    buf518 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf518, (1, 192, 1, 1), is_leaf=True)  # unsqueeze_785
    buf519 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf519, (1, 192, 1, 1), is_leaf=True)  # unsqueeze_797
    buf520 = reader.storage(None, 1775616, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf520, (32, 192, 17, 17), (55488, 1, 3264, 192), dtype=torch.bool, is_leaf=True)  # le_35
    buf521 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf521, (1, 192, 1, 1), is_leaf=True)  # unsqueeze_809
    buf522 = reader.storage(None, 1775616, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf522, (32, 192, 17, 17), (55488, 1, 3264, 192), dtype=torch.bool, is_leaf=True)  # le_36
    buf523 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf523, (1, 192, 1, 1), is_leaf=True)  # unsqueeze_821
    buf524 = reader.storage(None, 1775616, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf524, (32, 192, 17, 17), (55488, 1, 3264, 192), dtype=torch.bool, is_leaf=True)  # le_37
    buf525 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf525, (1, 192, 1, 1), is_leaf=True)  # unsqueeze_833
    buf526 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf526, (1, 160, 1, 1), is_leaf=True)  # unsqueeze_845
    buf527 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf527, (1, 160, 1, 1), is_leaf=True)  # unsqueeze_857
    buf528 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf528, (1, 160, 1, 1), is_leaf=True)  # unsqueeze_869
    buf529 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf529, (1, 160, 1, 1), is_leaf=True)  # unsqueeze_881
    buf530 = reader.storage(None, 1775616, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf530, (32, 192, 17, 17), (55488, 1, 3264, 192), dtype=torch.bool, is_leaf=True)  # le_42
    buf531 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf531, (1, 192, 1, 1), is_leaf=True)  # unsqueeze_893
    buf532 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf532, (1, 160, 1, 1), is_leaf=True)  # unsqueeze_905
    buf533 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf533, (1, 160, 1, 1), is_leaf=True)  # unsqueeze_917
    buf534 = reader.storage(None, 1775616, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf534, (32, 192, 17, 17), (55488, 1, 3264, 192), dtype=torch.bool, is_leaf=True)  # le_45
    buf535 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf535, (1, 192, 1, 1), is_leaf=True)  # unsqueeze_929
    buf536 = reader.storage(None, 1775616, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf536, (32, 192, 17, 17), (55488, 1, 3264, 192), dtype=torch.bool, is_leaf=True)  # le_46
    buf537 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf537, (1, 192, 1, 1), is_leaf=True)  # unsqueeze_941
    buf538 = reader.storage(None, 1775616, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf538, (32, 192, 17, 17), (55488, 1, 3264, 192), dtype=torch.bool, is_leaf=True)  # le_47
    buf539 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf539, (1, 192, 1, 1), is_leaf=True)  # unsqueeze_953
    buf540 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf540, (1, 160, 1, 1), is_leaf=True)  # unsqueeze_965
    buf541 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf541, (1, 160, 1, 1), is_leaf=True)  # unsqueeze_977
    buf542 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf542, (1, 160, 1, 1), is_leaf=True)  # unsqueeze_989
    buf543 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf543, (1, 160, 1, 1), is_leaf=True)  # unsqueeze_1001
    buf544 = reader.storage(None, 1775616, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf544, (32, 192, 17, 17), (55488, 1, 3264, 192), dtype=torch.bool, is_leaf=True)  # le_52
    buf545 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf545, (1, 192, 1, 1), is_leaf=True)  # unsqueeze_1013
    buf546 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf546, (1, 160, 1, 1), is_leaf=True)  # unsqueeze_1025
    buf547 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf547, (1, 160, 1, 1), is_leaf=True)  # unsqueeze_1037
    buf548 = reader.storage(None, 1775616, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf548, (32, 192, 17, 17), (55488, 1, 3264, 192), dtype=torch.bool, is_leaf=True)  # le_55
    buf549 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf549, (1, 192, 1, 1), is_leaf=True)  # unsqueeze_1049
    buf550 = reader.storage(None, 1775616, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf550, (32, 192, 17, 17), (55488, 1, 3264, 192), dtype=torch.bool, is_leaf=True)  # le_56
    buf551 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf551, (1, 192, 1, 1), is_leaf=True)  # unsqueeze_1061
    buf552 = reader.storage(None, 1775616, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf552, (32, 192, 17, 17), (55488, 1, 3264, 192), dtype=torch.bool, is_leaf=True)  # le_57
    buf553 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf553, (1, 192, 1, 1), is_leaf=True)  # unsqueeze_1073
    buf554 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf554, (1, 128, 1, 1), is_leaf=True)  # unsqueeze_1085
    buf555 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf555, (1, 128, 1, 1), is_leaf=True)  # unsqueeze_1097
    buf556 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf556, (1, 128, 1, 1), is_leaf=True)  # unsqueeze_1109
    buf557 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf557, (1, 128, 1, 1), is_leaf=True)  # unsqueeze_1121
    buf558 = reader.storage(None, 1775616, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf558, (32, 192, 17, 17), (55488, 1, 3264, 192), dtype=torch.bool, is_leaf=True)  # le_62
    buf559 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf559, (1, 192, 1, 1), is_leaf=True)  # unsqueeze_1133
    buf560 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf560, (1, 128, 1, 1), is_leaf=True)  # unsqueeze_1145
    buf561 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf561, (1, 128, 1, 1), is_leaf=True)  # unsqueeze_1157
    buf562 = reader.storage(None, 1775616, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf562, (32, 192, 17, 17), (55488, 1, 3264, 192), dtype=torch.bool, is_leaf=True)  # le_65
    buf563 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf563, (1, 192, 1, 1), is_leaf=True)  # unsqueeze_1169
    buf564 = reader.storage(None, 887808, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf564, (32, 96, 17, 17), (27744, 1, 1632, 96), dtype=torch.bool, is_leaf=True)  # le_66
    buf565 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf565, (1, 96, 1, 1), is_leaf=True)  # unsqueeze_1181
    buf566 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf566, (1, 96, 1, 1), is_leaf=True)  # unsqueeze_1193
    buf567 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf567, (1, 64, 1, 1), is_leaf=True)  # unsqueeze_1205
    buf568 = reader.storage(None, 3551232, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf568, (32, 384, 17, 17), (110976, 1, 6528, 384), dtype=torch.bool, is_leaf=True)  # le_69
    buf569 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf569, (1, 384, 1, 1), is_leaf=True)  # unsqueeze_1217
    buf570 = reader.storage(None, 2508800, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf570, (32, 64, 35, 35), (78400, 1, 2240, 64), dtype=torch.bool, is_leaf=True)  # le_70
    buf571 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf571, (1, 64, 1, 1), is_leaf=True)  # unsqueeze_1229
    buf572 = reader.storage(None, 3763200, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf572, (32, 96, 35, 35), (117600, 1, 3360, 96), dtype=torch.bool, is_leaf=True)  # le_71
    buf573 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf573, (1, 96, 1, 1), is_leaf=True)  # unsqueeze_1241
    buf574 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf574, (1, 96, 1, 1), is_leaf=True)  # unsqueeze_1253
    buf575 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf575, (1, 64, 1, 1), is_leaf=True)  # unsqueeze_1265
    buf576 = reader.storage(None, 2508800, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf576, (32, 64, 35, 35), (78400, 1, 2240, 64), dtype=torch.bool, is_leaf=True)  # le_74
    buf577 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf577, (1, 64, 1, 1), is_leaf=True)  # unsqueeze_1277
    buf578 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf578, (1, 48, 1, 1), is_leaf=True)  # unsqueeze_1289
    buf579 = reader.storage(None, 2508800, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf579, (32, 64, 35, 35), (78400, 1, 2240, 64), dtype=torch.bool, is_leaf=True)  # le_76
    buf580 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf580, (1, 64, 1, 1), is_leaf=True)  # unsqueeze_1301
    buf581 = reader.storage(None, 2508800, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf581, (32, 64, 35, 35), (78400, 1, 2240, 64), dtype=torch.bool, is_leaf=True)  # le_77
    buf582 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf582, (1, 64, 1, 1), is_leaf=True)  # unsqueeze_1313
    buf583 = reader.storage(None, 3763200, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf583, (32, 96, 35, 35), (117600, 1, 3360, 96), dtype=torch.bool, is_leaf=True)  # le_78
    buf584 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf584, (1, 96, 1, 1), is_leaf=True)  # unsqueeze_1325
    buf585 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf585, (1, 96, 1, 1), is_leaf=True)  # unsqueeze_1337
    buf586 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf586, (1, 64, 1, 1), is_leaf=True)  # unsqueeze_1349
    buf587 = reader.storage(None, 2508800, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf587, (32, 64, 35, 35), (78400, 1, 2240, 64), dtype=torch.bool, is_leaf=True)  # le_81
    buf588 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf588, (1, 64, 1, 1), is_leaf=True)  # unsqueeze_1361
    buf589 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf589, (1, 48, 1, 1), is_leaf=True)  # unsqueeze_1373
    buf590 = reader.storage(None, 2508800, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf590, (32, 64, 35, 35), (78400, 1, 2240, 64), dtype=torch.bool, is_leaf=True)  # le_83
    buf591 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf591, (1, 64, 1, 1), is_leaf=True)  # unsqueeze_1385
    buf592 = reader.storage(None, 1254400, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf592, (32, 32, 35, 35), (39200, 1, 1120, 32), dtype=torch.bool, is_leaf=True)  # le_84
    buf593 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf593, (1, 32, 1, 1), is_leaf=True)  # unsqueeze_1397
    buf594 = reader.storage(None, 3763200, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf594, (32, 96, 35, 35), (117600, 1, 3360, 96), dtype=torch.bool, is_leaf=True)  # le_85
    buf595 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf595, (1, 96, 1, 1), is_leaf=True)  # unsqueeze_1409
    buf596 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf596, (1, 96, 1, 1), is_leaf=True)  # unsqueeze_1421
    buf597 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf597, (1, 64, 1, 1), is_leaf=True)  # unsqueeze_1433
    buf598 = reader.storage(None, 2508800, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf598, (32, 64, 35, 35), (78400, 1, 2240, 64), dtype=torch.bool, is_leaf=True)  # le_88
    buf599 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf599, (1, 64, 1, 1), is_leaf=True)  # unsqueeze_1445
    buf600 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf600, (1, 48, 1, 1), is_leaf=True)  # unsqueeze_1457
    buf601 = reader.storage(None, 2508800, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf601, (32, 64, 35, 35), (78400, 1, 2240, 64), dtype=torch.bool, is_leaf=True)  # le_90
    buf602 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf602, (1, 64, 1, 1), is_leaf=True)  # unsqueeze_1469
    buf603 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf603, (1, 192, 1, 1), is_leaf=True)  # unsqueeze_1481
    buf604 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf604, (1, 80, 1, 1), is_leaf=True)  # unsqueeze_1493
    buf605 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf605, (1, 64, 1, 1), is_leaf=True)  # unsqueeze_1505
    buf606 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf606, (1, 32, 1, 1), is_leaf=True)  # unsqueeze_1517
    buf607 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf607, (1, 32, 1, 1), is_leaf=True)  # unsqueeze_1529
    buf608 = reader.storage(None, 128000, device=device(type='cuda', index=0))
    reader.tensor(buf608, (32, 1000), is_leaf=True)  # tangents_1
    buf609 = reader.storage(None, 128000, device=device(type='cuda', index=0))
    reader.tensor(buf609, (32, 1000), is_leaf=True)  # tangents_2
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
