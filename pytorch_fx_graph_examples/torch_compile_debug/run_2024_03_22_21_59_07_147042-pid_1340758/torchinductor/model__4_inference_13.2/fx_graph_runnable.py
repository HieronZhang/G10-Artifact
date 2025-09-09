
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



# torch version: 2.2.1+cu121
# torch cuda version: 12.1
# torch git version: 6c8c5ad5eaf47a62fafbb4a2747198cbffbf1ff0


# CUDA Info: 
# nvcc not found
# GPU Hardware Info: 
# NVIDIA A100-PCIE-40GB : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1):
        _foreach_add = torch.ops.aten._foreach_add.Scalar([arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1], 1)
        getitem = _foreach_add[0]
        getitem_1 = _foreach_add[1]
        getitem_2 = _foreach_add[2]
        getitem_3 = _foreach_add[3]
        getitem_4 = _foreach_add[4]
        getitem_5 = _foreach_add[5];  _foreach_add = None
        _foreach_sub = torch.ops.aten._foreach_sub.List([arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1], [arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1])
        getitem_6 = _foreach_sub[0]
        getitem_7 = _foreach_sub[1]
        getitem_8 = _foreach_sub[2]
        getitem_9 = _foreach_sub[3]
        getitem_10 = _foreach_sub[4]
        getitem_11 = _foreach_sub[5];  _foreach_sub = None
        _foreach_mul = torch.ops.aten._foreach_mul.Scalar([getitem_6, getitem_7, getitem_8, getitem_9, getitem_10, getitem_11], 0.09999999999999998);  getitem_6 = getitem_7 = getitem_8 = getitem_9 = getitem_10 = getitem_11 = None
        getitem_12 = _foreach_mul[0]
        getitem_13 = _foreach_mul[1]
        getitem_14 = _foreach_mul[2]
        getitem_15 = _foreach_mul[3]
        getitem_16 = _foreach_mul[4]
        getitem_17 = _foreach_mul[5];  _foreach_mul = None
        _foreach_add_1 = torch.ops.aten._foreach_add.List([arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1], [getitem_12, getitem_13, getitem_14, getitem_15, getitem_16, getitem_17]);  getitem_12 = getitem_13 = getitem_14 = getitem_15 = getitem_16 = getitem_17 = None
        getitem_18 = _foreach_add_1[0]
        getitem_19 = _foreach_add_1[1]
        getitem_20 = _foreach_add_1[2]
        getitem_21 = _foreach_add_1[3]
        getitem_22 = _foreach_add_1[4]
        getitem_23 = _foreach_add_1[5];  _foreach_add_1 = None
        _foreach_mul_1 = torch.ops.aten._foreach_mul.Scalar([arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1], 0.999)
        getitem_24 = _foreach_mul_1[0]
        getitem_25 = _foreach_mul_1[1]
        getitem_26 = _foreach_mul_1[2]
        getitem_27 = _foreach_mul_1[3]
        getitem_28 = _foreach_mul_1[4]
        getitem_29 = _foreach_mul_1[5];  _foreach_mul_1 = None
        _foreach_mul_2 = torch.ops.aten._foreach_mul.List([arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1], [arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1]);  arg24_1 = arg25_1 = arg26_1 = arg27_1 = arg28_1 = arg29_1 = None
        getitem_30 = _foreach_mul_2[0]
        getitem_31 = _foreach_mul_2[1]
        getitem_32 = _foreach_mul_2[2]
        getitem_33 = _foreach_mul_2[3]
        getitem_34 = _foreach_mul_2[4]
        getitem_35 = _foreach_mul_2[5];  _foreach_mul_2 = None
        _foreach_add_2 = torch.ops.aten._foreach_add.List([getitem_24, getitem_25, getitem_26, getitem_27, getitem_28, getitem_29], [getitem_30, getitem_31, getitem_32, getitem_33, getitem_34, getitem_35], alpha = 0.0010000000000000009);  getitem_24 = getitem_25 = getitem_26 = getitem_27 = getitem_28 = getitem_29 = getitem_30 = getitem_31 = getitem_32 = getitem_33 = getitem_34 = getitem_35 = None
        getitem_36 = _foreach_add_2[0]
        getitem_37 = _foreach_add_2[1]
        getitem_38 = _foreach_add_2[2]
        getitem_39 = _foreach_add_2[3]
        getitem_40 = _foreach_add_2[4]
        getitem_41 = _foreach_add_2[5];  _foreach_add_2 = None
        _foreach_pow = torch.ops.aten._foreach_pow.ScalarAndTensor(0.9, [getitem, getitem_1, getitem_2, getitem_3, getitem_4, getitem_5])
        getitem_42 = _foreach_pow[0]
        getitem_43 = _foreach_pow[1]
        getitem_44 = _foreach_pow[2]
        getitem_45 = _foreach_pow[3]
        getitem_46 = _foreach_pow[4]
        getitem_47 = _foreach_pow[5];  _foreach_pow = None
        _foreach_pow_1 = torch.ops.aten._foreach_pow.ScalarAndTensor(0.999, [getitem, getitem_1, getitem_2, getitem_3, getitem_4, getitem_5])
        getitem_48 = _foreach_pow_1[0]
        getitem_49 = _foreach_pow_1[1]
        getitem_50 = _foreach_pow_1[2]
        getitem_51 = _foreach_pow_1[3]
        getitem_52 = _foreach_pow_1[4]
        getitem_53 = _foreach_pow_1[5];  _foreach_pow_1 = None
        _foreach_sub_1 = torch.ops.aten._foreach_sub.Scalar([getitem_42, getitem_43, getitem_44, getitem_45, getitem_46, getitem_47], 1);  getitem_42 = getitem_43 = getitem_44 = getitem_45 = getitem_46 = getitem_47 = None
        getitem_54 = _foreach_sub_1[0]
        getitem_55 = _foreach_sub_1[1]
        getitem_56 = _foreach_sub_1[2]
        getitem_57 = _foreach_sub_1[3]
        getitem_58 = _foreach_sub_1[4]
        getitem_59 = _foreach_sub_1[5];  _foreach_sub_1 = None
        _foreach_sub_2 = torch.ops.aten._foreach_sub.Scalar([getitem_48, getitem_49, getitem_50, getitem_51, getitem_52, getitem_53], 1);  getitem_48 = getitem_49 = getitem_50 = getitem_51 = getitem_52 = getitem_53 = None
        getitem_60 = _foreach_sub_2[0]
        getitem_61 = _foreach_sub_2[1]
        getitem_62 = _foreach_sub_2[2]
        getitem_63 = _foreach_sub_2[3]
        getitem_64 = _foreach_sub_2[4]
        getitem_65 = _foreach_sub_2[5];  _foreach_sub_2 = None
        _foreach_neg = torch.ops.aten._foreach_neg.default([getitem_60, getitem_61, getitem_62, getitem_63, getitem_64, getitem_65]);  getitem_60 = getitem_61 = getitem_62 = getitem_63 = getitem_64 = getitem_65 = None
        getitem_66 = _foreach_neg[0]
        getitem_67 = _foreach_neg[1]
        getitem_68 = _foreach_neg[2]
        getitem_69 = _foreach_neg[3]
        getitem_70 = _foreach_neg[4]
        getitem_71 = _foreach_neg[5];  _foreach_neg = None
        _foreach_div = torch.ops.aten._foreach_div.Scalar([getitem_54, getitem_55, getitem_56, getitem_57, getitem_58, getitem_59], 0.001);  getitem_54 = getitem_55 = getitem_56 = getitem_57 = getitem_58 = getitem_59 = None
        getitem_72 = _foreach_div[0]
        getitem_73 = _foreach_div[1]
        getitem_74 = _foreach_div[2]
        getitem_75 = _foreach_div[3]
        getitem_76 = _foreach_div[4]
        getitem_77 = _foreach_div[5];  _foreach_div = None
        _foreach_reciprocal = torch.ops.aten._foreach_reciprocal.default([getitem_72, getitem_73, getitem_74, getitem_75, getitem_76, getitem_77]);  getitem_72 = getitem_73 = getitem_74 = getitem_75 = getitem_76 = getitem_77 = None
        getitem_78 = _foreach_reciprocal[0]
        getitem_79 = _foreach_reciprocal[1]
        getitem_80 = _foreach_reciprocal[2]
        getitem_81 = _foreach_reciprocal[3]
        getitem_82 = _foreach_reciprocal[4]
        getitem_83 = _foreach_reciprocal[5];  _foreach_reciprocal = None
        _foreach_sqrt = torch.ops.aten._foreach_sqrt.default([getitem_66, getitem_67, getitem_68, getitem_69, getitem_70, getitem_71]);  getitem_66 = getitem_67 = getitem_68 = getitem_69 = getitem_70 = getitem_71 = None
        getitem_84 = _foreach_sqrt[0]
        getitem_85 = _foreach_sqrt[1]
        getitem_86 = _foreach_sqrt[2]
        getitem_87 = _foreach_sqrt[3]
        getitem_88 = _foreach_sqrt[4]
        getitem_89 = _foreach_sqrt[5];  _foreach_sqrt = None
        _foreach_sqrt_1 = torch.ops.aten._foreach_sqrt.default([getitem_36, getitem_37, getitem_38, getitem_39, getitem_40, getitem_41])
        getitem_90 = _foreach_sqrt_1[0]
        getitem_91 = _foreach_sqrt_1[1]
        getitem_92 = _foreach_sqrt_1[2]
        getitem_93 = _foreach_sqrt_1[3]
        getitem_94 = _foreach_sqrt_1[4]
        getitem_95 = _foreach_sqrt_1[5];  _foreach_sqrt_1 = None
        _foreach_div_1 = torch.ops.aten._foreach_div.List([getitem_90, getitem_91, getitem_92, getitem_93, getitem_94, getitem_95], [getitem_84, getitem_85, getitem_86, getitem_87, getitem_88, getitem_89]);  getitem_90 = getitem_91 = getitem_92 = getitem_93 = getitem_94 = getitem_95 = getitem_84 = getitem_85 = getitem_86 = getitem_87 = getitem_88 = getitem_89 = None
        getitem_96 = _foreach_div_1[0]
        getitem_97 = _foreach_div_1[1]
        getitem_98 = _foreach_div_1[2]
        getitem_99 = _foreach_div_1[3]
        getitem_100 = _foreach_div_1[4]
        getitem_101 = _foreach_div_1[5];  _foreach_div_1 = None
        _foreach_add_3 = torch.ops.aten._foreach_add.Scalar([getitem_96, getitem_97, getitem_98, getitem_99, getitem_100, getitem_101], 1e-08);  getitem_96 = getitem_97 = getitem_98 = getitem_99 = getitem_100 = getitem_101 = None
        getitem_102 = _foreach_add_3[0]
        getitem_103 = _foreach_add_3[1]
        getitem_104 = _foreach_add_3[2]
        getitem_105 = _foreach_add_3[3]
        getitem_106 = _foreach_add_3[4]
        getitem_107 = _foreach_add_3[5];  _foreach_add_3 = None
        _foreach_div_2 = torch.ops.aten._foreach_div.List([getitem_102, getitem_103, getitem_104, getitem_105, getitem_106, getitem_107], [getitem_78, getitem_79, getitem_80, getitem_81, getitem_82, getitem_83]);  getitem_102 = getitem_103 = getitem_104 = getitem_105 = getitem_106 = getitem_107 = getitem_78 = getitem_79 = getitem_80 = getitem_81 = getitem_82 = getitem_83 = None
        getitem_108 = _foreach_div_2[0]
        getitem_109 = _foreach_div_2[1]
        getitem_110 = _foreach_div_2[2]
        getitem_111 = _foreach_div_2[3]
        getitem_112 = _foreach_div_2[4]
        getitem_113 = _foreach_div_2[5];  _foreach_div_2 = None
        _foreach_div_3 = torch.ops.aten._foreach_div.List([getitem_18, getitem_19, getitem_20, getitem_21, getitem_22, getitem_23], [getitem_108, getitem_109, getitem_110, getitem_111, getitem_112, getitem_113]);  getitem_108 = getitem_109 = getitem_110 = getitem_111 = getitem_112 = getitem_113 = None
        getitem_114 = _foreach_div_3[0]
        getitem_115 = _foreach_div_3[1]
        getitem_116 = _foreach_div_3[2]
        getitem_117 = _foreach_div_3[3]
        getitem_118 = _foreach_div_3[4]
        getitem_119 = _foreach_div_3[5];  _foreach_div_3 = None
        _foreach_add_4 = torch.ops.aten._foreach_add.List([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1], [getitem_114, getitem_115, getitem_116, getitem_117, getitem_118, getitem_119]);  getitem_114 = getitem_115 = getitem_116 = getitem_117 = getitem_118 = getitem_119 = None
        getitem_120 = _foreach_add_4[0]
        getitem_121 = _foreach_add_4[1]
        getitem_122 = _foreach_add_4[2]
        getitem_123 = _foreach_add_4[3]
        getitem_124 = _foreach_add_4[4]
        getitem_125 = _foreach_add_4[5];  _foreach_add_4 = None
        copy_ = torch.ops.aten.copy_.default(arg0_1, getitem_120);  arg0_1 = getitem_120 = None
        copy__1 = torch.ops.aten.copy_.default(arg1_1, getitem_121);  arg1_1 = getitem_121 = None
        copy__2 = torch.ops.aten.copy_.default(arg2_1, getitem_122);  arg2_1 = getitem_122 = None
        copy__3 = torch.ops.aten.copy_.default(arg3_1, getitem_123);  arg3_1 = getitem_123 = None
        copy__4 = torch.ops.aten.copy_.default(arg4_1, getitem_124);  arg4_1 = getitem_124 = None
        copy__5 = torch.ops.aten.copy_.default(arg5_1, getitem_125);  arg5_1 = getitem_125 = None
        copy__6 = torch.ops.aten.copy_.default(arg6_1, getitem_18);  arg6_1 = getitem_18 = None
        copy__7 = torch.ops.aten.copy_.default(arg7_1, getitem_19);  arg7_1 = getitem_19 = None
        copy__8 = torch.ops.aten.copy_.default(arg8_1, getitem_20);  arg8_1 = getitem_20 = None
        copy__9 = torch.ops.aten.copy_.default(arg9_1, getitem_21);  arg9_1 = getitem_21 = None
        copy__10 = torch.ops.aten.copy_.default(arg10_1, getitem_22);  arg10_1 = getitem_22 = None
        copy__11 = torch.ops.aten.copy_.default(arg11_1, getitem_23);  arg11_1 = getitem_23 = None
        copy__12 = torch.ops.aten.copy_.default(arg12_1, getitem_36);  arg12_1 = getitem_36 = None
        copy__13 = torch.ops.aten.copy_.default(arg13_1, getitem_37);  arg13_1 = getitem_37 = None
        copy__14 = torch.ops.aten.copy_.default(arg14_1, getitem_38);  arg14_1 = getitem_38 = None
        copy__15 = torch.ops.aten.copy_.default(arg15_1, getitem_39);  arg15_1 = getitem_39 = None
        copy__16 = torch.ops.aten.copy_.default(arg16_1, getitem_40);  arg16_1 = getitem_40 = None
        copy__17 = torch.ops.aten.copy_.default(arg17_1, getitem_41);  arg17_1 = getitem_41 = None
        copy__18 = torch.ops.aten.copy_.default(arg18_1, getitem);  arg18_1 = getitem = None
        copy__19 = torch.ops.aten.copy_.default(arg19_1, getitem_1);  arg19_1 = getitem_1 = None
        copy__20 = torch.ops.aten.copy_.default(arg20_1, getitem_2);  arg20_1 = getitem_2 = None
        copy__21 = torch.ops.aten.copy_.default(arg21_1, getitem_3);  arg21_1 = getitem_3 = None
        copy__22 = torch.ops.aten.copy_.default(arg22_1, getitem_4);  arg22_1 = getitem_4 = None
        copy__23 = torch.ops.aten.copy_.default(arg23_1, getitem_5);  arg23_1 = getitem_5 = None
        return ()
        
def load_args(reader):
    buf0 = reader.storage(None, 1605632, device=device(type='cuda', index=0))
    reader.tensor(buf0, (512, 784), requires_grad=True, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf1, (512,), requires_grad=True, is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf2, (512, 512), requires_grad=True, is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf3, (512,), requires_grad=True, is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 20480, device=device(type='cuda', index=0))
    reader.tensor(buf4, (10, 512), requires_grad=True, is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 40, device=device(type='cuda', index=0))
    reader.tensor(buf5, (10,), requires_grad=True, is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 1605632, device=device(type='cuda', index=0))
    reader.tensor(buf6, (512, 784), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf7, (512,), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf8, (512, 512), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf9, (512,), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 20480, device=device(type='cuda', index=0))
    reader.tensor(buf10, (10, 512), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 40, device=device(type='cuda', index=0))
    reader.tensor(buf11, (10,), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 1605632, device=device(type='cuda', index=0))
    reader.tensor(buf12, (512, 784), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf13, (512,), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf14, (512, 512), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf15, (512,), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 20480, device=device(type='cuda', index=0))
    reader.tensor(buf16, (10, 512), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 40, device=device(type='cuda', index=0))
    reader.tensor(buf17, (10,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf18, (), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf19, (), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf20, (), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf21, (), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf22, (), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf23, (), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 1605632, device=device(type='cuda', index=0))
    reader.tensor(buf24, (512, 784), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf25, (512,), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf26, (512, 512), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf27, (512,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 20480, device=device(type='cuda', index=0))
    reader.tensor(buf28, (10, 512), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 40, device=device(type='cuda', index=0))
    reader.tensor(buf29, (10,), is_leaf=True)  # arg29_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
