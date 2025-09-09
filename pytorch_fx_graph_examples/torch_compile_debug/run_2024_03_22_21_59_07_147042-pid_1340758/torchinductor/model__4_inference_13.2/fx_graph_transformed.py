class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[512, 784]", arg1_1: "f32[512]", arg2_1: "f32[512, 512]", arg3_1: "f32[512]", arg4_1: "f32[10, 512]", arg5_1: "f32[10]", arg6_1: "f32[512, 784]", arg7_1: "f32[512]", arg8_1: "f32[512, 512]", arg9_1: "f32[512]", arg10_1: "f32[10, 512]", arg11_1: "f32[10]", arg12_1: "f32[512, 784]", arg13_1: "f32[512]", arg14_1: "f32[512, 512]", arg15_1: "f32[512]", arg16_1: "f32[10, 512]", arg17_1: "f32[10]", arg18_1: "f32[]", arg19_1: "f32[]", arg20_1: "f32[]", arg21_1: "f32[]", arg22_1: "f32[]", arg23_1: "f32[]", arg24_1: "f32[512, 784]", arg25_1: "f32[512]", arg26_1: "f32[512, 512]", arg27_1: "f32[512]", arg28_1: "f32[10, 512]", arg29_1: "f32[10]"):
        # File: /home/zhang402/.local/lib/python3.8/site-packages/torch/optim/adam.py:510, code: torch._foreach_add_(device_state_steps, 1)
        _foreach_add = torch.ops.aten._foreach_add_.Scalar([arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1], 1);  arg18_1 = arg19_1 = arg20_1 = arg21_1 = arg22_1 = arg23_1 = None
        getitem: "f32[]" = _foreach_add[0]
        getitem_1: "f32[]" = _foreach_add[1]
        getitem_2: "f32[]" = _foreach_add[2]
        getitem_3: "f32[]" = _foreach_add[3]
        getitem_4: "f32[]" = _foreach_add[4]
        getitem_5: "f32[]" = _foreach_add[5];  _foreach_add = None
        
        # File: /home/zhang402/.local/lib/python3.8/site-packages/torch/optim/adam.py:520, code: torch._foreach_lerp_(device_exp_avgs, device_grads, 1 - beta1)
        _foreach_sub = torch.ops.aten._foreach_sub.List([arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1], [arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1])
        getitem_6: "f32[512, 784]" = _foreach_sub[0]
        getitem_7: "f32[512]" = _foreach_sub[1]
        getitem_8: "f32[512, 512]" = _foreach_sub[2]
        getitem_9: "f32[512]" = _foreach_sub[3]
        getitem_10: "f32[10, 512]" = _foreach_sub[4]
        getitem_11: "f32[10]" = _foreach_sub[5];  _foreach_sub = None
        _foreach_mul = torch.ops.aten._foreach_mul.Scalar([getitem_6, getitem_7, getitem_8, getitem_9, getitem_10, getitem_11], 0.09999999999999998);  getitem_6 = getitem_7 = getitem_8 = getitem_9 = getitem_10 = getitem_11 = None
        getitem_12: "f32[512, 784]" = _foreach_mul[0]
        getitem_13: "f32[512]" = _foreach_mul[1]
        getitem_14: "f32[512, 512]" = _foreach_mul[2]
        getitem_15: "f32[512]" = _foreach_mul[3]
        getitem_16: "f32[10, 512]" = _foreach_mul[4]
        getitem_17: "f32[10]" = _foreach_mul[5];  _foreach_mul = None
        _foreach_add_1 = torch.ops.aten._foreach_add_.List([arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1], [getitem_12, getitem_13, getitem_14, getitem_15, getitem_16, getitem_17]);  arg6_1 = arg7_1 = arg8_1 = arg9_1 = arg10_1 = arg11_1 = getitem_12 = getitem_13 = getitem_14 = getitem_15 = getitem_16 = getitem_17 = None
        getitem_18: "f32[512, 784]" = _foreach_add_1[0]
        getitem_19: "f32[512]" = _foreach_add_1[1]
        getitem_20: "f32[512, 512]" = _foreach_add_1[2]
        getitem_21: "f32[512]" = _foreach_add_1[3]
        getitem_22: "f32[10, 512]" = _foreach_add_1[4]
        getitem_23: "f32[10]" = _foreach_add_1[5];  _foreach_add_1 = None
        
        # File: /home/zhang402/.local/lib/python3.8/site-packages/torch/optim/adam.py:522, code: torch._foreach_mul_(device_exp_avg_sqs, beta2)
        _foreach_mul_1 = torch.ops.aten._foreach_mul.Scalar([arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1], 0.999)
        getitem_24: "f32[512, 784]" = _foreach_mul_1[0]
        getitem_25: "f32[512]" = _foreach_mul_1[1]
        getitem_26: "f32[512, 512]" = _foreach_mul_1[2]
        getitem_27: "f32[512]" = _foreach_mul_1[3]
        getitem_28: "f32[10, 512]" = _foreach_mul_1[4]
        getitem_29: "f32[10]" = _foreach_mul_1[5];  _foreach_mul_1 = None
        
        # File: /home/zhang402/.local/lib/python3.8/site-packages/torch/optim/adam.py:523, code: torch._foreach_addcmul_(device_exp_avg_sqs, device_grads, device_grads, 1 - beta2)
        _foreach_mul_2 = torch.ops.aten._foreach_mul.List([arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1], [arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1]);  arg24_1 = arg25_1 = arg26_1 = arg27_1 = arg28_1 = arg29_1 = None
        getitem_30: "f32[512, 784]" = _foreach_mul_2[0]
        getitem_31: "f32[512]" = _foreach_mul_2[1]
        getitem_32: "f32[512, 512]" = _foreach_mul_2[2]
        getitem_33: "f32[512]" = _foreach_mul_2[3]
        getitem_34: "f32[10, 512]" = _foreach_mul_2[4]
        getitem_35: "f32[10]" = _foreach_mul_2[5];  _foreach_mul_2 = None
        _foreach_add_2 = torch.ops.aten._foreach_add.List([getitem_24, getitem_25, getitem_26, getitem_27, getitem_28, getitem_29], [getitem_30, getitem_31, getitem_32, getitem_33, getitem_34, getitem_35], alpha = 0.0010000000000000009);  getitem_24 = getitem_25 = getitem_26 = getitem_27 = getitem_28 = getitem_29 = getitem_30 = getitem_31 = getitem_32 = getitem_33 = getitem_34 = getitem_35 = None
        getitem_36: "f32[512, 784]" = _foreach_add_2[0]
        getitem_37: "f32[512]" = _foreach_add_2[1]
        getitem_38: "f32[512, 512]" = _foreach_add_2[2]
        getitem_39: "f32[512]" = _foreach_add_2[3]
        getitem_40: "f32[10, 512]" = _foreach_add_2[4]
        getitem_41: "f32[10]" = _foreach_add_2[5];  _foreach_add_2 = None
        
        # File: /home/zhang402/.local/lib/python3.8/site-packages/torch/optim/adam.py:529, code: bias_correction1 = torch._foreach_pow(beta1, device_state_steps)
        _foreach_pow = torch.ops.aten._foreach_pow.ScalarAndTensor(0.9, [getitem, getitem_1, getitem_2, getitem_3, getitem_4, getitem_5])
        getitem_42: "f32[]" = _foreach_pow[0]
        getitem_43: "f32[]" = _foreach_pow[1]
        getitem_44: "f32[]" = _foreach_pow[2]
        getitem_45: "f32[]" = _foreach_pow[3]
        getitem_46: "f32[]" = _foreach_pow[4]
        getitem_47: "f32[]" = _foreach_pow[5];  _foreach_pow = None
        
        # File: /home/zhang402/.local/lib/python3.8/site-packages/torch/optim/adam.py:530, code: bias_correction2 = torch._foreach_pow(beta2, device_state_steps)
        _foreach_pow_1 = torch.ops.aten._foreach_pow.ScalarAndTensor(0.999, [getitem, getitem_1, getitem_2, getitem_3, getitem_4, getitem_5]);  getitem = getitem_1 = getitem_2 = getitem_3 = getitem_4 = getitem_5 = None
        getitem_48: "f32[]" = _foreach_pow_1[0]
        getitem_49: "f32[]" = _foreach_pow_1[1]
        getitem_50: "f32[]" = _foreach_pow_1[2]
        getitem_51: "f32[]" = _foreach_pow_1[3]
        getitem_52: "f32[]" = _foreach_pow_1[4]
        getitem_53: "f32[]" = _foreach_pow_1[5];  _foreach_pow_1 = None
        
        # File: /home/zhang402/.local/lib/python3.8/site-packages/torch/optim/adam.py:532, code: torch._foreach_sub_(bias_correction1, 1)
        _foreach_sub_1 = torch.ops.aten._foreach_sub.Scalar([getitem_42, getitem_43, getitem_44, getitem_45, getitem_46, getitem_47], 1);  getitem_42 = getitem_43 = getitem_44 = getitem_45 = getitem_46 = getitem_47 = None
        getitem_54: "f32[]" = _foreach_sub_1[0]
        getitem_55: "f32[]" = _foreach_sub_1[1]
        getitem_56: "f32[]" = _foreach_sub_1[2]
        getitem_57: "f32[]" = _foreach_sub_1[3]
        getitem_58: "f32[]" = _foreach_sub_1[4]
        getitem_59: "f32[]" = _foreach_sub_1[5];  _foreach_sub_1 = None
        
        # File: /home/zhang402/.local/lib/python3.8/site-packages/torch/optim/adam.py:533, code: torch._foreach_sub_(bias_correction2, 1)
        _foreach_sub_2 = torch.ops.aten._foreach_sub.Scalar([getitem_48, getitem_49, getitem_50, getitem_51, getitem_52, getitem_53], 1);  getitem_48 = getitem_49 = getitem_50 = getitem_51 = getitem_52 = getitem_53 = None
        getitem_60: "f32[]" = _foreach_sub_2[0]
        getitem_61: "f32[]" = _foreach_sub_2[1]
        getitem_62: "f32[]" = _foreach_sub_2[2]
        getitem_63: "f32[]" = _foreach_sub_2[3]
        getitem_64: "f32[]" = _foreach_sub_2[4]
        getitem_65: "f32[]" = _foreach_sub_2[5];  _foreach_sub_2 = None
        
        # File: /home/zhang402/.local/lib/python3.8/site-packages/torch/optim/adam.py:535, code: torch._foreach_neg_(bias_correction2)
        _foreach_neg = torch.ops.aten._foreach_neg.default([getitem_60, getitem_61, getitem_62, getitem_63, getitem_64, getitem_65]);  getitem_60 = getitem_61 = getitem_62 = getitem_63 = getitem_64 = getitem_65 = None
        getitem_66: "f32[]" = _foreach_neg[0]
        getitem_67: "f32[]" = _foreach_neg[1]
        getitem_68: "f32[]" = _foreach_neg[2]
        getitem_69: "f32[]" = _foreach_neg[3]
        getitem_70: "f32[]" = _foreach_neg[4]
        getitem_71: "f32[]" = _foreach_neg[5];  _foreach_neg = None
        
        # File: /home/zhang402/.local/lib/python3.8/site-packages/torch/optim/adam.py:538, code: torch._foreach_div_(bias_correction1, lr)
        _foreach_div = torch.ops.aten._foreach_div.Scalar([getitem_54, getitem_55, getitem_56, getitem_57, getitem_58, getitem_59], 0.001);  getitem_54 = getitem_55 = getitem_56 = getitem_57 = getitem_58 = getitem_59 = None
        getitem_72: "f32[]" = _foreach_div[0]
        getitem_73: "f32[]" = _foreach_div[1]
        getitem_74: "f32[]" = _foreach_div[2]
        getitem_75: "f32[]" = _foreach_div[3]
        getitem_76: "f32[]" = _foreach_div[4]
        getitem_77: "f32[]" = _foreach_div[5];  _foreach_div = None
        
        # File: /home/zhang402/.local/lib/python3.8/site-packages/torch/optim/adam.py:539, code: torch._foreach_reciprocal_(bias_correction1)
        _foreach_reciprocal = torch.ops.aten._foreach_reciprocal.default([getitem_72, getitem_73, getitem_74, getitem_75, getitem_76, getitem_77]);  getitem_72 = getitem_73 = getitem_74 = getitem_75 = getitem_76 = getitem_77 = None
        getitem_78: "f32[]" = _foreach_reciprocal[0]
        getitem_79: "f32[]" = _foreach_reciprocal[1]
        getitem_80: "f32[]" = _foreach_reciprocal[2]
        getitem_81: "f32[]" = _foreach_reciprocal[3]
        getitem_82: "f32[]" = _foreach_reciprocal[4]
        getitem_83: "f32[]" = _foreach_reciprocal[5];  _foreach_reciprocal = None
        
        # File: /home/zhang402/.local/lib/python3.8/site-packages/torch/optim/adam.py:541, code: torch._foreach_sqrt_(bias_correction2)
        _foreach_sqrt = torch.ops.aten._foreach_sqrt.default([getitem_66, getitem_67, getitem_68, getitem_69, getitem_70, getitem_71]);  getitem_66 = getitem_67 = getitem_68 = getitem_69 = getitem_70 = getitem_71 = None
        getitem_84: "f32[]" = _foreach_sqrt[0]
        getitem_85: "f32[]" = _foreach_sqrt[1]
        getitem_86: "f32[]" = _foreach_sqrt[2]
        getitem_87: "f32[]" = _foreach_sqrt[3]
        getitem_88: "f32[]" = _foreach_sqrt[4]
        getitem_89: "f32[]" = _foreach_sqrt[5];  _foreach_sqrt = None
        
        # File: /home/zhang402/.local/lib/python3.8/site-packages/torch/optim/adam.py:556, code: exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)
        _foreach_sqrt_1 = torch.ops.aten._foreach_sqrt.default([getitem_36, getitem_37, getitem_38, getitem_39, getitem_40, getitem_41])
        getitem_90: "f32[512, 784]" = _foreach_sqrt_1[0]
        getitem_91: "f32[512]" = _foreach_sqrt_1[1]
        getitem_92: "f32[512, 512]" = _foreach_sqrt_1[2]
        getitem_93: "f32[512]" = _foreach_sqrt_1[3]
        getitem_94: "f32[10, 512]" = _foreach_sqrt_1[4]
        getitem_95: "f32[10]" = _foreach_sqrt_1[5];  _foreach_sqrt_1 = None
        
        # File: /home/zhang402/.local/lib/python3.8/site-packages/torch/optim/adam.py:558, code: torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
        _foreach_div_1 = torch.ops.aten._foreach_div.List([getitem_90, getitem_91, getitem_92, getitem_93, getitem_94, getitem_95], [getitem_84, getitem_85, getitem_86, getitem_87, getitem_88, getitem_89]);  getitem_90 = getitem_91 = getitem_92 = getitem_93 = getitem_94 = getitem_95 = getitem_84 = getitem_85 = getitem_86 = getitem_87 = getitem_88 = getitem_89 = None
        getitem_96: "f32[512, 784]" = _foreach_div_1[0]
        getitem_97: "f32[512]" = _foreach_div_1[1]
        getitem_98: "f32[512, 512]" = _foreach_div_1[2]
        getitem_99: "f32[512]" = _foreach_div_1[3]
        getitem_100: "f32[10, 512]" = _foreach_div_1[4]
        getitem_101: "f32[10]" = _foreach_div_1[5];  _foreach_div_1 = None
        
        # File: /home/zhang402/.local/lib/python3.8/site-packages/torch/optim/adam.py:559, code: torch._foreach_add_(exp_avg_sq_sqrt, eps)
        _foreach_add_3 = torch.ops.aten._foreach_add.Scalar([getitem_96, getitem_97, getitem_98, getitem_99, getitem_100, getitem_101], 1e-08);  getitem_96 = getitem_97 = getitem_98 = getitem_99 = getitem_100 = getitem_101 = None
        getitem_102: "f32[512, 784]" = _foreach_add_3[0]
        getitem_103: "f32[512]" = _foreach_add_3[1]
        getitem_104: "f32[512, 512]" = _foreach_add_3[2]
        getitem_105: "f32[512]" = _foreach_add_3[3]
        getitem_106: "f32[10, 512]" = _foreach_add_3[4]
        getitem_107: "f32[10]" = _foreach_add_3[5];  _foreach_add_3 = None
        
        # File: /home/zhang402/.local/lib/python3.8/site-packages/torch/optim/adam.py:560, code: torch._foreach_div_(exp_avg_sq_sqrt, step_size)
        _foreach_div_2 = torch.ops.aten._foreach_div.List([getitem_102, getitem_103, getitem_104, getitem_105, getitem_106, getitem_107], [getitem_78, getitem_79, getitem_80, getitem_81, getitem_82, getitem_83]);  getitem_102 = getitem_103 = getitem_104 = getitem_105 = getitem_106 = getitem_107 = getitem_78 = getitem_79 = getitem_80 = getitem_81 = getitem_82 = getitem_83 = None
        getitem_108: "f32[512, 784]" = _foreach_div_2[0]
        getitem_109: "f32[512]" = _foreach_div_2[1]
        getitem_110: "f32[512, 512]" = _foreach_div_2[2]
        getitem_111: "f32[512]" = _foreach_div_2[3]
        getitem_112: "f32[10, 512]" = _foreach_div_2[4]
        getitem_113: "f32[10]" = _foreach_div_2[5];  _foreach_div_2 = None
        
        # File: /home/zhang402/.local/lib/python3.8/site-packages/torch/optim/adam.py:563, code: torch._foreach_addcdiv_(device_params, device_exp_avgs, exp_avg_sq_sqrt)
        _foreach_div_3 = torch.ops.aten._foreach_div.List([getitem_18, getitem_19, getitem_20, getitem_21, getitem_22, getitem_23], [getitem_108, getitem_109, getitem_110, getitem_111, getitem_112, getitem_113]);  getitem_18 = getitem_19 = getitem_20 = getitem_21 = getitem_22 = getitem_23 = getitem_108 = getitem_109 = getitem_110 = getitem_111 = getitem_112 = getitem_113 = None
        getitem_114: "f32[512, 784]" = _foreach_div_3[0]
        getitem_115: "f32[512]" = _foreach_div_3[1]
        getitem_116: "f32[512, 512]" = _foreach_div_3[2]
        getitem_117: "f32[512]" = _foreach_div_3[3]
        getitem_118: "f32[10, 512]" = _foreach_div_3[4]
        getitem_119: "f32[10]" = _foreach_div_3[5];  _foreach_div_3 = None
        _foreach_add_4 = torch.ops.aten._foreach_add_.List([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1], [getitem_114, getitem_115, getitem_116, getitem_117, getitem_118, getitem_119]);  arg0_1 = arg1_1 = arg2_1 = arg3_1 = arg4_1 = arg5_1 = getitem_114 = getitem_115 = getitem_116 = getitem_117 = getitem_118 = getitem_119 = None
        getitem_120: "f32[512, 784]" = _foreach_add_4[0]
        getitem_121: "f32[512]" = _foreach_add_4[1]
        getitem_122: "f32[512, 512]" = _foreach_add_4[2]
        getitem_123: "f32[512]" = _foreach_add_4[3]
        getitem_124: "f32[10, 512]" = _foreach_add_4[4]
        getitem_125: "f32[10]" = _foreach_add_4[5];  _foreach_add_4 = None
        
        # No stacktrace found for following nodes
        copy__12: "f32[512, 784]" = torch.ops.aten.copy_.default(arg12_1, getitem_36);  arg12_1 = getitem_36 = None
        copy__13: "f32[512]" = torch.ops.aten.copy_.default(arg13_1, getitem_37);  arg13_1 = getitem_37 = None
        copy__14: "f32[512, 512]" = torch.ops.aten.copy_.default(arg14_1, getitem_38);  arg14_1 = getitem_38 = None
        copy__15: "f32[512]" = torch.ops.aten.copy_.default(arg15_1, getitem_39);  arg15_1 = getitem_39 = None
        copy__16: "f32[10, 512]" = torch.ops.aten.copy_.default(arg16_1, getitem_40);  arg16_1 = getitem_40 = None
        copy__17: "f32[10]" = torch.ops.aten.copy_.default(arg17_1, getitem_41);  arg17_1 = getitem_41 = None
        return ()
        