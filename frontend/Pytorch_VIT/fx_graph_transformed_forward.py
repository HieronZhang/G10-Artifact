class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[1, 1, 768]", primals_2: "f32[1, 50, 768]", primals_3: "f32[768, 3, 32, 32]", primals_4: "f32[768]", primals_5: "f32[768]", primals_6: "f32[768]", primals_7: "f32[2304, 768]", primals_8: "f32[2304]", primals_9: "f32[768, 768]", primals_10: "f32[768]", primals_11: "f32[768]", primals_12: "f32[768]", primals_13: "f32[3072, 768]", primals_14: "f32[3072]", primals_15: "f32[768, 3072]", primals_16: "f32[768]", primals_17: "f32[768]", primals_18: "f32[768]", primals_19: "f32[2304, 768]", primals_20: "f32[2304]", primals_21: "f32[768, 768]", primals_22: "f32[768]", primals_23: "f32[768]", primals_24: "f32[768]", primals_25: "f32[3072, 768]", primals_26: "f32[3072]", primals_27: "f32[768, 3072]", primals_28: "f32[768]", primals_29: "f32[768]", primals_30: "f32[768]", primals_31: "f32[2304, 768]", primals_32: "f32[2304]", primals_33: "f32[768, 768]", primals_34: "f32[768]", primals_35: "f32[768]", primals_36: "f32[768]", primals_37: "f32[3072, 768]", primals_38: "f32[3072]", primals_39: "f32[768, 3072]", primals_40: "f32[768]", primals_41: "f32[768]", primals_42: "f32[768]", primals_43: "f32[2304, 768]", primals_44: "f32[2304]", primals_45: "f32[768, 768]", primals_46: "f32[768]", primals_47: "f32[768]", primals_48: "f32[768]", primals_49: "f32[3072, 768]", primals_50: "f32[3072]", primals_51: "f32[768, 3072]", primals_52: "f32[768]", primals_53: "f32[768]", primals_54: "f32[768]", primals_55: "f32[2304, 768]", primals_56: "f32[2304]", primals_57: "f32[768, 768]", primals_58: "f32[768]", primals_59: "f32[768]", primals_60: "f32[768]", primals_61: "f32[3072, 768]", primals_62: "f32[3072]", primals_63: "f32[768, 3072]", primals_64: "f32[768]", primals_65: "f32[768]", primals_66: "f32[768]", primals_67: "f32[2304, 768]", primals_68: "f32[2304]", primals_69: "f32[768, 768]", primals_70: "f32[768]", primals_71: "f32[768]", primals_72: "f32[768]", primals_73: "f32[3072, 768]", primals_74: "f32[3072]", primals_75: "f32[768, 3072]", primals_76: "f32[768]", primals_77: "f32[768]", primals_78: "f32[768]", primals_79: "f32[2304, 768]", primals_80: "f32[2304]", primals_81: "f32[768, 768]", primals_82: "f32[768]", primals_83: "f32[768]", primals_84: "f32[768]", primals_85: "f32[3072, 768]", primals_86: "f32[3072]", primals_87: "f32[768, 3072]", primals_88: "f32[768]", primals_89: "f32[768]", primals_90: "f32[768]", primals_91: "f32[2304, 768]", primals_92: "f32[2304]", primals_93: "f32[768, 768]", primals_94: "f32[768]", primals_95: "f32[768]", primals_96: "f32[768]", primals_97: "f32[3072, 768]", primals_98: "f32[3072]", primals_99: "f32[768, 3072]", primals_100: "f32[768]", primals_101: "f32[768]", primals_102: "f32[768]", primals_103: "f32[2304, 768]", primals_104: "f32[2304]", primals_105: "f32[768, 768]", primals_106: "f32[768]", primals_107: "f32[768]", primals_108: "f32[768]", primals_109: "f32[3072, 768]", primals_110: "f32[3072]", primals_111: "f32[768, 3072]", primals_112: "f32[768]", primals_113: "f32[768]", primals_114: "f32[768]", primals_115: "f32[2304, 768]", primals_116: "f32[2304]", primals_117: "f32[768, 768]", primals_118: "f32[768]", primals_119: "f32[768]", primals_120: "f32[768]", primals_121: "f32[3072, 768]", primals_122: "f32[3072]", primals_123: "f32[768, 3072]", primals_124: "f32[768]", primals_125: "f32[768]", primals_126: "f32[768]", primals_127: "f32[2304, 768]", primals_128: "f32[2304]", primals_129: "f32[768, 768]", primals_130: "f32[768]", primals_131: "f32[768]", primals_132: "f32[768]", primals_133: "f32[3072, 768]", primals_134: "f32[3072]", primals_135: "f32[768, 3072]", primals_136: "f32[768]", primals_137: "f32[768]", primals_138: "f32[768]", primals_139: "f32[2304, 768]", primals_140: "f32[2304]", primals_141: "f32[768, 768]", primals_142: "f32[768]", primals_143: "f32[768]", primals_144: "f32[768]", primals_145: "f32[3072, 768]", primals_146: "f32[3072]", primals_147: "f32[768, 3072]", primals_148: "f32[768]", primals_149: "f32[768]", primals_150: "f32[768]", primals_151: "f32[1000, 768]", primals_152: "f32[1000]", primals_153: "f32[64, 3, 224, 224]"):
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:277, code: x = self.conv_proj(x)
        convolution: "f32[64, 768, 7, 7]" = torch.ops.aten.convolution.default(primals_153, primals_3, primals_4, [32, 32], [0, 0], [1, 1], False, [0, 0], 1);  primals_4 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:279, code: x = x.reshape(n, self.hidden_dim, n_h * n_w)
        view: "f32[64, 768, 49]" = torch.ops.aten.reshape.default(convolution, [64, 768, 49]);  convolution = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:285, code: x = x.permute(0, 2, 1)
        permute: "f32[64, 49, 768]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:295, code: batch_class_token = self.class_token.expand(n, -1, -1)
        expand: "f32[64, 1, 768]" = torch.ops.aten.expand.default(primals_1, [64, -1, -1]);  primals_1 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:296, code: x = torch.cat([batch_class_token, x], dim=1)
        cat: "f32[64, 50, 768]" = torch.ops.aten.cat.default([expand, permute], 1);  expand = permute = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:156, code: input = input + self.pos_embedding
        add: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(cat, primals_2);  cat = primals_2 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:112, code: x = self.ln_1(input)
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem: "f32[64, 50, 1]" = var_mean[0]
        getitem_1: "f32[64, 50, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[64, 50, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt: "f32[64, 50, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(add, getitem_1);  getitem_1 = None
        mul: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul, primals_5)
        add_2: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(mul_1, primals_6);  mul_1 = primals_6 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_1: "f32[50, 64, 768]" = torch.ops.aten.permute.default(add_2, [1, 0, 2]);  add_2 = None
        permute_2: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
        clone_1: "f32[50, 64, 768]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_1: "f32[3200, 768]" = torch.ops.aten.reshape.default(clone_1, [3200, 768]);  clone_1 = None
        mm: "f32[3200, 2304]" = torch.ops.aten.mm.default(view_1, permute_2)
        view_2: "f32[50, 64, 2304]" = torch.ops.aten.reshape.default(mm, [50, 64, 2304]);  mm = None
        add_3: "f32[50, 64, 2304]" = torch.ops.aten.add.Tensor(view_2, primals_8);  view_2 = primals_8 = None
        view_3: "f32[50, 64, 3, 768]" = torch.ops.aten.reshape.default(add_3, [50, 64, 3, 768]);  add_3 = None
        unsqueeze: "f32[1, 50, 64, 3, 768]" = torch.ops.aten.unsqueeze.default(view_3, 0);  view_3 = None
        permute_3: "f32[3, 50, 64, 1, 768]" = torch.ops.aten.permute.default(unsqueeze, [3, 1, 2, 0, 4]);  unsqueeze = None
        squeeze: "f32[3, 50, 64, 768]" = torch.ops.aten.squeeze.dim(permute_3, -2);  permute_3 = None
        clone_2: "f32[3, 50, 64, 768]" = torch.ops.aten.clone.default(squeeze, memory_format = torch.contiguous_format);  squeeze = None
        select: "f32[50, 64, 768]" = torch.ops.aten.select.int(clone_2, 0, 0)
        select_1: "f32[50, 64, 768]" = torch.ops.aten.select.int(clone_2, 0, 1)
        select_2: "f32[50, 64, 768]" = torch.ops.aten.select.int(clone_2, 0, 2);  clone_2 = None
        view_4: "f32[50, 768, 64]" = torch.ops.aten.reshape.default(select, [50, 768, 64]);  select = None
        permute_4: "f32[768, 50, 64]" = torch.ops.aten.permute.default(view_4, [1, 0, 2]);  view_4 = None
        view_5: "f32[50, 768, 64]" = torch.ops.aten.reshape.default(select_1, [50, 768, 64]);  select_1 = None
        permute_5: "f32[768, 50, 64]" = torch.ops.aten.permute.default(view_5, [1, 0, 2]);  view_5 = None
        view_6: "f32[50, 768, 64]" = torch.ops.aten.reshape.default(select_2, [50, 768, 64]);  select_2 = None
        permute_6: "f32[768, 50, 64]" = torch.ops.aten.permute.default(view_6, [1, 0, 2]);  view_6 = None
        view_7: "f32[64, 12, 50, 64]" = torch.ops.aten.reshape.default(permute_4, [64, 12, 50, 64]);  permute_4 = None
        view_8: "f32[64, 12, 50, 64]" = torch.ops.aten.reshape.default(permute_5, [64, 12, 50, 64]);  permute_5 = None
        view_9: "f32[64, 12, 50, 64]" = torch.ops.aten.reshape.default(permute_6, [64, 12, 50, 64]);  permute_6 = None
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(view_7, view_8, view_9, None, True)
        getitem_2: "f32[64, 12, 50, 64]" = _scaled_dot_product_efficient_attention[0]
        getitem_3: "f32[64, 12, 64]" = _scaled_dot_product_efficient_attention[1]
        getitem_4: "i64[]" = _scaled_dot_product_efficient_attention[2]
        getitem_5: "i64[]" = _scaled_dot_product_efficient_attention[3];  _scaled_dot_product_efficient_attention = None
        alias: "f32[64, 12, 50, 64]" = torch.ops.aten.alias.default(getitem_2)
        permute_7: "f32[50, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_2, [2, 0, 1, 3]);  getitem_2 = None
        clone_3: "f32[50, 64, 12, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_10: "f32[3200, 768]" = torch.ops.aten.reshape.default(clone_3, [3200, 768]);  clone_3 = None
        permute_8: "f32[768, 768]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
        
        # No stacktrace found for following nodes
        mm_default_23: "f32[3200, 768]" = torch.ops.aten.mm.default(view_10, permute_8)
        add_tensor_23: "f32[3200, 768]" = torch.ops.aten.add.Tensor(mm_default_23, primals_10);  mm_default_23 = primals_10 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        view_11: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(add_tensor_23, [50, 64, 768]);  add_tensor_23 = None
        permute_9: "f32[64, 50, 768]" = torch.ops.aten.permute.default(view_11, [1, 0, 2]);  view_11 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:115, code: x = x + input
        add_4: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(permute_9, add);  permute_9 = add = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:117, code: y = self.ln_2(x)
        clone_5: "f32[64, 50, 768]" = torch.ops.aten.clone.default(add_4, memory_format = torch.contiguous_format)
        var_mean_1 = torch.ops.aten.var_mean.correction(clone_5, [2], correction = 0, keepdim = True)
        getitem_6: "f32[64, 50, 1]" = var_mean_1[0]
        getitem_7: "f32[64, 50, 1]" = var_mean_1[1];  var_mean_1 = None
        add_5: "f32[64, 50, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-06);  getitem_6 = None
        rsqrt_1: "f32[64, 50, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        sub_1: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(clone_5, getitem_7);  clone_5 = getitem_7 = None
        mul_2: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_3: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_2, primals_11)
        add_6: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(mul_3, primals_12);  mul_3 = primals_12 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        view_12: "f32[3200, 768]" = torch.ops.aten.reshape.default(add_6, [3200, 768]);  add_6 = None
        permute_10: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
        addmm_1: "f32[3200, 3072]" = torch.ops.aten.addmm.default(primals_14, view_12, permute_10);  primals_14 = None
        view_13: "f32[64, 50, 3072]" = torch.ops.aten.reshape.default(addmm_1, [64, 50, 3072])
        mul_4: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_13, 0.5)
        mul_5: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_13, 0.7071067811865476);  view_13 = None
        erf: "f32[64, 50, 3072]" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
        add_7: "f32[64, 50, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_6: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(mul_4, add_7);  mul_4 = add_7 = None
        view_14: "f32[3200, 3072]" = torch.ops.aten.reshape.default(mul_6, [3200, 3072]);  mul_6 = None
        permute_11: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
        
        # No stacktrace found for following nodes
        mm_default_22: "f32[3200, 768]" = torch.ops.aten.mm.default(view_14, permute_11)
        add_tensor_22: "f32[3200, 768]" = torch.ops.aten.add.Tensor(mm_default_22, primals_16);  mm_default_22 = primals_16 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        view_15: "f32[64, 50, 768]" = torch.ops.aten.reshape.default(add_tensor_22, [64, 50, 768]);  add_tensor_22 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:119, code: return x + y
        add_8: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(add_4, view_15);  add_4 = view_15 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:112, code: x = self.ln_1(input)
        clone_8: "f32[64, 50, 768]" = torch.ops.aten.clone.default(add_8, memory_format = torch.contiguous_format)
        var_mean_2 = torch.ops.aten.var_mean.correction(clone_8, [2], correction = 0, keepdim = True)
        getitem_8: "f32[64, 50, 1]" = var_mean_2[0]
        getitem_9: "f32[64, 50, 1]" = var_mean_2[1];  var_mean_2 = None
        add_9: "f32[64, 50, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-06);  getitem_8 = None
        rsqrt_2: "f32[64, 50, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        sub_2: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(clone_8, getitem_9);  clone_8 = getitem_9 = None
        mul_7: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        mul_8: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_7, primals_17)
        add_10: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(mul_8, primals_18);  mul_8 = primals_18 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_12: "f32[50, 64, 768]" = torch.ops.aten.permute.default(add_10, [1, 0, 2]);  add_10 = None
        permute_13: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_19, [1, 0]);  primals_19 = None
        clone_9: "f32[50, 64, 768]" = torch.ops.aten.clone.default(permute_12, memory_format = torch.contiguous_format);  permute_12 = None
        view_16: "f32[3200, 768]" = torch.ops.aten.reshape.default(clone_9, [3200, 768]);  clone_9 = None
        mm_1: "f32[3200, 2304]" = torch.ops.aten.mm.default(view_16, permute_13)
        view_17: "f32[50, 64, 2304]" = torch.ops.aten.reshape.default(mm_1, [50, 64, 2304]);  mm_1 = None
        add_11: "f32[50, 64, 2304]" = torch.ops.aten.add.Tensor(view_17, primals_20);  view_17 = primals_20 = None
        view_18: "f32[50, 64, 3, 768]" = torch.ops.aten.reshape.default(add_11, [50, 64, 3, 768]);  add_11 = None
        unsqueeze_1: "f32[1, 50, 64, 3, 768]" = torch.ops.aten.unsqueeze.default(view_18, 0);  view_18 = None
        permute_14: "f32[3, 50, 64, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_1, [3, 1, 2, 0, 4]);  unsqueeze_1 = None
        squeeze_1: "f32[3, 50, 64, 768]" = torch.ops.aten.squeeze.dim(permute_14, -2);  permute_14 = None
        clone_10: "f32[3, 50, 64, 768]" = torch.ops.aten.clone.default(squeeze_1, memory_format = torch.contiguous_format);  squeeze_1 = None
        select_3: "f32[50, 64, 768]" = torch.ops.aten.select.int(clone_10, 0, 0)
        select_4: "f32[50, 64, 768]" = torch.ops.aten.select.int(clone_10, 0, 1)
        select_5: "f32[50, 64, 768]" = torch.ops.aten.select.int(clone_10, 0, 2);  clone_10 = None
        view_19: "f32[50, 768, 64]" = torch.ops.aten.reshape.default(select_3, [50, 768, 64]);  select_3 = None
        permute_15: "f32[768, 50, 64]" = torch.ops.aten.permute.default(view_19, [1, 0, 2]);  view_19 = None
        view_20: "f32[50, 768, 64]" = torch.ops.aten.reshape.default(select_4, [50, 768, 64]);  select_4 = None
        permute_16: "f32[768, 50, 64]" = torch.ops.aten.permute.default(view_20, [1, 0, 2]);  view_20 = None
        view_21: "f32[50, 768, 64]" = torch.ops.aten.reshape.default(select_5, [50, 768, 64]);  select_5 = None
        permute_17: "f32[768, 50, 64]" = torch.ops.aten.permute.default(view_21, [1, 0, 2]);  view_21 = None
        view_22: "f32[64, 12, 50, 64]" = torch.ops.aten.reshape.default(permute_15, [64, 12, 50, 64]);  permute_15 = None
        view_23: "f32[64, 12, 50, 64]" = torch.ops.aten.reshape.default(permute_16, [64, 12, 50, 64]);  permute_16 = None
        view_24: "f32[64, 12, 50, 64]" = torch.ops.aten.reshape.default(permute_17, [64, 12, 50, 64]);  permute_17 = None
        _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(view_22, view_23, view_24, None, True)
        getitem_10: "f32[64, 12, 50, 64]" = _scaled_dot_product_efficient_attention_1[0]
        getitem_11: "f32[64, 12, 64]" = _scaled_dot_product_efficient_attention_1[1]
        getitem_12: "i64[]" = _scaled_dot_product_efficient_attention_1[2]
        getitem_13: "i64[]" = _scaled_dot_product_efficient_attention_1[3];  _scaled_dot_product_efficient_attention_1 = None
        alias_1: "f32[64, 12, 50, 64]" = torch.ops.aten.alias.default(getitem_10)
        permute_18: "f32[50, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_10, [2, 0, 1, 3]);  getitem_10 = None
        clone_11: "f32[50, 64, 12, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_25: "f32[3200, 768]" = torch.ops.aten.reshape.default(clone_11, [3200, 768]);  clone_11 = None
        permute_19: "f32[768, 768]" = torch.ops.aten.permute.default(primals_21, [1, 0]);  primals_21 = None
        
        # No stacktrace found for following nodes
        mm_default_21: "f32[3200, 768]" = torch.ops.aten.mm.default(view_25, permute_19)
        add_tensor_21: "f32[3200, 768]" = torch.ops.aten.add.Tensor(mm_default_21, primals_22);  mm_default_21 = primals_22 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        view_26: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(add_tensor_21, [50, 64, 768]);  add_tensor_21 = None
        permute_20: "f32[64, 50, 768]" = torch.ops.aten.permute.default(view_26, [1, 0, 2]);  view_26 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:115, code: x = x + input
        add_12: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(permute_20, add_8);  permute_20 = add_8 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:117, code: y = self.ln_2(x)
        clone_13: "f32[64, 50, 768]" = torch.ops.aten.clone.default(add_12, memory_format = torch.contiguous_format)
        var_mean_3 = torch.ops.aten.var_mean.correction(clone_13, [2], correction = 0, keepdim = True)
        getitem_14: "f32[64, 50, 1]" = var_mean_3[0]
        getitem_15: "f32[64, 50, 1]" = var_mean_3[1];  var_mean_3 = None
        add_13: "f32[64, 50, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-06);  getitem_14 = None
        rsqrt_3: "f32[64, 50, 1]" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        sub_3: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(clone_13, getitem_15);  clone_13 = getitem_15 = None
        mul_9: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        mul_10: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_9, primals_23)
        add_14: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(mul_10, primals_24);  mul_10 = primals_24 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        view_27: "f32[3200, 768]" = torch.ops.aten.reshape.default(add_14, [3200, 768]);  add_14 = None
        permute_21: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_25, [1, 0]);  primals_25 = None
        addmm_4: "f32[3200, 3072]" = torch.ops.aten.addmm.default(primals_26, view_27, permute_21);  primals_26 = None
        view_28: "f32[64, 50, 3072]" = torch.ops.aten.reshape.default(addmm_4, [64, 50, 3072])
        mul_11: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_28, 0.5)
        mul_12: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_28, 0.7071067811865476);  view_28 = None
        erf_1: "f32[64, 50, 3072]" = torch.ops.aten.erf.default(mul_12);  mul_12 = None
        add_15: "f32[64, 50, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_13: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(mul_11, add_15);  mul_11 = add_15 = None
        view_29: "f32[3200, 3072]" = torch.ops.aten.reshape.default(mul_13, [3200, 3072]);  mul_13 = None
        permute_22: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_27, [1, 0]);  primals_27 = None
        
        # No stacktrace found for following nodes
        mm_default_20: "f32[3200, 768]" = torch.ops.aten.mm.default(view_29, permute_22)
        add_tensor_20: "f32[3200, 768]" = torch.ops.aten.add.Tensor(mm_default_20, primals_28);  mm_default_20 = primals_28 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        view_30: "f32[64, 50, 768]" = torch.ops.aten.reshape.default(add_tensor_20, [64, 50, 768]);  add_tensor_20 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:119, code: return x + y
        add_16: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(add_12, view_30);  add_12 = view_30 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:112, code: x = self.ln_1(input)
        clone_16: "f32[64, 50, 768]" = torch.ops.aten.clone.default(add_16, memory_format = torch.contiguous_format)
        var_mean_4 = torch.ops.aten.var_mean.correction(clone_16, [2], correction = 0, keepdim = True)
        getitem_16: "f32[64, 50, 1]" = var_mean_4[0]
        getitem_17: "f32[64, 50, 1]" = var_mean_4[1];  var_mean_4 = None
        add_17: "f32[64, 50, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-06);  getitem_16 = None
        rsqrt_4: "f32[64, 50, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        sub_4: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(clone_16, getitem_17);  clone_16 = getitem_17 = None
        mul_14: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        mul_15: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_14, primals_29)
        add_18: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(mul_15, primals_30);  mul_15 = primals_30 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_23: "f32[50, 64, 768]" = torch.ops.aten.permute.default(add_18, [1, 0, 2]);  add_18 = None
        permute_24: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_31, [1, 0]);  primals_31 = None
        clone_17: "f32[50, 64, 768]" = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
        view_31: "f32[3200, 768]" = torch.ops.aten.reshape.default(clone_17, [3200, 768]);  clone_17 = None
        mm_2: "f32[3200, 2304]" = torch.ops.aten.mm.default(view_31, permute_24)
        view_32: "f32[50, 64, 2304]" = torch.ops.aten.reshape.default(mm_2, [50, 64, 2304]);  mm_2 = None
        add_19: "f32[50, 64, 2304]" = torch.ops.aten.add.Tensor(view_32, primals_32);  view_32 = primals_32 = None
        view_33: "f32[50, 64, 3, 768]" = torch.ops.aten.reshape.default(add_19, [50, 64, 3, 768]);  add_19 = None
        unsqueeze_2: "f32[1, 50, 64, 3, 768]" = torch.ops.aten.unsqueeze.default(view_33, 0);  view_33 = None
        permute_25: "f32[3, 50, 64, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_2, [3, 1, 2, 0, 4]);  unsqueeze_2 = None
        squeeze_2: "f32[3, 50, 64, 768]" = torch.ops.aten.squeeze.dim(permute_25, -2);  permute_25 = None
        clone_18: "f32[3, 50, 64, 768]" = torch.ops.aten.clone.default(squeeze_2, memory_format = torch.contiguous_format);  squeeze_2 = None
        select_6: "f32[50, 64, 768]" = torch.ops.aten.select.int(clone_18, 0, 0)
        select_7: "f32[50, 64, 768]" = torch.ops.aten.select.int(clone_18, 0, 1)
        select_8: "f32[50, 64, 768]" = torch.ops.aten.select.int(clone_18, 0, 2);  clone_18 = None
        view_34: "f32[50, 768, 64]" = torch.ops.aten.reshape.default(select_6, [50, 768, 64]);  select_6 = None
        permute_26: "f32[768, 50, 64]" = torch.ops.aten.permute.default(view_34, [1, 0, 2]);  view_34 = None
        view_35: "f32[50, 768, 64]" = torch.ops.aten.reshape.default(select_7, [50, 768, 64]);  select_7 = None
        permute_27: "f32[768, 50, 64]" = torch.ops.aten.permute.default(view_35, [1, 0, 2]);  view_35 = None
        view_36: "f32[50, 768, 64]" = torch.ops.aten.reshape.default(select_8, [50, 768, 64]);  select_8 = None
        permute_28: "f32[768, 50, 64]" = torch.ops.aten.permute.default(view_36, [1, 0, 2]);  view_36 = None
        view_37: "f32[64, 12, 50, 64]" = torch.ops.aten.reshape.default(permute_26, [64, 12, 50, 64]);  permute_26 = None
        view_38: "f32[64, 12, 50, 64]" = torch.ops.aten.reshape.default(permute_27, [64, 12, 50, 64]);  permute_27 = None
        view_39: "f32[64, 12, 50, 64]" = torch.ops.aten.reshape.default(permute_28, [64, 12, 50, 64]);  permute_28 = None
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(view_37, view_38, view_39, None, True)
        getitem_18: "f32[64, 12, 50, 64]" = _scaled_dot_product_efficient_attention_2[0]
        getitem_19: "f32[64, 12, 64]" = _scaled_dot_product_efficient_attention_2[1]
        getitem_20: "i64[]" = _scaled_dot_product_efficient_attention_2[2]
        getitem_21: "i64[]" = _scaled_dot_product_efficient_attention_2[3];  _scaled_dot_product_efficient_attention_2 = None
        alias_2: "f32[64, 12, 50, 64]" = torch.ops.aten.alias.default(getitem_18)
        permute_29: "f32[50, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_18, [2, 0, 1, 3]);  getitem_18 = None
        clone_19: "f32[50, 64, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_40: "f32[3200, 768]" = torch.ops.aten.reshape.default(clone_19, [3200, 768]);  clone_19 = None
        permute_30: "f32[768, 768]" = torch.ops.aten.permute.default(primals_33, [1, 0]);  primals_33 = None
        
        # No stacktrace found for following nodes
        mm_default_19: "f32[3200, 768]" = torch.ops.aten.mm.default(view_40, permute_30)
        add_tensor_19: "f32[3200, 768]" = torch.ops.aten.add.Tensor(mm_default_19, primals_34);  mm_default_19 = primals_34 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        view_41: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(add_tensor_19, [50, 64, 768]);  add_tensor_19 = None
        permute_31: "f32[64, 50, 768]" = torch.ops.aten.permute.default(view_41, [1, 0, 2]);  view_41 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:115, code: x = x + input
        add_20: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(permute_31, add_16);  permute_31 = add_16 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:117, code: y = self.ln_2(x)
        clone_21: "f32[64, 50, 768]" = torch.ops.aten.clone.default(add_20, memory_format = torch.contiguous_format)
        var_mean_5 = torch.ops.aten.var_mean.correction(clone_21, [2], correction = 0, keepdim = True)
        getitem_22: "f32[64, 50, 1]" = var_mean_5[0]
        getitem_23: "f32[64, 50, 1]" = var_mean_5[1];  var_mean_5 = None
        add_21: "f32[64, 50, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-06);  getitem_22 = None
        rsqrt_5: "f32[64, 50, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        sub_5: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(clone_21, getitem_23);  clone_21 = getitem_23 = None
        mul_16: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        mul_17: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_16, primals_35)
        add_22: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(mul_17, primals_36);  mul_17 = primals_36 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        view_42: "f32[3200, 768]" = torch.ops.aten.reshape.default(add_22, [3200, 768]);  add_22 = None
        permute_32: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_37, [1, 0]);  primals_37 = None
        addmm_7: "f32[3200, 3072]" = torch.ops.aten.addmm.default(primals_38, view_42, permute_32);  primals_38 = None
        view_43: "f32[64, 50, 3072]" = torch.ops.aten.reshape.default(addmm_7, [64, 50, 3072])
        mul_18: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_43, 0.5)
        mul_19: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_43, 0.7071067811865476);  view_43 = None
        erf_2: "f32[64, 50, 3072]" = torch.ops.aten.erf.default(mul_19);  mul_19 = None
        add_23: "f32[64, 50, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_20: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(mul_18, add_23);  mul_18 = add_23 = None
        view_44: "f32[3200, 3072]" = torch.ops.aten.reshape.default(mul_20, [3200, 3072]);  mul_20 = None
        permute_33: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_39, [1, 0]);  primals_39 = None
        
        # No stacktrace found for following nodes
        mm_default_18: "f32[3200, 768]" = torch.ops.aten.mm.default(view_44, permute_33)
        add_tensor_18: "f32[3200, 768]" = torch.ops.aten.add.Tensor(mm_default_18, primals_40);  mm_default_18 = primals_40 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        view_45: "f32[64, 50, 768]" = torch.ops.aten.reshape.default(add_tensor_18, [64, 50, 768]);  add_tensor_18 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:119, code: return x + y
        add_24: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(add_20, view_45);  add_20 = view_45 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:112, code: x = self.ln_1(input)
        clone_24: "f32[64, 50, 768]" = torch.ops.aten.clone.default(add_24, memory_format = torch.contiguous_format)
        var_mean_6 = torch.ops.aten.var_mean.correction(clone_24, [2], correction = 0, keepdim = True)
        getitem_24: "f32[64, 50, 1]" = var_mean_6[0]
        getitem_25: "f32[64, 50, 1]" = var_mean_6[1];  var_mean_6 = None
        add_25: "f32[64, 50, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-06);  getitem_24 = None
        rsqrt_6: "f32[64, 50, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        sub_6: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(clone_24, getitem_25);  clone_24 = getitem_25 = None
        mul_21: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
        mul_22: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_21, primals_41)
        add_26: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(mul_22, primals_42);  mul_22 = primals_42 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_34: "f32[50, 64, 768]" = torch.ops.aten.permute.default(add_26, [1, 0, 2]);  add_26 = None
        permute_35: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_43, [1, 0]);  primals_43 = None
        clone_25: "f32[50, 64, 768]" = torch.ops.aten.clone.default(permute_34, memory_format = torch.contiguous_format);  permute_34 = None
        view_46: "f32[3200, 768]" = torch.ops.aten.reshape.default(clone_25, [3200, 768]);  clone_25 = None
        mm_3: "f32[3200, 2304]" = torch.ops.aten.mm.default(view_46, permute_35)
        view_47: "f32[50, 64, 2304]" = torch.ops.aten.reshape.default(mm_3, [50, 64, 2304]);  mm_3 = None
        add_27: "f32[50, 64, 2304]" = torch.ops.aten.add.Tensor(view_47, primals_44);  view_47 = primals_44 = None
        view_48: "f32[50, 64, 3, 768]" = torch.ops.aten.reshape.default(add_27, [50, 64, 3, 768]);  add_27 = None
        unsqueeze_3: "f32[1, 50, 64, 3, 768]" = torch.ops.aten.unsqueeze.default(view_48, 0);  view_48 = None
        permute_36: "f32[3, 50, 64, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_3, [3, 1, 2, 0, 4]);  unsqueeze_3 = None
        squeeze_3: "f32[3, 50, 64, 768]" = torch.ops.aten.squeeze.dim(permute_36, -2);  permute_36 = None
        clone_26: "f32[3, 50, 64, 768]" = torch.ops.aten.clone.default(squeeze_3, memory_format = torch.contiguous_format);  squeeze_3 = None
        select_9: "f32[50, 64, 768]" = torch.ops.aten.select.int(clone_26, 0, 0)
        select_10: "f32[50, 64, 768]" = torch.ops.aten.select.int(clone_26, 0, 1)
        select_11: "f32[50, 64, 768]" = torch.ops.aten.select.int(clone_26, 0, 2);  clone_26 = None
        view_49: "f32[50, 768, 64]" = torch.ops.aten.reshape.default(select_9, [50, 768, 64]);  select_9 = None
        permute_37: "f32[768, 50, 64]" = torch.ops.aten.permute.default(view_49, [1, 0, 2]);  view_49 = None
        view_50: "f32[50, 768, 64]" = torch.ops.aten.reshape.default(select_10, [50, 768, 64]);  select_10 = None
        permute_38: "f32[768, 50, 64]" = torch.ops.aten.permute.default(view_50, [1, 0, 2]);  view_50 = None
        view_51: "f32[50, 768, 64]" = torch.ops.aten.reshape.default(select_11, [50, 768, 64]);  select_11 = None
        permute_39: "f32[768, 50, 64]" = torch.ops.aten.permute.default(view_51, [1, 0, 2]);  view_51 = None
        view_52: "f32[64, 12, 50, 64]" = torch.ops.aten.reshape.default(permute_37, [64, 12, 50, 64]);  permute_37 = None
        view_53: "f32[64, 12, 50, 64]" = torch.ops.aten.reshape.default(permute_38, [64, 12, 50, 64]);  permute_38 = None
        view_54: "f32[64, 12, 50, 64]" = torch.ops.aten.reshape.default(permute_39, [64, 12, 50, 64]);  permute_39 = None
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(view_52, view_53, view_54, None, True)
        getitem_26: "f32[64, 12, 50, 64]" = _scaled_dot_product_efficient_attention_3[0]
        getitem_27: "f32[64, 12, 64]" = _scaled_dot_product_efficient_attention_3[1]
        getitem_28: "i64[]" = _scaled_dot_product_efficient_attention_3[2]
        getitem_29: "i64[]" = _scaled_dot_product_efficient_attention_3[3];  _scaled_dot_product_efficient_attention_3 = None
        alias_3: "f32[64, 12, 50, 64]" = torch.ops.aten.alias.default(getitem_26)
        permute_40: "f32[50, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_26, [2, 0, 1, 3]);  getitem_26 = None
        clone_27: "f32[50, 64, 12, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_55: "f32[3200, 768]" = torch.ops.aten.reshape.default(clone_27, [3200, 768]);  clone_27 = None
        permute_41: "f32[768, 768]" = torch.ops.aten.permute.default(primals_45, [1, 0]);  primals_45 = None
        
        # No stacktrace found for following nodes
        mm_default_17: "f32[3200, 768]" = torch.ops.aten.mm.default(view_55, permute_41)
        add_tensor_17: "f32[3200, 768]" = torch.ops.aten.add.Tensor(mm_default_17, primals_46);  mm_default_17 = primals_46 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        view_56: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(add_tensor_17, [50, 64, 768]);  add_tensor_17 = None
        permute_42: "f32[64, 50, 768]" = torch.ops.aten.permute.default(view_56, [1, 0, 2]);  view_56 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:115, code: x = x + input
        add_28: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(permute_42, add_24);  permute_42 = add_24 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:117, code: y = self.ln_2(x)
        clone_29: "f32[64, 50, 768]" = torch.ops.aten.clone.default(add_28, memory_format = torch.contiguous_format)
        var_mean_7 = torch.ops.aten.var_mean.correction(clone_29, [2], correction = 0, keepdim = True)
        getitem_30: "f32[64, 50, 1]" = var_mean_7[0]
        getitem_31: "f32[64, 50, 1]" = var_mean_7[1];  var_mean_7 = None
        add_29: "f32[64, 50, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-06);  getitem_30 = None
        rsqrt_7: "f32[64, 50, 1]" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
        sub_7: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(clone_29, getitem_31);  clone_29 = getitem_31 = None
        mul_23: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
        mul_24: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_23, primals_47)
        add_30: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(mul_24, primals_48);  mul_24 = primals_48 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        view_57: "f32[3200, 768]" = torch.ops.aten.reshape.default(add_30, [3200, 768]);  add_30 = None
        permute_43: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_49, [1, 0]);  primals_49 = None
        addmm_10: "f32[3200, 3072]" = torch.ops.aten.addmm.default(primals_50, view_57, permute_43);  primals_50 = None
        view_58: "f32[64, 50, 3072]" = torch.ops.aten.reshape.default(addmm_10, [64, 50, 3072])
        mul_25: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_58, 0.5)
        mul_26: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_58, 0.7071067811865476);  view_58 = None
        erf_3: "f32[64, 50, 3072]" = torch.ops.aten.erf.default(mul_26);  mul_26 = None
        add_31: "f32[64, 50, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_27: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(mul_25, add_31);  mul_25 = add_31 = None
        view_59: "f32[3200, 3072]" = torch.ops.aten.reshape.default(mul_27, [3200, 3072]);  mul_27 = None
        permute_44: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_51, [1, 0]);  primals_51 = None
        
        # No stacktrace found for following nodes
        mm_default_16: "f32[3200, 768]" = torch.ops.aten.mm.default(view_59, permute_44)
        add_tensor_16: "f32[3200, 768]" = torch.ops.aten.add.Tensor(mm_default_16, primals_52);  mm_default_16 = primals_52 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        view_60: "f32[64, 50, 768]" = torch.ops.aten.reshape.default(add_tensor_16, [64, 50, 768]);  add_tensor_16 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:119, code: return x + y
        add_32: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(add_28, view_60);  add_28 = view_60 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:112, code: x = self.ln_1(input)
        clone_32: "f32[64, 50, 768]" = torch.ops.aten.clone.default(add_32, memory_format = torch.contiguous_format)
        var_mean_8 = torch.ops.aten.var_mean.correction(clone_32, [2], correction = 0, keepdim = True)
        getitem_32: "f32[64, 50, 1]" = var_mean_8[0]
        getitem_33: "f32[64, 50, 1]" = var_mean_8[1];  var_mean_8 = None
        add_33: "f32[64, 50, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-06);  getitem_32 = None
        rsqrt_8: "f32[64, 50, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
        sub_8: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(clone_32, getitem_33);  clone_32 = getitem_33 = None
        mul_28: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
        mul_29: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_28, primals_53)
        add_34: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(mul_29, primals_54);  mul_29 = primals_54 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_45: "f32[50, 64, 768]" = torch.ops.aten.permute.default(add_34, [1, 0, 2]);  add_34 = None
        permute_46: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_55, [1, 0]);  primals_55 = None
        clone_33: "f32[50, 64, 768]" = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
        view_61: "f32[3200, 768]" = torch.ops.aten.reshape.default(clone_33, [3200, 768]);  clone_33 = None
        mm_4: "f32[3200, 2304]" = torch.ops.aten.mm.default(view_61, permute_46)
        view_62: "f32[50, 64, 2304]" = torch.ops.aten.reshape.default(mm_4, [50, 64, 2304]);  mm_4 = None
        add_35: "f32[50, 64, 2304]" = torch.ops.aten.add.Tensor(view_62, primals_56);  view_62 = primals_56 = None
        view_63: "f32[50, 64, 3, 768]" = torch.ops.aten.reshape.default(add_35, [50, 64, 3, 768]);  add_35 = None
        unsqueeze_4: "f32[1, 50, 64, 3, 768]" = torch.ops.aten.unsqueeze.default(view_63, 0);  view_63 = None
        permute_47: "f32[3, 50, 64, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_4, [3, 1, 2, 0, 4]);  unsqueeze_4 = None
        squeeze_4: "f32[3, 50, 64, 768]" = torch.ops.aten.squeeze.dim(permute_47, -2);  permute_47 = None
        clone_34: "f32[3, 50, 64, 768]" = torch.ops.aten.clone.default(squeeze_4, memory_format = torch.contiguous_format);  squeeze_4 = None
        select_12: "f32[50, 64, 768]" = torch.ops.aten.select.int(clone_34, 0, 0)
        select_13: "f32[50, 64, 768]" = torch.ops.aten.select.int(clone_34, 0, 1)
        select_14: "f32[50, 64, 768]" = torch.ops.aten.select.int(clone_34, 0, 2);  clone_34 = None
        view_64: "f32[50, 768, 64]" = torch.ops.aten.reshape.default(select_12, [50, 768, 64]);  select_12 = None
        permute_48: "f32[768, 50, 64]" = torch.ops.aten.permute.default(view_64, [1, 0, 2]);  view_64 = None
        view_65: "f32[50, 768, 64]" = torch.ops.aten.reshape.default(select_13, [50, 768, 64]);  select_13 = None
        permute_49: "f32[768, 50, 64]" = torch.ops.aten.permute.default(view_65, [1, 0, 2]);  view_65 = None
        view_66: "f32[50, 768, 64]" = torch.ops.aten.reshape.default(select_14, [50, 768, 64]);  select_14 = None
        permute_50: "f32[768, 50, 64]" = torch.ops.aten.permute.default(view_66, [1, 0, 2]);  view_66 = None
        view_67: "f32[64, 12, 50, 64]" = torch.ops.aten.reshape.default(permute_48, [64, 12, 50, 64]);  permute_48 = None
        view_68: "f32[64, 12, 50, 64]" = torch.ops.aten.reshape.default(permute_49, [64, 12, 50, 64]);  permute_49 = None
        view_69: "f32[64, 12, 50, 64]" = torch.ops.aten.reshape.default(permute_50, [64, 12, 50, 64]);  permute_50 = None
        _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(view_67, view_68, view_69, None, True)
        getitem_34: "f32[64, 12, 50, 64]" = _scaled_dot_product_efficient_attention_4[0]
        getitem_35: "f32[64, 12, 64]" = _scaled_dot_product_efficient_attention_4[1]
        getitem_36: "i64[]" = _scaled_dot_product_efficient_attention_4[2]
        getitem_37: "i64[]" = _scaled_dot_product_efficient_attention_4[3];  _scaled_dot_product_efficient_attention_4 = None
        alias_4: "f32[64, 12, 50, 64]" = torch.ops.aten.alias.default(getitem_34)
        permute_51: "f32[50, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_34, [2, 0, 1, 3]);  getitem_34 = None
        clone_35: "f32[50, 64, 12, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
        view_70: "f32[3200, 768]" = torch.ops.aten.reshape.default(clone_35, [3200, 768]);  clone_35 = None
        permute_52: "f32[768, 768]" = torch.ops.aten.permute.default(primals_57, [1, 0]);  primals_57 = None
        
        # No stacktrace found for following nodes
        mm_default_15: "f32[3200, 768]" = torch.ops.aten.mm.default(view_70, permute_52)
        add_tensor_15: "f32[3200, 768]" = torch.ops.aten.add.Tensor(mm_default_15, primals_58);  mm_default_15 = primals_58 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        view_71: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(add_tensor_15, [50, 64, 768]);  add_tensor_15 = None
        permute_53: "f32[64, 50, 768]" = torch.ops.aten.permute.default(view_71, [1, 0, 2]);  view_71 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:115, code: x = x + input
        add_36: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(permute_53, add_32);  permute_53 = add_32 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:117, code: y = self.ln_2(x)
        clone_37: "f32[64, 50, 768]" = torch.ops.aten.clone.default(add_36, memory_format = torch.contiguous_format)
        var_mean_9 = torch.ops.aten.var_mean.correction(clone_37, [2], correction = 0, keepdim = True)
        getitem_38: "f32[64, 50, 1]" = var_mean_9[0]
        getitem_39: "f32[64, 50, 1]" = var_mean_9[1];  var_mean_9 = None
        add_37: "f32[64, 50, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-06);  getitem_38 = None
        rsqrt_9: "f32[64, 50, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        sub_9: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(clone_37, getitem_39);  clone_37 = getitem_39 = None
        mul_30: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
        mul_31: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_30, primals_59)
        add_38: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(mul_31, primals_60);  mul_31 = primals_60 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        view_72: "f32[3200, 768]" = torch.ops.aten.reshape.default(add_38, [3200, 768]);  add_38 = None
        permute_54: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_61, [1, 0]);  primals_61 = None
        addmm_13: "f32[3200, 3072]" = torch.ops.aten.addmm.default(primals_62, view_72, permute_54);  primals_62 = None
        view_73: "f32[64, 50, 3072]" = torch.ops.aten.reshape.default(addmm_13, [64, 50, 3072])
        mul_32: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_73, 0.5)
        mul_33: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_73, 0.7071067811865476);  view_73 = None
        erf_4: "f32[64, 50, 3072]" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
        add_39: "f32[64, 50, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_34: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(mul_32, add_39);  mul_32 = add_39 = None
        view_74: "f32[3200, 3072]" = torch.ops.aten.reshape.default(mul_34, [3200, 3072]);  mul_34 = None
        permute_55: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_63, [1, 0]);  primals_63 = None
        
        # No stacktrace found for following nodes
        mm_default_14: "f32[3200, 768]" = torch.ops.aten.mm.default(view_74, permute_55)
        add_tensor_14: "f32[3200, 768]" = torch.ops.aten.add.Tensor(mm_default_14, primals_64);  mm_default_14 = primals_64 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        view_75: "f32[64, 50, 768]" = torch.ops.aten.reshape.default(add_tensor_14, [64, 50, 768]);  add_tensor_14 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:119, code: return x + y
        add_40: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(add_36, view_75);  add_36 = view_75 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:112, code: x = self.ln_1(input)
        clone_40: "f32[64, 50, 768]" = torch.ops.aten.clone.default(add_40, memory_format = torch.contiguous_format)
        var_mean_10 = torch.ops.aten.var_mean.correction(clone_40, [2], correction = 0, keepdim = True)
        getitem_40: "f32[64, 50, 1]" = var_mean_10[0]
        getitem_41: "f32[64, 50, 1]" = var_mean_10[1];  var_mean_10 = None
        add_41: "f32[64, 50, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-06);  getitem_40 = None
        rsqrt_10: "f32[64, 50, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
        sub_10: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(clone_40, getitem_41);  clone_40 = getitem_41 = None
        mul_35: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
        mul_36: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_35, primals_65)
        add_42: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(mul_36, primals_66);  mul_36 = primals_66 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_56: "f32[50, 64, 768]" = torch.ops.aten.permute.default(add_42, [1, 0, 2]);  add_42 = None
        permute_57: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_67, [1, 0]);  primals_67 = None
        clone_41: "f32[50, 64, 768]" = torch.ops.aten.clone.default(permute_56, memory_format = torch.contiguous_format);  permute_56 = None
        view_76: "f32[3200, 768]" = torch.ops.aten.reshape.default(clone_41, [3200, 768]);  clone_41 = None
        mm_5: "f32[3200, 2304]" = torch.ops.aten.mm.default(view_76, permute_57)
        view_77: "f32[50, 64, 2304]" = torch.ops.aten.reshape.default(mm_5, [50, 64, 2304]);  mm_5 = None
        add_43: "f32[50, 64, 2304]" = torch.ops.aten.add.Tensor(view_77, primals_68);  view_77 = primals_68 = None
        view_78: "f32[50, 64, 3, 768]" = torch.ops.aten.reshape.default(add_43, [50, 64, 3, 768]);  add_43 = None
        unsqueeze_5: "f32[1, 50, 64, 3, 768]" = torch.ops.aten.unsqueeze.default(view_78, 0);  view_78 = None
        permute_58: "f32[3, 50, 64, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_5, [3, 1, 2, 0, 4]);  unsqueeze_5 = None
        squeeze_5: "f32[3, 50, 64, 768]" = torch.ops.aten.squeeze.dim(permute_58, -2);  permute_58 = None
        clone_42: "f32[3, 50, 64, 768]" = torch.ops.aten.clone.default(squeeze_5, memory_format = torch.contiguous_format);  squeeze_5 = None
        select_15: "f32[50, 64, 768]" = torch.ops.aten.select.int(clone_42, 0, 0)
        select_16: "f32[50, 64, 768]" = torch.ops.aten.select.int(clone_42, 0, 1)
        select_17: "f32[50, 64, 768]" = torch.ops.aten.select.int(clone_42, 0, 2);  clone_42 = None
        view_79: "f32[50, 768, 64]" = torch.ops.aten.reshape.default(select_15, [50, 768, 64]);  select_15 = None
        permute_59: "f32[768, 50, 64]" = torch.ops.aten.permute.default(view_79, [1, 0, 2]);  view_79 = None
        view_80: "f32[50, 768, 64]" = torch.ops.aten.reshape.default(select_16, [50, 768, 64]);  select_16 = None
        permute_60: "f32[768, 50, 64]" = torch.ops.aten.permute.default(view_80, [1, 0, 2]);  view_80 = None
        view_81: "f32[50, 768, 64]" = torch.ops.aten.reshape.default(select_17, [50, 768, 64]);  select_17 = None
        permute_61: "f32[768, 50, 64]" = torch.ops.aten.permute.default(view_81, [1, 0, 2]);  view_81 = None
        view_82: "f32[64, 12, 50, 64]" = torch.ops.aten.reshape.default(permute_59, [64, 12, 50, 64]);  permute_59 = None
        view_83: "f32[64, 12, 50, 64]" = torch.ops.aten.reshape.default(permute_60, [64, 12, 50, 64]);  permute_60 = None
        view_84: "f32[64, 12, 50, 64]" = torch.ops.aten.reshape.default(permute_61, [64, 12, 50, 64]);  permute_61 = None
        _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(view_82, view_83, view_84, None, True)
        getitem_42: "f32[64, 12, 50, 64]" = _scaled_dot_product_efficient_attention_5[0]
        getitem_43: "f32[64, 12, 64]" = _scaled_dot_product_efficient_attention_5[1]
        getitem_44: "i64[]" = _scaled_dot_product_efficient_attention_5[2]
        getitem_45: "i64[]" = _scaled_dot_product_efficient_attention_5[3];  _scaled_dot_product_efficient_attention_5 = None
        alias_5: "f32[64, 12, 50, 64]" = torch.ops.aten.alias.default(getitem_42)
        permute_62: "f32[50, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_42, [2, 0, 1, 3]);  getitem_42 = None
        clone_43: "f32[50, 64, 12, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_85: "f32[3200, 768]" = torch.ops.aten.reshape.default(clone_43, [3200, 768]);  clone_43 = None
        permute_63: "f32[768, 768]" = torch.ops.aten.permute.default(primals_69, [1, 0]);  primals_69 = None
        
        # No stacktrace found for following nodes
        mm_default_13: "f32[3200, 768]" = torch.ops.aten.mm.default(view_85, permute_63)
        add_tensor_13: "f32[3200, 768]" = torch.ops.aten.add.Tensor(mm_default_13, primals_70);  mm_default_13 = primals_70 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        view_86: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(add_tensor_13, [50, 64, 768]);  add_tensor_13 = None
        permute_64: "f32[64, 50, 768]" = torch.ops.aten.permute.default(view_86, [1, 0, 2]);  view_86 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:115, code: x = x + input
        add_44: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(permute_64, add_40);  permute_64 = add_40 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:117, code: y = self.ln_2(x)
        clone_45: "f32[64, 50, 768]" = torch.ops.aten.clone.default(add_44, memory_format = torch.contiguous_format)
        var_mean_11 = torch.ops.aten.var_mean.correction(clone_45, [2], correction = 0, keepdim = True)
        getitem_46: "f32[64, 50, 1]" = var_mean_11[0]
        getitem_47: "f32[64, 50, 1]" = var_mean_11[1];  var_mean_11 = None
        add_45: "f32[64, 50, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-06);  getitem_46 = None
        rsqrt_11: "f32[64, 50, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
        sub_11: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(clone_45, getitem_47);  clone_45 = getitem_47 = None
        mul_37: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
        mul_38: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_37, primals_71)
        add_46: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(mul_38, primals_72);  mul_38 = primals_72 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        view_87: "f32[3200, 768]" = torch.ops.aten.reshape.default(add_46, [3200, 768]);  add_46 = None
        permute_65: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_73, [1, 0]);  primals_73 = None
        addmm_16: "f32[3200, 3072]" = torch.ops.aten.addmm.default(primals_74, view_87, permute_65);  primals_74 = None
        view_88: "f32[64, 50, 3072]" = torch.ops.aten.reshape.default(addmm_16, [64, 50, 3072])
        mul_39: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_88, 0.5)
        mul_40: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_88, 0.7071067811865476);  view_88 = None
        erf_5: "f32[64, 50, 3072]" = torch.ops.aten.erf.default(mul_40);  mul_40 = None
        add_47: "f32[64, 50, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_41: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(mul_39, add_47);  mul_39 = add_47 = None
        view_89: "f32[3200, 3072]" = torch.ops.aten.reshape.default(mul_41, [3200, 3072]);  mul_41 = None
        permute_66: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_75, [1, 0]);  primals_75 = None
        
        # No stacktrace found for following nodes
        mm_default_12: "f32[3200, 768]" = torch.ops.aten.mm.default(view_89, permute_66)
        add_tensor_12: "f32[3200, 768]" = torch.ops.aten.add.Tensor(mm_default_12, primals_76);  mm_default_12 = primals_76 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        view_90: "f32[64, 50, 768]" = torch.ops.aten.reshape.default(add_tensor_12, [64, 50, 768]);  add_tensor_12 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:119, code: return x + y
        add_48: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(add_44, view_90);  add_44 = view_90 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:112, code: x = self.ln_1(input)
        clone_48: "f32[64, 50, 768]" = torch.ops.aten.clone.default(add_48, memory_format = torch.contiguous_format)
        var_mean_12 = torch.ops.aten.var_mean.correction(clone_48, [2], correction = 0, keepdim = True)
        getitem_48: "f32[64, 50, 1]" = var_mean_12[0]
        getitem_49: "f32[64, 50, 1]" = var_mean_12[1];  var_mean_12 = None
        add_49: "f32[64, 50, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-06);  getitem_48 = None
        rsqrt_12: "f32[64, 50, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        sub_12: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(clone_48, getitem_49);  clone_48 = getitem_49 = None
        mul_42: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
        mul_43: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_42, primals_77)
        add_50: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(mul_43, primals_78);  mul_43 = primals_78 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_67: "f32[50, 64, 768]" = torch.ops.aten.permute.default(add_50, [1, 0, 2]);  add_50 = None
        permute_68: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_79, [1, 0]);  primals_79 = None
        clone_49: "f32[50, 64, 768]" = torch.ops.aten.clone.default(permute_67, memory_format = torch.contiguous_format);  permute_67 = None
        view_91: "f32[3200, 768]" = torch.ops.aten.reshape.default(clone_49, [3200, 768]);  clone_49 = None
        mm_6: "f32[3200, 2304]" = torch.ops.aten.mm.default(view_91, permute_68)
        view_92: "f32[50, 64, 2304]" = torch.ops.aten.reshape.default(mm_6, [50, 64, 2304]);  mm_6 = None
        add_51: "f32[50, 64, 2304]" = torch.ops.aten.add.Tensor(view_92, primals_80);  view_92 = primals_80 = None
        view_93: "f32[50, 64, 3, 768]" = torch.ops.aten.reshape.default(add_51, [50, 64, 3, 768]);  add_51 = None
        unsqueeze_6: "f32[1, 50, 64, 3, 768]" = torch.ops.aten.unsqueeze.default(view_93, 0);  view_93 = None
        permute_69: "f32[3, 50, 64, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_6, [3, 1, 2, 0, 4]);  unsqueeze_6 = None
        squeeze_6: "f32[3, 50, 64, 768]" = torch.ops.aten.squeeze.dim(permute_69, -2);  permute_69 = None
        clone_50: "f32[3, 50, 64, 768]" = torch.ops.aten.clone.default(squeeze_6, memory_format = torch.contiguous_format);  squeeze_6 = None
        select_18: "f32[50, 64, 768]" = torch.ops.aten.select.int(clone_50, 0, 0)
        select_19: "f32[50, 64, 768]" = torch.ops.aten.select.int(clone_50, 0, 1)
        select_20: "f32[50, 64, 768]" = torch.ops.aten.select.int(clone_50, 0, 2);  clone_50 = None
        view_94: "f32[50, 768, 64]" = torch.ops.aten.reshape.default(select_18, [50, 768, 64]);  select_18 = None
        permute_70: "f32[768, 50, 64]" = torch.ops.aten.permute.default(view_94, [1, 0, 2]);  view_94 = None
        view_95: "f32[50, 768, 64]" = torch.ops.aten.reshape.default(select_19, [50, 768, 64]);  select_19 = None
        permute_71: "f32[768, 50, 64]" = torch.ops.aten.permute.default(view_95, [1, 0, 2]);  view_95 = None
        view_96: "f32[50, 768, 64]" = torch.ops.aten.reshape.default(select_20, [50, 768, 64]);  select_20 = None
        permute_72: "f32[768, 50, 64]" = torch.ops.aten.permute.default(view_96, [1, 0, 2]);  view_96 = None
        view_97: "f32[64, 12, 50, 64]" = torch.ops.aten.reshape.default(permute_70, [64, 12, 50, 64]);  permute_70 = None
        view_98: "f32[64, 12, 50, 64]" = torch.ops.aten.reshape.default(permute_71, [64, 12, 50, 64]);  permute_71 = None
        view_99: "f32[64, 12, 50, 64]" = torch.ops.aten.reshape.default(permute_72, [64, 12, 50, 64]);  permute_72 = None
        _scaled_dot_product_efficient_attention_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(view_97, view_98, view_99, None, True)
        getitem_50: "f32[64, 12, 50, 64]" = _scaled_dot_product_efficient_attention_6[0]
        getitem_51: "f32[64, 12, 64]" = _scaled_dot_product_efficient_attention_6[1]
        getitem_52: "i64[]" = _scaled_dot_product_efficient_attention_6[2]
        getitem_53: "i64[]" = _scaled_dot_product_efficient_attention_6[3];  _scaled_dot_product_efficient_attention_6 = None
        alias_6: "f32[64, 12, 50, 64]" = torch.ops.aten.alias.default(getitem_50)
        permute_73: "f32[50, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_50, [2, 0, 1, 3]);  getitem_50 = None
        clone_51: "f32[50, 64, 12, 64]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
        view_100: "f32[3200, 768]" = torch.ops.aten.reshape.default(clone_51, [3200, 768]);  clone_51 = None
        permute_74: "f32[768, 768]" = torch.ops.aten.permute.default(primals_81, [1, 0]);  primals_81 = None
        
        # No stacktrace found for following nodes
        mm_default_11: "f32[3200, 768]" = torch.ops.aten.mm.default(view_100, permute_74)
        add_tensor_11: "f32[3200, 768]" = torch.ops.aten.add.Tensor(mm_default_11, primals_82);  mm_default_11 = primals_82 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        view_101: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(add_tensor_11, [50, 64, 768]);  add_tensor_11 = None
        permute_75: "f32[64, 50, 768]" = torch.ops.aten.permute.default(view_101, [1, 0, 2]);  view_101 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:115, code: x = x + input
        add_52: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(permute_75, add_48);  permute_75 = add_48 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:117, code: y = self.ln_2(x)
        clone_53: "f32[64, 50, 768]" = torch.ops.aten.clone.default(add_52, memory_format = torch.contiguous_format)
        var_mean_13 = torch.ops.aten.var_mean.correction(clone_53, [2], correction = 0, keepdim = True)
        getitem_54: "f32[64, 50, 1]" = var_mean_13[0]
        getitem_55: "f32[64, 50, 1]" = var_mean_13[1];  var_mean_13 = None
        add_53: "f32[64, 50, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-06);  getitem_54 = None
        rsqrt_13: "f32[64, 50, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        sub_13: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(clone_53, getitem_55);  clone_53 = getitem_55 = None
        mul_44: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
        mul_45: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_44, primals_83)
        add_54: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(mul_45, primals_84);  mul_45 = primals_84 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        view_102: "f32[3200, 768]" = torch.ops.aten.reshape.default(add_54, [3200, 768]);  add_54 = None
        permute_76: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_85, [1, 0]);  primals_85 = None
        addmm_19: "f32[3200, 3072]" = torch.ops.aten.addmm.default(primals_86, view_102, permute_76);  primals_86 = None
        view_103: "f32[64, 50, 3072]" = torch.ops.aten.reshape.default(addmm_19, [64, 50, 3072])
        mul_46: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_103, 0.5)
        mul_47: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_103, 0.7071067811865476);  view_103 = None
        erf_6: "f32[64, 50, 3072]" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
        add_55: "f32[64, 50, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_48: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(mul_46, add_55);  mul_46 = add_55 = None
        view_104: "f32[3200, 3072]" = torch.ops.aten.reshape.default(mul_48, [3200, 3072]);  mul_48 = None
        permute_77: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_87, [1, 0]);  primals_87 = None
        
        # No stacktrace found for following nodes
        mm_default_10: "f32[3200, 768]" = torch.ops.aten.mm.default(view_104, permute_77)
        add_tensor_10: "f32[3200, 768]" = torch.ops.aten.add.Tensor(mm_default_10, primals_88);  mm_default_10 = primals_88 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        view_105: "f32[64, 50, 768]" = torch.ops.aten.reshape.default(add_tensor_10, [64, 50, 768]);  add_tensor_10 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:119, code: return x + y
        add_56: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(add_52, view_105);  add_52 = view_105 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:112, code: x = self.ln_1(input)
        clone_56: "f32[64, 50, 768]" = torch.ops.aten.clone.default(add_56, memory_format = torch.contiguous_format)
        var_mean_14 = torch.ops.aten.var_mean.correction(clone_56, [2], correction = 0, keepdim = True)
        getitem_56: "f32[64, 50, 1]" = var_mean_14[0]
        getitem_57: "f32[64, 50, 1]" = var_mean_14[1];  var_mean_14 = None
        add_57: "f32[64, 50, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-06);  getitem_56 = None
        rsqrt_14: "f32[64, 50, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
        sub_14: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(clone_56, getitem_57);  clone_56 = getitem_57 = None
        mul_49: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
        mul_50: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_49, primals_89)
        add_58: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(mul_50, primals_90);  mul_50 = primals_90 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_78: "f32[50, 64, 768]" = torch.ops.aten.permute.default(add_58, [1, 0, 2]);  add_58 = None
        permute_79: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_91, [1, 0]);  primals_91 = None
        clone_57: "f32[50, 64, 768]" = torch.ops.aten.clone.default(permute_78, memory_format = torch.contiguous_format);  permute_78 = None
        view_106: "f32[3200, 768]" = torch.ops.aten.reshape.default(clone_57, [3200, 768]);  clone_57 = None
        mm_7: "f32[3200, 2304]" = torch.ops.aten.mm.default(view_106, permute_79)
        view_107: "f32[50, 64, 2304]" = torch.ops.aten.reshape.default(mm_7, [50, 64, 2304]);  mm_7 = None
        add_59: "f32[50, 64, 2304]" = torch.ops.aten.add.Tensor(view_107, primals_92);  view_107 = primals_92 = None
        view_108: "f32[50, 64, 3, 768]" = torch.ops.aten.reshape.default(add_59, [50, 64, 3, 768]);  add_59 = None
        unsqueeze_7: "f32[1, 50, 64, 3, 768]" = torch.ops.aten.unsqueeze.default(view_108, 0);  view_108 = None
        permute_80: "f32[3, 50, 64, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_7, [3, 1, 2, 0, 4]);  unsqueeze_7 = None
        squeeze_7: "f32[3, 50, 64, 768]" = torch.ops.aten.squeeze.dim(permute_80, -2);  permute_80 = None
        clone_58: "f32[3, 50, 64, 768]" = torch.ops.aten.clone.default(squeeze_7, memory_format = torch.contiguous_format);  squeeze_7 = None
        select_21: "f32[50, 64, 768]" = torch.ops.aten.select.int(clone_58, 0, 0)
        select_22: "f32[50, 64, 768]" = torch.ops.aten.select.int(clone_58, 0, 1)
        select_23: "f32[50, 64, 768]" = torch.ops.aten.select.int(clone_58, 0, 2);  clone_58 = None
        view_109: "f32[50, 768, 64]" = torch.ops.aten.reshape.default(select_21, [50, 768, 64]);  select_21 = None
        permute_81: "f32[768, 50, 64]" = torch.ops.aten.permute.default(view_109, [1, 0, 2]);  view_109 = None
        view_110: "f32[50, 768, 64]" = torch.ops.aten.reshape.default(select_22, [50, 768, 64]);  select_22 = None
        permute_82: "f32[768, 50, 64]" = torch.ops.aten.permute.default(view_110, [1, 0, 2]);  view_110 = None
        view_111: "f32[50, 768, 64]" = torch.ops.aten.reshape.default(select_23, [50, 768, 64]);  select_23 = None
        permute_83: "f32[768, 50, 64]" = torch.ops.aten.permute.default(view_111, [1, 0, 2]);  view_111 = None
        view_112: "f32[64, 12, 50, 64]" = torch.ops.aten.reshape.default(permute_81, [64, 12, 50, 64]);  permute_81 = None
        view_113: "f32[64, 12, 50, 64]" = torch.ops.aten.reshape.default(permute_82, [64, 12, 50, 64]);  permute_82 = None
        view_114: "f32[64, 12, 50, 64]" = torch.ops.aten.reshape.default(permute_83, [64, 12, 50, 64]);  permute_83 = None
        _scaled_dot_product_efficient_attention_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(view_112, view_113, view_114, None, True)
        getitem_58: "f32[64, 12, 50, 64]" = _scaled_dot_product_efficient_attention_7[0]
        getitem_59: "f32[64, 12, 64]" = _scaled_dot_product_efficient_attention_7[1]
        getitem_60: "i64[]" = _scaled_dot_product_efficient_attention_7[2]
        getitem_61: "i64[]" = _scaled_dot_product_efficient_attention_7[3];  _scaled_dot_product_efficient_attention_7 = None
        alias_7: "f32[64, 12, 50, 64]" = torch.ops.aten.alias.default(getitem_58)
        permute_84: "f32[50, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_58, [2, 0, 1, 3]);  getitem_58 = None
        clone_59: "f32[50, 64, 12, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
        view_115: "f32[3200, 768]" = torch.ops.aten.reshape.default(clone_59, [3200, 768]);  clone_59 = None
        permute_85: "f32[768, 768]" = torch.ops.aten.permute.default(primals_93, [1, 0]);  primals_93 = None
        
        # No stacktrace found for following nodes
        mm_default_9: "f32[3200, 768]" = torch.ops.aten.mm.default(view_115, permute_85)
        add_tensor_9: "f32[3200, 768]" = torch.ops.aten.add.Tensor(mm_default_9, primals_94);  mm_default_9 = primals_94 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        view_116: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(add_tensor_9, [50, 64, 768]);  add_tensor_9 = None
        permute_86: "f32[64, 50, 768]" = torch.ops.aten.permute.default(view_116, [1, 0, 2]);  view_116 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:115, code: x = x + input
        add_60: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(permute_86, add_56);  permute_86 = add_56 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:117, code: y = self.ln_2(x)
        clone_61: "f32[64, 50, 768]" = torch.ops.aten.clone.default(add_60, memory_format = torch.contiguous_format)
        var_mean_15 = torch.ops.aten.var_mean.correction(clone_61, [2], correction = 0, keepdim = True)
        getitem_62: "f32[64, 50, 1]" = var_mean_15[0]
        getitem_63: "f32[64, 50, 1]" = var_mean_15[1];  var_mean_15 = None
        add_61: "f32[64, 50, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-06);  getitem_62 = None
        rsqrt_15: "f32[64, 50, 1]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
        sub_15: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(clone_61, getitem_63);  clone_61 = getitem_63 = None
        mul_51: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
        mul_52: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_51, primals_95)
        add_62: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(mul_52, primals_96);  mul_52 = primals_96 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        view_117: "f32[3200, 768]" = torch.ops.aten.reshape.default(add_62, [3200, 768]);  add_62 = None
        permute_87: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_97, [1, 0]);  primals_97 = None
        addmm_22: "f32[3200, 3072]" = torch.ops.aten.addmm.default(primals_98, view_117, permute_87);  primals_98 = None
        view_118: "f32[64, 50, 3072]" = torch.ops.aten.reshape.default(addmm_22, [64, 50, 3072])
        mul_53: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_118, 0.5)
        mul_54: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_118, 0.7071067811865476);  view_118 = None
        erf_7: "f32[64, 50, 3072]" = torch.ops.aten.erf.default(mul_54);  mul_54 = None
        add_63: "f32[64, 50, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_55: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(mul_53, add_63);  mul_53 = add_63 = None
        view_119: "f32[3200, 3072]" = torch.ops.aten.reshape.default(mul_55, [3200, 3072]);  mul_55 = None
        permute_88: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_99, [1, 0]);  primals_99 = None
        
        # No stacktrace found for following nodes
        mm_default_8: "f32[3200, 768]" = torch.ops.aten.mm.default(view_119, permute_88)
        add_tensor_8: "f32[3200, 768]" = torch.ops.aten.add.Tensor(mm_default_8, primals_100);  mm_default_8 = primals_100 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        view_120: "f32[64, 50, 768]" = torch.ops.aten.reshape.default(add_tensor_8, [64, 50, 768]);  add_tensor_8 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:119, code: return x + y
        add_64: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(add_60, view_120);  add_60 = view_120 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:112, code: x = self.ln_1(input)
        clone_64: "f32[64, 50, 768]" = torch.ops.aten.clone.default(add_64, memory_format = torch.contiguous_format)
        var_mean_16 = torch.ops.aten.var_mean.correction(clone_64, [2], correction = 0, keepdim = True)
        getitem_64: "f32[64, 50, 1]" = var_mean_16[0]
        getitem_65: "f32[64, 50, 1]" = var_mean_16[1];  var_mean_16 = None
        add_65: "f32[64, 50, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-06);  getitem_64 = None
        rsqrt_16: "f32[64, 50, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
        sub_16: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(clone_64, getitem_65);  clone_64 = getitem_65 = None
        mul_56: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
        mul_57: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_56, primals_101)
        add_66: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(mul_57, primals_102);  mul_57 = primals_102 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_89: "f32[50, 64, 768]" = torch.ops.aten.permute.default(add_66, [1, 0, 2]);  add_66 = None
        permute_90: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_103, [1, 0]);  primals_103 = None
        clone_65: "f32[50, 64, 768]" = torch.ops.aten.clone.default(permute_89, memory_format = torch.contiguous_format);  permute_89 = None
        view_121: "f32[3200, 768]" = torch.ops.aten.reshape.default(clone_65, [3200, 768]);  clone_65 = None
        mm_8: "f32[3200, 2304]" = torch.ops.aten.mm.default(view_121, permute_90)
        view_122: "f32[50, 64, 2304]" = torch.ops.aten.reshape.default(mm_8, [50, 64, 2304]);  mm_8 = None
        add_67: "f32[50, 64, 2304]" = torch.ops.aten.add.Tensor(view_122, primals_104);  view_122 = primals_104 = None
        view_123: "f32[50, 64, 3, 768]" = torch.ops.aten.reshape.default(add_67, [50, 64, 3, 768]);  add_67 = None
        unsqueeze_8: "f32[1, 50, 64, 3, 768]" = torch.ops.aten.unsqueeze.default(view_123, 0);  view_123 = None
        permute_91: "f32[3, 50, 64, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_8, [3, 1, 2, 0, 4]);  unsqueeze_8 = None
        squeeze_8: "f32[3, 50, 64, 768]" = torch.ops.aten.squeeze.dim(permute_91, -2);  permute_91 = None
        clone_66: "f32[3, 50, 64, 768]" = torch.ops.aten.clone.default(squeeze_8, memory_format = torch.contiguous_format);  squeeze_8 = None
        select_24: "f32[50, 64, 768]" = torch.ops.aten.select.int(clone_66, 0, 0)
        select_25: "f32[50, 64, 768]" = torch.ops.aten.select.int(clone_66, 0, 1)
        select_26: "f32[50, 64, 768]" = torch.ops.aten.select.int(clone_66, 0, 2);  clone_66 = None
        view_124: "f32[50, 768, 64]" = torch.ops.aten.reshape.default(select_24, [50, 768, 64]);  select_24 = None
        permute_92: "f32[768, 50, 64]" = torch.ops.aten.permute.default(view_124, [1, 0, 2]);  view_124 = None
        view_125: "f32[50, 768, 64]" = torch.ops.aten.reshape.default(select_25, [50, 768, 64]);  select_25 = None
        permute_93: "f32[768, 50, 64]" = torch.ops.aten.permute.default(view_125, [1, 0, 2]);  view_125 = None
        view_126: "f32[50, 768, 64]" = torch.ops.aten.reshape.default(select_26, [50, 768, 64]);  select_26 = None
        permute_94: "f32[768, 50, 64]" = torch.ops.aten.permute.default(view_126, [1, 0, 2]);  view_126 = None
        view_127: "f32[64, 12, 50, 64]" = torch.ops.aten.reshape.default(permute_92, [64, 12, 50, 64]);  permute_92 = None
        view_128: "f32[64, 12, 50, 64]" = torch.ops.aten.reshape.default(permute_93, [64, 12, 50, 64]);  permute_93 = None
        view_129: "f32[64, 12, 50, 64]" = torch.ops.aten.reshape.default(permute_94, [64, 12, 50, 64]);  permute_94 = None
        _scaled_dot_product_efficient_attention_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(view_127, view_128, view_129, None, True)
        getitem_66: "f32[64, 12, 50, 64]" = _scaled_dot_product_efficient_attention_8[0]
        getitem_67: "f32[64, 12, 64]" = _scaled_dot_product_efficient_attention_8[1]
        getitem_68: "i64[]" = _scaled_dot_product_efficient_attention_8[2]
        getitem_69: "i64[]" = _scaled_dot_product_efficient_attention_8[3];  _scaled_dot_product_efficient_attention_8 = None
        alias_8: "f32[64, 12, 50, 64]" = torch.ops.aten.alias.default(getitem_66)
        permute_95: "f32[50, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_66, [2, 0, 1, 3]);  getitem_66 = None
        clone_67: "f32[50, 64, 12, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        view_130: "f32[3200, 768]" = torch.ops.aten.reshape.default(clone_67, [3200, 768]);  clone_67 = None
        permute_96: "f32[768, 768]" = torch.ops.aten.permute.default(primals_105, [1, 0]);  primals_105 = None
        
        # No stacktrace found for following nodes
        mm_default_7: "f32[3200, 768]" = torch.ops.aten.mm.default(view_130, permute_96)
        add_tensor_7: "f32[3200, 768]" = torch.ops.aten.add.Tensor(mm_default_7, primals_106);  mm_default_7 = primals_106 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        view_131: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(add_tensor_7, [50, 64, 768]);  add_tensor_7 = None
        permute_97: "f32[64, 50, 768]" = torch.ops.aten.permute.default(view_131, [1, 0, 2]);  view_131 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:115, code: x = x + input
        add_68: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(permute_97, add_64);  permute_97 = add_64 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:117, code: y = self.ln_2(x)
        clone_69: "f32[64, 50, 768]" = torch.ops.aten.clone.default(add_68, memory_format = torch.contiguous_format)
        var_mean_17 = torch.ops.aten.var_mean.correction(clone_69, [2], correction = 0, keepdim = True)
        getitem_70: "f32[64, 50, 1]" = var_mean_17[0]
        getitem_71: "f32[64, 50, 1]" = var_mean_17[1];  var_mean_17 = None
        add_69: "f32[64, 50, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-06);  getitem_70 = None
        rsqrt_17: "f32[64, 50, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        sub_17: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(clone_69, getitem_71);  clone_69 = getitem_71 = None
        mul_58: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
        mul_59: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_58, primals_107)
        add_70: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(mul_59, primals_108);  mul_59 = primals_108 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        view_132: "f32[3200, 768]" = torch.ops.aten.reshape.default(add_70, [3200, 768]);  add_70 = None
        permute_98: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_109, [1, 0]);  primals_109 = None
        addmm_25: "f32[3200, 3072]" = torch.ops.aten.addmm.default(primals_110, view_132, permute_98);  primals_110 = None
        view_133: "f32[64, 50, 3072]" = torch.ops.aten.reshape.default(addmm_25, [64, 50, 3072])
        mul_60: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_133, 0.5)
        mul_61: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_133, 0.7071067811865476);  view_133 = None
        erf_8: "f32[64, 50, 3072]" = torch.ops.aten.erf.default(mul_61);  mul_61 = None
        add_71: "f32[64, 50, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_62: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(mul_60, add_71);  mul_60 = add_71 = None
        view_134: "f32[3200, 3072]" = torch.ops.aten.reshape.default(mul_62, [3200, 3072]);  mul_62 = None
        permute_99: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_111, [1, 0]);  primals_111 = None
        
        # No stacktrace found for following nodes
        mm_default_6: "f32[3200, 768]" = torch.ops.aten.mm.default(view_134, permute_99)
        add_tensor_6: "f32[3200, 768]" = torch.ops.aten.add.Tensor(mm_default_6, primals_112);  mm_default_6 = primals_112 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        view_135: "f32[64, 50, 768]" = torch.ops.aten.reshape.default(add_tensor_6, [64, 50, 768]);  add_tensor_6 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:119, code: return x + y
        add_72: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(add_68, view_135);  add_68 = view_135 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:112, code: x = self.ln_1(input)
        clone_72: "f32[64, 50, 768]" = torch.ops.aten.clone.default(add_72, memory_format = torch.contiguous_format)
        var_mean_18 = torch.ops.aten.var_mean.correction(clone_72, [2], correction = 0, keepdim = True)
        getitem_72: "f32[64, 50, 1]" = var_mean_18[0]
        getitem_73: "f32[64, 50, 1]" = var_mean_18[1];  var_mean_18 = None
        add_73: "f32[64, 50, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-06);  getitem_72 = None
        rsqrt_18: "f32[64, 50, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
        sub_18: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(clone_72, getitem_73);  clone_72 = getitem_73 = None
        mul_63: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
        mul_64: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_63, primals_113)
        add_74: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(mul_64, primals_114);  mul_64 = primals_114 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_100: "f32[50, 64, 768]" = torch.ops.aten.permute.default(add_74, [1, 0, 2]);  add_74 = None
        permute_101: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
        clone_73: "f32[50, 64, 768]" = torch.ops.aten.clone.default(permute_100, memory_format = torch.contiguous_format);  permute_100 = None
        view_136: "f32[3200, 768]" = torch.ops.aten.reshape.default(clone_73, [3200, 768]);  clone_73 = None
        mm_9: "f32[3200, 2304]" = torch.ops.aten.mm.default(view_136, permute_101)
        view_137: "f32[50, 64, 2304]" = torch.ops.aten.reshape.default(mm_9, [50, 64, 2304]);  mm_9 = None
        add_75: "f32[50, 64, 2304]" = torch.ops.aten.add.Tensor(view_137, primals_116);  view_137 = primals_116 = None
        view_138: "f32[50, 64, 3, 768]" = torch.ops.aten.reshape.default(add_75, [50, 64, 3, 768]);  add_75 = None
        unsqueeze_9: "f32[1, 50, 64, 3, 768]" = torch.ops.aten.unsqueeze.default(view_138, 0);  view_138 = None
        permute_102: "f32[3, 50, 64, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_9, [3, 1, 2, 0, 4]);  unsqueeze_9 = None
        squeeze_9: "f32[3, 50, 64, 768]" = torch.ops.aten.squeeze.dim(permute_102, -2);  permute_102 = None
        clone_74: "f32[3, 50, 64, 768]" = torch.ops.aten.clone.default(squeeze_9, memory_format = torch.contiguous_format);  squeeze_9 = None
        select_27: "f32[50, 64, 768]" = torch.ops.aten.select.int(clone_74, 0, 0)
        select_28: "f32[50, 64, 768]" = torch.ops.aten.select.int(clone_74, 0, 1)
        select_29: "f32[50, 64, 768]" = torch.ops.aten.select.int(clone_74, 0, 2);  clone_74 = None
        view_139: "f32[50, 768, 64]" = torch.ops.aten.reshape.default(select_27, [50, 768, 64]);  select_27 = None
        permute_103: "f32[768, 50, 64]" = torch.ops.aten.permute.default(view_139, [1, 0, 2]);  view_139 = None
        view_140: "f32[50, 768, 64]" = torch.ops.aten.reshape.default(select_28, [50, 768, 64]);  select_28 = None
        permute_104: "f32[768, 50, 64]" = torch.ops.aten.permute.default(view_140, [1, 0, 2]);  view_140 = None
        view_141: "f32[50, 768, 64]" = torch.ops.aten.reshape.default(select_29, [50, 768, 64]);  select_29 = None
        permute_105: "f32[768, 50, 64]" = torch.ops.aten.permute.default(view_141, [1, 0, 2]);  view_141 = None
        view_142: "f32[64, 12, 50, 64]" = torch.ops.aten.reshape.default(permute_103, [64, 12, 50, 64]);  permute_103 = None
        view_143: "f32[64, 12, 50, 64]" = torch.ops.aten.reshape.default(permute_104, [64, 12, 50, 64]);  permute_104 = None
        view_144: "f32[64, 12, 50, 64]" = torch.ops.aten.reshape.default(permute_105, [64, 12, 50, 64]);  permute_105 = None
        _scaled_dot_product_efficient_attention_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(view_142, view_143, view_144, None, True)
        getitem_74: "f32[64, 12, 50, 64]" = _scaled_dot_product_efficient_attention_9[0]
        getitem_75: "f32[64, 12, 64]" = _scaled_dot_product_efficient_attention_9[1]
        getitem_76: "i64[]" = _scaled_dot_product_efficient_attention_9[2]
        getitem_77: "i64[]" = _scaled_dot_product_efficient_attention_9[3];  _scaled_dot_product_efficient_attention_9 = None
        alias_9: "f32[64, 12, 50, 64]" = torch.ops.aten.alias.default(getitem_74)
        permute_106: "f32[50, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_74, [2, 0, 1, 3]);  getitem_74 = None
        clone_75: "f32[50, 64, 12, 64]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
        view_145: "f32[3200, 768]" = torch.ops.aten.reshape.default(clone_75, [3200, 768]);  clone_75 = None
        permute_107: "f32[768, 768]" = torch.ops.aten.permute.default(primals_117, [1, 0]);  primals_117 = None
        
        # No stacktrace found for following nodes
        mm_default_5: "f32[3200, 768]" = torch.ops.aten.mm.default(view_145, permute_107)
        add_tensor_5: "f32[3200, 768]" = torch.ops.aten.add.Tensor(mm_default_5, primals_118);  mm_default_5 = primals_118 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        view_146: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(add_tensor_5, [50, 64, 768]);  add_tensor_5 = None
        permute_108: "f32[64, 50, 768]" = torch.ops.aten.permute.default(view_146, [1, 0, 2]);  view_146 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:115, code: x = x + input
        add_76: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(permute_108, add_72);  permute_108 = add_72 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:117, code: y = self.ln_2(x)
        clone_77: "f32[64, 50, 768]" = torch.ops.aten.clone.default(add_76, memory_format = torch.contiguous_format)
        var_mean_19 = torch.ops.aten.var_mean.correction(clone_77, [2], correction = 0, keepdim = True)
        getitem_78: "f32[64, 50, 1]" = var_mean_19[0]
        getitem_79: "f32[64, 50, 1]" = var_mean_19[1];  var_mean_19 = None
        add_77: "f32[64, 50, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-06);  getitem_78 = None
        rsqrt_19: "f32[64, 50, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
        sub_19: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(clone_77, getitem_79);  clone_77 = getitem_79 = None
        mul_65: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
        mul_66: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_65, primals_119)
        add_78: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(mul_66, primals_120);  mul_66 = primals_120 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        view_147: "f32[3200, 768]" = torch.ops.aten.reshape.default(add_78, [3200, 768]);  add_78 = None
        permute_109: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_121, [1, 0]);  primals_121 = None
        addmm_28: "f32[3200, 3072]" = torch.ops.aten.addmm.default(primals_122, view_147, permute_109);  primals_122 = None
        view_148: "f32[64, 50, 3072]" = torch.ops.aten.reshape.default(addmm_28, [64, 50, 3072])
        mul_67: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_148, 0.5)
        mul_68: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_148, 0.7071067811865476);  view_148 = None
        erf_9: "f32[64, 50, 3072]" = torch.ops.aten.erf.default(mul_68);  mul_68 = None
        add_79: "f32[64, 50, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_69: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(mul_67, add_79);  mul_67 = add_79 = None
        view_149: "f32[3200, 3072]" = torch.ops.aten.reshape.default(mul_69, [3200, 3072]);  mul_69 = None
        permute_110: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_123, [1, 0]);  primals_123 = None
        
        # No stacktrace found for following nodes
        mm_default_4: "f32[3200, 768]" = torch.ops.aten.mm.default(view_149, permute_110)
        add_tensor_4: "f32[3200, 768]" = torch.ops.aten.add.Tensor(mm_default_4, primals_124);  mm_default_4 = primals_124 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        view_150: "f32[64, 50, 768]" = torch.ops.aten.reshape.default(add_tensor_4, [64, 50, 768]);  add_tensor_4 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:119, code: return x + y
        add_80: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(add_76, view_150);  add_76 = view_150 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:112, code: x = self.ln_1(input)
        clone_80: "f32[64, 50, 768]" = torch.ops.aten.clone.default(add_80, memory_format = torch.contiguous_format)
        var_mean_20 = torch.ops.aten.var_mean.correction(clone_80, [2], correction = 0, keepdim = True)
        getitem_80: "f32[64, 50, 1]" = var_mean_20[0]
        getitem_81: "f32[64, 50, 1]" = var_mean_20[1];  var_mean_20 = None
        add_81: "f32[64, 50, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-06);  getitem_80 = None
        rsqrt_20: "f32[64, 50, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
        sub_20: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(clone_80, getitem_81);  clone_80 = getitem_81 = None
        mul_70: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
        mul_71: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_70, primals_125)
        add_82: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(mul_71, primals_126);  mul_71 = primals_126 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_111: "f32[50, 64, 768]" = torch.ops.aten.permute.default(add_82, [1, 0, 2]);  add_82 = None
        permute_112: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_127, [1, 0]);  primals_127 = None
        clone_81: "f32[50, 64, 768]" = torch.ops.aten.clone.default(permute_111, memory_format = torch.contiguous_format);  permute_111 = None
        view_151: "f32[3200, 768]" = torch.ops.aten.reshape.default(clone_81, [3200, 768]);  clone_81 = None
        mm_10: "f32[3200, 2304]" = torch.ops.aten.mm.default(view_151, permute_112)
        view_152: "f32[50, 64, 2304]" = torch.ops.aten.reshape.default(mm_10, [50, 64, 2304]);  mm_10 = None
        add_83: "f32[50, 64, 2304]" = torch.ops.aten.add.Tensor(view_152, primals_128);  view_152 = primals_128 = None
        view_153: "f32[50, 64, 3, 768]" = torch.ops.aten.reshape.default(add_83, [50, 64, 3, 768]);  add_83 = None
        unsqueeze_10: "f32[1, 50, 64, 3, 768]" = torch.ops.aten.unsqueeze.default(view_153, 0);  view_153 = None
        permute_113: "f32[3, 50, 64, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_10, [3, 1, 2, 0, 4]);  unsqueeze_10 = None
        squeeze_10: "f32[3, 50, 64, 768]" = torch.ops.aten.squeeze.dim(permute_113, -2);  permute_113 = None
        clone_82: "f32[3, 50, 64, 768]" = torch.ops.aten.clone.default(squeeze_10, memory_format = torch.contiguous_format);  squeeze_10 = None
        select_30: "f32[50, 64, 768]" = torch.ops.aten.select.int(clone_82, 0, 0)
        select_31: "f32[50, 64, 768]" = torch.ops.aten.select.int(clone_82, 0, 1)
        select_32: "f32[50, 64, 768]" = torch.ops.aten.select.int(clone_82, 0, 2);  clone_82 = None
        view_154: "f32[50, 768, 64]" = torch.ops.aten.reshape.default(select_30, [50, 768, 64]);  select_30 = None
        permute_114: "f32[768, 50, 64]" = torch.ops.aten.permute.default(view_154, [1, 0, 2]);  view_154 = None
        view_155: "f32[50, 768, 64]" = torch.ops.aten.reshape.default(select_31, [50, 768, 64]);  select_31 = None
        permute_115: "f32[768, 50, 64]" = torch.ops.aten.permute.default(view_155, [1, 0, 2]);  view_155 = None
        view_156: "f32[50, 768, 64]" = torch.ops.aten.reshape.default(select_32, [50, 768, 64]);  select_32 = None
        permute_116: "f32[768, 50, 64]" = torch.ops.aten.permute.default(view_156, [1, 0, 2]);  view_156 = None
        view_157: "f32[64, 12, 50, 64]" = torch.ops.aten.reshape.default(permute_114, [64, 12, 50, 64]);  permute_114 = None
        view_158: "f32[64, 12, 50, 64]" = torch.ops.aten.reshape.default(permute_115, [64, 12, 50, 64]);  permute_115 = None
        view_159: "f32[64, 12, 50, 64]" = torch.ops.aten.reshape.default(permute_116, [64, 12, 50, 64]);  permute_116 = None
        _scaled_dot_product_efficient_attention_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(view_157, view_158, view_159, None, True)
        getitem_82: "f32[64, 12, 50, 64]" = _scaled_dot_product_efficient_attention_10[0]
        getitem_83: "f32[64, 12, 64]" = _scaled_dot_product_efficient_attention_10[1]
        getitem_84: "i64[]" = _scaled_dot_product_efficient_attention_10[2]
        getitem_85: "i64[]" = _scaled_dot_product_efficient_attention_10[3];  _scaled_dot_product_efficient_attention_10 = None
        alias_10: "f32[64, 12, 50, 64]" = torch.ops.aten.alias.default(getitem_82)
        permute_117: "f32[50, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_82, [2, 0, 1, 3]);  getitem_82 = None
        clone_83: "f32[50, 64, 12, 64]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
        view_160: "f32[3200, 768]" = torch.ops.aten.reshape.default(clone_83, [3200, 768]);  clone_83 = None
        permute_118: "f32[768, 768]" = torch.ops.aten.permute.default(primals_129, [1, 0]);  primals_129 = None
        
        # No stacktrace found for following nodes
        mm_default_3: "f32[3200, 768]" = torch.ops.aten.mm.default(view_160, permute_118)
        add_tensor_3: "f32[3200, 768]" = torch.ops.aten.add.Tensor(mm_default_3, primals_130);  mm_default_3 = primals_130 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        view_161: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(add_tensor_3, [50, 64, 768]);  add_tensor_3 = None
        permute_119: "f32[64, 50, 768]" = torch.ops.aten.permute.default(view_161, [1, 0, 2]);  view_161 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:115, code: x = x + input
        add_84: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(permute_119, add_80);  permute_119 = add_80 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:117, code: y = self.ln_2(x)
        clone_85: "f32[64, 50, 768]" = torch.ops.aten.clone.default(add_84, memory_format = torch.contiguous_format)
        var_mean_21 = torch.ops.aten.var_mean.correction(clone_85, [2], correction = 0, keepdim = True)
        getitem_86: "f32[64, 50, 1]" = var_mean_21[0]
        getitem_87: "f32[64, 50, 1]" = var_mean_21[1];  var_mean_21 = None
        add_85: "f32[64, 50, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-06);  getitem_86 = None
        rsqrt_21: "f32[64, 50, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        sub_21: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(clone_85, getitem_87);  clone_85 = getitem_87 = None
        mul_72: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
        mul_73: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_72, primals_131)
        add_86: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(mul_73, primals_132);  mul_73 = primals_132 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        view_162: "f32[3200, 768]" = torch.ops.aten.reshape.default(add_86, [3200, 768]);  add_86 = None
        permute_120: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_133, [1, 0]);  primals_133 = None
        addmm_31: "f32[3200, 3072]" = torch.ops.aten.addmm.default(primals_134, view_162, permute_120);  primals_134 = None
        view_163: "f32[64, 50, 3072]" = torch.ops.aten.reshape.default(addmm_31, [64, 50, 3072])
        mul_74: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_163, 0.5)
        mul_75: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_163, 0.7071067811865476);  view_163 = None
        erf_10: "f32[64, 50, 3072]" = torch.ops.aten.erf.default(mul_75);  mul_75 = None
        add_87: "f32[64, 50, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_76: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(mul_74, add_87);  mul_74 = add_87 = None
        view_164: "f32[3200, 3072]" = torch.ops.aten.reshape.default(mul_76, [3200, 3072]);  mul_76 = None
        permute_121: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_135, [1, 0]);  primals_135 = None
        
        # No stacktrace found for following nodes
        mm_default_2: "f32[3200, 768]" = torch.ops.aten.mm.default(view_164, permute_121)
        add_tensor_2: "f32[3200, 768]" = torch.ops.aten.add.Tensor(mm_default_2, primals_136);  mm_default_2 = primals_136 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        view_165: "f32[64, 50, 768]" = torch.ops.aten.reshape.default(add_tensor_2, [64, 50, 768]);  add_tensor_2 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:119, code: return x + y
        add_88: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(add_84, view_165);  add_84 = view_165 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:112, code: x = self.ln_1(input)
        clone_88: "f32[64, 50, 768]" = torch.ops.aten.clone.default(add_88, memory_format = torch.contiguous_format)
        var_mean_22 = torch.ops.aten.var_mean.correction(clone_88, [2], correction = 0, keepdim = True)
        getitem_88: "f32[64, 50, 1]" = var_mean_22[0]
        getitem_89: "f32[64, 50, 1]" = var_mean_22[1];  var_mean_22 = None
        add_89: "f32[64, 50, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-06);  getitem_88 = None
        rsqrt_22: "f32[64, 50, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
        sub_22: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(clone_88, getitem_89);  clone_88 = getitem_89 = None
        mul_77: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
        mul_78: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_77, primals_137)
        add_90: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(mul_78, primals_138);  mul_78 = primals_138 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_122: "f32[50, 64, 768]" = torch.ops.aten.permute.default(add_90, [1, 0, 2]);  add_90 = None
        permute_123: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_139, [1, 0]);  primals_139 = None
        clone_89: "f32[50, 64, 768]" = torch.ops.aten.clone.default(permute_122, memory_format = torch.contiguous_format);  permute_122 = None
        view_166: "f32[3200, 768]" = torch.ops.aten.reshape.default(clone_89, [3200, 768]);  clone_89 = None
        mm_11: "f32[3200, 2304]" = torch.ops.aten.mm.default(view_166, permute_123)
        view_167: "f32[50, 64, 2304]" = torch.ops.aten.reshape.default(mm_11, [50, 64, 2304]);  mm_11 = None
        add_91: "f32[50, 64, 2304]" = torch.ops.aten.add.Tensor(view_167, primals_140);  view_167 = primals_140 = None
        view_168: "f32[50, 64, 3, 768]" = torch.ops.aten.reshape.default(add_91, [50, 64, 3, 768]);  add_91 = None
        unsqueeze_11: "f32[1, 50, 64, 3, 768]" = torch.ops.aten.unsqueeze.default(view_168, 0);  view_168 = None
        permute_124: "f32[3, 50, 64, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_11, [3, 1, 2, 0, 4]);  unsqueeze_11 = None
        squeeze_11: "f32[3, 50, 64, 768]" = torch.ops.aten.squeeze.dim(permute_124, -2);  permute_124 = None
        clone_90: "f32[3, 50, 64, 768]" = torch.ops.aten.clone.default(squeeze_11, memory_format = torch.contiguous_format);  squeeze_11 = None
        select_33: "f32[50, 64, 768]" = torch.ops.aten.select.int(clone_90, 0, 0)
        select_34: "f32[50, 64, 768]" = torch.ops.aten.select.int(clone_90, 0, 1)
        select_35: "f32[50, 64, 768]" = torch.ops.aten.select.int(clone_90, 0, 2);  clone_90 = None
        view_169: "f32[50, 768, 64]" = torch.ops.aten.reshape.default(select_33, [50, 768, 64]);  select_33 = None
        permute_125: "f32[768, 50, 64]" = torch.ops.aten.permute.default(view_169, [1, 0, 2]);  view_169 = None
        view_170: "f32[50, 768, 64]" = torch.ops.aten.reshape.default(select_34, [50, 768, 64]);  select_34 = None
        permute_126: "f32[768, 50, 64]" = torch.ops.aten.permute.default(view_170, [1, 0, 2]);  view_170 = None
        view_171: "f32[50, 768, 64]" = torch.ops.aten.reshape.default(select_35, [50, 768, 64]);  select_35 = None
        permute_127: "f32[768, 50, 64]" = torch.ops.aten.permute.default(view_171, [1, 0, 2]);  view_171 = None
        view_172: "f32[64, 12, 50, 64]" = torch.ops.aten.reshape.default(permute_125, [64, 12, 50, 64]);  permute_125 = None
        view_173: "f32[64, 12, 50, 64]" = torch.ops.aten.reshape.default(permute_126, [64, 12, 50, 64]);  permute_126 = None
        view_174: "f32[64, 12, 50, 64]" = torch.ops.aten.reshape.default(permute_127, [64, 12, 50, 64]);  permute_127 = None
        _scaled_dot_product_efficient_attention_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(view_172, view_173, view_174, None, True)
        getitem_90: "f32[64, 12, 50, 64]" = _scaled_dot_product_efficient_attention_11[0]
        getitem_91: "f32[64, 12, 64]" = _scaled_dot_product_efficient_attention_11[1]
        getitem_92: "i64[]" = _scaled_dot_product_efficient_attention_11[2]
        getitem_93: "i64[]" = _scaled_dot_product_efficient_attention_11[3];  _scaled_dot_product_efficient_attention_11 = None
        alias_11: "f32[64, 12, 50, 64]" = torch.ops.aten.alias.default(getitem_90)
        permute_128: "f32[50, 64, 12, 64]" = torch.ops.aten.permute.default(getitem_90, [2, 0, 1, 3]);  getitem_90 = None
        clone_91: "f32[50, 64, 12, 64]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
        view_175: "f32[3200, 768]" = torch.ops.aten.reshape.default(clone_91, [3200, 768]);  clone_91 = None
        permute_129: "f32[768, 768]" = torch.ops.aten.permute.default(primals_141, [1, 0]);  primals_141 = None
        
        # No stacktrace found for following nodes
        mm_default_1: "f32[3200, 768]" = torch.ops.aten.mm.default(view_175, permute_129)
        add_tensor_1: "f32[3200, 768]" = torch.ops.aten.add.Tensor(mm_default_1, primals_142);  mm_default_1 = primals_142 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        view_176: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(add_tensor_1, [50, 64, 768]);  add_tensor_1 = None
        permute_130: "f32[64, 50, 768]" = torch.ops.aten.permute.default(view_176, [1, 0, 2]);  view_176 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:115, code: x = x + input
        add_92: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(permute_130, add_88);  permute_130 = add_88 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:117, code: y = self.ln_2(x)
        clone_93: "f32[64, 50, 768]" = torch.ops.aten.clone.default(add_92, memory_format = torch.contiguous_format)
        var_mean_23 = torch.ops.aten.var_mean.correction(clone_93, [2], correction = 0, keepdim = True)
        getitem_94: "f32[64, 50, 1]" = var_mean_23[0]
        getitem_95: "f32[64, 50, 1]" = var_mean_23[1];  var_mean_23 = None
        add_93: "f32[64, 50, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-06);  getitem_94 = None
        rsqrt_23: "f32[64, 50, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        sub_23: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(clone_93, getitem_95);  clone_93 = getitem_95 = None
        mul_79: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
        mul_80: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_79, primals_143)
        add_94: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(mul_80, primals_144);  mul_80 = primals_144 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        view_177: "f32[3200, 768]" = torch.ops.aten.reshape.default(add_94, [3200, 768]);  add_94 = None
        permute_131: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_145, [1, 0]);  primals_145 = None
        addmm_34: "f32[3200, 3072]" = torch.ops.aten.addmm.default(primals_146, view_177, permute_131);  primals_146 = None
        view_178: "f32[64, 50, 3072]" = torch.ops.aten.reshape.default(addmm_34, [64, 50, 3072])
        mul_81: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_178, 0.5)
        mul_82: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_178, 0.7071067811865476);  view_178 = None
        erf_11: "f32[64, 50, 3072]" = torch.ops.aten.erf.default(mul_82);  mul_82 = None
        add_95: "f32[64, 50, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_83: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(mul_81, add_95);  mul_81 = add_95 = None
        view_179: "f32[3200, 3072]" = torch.ops.aten.reshape.default(mul_83, [3200, 3072]);  mul_83 = None
        permute_132: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_147, [1, 0]);  primals_147 = None
        
        # No stacktrace found for following nodes
        mm_default: "f32[3200, 768]" = torch.ops.aten.mm.default(view_179, permute_132)
        add_tensor: "f32[3200, 768]" = torch.ops.aten.add.Tensor(mm_default, primals_148);  mm_default = primals_148 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        view_180: "f32[64, 50, 768]" = torch.ops.aten.reshape.default(add_tensor, [64, 50, 768]);  add_tensor = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:119, code: return x + y
        add_96: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(add_92, view_180);  add_92 = view_180 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:157, code: return self.ln(self.layers(self.dropout(input)))
        clone_96: "f32[64, 50, 768]" = torch.ops.aten.clone.default(add_96, memory_format = torch.contiguous_format);  add_96 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(clone_96, [2], correction = 0, keepdim = True)
        getitem_96: "f32[64, 50, 1]" = var_mean_24[0]
        getitem_97: "f32[64, 50, 1]" = var_mean_24[1];  var_mean_24 = None
        add_97: "f32[64, 50, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-06);  getitem_96 = None
        rsqrt_24: "f32[64, 50, 1]" = torch.ops.aten.rsqrt.default(add_97);  add_97 = None
        sub_24: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(clone_96, getitem_97);  clone_96 = getitem_97 = None
        mul_84: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
        mul_85: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_84, primals_149)
        add_98: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(mul_85, primals_150);  mul_85 = primals_150 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:301, code: x = x[:, 0]
        slice_1: "f32[64, 50, 768]" = torch.ops.aten.slice.Tensor(add_98, 0, 0, 9223372036854775807);  add_98 = None
        select_36: "f32[64, 768]" = torch.ops.aten.select.int(slice_1, 1, 0);  slice_1 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:303, code: x = self.heads(x)
        permute_133: "f32[768, 1000]" = torch.ops.aten.permute.default(primals_151, [1, 0]);  primals_151 = None
        addmm_36: "f32[64, 1000]" = torch.ops.aten.addmm.default(primals_152, select_36, permute_133);  primals_152 = None
        permute_134: "f32[1000, 768]" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:157, code: return self.ln(self.layers(self.dropout(input)))
        div: "f32[64, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        permute_138: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
        permute_142: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:117, code: y = self.ln_2(x)
        div_1: "f32[64, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_147: "f32[768, 768]" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
        alias_12: "f32[64, 12, 50, 64]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
        permute_158: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:112, code: x = self.ln_1(input)
        div_2: "f32[64, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        permute_161: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
        permute_165: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:117, code: y = self.ln_2(x)
        div_3: "f32[64, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_170: "f32[768, 768]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
        alias_13: "f32[64, 12, 50, 64]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
        permute_181: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:112, code: x = self.ln_1(input)
        div_4: "f32[64, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        permute_184: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
        permute_188: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:117, code: y = self.ln_2(x)
        div_5: "f32[64, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_193: "f32[768, 768]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
        alias_14: "f32[64, 12, 50, 64]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
        permute_204: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:112, code: x = self.ln_1(input)
        div_6: "f32[64, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        permute_207: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
        permute_211: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:117, code: y = self.ln_2(x)
        div_7: "f32[64, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_216: "f32[768, 768]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
        alias_15: "f32[64, 12, 50, 64]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
        permute_227: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:112, code: x = self.ln_1(input)
        div_8: "f32[64, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        permute_230: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
        permute_234: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:117, code: y = self.ln_2(x)
        div_9: "f32[64, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_239: "f32[768, 768]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
        alias_16: "f32[64, 12, 50, 64]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
        permute_250: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:112, code: x = self.ln_1(input)
        div_10: "f32[64, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        permute_253: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
        permute_257: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:117, code: y = self.ln_2(x)
        div_11: "f32[64, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_262: "f32[768, 768]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
        alias_17: "f32[64, 12, 50, 64]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
        permute_273: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:112, code: x = self.ln_1(input)
        div_12: "f32[64, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        permute_276: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
        permute_280: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:117, code: y = self.ln_2(x)
        div_13: "f32[64, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_285: "f32[768, 768]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
        alias_18: "f32[64, 12, 50, 64]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
        permute_296: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:112, code: x = self.ln_1(input)
        div_14: "f32[64, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        permute_299: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
        permute_303: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:117, code: y = self.ln_2(x)
        div_15: "f32[64, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_308: "f32[768, 768]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
        alias_19: "f32[64, 12, 50, 64]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
        permute_319: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:112, code: x = self.ln_1(input)
        div_16: "f32[64, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        permute_322: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
        permute_326: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:117, code: y = self.ln_2(x)
        div_17: "f32[64, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_331: "f32[768, 768]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
        alias_20: "f32[64, 12, 50, 64]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
        permute_342: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:112, code: x = self.ln_1(input)
        div_18: "f32[64, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        permute_345: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
        permute_349: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:117, code: y = self.ln_2(x)
        div_19: "f32[64, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_354: "f32[768, 768]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
        alias_21: "f32[64, 12, 50, 64]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
        permute_365: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:112, code: x = self.ln_1(input)
        div_20: "f32[64, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        permute_368: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        permute_372: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:117, code: y = self.ln_2(x)
        div_21: "f32[64, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_377: "f32[768, 768]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
        alias_22: "f32[64, 12, 50, 64]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
        permute_388: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:112, code: x = self.ln_1(input)
        div_22: "f32[64, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        permute_391: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        permute_395: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:117, code: y = self.ln_2(x)
        div_23: "f32[64, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_400: "f32[768, 768]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        alias_23: "f32[64, 12, 50, 64]" = torch.ops.aten.alias.default(alias);  alias = None
        permute_411: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:112, code: x = self.ln_1(input)
        div_24: "f32[64, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        return [addmm_36, primals_3, primals_5, primals_11, primals_17, primals_23, primals_29, primals_35, primals_41, primals_47, primals_53, primals_59, primals_65, primals_71, primals_77, primals_83, primals_89, primals_95, primals_101, primals_107, primals_113, primals_119, primals_125, primals_131, primals_137, primals_143, primals_149, primals_153, mul, view_1, view_7, view_8, view_9, getitem_3, getitem_4, getitem_5, view_10, mul_2, view_12, addmm_1, view_14, mul_7, view_16, view_22, view_23, view_24, getitem_11, getitem_12, getitem_13, view_25, mul_9, view_27, addmm_4, view_29, mul_14, view_31, view_37, view_38, view_39, getitem_19, getitem_20, getitem_21, view_40, mul_16, view_42, addmm_7, view_44, mul_21, view_46, view_52, view_53, view_54, getitem_27, getitem_28, getitem_29, view_55, mul_23, view_57, addmm_10, view_59, mul_28, view_61, view_67, view_68, view_69, getitem_35, getitem_36, getitem_37, view_70, mul_30, view_72, addmm_13, view_74, mul_35, view_76, view_82, view_83, view_84, getitem_43, getitem_44, getitem_45, view_85, mul_37, view_87, addmm_16, view_89, mul_42, view_91, view_97, view_98, view_99, getitem_51, getitem_52, getitem_53, view_100, mul_44, view_102, addmm_19, view_104, mul_49, view_106, view_112, view_113, view_114, getitem_59, getitem_60, getitem_61, view_115, mul_51, view_117, addmm_22, view_119, mul_56, view_121, view_127, view_128, view_129, getitem_67, getitem_68, getitem_69, view_130, mul_58, view_132, addmm_25, view_134, mul_63, view_136, view_142, view_143, view_144, getitem_75, getitem_76, getitem_77, view_145, mul_65, view_147, addmm_28, view_149, mul_70, view_151, view_157, view_158, view_159, getitem_83, getitem_84, getitem_85, view_160, mul_72, view_162, addmm_31, view_164, mul_77, view_166, view_172, view_173, view_174, getitem_91, getitem_92, getitem_93, view_175, mul_79, view_177, addmm_34, view_179, mul_84, select_36, permute_134, div, permute_138, permute_142, div_1, permute_147, alias_12, permute_158, div_2, permute_161, permute_165, div_3, permute_170, alias_13, permute_181, div_4, permute_184, permute_188, div_5, permute_193, alias_14, permute_204, div_6, permute_207, permute_211, div_7, permute_216, alias_15, permute_227, div_8, permute_230, permute_234, div_9, permute_239, alias_16, permute_250, div_10, permute_253, permute_257, div_11, permute_262, alias_17, permute_273, div_12, permute_276, permute_280, div_13, permute_285, alias_18, permute_296, div_14, permute_299, permute_303, div_15, permute_308, alias_19, permute_319, div_16, permute_322, permute_326, div_17, permute_331, alias_20, permute_342, div_18, permute_345, permute_349, div_19, permute_354, alias_21, permute_365, div_20, permute_368, permute_372, div_21, permute_377, alias_22, permute_388, div_22, permute_391, permute_395, div_23, permute_400, alias_23, permute_411, div_24]
        