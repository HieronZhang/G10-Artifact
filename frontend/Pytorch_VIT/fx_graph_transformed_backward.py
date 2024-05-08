class GraphModule(torch.nn.Module):
    def forward(self, primals_3: "f32[768, 3, 32, 32]", primals_5: "f32[768]", primals_11: "f32[768]", primals_17: "f32[768]", primals_23: "f32[768]", primals_29: "f32[768]", primals_35: "f32[768]", primals_41: "f32[768]", primals_47: "f32[768]", primals_53: "f32[768]", primals_59: "f32[768]", primals_65: "f32[768]", primals_71: "f32[768]", primals_77: "f32[768]", primals_83: "f32[768]", primals_89: "f32[768]", primals_95: "f32[768]", primals_101: "f32[768]", primals_107: "f32[768]", primals_113: "f32[768]", primals_119: "f32[768]", primals_125: "f32[768]", primals_131: "f32[768]", primals_137: "f32[768]", primals_143: "f32[768]", primals_149: "f32[768]", primals_153: "f32[64, 3, 224, 224]", mul: "f32[64, 50, 768]", view_1: "f32[3200, 768]", view_7: "f32[64, 12, 50, 64]", view_8: "f32[64, 12, 50, 64]", view_9: "f32[64, 12, 50, 64]", getitem_3: "f32[64, 12, 64]", getitem_4: "i64[]", getitem_5: "i64[]", view_10: "f32[3200, 768]", mul_2: "f32[64, 50, 768]", view_12: "f32[3200, 768]", addmm_1: "f32[3200, 3072]", view_14: "f32[3200, 3072]", mul_7: "f32[64, 50, 768]", view_16: "f32[3200, 768]", view_22: "f32[64, 12, 50, 64]", view_23: "f32[64, 12, 50, 64]", view_24: "f32[64, 12, 50, 64]", getitem_11: "f32[64, 12, 64]", getitem_12: "i64[]", getitem_13: "i64[]", view_25: "f32[3200, 768]", mul_9: "f32[64, 50, 768]", view_27: "f32[3200, 768]", addmm_4: "f32[3200, 3072]", view_29: "f32[3200, 3072]", mul_14: "f32[64, 50, 768]", view_31: "f32[3200, 768]", view_37: "f32[64, 12, 50, 64]", view_38: "f32[64, 12, 50, 64]", view_39: "f32[64, 12, 50, 64]", getitem_19: "f32[64, 12, 64]", getitem_20: "i64[]", getitem_21: "i64[]", view_40: "f32[3200, 768]", mul_16: "f32[64, 50, 768]", view_42: "f32[3200, 768]", addmm_7: "f32[3200, 3072]", view_44: "f32[3200, 3072]", mul_21: "f32[64, 50, 768]", view_46: "f32[3200, 768]", view_52: "f32[64, 12, 50, 64]", view_53: "f32[64, 12, 50, 64]", view_54: "f32[64, 12, 50, 64]", getitem_27: "f32[64, 12, 64]", getitem_28: "i64[]", getitem_29: "i64[]", view_55: "f32[3200, 768]", mul_23: "f32[64, 50, 768]", view_57: "f32[3200, 768]", addmm_10: "f32[3200, 3072]", view_59: "f32[3200, 3072]", mul_28: "f32[64, 50, 768]", view_61: "f32[3200, 768]", view_67: "f32[64, 12, 50, 64]", view_68: "f32[64, 12, 50, 64]", view_69: "f32[64, 12, 50, 64]", getitem_35: "f32[64, 12, 64]", getitem_36: "i64[]", getitem_37: "i64[]", view_70: "f32[3200, 768]", mul_30: "f32[64, 50, 768]", view_72: "f32[3200, 768]", addmm_13: "f32[3200, 3072]", view_74: "f32[3200, 3072]", mul_35: "f32[64, 50, 768]", view_76: "f32[3200, 768]", view_82: "f32[64, 12, 50, 64]", view_83: "f32[64, 12, 50, 64]", view_84: "f32[64, 12, 50, 64]", getitem_43: "f32[64, 12, 64]", getitem_44: "i64[]", getitem_45: "i64[]", view_85: "f32[3200, 768]", mul_37: "f32[64, 50, 768]", view_87: "f32[3200, 768]", addmm_16: "f32[3200, 3072]", view_89: "f32[3200, 3072]", mul_42: "f32[64, 50, 768]", view_91: "f32[3200, 768]", view_97: "f32[64, 12, 50, 64]", view_98: "f32[64, 12, 50, 64]", view_99: "f32[64, 12, 50, 64]", getitem_51: "f32[64, 12, 64]", getitem_52: "i64[]", getitem_53: "i64[]", view_100: "f32[3200, 768]", mul_44: "f32[64, 50, 768]", view_102: "f32[3200, 768]", addmm_19: "f32[3200, 3072]", view_104: "f32[3200, 3072]", mul_49: "f32[64, 50, 768]", view_106: "f32[3200, 768]", view_112: "f32[64, 12, 50, 64]", view_113: "f32[64, 12, 50, 64]", view_114: "f32[64, 12, 50, 64]", getitem_59: "f32[64, 12, 64]", getitem_60: "i64[]", getitem_61: "i64[]", view_115: "f32[3200, 768]", mul_51: "f32[64, 50, 768]", view_117: "f32[3200, 768]", addmm_22: "f32[3200, 3072]", view_119: "f32[3200, 3072]", mul_56: "f32[64, 50, 768]", view_121: "f32[3200, 768]", view_127: "f32[64, 12, 50, 64]", view_128: "f32[64, 12, 50, 64]", view_129: "f32[64, 12, 50, 64]", getitem_67: "f32[64, 12, 64]", getitem_68: "i64[]", getitem_69: "i64[]", view_130: "f32[3200, 768]", mul_58: "f32[64, 50, 768]", view_132: "f32[3200, 768]", addmm_25: "f32[3200, 3072]", view_134: "f32[3200, 3072]", mul_63: "f32[64, 50, 768]", view_136: "f32[3200, 768]", view_142: "f32[64, 12, 50, 64]", view_143: "f32[64, 12, 50, 64]", view_144: "f32[64, 12, 50, 64]", getitem_75: "f32[64, 12, 64]", getitem_76: "i64[]", getitem_77: "i64[]", view_145: "f32[3200, 768]", mul_65: "f32[64, 50, 768]", view_147: "f32[3200, 768]", addmm_28: "f32[3200, 3072]", view_149: "f32[3200, 3072]", mul_70: "f32[64, 50, 768]", view_151: "f32[3200, 768]", view_157: "f32[64, 12, 50, 64]", view_158: "f32[64, 12, 50, 64]", view_159: "f32[64, 12, 50, 64]", getitem_83: "f32[64, 12, 64]", getitem_84: "i64[]", getitem_85: "i64[]", view_160: "f32[3200, 768]", mul_72: "f32[64, 50, 768]", view_162: "f32[3200, 768]", addmm_31: "f32[3200, 3072]", view_164: "f32[3200, 3072]", mul_77: "f32[64, 50, 768]", view_166: "f32[3200, 768]", view_172: "f32[64, 12, 50, 64]", view_173: "f32[64, 12, 50, 64]", view_174: "f32[64, 12, 50, 64]", getitem_91: "f32[64, 12, 64]", getitem_92: "i64[]", getitem_93: "i64[]", view_175: "f32[3200, 768]", mul_79: "f32[64, 50, 768]", view_177: "f32[3200, 768]", addmm_34: "f32[3200, 3072]", view_179: "f32[3200, 3072]", mul_84: "f32[64, 50, 768]", select_36: "f32[64, 768]", permute_134: "f32[1000, 768]", div: "f32[64, 50, 1]", permute_138: "f32[768, 3072]", permute_142: "f32[3072, 768]", div_1: "f32[64, 50, 1]", permute_147: "f32[768, 768]", alias_12: "f32[64, 12, 50, 64]", permute_158: "f32[2304, 768]", div_2: "f32[64, 50, 1]", permute_161: "f32[768, 3072]", permute_165: "f32[3072, 768]", div_3: "f32[64, 50, 1]", permute_170: "f32[768, 768]", alias_13: "f32[64, 12, 50, 64]", permute_181: "f32[2304, 768]", div_4: "f32[64, 50, 1]", permute_184: "f32[768, 3072]", permute_188: "f32[3072, 768]", div_5: "f32[64, 50, 1]", permute_193: "f32[768, 768]", alias_14: "f32[64, 12, 50, 64]", permute_204: "f32[2304, 768]", div_6: "f32[64, 50, 1]", permute_207: "f32[768, 3072]", permute_211: "f32[3072, 768]", div_7: "f32[64, 50, 1]", permute_216: "f32[768, 768]", alias_15: "f32[64, 12, 50, 64]", permute_227: "f32[2304, 768]", div_8: "f32[64, 50, 1]", permute_230: "f32[768, 3072]", permute_234: "f32[3072, 768]", div_9: "f32[64, 50, 1]", permute_239: "f32[768, 768]", alias_16: "f32[64, 12, 50, 64]", permute_250: "f32[2304, 768]", div_10: "f32[64, 50, 1]", permute_253: "f32[768, 3072]", permute_257: "f32[3072, 768]", div_11: "f32[64, 50, 1]", permute_262: "f32[768, 768]", alias_17: "f32[64, 12, 50, 64]", permute_273: "f32[2304, 768]", div_12: "f32[64, 50, 1]", permute_276: "f32[768, 3072]", permute_280: "f32[3072, 768]", div_13: "f32[64, 50, 1]", permute_285: "f32[768, 768]", alias_18: "f32[64, 12, 50, 64]", permute_296: "f32[2304, 768]", div_14: "f32[64, 50, 1]", permute_299: "f32[768, 3072]", permute_303: "f32[3072, 768]", div_15: "f32[64, 50, 1]", permute_308: "f32[768, 768]", alias_19: "f32[64, 12, 50, 64]", permute_319: "f32[2304, 768]", div_16: "f32[64, 50, 1]", permute_322: "f32[768, 3072]", permute_326: "f32[3072, 768]", div_17: "f32[64, 50, 1]", permute_331: "f32[768, 768]", alias_20: "f32[64, 12, 50, 64]", permute_342: "f32[2304, 768]", div_18: "f32[64, 50, 1]", permute_345: "f32[768, 3072]", permute_349: "f32[3072, 768]", div_19: "f32[64, 50, 1]", permute_354: "f32[768, 768]", alias_21: "f32[64, 12, 50, 64]", permute_365: "f32[2304, 768]", div_20: "f32[64, 50, 1]", permute_368: "f32[768, 3072]", permute_372: "f32[3072, 768]", div_21: "f32[64, 50, 1]", permute_377: "f32[768, 768]", alias_22: "f32[64, 12, 50, 64]", permute_388: "f32[2304, 768]", div_22: "f32[64, 50, 1]", permute_391: "f32[768, 3072]", permute_395: "f32[3072, 768]", div_23: "f32[64, 50, 1]", permute_400: "f32[768, 768]", alias_23: "f32[64, 12, 50, 64]", permute_411: "f32[2304, 768]", div_24: "f32[64, 50, 1]", tangents_1: "f32[64, 1000]"):
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        view_13: "f32[64, 50, 3072]" = torch.ops.aten.reshape.default(addmm_1, [64, 50, 3072]);  addmm_1 = None
        mul_5: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_13, 0.7071067811865476)
        erf: "f32[64, 50, 3072]" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
        add_7: "f32[64, 50, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        view_28: "f32[64, 50, 3072]" = torch.ops.aten.reshape.default(addmm_4, [64, 50, 3072]);  addmm_4 = None
        mul_12: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_28, 0.7071067811865476)
        erf_1: "f32[64, 50, 3072]" = torch.ops.aten.erf.default(mul_12);  mul_12 = None
        add_15: "f32[64, 50, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        view_43: "f32[64, 50, 3072]" = torch.ops.aten.reshape.default(addmm_7, [64, 50, 3072]);  addmm_7 = None
        mul_19: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_43, 0.7071067811865476)
        erf_2: "f32[64, 50, 3072]" = torch.ops.aten.erf.default(mul_19);  mul_19 = None
        add_23: "f32[64, 50, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        view_58: "f32[64, 50, 3072]" = torch.ops.aten.reshape.default(addmm_10, [64, 50, 3072]);  addmm_10 = None
        mul_26: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_58, 0.7071067811865476)
        erf_3: "f32[64, 50, 3072]" = torch.ops.aten.erf.default(mul_26);  mul_26 = None
        add_31: "f32[64, 50, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        view_73: "f32[64, 50, 3072]" = torch.ops.aten.reshape.default(addmm_13, [64, 50, 3072]);  addmm_13 = None
        mul_33: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_73, 0.7071067811865476)
        erf_4: "f32[64, 50, 3072]" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
        add_39: "f32[64, 50, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        view_88: "f32[64, 50, 3072]" = torch.ops.aten.reshape.default(addmm_16, [64, 50, 3072]);  addmm_16 = None
        mul_40: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_88, 0.7071067811865476)
        erf_5: "f32[64, 50, 3072]" = torch.ops.aten.erf.default(mul_40);  mul_40 = None
        add_47: "f32[64, 50, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        view_103: "f32[64, 50, 3072]" = torch.ops.aten.reshape.default(addmm_19, [64, 50, 3072]);  addmm_19 = None
        mul_47: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_103, 0.7071067811865476)
        erf_6: "f32[64, 50, 3072]" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
        add_55: "f32[64, 50, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        view_118: "f32[64, 50, 3072]" = torch.ops.aten.reshape.default(addmm_22, [64, 50, 3072]);  addmm_22 = None
        mul_54: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_118, 0.7071067811865476)
        erf_7: "f32[64, 50, 3072]" = torch.ops.aten.erf.default(mul_54);  mul_54 = None
        add_63: "f32[64, 50, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        view_133: "f32[64, 50, 3072]" = torch.ops.aten.reshape.default(addmm_25, [64, 50, 3072]);  addmm_25 = None
        mul_61: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_133, 0.7071067811865476)
        erf_8: "f32[64, 50, 3072]" = torch.ops.aten.erf.default(mul_61);  mul_61 = None
        add_71: "f32[64, 50, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        view_148: "f32[64, 50, 3072]" = torch.ops.aten.reshape.default(addmm_28, [64, 50, 3072]);  addmm_28 = None
        mul_68: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_148, 0.7071067811865476)
        erf_9: "f32[64, 50, 3072]" = torch.ops.aten.erf.default(mul_68);  mul_68 = None
        add_79: "f32[64, 50, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        view_163: "f32[64, 50, 3072]" = torch.ops.aten.reshape.default(addmm_31, [64, 50, 3072]);  addmm_31 = None
        mul_75: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_163, 0.7071067811865476)
        erf_10: "f32[64, 50, 3072]" = torch.ops.aten.erf.default(mul_75);  mul_75 = None
        add_87: "f32[64, 50, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        view_178: "f32[64, 50, 3072]" = torch.ops.aten.reshape.default(addmm_34, [64, 50, 3072]);  addmm_34 = None
        mul_82: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_178, 0.7071067811865476)
        erf_11: "f32[64, 50, 3072]" = torch.ops.aten.erf.default(mul_82);  mul_82 = None
        add_95: "f32[64, 50, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:303, code: x = self.heads(x)
        mm_12: "f32[64, 768]" = torch.ops.aten.mm.default(tangents_1, permute_134);  permute_134 = None
        permute_135: "f32[1000, 64]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_13: "f32[1000, 768]" = torch.ops.aten.mm.default(permute_135, select_36);  permute_135 = select_36 = None
        permute_136: "f32[768, 1000]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
        sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        view_181: "f32[1000]" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None
        permute_137: "f32[1000, 768]" = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:301, code: x = x[:, 0]
        full_default: "f32[64, 50, 768]" = torch.ops.aten.full.default([64, 50, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter: "f32[64, 50, 768]" = torch.ops.aten.select_scatter.default(full_default, mm_12, 1, 0);  full_default = mm_12 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:157, code: return self.ln(self.layers(self.dropout(input)))
        mul_87: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(select_scatter, primals_149);  primals_149 = None
        mul_88: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_87, 768)
        sum_2: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_87, [2], True)
        mul_89: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_87, mul_84);  mul_87 = None
        sum_3: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_89, [2], True);  mul_89 = None
        mul_90: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_84, sum_3);  sum_3 = None
        sub_26: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(mul_88, sum_2);  mul_88 = sum_2 = None
        sub_27: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(sub_26, mul_90);  sub_26 = mul_90 = None
        mul_91: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(div, sub_27);  div = sub_27 = None
        mul_92: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(select_scatter, mul_84);  mul_84 = None
        sum_4: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_92, [0, 1]);  mul_92 = None
        sum_5: "f32[768]" = torch.ops.aten.sum.dim_IntList(select_scatter, [0, 1]);  select_scatter = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        view_182: "f32[3200, 768]" = torch.ops.aten.reshape.default(mul_91, [3200, 768])
        mm_14: "f32[3200, 3072]" = torch.ops.aten.mm.default(view_182, permute_138);  permute_138 = None
        permute_139: "f32[768, 3200]" = torch.ops.aten.permute.default(view_182, [1, 0])
        mm_15: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_139, view_179);  permute_139 = view_179 = None
        permute_140: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
        sum_6: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_182, [0], True);  view_182 = None
        view_183: "f32[768]" = torch.ops.aten.reshape.default(sum_6, [768]);  sum_6 = None
        permute_141: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
        view_184: "f32[64, 50, 3072]" = torch.ops.aten.reshape.default(mm_14, [64, 50, 3072]);  mm_14 = None
        mul_94: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(add_95, 0.5);  add_95 = None
        mul_95: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_178, view_178)
        mul_96: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(mul_95, -0.5);  mul_95 = None
        exp: "f32[64, 50, 3072]" = torch.ops.aten.exp.default(mul_96);  mul_96 = None
        mul_97: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_98: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_178, mul_97);  view_178 = mul_97 = None
        add_100: "f32[64, 50, 3072]" = torch.ops.aten.add.Tensor(mul_94, mul_98);  mul_94 = mul_98 = None
        mul_99: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_184, add_100);  view_184 = add_100 = None
        view_185: "f32[3200, 3072]" = torch.ops.aten.reshape.default(mul_99, [3200, 3072]);  mul_99 = None
        mm_16: "f32[3200, 768]" = torch.ops.aten.mm.default(view_185, permute_142);  permute_142 = None
        permute_143: "f32[3072, 3200]" = torch.ops.aten.permute.default(view_185, [1, 0])
        mm_17: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_143, view_177);  permute_143 = view_177 = None
        permute_144: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
        sum_7: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_185, [0], True);  view_185 = None
        view_186: "f32[3072]" = torch.ops.aten.reshape.default(sum_7, [3072]);  sum_7 = None
        permute_145: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None
        view_187: "f32[64, 50, 768]" = torch.ops.aten.reshape.default(mm_16, [64, 50, 768]);  mm_16 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:117, code: y = self.ln_2(x)
        mul_101: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(view_187, primals_143);  primals_143 = None
        mul_102: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_101, 768)
        sum_8: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_101, [2], True)
        mul_103: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_101, mul_79);  mul_101 = None
        sum_9: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_103, [2], True);  mul_103 = None
        mul_104: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_79, sum_9);  sum_9 = None
        sub_29: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(mul_102, sum_8);  mul_102 = sum_8 = None
        sub_30: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(sub_29, mul_104);  sub_29 = mul_104 = None
        mul_105: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(div_1, sub_30);  div_1 = sub_30 = None
        mul_106: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(view_187, mul_79);  mul_79 = None
        sum_10: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_106, [0, 1]);  mul_106 = None
        sum_11: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_187, [0, 1]);  view_187 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:117, code: y = self.ln_2(x)
        add_101: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(mul_91, mul_105);  mul_91 = mul_105 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_146: "f32[50, 64, 768]" = torch.ops.aten.permute.default(add_101, [1, 0, 2])
        clone_99: "f32[50, 64, 768]" = torch.ops.aten.clone.default(permute_146, memory_format = torch.contiguous_format);  permute_146 = None
        view_188: "f32[3200, 768]" = torch.ops.aten.reshape.default(clone_99, [3200, 768]);  clone_99 = None
        mm_18: "f32[3200, 768]" = torch.ops.aten.mm.default(view_188, permute_147);  permute_147 = None
        permute_148: "f32[768, 3200]" = torch.ops.aten.permute.default(view_188, [1, 0])
        mm_19: "f32[768, 768]" = torch.ops.aten.mm.default(permute_148, view_175);  permute_148 = view_175 = None
        permute_149: "f32[768, 768]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
        sum_12: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_188, [0], True);  view_188 = None
        view_189: "f32[768]" = torch.ops.aten.reshape.default(sum_12, [768]);  sum_12 = None
        permute_150: "f32[768, 768]" = torch.ops.aten.permute.default(permute_149, [1, 0]);  permute_149 = None
        view_190: "f32[50, 64, 12, 64]" = torch.ops.aten.reshape.default(mm_18, [50, 64, 12, 64]);  mm_18 = None
        permute_151: "f32[64, 12, 50, 64]" = torch.ops.aten.permute.default(view_190, [1, 2, 0, 3]);  view_190 = None
        _scaled_dot_product_efficient_attention_backward = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_151, view_172, view_173, view_174, None, alias_12, getitem_91, getitem_92, getitem_93, 0.0, [True, True, True, False]);  permute_151 = view_172 = view_173 = view_174 = alias_12 = getitem_91 = getitem_92 = getitem_93 = None
        getitem_98: "f32[64, 12, 50, 64]" = _scaled_dot_product_efficient_attention_backward[0]
        getitem_99: "f32[64, 12, 50, 64]" = _scaled_dot_product_efficient_attention_backward[1]
        getitem_100: "f32[64, 12, 50, 64]" = _scaled_dot_product_efficient_attention_backward[2];  _scaled_dot_product_efficient_attention_backward = None
        clone_100: "f32[64, 12, 50, 64]" = torch.ops.aten.clone.default(getitem_100, memory_format = torch.contiguous_format);  getitem_100 = None
        view_191: "f32[768, 50, 64]" = torch.ops.aten.reshape.default(clone_100, [768, 50, 64]);  clone_100 = None
        clone_101: "f32[64, 12, 50, 64]" = torch.ops.aten.clone.default(getitem_99, memory_format = torch.contiguous_format);  getitem_99 = None
        view_192: "f32[768, 50, 64]" = torch.ops.aten.reshape.default(clone_101, [768, 50, 64]);  clone_101 = None
        clone_102: "f32[64, 12, 50, 64]" = torch.ops.aten.clone.default(getitem_98, memory_format = torch.contiguous_format);  getitem_98 = None
        view_193: "f32[768, 50, 64]" = torch.ops.aten.reshape.default(clone_102, [768, 50, 64]);  clone_102 = None
        permute_152: "f32[50, 768, 64]" = torch.ops.aten.permute.default(view_191, [1, 0, 2]);  view_191 = None
        clone_103: "f32[50, 768, 64]" = torch.ops.aten.clone.default(permute_152, memory_format = torch.contiguous_format);  permute_152 = None
        view_194: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(clone_103, [50, 64, 768]);  clone_103 = None
        permute_153: "f32[50, 768, 64]" = torch.ops.aten.permute.default(view_192, [1, 0, 2]);  view_192 = None
        clone_104: "f32[50, 768, 64]" = torch.ops.aten.clone.default(permute_153, memory_format = torch.contiguous_format);  permute_153 = None
        view_195: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(clone_104, [50, 64, 768]);  clone_104 = None
        permute_154: "f32[50, 768, 64]" = torch.ops.aten.permute.default(view_193, [1, 0, 2]);  view_193 = None
        clone_105: "f32[50, 768, 64]" = torch.ops.aten.clone.default(permute_154, memory_format = torch.contiguous_format);  permute_154 = None
        view_196: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(clone_105, [50, 64, 768]);  clone_105 = None
        full_default_2: "f32[3, 50, 64, 768]" = torch.ops.aten.full.default([3, 50, 64, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_1: "f32[3, 50, 64, 768]" = torch.ops.aten.select_scatter.default(full_default_2, view_194, 0, 2);  view_194 = None
        select_scatter_2: "f32[3, 50, 64, 768]" = torch.ops.aten.select_scatter.default(full_default_2, view_195, 0, 1);  view_195 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        add_102: "f32[3, 50, 64, 768]" = torch.ops.aten.add.Tensor(select_scatter_1, select_scatter_2);  select_scatter_1 = select_scatter_2 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        select_scatter_3: "f32[3, 50, 64, 768]" = torch.ops.aten.select_scatter.default(full_default_2, view_196, 0, 0);  view_196 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        add_103: "f32[3, 50, 64, 768]" = torch.ops.aten.add.Tensor(add_102, select_scatter_3);  add_102 = select_scatter_3 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        unsqueeze_12: "f32[3, 50, 64, 1, 768]" = torch.ops.aten.unsqueeze.default(add_103, 3);  add_103 = None
        permute_155: "f32[1, 50, 64, 3, 768]" = torch.ops.aten.permute.default(unsqueeze_12, [3, 1, 2, 0, 4]);  unsqueeze_12 = None
        squeeze_12: "f32[50, 64, 3, 768]" = torch.ops.aten.squeeze.dim(permute_155, 0);  permute_155 = None
        clone_106: "f32[50, 64, 3, 768]" = torch.ops.aten.clone.default(squeeze_12, memory_format = torch.contiguous_format);  squeeze_12 = None
        view_197: "f32[50, 64, 2304]" = torch.ops.aten.reshape.default(clone_106, [50, 64, 2304]);  clone_106 = None
        sum_13: "f32[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_197, [0, 1], True)
        view_198: "f32[2304]" = torch.ops.aten.reshape.default(sum_13, [2304]);  sum_13 = None
        view_199: "f32[3200, 2304]" = torch.ops.aten.reshape.default(view_197, [3200, 2304]);  view_197 = None
        permute_156: "f32[2304, 3200]" = torch.ops.aten.permute.default(view_199, [1, 0])
        mm_20: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_156, view_166);  permute_156 = view_166 = None
        permute_157: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_20, [1, 0]);  mm_20 = None
        mm_21: "f32[3200, 768]" = torch.ops.aten.mm.default(view_199, permute_158);  view_199 = permute_158 = None
        view_200: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(mm_21, [50, 64, 768]);  mm_21 = None
        permute_159: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_157, [1, 0]);  permute_157 = None
        permute_160: "f32[64, 50, 768]" = torch.ops.aten.permute.default(view_200, [1, 0, 2]);  view_200 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:112, code: x = self.ln_1(input)
        clone_107: "f32[64, 50, 768]" = torch.ops.aten.clone.default(permute_160, memory_format = torch.contiguous_format);  permute_160 = None
        mul_108: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(clone_107, primals_137);  primals_137 = None
        mul_109: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_108, 768)
        sum_14: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_108, [2], True)
        mul_110: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_108, mul_77);  mul_108 = None
        sum_15: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_110, [2], True);  mul_110 = None
        mul_111: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_77, sum_15);  sum_15 = None
        sub_32: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(mul_109, sum_14);  mul_109 = sum_14 = None
        sub_33: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(sub_32, mul_111);  sub_32 = mul_111 = None
        mul_112: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(div_2, sub_33);  div_2 = sub_33 = None
        mul_113: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(clone_107, mul_77);  mul_77 = None
        sum_16: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_113, [0, 1]);  mul_113 = None
        sum_17: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_107, [0, 1]);  clone_107 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:112, code: x = self.ln_1(input)
        add_104: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(add_101, mul_112);  add_101 = mul_112 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        view_201: "f32[3200, 768]" = torch.ops.aten.reshape.default(add_104, [3200, 768])
        mm_22: "f32[3200, 3072]" = torch.ops.aten.mm.default(view_201, permute_161);  permute_161 = None
        permute_162: "f32[768, 3200]" = torch.ops.aten.permute.default(view_201, [1, 0])
        mm_23: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_162, view_164);  permute_162 = view_164 = None
        permute_163: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
        sum_18: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_201, [0], True);  view_201 = None
        view_202: "f32[768]" = torch.ops.aten.reshape.default(sum_18, [768]);  sum_18 = None
        permute_164: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_163, [1, 0]);  permute_163 = None
        view_203: "f32[64, 50, 3072]" = torch.ops.aten.reshape.default(mm_22, [64, 50, 3072]);  mm_22 = None
        mul_115: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(add_87, 0.5);  add_87 = None
        mul_116: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_163, view_163)
        mul_117: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(mul_116, -0.5);  mul_116 = None
        exp_1: "f32[64, 50, 3072]" = torch.ops.aten.exp.default(mul_117);  mul_117 = None
        mul_118: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
        mul_119: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_163, mul_118);  view_163 = mul_118 = None
        add_106: "f32[64, 50, 3072]" = torch.ops.aten.add.Tensor(mul_115, mul_119);  mul_115 = mul_119 = None
        mul_120: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_203, add_106);  view_203 = add_106 = None
        view_204: "f32[3200, 3072]" = torch.ops.aten.reshape.default(mul_120, [3200, 3072]);  mul_120 = None
        mm_24: "f32[3200, 768]" = torch.ops.aten.mm.default(view_204, permute_165);  permute_165 = None
        permute_166: "f32[3072, 3200]" = torch.ops.aten.permute.default(view_204, [1, 0])
        mm_25: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_166, view_162);  permute_166 = view_162 = None
        permute_167: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
        sum_19: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_204, [0], True);  view_204 = None
        view_205: "f32[3072]" = torch.ops.aten.reshape.default(sum_19, [3072]);  sum_19 = None
        permute_168: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_167, [1, 0]);  permute_167 = None
        view_206: "f32[64, 50, 768]" = torch.ops.aten.reshape.default(mm_24, [64, 50, 768]);  mm_24 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:117, code: y = self.ln_2(x)
        mul_122: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(view_206, primals_131);  primals_131 = None
        mul_123: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_122, 768)
        sum_20: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_122, [2], True)
        mul_124: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_122, mul_72);  mul_122 = None
        sum_21: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_124, [2], True);  mul_124 = None
        mul_125: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_72, sum_21);  sum_21 = None
        sub_35: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(mul_123, sum_20);  mul_123 = sum_20 = None
        sub_36: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(sub_35, mul_125);  sub_35 = mul_125 = None
        mul_126: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(div_3, sub_36);  div_3 = sub_36 = None
        mul_127: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(view_206, mul_72);  mul_72 = None
        sum_22: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_127, [0, 1]);  mul_127 = None
        sum_23: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_206, [0, 1]);  view_206 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:117, code: y = self.ln_2(x)
        add_107: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(add_104, mul_126);  add_104 = mul_126 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_169: "f32[50, 64, 768]" = torch.ops.aten.permute.default(add_107, [1, 0, 2])
        clone_110: "f32[50, 64, 768]" = torch.ops.aten.clone.default(permute_169, memory_format = torch.contiguous_format);  permute_169 = None
        view_207: "f32[3200, 768]" = torch.ops.aten.reshape.default(clone_110, [3200, 768]);  clone_110 = None
        mm_26: "f32[3200, 768]" = torch.ops.aten.mm.default(view_207, permute_170);  permute_170 = None
        permute_171: "f32[768, 3200]" = torch.ops.aten.permute.default(view_207, [1, 0])
        mm_27: "f32[768, 768]" = torch.ops.aten.mm.default(permute_171, view_160);  permute_171 = view_160 = None
        permute_172: "f32[768, 768]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
        sum_24: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_207, [0], True);  view_207 = None
        view_208: "f32[768]" = torch.ops.aten.reshape.default(sum_24, [768]);  sum_24 = None
        permute_173: "f32[768, 768]" = torch.ops.aten.permute.default(permute_172, [1, 0]);  permute_172 = None
        view_209: "f32[50, 64, 12, 64]" = torch.ops.aten.reshape.default(mm_26, [50, 64, 12, 64]);  mm_26 = None
        permute_174: "f32[64, 12, 50, 64]" = torch.ops.aten.permute.default(view_209, [1, 2, 0, 3]);  view_209 = None
        _scaled_dot_product_efficient_attention_backward_1 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_174, view_157, view_158, view_159, None, alias_13, getitem_83, getitem_84, getitem_85, 0.0, [True, True, True, False]);  permute_174 = view_157 = view_158 = view_159 = alias_13 = getitem_83 = getitem_84 = getitem_85 = None
        getitem_102: "f32[64, 12, 50, 64]" = _scaled_dot_product_efficient_attention_backward_1[0]
        getitem_103: "f32[64, 12, 50, 64]" = _scaled_dot_product_efficient_attention_backward_1[1]
        getitem_104: "f32[64, 12, 50, 64]" = _scaled_dot_product_efficient_attention_backward_1[2];  _scaled_dot_product_efficient_attention_backward_1 = None
        clone_111: "f32[64, 12, 50, 64]" = torch.ops.aten.clone.default(getitem_104, memory_format = torch.contiguous_format);  getitem_104 = None
        view_210: "f32[768, 50, 64]" = torch.ops.aten.reshape.default(clone_111, [768, 50, 64]);  clone_111 = None
        clone_112: "f32[64, 12, 50, 64]" = torch.ops.aten.clone.default(getitem_103, memory_format = torch.contiguous_format);  getitem_103 = None
        view_211: "f32[768, 50, 64]" = torch.ops.aten.reshape.default(clone_112, [768, 50, 64]);  clone_112 = None
        clone_113: "f32[64, 12, 50, 64]" = torch.ops.aten.clone.default(getitem_102, memory_format = torch.contiguous_format);  getitem_102 = None
        view_212: "f32[768, 50, 64]" = torch.ops.aten.reshape.default(clone_113, [768, 50, 64]);  clone_113 = None
        permute_175: "f32[50, 768, 64]" = torch.ops.aten.permute.default(view_210, [1, 0, 2]);  view_210 = None
        clone_114: "f32[50, 768, 64]" = torch.ops.aten.clone.default(permute_175, memory_format = torch.contiguous_format);  permute_175 = None
        view_213: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(clone_114, [50, 64, 768]);  clone_114 = None
        permute_176: "f32[50, 768, 64]" = torch.ops.aten.permute.default(view_211, [1, 0, 2]);  view_211 = None
        clone_115: "f32[50, 768, 64]" = torch.ops.aten.clone.default(permute_176, memory_format = torch.contiguous_format);  permute_176 = None
        view_214: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(clone_115, [50, 64, 768]);  clone_115 = None
        permute_177: "f32[50, 768, 64]" = torch.ops.aten.permute.default(view_212, [1, 0, 2]);  view_212 = None
        clone_116: "f32[50, 768, 64]" = torch.ops.aten.clone.default(permute_177, memory_format = torch.contiguous_format);  permute_177 = None
        view_215: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(clone_116, [50, 64, 768]);  clone_116 = None
        select_scatter_4: "f32[3, 50, 64, 768]" = torch.ops.aten.select_scatter.default(full_default_2, view_213, 0, 2);  view_213 = None
        select_scatter_5: "f32[3, 50, 64, 768]" = torch.ops.aten.select_scatter.default(full_default_2, view_214, 0, 1);  view_214 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        add_108: "f32[3, 50, 64, 768]" = torch.ops.aten.add.Tensor(select_scatter_4, select_scatter_5);  select_scatter_4 = select_scatter_5 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        select_scatter_6: "f32[3, 50, 64, 768]" = torch.ops.aten.select_scatter.default(full_default_2, view_215, 0, 0);  view_215 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        add_109: "f32[3, 50, 64, 768]" = torch.ops.aten.add.Tensor(add_108, select_scatter_6);  add_108 = select_scatter_6 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        unsqueeze_13: "f32[3, 50, 64, 1, 768]" = torch.ops.aten.unsqueeze.default(add_109, 3);  add_109 = None
        permute_178: "f32[1, 50, 64, 3, 768]" = torch.ops.aten.permute.default(unsqueeze_13, [3, 1, 2, 0, 4]);  unsqueeze_13 = None
        squeeze_13: "f32[50, 64, 3, 768]" = torch.ops.aten.squeeze.dim(permute_178, 0);  permute_178 = None
        clone_117: "f32[50, 64, 3, 768]" = torch.ops.aten.clone.default(squeeze_13, memory_format = torch.contiguous_format);  squeeze_13 = None
        view_216: "f32[50, 64, 2304]" = torch.ops.aten.reshape.default(clone_117, [50, 64, 2304]);  clone_117 = None
        sum_25: "f32[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_216, [0, 1], True)
        view_217: "f32[2304]" = torch.ops.aten.reshape.default(sum_25, [2304]);  sum_25 = None
        view_218: "f32[3200, 2304]" = torch.ops.aten.reshape.default(view_216, [3200, 2304]);  view_216 = None
        permute_179: "f32[2304, 3200]" = torch.ops.aten.permute.default(view_218, [1, 0])
        mm_28: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_179, view_151);  permute_179 = view_151 = None
        permute_180: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_28, [1, 0]);  mm_28 = None
        mm_29: "f32[3200, 768]" = torch.ops.aten.mm.default(view_218, permute_181);  view_218 = permute_181 = None
        view_219: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(mm_29, [50, 64, 768]);  mm_29 = None
        permute_182: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_180, [1, 0]);  permute_180 = None
        permute_183: "f32[64, 50, 768]" = torch.ops.aten.permute.default(view_219, [1, 0, 2]);  view_219 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:112, code: x = self.ln_1(input)
        clone_118: "f32[64, 50, 768]" = torch.ops.aten.clone.default(permute_183, memory_format = torch.contiguous_format);  permute_183 = None
        mul_129: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(clone_118, primals_125);  primals_125 = None
        mul_130: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_129, 768)
        sum_26: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_129, [2], True)
        mul_131: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_129, mul_70);  mul_129 = None
        sum_27: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_131, [2], True);  mul_131 = None
        mul_132: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_70, sum_27);  sum_27 = None
        sub_38: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(mul_130, sum_26);  mul_130 = sum_26 = None
        sub_39: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(sub_38, mul_132);  sub_38 = mul_132 = None
        mul_133: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(div_4, sub_39);  div_4 = sub_39 = None
        mul_134: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(clone_118, mul_70);  mul_70 = None
        sum_28: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_134, [0, 1]);  mul_134 = None
        sum_29: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_118, [0, 1]);  clone_118 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:112, code: x = self.ln_1(input)
        add_110: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(add_107, mul_133);  add_107 = mul_133 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        view_220: "f32[3200, 768]" = torch.ops.aten.reshape.default(add_110, [3200, 768])
        mm_30: "f32[3200, 3072]" = torch.ops.aten.mm.default(view_220, permute_184);  permute_184 = None
        permute_185: "f32[768, 3200]" = torch.ops.aten.permute.default(view_220, [1, 0])
        mm_31: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_185, view_149);  permute_185 = view_149 = None
        permute_186: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
        sum_30: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_220, [0], True);  view_220 = None
        view_221: "f32[768]" = torch.ops.aten.reshape.default(sum_30, [768]);  sum_30 = None
        permute_187: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
        view_222: "f32[64, 50, 3072]" = torch.ops.aten.reshape.default(mm_30, [64, 50, 3072]);  mm_30 = None
        mul_136: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(add_79, 0.5);  add_79 = None
        mul_137: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_148, view_148)
        mul_138: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(mul_137, -0.5);  mul_137 = None
        exp_2: "f32[64, 50, 3072]" = torch.ops.aten.exp.default(mul_138);  mul_138 = None
        mul_139: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(exp_2, 0.3989422804014327);  exp_2 = None
        mul_140: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_148, mul_139);  view_148 = mul_139 = None
        add_112: "f32[64, 50, 3072]" = torch.ops.aten.add.Tensor(mul_136, mul_140);  mul_136 = mul_140 = None
        mul_141: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_222, add_112);  view_222 = add_112 = None
        view_223: "f32[3200, 3072]" = torch.ops.aten.reshape.default(mul_141, [3200, 3072]);  mul_141 = None
        mm_32: "f32[3200, 768]" = torch.ops.aten.mm.default(view_223, permute_188);  permute_188 = None
        permute_189: "f32[3072, 3200]" = torch.ops.aten.permute.default(view_223, [1, 0])
        mm_33: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_189, view_147);  permute_189 = view_147 = None
        permute_190: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
        sum_31: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_223, [0], True);  view_223 = None
        view_224: "f32[3072]" = torch.ops.aten.reshape.default(sum_31, [3072]);  sum_31 = None
        permute_191: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_190, [1, 0]);  permute_190 = None
        view_225: "f32[64, 50, 768]" = torch.ops.aten.reshape.default(mm_32, [64, 50, 768]);  mm_32 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:117, code: y = self.ln_2(x)
        mul_143: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(view_225, primals_119);  primals_119 = None
        mul_144: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_143, 768)
        sum_32: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_143, [2], True)
        mul_145: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_143, mul_65);  mul_143 = None
        sum_33: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_145, [2], True);  mul_145 = None
        mul_146: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_65, sum_33);  sum_33 = None
        sub_41: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(mul_144, sum_32);  mul_144 = sum_32 = None
        sub_42: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(sub_41, mul_146);  sub_41 = mul_146 = None
        mul_147: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(div_5, sub_42);  div_5 = sub_42 = None
        mul_148: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(view_225, mul_65);  mul_65 = None
        sum_34: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_148, [0, 1]);  mul_148 = None
        sum_35: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_225, [0, 1]);  view_225 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:117, code: y = self.ln_2(x)
        add_113: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(add_110, mul_147);  add_110 = mul_147 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_192: "f32[50, 64, 768]" = torch.ops.aten.permute.default(add_113, [1, 0, 2])
        clone_121: "f32[50, 64, 768]" = torch.ops.aten.clone.default(permute_192, memory_format = torch.contiguous_format);  permute_192 = None
        view_226: "f32[3200, 768]" = torch.ops.aten.reshape.default(clone_121, [3200, 768]);  clone_121 = None
        mm_34: "f32[3200, 768]" = torch.ops.aten.mm.default(view_226, permute_193);  permute_193 = None
        permute_194: "f32[768, 3200]" = torch.ops.aten.permute.default(view_226, [1, 0])
        mm_35: "f32[768, 768]" = torch.ops.aten.mm.default(permute_194, view_145);  permute_194 = view_145 = None
        permute_195: "f32[768, 768]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
        sum_36: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_226, [0], True);  view_226 = None
        view_227: "f32[768]" = torch.ops.aten.reshape.default(sum_36, [768]);  sum_36 = None
        permute_196: "f32[768, 768]" = torch.ops.aten.permute.default(permute_195, [1, 0]);  permute_195 = None
        view_228: "f32[50, 64, 12, 64]" = torch.ops.aten.reshape.default(mm_34, [50, 64, 12, 64]);  mm_34 = None
        permute_197: "f32[64, 12, 50, 64]" = torch.ops.aten.permute.default(view_228, [1, 2, 0, 3]);  view_228 = None
        _scaled_dot_product_efficient_attention_backward_2 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_197, view_142, view_143, view_144, None, alias_14, getitem_75, getitem_76, getitem_77, 0.0, [True, True, True, False]);  permute_197 = view_142 = view_143 = view_144 = alias_14 = getitem_75 = getitem_76 = getitem_77 = None
        getitem_106: "f32[64, 12, 50, 64]" = _scaled_dot_product_efficient_attention_backward_2[0]
        getitem_107: "f32[64, 12, 50, 64]" = _scaled_dot_product_efficient_attention_backward_2[1]
        getitem_108: "f32[64, 12, 50, 64]" = _scaled_dot_product_efficient_attention_backward_2[2];  _scaled_dot_product_efficient_attention_backward_2 = None
        clone_122: "f32[64, 12, 50, 64]" = torch.ops.aten.clone.default(getitem_108, memory_format = torch.contiguous_format);  getitem_108 = None
        view_229: "f32[768, 50, 64]" = torch.ops.aten.reshape.default(clone_122, [768, 50, 64]);  clone_122 = None
        clone_123: "f32[64, 12, 50, 64]" = torch.ops.aten.clone.default(getitem_107, memory_format = torch.contiguous_format);  getitem_107 = None
        view_230: "f32[768, 50, 64]" = torch.ops.aten.reshape.default(clone_123, [768, 50, 64]);  clone_123 = None
        clone_124: "f32[64, 12, 50, 64]" = torch.ops.aten.clone.default(getitem_106, memory_format = torch.contiguous_format);  getitem_106 = None
        view_231: "f32[768, 50, 64]" = torch.ops.aten.reshape.default(clone_124, [768, 50, 64]);  clone_124 = None
        permute_198: "f32[50, 768, 64]" = torch.ops.aten.permute.default(view_229, [1, 0, 2]);  view_229 = None
        clone_125: "f32[50, 768, 64]" = torch.ops.aten.clone.default(permute_198, memory_format = torch.contiguous_format);  permute_198 = None
        view_232: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(clone_125, [50, 64, 768]);  clone_125 = None
        permute_199: "f32[50, 768, 64]" = torch.ops.aten.permute.default(view_230, [1, 0, 2]);  view_230 = None
        clone_126: "f32[50, 768, 64]" = torch.ops.aten.clone.default(permute_199, memory_format = torch.contiguous_format);  permute_199 = None
        view_233: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(clone_126, [50, 64, 768]);  clone_126 = None
        permute_200: "f32[50, 768, 64]" = torch.ops.aten.permute.default(view_231, [1, 0, 2]);  view_231 = None
        clone_127: "f32[50, 768, 64]" = torch.ops.aten.clone.default(permute_200, memory_format = torch.contiguous_format);  permute_200 = None
        view_234: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(clone_127, [50, 64, 768]);  clone_127 = None
        select_scatter_7: "f32[3, 50, 64, 768]" = torch.ops.aten.select_scatter.default(full_default_2, view_232, 0, 2);  view_232 = None
        select_scatter_8: "f32[3, 50, 64, 768]" = torch.ops.aten.select_scatter.default(full_default_2, view_233, 0, 1);  view_233 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        add_114: "f32[3, 50, 64, 768]" = torch.ops.aten.add.Tensor(select_scatter_7, select_scatter_8);  select_scatter_7 = select_scatter_8 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        select_scatter_9: "f32[3, 50, 64, 768]" = torch.ops.aten.select_scatter.default(full_default_2, view_234, 0, 0);  view_234 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        add_115: "f32[3, 50, 64, 768]" = torch.ops.aten.add.Tensor(add_114, select_scatter_9);  add_114 = select_scatter_9 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        unsqueeze_14: "f32[3, 50, 64, 1, 768]" = torch.ops.aten.unsqueeze.default(add_115, 3);  add_115 = None
        permute_201: "f32[1, 50, 64, 3, 768]" = torch.ops.aten.permute.default(unsqueeze_14, [3, 1, 2, 0, 4]);  unsqueeze_14 = None
        squeeze_14: "f32[50, 64, 3, 768]" = torch.ops.aten.squeeze.dim(permute_201, 0);  permute_201 = None
        clone_128: "f32[50, 64, 3, 768]" = torch.ops.aten.clone.default(squeeze_14, memory_format = torch.contiguous_format);  squeeze_14 = None
        view_235: "f32[50, 64, 2304]" = torch.ops.aten.reshape.default(clone_128, [50, 64, 2304]);  clone_128 = None
        sum_37: "f32[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_235, [0, 1], True)
        view_236: "f32[2304]" = torch.ops.aten.reshape.default(sum_37, [2304]);  sum_37 = None
        view_237: "f32[3200, 2304]" = torch.ops.aten.reshape.default(view_235, [3200, 2304]);  view_235 = None
        permute_202: "f32[2304, 3200]" = torch.ops.aten.permute.default(view_237, [1, 0])
        mm_36: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_202, view_136);  permute_202 = view_136 = None
        permute_203: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_36, [1, 0]);  mm_36 = None
        mm_37: "f32[3200, 768]" = torch.ops.aten.mm.default(view_237, permute_204);  view_237 = permute_204 = None
        view_238: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(mm_37, [50, 64, 768]);  mm_37 = None
        permute_205: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_203, [1, 0]);  permute_203 = None
        permute_206: "f32[64, 50, 768]" = torch.ops.aten.permute.default(view_238, [1, 0, 2]);  view_238 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:112, code: x = self.ln_1(input)
        clone_129: "f32[64, 50, 768]" = torch.ops.aten.clone.default(permute_206, memory_format = torch.contiguous_format);  permute_206 = None
        mul_150: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(clone_129, primals_113);  primals_113 = None
        mul_151: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_150, 768)
        sum_38: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_150, [2], True)
        mul_152: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_150, mul_63);  mul_150 = None
        sum_39: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_152, [2], True);  mul_152 = None
        mul_153: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_63, sum_39);  sum_39 = None
        sub_44: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(mul_151, sum_38);  mul_151 = sum_38 = None
        sub_45: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(sub_44, mul_153);  sub_44 = mul_153 = None
        mul_154: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(div_6, sub_45);  div_6 = sub_45 = None
        mul_155: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(clone_129, mul_63);  mul_63 = None
        sum_40: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_155, [0, 1]);  mul_155 = None
        sum_41: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_129, [0, 1]);  clone_129 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:112, code: x = self.ln_1(input)
        add_116: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(add_113, mul_154);  add_113 = mul_154 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        view_239: "f32[3200, 768]" = torch.ops.aten.reshape.default(add_116, [3200, 768])
        mm_38: "f32[3200, 3072]" = torch.ops.aten.mm.default(view_239, permute_207);  permute_207 = None
        permute_208: "f32[768, 3200]" = torch.ops.aten.permute.default(view_239, [1, 0])
        mm_39: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_208, view_134);  permute_208 = view_134 = None
        permute_209: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
        sum_42: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_239, [0], True);  view_239 = None
        view_240: "f32[768]" = torch.ops.aten.reshape.default(sum_42, [768]);  sum_42 = None
        permute_210: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_209, [1, 0]);  permute_209 = None
        view_241: "f32[64, 50, 3072]" = torch.ops.aten.reshape.default(mm_38, [64, 50, 3072]);  mm_38 = None
        mul_157: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(add_71, 0.5);  add_71 = None
        mul_158: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_133, view_133)
        mul_159: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(mul_158, -0.5);  mul_158 = None
        exp_3: "f32[64, 50, 3072]" = torch.ops.aten.exp.default(mul_159);  mul_159 = None
        mul_160: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(exp_3, 0.3989422804014327);  exp_3 = None
        mul_161: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_133, mul_160);  view_133 = mul_160 = None
        add_118: "f32[64, 50, 3072]" = torch.ops.aten.add.Tensor(mul_157, mul_161);  mul_157 = mul_161 = None
        mul_162: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_241, add_118);  view_241 = add_118 = None
        view_242: "f32[3200, 3072]" = torch.ops.aten.reshape.default(mul_162, [3200, 3072]);  mul_162 = None
        mm_40: "f32[3200, 768]" = torch.ops.aten.mm.default(view_242, permute_211);  permute_211 = None
        permute_212: "f32[3072, 3200]" = torch.ops.aten.permute.default(view_242, [1, 0])
        mm_41: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_212, view_132);  permute_212 = view_132 = None
        permute_213: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
        sum_43: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_242, [0], True);  view_242 = None
        view_243: "f32[3072]" = torch.ops.aten.reshape.default(sum_43, [3072]);  sum_43 = None
        permute_214: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_213, [1, 0]);  permute_213 = None
        view_244: "f32[64, 50, 768]" = torch.ops.aten.reshape.default(mm_40, [64, 50, 768]);  mm_40 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:117, code: y = self.ln_2(x)
        mul_164: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(view_244, primals_107);  primals_107 = None
        mul_165: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_164, 768)
        sum_44: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_164, [2], True)
        mul_166: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_164, mul_58);  mul_164 = None
        sum_45: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_166, [2], True);  mul_166 = None
        mul_167: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_58, sum_45);  sum_45 = None
        sub_47: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(mul_165, sum_44);  mul_165 = sum_44 = None
        sub_48: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(sub_47, mul_167);  sub_47 = mul_167 = None
        mul_168: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(div_7, sub_48);  div_7 = sub_48 = None
        mul_169: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(view_244, mul_58);  mul_58 = None
        sum_46: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_169, [0, 1]);  mul_169 = None
        sum_47: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_244, [0, 1]);  view_244 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:117, code: y = self.ln_2(x)
        add_119: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(add_116, mul_168);  add_116 = mul_168 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_215: "f32[50, 64, 768]" = torch.ops.aten.permute.default(add_119, [1, 0, 2])
        clone_132: "f32[50, 64, 768]" = torch.ops.aten.clone.default(permute_215, memory_format = torch.contiguous_format);  permute_215 = None
        view_245: "f32[3200, 768]" = torch.ops.aten.reshape.default(clone_132, [3200, 768]);  clone_132 = None
        mm_42: "f32[3200, 768]" = torch.ops.aten.mm.default(view_245, permute_216);  permute_216 = None
        permute_217: "f32[768, 3200]" = torch.ops.aten.permute.default(view_245, [1, 0])
        mm_43: "f32[768, 768]" = torch.ops.aten.mm.default(permute_217, view_130);  permute_217 = view_130 = None
        permute_218: "f32[768, 768]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
        sum_48: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_245, [0], True);  view_245 = None
        view_246: "f32[768]" = torch.ops.aten.reshape.default(sum_48, [768]);  sum_48 = None
        permute_219: "f32[768, 768]" = torch.ops.aten.permute.default(permute_218, [1, 0]);  permute_218 = None
        view_247: "f32[50, 64, 12, 64]" = torch.ops.aten.reshape.default(mm_42, [50, 64, 12, 64]);  mm_42 = None
        permute_220: "f32[64, 12, 50, 64]" = torch.ops.aten.permute.default(view_247, [1, 2, 0, 3]);  view_247 = None
        _scaled_dot_product_efficient_attention_backward_3 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_220, view_127, view_128, view_129, None, alias_15, getitem_67, getitem_68, getitem_69, 0.0, [True, True, True, False]);  permute_220 = view_127 = view_128 = view_129 = alias_15 = getitem_67 = getitem_68 = getitem_69 = None
        getitem_110: "f32[64, 12, 50, 64]" = _scaled_dot_product_efficient_attention_backward_3[0]
        getitem_111: "f32[64, 12, 50, 64]" = _scaled_dot_product_efficient_attention_backward_3[1]
        getitem_112: "f32[64, 12, 50, 64]" = _scaled_dot_product_efficient_attention_backward_3[2];  _scaled_dot_product_efficient_attention_backward_3 = None
        clone_133: "f32[64, 12, 50, 64]" = torch.ops.aten.clone.default(getitem_112, memory_format = torch.contiguous_format);  getitem_112 = None
        view_248: "f32[768, 50, 64]" = torch.ops.aten.reshape.default(clone_133, [768, 50, 64]);  clone_133 = None
        clone_134: "f32[64, 12, 50, 64]" = torch.ops.aten.clone.default(getitem_111, memory_format = torch.contiguous_format);  getitem_111 = None
        view_249: "f32[768, 50, 64]" = torch.ops.aten.reshape.default(clone_134, [768, 50, 64]);  clone_134 = None
        clone_135: "f32[64, 12, 50, 64]" = torch.ops.aten.clone.default(getitem_110, memory_format = torch.contiguous_format);  getitem_110 = None
        view_250: "f32[768, 50, 64]" = torch.ops.aten.reshape.default(clone_135, [768, 50, 64]);  clone_135 = None
        permute_221: "f32[50, 768, 64]" = torch.ops.aten.permute.default(view_248, [1, 0, 2]);  view_248 = None
        clone_136: "f32[50, 768, 64]" = torch.ops.aten.clone.default(permute_221, memory_format = torch.contiguous_format);  permute_221 = None
        view_251: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(clone_136, [50, 64, 768]);  clone_136 = None
        permute_222: "f32[50, 768, 64]" = torch.ops.aten.permute.default(view_249, [1, 0, 2]);  view_249 = None
        clone_137: "f32[50, 768, 64]" = torch.ops.aten.clone.default(permute_222, memory_format = torch.contiguous_format);  permute_222 = None
        view_252: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(clone_137, [50, 64, 768]);  clone_137 = None
        permute_223: "f32[50, 768, 64]" = torch.ops.aten.permute.default(view_250, [1, 0, 2]);  view_250 = None
        clone_138: "f32[50, 768, 64]" = torch.ops.aten.clone.default(permute_223, memory_format = torch.contiguous_format);  permute_223 = None
        view_253: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(clone_138, [50, 64, 768]);  clone_138 = None
        select_scatter_10: "f32[3, 50, 64, 768]" = torch.ops.aten.select_scatter.default(full_default_2, view_251, 0, 2);  view_251 = None
        select_scatter_11: "f32[3, 50, 64, 768]" = torch.ops.aten.select_scatter.default(full_default_2, view_252, 0, 1);  view_252 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        add_120: "f32[3, 50, 64, 768]" = torch.ops.aten.add.Tensor(select_scatter_10, select_scatter_11);  select_scatter_10 = select_scatter_11 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        select_scatter_12: "f32[3, 50, 64, 768]" = torch.ops.aten.select_scatter.default(full_default_2, view_253, 0, 0);  view_253 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        add_121: "f32[3, 50, 64, 768]" = torch.ops.aten.add.Tensor(add_120, select_scatter_12);  add_120 = select_scatter_12 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        unsqueeze_15: "f32[3, 50, 64, 1, 768]" = torch.ops.aten.unsqueeze.default(add_121, 3);  add_121 = None
        permute_224: "f32[1, 50, 64, 3, 768]" = torch.ops.aten.permute.default(unsqueeze_15, [3, 1, 2, 0, 4]);  unsqueeze_15 = None
        squeeze_15: "f32[50, 64, 3, 768]" = torch.ops.aten.squeeze.dim(permute_224, 0);  permute_224 = None
        clone_139: "f32[50, 64, 3, 768]" = torch.ops.aten.clone.default(squeeze_15, memory_format = torch.contiguous_format);  squeeze_15 = None
        view_254: "f32[50, 64, 2304]" = torch.ops.aten.reshape.default(clone_139, [50, 64, 2304]);  clone_139 = None
        sum_49: "f32[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_254, [0, 1], True)
        view_255: "f32[2304]" = torch.ops.aten.reshape.default(sum_49, [2304]);  sum_49 = None
        view_256: "f32[3200, 2304]" = torch.ops.aten.reshape.default(view_254, [3200, 2304]);  view_254 = None
        permute_225: "f32[2304, 3200]" = torch.ops.aten.permute.default(view_256, [1, 0])
        mm_44: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_225, view_121);  permute_225 = view_121 = None
        permute_226: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_44, [1, 0]);  mm_44 = None
        mm_45: "f32[3200, 768]" = torch.ops.aten.mm.default(view_256, permute_227);  view_256 = permute_227 = None
        view_257: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(mm_45, [50, 64, 768]);  mm_45 = None
        permute_228: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_226, [1, 0]);  permute_226 = None
        permute_229: "f32[64, 50, 768]" = torch.ops.aten.permute.default(view_257, [1, 0, 2]);  view_257 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:112, code: x = self.ln_1(input)
        clone_140: "f32[64, 50, 768]" = torch.ops.aten.clone.default(permute_229, memory_format = torch.contiguous_format);  permute_229 = None
        mul_171: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(clone_140, primals_101);  primals_101 = None
        mul_172: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_171, 768)
        sum_50: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_171, [2], True)
        mul_173: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_171, mul_56);  mul_171 = None
        sum_51: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_173, [2], True);  mul_173 = None
        mul_174: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_56, sum_51);  sum_51 = None
        sub_50: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(mul_172, sum_50);  mul_172 = sum_50 = None
        sub_51: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(sub_50, mul_174);  sub_50 = mul_174 = None
        mul_175: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(div_8, sub_51);  div_8 = sub_51 = None
        mul_176: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(clone_140, mul_56);  mul_56 = None
        sum_52: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_176, [0, 1]);  mul_176 = None
        sum_53: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_140, [0, 1]);  clone_140 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:112, code: x = self.ln_1(input)
        add_122: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(add_119, mul_175);  add_119 = mul_175 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        view_258: "f32[3200, 768]" = torch.ops.aten.reshape.default(add_122, [3200, 768])
        mm_46: "f32[3200, 3072]" = torch.ops.aten.mm.default(view_258, permute_230);  permute_230 = None
        permute_231: "f32[768, 3200]" = torch.ops.aten.permute.default(view_258, [1, 0])
        mm_47: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_231, view_119);  permute_231 = view_119 = None
        permute_232: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
        sum_54: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_258, [0], True);  view_258 = None
        view_259: "f32[768]" = torch.ops.aten.reshape.default(sum_54, [768]);  sum_54 = None
        permute_233: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_232, [1, 0]);  permute_232 = None
        view_260: "f32[64, 50, 3072]" = torch.ops.aten.reshape.default(mm_46, [64, 50, 3072]);  mm_46 = None
        mul_178: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(add_63, 0.5);  add_63 = None
        mul_179: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_118, view_118)
        mul_180: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(mul_179, -0.5);  mul_179 = None
        exp_4: "f32[64, 50, 3072]" = torch.ops.aten.exp.default(mul_180);  mul_180 = None
        mul_181: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(exp_4, 0.3989422804014327);  exp_4 = None
        mul_182: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_118, mul_181);  view_118 = mul_181 = None
        add_124: "f32[64, 50, 3072]" = torch.ops.aten.add.Tensor(mul_178, mul_182);  mul_178 = mul_182 = None
        mul_183: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_260, add_124);  view_260 = add_124 = None
        view_261: "f32[3200, 3072]" = torch.ops.aten.reshape.default(mul_183, [3200, 3072]);  mul_183 = None
        mm_48: "f32[3200, 768]" = torch.ops.aten.mm.default(view_261, permute_234);  permute_234 = None
        permute_235: "f32[3072, 3200]" = torch.ops.aten.permute.default(view_261, [1, 0])
        mm_49: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_235, view_117);  permute_235 = view_117 = None
        permute_236: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
        sum_55: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_261, [0], True);  view_261 = None
        view_262: "f32[3072]" = torch.ops.aten.reshape.default(sum_55, [3072]);  sum_55 = None
        permute_237: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_236, [1, 0]);  permute_236 = None
        view_263: "f32[64, 50, 768]" = torch.ops.aten.reshape.default(mm_48, [64, 50, 768]);  mm_48 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:117, code: y = self.ln_2(x)
        mul_185: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(view_263, primals_95);  primals_95 = None
        mul_186: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_185, 768)
        sum_56: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_185, [2], True)
        mul_187: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_185, mul_51);  mul_185 = None
        sum_57: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_187, [2], True);  mul_187 = None
        mul_188: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_51, sum_57);  sum_57 = None
        sub_53: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(mul_186, sum_56);  mul_186 = sum_56 = None
        sub_54: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(sub_53, mul_188);  sub_53 = mul_188 = None
        mul_189: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(div_9, sub_54);  div_9 = sub_54 = None
        mul_190: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(view_263, mul_51);  mul_51 = None
        sum_58: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_190, [0, 1]);  mul_190 = None
        sum_59: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_263, [0, 1]);  view_263 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:117, code: y = self.ln_2(x)
        add_125: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(add_122, mul_189);  add_122 = mul_189 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_238: "f32[50, 64, 768]" = torch.ops.aten.permute.default(add_125, [1, 0, 2])
        clone_143: "f32[50, 64, 768]" = torch.ops.aten.clone.default(permute_238, memory_format = torch.contiguous_format);  permute_238 = None
        view_264: "f32[3200, 768]" = torch.ops.aten.reshape.default(clone_143, [3200, 768]);  clone_143 = None
        mm_50: "f32[3200, 768]" = torch.ops.aten.mm.default(view_264, permute_239);  permute_239 = None
        permute_240: "f32[768, 3200]" = torch.ops.aten.permute.default(view_264, [1, 0])
        mm_51: "f32[768, 768]" = torch.ops.aten.mm.default(permute_240, view_115);  permute_240 = view_115 = None
        permute_241: "f32[768, 768]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
        sum_60: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_264, [0], True);  view_264 = None
        view_265: "f32[768]" = torch.ops.aten.reshape.default(sum_60, [768]);  sum_60 = None
        permute_242: "f32[768, 768]" = torch.ops.aten.permute.default(permute_241, [1, 0]);  permute_241 = None
        view_266: "f32[50, 64, 12, 64]" = torch.ops.aten.reshape.default(mm_50, [50, 64, 12, 64]);  mm_50 = None
        permute_243: "f32[64, 12, 50, 64]" = torch.ops.aten.permute.default(view_266, [1, 2, 0, 3]);  view_266 = None
        _scaled_dot_product_efficient_attention_backward_4 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_243, view_112, view_113, view_114, None, alias_16, getitem_59, getitem_60, getitem_61, 0.0, [True, True, True, False]);  permute_243 = view_112 = view_113 = view_114 = alias_16 = getitem_59 = getitem_60 = getitem_61 = None
        getitem_114: "f32[64, 12, 50, 64]" = _scaled_dot_product_efficient_attention_backward_4[0]
        getitem_115: "f32[64, 12, 50, 64]" = _scaled_dot_product_efficient_attention_backward_4[1]
        getitem_116: "f32[64, 12, 50, 64]" = _scaled_dot_product_efficient_attention_backward_4[2];  _scaled_dot_product_efficient_attention_backward_4 = None
        clone_144: "f32[64, 12, 50, 64]" = torch.ops.aten.clone.default(getitem_116, memory_format = torch.contiguous_format);  getitem_116 = None
        view_267: "f32[768, 50, 64]" = torch.ops.aten.reshape.default(clone_144, [768, 50, 64]);  clone_144 = None
        clone_145: "f32[64, 12, 50, 64]" = torch.ops.aten.clone.default(getitem_115, memory_format = torch.contiguous_format);  getitem_115 = None
        view_268: "f32[768, 50, 64]" = torch.ops.aten.reshape.default(clone_145, [768, 50, 64]);  clone_145 = None
        clone_146: "f32[64, 12, 50, 64]" = torch.ops.aten.clone.default(getitem_114, memory_format = torch.contiguous_format);  getitem_114 = None
        view_269: "f32[768, 50, 64]" = torch.ops.aten.reshape.default(clone_146, [768, 50, 64]);  clone_146 = None
        permute_244: "f32[50, 768, 64]" = torch.ops.aten.permute.default(view_267, [1, 0, 2]);  view_267 = None
        clone_147: "f32[50, 768, 64]" = torch.ops.aten.clone.default(permute_244, memory_format = torch.contiguous_format);  permute_244 = None
        view_270: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(clone_147, [50, 64, 768]);  clone_147 = None
        permute_245: "f32[50, 768, 64]" = torch.ops.aten.permute.default(view_268, [1, 0, 2]);  view_268 = None
        clone_148: "f32[50, 768, 64]" = torch.ops.aten.clone.default(permute_245, memory_format = torch.contiguous_format);  permute_245 = None
        view_271: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(clone_148, [50, 64, 768]);  clone_148 = None
        permute_246: "f32[50, 768, 64]" = torch.ops.aten.permute.default(view_269, [1, 0, 2]);  view_269 = None
        clone_149: "f32[50, 768, 64]" = torch.ops.aten.clone.default(permute_246, memory_format = torch.contiguous_format);  permute_246 = None
        view_272: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(clone_149, [50, 64, 768]);  clone_149 = None
        select_scatter_13: "f32[3, 50, 64, 768]" = torch.ops.aten.select_scatter.default(full_default_2, view_270, 0, 2);  view_270 = None
        select_scatter_14: "f32[3, 50, 64, 768]" = torch.ops.aten.select_scatter.default(full_default_2, view_271, 0, 1);  view_271 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        add_126: "f32[3, 50, 64, 768]" = torch.ops.aten.add.Tensor(select_scatter_13, select_scatter_14);  select_scatter_13 = select_scatter_14 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        select_scatter_15: "f32[3, 50, 64, 768]" = torch.ops.aten.select_scatter.default(full_default_2, view_272, 0, 0);  view_272 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        add_127: "f32[3, 50, 64, 768]" = torch.ops.aten.add.Tensor(add_126, select_scatter_15);  add_126 = select_scatter_15 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        unsqueeze_16: "f32[3, 50, 64, 1, 768]" = torch.ops.aten.unsqueeze.default(add_127, 3);  add_127 = None
        permute_247: "f32[1, 50, 64, 3, 768]" = torch.ops.aten.permute.default(unsqueeze_16, [3, 1, 2, 0, 4]);  unsqueeze_16 = None
        squeeze_16: "f32[50, 64, 3, 768]" = torch.ops.aten.squeeze.dim(permute_247, 0);  permute_247 = None
        clone_150: "f32[50, 64, 3, 768]" = torch.ops.aten.clone.default(squeeze_16, memory_format = torch.contiguous_format);  squeeze_16 = None
        view_273: "f32[50, 64, 2304]" = torch.ops.aten.reshape.default(clone_150, [50, 64, 2304]);  clone_150 = None
        sum_61: "f32[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_273, [0, 1], True)
        view_274: "f32[2304]" = torch.ops.aten.reshape.default(sum_61, [2304]);  sum_61 = None
        view_275: "f32[3200, 2304]" = torch.ops.aten.reshape.default(view_273, [3200, 2304]);  view_273 = None
        permute_248: "f32[2304, 3200]" = torch.ops.aten.permute.default(view_275, [1, 0])
        mm_52: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_248, view_106);  permute_248 = view_106 = None
        permute_249: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_52, [1, 0]);  mm_52 = None
        mm_53: "f32[3200, 768]" = torch.ops.aten.mm.default(view_275, permute_250);  view_275 = permute_250 = None
        view_276: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(mm_53, [50, 64, 768]);  mm_53 = None
        permute_251: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_249, [1, 0]);  permute_249 = None
        permute_252: "f32[64, 50, 768]" = torch.ops.aten.permute.default(view_276, [1, 0, 2]);  view_276 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:112, code: x = self.ln_1(input)
        clone_151: "f32[64, 50, 768]" = torch.ops.aten.clone.default(permute_252, memory_format = torch.contiguous_format);  permute_252 = None
        mul_192: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(clone_151, primals_89);  primals_89 = None
        mul_193: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_192, 768)
        sum_62: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_192, [2], True)
        mul_194: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_192, mul_49);  mul_192 = None
        sum_63: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_194, [2], True);  mul_194 = None
        mul_195: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_49, sum_63);  sum_63 = None
        sub_56: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(mul_193, sum_62);  mul_193 = sum_62 = None
        sub_57: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(sub_56, mul_195);  sub_56 = mul_195 = None
        mul_196: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(div_10, sub_57);  div_10 = sub_57 = None
        mul_197: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(clone_151, mul_49);  mul_49 = None
        sum_64: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_197, [0, 1]);  mul_197 = None
        sum_65: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_151, [0, 1]);  clone_151 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:112, code: x = self.ln_1(input)
        add_128: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(add_125, mul_196);  add_125 = mul_196 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        view_277: "f32[3200, 768]" = torch.ops.aten.reshape.default(add_128, [3200, 768])
        mm_54: "f32[3200, 3072]" = torch.ops.aten.mm.default(view_277, permute_253);  permute_253 = None
        permute_254: "f32[768, 3200]" = torch.ops.aten.permute.default(view_277, [1, 0])
        mm_55: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_254, view_104);  permute_254 = view_104 = None
        permute_255: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
        sum_66: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_277, [0], True);  view_277 = None
        view_278: "f32[768]" = torch.ops.aten.reshape.default(sum_66, [768]);  sum_66 = None
        permute_256: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_255, [1, 0]);  permute_255 = None
        view_279: "f32[64, 50, 3072]" = torch.ops.aten.reshape.default(mm_54, [64, 50, 3072]);  mm_54 = None
        mul_199: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(add_55, 0.5);  add_55 = None
        mul_200: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_103, view_103)
        mul_201: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(mul_200, -0.5);  mul_200 = None
        exp_5: "f32[64, 50, 3072]" = torch.ops.aten.exp.default(mul_201);  mul_201 = None
        mul_202: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(exp_5, 0.3989422804014327);  exp_5 = None
        mul_203: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_103, mul_202);  view_103 = mul_202 = None
        add_130: "f32[64, 50, 3072]" = torch.ops.aten.add.Tensor(mul_199, mul_203);  mul_199 = mul_203 = None
        mul_204: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_279, add_130);  view_279 = add_130 = None
        view_280: "f32[3200, 3072]" = torch.ops.aten.reshape.default(mul_204, [3200, 3072]);  mul_204 = None
        mm_56: "f32[3200, 768]" = torch.ops.aten.mm.default(view_280, permute_257);  permute_257 = None
        permute_258: "f32[3072, 3200]" = torch.ops.aten.permute.default(view_280, [1, 0])
        mm_57: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_258, view_102);  permute_258 = view_102 = None
        permute_259: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
        sum_67: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_280, [0], True);  view_280 = None
        view_281: "f32[3072]" = torch.ops.aten.reshape.default(sum_67, [3072]);  sum_67 = None
        permute_260: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_259, [1, 0]);  permute_259 = None
        view_282: "f32[64, 50, 768]" = torch.ops.aten.reshape.default(mm_56, [64, 50, 768]);  mm_56 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:117, code: y = self.ln_2(x)
        mul_206: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(view_282, primals_83);  primals_83 = None
        mul_207: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_206, 768)
        sum_68: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_206, [2], True)
        mul_208: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_206, mul_44);  mul_206 = None
        sum_69: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_208, [2], True);  mul_208 = None
        mul_209: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_44, sum_69);  sum_69 = None
        sub_59: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(mul_207, sum_68);  mul_207 = sum_68 = None
        sub_60: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(sub_59, mul_209);  sub_59 = mul_209 = None
        mul_210: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(div_11, sub_60);  div_11 = sub_60 = None
        mul_211: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(view_282, mul_44);  mul_44 = None
        sum_70: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_211, [0, 1]);  mul_211 = None
        sum_71: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_282, [0, 1]);  view_282 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:117, code: y = self.ln_2(x)
        add_131: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(add_128, mul_210);  add_128 = mul_210 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_261: "f32[50, 64, 768]" = torch.ops.aten.permute.default(add_131, [1, 0, 2])
        clone_154: "f32[50, 64, 768]" = torch.ops.aten.clone.default(permute_261, memory_format = torch.contiguous_format);  permute_261 = None
        view_283: "f32[3200, 768]" = torch.ops.aten.reshape.default(clone_154, [3200, 768]);  clone_154 = None
        mm_58: "f32[3200, 768]" = torch.ops.aten.mm.default(view_283, permute_262);  permute_262 = None
        permute_263: "f32[768, 3200]" = torch.ops.aten.permute.default(view_283, [1, 0])
        mm_59: "f32[768, 768]" = torch.ops.aten.mm.default(permute_263, view_100);  permute_263 = view_100 = None
        permute_264: "f32[768, 768]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
        sum_72: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_283, [0], True);  view_283 = None
        view_284: "f32[768]" = torch.ops.aten.reshape.default(sum_72, [768]);  sum_72 = None
        permute_265: "f32[768, 768]" = torch.ops.aten.permute.default(permute_264, [1, 0]);  permute_264 = None
        view_285: "f32[50, 64, 12, 64]" = torch.ops.aten.reshape.default(mm_58, [50, 64, 12, 64]);  mm_58 = None
        permute_266: "f32[64, 12, 50, 64]" = torch.ops.aten.permute.default(view_285, [1, 2, 0, 3]);  view_285 = None
        _scaled_dot_product_efficient_attention_backward_5 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_266, view_97, view_98, view_99, None, alias_17, getitem_51, getitem_52, getitem_53, 0.0, [True, True, True, False]);  permute_266 = view_97 = view_98 = view_99 = alias_17 = getitem_51 = getitem_52 = getitem_53 = None
        getitem_118: "f32[64, 12, 50, 64]" = _scaled_dot_product_efficient_attention_backward_5[0]
        getitem_119: "f32[64, 12, 50, 64]" = _scaled_dot_product_efficient_attention_backward_5[1]
        getitem_120: "f32[64, 12, 50, 64]" = _scaled_dot_product_efficient_attention_backward_5[2];  _scaled_dot_product_efficient_attention_backward_5 = None
        clone_155: "f32[64, 12, 50, 64]" = torch.ops.aten.clone.default(getitem_120, memory_format = torch.contiguous_format);  getitem_120 = None
        view_286: "f32[768, 50, 64]" = torch.ops.aten.reshape.default(clone_155, [768, 50, 64]);  clone_155 = None
        clone_156: "f32[64, 12, 50, 64]" = torch.ops.aten.clone.default(getitem_119, memory_format = torch.contiguous_format);  getitem_119 = None
        view_287: "f32[768, 50, 64]" = torch.ops.aten.reshape.default(clone_156, [768, 50, 64]);  clone_156 = None
        clone_157: "f32[64, 12, 50, 64]" = torch.ops.aten.clone.default(getitem_118, memory_format = torch.contiguous_format);  getitem_118 = None
        view_288: "f32[768, 50, 64]" = torch.ops.aten.reshape.default(clone_157, [768, 50, 64]);  clone_157 = None
        permute_267: "f32[50, 768, 64]" = torch.ops.aten.permute.default(view_286, [1, 0, 2]);  view_286 = None
        clone_158: "f32[50, 768, 64]" = torch.ops.aten.clone.default(permute_267, memory_format = torch.contiguous_format);  permute_267 = None
        view_289: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(clone_158, [50, 64, 768]);  clone_158 = None
        permute_268: "f32[50, 768, 64]" = torch.ops.aten.permute.default(view_287, [1, 0, 2]);  view_287 = None
        clone_159: "f32[50, 768, 64]" = torch.ops.aten.clone.default(permute_268, memory_format = torch.contiguous_format);  permute_268 = None
        view_290: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(clone_159, [50, 64, 768]);  clone_159 = None
        permute_269: "f32[50, 768, 64]" = torch.ops.aten.permute.default(view_288, [1, 0, 2]);  view_288 = None
        clone_160: "f32[50, 768, 64]" = torch.ops.aten.clone.default(permute_269, memory_format = torch.contiguous_format);  permute_269 = None
        view_291: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(clone_160, [50, 64, 768]);  clone_160 = None
        select_scatter_16: "f32[3, 50, 64, 768]" = torch.ops.aten.select_scatter.default(full_default_2, view_289, 0, 2);  view_289 = None
        select_scatter_17: "f32[3, 50, 64, 768]" = torch.ops.aten.select_scatter.default(full_default_2, view_290, 0, 1);  view_290 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        add_132: "f32[3, 50, 64, 768]" = torch.ops.aten.add.Tensor(select_scatter_16, select_scatter_17);  select_scatter_16 = select_scatter_17 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        select_scatter_18: "f32[3, 50, 64, 768]" = torch.ops.aten.select_scatter.default(full_default_2, view_291, 0, 0);  view_291 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        add_133: "f32[3, 50, 64, 768]" = torch.ops.aten.add.Tensor(add_132, select_scatter_18);  add_132 = select_scatter_18 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        unsqueeze_17: "f32[3, 50, 64, 1, 768]" = torch.ops.aten.unsqueeze.default(add_133, 3);  add_133 = None
        permute_270: "f32[1, 50, 64, 3, 768]" = torch.ops.aten.permute.default(unsqueeze_17, [3, 1, 2, 0, 4]);  unsqueeze_17 = None
        squeeze_17: "f32[50, 64, 3, 768]" = torch.ops.aten.squeeze.dim(permute_270, 0);  permute_270 = None
        clone_161: "f32[50, 64, 3, 768]" = torch.ops.aten.clone.default(squeeze_17, memory_format = torch.contiguous_format);  squeeze_17 = None
        view_292: "f32[50, 64, 2304]" = torch.ops.aten.reshape.default(clone_161, [50, 64, 2304]);  clone_161 = None
        sum_73: "f32[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_292, [0, 1], True)
        view_293: "f32[2304]" = torch.ops.aten.reshape.default(sum_73, [2304]);  sum_73 = None
        view_294: "f32[3200, 2304]" = torch.ops.aten.reshape.default(view_292, [3200, 2304]);  view_292 = None
        permute_271: "f32[2304, 3200]" = torch.ops.aten.permute.default(view_294, [1, 0])
        mm_60: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_271, view_91);  permute_271 = view_91 = None
        permute_272: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_60, [1, 0]);  mm_60 = None
        mm_61: "f32[3200, 768]" = torch.ops.aten.mm.default(view_294, permute_273);  view_294 = permute_273 = None
        view_295: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(mm_61, [50, 64, 768]);  mm_61 = None
        permute_274: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_272, [1, 0]);  permute_272 = None
        permute_275: "f32[64, 50, 768]" = torch.ops.aten.permute.default(view_295, [1, 0, 2]);  view_295 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:112, code: x = self.ln_1(input)
        clone_162: "f32[64, 50, 768]" = torch.ops.aten.clone.default(permute_275, memory_format = torch.contiguous_format);  permute_275 = None
        mul_213: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(clone_162, primals_77);  primals_77 = None
        mul_214: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_213, 768)
        sum_74: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_213, [2], True)
        mul_215: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_213, mul_42);  mul_213 = None
        sum_75: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_215, [2], True);  mul_215 = None
        mul_216: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_42, sum_75);  sum_75 = None
        sub_62: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(mul_214, sum_74);  mul_214 = sum_74 = None
        sub_63: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(sub_62, mul_216);  sub_62 = mul_216 = None
        mul_217: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(div_12, sub_63);  div_12 = sub_63 = None
        mul_218: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(clone_162, mul_42);  mul_42 = None
        sum_76: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_218, [0, 1]);  mul_218 = None
        sum_77: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_162, [0, 1]);  clone_162 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:112, code: x = self.ln_1(input)
        add_134: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(add_131, mul_217);  add_131 = mul_217 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        view_296: "f32[3200, 768]" = torch.ops.aten.reshape.default(add_134, [3200, 768])
        mm_62: "f32[3200, 3072]" = torch.ops.aten.mm.default(view_296, permute_276);  permute_276 = None
        permute_277: "f32[768, 3200]" = torch.ops.aten.permute.default(view_296, [1, 0])
        mm_63: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_277, view_89);  permute_277 = view_89 = None
        permute_278: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
        sum_78: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_296, [0], True);  view_296 = None
        view_297: "f32[768]" = torch.ops.aten.reshape.default(sum_78, [768]);  sum_78 = None
        permute_279: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_278, [1, 0]);  permute_278 = None
        view_298: "f32[64, 50, 3072]" = torch.ops.aten.reshape.default(mm_62, [64, 50, 3072]);  mm_62 = None
        mul_220: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(add_47, 0.5);  add_47 = None
        mul_221: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_88, view_88)
        mul_222: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(mul_221, -0.5);  mul_221 = None
        exp_6: "f32[64, 50, 3072]" = torch.ops.aten.exp.default(mul_222);  mul_222 = None
        mul_223: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(exp_6, 0.3989422804014327);  exp_6 = None
        mul_224: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_88, mul_223);  view_88 = mul_223 = None
        add_136: "f32[64, 50, 3072]" = torch.ops.aten.add.Tensor(mul_220, mul_224);  mul_220 = mul_224 = None
        mul_225: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_298, add_136);  view_298 = add_136 = None
        view_299: "f32[3200, 3072]" = torch.ops.aten.reshape.default(mul_225, [3200, 3072]);  mul_225 = None
        mm_64: "f32[3200, 768]" = torch.ops.aten.mm.default(view_299, permute_280);  permute_280 = None
        permute_281: "f32[3072, 3200]" = torch.ops.aten.permute.default(view_299, [1, 0])
        mm_65: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_281, view_87);  permute_281 = view_87 = None
        permute_282: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
        sum_79: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_299, [0], True);  view_299 = None
        view_300: "f32[3072]" = torch.ops.aten.reshape.default(sum_79, [3072]);  sum_79 = None
        permute_283: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_282, [1, 0]);  permute_282 = None
        view_301: "f32[64, 50, 768]" = torch.ops.aten.reshape.default(mm_64, [64, 50, 768]);  mm_64 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:117, code: y = self.ln_2(x)
        mul_227: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(view_301, primals_71);  primals_71 = None
        mul_228: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_227, 768)
        sum_80: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_227, [2], True)
        mul_229: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_227, mul_37);  mul_227 = None
        sum_81: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_229, [2], True);  mul_229 = None
        mul_230: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_37, sum_81);  sum_81 = None
        sub_65: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(mul_228, sum_80);  mul_228 = sum_80 = None
        sub_66: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(sub_65, mul_230);  sub_65 = mul_230 = None
        mul_231: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(div_13, sub_66);  div_13 = sub_66 = None
        mul_232: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(view_301, mul_37);  mul_37 = None
        sum_82: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_232, [0, 1]);  mul_232 = None
        sum_83: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_301, [0, 1]);  view_301 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:117, code: y = self.ln_2(x)
        add_137: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(add_134, mul_231);  add_134 = mul_231 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_284: "f32[50, 64, 768]" = torch.ops.aten.permute.default(add_137, [1, 0, 2])
        clone_165: "f32[50, 64, 768]" = torch.ops.aten.clone.default(permute_284, memory_format = torch.contiguous_format);  permute_284 = None
        view_302: "f32[3200, 768]" = torch.ops.aten.reshape.default(clone_165, [3200, 768]);  clone_165 = None
        mm_66: "f32[3200, 768]" = torch.ops.aten.mm.default(view_302, permute_285);  permute_285 = None
        permute_286: "f32[768, 3200]" = torch.ops.aten.permute.default(view_302, [1, 0])
        mm_67: "f32[768, 768]" = torch.ops.aten.mm.default(permute_286, view_85);  permute_286 = view_85 = None
        permute_287: "f32[768, 768]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
        sum_84: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_302, [0], True);  view_302 = None
        view_303: "f32[768]" = torch.ops.aten.reshape.default(sum_84, [768]);  sum_84 = None
        permute_288: "f32[768, 768]" = torch.ops.aten.permute.default(permute_287, [1, 0]);  permute_287 = None
        view_304: "f32[50, 64, 12, 64]" = torch.ops.aten.reshape.default(mm_66, [50, 64, 12, 64]);  mm_66 = None
        permute_289: "f32[64, 12, 50, 64]" = torch.ops.aten.permute.default(view_304, [1, 2, 0, 3]);  view_304 = None
        _scaled_dot_product_efficient_attention_backward_6 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_289, view_82, view_83, view_84, None, alias_18, getitem_43, getitem_44, getitem_45, 0.0, [True, True, True, False]);  permute_289 = view_82 = view_83 = view_84 = alias_18 = getitem_43 = getitem_44 = getitem_45 = None
        getitem_122: "f32[64, 12, 50, 64]" = _scaled_dot_product_efficient_attention_backward_6[0]
        getitem_123: "f32[64, 12, 50, 64]" = _scaled_dot_product_efficient_attention_backward_6[1]
        getitem_124: "f32[64, 12, 50, 64]" = _scaled_dot_product_efficient_attention_backward_6[2];  _scaled_dot_product_efficient_attention_backward_6 = None
        clone_166: "f32[64, 12, 50, 64]" = torch.ops.aten.clone.default(getitem_124, memory_format = torch.contiguous_format);  getitem_124 = None
        view_305: "f32[768, 50, 64]" = torch.ops.aten.reshape.default(clone_166, [768, 50, 64]);  clone_166 = None
        clone_167: "f32[64, 12, 50, 64]" = torch.ops.aten.clone.default(getitem_123, memory_format = torch.contiguous_format);  getitem_123 = None
        view_306: "f32[768, 50, 64]" = torch.ops.aten.reshape.default(clone_167, [768, 50, 64]);  clone_167 = None
        clone_168: "f32[64, 12, 50, 64]" = torch.ops.aten.clone.default(getitem_122, memory_format = torch.contiguous_format);  getitem_122 = None
        view_307: "f32[768, 50, 64]" = torch.ops.aten.reshape.default(clone_168, [768, 50, 64]);  clone_168 = None
        permute_290: "f32[50, 768, 64]" = torch.ops.aten.permute.default(view_305, [1, 0, 2]);  view_305 = None
        clone_169: "f32[50, 768, 64]" = torch.ops.aten.clone.default(permute_290, memory_format = torch.contiguous_format);  permute_290 = None
        view_308: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(clone_169, [50, 64, 768]);  clone_169 = None
        permute_291: "f32[50, 768, 64]" = torch.ops.aten.permute.default(view_306, [1, 0, 2]);  view_306 = None
        clone_170: "f32[50, 768, 64]" = torch.ops.aten.clone.default(permute_291, memory_format = torch.contiguous_format);  permute_291 = None
        view_309: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(clone_170, [50, 64, 768]);  clone_170 = None
        permute_292: "f32[50, 768, 64]" = torch.ops.aten.permute.default(view_307, [1, 0, 2]);  view_307 = None
        clone_171: "f32[50, 768, 64]" = torch.ops.aten.clone.default(permute_292, memory_format = torch.contiguous_format);  permute_292 = None
        view_310: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(clone_171, [50, 64, 768]);  clone_171 = None
        select_scatter_19: "f32[3, 50, 64, 768]" = torch.ops.aten.select_scatter.default(full_default_2, view_308, 0, 2);  view_308 = None
        select_scatter_20: "f32[3, 50, 64, 768]" = torch.ops.aten.select_scatter.default(full_default_2, view_309, 0, 1);  view_309 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        add_138: "f32[3, 50, 64, 768]" = torch.ops.aten.add.Tensor(select_scatter_19, select_scatter_20);  select_scatter_19 = select_scatter_20 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        select_scatter_21: "f32[3, 50, 64, 768]" = torch.ops.aten.select_scatter.default(full_default_2, view_310, 0, 0);  view_310 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        add_139: "f32[3, 50, 64, 768]" = torch.ops.aten.add.Tensor(add_138, select_scatter_21);  add_138 = select_scatter_21 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        unsqueeze_18: "f32[3, 50, 64, 1, 768]" = torch.ops.aten.unsqueeze.default(add_139, 3);  add_139 = None
        permute_293: "f32[1, 50, 64, 3, 768]" = torch.ops.aten.permute.default(unsqueeze_18, [3, 1, 2, 0, 4]);  unsqueeze_18 = None
        squeeze_18: "f32[50, 64, 3, 768]" = torch.ops.aten.squeeze.dim(permute_293, 0);  permute_293 = None
        clone_172: "f32[50, 64, 3, 768]" = torch.ops.aten.clone.default(squeeze_18, memory_format = torch.contiguous_format);  squeeze_18 = None
        view_311: "f32[50, 64, 2304]" = torch.ops.aten.reshape.default(clone_172, [50, 64, 2304]);  clone_172 = None
        sum_85: "f32[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_311, [0, 1], True)
        view_312: "f32[2304]" = torch.ops.aten.reshape.default(sum_85, [2304]);  sum_85 = None
        view_313: "f32[3200, 2304]" = torch.ops.aten.reshape.default(view_311, [3200, 2304]);  view_311 = None
        permute_294: "f32[2304, 3200]" = torch.ops.aten.permute.default(view_313, [1, 0])
        mm_68: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_294, view_76);  permute_294 = view_76 = None
        permute_295: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_68, [1, 0]);  mm_68 = None
        mm_69: "f32[3200, 768]" = torch.ops.aten.mm.default(view_313, permute_296);  view_313 = permute_296 = None
        view_314: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(mm_69, [50, 64, 768]);  mm_69 = None
        permute_297: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_295, [1, 0]);  permute_295 = None
        permute_298: "f32[64, 50, 768]" = torch.ops.aten.permute.default(view_314, [1, 0, 2]);  view_314 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:112, code: x = self.ln_1(input)
        clone_173: "f32[64, 50, 768]" = torch.ops.aten.clone.default(permute_298, memory_format = torch.contiguous_format);  permute_298 = None
        mul_234: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(clone_173, primals_65);  primals_65 = None
        mul_235: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_234, 768)
        sum_86: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_234, [2], True)
        mul_236: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_234, mul_35);  mul_234 = None
        sum_87: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_236, [2], True);  mul_236 = None
        mul_237: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_35, sum_87);  sum_87 = None
        sub_68: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(mul_235, sum_86);  mul_235 = sum_86 = None
        sub_69: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(sub_68, mul_237);  sub_68 = mul_237 = None
        mul_238: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(div_14, sub_69);  div_14 = sub_69 = None
        mul_239: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(clone_173, mul_35);  mul_35 = None
        sum_88: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_239, [0, 1]);  mul_239 = None
        sum_89: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_173, [0, 1]);  clone_173 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:112, code: x = self.ln_1(input)
        add_140: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(add_137, mul_238);  add_137 = mul_238 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        view_315: "f32[3200, 768]" = torch.ops.aten.reshape.default(add_140, [3200, 768])
        mm_70: "f32[3200, 3072]" = torch.ops.aten.mm.default(view_315, permute_299);  permute_299 = None
        permute_300: "f32[768, 3200]" = torch.ops.aten.permute.default(view_315, [1, 0])
        mm_71: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_300, view_74);  permute_300 = view_74 = None
        permute_301: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
        sum_90: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_315, [0], True);  view_315 = None
        view_316: "f32[768]" = torch.ops.aten.reshape.default(sum_90, [768]);  sum_90 = None
        permute_302: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_301, [1, 0]);  permute_301 = None
        view_317: "f32[64, 50, 3072]" = torch.ops.aten.reshape.default(mm_70, [64, 50, 3072]);  mm_70 = None
        mul_241: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(add_39, 0.5);  add_39 = None
        mul_242: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_73, view_73)
        mul_243: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(mul_242, -0.5);  mul_242 = None
        exp_7: "f32[64, 50, 3072]" = torch.ops.aten.exp.default(mul_243);  mul_243 = None
        mul_244: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(exp_7, 0.3989422804014327);  exp_7 = None
        mul_245: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_73, mul_244);  view_73 = mul_244 = None
        add_142: "f32[64, 50, 3072]" = torch.ops.aten.add.Tensor(mul_241, mul_245);  mul_241 = mul_245 = None
        mul_246: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_317, add_142);  view_317 = add_142 = None
        view_318: "f32[3200, 3072]" = torch.ops.aten.reshape.default(mul_246, [3200, 3072]);  mul_246 = None
        mm_72: "f32[3200, 768]" = torch.ops.aten.mm.default(view_318, permute_303);  permute_303 = None
        permute_304: "f32[3072, 3200]" = torch.ops.aten.permute.default(view_318, [1, 0])
        mm_73: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_304, view_72);  permute_304 = view_72 = None
        permute_305: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
        sum_91: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_318, [0], True);  view_318 = None
        view_319: "f32[3072]" = torch.ops.aten.reshape.default(sum_91, [3072]);  sum_91 = None
        permute_306: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_305, [1, 0]);  permute_305 = None
        view_320: "f32[64, 50, 768]" = torch.ops.aten.reshape.default(mm_72, [64, 50, 768]);  mm_72 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:117, code: y = self.ln_2(x)
        mul_248: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(view_320, primals_59);  primals_59 = None
        mul_249: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_248, 768)
        sum_92: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_248, [2], True)
        mul_250: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_248, mul_30);  mul_248 = None
        sum_93: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_250, [2], True);  mul_250 = None
        mul_251: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_30, sum_93);  sum_93 = None
        sub_71: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(mul_249, sum_92);  mul_249 = sum_92 = None
        sub_72: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(sub_71, mul_251);  sub_71 = mul_251 = None
        mul_252: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(div_15, sub_72);  div_15 = sub_72 = None
        mul_253: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(view_320, mul_30);  mul_30 = None
        sum_94: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_253, [0, 1]);  mul_253 = None
        sum_95: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_320, [0, 1]);  view_320 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:117, code: y = self.ln_2(x)
        add_143: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(add_140, mul_252);  add_140 = mul_252 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_307: "f32[50, 64, 768]" = torch.ops.aten.permute.default(add_143, [1, 0, 2])
        clone_176: "f32[50, 64, 768]" = torch.ops.aten.clone.default(permute_307, memory_format = torch.contiguous_format);  permute_307 = None
        view_321: "f32[3200, 768]" = torch.ops.aten.reshape.default(clone_176, [3200, 768]);  clone_176 = None
        mm_74: "f32[3200, 768]" = torch.ops.aten.mm.default(view_321, permute_308);  permute_308 = None
        permute_309: "f32[768, 3200]" = torch.ops.aten.permute.default(view_321, [1, 0])
        mm_75: "f32[768, 768]" = torch.ops.aten.mm.default(permute_309, view_70);  permute_309 = view_70 = None
        permute_310: "f32[768, 768]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
        sum_96: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_321, [0], True);  view_321 = None
        view_322: "f32[768]" = torch.ops.aten.reshape.default(sum_96, [768]);  sum_96 = None
        permute_311: "f32[768, 768]" = torch.ops.aten.permute.default(permute_310, [1, 0]);  permute_310 = None
        view_323: "f32[50, 64, 12, 64]" = torch.ops.aten.reshape.default(mm_74, [50, 64, 12, 64]);  mm_74 = None
        permute_312: "f32[64, 12, 50, 64]" = torch.ops.aten.permute.default(view_323, [1, 2, 0, 3]);  view_323 = None
        _scaled_dot_product_efficient_attention_backward_7 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_312, view_67, view_68, view_69, None, alias_19, getitem_35, getitem_36, getitem_37, 0.0, [True, True, True, False]);  permute_312 = view_67 = view_68 = view_69 = alias_19 = getitem_35 = getitem_36 = getitem_37 = None
        getitem_126: "f32[64, 12, 50, 64]" = _scaled_dot_product_efficient_attention_backward_7[0]
        getitem_127: "f32[64, 12, 50, 64]" = _scaled_dot_product_efficient_attention_backward_7[1]
        getitem_128: "f32[64, 12, 50, 64]" = _scaled_dot_product_efficient_attention_backward_7[2];  _scaled_dot_product_efficient_attention_backward_7 = None
        clone_177: "f32[64, 12, 50, 64]" = torch.ops.aten.clone.default(getitem_128, memory_format = torch.contiguous_format);  getitem_128 = None
        view_324: "f32[768, 50, 64]" = torch.ops.aten.reshape.default(clone_177, [768, 50, 64]);  clone_177 = None
        clone_178: "f32[64, 12, 50, 64]" = torch.ops.aten.clone.default(getitem_127, memory_format = torch.contiguous_format);  getitem_127 = None
        view_325: "f32[768, 50, 64]" = torch.ops.aten.reshape.default(clone_178, [768, 50, 64]);  clone_178 = None
        clone_179: "f32[64, 12, 50, 64]" = torch.ops.aten.clone.default(getitem_126, memory_format = torch.contiguous_format);  getitem_126 = None
        view_326: "f32[768, 50, 64]" = torch.ops.aten.reshape.default(clone_179, [768, 50, 64]);  clone_179 = None
        permute_313: "f32[50, 768, 64]" = torch.ops.aten.permute.default(view_324, [1, 0, 2]);  view_324 = None
        clone_180: "f32[50, 768, 64]" = torch.ops.aten.clone.default(permute_313, memory_format = torch.contiguous_format);  permute_313 = None
        view_327: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(clone_180, [50, 64, 768]);  clone_180 = None
        permute_314: "f32[50, 768, 64]" = torch.ops.aten.permute.default(view_325, [1, 0, 2]);  view_325 = None
        clone_181: "f32[50, 768, 64]" = torch.ops.aten.clone.default(permute_314, memory_format = torch.contiguous_format);  permute_314 = None
        view_328: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(clone_181, [50, 64, 768]);  clone_181 = None
        permute_315: "f32[50, 768, 64]" = torch.ops.aten.permute.default(view_326, [1, 0, 2]);  view_326 = None
        clone_182: "f32[50, 768, 64]" = torch.ops.aten.clone.default(permute_315, memory_format = torch.contiguous_format);  permute_315 = None
        view_329: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(clone_182, [50, 64, 768]);  clone_182 = None
        select_scatter_22: "f32[3, 50, 64, 768]" = torch.ops.aten.select_scatter.default(full_default_2, view_327, 0, 2);  view_327 = None
        select_scatter_23: "f32[3, 50, 64, 768]" = torch.ops.aten.select_scatter.default(full_default_2, view_328, 0, 1);  view_328 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        add_144: "f32[3, 50, 64, 768]" = torch.ops.aten.add.Tensor(select_scatter_22, select_scatter_23);  select_scatter_22 = select_scatter_23 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        select_scatter_24: "f32[3, 50, 64, 768]" = torch.ops.aten.select_scatter.default(full_default_2, view_329, 0, 0);  view_329 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        add_145: "f32[3, 50, 64, 768]" = torch.ops.aten.add.Tensor(add_144, select_scatter_24);  add_144 = select_scatter_24 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        unsqueeze_19: "f32[3, 50, 64, 1, 768]" = torch.ops.aten.unsqueeze.default(add_145, 3);  add_145 = None
        permute_316: "f32[1, 50, 64, 3, 768]" = torch.ops.aten.permute.default(unsqueeze_19, [3, 1, 2, 0, 4]);  unsqueeze_19 = None
        squeeze_19: "f32[50, 64, 3, 768]" = torch.ops.aten.squeeze.dim(permute_316, 0);  permute_316 = None
        clone_183: "f32[50, 64, 3, 768]" = torch.ops.aten.clone.default(squeeze_19, memory_format = torch.contiguous_format);  squeeze_19 = None
        view_330: "f32[50, 64, 2304]" = torch.ops.aten.reshape.default(clone_183, [50, 64, 2304]);  clone_183 = None
        sum_97: "f32[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_330, [0, 1], True)
        view_331: "f32[2304]" = torch.ops.aten.reshape.default(sum_97, [2304]);  sum_97 = None
        view_332: "f32[3200, 2304]" = torch.ops.aten.reshape.default(view_330, [3200, 2304]);  view_330 = None
        permute_317: "f32[2304, 3200]" = torch.ops.aten.permute.default(view_332, [1, 0])
        mm_76: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_317, view_61);  permute_317 = view_61 = None
        permute_318: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_76, [1, 0]);  mm_76 = None
        mm_77: "f32[3200, 768]" = torch.ops.aten.mm.default(view_332, permute_319);  view_332 = permute_319 = None
        view_333: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(mm_77, [50, 64, 768]);  mm_77 = None
        permute_320: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_318, [1, 0]);  permute_318 = None
        permute_321: "f32[64, 50, 768]" = torch.ops.aten.permute.default(view_333, [1, 0, 2]);  view_333 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:112, code: x = self.ln_1(input)
        clone_184: "f32[64, 50, 768]" = torch.ops.aten.clone.default(permute_321, memory_format = torch.contiguous_format);  permute_321 = None
        mul_255: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(clone_184, primals_53);  primals_53 = None
        mul_256: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_255, 768)
        sum_98: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_255, [2], True)
        mul_257: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_255, mul_28);  mul_255 = None
        sum_99: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_257, [2], True);  mul_257 = None
        mul_258: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_28, sum_99);  sum_99 = None
        sub_74: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(mul_256, sum_98);  mul_256 = sum_98 = None
        sub_75: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(sub_74, mul_258);  sub_74 = mul_258 = None
        mul_259: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(div_16, sub_75);  div_16 = sub_75 = None
        mul_260: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(clone_184, mul_28);  mul_28 = None
        sum_100: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_260, [0, 1]);  mul_260 = None
        sum_101: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_184, [0, 1]);  clone_184 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:112, code: x = self.ln_1(input)
        add_146: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(add_143, mul_259);  add_143 = mul_259 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        view_334: "f32[3200, 768]" = torch.ops.aten.reshape.default(add_146, [3200, 768])
        mm_78: "f32[3200, 3072]" = torch.ops.aten.mm.default(view_334, permute_322);  permute_322 = None
        permute_323: "f32[768, 3200]" = torch.ops.aten.permute.default(view_334, [1, 0])
        mm_79: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_323, view_59);  permute_323 = view_59 = None
        permute_324: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
        sum_102: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_334, [0], True);  view_334 = None
        view_335: "f32[768]" = torch.ops.aten.reshape.default(sum_102, [768]);  sum_102 = None
        permute_325: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_324, [1, 0]);  permute_324 = None
        view_336: "f32[64, 50, 3072]" = torch.ops.aten.reshape.default(mm_78, [64, 50, 3072]);  mm_78 = None
        mul_262: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(add_31, 0.5);  add_31 = None
        mul_263: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_58, view_58)
        mul_264: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(mul_263, -0.5);  mul_263 = None
        exp_8: "f32[64, 50, 3072]" = torch.ops.aten.exp.default(mul_264);  mul_264 = None
        mul_265: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(exp_8, 0.3989422804014327);  exp_8 = None
        mul_266: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_58, mul_265);  view_58 = mul_265 = None
        add_148: "f32[64, 50, 3072]" = torch.ops.aten.add.Tensor(mul_262, mul_266);  mul_262 = mul_266 = None
        mul_267: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_336, add_148);  view_336 = add_148 = None
        view_337: "f32[3200, 3072]" = torch.ops.aten.reshape.default(mul_267, [3200, 3072]);  mul_267 = None
        mm_80: "f32[3200, 768]" = torch.ops.aten.mm.default(view_337, permute_326);  permute_326 = None
        permute_327: "f32[3072, 3200]" = torch.ops.aten.permute.default(view_337, [1, 0])
        mm_81: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_327, view_57);  permute_327 = view_57 = None
        permute_328: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
        sum_103: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_337, [0], True);  view_337 = None
        view_338: "f32[3072]" = torch.ops.aten.reshape.default(sum_103, [3072]);  sum_103 = None
        permute_329: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_328, [1, 0]);  permute_328 = None
        view_339: "f32[64, 50, 768]" = torch.ops.aten.reshape.default(mm_80, [64, 50, 768]);  mm_80 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:117, code: y = self.ln_2(x)
        mul_269: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(view_339, primals_47);  primals_47 = None
        mul_270: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_269, 768)
        sum_104: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_269, [2], True)
        mul_271: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_269, mul_23);  mul_269 = None
        sum_105: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_271, [2], True);  mul_271 = None
        mul_272: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_23, sum_105);  sum_105 = None
        sub_77: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(mul_270, sum_104);  mul_270 = sum_104 = None
        sub_78: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(sub_77, mul_272);  sub_77 = mul_272 = None
        mul_273: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(div_17, sub_78);  div_17 = sub_78 = None
        mul_274: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(view_339, mul_23);  mul_23 = None
        sum_106: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_274, [0, 1]);  mul_274 = None
        sum_107: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_339, [0, 1]);  view_339 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:117, code: y = self.ln_2(x)
        add_149: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(add_146, mul_273);  add_146 = mul_273 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_330: "f32[50, 64, 768]" = torch.ops.aten.permute.default(add_149, [1, 0, 2])
        clone_187: "f32[50, 64, 768]" = torch.ops.aten.clone.default(permute_330, memory_format = torch.contiguous_format);  permute_330 = None
        view_340: "f32[3200, 768]" = torch.ops.aten.reshape.default(clone_187, [3200, 768]);  clone_187 = None
        mm_82: "f32[3200, 768]" = torch.ops.aten.mm.default(view_340, permute_331);  permute_331 = None
        permute_332: "f32[768, 3200]" = torch.ops.aten.permute.default(view_340, [1, 0])
        mm_83: "f32[768, 768]" = torch.ops.aten.mm.default(permute_332, view_55);  permute_332 = view_55 = None
        permute_333: "f32[768, 768]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
        sum_108: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_340, [0], True);  view_340 = None
        view_341: "f32[768]" = torch.ops.aten.reshape.default(sum_108, [768]);  sum_108 = None
        permute_334: "f32[768, 768]" = torch.ops.aten.permute.default(permute_333, [1, 0]);  permute_333 = None
        view_342: "f32[50, 64, 12, 64]" = torch.ops.aten.reshape.default(mm_82, [50, 64, 12, 64]);  mm_82 = None
        permute_335: "f32[64, 12, 50, 64]" = torch.ops.aten.permute.default(view_342, [1, 2, 0, 3]);  view_342 = None
        _scaled_dot_product_efficient_attention_backward_8 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_335, view_52, view_53, view_54, None, alias_20, getitem_27, getitem_28, getitem_29, 0.0, [True, True, True, False]);  permute_335 = view_52 = view_53 = view_54 = alias_20 = getitem_27 = getitem_28 = getitem_29 = None
        getitem_130: "f32[64, 12, 50, 64]" = _scaled_dot_product_efficient_attention_backward_8[0]
        getitem_131: "f32[64, 12, 50, 64]" = _scaled_dot_product_efficient_attention_backward_8[1]
        getitem_132: "f32[64, 12, 50, 64]" = _scaled_dot_product_efficient_attention_backward_8[2];  _scaled_dot_product_efficient_attention_backward_8 = None
        clone_188: "f32[64, 12, 50, 64]" = torch.ops.aten.clone.default(getitem_132, memory_format = torch.contiguous_format);  getitem_132 = None
        view_343: "f32[768, 50, 64]" = torch.ops.aten.reshape.default(clone_188, [768, 50, 64]);  clone_188 = None
        clone_189: "f32[64, 12, 50, 64]" = torch.ops.aten.clone.default(getitem_131, memory_format = torch.contiguous_format);  getitem_131 = None
        view_344: "f32[768, 50, 64]" = torch.ops.aten.reshape.default(clone_189, [768, 50, 64]);  clone_189 = None
        clone_190: "f32[64, 12, 50, 64]" = torch.ops.aten.clone.default(getitem_130, memory_format = torch.contiguous_format);  getitem_130 = None
        view_345: "f32[768, 50, 64]" = torch.ops.aten.reshape.default(clone_190, [768, 50, 64]);  clone_190 = None
        permute_336: "f32[50, 768, 64]" = torch.ops.aten.permute.default(view_343, [1, 0, 2]);  view_343 = None
        clone_191: "f32[50, 768, 64]" = torch.ops.aten.clone.default(permute_336, memory_format = torch.contiguous_format);  permute_336 = None
        view_346: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(clone_191, [50, 64, 768]);  clone_191 = None
        permute_337: "f32[50, 768, 64]" = torch.ops.aten.permute.default(view_344, [1, 0, 2]);  view_344 = None
        clone_192: "f32[50, 768, 64]" = torch.ops.aten.clone.default(permute_337, memory_format = torch.contiguous_format);  permute_337 = None
        view_347: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(clone_192, [50, 64, 768]);  clone_192 = None
        permute_338: "f32[50, 768, 64]" = torch.ops.aten.permute.default(view_345, [1, 0, 2]);  view_345 = None
        clone_193: "f32[50, 768, 64]" = torch.ops.aten.clone.default(permute_338, memory_format = torch.contiguous_format);  permute_338 = None
        view_348: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(clone_193, [50, 64, 768]);  clone_193 = None
        select_scatter_25: "f32[3, 50, 64, 768]" = torch.ops.aten.select_scatter.default(full_default_2, view_346, 0, 2);  view_346 = None
        select_scatter_26: "f32[3, 50, 64, 768]" = torch.ops.aten.select_scatter.default(full_default_2, view_347, 0, 1);  view_347 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        add_150: "f32[3, 50, 64, 768]" = torch.ops.aten.add.Tensor(select_scatter_25, select_scatter_26);  select_scatter_25 = select_scatter_26 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        select_scatter_27: "f32[3, 50, 64, 768]" = torch.ops.aten.select_scatter.default(full_default_2, view_348, 0, 0);  view_348 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        add_151: "f32[3, 50, 64, 768]" = torch.ops.aten.add.Tensor(add_150, select_scatter_27);  add_150 = select_scatter_27 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        unsqueeze_20: "f32[3, 50, 64, 1, 768]" = torch.ops.aten.unsqueeze.default(add_151, 3);  add_151 = None
        permute_339: "f32[1, 50, 64, 3, 768]" = torch.ops.aten.permute.default(unsqueeze_20, [3, 1, 2, 0, 4]);  unsqueeze_20 = None
        squeeze_20: "f32[50, 64, 3, 768]" = torch.ops.aten.squeeze.dim(permute_339, 0);  permute_339 = None
        clone_194: "f32[50, 64, 3, 768]" = torch.ops.aten.clone.default(squeeze_20, memory_format = torch.contiguous_format);  squeeze_20 = None
        view_349: "f32[50, 64, 2304]" = torch.ops.aten.reshape.default(clone_194, [50, 64, 2304]);  clone_194 = None
        sum_109: "f32[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_349, [0, 1], True)
        view_350: "f32[2304]" = torch.ops.aten.reshape.default(sum_109, [2304]);  sum_109 = None
        view_351: "f32[3200, 2304]" = torch.ops.aten.reshape.default(view_349, [3200, 2304]);  view_349 = None
        permute_340: "f32[2304, 3200]" = torch.ops.aten.permute.default(view_351, [1, 0])
        mm_84: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_340, view_46);  permute_340 = view_46 = None
        permute_341: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_84, [1, 0]);  mm_84 = None
        mm_85: "f32[3200, 768]" = torch.ops.aten.mm.default(view_351, permute_342);  view_351 = permute_342 = None
        view_352: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(mm_85, [50, 64, 768]);  mm_85 = None
        permute_343: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_341, [1, 0]);  permute_341 = None
        permute_344: "f32[64, 50, 768]" = torch.ops.aten.permute.default(view_352, [1, 0, 2]);  view_352 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:112, code: x = self.ln_1(input)
        clone_195: "f32[64, 50, 768]" = torch.ops.aten.clone.default(permute_344, memory_format = torch.contiguous_format);  permute_344 = None
        mul_276: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(clone_195, primals_41);  primals_41 = None
        mul_277: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_276, 768)
        sum_110: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_276, [2], True)
        mul_278: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_276, mul_21);  mul_276 = None
        sum_111: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_278, [2], True);  mul_278 = None
        mul_279: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_21, sum_111);  sum_111 = None
        sub_80: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(mul_277, sum_110);  mul_277 = sum_110 = None
        sub_81: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(sub_80, mul_279);  sub_80 = mul_279 = None
        mul_280: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(div_18, sub_81);  div_18 = sub_81 = None
        mul_281: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(clone_195, mul_21);  mul_21 = None
        sum_112: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_281, [0, 1]);  mul_281 = None
        sum_113: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_195, [0, 1]);  clone_195 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:112, code: x = self.ln_1(input)
        add_152: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(add_149, mul_280);  add_149 = mul_280 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        view_353: "f32[3200, 768]" = torch.ops.aten.reshape.default(add_152, [3200, 768])
        mm_86: "f32[3200, 3072]" = torch.ops.aten.mm.default(view_353, permute_345);  permute_345 = None
        permute_346: "f32[768, 3200]" = torch.ops.aten.permute.default(view_353, [1, 0])
        mm_87: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_346, view_44);  permute_346 = view_44 = None
        permute_347: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
        sum_114: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_353, [0], True);  view_353 = None
        view_354: "f32[768]" = torch.ops.aten.reshape.default(sum_114, [768]);  sum_114 = None
        permute_348: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_347, [1, 0]);  permute_347 = None
        view_355: "f32[64, 50, 3072]" = torch.ops.aten.reshape.default(mm_86, [64, 50, 3072]);  mm_86 = None
        mul_283: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(add_23, 0.5);  add_23 = None
        mul_284: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_43, view_43)
        mul_285: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(mul_284, -0.5);  mul_284 = None
        exp_9: "f32[64, 50, 3072]" = torch.ops.aten.exp.default(mul_285);  mul_285 = None
        mul_286: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(exp_9, 0.3989422804014327);  exp_9 = None
        mul_287: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_43, mul_286);  view_43 = mul_286 = None
        add_154: "f32[64, 50, 3072]" = torch.ops.aten.add.Tensor(mul_283, mul_287);  mul_283 = mul_287 = None
        mul_288: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_355, add_154);  view_355 = add_154 = None
        view_356: "f32[3200, 3072]" = torch.ops.aten.reshape.default(mul_288, [3200, 3072]);  mul_288 = None
        mm_88: "f32[3200, 768]" = torch.ops.aten.mm.default(view_356, permute_349);  permute_349 = None
        permute_350: "f32[3072, 3200]" = torch.ops.aten.permute.default(view_356, [1, 0])
        mm_89: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_350, view_42);  permute_350 = view_42 = None
        permute_351: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
        sum_115: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_356, [0], True);  view_356 = None
        view_357: "f32[3072]" = torch.ops.aten.reshape.default(sum_115, [3072]);  sum_115 = None
        permute_352: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_351, [1, 0]);  permute_351 = None
        view_358: "f32[64, 50, 768]" = torch.ops.aten.reshape.default(mm_88, [64, 50, 768]);  mm_88 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:117, code: y = self.ln_2(x)
        mul_290: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(view_358, primals_35);  primals_35 = None
        mul_291: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_290, 768)
        sum_116: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_290, [2], True)
        mul_292: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_290, mul_16);  mul_290 = None
        sum_117: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_292, [2], True);  mul_292 = None
        mul_293: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_16, sum_117);  sum_117 = None
        sub_83: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(mul_291, sum_116);  mul_291 = sum_116 = None
        sub_84: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(sub_83, mul_293);  sub_83 = mul_293 = None
        mul_294: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(div_19, sub_84);  div_19 = sub_84 = None
        mul_295: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(view_358, mul_16);  mul_16 = None
        sum_118: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_295, [0, 1]);  mul_295 = None
        sum_119: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_358, [0, 1]);  view_358 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:117, code: y = self.ln_2(x)
        add_155: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(add_152, mul_294);  add_152 = mul_294 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_353: "f32[50, 64, 768]" = torch.ops.aten.permute.default(add_155, [1, 0, 2])
        clone_198: "f32[50, 64, 768]" = torch.ops.aten.clone.default(permute_353, memory_format = torch.contiguous_format);  permute_353 = None
        view_359: "f32[3200, 768]" = torch.ops.aten.reshape.default(clone_198, [3200, 768]);  clone_198 = None
        mm_90: "f32[3200, 768]" = torch.ops.aten.mm.default(view_359, permute_354);  permute_354 = None
        permute_355: "f32[768, 3200]" = torch.ops.aten.permute.default(view_359, [1, 0])
        mm_91: "f32[768, 768]" = torch.ops.aten.mm.default(permute_355, view_40);  permute_355 = view_40 = None
        permute_356: "f32[768, 768]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
        sum_120: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_359, [0], True);  view_359 = None
        view_360: "f32[768]" = torch.ops.aten.reshape.default(sum_120, [768]);  sum_120 = None
        permute_357: "f32[768, 768]" = torch.ops.aten.permute.default(permute_356, [1, 0]);  permute_356 = None
        view_361: "f32[50, 64, 12, 64]" = torch.ops.aten.reshape.default(mm_90, [50, 64, 12, 64]);  mm_90 = None
        permute_358: "f32[64, 12, 50, 64]" = torch.ops.aten.permute.default(view_361, [1, 2, 0, 3]);  view_361 = None
        _scaled_dot_product_efficient_attention_backward_9 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_358, view_37, view_38, view_39, None, alias_21, getitem_19, getitem_20, getitem_21, 0.0, [True, True, True, False]);  permute_358 = view_37 = view_38 = view_39 = alias_21 = getitem_19 = getitem_20 = getitem_21 = None
        getitem_134: "f32[64, 12, 50, 64]" = _scaled_dot_product_efficient_attention_backward_9[0]
        getitem_135: "f32[64, 12, 50, 64]" = _scaled_dot_product_efficient_attention_backward_9[1]
        getitem_136: "f32[64, 12, 50, 64]" = _scaled_dot_product_efficient_attention_backward_9[2];  _scaled_dot_product_efficient_attention_backward_9 = None
        clone_199: "f32[64, 12, 50, 64]" = torch.ops.aten.clone.default(getitem_136, memory_format = torch.contiguous_format);  getitem_136 = None
        view_362: "f32[768, 50, 64]" = torch.ops.aten.reshape.default(clone_199, [768, 50, 64]);  clone_199 = None
        clone_200: "f32[64, 12, 50, 64]" = torch.ops.aten.clone.default(getitem_135, memory_format = torch.contiguous_format);  getitem_135 = None
        view_363: "f32[768, 50, 64]" = torch.ops.aten.reshape.default(clone_200, [768, 50, 64]);  clone_200 = None
        clone_201: "f32[64, 12, 50, 64]" = torch.ops.aten.clone.default(getitem_134, memory_format = torch.contiguous_format);  getitem_134 = None
        view_364: "f32[768, 50, 64]" = torch.ops.aten.reshape.default(clone_201, [768, 50, 64]);  clone_201 = None
        permute_359: "f32[50, 768, 64]" = torch.ops.aten.permute.default(view_362, [1, 0, 2]);  view_362 = None
        clone_202: "f32[50, 768, 64]" = torch.ops.aten.clone.default(permute_359, memory_format = torch.contiguous_format);  permute_359 = None
        view_365: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(clone_202, [50, 64, 768]);  clone_202 = None
        permute_360: "f32[50, 768, 64]" = torch.ops.aten.permute.default(view_363, [1, 0, 2]);  view_363 = None
        clone_203: "f32[50, 768, 64]" = torch.ops.aten.clone.default(permute_360, memory_format = torch.contiguous_format);  permute_360 = None
        view_366: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(clone_203, [50, 64, 768]);  clone_203 = None
        permute_361: "f32[50, 768, 64]" = torch.ops.aten.permute.default(view_364, [1, 0, 2]);  view_364 = None
        clone_204: "f32[50, 768, 64]" = torch.ops.aten.clone.default(permute_361, memory_format = torch.contiguous_format);  permute_361 = None
        view_367: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(clone_204, [50, 64, 768]);  clone_204 = None
        select_scatter_28: "f32[3, 50, 64, 768]" = torch.ops.aten.select_scatter.default(full_default_2, view_365, 0, 2);  view_365 = None
        select_scatter_29: "f32[3, 50, 64, 768]" = torch.ops.aten.select_scatter.default(full_default_2, view_366, 0, 1);  view_366 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        add_156: "f32[3, 50, 64, 768]" = torch.ops.aten.add.Tensor(select_scatter_28, select_scatter_29);  select_scatter_28 = select_scatter_29 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        select_scatter_30: "f32[3, 50, 64, 768]" = torch.ops.aten.select_scatter.default(full_default_2, view_367, 0, 0);  view_367 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        add_157: "f32[3, 50, 64, 768]" = torch.ops.aten.add.Tensor(add_156, select_scatter_30);  add_156 = select_scatter_30 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        unsqueeze_21: "f32[3, 50, 64, 1, 768]" = torch.ops.aten.unsqueeze.default(add_157, 3);  add_157 = None
        permute_362: "f32[1, 50, 64, 3, 768]" = torch.ops.aten.permute.default(unsqueeze_21, [3, 1, 2, 0, 4]);  unsqueeze_21 = None
        squeeze_21: "f32[50, 64, 3, 768]" = torch.ops.aten.squeeze.dim(permute_362, 0);  permute_362 = None
        clone_205: "f32[50, 64, 3, 768]" = torch.ops.aten.clone.default(squeeze_21, memory_format = torch.contiguous_format);  squeeze_21 = None
        view_368: "f32[50, 64, 2304]" = torch.ops.aten.reshape.default(clone_205, [50, 64, 2304]);  clone_205 = None
        sum_121: "f32[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_368, [0, 1], True)
        view_369: "f32[2304]" = torch.ops.aten.reshape.default(sum_121, [2304]);  sum_121 = None
        view_370: "f32[3200, 2304]" = torch.ops.aten.reshape.default(view_368, [3200, 2304]);  view_368 = None
        permute_363: "f32[2304, 3200]" = torch.ops.aten.permute.default(view_370, [1, 0])
        mm_92: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_363, view_31);  permute_363 = view_31 = None
        permute_364: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_92, [1, 0]);  mm_92 = None
        mm_93: "f32[3200, 768]" = torch.ops.aten.mm.default(view_370, permute_365);  view_370 = permute_365 = None
        view_371: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(mm_93, [50, 64, 768]);  mm_93 = None
        permute_366: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_364, [1, 0]);  permute_364 = None
        permute_367: "f32[64, 50, 768]" = torch.ops.aten.permute.default(view_371, [1, 0, 2]);  view_371 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:112, code: x = self.ln_1(input)
        clone_206: "f32[64, 50, 768]" = torch.ops.aten.clone.default(permute_367, memory_format = torch.contiguous_format);  permute_367 = None
        mul_297: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(clone_206, primals_29);  primals_29 = None
        mul_298: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_297, 768)
        sum_122: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_297, [2], True)
        mul_299: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_297, mul_14);  mul_297 = None
        sum_123: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_299, [2], True);  mul_299 = None
        mul_300: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_14, sum_123);  sum_123 = None
        sub_86: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(mul_298, sum_122);  mul_298 = sum_122 = None
        sub_87: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(sub_86, mul_300);  sub_86 = mul_300 = None
        mul_301: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(div_20, sub_87);  div_20 = sub_87 = None
        mul_302: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(clone_206, mul_14);  mul_14 = None
        sum_124: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_302, [0, 1]);  mul_302 = None
        sum_125: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_206, [0, 1]);  clone_206 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:112, code: x = self.ln_1(input)
        add_158: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(add_155, mul_301);  add_155 = mul_301 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        view_372: "f32[3200, 768]" = torch.ops.aten.reshape.default(add_158, [3200, 768])
        mm_94: "f32[3200, 3072]" = torch.ops.aten.mm.default(view_372, permute_368);  permute_368 = None
        permute_369: "f32[768, 3200]" = torch.ops.aten.permute.default(view_372, [1, 0])
        mm_95: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_369, view_29);  permute_369 = view_29 = None
        permute_370: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
        sum_126: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_372, [0], True);  view_372 = None
        view_373: "f32[768]" = torch.ops.aten.reshape.default(sum_126, [768]);  sum_126 = None
        permute_371: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_370, [1, 0]);  permute_370 = None
        view_374: "f32[64, 50, 3072]" = torch.ops.aten.reshape.default(mm_94, [64, 50, 3072]);  mm_94 = None
        mul_304: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(add_15, 0.5);  add_15 = None
        mul_305: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_28, view_28)
        mul_306: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(mul_305, -0.5);  mul_305 = None
        exp_10: "f32[64, 50, 3072]" = torch.ops.aten.exp.default(mul_306);  mul_306 = None
        mul_307: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(exp_10, 0.3989422804014327);  exp_10 = None
        mul_308: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_28, mul_307);  view_28 = mul_307 = None
        add_160: "f32[64, 50, 3072]" = torch.ops.aten.add.Tensor(mul_304, mul_308);  mul_304 = mul_308 = None
        mul_309: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_374, add_160);  view_374 = add_160 = None
        view_375: "f32[3200, 3072]" = torch.ops.aten.reshape.default(mul_309, [3200, 3072]);  mul_309 = None
        mm_96: "f32[3200, 768]" = torch.ops.aten.mm.default(view_375, permute_372);  permute_372 = None
        permute_373: "f32[3072, 3200]" = torch.ops.aten.permute.default(view_375, [1, 0])
        mm_97: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_373, view_27);  permute_373 = view_27 = None
        permute_374: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
        sum_127: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_375, [0], True);  view_375 = None
        view_376: "f32[3072]" = torch.ops.aten.reshape.default(sum_127, [3072]);  sum_127 = None
        permute_375: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_374, [1, 0]);  permute_374 = None
        view_377: "f32[64, 50, 768]" = torch.ops.aten.reshape.default(mm_96, [64, 50, 768]);  mm_96 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:117, code: y = self.ln_2(x)
        mul_311: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(view_377, primals_23);  primals_23 = None
        mul_312: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_311, 768)
        sum_128: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_311, [2], True)
        mul_313: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_311, mul_9);  mul_311 = None
        sum_129: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_313, [2], True);  mul_313 = None
        mul_314: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_9, sum_129);  sum_129 = None
        sub_89: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(mul_312, sum_128);  mul_312 = sum_128 = None
        sub_90: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(sub_89, mul_314);  sub_89 = mul_314 = None
        mul_315: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(div_21, sub_90);  div_21 = sub_90 = None
        mul_316: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(view_377, mul_9);  mul_9 = None
        sum_130: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_316, [0, 1]);  mul_316 = None
        sum_131: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_377, [0, 1]);  view_377 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:117, code: y = self.ln_2(x)
        add_161: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(add_158, mul_315);  add_158 = mul_315 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_376: "f32[50, 64, 768]" = torch.ops.aten.permute.default(add_161, [1, 0, 2])
        clone_209: "f32[50, 64, 768]" = torch.ops.aten.clone.default(permute_376, memory_format = torch.contiguous_format);  permute_376 = None
        view_378: "f32[3200, 768]" = torch.ops.aten.reshape.default(clone_209, [3200, 768]);  clone_209 = None
        mm_98: "f32[3200, 768]" = torch.ops.aten.mm.default(view_378, permute_377);  permute_377 = None
        permute_378: "f32[768, 3200]" = torch.ops.aten.permute.default(view_378, [1, 0])
        mm_99: "f32[768, 768]" = torch.ops.aten.mm.default(permute_378, view_25);  permute_378 = view_25 = None
        permute_379: "f32[768, 768]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
        sum_132: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_378, [0], True);  view_378 = None
        view_379: "f32[768]" = torch.ops.aten.reshape.default(sum_132, [768]);  sum_132 = None
        permute_380: "f32[768, 768]" = torch.ops.aten.permute.default(permute_379, [1, 0]);  permute_379 = None
        view_380: "f32[50, 64, 12, 64]" = torch.ops.aten.reshape.default(mm_98, [50, 64, 12, 64]);  mm_98 = None
        permute_381: "f32[64, 12, 50, 64]" = torch.ops.aten.permute.default(view_380, [1, 2, 0, 3]);  view_380 = None
        _scaled_dot_product_efficient_attention_backward_10 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_381, view_22, view_23, view_24, None, alias_22, getitem_11, getitem_12, getitem_13, 0.0, [True, True, True, False]);  permute_381 = view_22 = view_23 = view_24 = alias_22 = getitem_11 = getitem_12 = getitem_13 = None
        getitem_138: "f32[64, 12, 50, 64]" = _scaled_dot_product_efficient_attention_backward_10[0]
        getitem_139: "f32[64, 12, 50, 64]" = _scaled_dot_product_efficient_attention_backward_10[1]
        getitem_140: "f32[64, 12, 50, 64]" = _scaled_dot_product_efficient_attention_backward_10[2];  _scaled_dot_product_efficient_attention_backward_10 = None
        clone_210: "f32[64, 12, 50, 64]" = torch.ops.aten.clone.default(getitem_140, memory_format = torch.contiguous_format);  getitem_140 = None
        view_381: "f32[768, 50, 64]" = torch.ops.aten.reshape.default(clone_210, [768, 50, 64]);  clone_210 = None
        clone_211: "f32[64, 12, 50, 64]" = torch.ops.aten.clone.default(getitem_139, memory_format = torch.contiguous_format);  getitem_139 = None
        view_382: "f32[768, 50, 64]" = torch.ops.aten.reshape.default(clone_211, [768, 50, 64]);  clone_211 = None
        clone_212: "f32[64, 12, 50, 64]" = torch.ops.aten.clone.default(getitem_138, memory_format = torch.contiguous_format);  getitem_138 = None
        view_383: "f32[768, 50, 64]" = torch.ops.aten.reshape.default(clone_212, [768, 50, 64]);  clone_212 = None
        permute_382: "f32[50, 768, 64]" = torch.ops.aten.permute.default(view_381, [1, 0, 2]);  view_381 = None
        clone_213: "f32[50, 768, 64]" = torch.ops.aten.clone.default(permute_382, memory_format = torch.contiguous_format);  permute_382 = None
        view_384: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(clone_213, [50, 64, 768]);  clone_213 = None
        permute_383: "f32[50, 768, 64]" = torch.ops.aten.permute.default(view_382, [1, 0, 2]);  view_382 = None
        clone_214: "f32[50, 768, 64]" = torch.ops.aten.clone.default(permute_383, memory_format = torch.contiguous_format);  permute_383 = None
        view_385: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(clone_214, [50, 64, 768]);  clone_214 = None
        permute_384: "f32[50, 768, 64]" = torch.ops.aten.permute.default(view_383, [1, 0, 2]);  view_383 = None
        clone_215: "f32[50, 768, 64]" = torch.ops.aten.clone.default(permute_384, memory_format = torch.contiguous_format);  permute_384 = None
        view_386: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(clone_215, [50, 64, 768]);  clone_215 = None
        select_scatter_31: "f32[3, 50, 64, 768]" = torch.ops.aten.select_scatter.default(full_default_2, view_384, 0, 2);  view_384 = None
        select_scatter_32: "f32[3, 50, 64, 768]" = torch.ops.aten.select_scatter.default(full_default_2, view_385, 0, 1);  view_385 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        add_162: "f32[3, 50, 64, 768]" = torch.ops.aten.add.Tensor(select_scatter_31, select_scatter_32);  select_scatter_31 = select_scatter_32 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        select_scatter_33: "f32[3, 50, 64, 768]" = torch.ops.aten.select_scatter.default(full_default_2, view_386, 0, 0);  view_386 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        add_163: "f32[3, 50, 64, 768]" = torch.ops.aten.add.Tensor(add_162, select_scatter_33);  add_162 = select_scatter_33 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        unsqueeze_22: "f32[3, 50, 64, 1, 768]" = torch.ops.aten.unsqueeze.default(add_163, 3);  add_163 = None
        permute_385: "f32[1, 50, 64, 3, 768]" = torch.ops.aten.permute.default(unsqueeze_22, [3, 1, 2, 0, 4]);  unsqueeze_22 = None
        squeeze_22: "f32[50, 64, 3, 768]" = torch.ops.aten.squeeze.dim(permute_385, 0);  permute_385 = None
        clone_216: "f32[50, 64, 3, 768]" = torch.ops.aten.clone.default(squeeze_22, memory_format = torch.contiguous_format);  squeeze_22 = None
        view_387: "f32[50, 64, 2304]" = torch.ops.aten.reshape.default(clone_216, [50, 64, 2304]);  clone_216 = None
        sum_133: "f32[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_387, [0, 1], True)
        view_388: "f32[2304]" = torch.ops.aten.reshape.default(sum_133, [2304]);  sum_133 = None
        view_389: "f32[3200, 2304]" = torch.ops.aten.reshape.default(view_387, [3200, 2304]);  view_387 = None
        permute_386: "f32[2304, 3200]" = torch.ops.aten.permute.default(view_389, [1, 0])
        mm_100: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_386, view_16);  permute_386 = view_16 = None
        permute_387: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_100, [1, 0]);  mm_100 = None
        mm_101: "f32[3200, 768]" = torch.ops.aten.mm.default(view_389, permute_388);  view_389 = permute_388 = None
        view_390: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(mm_101, [50, 64, 768]);  mm_101 = None
        permute_389: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_387, [1, 0]);  permute_387 = None
        permute_390: "f32[64, 50, 768]" = torch.ops.aten.permute.default(view_390, [1, 0, 2]);  view_390 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:112, code: x = self.ln_1(input)
        clone_217: "f32[64, 50, 768]" = torch.ops.aten.clone.default(permute_390, memory_format = torch.contiguous_format);  permute_390 = None
        mul_318: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(clone_217, primals_17);  primals_17 = None
        mul_319: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_318, 768)
        sum_134: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_318, [2], True)
        mul_320: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_318, mul_7);  mul_318 = None
        sum_135: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_320, [2], True);  mul_320 = None
        mul_321: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_7, sum_135);  sum_135 = None
        sub_92: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(mul_319, sum_134);  mul_319 = sum_134 = None
        sub_93: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(sub_92, mul_321);  sub_92 = mul_321 = None
        mul_322: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(div_22, sub_93);  div_22 = sub_93 = None
        mul_323: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(clone_217, mul_7);  mul_7 = None
        sum_136: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_323, [0, 1]);  mul_323 = None
        sum_137: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_217, [0, 1]);  clone_217 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:112, code: x = self.ln_1(input)
        add_164: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(add_161, mul_322);  add_161 = mul_322 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:118, code: y = self.mlp(y)
        view_391: "f32[3200, 768]" = torch.ops.aten.reshape.default(add_164, [3200, 768])
        mm_102: "f32[3200, 3072]" = torch.ops.aten.mm.default(view_391, permute_391);  permute_391 = None
        permute_392: "f32[768, 3200]" = torch.ops.aten.permute.default(view_391, [1, 0])
        mm_103: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_392, view_14);  permute_392 = view_14 = None
        permute_393: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
        sum_138: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_391, [0], True);  view_391 = None
        view_392: "f32[768]" = torch.ops.aten.reshape.default(sum_138, [768]);  sum_138 = None
        permute_394: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_393, [1, 0]);  permute_393 = None
        view_393: "f32[64, 50, 3072]" = torch.ops.aten.reshape.default(mm_102, [64, 50, 3072]);  mm_102 = None
        mul_325: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(add_7, 0.5);  add_7 = None
        mul_326: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_13, view_13)
        mul_327: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(mul_326, -0.5);  mul_326 = None
        exp_11: "f32[64, 50, 3072]" = torch.ops.aten.exp.default(mul_327);  mul_327 = None
        mul_328: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(exp_11, 0.3989422804014327);  exp_11 = None
        mul_329: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_13, mul_328);  view_13 = mul_328 = None
        add_166: "f32[64, 50, 3072]" = torch.ops.aten.add.Tensor(mul_325, mul_329);  mul_325 = mul_329 = None
        mul_330: "f32[64, 50, 3072]" = torch.ops.aten.mul.Tensor(view_393, add_166);  view_393 = add_166 = None
        view_394: "f32[3200, 3072]" = torch.ops.aten.reshape.default(mul_330, [3200, 3072]);  mul_330 = None
        mm_104: "f32[3200, 768]" = torch.ops.aten.mm.default(view_394, permute_395);  permute_395 = None
        permute_396: "f32[3072, 3200]" = torch.ops.aten.permute.default(view_394, [1, 0])
        mm_105: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_396, view_12);  permute_396 = view_12 = None
        permute_397: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
        sum_139: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_394, [0], True);  view_394 = None
        view_395: "f32[3072]" = torch.ops.aten.reshape.default(sum_139, [3072]);  sum_139 = None
        permute_398: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_397, [1, 0]);  permute_397 = None
        view_396: "f32[64, 50, 768]" = torch.ops.aten.reshape.default(mm_104, [64, 50, 768]);  mm_104 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:117, code: y = self.ln_2(x)
        mul_332: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(view_396, primals_11);  primals_11 = None
        mul_333: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_332, 768)
        sum_140: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_332, [2], True)
        mul_334: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_332, mul_2);  mul_332 = None
        sum_141: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_334, [2], True);  mul_334 = None
        mul_335: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_2, sum_141);  sum_141 = None
        sub_95: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(mul_333, sum_140);  mul_333 = sum_140 = None
        sub_96: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(sub_95, mul_335);  sub_95 = mul_335 = None
        mul_336: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(div_23, sub_96);  div_23 = sub_96 = None
        mul_337: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(view_396, mul_2);  mul_2 = None
        sum_142: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_337, [0, 1]);  mul_337 = None
        sum_143: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_396, [0, 1]);  view_396 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:117, code: y = self.ln_2(x)
        add_167: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(add_164, mul_336);  add_164 = mul_336 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_399: "f32[50, 64, 768]" = torch.ops.aten.permute.default(add_167, [1, 0, 2])
        clone_220: "f32[50, 64, 768]" = torch.ops.aten.clone.default(permute_399, memory_format = torch.contiguous_format);  permute_399 = None
        view_397: "f32[3200, 768]" = torch.ops.aten.reshape.default(clone_220, [3200, 768]);  clone_220 = None
        mm_106: "f32[3200, 768]" = torch.ops.aten.mm.default(view_397, permute_400);  permute_400 = None
        permute_401: "f32[768, 3200]" = torch.ops.aten.permute.default(view_397, [1, 0])
        mm_107: "f32[768, 768]" = torch.ops.aten.mm.default(permute_401, view_10);  permute_401 = view_10 = None
        permute_402: "f32[768, 768]" = torch.ops.aten.permute.default(mm_107, [1, 0]);  mm_107 = None
        sum_144: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_397, [0], True);  view_397 = None
        view_398: "f32[768]" = torch.ops.aten.reshape.default(sum_144, [768]);  sum_144 = None
        permute_403: "f32[768, 768]" = torch.ops.aten.permute.default(permute_402, [1, 0]);  permute_402 = None
        view_399: "f32[50, 64, 12, 64]" = torch.ops.aten.reshape.default(mm_106, [50, 64, 12, 64]);  mm_106 = None
        permute_404: "f32[64, 12, 50, 64]" = torch.ops.aten.permute.default(view_399, [1, 2, 0, 3]);  view_399 = None
        _scaled_dot_product_efficient_attention_backward_11 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_404, view_7, view_8, view_9, None, alias_23, getitem_3, getitem_4, getitem_5, 0.0, [True, True, True, False]);  permute_404 = view_7 = view_8 = view_9 = alias_23 = getitem_3 = getitem_4 = getitem_5 = None
        getitem_142: "f32[64, 12, 50, 64]" = _scaled_dot_product_efficient_attention_backward_11[0]
        getitem_143: "f32[64, 12, 50, 64]" = _scaled_dot_product_efficient_attention_backward_11[1]
        getitem_144: "f32[64, 12, 50, 64]" = _scaled_dot_product_efficient_attention_backward_11[2];  _scaled_dot_product_efficient_attention_backward_11 = None
        clone_221: "f32[64, 12, 50, 64]" = torch.ops.aten.clone.default(getitem_144, memory_format = torch.contiguous_format);  getitem_144 = None
        view_400: "f32[768, 50, 64]" = torch.ops.aten.reshape.default(clone_221, [768, 50, 64]);  clone_221 = None
        clone_222: "f32[64, 12, 50, 64]" = torch.ops.aten.clone.default(getitem_143, memory_format = torch.contiguous_format);  getitem_143 = None
        view_401: "f32[768, 50, 64]" = torch.ops.aten.reshape.default(clone_222, [768, 50, 64]);  clone_222 = None
        clone_223: "f32[64, 12, 50, 64]" = torch.ops.aten.clone.default(getitem_142, memory_format = torch.contiguous_format);  getitem_142 = None
        view_402: "f32[768, 50, 64]" = torch.ops.aten.reshape.default(clone_223, [768, 50, 64]);  clone_223 = None
        permute_405: "f32[50, 768, 64]" = torch.ops.aten.permute.default(view_400, [1, 0, 2]);  view_400 = None
        clone_224: "f32[50, 768, 64]" = torch.ops.aten.clone.default(permute_405, memory_format = torch.contiguous_format);  permute_405 = None
        view_403: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(clone_224, [50, 64, 768]);  clone_224 = None
        permute_406: "f32[50, 768, 64]" = torch.ops.aten.permute.default(view_401, [1, 0, 2]);  view_401 = None
        clone_225: "f32[50, 768, 64]" = torch.ops.aten.clone.default(permute_406, memory_format = torch.contiguous_format);  permute_406 = None
        view_404: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(clone_225, [50, 64, 768]);  clone_225 = None
        permute_407: "f32[50, 768, 64]" = torch.ops.aten.permute.default(view_402, [1, 0, 2]);  view_402 = None
        clone_226: "f32[50, 768, 64]" = torch.ops.aten.clone.default(permute_407, memory_format = torch.contiguous_format);  permute_407 = None
        view_405: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(clone_226, [50, 64, 768]);  clone_226 = None
        select_scatter_34: "f32[3, 50, 64, 768]" = torch.ops.aten.select_scatter.default(full_default_2, view_403, 0, 2);  view_403 = None
        select_scatter_35: "f32[3, 50, 64, 768]" = torch.ops.aten.select_scatter.default(full_default_2, view_404, 0, 1);  view_404 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        add_168: "f32[3, 50, 64, 768]" = torch.ops.aten.add.Tensor(select_scatter_34, select_scatter_35);  select_scatter_34 = select_scatter_35 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        select_scatter_36: "f32[3, 50, 64, 768]" = torch.ops.aten.select_scatter.default(full_default_2, view_405, 0, 0);  full_default_2 = view_405 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        add_169: "f32[3, 50, 64, 768]" = torch.ops.aten.add.Tensor(add_168, select_scatter_36);  add_168 = select_scatter_36 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:113, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        unsqueeze_23: "f32[3, 50, 64, 1, 768]" = torch.ops.aten.unsqueeze.default(add_169, 3);  add_169 = None
        permute_408: "f32[1, 50, 64, 3, 768]" = torch.ops.aten.permute.default(unsqueeze_23, [3, 1, 2, 0, 4]);  unsqueeze_23 = None
        squeeze_23: "f32[50, 64, 3, 768]" = torch.ops.aten.squeeze.dim(permute_408, 0);  permute_408 = None
        clone_227: "f32[50, 64, 3, 768]" = torch.ops.aten.clone.default(squeeze_23, memory_format = torch.contiguous_format);  squeeze_23 = None
        view_406: "f32[50, 64, 2304]" = torch.ops.aten.reshape.default(clone_227, [50, 64, 2304]);  clone_227 = None
        sum_145: "f32[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_406, [0, 1], True)
        view_407: "f32[2304]" = torch.ops.aten.reshape.default(sum_145, [2304]);  sum_145 = None
        view_408: "f32[3200, 2304]" = torch.ops.aten.reshape.default(view_406, [3200, 2304]);  view_406 = None
        permute_409: "f32[2304, 3200]" = torch.ops.aten.permute.default(view_408, [1, 0])
        mm_108: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_409, view_1);  permute_409 = view_1 = None
        permute_410: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_108, [1, 0]);  mm_108 = None
        mm_109: "f32[3200, 768]" = torch.ops.aten.mm.default(view_408, permute_411);  view_408 = permute_411 = None
        view_409: "f32[50, 64, 768]" = torch.ops.aten.reshape.default(mm_109, [50, 64, 768]);  mm_109 = None
        permute_412: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_410, [1, 0]);  permute_410 = None
        permute_413: "f32[64, 50, 768]" = torch.ops.aten.permute.default(view_409, [1, 0, 2]);  view_409 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:112, code: x = self.ln_1(input)
        clone_228: "f32[64, 50, 768]" = torch.ops.aten.clone.default(permute_413, memory_format = torch.contiguous_format);  permute_413 = None
        mul_339: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(clone_228, primals_5);  primals_5 = None
        mul_340: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_339, 768)
        sum_146: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_339, [2], True)
        mul_341: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul_339, mul);  mul_339 = None
        sum_147: "f32[64, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_341, [2], True);  mul_341 = None
        mul_342: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(mul, sum_147);  sum_147 = None
        sub_98: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(mul_340, sum_146);  mul_340 = sum_146 = None
        sub_99: "f32[64, 50, 768]" = torch.ops.aten.sub.Tensor(sub_98, mul_342);  sub_98 = mul_342 = None
        mul_343: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(div_24, sub_99);  div_24 = sub_99 = None
        mul_344: "f32[64, 50, 768]" = torch.ops.aten.mul.Tensor(clone_228, mul);  mul = None
        sum_148: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_344, [0, 1]);  mul_344 = None
        sum_149: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_228, [0, 1]);  clone_228 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:112, code: x = self.ln_1(input)
        add_170: "f32[64, 50, 768]" = torch.ops.aten.add.Tensor(add_167, mul_343);  add_167 = mul_343 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:156, code: input = input + self.pos_embedding
        sum_150: "f32[1, 50, 768]" = torch.ops.aten.sum.dim_IntList(add_170, [0], True)
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:296, code: x = torch.cat([batch_class_token, x], dim=1)
        slice_2: "f32[64, 1, 768]" = torch.ops.aten.slice.Tensor(add_170, 1, 0, 1)
        slice_3: "f32[64, 49, 768]" = torch.ops.aten.slice.Tensor(add_170, 1, 1, 50);  add_170 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:295, code: batch_class_token = self.class_token.expand(n, -1, -1)
        sum_151: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(slice_2, [0], True);  slice_2 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:285, code: x = x.permute(0, 2, 1)
        permute_414: "f32[64, 768, 49]" = torch.ops.aten.permute.default(slice_3, [0, 2, 1]);  slice_3 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:279, code: x = x.reshape(n, self.hidden_dim, n_h * n_w)
        view_410: "f32[64, 768, 7, 7]" = torch.ops.aten.reshape.default(permute_414, [64, 768, 7, 7]);  permute_414 = None
        
        # File: /home/zibo/.local/lib/python3.10/site-packages/torchvision/models/vision_transformer.py:277, code: x = self.conv_proj(x)
        sum_152: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_410, [0, 2, 3])
        convolution_backward = torch.ops.aten.convolution_backward.default(view_410, primals_153, primals_3, [768], [32, 32], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_410 = primals_153 = primals_3 = None
        getitem_146: "f32[64, 3, 224, 224]" = convolution_backward[0]
        getitem_147: "f32[768, 3, 32, 32]" = convolution_backward[1];  convolution_backward = None
        return [sum_151, sum_150, getitem_147, sum_152, sum_148, sum_149, permute_412, view_407, permute_403, view_398, sum_142, sum_143, permute_398, view_395, permute_394, view_392, sum_136, sum_137, permute_389, view_388, permute_380, view_379, sum_130, sum_131, permute_375, view_376, permute_371, view_373, sum_124, sum_125, permute_366, view_369, permute_357, view_360, sum_118, sum_119, permute_352, view_357, permute_348, view_354, sum_112, sum_113, permute_343, view_350, permute_334, view_341, sum_106, sum_107, permute_329, view_338, permute_325, view_335, sum_100, sum_101, permute_320, view_331, permute_311, view_322, sum_94, sum_95, permute_306, view_319, permute_302, view_316, sum_88, sum_89, permute_297, view_312, permute_288, view_303, sum_82, sum_83, permute_283, view_300, permute_279, view_297, sum_76, sum_77, permute_274, view_293, permute_265, view_284, sum_70, sum_71, permute_260, view_281, permute_256, view_278, sum_64, sum_65, permute_251, view_274, permute_242, view_265, sum_58, sum_59, permute_237, view_262, permute_233, view_259, sum_52, sum_53, permute_228, view_255, permute_219, view_246, sum_46, sum_47, permute_214, view_243, permute_210, view_240, sum_40, sum_41, permute_205, view_236, permute_196, view_227, sum_34, sum_35, permute_191, view_224, permute_187, view_221, sum_28, sum_29, permute_182, view_217, permute_173, view_208, sum_22, sum_23, permute_168, view_205, permute_164, view_202, sum_16, sum_17, permute_159, view_198, permute_150, view_189, sum_10, sum_11, permute_145, view_186, permute_141, view_183, sum_4, sum_5, permute_137, view_181, getitem_146]
        