class GraphModule(torch.nn.Module):
    def forward(self, primals_4: "f32[768]", primals_14: "f32[768]", primals_20: "f32[768]", primals_30: "f32[768]", primals_36: "f32[768]", primals_46: "f32[768]", primals_52: "f32[768]", primals_62: "f32[768]", primals_68: "f32[768]", primals_78: "f32[768]", primals_84: "f32[768]", primals_94: "f32[768]", primals_100: "f32[768]", primals_110: "f32[768]", primals_116: "f32[768]", primals_126: "f32[768]", primals_132: "f32[768]", primals_142: "f32[768]", primals_148: "f32[768]", primals_158: "f32[768]", primals_164: "f32[768]", primals_174: "f32[768]", primals_180: "f32[768]", primals_190: "f32[768]", primals_196: "f32[768]", primals_202: "i64[3, 5]", expand: "i64[3, 5]", slice_6: "i64[1, 5]", mul_1: "f32[3, 5, 768]", gt: "b8[3, 5, 768]", view: "f32[15, 768]", gt_1: "b8[3, 12, 5, 5]", view_16: "f32[15, 768]", gt_2: "b8[3, 5, 768]", mul_9: "f32[3, 5, 768]", view_18: "f32[15, 768]", addmm_4: "f32[15, 3072]", view_20: "f32[15, 3072]", gt_3: "b8[3, 5, 768]", mul_16: "f32[3, 5, 768]", view_22: "f32[15, 768]", gt_4: "b8[3, 12, 5, 5]", view_38: "f32[15, 768]", gt_5: "b8[3, 5, 768]", mul_22: "f32[3, 5, 768]", view_40: "f32[15, 768]", addmm_10: "f32[15, 3072]", view_42: "f32[15, 3072]", gt_6: "b8[3, 5, 768]", mul_29: "f32[3, 5, 768]", view_44: "f32[15, 768]", gt_7: "b8[3, 12, 5, 5]", view_60: "f32[15, 768]", gt_8: "b8[3, 5, 768]", mul_35: "f32[3, 5, 768]", view_62: "f32[15, 768]", addmm_16: "f32[15, 3072]", view_64: "f32[15, 3072]", gt_9: "b8[3, 5, 768]", mul_42: "f32[3, 5, 768]", view_66: "f32[15, 768]", gt_10: "b8[3, 12, 5, 5]", view_82: "f32[15, 768]", gt_11: "b8[3, 5, 768]", mul_48: "f32[3, 5, 768]", view_84: "f32[15, 768]", addmm_22: "f32[15, 3072]", view_86: "f32[15, 3072]", gt_12: "b8[3, 5, 768]", mul_55: "f32[3, 5, 768]", view_88: "f32[15, 768]", gt_13: "b8[3, 12, 5, 5]", view_104: "f32[15, 768]", gt_14: "b8[3, 5, 768]", mul_61: "f32[3, 5, 768]", view_106: "f32[15, 768]", addmm_28: "f32[15, 3072]", view_108: "f32[15, 3072]", gt_15: "b8[3, 5, 768]", mul_68: "f32[3, 5, 768]", view_110: "f32[15, 768]", gt_16: "b8[3, 12, 5, 5]", view_126: "f32[15, 768]", gt_17: "b8[3, 5, 768]", mul_74: "f32[3, 5, 768]", view_128: "f32[15, 768]", addmm_34: "f32[15, 3072]", view_130: "f32[15, 3072]", gt_18: "b8[3, 5, 768]", mul_81: "f32[3, 5, 768]", view_132: "f32[15, 768]", gt_19: "b8[3, 12, 5, 5]", view_148: "f32[15, 768]", gt_20: "b8[3, 5, 768]", mul_87: "f32[3, 5, 768]", view_150: "f32[15, 768]", addmm_40: "f32[15, 3072]", view_152: "f32[15, 3072]", gt_21: "b8[3, 5, 768]", mul_94: "f32[3, 5, 768]", view_154: "f32[15, 768]", gt_22: "b8[3, 12, 5, 5]", view_170: "f32[15, 768]", gt_23: "b8[3, 5, 768]", mul_100: "f32[3, 5, 768]", view_172: "f32[15, 768]", addmm_46: "f32[15, 3072]", view_174: "f32[15, 3072]", gt_24: "b8[3, 5, 768]", mul_107: "f32[3, 5, 768]", view_176: "f32[15, 768]", gt_25: "b8[3, 12, 5, 5]", view_192: "f32[15, 768]", gt_26: "b8[3, 5, 768]", mul_113: "f32[3, 5, 768]", view_194: "f32[15, 768]", addmm_52: "f32[15, 3072]", view_196: "f32[15, 3072]", gt_27: "b8[3, 5, 768]", mul_120: "f32[3, 5, 768]", view_198: "f32[15, 768]", gt_28: "b8[3, 12, 5, 5]", view_214: "f32[15, 768]", gt_29: "b8[3, 5, 768]", mul_126: "f32[3, 5, 768]", view_216: "f32[15, 768]", addmm_58: "f32[15, 3072]", view_218: "f32[15, 3072]", gt_30: "b8[3, 5, 768]", mul_133: "f32[3, 5, 768]", view_220: "f32[15, 768]", gt_31: "b8[3, 12, 5, 5]", view_236: "f32[15, 768]", gt_32: "b8[3, 5, 768]", mul_139: "f32[3, 5, 768]", view_238: "f32[15, 768]", addmm_64: "f32[15, 3072]", view_240: "f32[15, 3072]", gt_33: "b8[3, 5, 768]", mul_146: "f32[3, 5, 768]", view_242: "f32[15, 768]", gt_34: "b8[3, 12, 5, 5]", view_258: "f32[15, 768]", gt_35: "b8[3, 5, 768]", mul_152: "f32[3, 5, 768]", view_260: "f32[15, 768]", addmm_70: "f32[15, 3072]", view_262: "f32[15, 3072]", gt_36: "b8[3, 5, 768]", mul_159: "f32[3, 5, 768]", select: "f32[3, 768]", tanh: "f32[3, 768]", permute_133: "f32[768, 768]", div_24: "f32[3, 5, 1]", permute_137: "f32[768, 3072]", permute_141: "f32[3072, 768]", div_25: "f32[3, 5, 1]", permute_145: "f32[768, 768]", permute_150: "f32[36, 5, 5]", permute_151: "f32[36, 64, 5]", alias_14: "f32[3, 12, 5, 5]", permute_152: "f32[36, 64, 5]", permute_153: "f32[36, 5, 64]", permute_157: "f32[768, 768]", permute_162: "f32[768, 768]", permute_166: "f32[768, 768]", div_27: "f32[3, 5, 1]", permute_170: "f32[768, 3072]", permute_174: "f32[3072, 768]", div_28: "f32[3, 5, 1]", permute_178: "f32[768, 768]", permute_183: "f32[36, 5, 5]", permute_184: "f32[36, 64, 5]", alias_15: "f32[3, 12, 5, 5]", permute_185: "f32[36, 64, 5]", permute_186: "f32[36, 5, 64]", permute_190: "f32[768, 768]", permute_195: "f32[768, 768]", permute_199: "f32[768, 768]", div_30: "f32[3, 5, 1]", permute_203: "f32[768, 3072]", permute_207: "f32[3072, 768]", div_31: "f32[3, 5, 1]", permute_211: "f32[768, 768]", permute_216: "f32[36, 5, 5]", permute_217: "f32[36, 64, 5]", alias_16: "f32[3, 12, 5, 5]", permute_218: "f32[36, 64, 5]", permute_219: "f32[36, 5, 64]", permute_223: "f32[768, 768]", permute_228: "f32[768, 768]", permute_232: "f32[768, 768]", div_33: "f32[3, 5, 1]", permute_236: "f32[768, 3072]", permute_240: "f32[3072, 768]", div_34: "f32[3, 5, 1]", permute_244: "f32[768, 768]", permute_249: "f32[36, 5, 5]", permute_250: "f32[36, 64, 5]", alias_17: "f32[3, 12, 5, 5]", permute_251: "f32[36, 64, 5]", permute_252: "f32[36, 5, 64]", permute_256: "f32[768, 768]", permute_261: "f32[768, 768]", permute_265: "f32[768, 768]", div_36: "f32[3, 5, 1]", permute_269: "f32[768, 3072]", permute_273: "f32[3072, 768]", div_37: "f32[3, 5, 1]", permute_277: "f32[768, 768]", permute_282: "f32[36, 5, 5]", permute_283: "f32[36, 64, 5]", alias_18: "f32[3, 12, 5, 5]", permute_284: "f32[36, 64, 5]", permute_285: "f32[36, 5, 64]", permute_289: "f32[768, 768]", permute_294: "f32[768, 768]", permute_298: "f32[768, 768]", div_39: "f32[3, 5, 1]", permute_302: "f32[768, 3072]", permute_306: "f32[3072, 768]", div_40: "f32[3, 5, 1]", permute_310: "f32[768, 768]", permute_315: "f32[36, 5, 5]", permute_316: "f32[36, 64, 5]", alias_19: "f32[3, 12, 5, 5]", permute_317: "f32[36, 64, 5]", permute_318: "f32[36, 5, 64]", permute_322: "f32[768, 768]", permute_327: "f32[768, 768]", permute_331: "f32[768, 768]", div_42: "f32[3, 5, 1]", permute_335: "f32[768, 3072]", permute_339: "f32[3072, 768]", div_43: "f32[3, 5, 1]", permute_343: "f32[768, 768]", permute_348: "f32[36, 5, 5]", permute_349: "f32[36, 64, 5]", alias_20: "f32[3, 12, 5, 5]", permute_350: "f32[36, 64, 5]", permute_351: "f32[36, 5, 64]", permute_355: "f32[768, 768]", permute_360: "f32[768, 768]", permute_364: "f32[768, 768]", div_45: "f32[3, 5, 1]", permute_368: "f32[768, 3072]", permute_372: "f32[3072, 768]", div_46: "f32[3, 5, 1]", permute_376: "f32[768, 768]", permute_381: "f32[36, 5, 5]", permute_382: "f32[36, 64, 5]", alias_21: "f32[3, 12, 5, 5]", permute_383: "f32[36, 64, 5]", permute_384: "f32[36, 5, 64]", permute_388: "f32[768, 768]", permute_393: "f32[768, 768]", permute_397: "f32[768, 768]", div_48: "f32[3, 5, 1]", permute_401: "f32[768, 3072]", permute_405: "f32[3072, 768]", div_49: "f32[3, 5, 1]", permute_409: "f32[768, 768]", permute_414: "f32[36, 5, 5]", permute_415: "f32[36, 64, 5]", alias_22: "f32[3, 12, 5, 5]", permute_416: "f32[36, 64, 5]", permute_417: "f32[36, 5, 64]", permute_421: "f32[768, 768]", permute_426: "f32[768, 768]", permute_430: "f32[768, 768]", div_51: "f32[3, 5, 1]", permute_434: "f32[768, 3072]", permute_438: "f32[3072, 768]", div_52: "f32[3, 5, 1]", permute_442: "f32[768, 768]", permute_447: "f32[36, 5, 5]", permute_448: "f32[36, 64, 5]", alias_23: "f32[3, 12, 5, 5]", permute_449: "f32[36, 64, 5]", permute_450: "f32[36, 5, 64]", permute_454: "f32[768, 768]", permute_459: "f32[768, 768]", permute_463: "f32[768, 768]", div_54: "f32[3, 5, 1]", permute_467: "f32[768, 3072]", permute_471: "f32[3072, 768]", div_55: "f32[3, 5, 1]", permute_475: "f32[768, 768]", permute_480: "f32[36, 5, 5]", permute_481: "f32[36, 64, 5]", alias_24: "f32[3, 12, 5, 5]", permute_482: "f32[36, 64, 5]", permute_483: "f32[36, 5, 64]", permute_487: "f32[768, 768]", permute_492: "f32[768, 768]", permute_496: "f32[768, 768]", div_57: "f32[3, 5, 1]", permute_500: "f32[768, 3072]", permute_504: "f32[3072, 768]", div_58: "f32[3, 5, 1]", permute_508: "f32[768, 768]", permute_513: "f32[36, 5, 5]", permute_514: "f32[36, 64, 5]", alias_25: "f32[3, 12, 5, 5]", permute_515: "f32[36, 64, 5]", permute_516: "f32[36, 5, 64]", permute_520: "f32[768, 768]", permute_525: "f32[768, 768]", permute_529: "f32[768, 768]", div_60: "f32[3, 5, 1]", tangents_1: "f32[3, 5, 768]", tangents_2: "f32[3, 768]"):
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        view_19: "f32[3, 5, 3072]" = torch.ops.aten.reshape.default(addmm_4, [3, 5, 3072]);  addmm_4 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_12: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476)
        erf: "f32[3, 5, 3072]" = torch.ops.aten.erf.default(mul_12);  mul_12 = None
        add_8: "f32[3, 5, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        view_41: "f32[3, 5, 3072]" = torch.ops.aten.reshape.default(addmm_10, [3, 5, 3072]);  addmm_10 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_25: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476)
        erf_1: "f32[3, 5, 3072]" = torch.ops.aten.erf.default(mul_25);  mul_25 = None
        add_16: "f32[3, 5, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        view_63: "f32[3, 5, 3072]" = torch.ops.aten.reshape.default(addmm_16, [3, 5, 3072]);  addmm_16 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_38: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476)
        erf_2: "f32[3, 5, 3072]" = torch.ops.aten.erf.default(mul_38);  mul_38 = None
        add_24: "f32[3, 5, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        view_85: "f32[3, 5, 3072]" = torch.ops.aten.reshape.default(addmm_22, [3, 5, 3072]);  addmm_22 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_51: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476)
        erf_3: "f32[3, 5, 3072]" = torch.ops.aten.erf.default(mul_51);  mul_51 = None
        add_32: "f32[3, 5, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        view_107: "f32[3, 5, 3072]" = torch.ops.aten.reshape.default(addmm_28, [3, 5, 3072]);  addmm_28 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_64: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.7071067811865476)
        erf_4: "f32[3, 5, 3072]" = torch.ops.aten.erf.default(mul_64);  mul_64 = None
        add_40: "f32[3, 5, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        view_129: "f32[3, 5, 3072]" = torch.ops.aten.reshape.default(addmm_34, [3, 5, 3072]);  addmm_34 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_77: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(view_129, 0.7071067811865476)
        erf_5: "f32[3, 5, 3072]" = torch.ops.aten.erf.default(mul_77);  mul_77 = None
        add_48: "f32[3, 5, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        view_151: "f32[3, 5, 3072]" = torch.ops.aten.reshape.default(addmm_40, [3, 5, 3072]);  addmm_40 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_90: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476)
        erf_6: "f32[3, 5, 3072]" = torch.ops.aten.erf.default(mul_90);  mul_90 = None
        add_56: "f32[3, 5, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        view_173: "f32[3, 5, 3072]" = torch.ops.aten.reshape.default(addmm_46, [3, 5, 3072]);  addmm_46 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_103: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476)
        erf_7: "f32[3, 5, 3072]" = torch.ops.aten.erf.default(mul_103);  mul_103 = None
        add_64: "f32[3, 5, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        view_195: "f32[3, 5, 3072]" = torch.ops.aten.reshape.default(addmm_52, [3, 5, 3072]);  addmm_52 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_116: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(view_195, 0.7071067811865476)
        erf_8: "f32[3, 5, 3072]" = torch.ops.aten.erf.default(mul_116);  mul_116 = None
        add_72: "f32[3, 5, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        view_217: "f32[3, 5, 3072]" = torch.ops.aten.reshape.default(addmm_58, [3, 5, 3072]);  addmm_58 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_129: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476)
        erf_9: "f32[3, 5, 3072]" = torch.ops.aten.erf.default(mul_129);  mul_129 = None
        add_80: "f32[3, 5, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        view_239: "f32[3, 5, 3072]" = torch.ops.aten.reshape.default(addmm_64, [3, 5, 3072]);  addmm_64 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_142: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(view_239, 0.7071067811865476)
        erf_10: "f32[3, 5, 3072]" = torch.ops.aten.erf.default(mul_142);  mul_142 = None
        add_88: "f32[3, 5, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        view_261: "f32[3, 5, 3072]" = torch.ops.aten.reshape.default(addmm_70, [3, 5, 3072]);  addmm_70 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_155: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(view_261, 0.7071067811865476)
        erf_11: "f32[3, 5, 3072]" = torch.ops.aten.erf.default(mul_155);  mul_155 = None
        add_96: "f32[3, 5, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:656, code: pooled_output = self.activation(pooled_output)
        mul_161: "f32[3, 768]" = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
        sub_38: "f32[3, 768]" = torch.ops.aten.sub.Tensor(1, mul_161);  mul_161 = None
        mul_162: "f32[3, 768]" = torch.ops.aten.mul.Tensor(tangents_2, sub_38);  tangents_2 = sub_38 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:655, code: pooled_output = self.dense(first_token_tensor)
        mm: "f32[3, 768]" = torch.ops.aten.mm.default(mul_162, permute_133);  permute_133 = None
        permute_134: "f32[768, 3]" = torch.ops.aten.permute.default(mul_162, [1, 0])
        mm_1: "f32[768, 768]" = torch.ops.aten.mm.default(permute_134, select);  permute_134 = select = None
        permute_135: "f32[768, 768]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
        sum_13: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(mul_162, [0], True);  mul_162 = None
        view_264: "f32[768]" = torch.ops.aten.reshape.default(sum_13, [768]);  sum_13 = None
        permute_136: "f32[768, 768]" = torch.ops.aten.permute.default(permute_135, [1, 0]);  permute_135 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:654, code: first_token_tensor = hidden_states[:, 0]
        full_default: "f32[3, 5, 768]" = torch.ops.aten.full.default([3, 5, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter: "f32[3, 5, 768]" = torch.ops.aten.select_scatter.default(full_default, mm, 1, 0);  full_default = mm = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:654, code: first_token_tensor = hidden_states[:, 0]
        add_100: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(tangents_1, select_scatter);  tangents_1 = select_scatter = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_164: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_100, primals_196);  primals_196 = None
        mul_165: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_164, 768)
        sum_14: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_164, [2], True)
        mul_166: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_164, mul_159);  mul_164 = None
        sum_15: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_166, [2], True);  mul_166 = None
        mul_167: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_159, sum_15);  sum_15 = None
        sub_40: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(mul_165, sum_14);  mul_165 = sum_14 = None
        sub_41: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(sub_40, mul_167);  sub_40 = mul_167 = None
        mul_168: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(div_24, sub_41);  div_24 = sub_41 = None
        mul_169: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_100, mul_159);  mul_159 = None
        sum_16: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_169, [0, 1]);  mul_169 = None
        sum_17: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_100, [0, 1]);  add_100 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:457, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1: "f32[3, 5, 768]" = torch.ops.prims.convert_element_type.default(gt_36, torch.float32);  gt_36 = None
        mul_170: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_171: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_168, mul_170);  mul_170 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:456, code: hidden_states = self.dense(hidden_states)
        view_265: "f32[15, 768]" = torch.ops.aten.reshape.default(mul_171, [15, 768]);  mul_171 = None
        mm_2: "f32[15, 3072]" = torch.ops.aten.mm.default(view_265, permute_137);  permute_137 = None
        permute_138: "f32[768, 15]" = torch.ops.aten.permute.default(view_265, [1, 0])
        mm_3: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_138, view_262);  permute_138 = view_262 = None
        permute_139: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
        sum_18: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_265, [0], True);  view_265 = None
        view_266: "f32[768]" = torch.ops.aten.reshape.default(sum_18, [768]);  sum_18 = None
        permute_140: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_139, [1, 0]);  permute_139 = None
        view_267: "f32[3, 5, 3072]" = torch.ops.aten.reshape.default(mm_2, [3, 5, 3072]);  mm_2 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_173: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(add_96, 0.5);  add_96 = None
        mul_174: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(view_261, view_261)
        mul_175: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(mul_174, -0.5);  mul_174 = None
        exp_12: "f32[3, 5, 3072]" = torch.ops.aten.exp.default(mul_175);  mul_175 = None
        mul_176: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(exp_12, 0.3989422804014327);  exp_12 = None
        mul_177: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(view_261, mul_176);  view_261 = mul_176 = None
        add_102: "f32[3, 5, 3072]" = torch.ops.aten.add.Tensor(mul_173, mul_177);  mul_173 = mul_177 = None
        mul_178: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(view_267, add_102);  view_267 = add_102 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        view_268: "f32[15, 3072]" = torch.ops.aten.reshape.default(mul_178, [15, 3072]);  mul_178 = None
        mm_4: "f32[15, 768]" = torch.ops.aten.mm.default(view_268, permute_141);  permute_141 = None
        permute_142: "f32[3072, 15]" = torch.ops.aten.permute.default(view_268, [1, 0])
        mm_5: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_142, view_260);  permute_142 = view_260 = None
        permute_143: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
        sum_19: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_268, [0], True);  view_268 = None
        view_269: "f32[3072]" = torch.ops.aten.reshape.default(sum_19, [3072]);  sum_19 = None
        permute_144: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
        view_270: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_4, [3, 5, 768]);  mm_4 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        add_103: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(mul_168, view_270);  mul_168 = view_270 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_180: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_103, primals_190);  primals_190 = None
        mul_181: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_180, 768)
        sum_20: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_180, [2], True)
        mul_182: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_180, mul_152);  mul_180 = None
        sum_21: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_182, [2], True);  mul_182 = None
        mul_183: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_152, sum_21);  sum_21 = None
        sub_43: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(mul_181, sum_20);  mul_181 = sum_20 = None
        sub_44: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(sub_43, mul_183);  sub_43 = mul_183 = None
        mul_184: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(div_25, sub_44);  div_25 = sub_44 = None
        mul_185: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_103, mul_152);  mul_152 = None
        sum_22: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_185, [0, 1]);  mul_185 = None
        sum_23: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_103, [0, 1]);  add_103 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:379, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_2: "f32[3, 5, 768]" = torch.ops.prims.convert_element_type.default(gt_35, torch.float32);  gt_35 = None
        mul_186: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
        mul_187: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_184, mul_186);  mul_186 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:378, code: hidden_states = self.dense(hidden_states)
        view_271: "f32[15, 768]" = torch.ops.aten.reshape.default(mul_187, [15, 768]);  mul_187 = None
        mm_6: "f32[15, 768]" = torch.ops.aten.mm.default(view_271, permute_145);  permute_145 = None
        permute_146: "f32[768, 15]" = torch.ops.aten.permute.default(view_271, [1, 0])
        mm_7: "f32[768, 768]" = torch.ops.aten.mm.default(permute_146, view_258);  permute_146 = view_258 = None
        permute_147: "f32[768, 768]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
        sum_24: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_271, [0], True);  view_271 = None
        view_272: "f32[768]" = torch.ops.aten.reshape.default(sum_24, [768]);  sum_24 = None
        permute_148: "f32[768, 768]" = torch.ops.aten.permute.default(permute_147, [1, 0]);  permute_147 = None
        view_273: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_6, [3, 5, 768]);  mm_6 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:361, code: context_layer = context_layer.view(new_context_layer_shape)
        view_274: "f32[3, 5, 12, 64]" = torch.ops.aten.reshape.default(view_273, [3, 5, 12, 64]);  view_273 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_149: "f32[3, 12, 5, 64]" = torch.ops.aten.permute.default(view_274, [0, 2, 1, 3]);  view_274 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:357, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_50: "f32[3, 12, 5, 64]" = torch.ops.aten.clone.default(permute_149, memory_format = torch.contiguous_format);  permute_149 = None
        view_275: "f32[36, 5, 64]" = torch.ops.aten.reshape.default(clone_50, [36, 5, 64]);  clone_50 = None
        bmm_24: "f32[36, 5, 64]" = torch.ops.aten.bmm.default(permute_150, view_275);  permute_150 = None
        bmm_25: "f32[36, 5, 5]" = torch.ops.aten.bmm.default(view_275, permute_151);  view_275 = permute_151 = None
        view_276: "f32[3, 12, 5, 64]" = torch.ops.aten.reshape.default(bmm_24, [3, 12, 5, 64]);  bmm_24 = None
        view_277: "f32[3, 12, 5, 5]" = torch.ops.aten.reshape.default(bmm_25, [3, 12, 5, 5]);  bmm_25 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:351, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_3: "f32[3, 12, 5, 5]" = torch.ops.prims.convert_element_type.default(gt_34, torch.float32);  gt_34 = None
        mul_188: "f32[3, 12, 5, 5]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1.1111111111111112);  convert_element_type_3 = None
        mul_189: "f32[3, 12, 5, 5]" = torch.ops.aten.mul.Tensor(view_277, mul_188);  view_277 = mul_188 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:347, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_190: "f32[3, 12, 5, 5]" = torch.ops.aten.mul.Tensor(mul_189, alias_14);  mul_189 = None
        sum_25: "f32[3, 12, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_190, [-1], True)
        mul_191: "f32[3, 12, 5, 5]" = torch.ops.aten.mul.Tensor(alias_14, sum_25);  alias_14 = sum_25 = None
        sub_45: "f32[3, 12, 5, 5]" = torch.ops.aten.sub.Tensor(mul_190, mul_191);  mul_190 = mul_191 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:341, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_26: "f32[3, 12, 5, 5]" = torch.ops.aten.div.Tensor(sub_45, 8.0);  sub_45 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:323, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        view_278: "f32[36, 5, 5]" = torch.ops.aten.reshape.default(div_26, [36, 5, 5]);  div_26 = None
        bmm_26: "f32[36, 64, 5]" = torch.ops.aten.bmm.default(permute_152, view_278);  permute_152 = None
        bmm_27: "f32[36, 5, 64]" = torch.ops.aten.bmm.default(view_278, permute_153);  view_278 = permute_153 = None
        view_279: "f32[3, 12, 64, 5]" = torch.ops.aten.reshape.default(bmm_26, [3, 12, 64, 5]);  bmm_26 = None
        view_280: "f32[3, 12, 5, 64]" = torch.ops.aten.reshape.default(bmm_27, [3, 12, 5, 64]);  bmm_27 = None
        permute_154: "f32[3, 12, 5, 64]" = torch.ops.aten.permute.default(view_279, [0, 1, 3, 2]);  view_279 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_155: "f32[3, 5, 12, 64]" = torch.ops.aten.permute.default(view_280, [0, 2, 1, 3]);  view_280 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        clone_52: "f32[3, 5, 12, 64]" = torch.ops.aten.clone.default(permute_155, memory_format = torch.contiguous_format);  permute_155 = None
        view_281: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(clone_52, [3, 5, 768]);  clone_52 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_156: "f32[3, 5, 12, 64]" = torch.ops.aten.permute.default(view_276, [0, 2, 1, 3]);  view_276 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        clone_53: "f32[3, 5, 12, 64]" = torch.ops.aten.clone.default(permute_156, memory_format = torch.contiguous_format);  permute_156 = None
        view_282: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(clone_53, [3, 5, 768]);  clone_53 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_283: "f32[15, 768]" = torch.ops.aten.reshape.default(view_282, [15, 768]);  view_282 = None
        mm_8: "f32[15, 768]" = torch.ops.aten.mm.default(view_283, permute_157);  permute_157 = None
        permute_158: "f32[768, 15]" = torch.ops.aten.permute.default(view_283, [1, 0])
        mm_9: "f32[768, 768]" = torch.ops.aten.mm.default(permute_158, view_242);  permute_158 = None
        permute_159: "f32[768, 768]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
        sum_26: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_283, [0], True);  view_283 = None
        view_284: "f32[768]" = torch.ops.aten.reshape.default(sum_26, [768]);  sum_26 = None
        permute_160: "f32[768, 768]" = torch.ops.aten.permute.default(permute_159, [1, 0]);  permute_159 = None
        view_285: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_8, [3, 5, 768]);  mm_8 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        add_104: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(mul_184, view_285);  mul_184 = view_285 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_161: "f32[3, 5, 12, 64]" = torch.ops.aten.permute.default(permute_154, [0, 2, 1, 3]);  permute_154 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_286: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(permute_161, [3, 5, 768]);  permute_161 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        clone_54: "f32[3, 5, 768]" = torch.ops.aten.clone.default(view_286, memory_format = torch.contiguous_format);  view_286 = None
        view_287: "f32[15, 768]" = torch.ops.aten.reshape.default(clone_54, [15, 768]);  clone_54 = None
        mm_10: "f32[15, 768]" = torch.ops.aten.mm.default(view_287, permute_162);  permute_162 = None
        permute_163: "f32[768, 15]" = torch.ops.aten.permute.default(view_287, [1, 0])
        mm_11: "f32[768, 768]" = torch.ops.aten.mm.default(permute_163, view_242);  permute_163 = None
        permute_164: "f32[768, 768]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
        sum_27: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_287, [0], True);  view_287 = None
        view_288: "f32[768]" = torch.ops.aten.reshape.default(sum_27, [768]);  sum_27 = None
        permute_165: "f32[768, 768]" = torch.ops.aten.permute.default(permute_164, [1, 0]);  permute_164 = None
        view_289: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_10, [3, 5, 768]);  mm_10 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        add_105: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(add_104, view_289);  add_104 = view_289 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        view_290: "f32[15, 768]" = torch.ops.aten.reshape.default(view_281, [15, 768]);  view_281 = None
        mm_12: "f32[15, 768]" = torch.ops.aten.mm.default(view_290, permute_166);  permute_166 = None
        permute_167: "f32[768, 15]" = torch.ops.aten.permute.default(view_290, [1, 0])
        mm_13: "f32[768, 768]" = torch.ops.aten.mm.default(permute_167, view_242);  permute_167 = view_242 = None
        permute_168: "f32[768, 768]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
        sum_28: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_290, [0], True);  view_290 = None
        view_291: "f32[768]" = torch.ops.aten.reshape.default(sum_28, [768]);  sum_28 = None
        permute_169: "f32[768, 768]" = torch.ops.aten.permute.default(permute_168, [1, 0]);  permute_168 = None
        view_292: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_12, [3, 5, 768]);  mm_12 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        add_106: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(add_105, view_292);  add_105 = view_292 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_193: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_106, primals_180);  primals_180 = None
        mul_194: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_193, 768)
        sum_29: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_193, [2], True)
        mul_195: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_193, mul_146);  mul_193 = None
        sum_30: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_195, [2], True);  mul_195 = None
        mul_196: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_146, sum_30);  sum_30 = None
        sub_47: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(mul_194, sum_29);  mul_194 = sum_29 = None
        sub_48: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(sub_47, mul_196);  sub_47 = mul_196 = None
        mul_197: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(div_27, sub_48);  div_27 = sub_48 = None
        mul_198: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_106, mul_146);  mul_146 = None
        sum_31: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_198, [0, 1]);  mul_198 = None
        sum_32: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_106, [0, 1]);  add_106 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:457, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_4: "f32[3, 5, 768]" = torch.ops.prims.convert_element_type.default(gt_33, torch.float32);  gt_33 = None
        mul_199: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 1.1111111111111112);  convert_element_type_4 = None
        mul_200: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_197, mul_199);  mul_199 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:456, code: hidden_states = self.dense(hidden_states)
        view_293: "f32[15, 768]" = torch.ops.aten.reshape.default(mul_200, [15, 768]);  mul_200 = None
        mm_14: "f32[15, 3072]" = torch.ops.aten.mm.default(view_293, permute_170);  permute_170 = None
        permute_171: "f32[768, 15]" = torch.ops.aten.permute.default(view_293, [1, 0])
        mm_15: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_171, view_240);  permute_171 = view_240 = None
        permute_172: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
        sum_33: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_293, [0], True);  view_293 = None
        view_294: "f32[768]" = torch.ops.aten.reshape.default(sum_33, [768]);  sum_33 = None
        permute_173: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_172, [1, 0]);  permute_172 = None
        view_295: "f32[3, 5, 3072]" = torch.ops.aten.reshape.default(mm_14, [3, 5, 3072]);  mm_14 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_202: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(add_88, 0.5);  add_88 = None
        mul_203: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(view_239, view_239)
        mul_204: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(mul_203, -0.5);  mul_203 = None
        exp_13: "f32[3, 5, 3072]" = torch.ops.aten.exp.default(mul_204);  mul_204 = None
        mul_205: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(exp_13, 0.3989422804014327);  exp_13 = None
        mul_206: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(view_239, mul_205);  view_239 = mul_205 = None
        add_108: "f32[3, 5, 3072]" = torch.ops.aten.add.Tensor(mul_202, mul_206);  mul_202 = mul_206 = None
        mul_207: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(view_295, add_108);  view_295 = add_108 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        view_296: "f32[15, 3072]" = torch.ops.aten.reshape.default(mul_207, [15, 3072]);  mul_207 = None
        mm_16: "f32[15, 768]" = torch.ops.aten.mm.default(view_296, permute_174);  permute_174 = None
        permute_175: "f32[3072, 15]" = torch.ops.aten.permute.default(view_296, [1, 0])
        mm_17: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_175, view_238);  permute_175 = view_238 = None
        permute_176: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
        sum_34: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_296, [0], True);  view_296 = None
        view_297: "f32[3072]" = torch.ops.aten.reshape.default(sum_34, [3072]);  sum_34 = None
        permute_177: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
        view_298: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_16, [3, 5, 768]);  mm_16 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        add_109: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(mul_197, view_298);  mul_197 = view_298 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_209: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_109, primals_174);  primals_174 = None
        mul_210: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_209, 768)
        sum_35: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_209, [2], True)
        mul_211: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_209, mul_139);  mul_209 = None
        sum_36: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_211, [2], True);  mul_211 = None
        mul_212: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_139, sum_36);  sum_36 = None
        sub_50: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(mul_210, sum_35);  mul_210 = sum_35 = None
        sub_51: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(sub_50, mul_212);  sub_50 = mul_212 = None
        mul_213: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(div_28, sub_51);  div_28 = sub_51 = None
        mul_214: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_109, mul_139);  mul_139 = None
        sum_37: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_214, [0, 1]);  mul_214 = None
        sum_38: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_109, [0, 1]);  add_109 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:379, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_5: "f32[3, 5, 768]" = torch.ops.prims.convert_element_type.default(gt_32, torch.float32);  gt_32 = None
        mul_215: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_5, 1.1111111111111112);  convert_element_type_5 = None
        mul_216: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_213, mul_215);  mul_215 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:378, code: hidden_states = self.dense(hidden_states)
        view_299: "f32[15, 768]" = torch.ops.aten.reshape.default(mul_216, [15, 768]);  mul_216 = None
        mm_18: "f32[15, 768]" = torch.ops.aten.mm.default(view_299, permute_178);  permute_178 = None
        permute_179: "f32[768, 15]" = torch.ops.aten.permute.default(view_299, [1, 0])
        mm_19: "f32[768, 768]" = torch.ops.aten.mm.default(permute_179, view_236);  permute_179 = view_236 = None
        permute_180: "f32[768, 768]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
        sum_39: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_299, [0], True);  view_299 = None
        view_300: "f32[768]" = torch.ops.aten.reshape.default(sum_39, [768]);  sum_39 = None
        permute_181: "f32[768, 768]" = torch.ops.aten.permute.default(permute_180, [1, 0]);  permute_180 = None
        view_301: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_18, [3, 5, 768]);  mm_18 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:361, code: context_layer = context_layer.view(new_context_layer_shape)
        view_302: "f32[3, 5, 12, 64]" = torch.ops.aten.reshape.default(view_301, [3, 5, 12, 64]);  view_301 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_182: "f32[3, 12, 5, 64]" = torch.ops.aten.permute.default(view_302, [0, 2, 1, 3]);  view_302 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:357, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_57: "f32[3, 12, 5, 64]" = torch.ops.aten.clone.default(permute_182, memory_format = torch.contiguous_format);  permute_182 = None
        view_303: "f32[36, 5, 64]" = torch.ops.aten.reshape.default(clone_57, [36, 5, 64]);  clone_57 = None
        bmm_28: "f32[36, 5, 64]" = torch.ops.aten.bmm.default(permute_183, view_303);  permute_183 = None
        bmm_29: "f32[36, 5, 5]" = torch.ops.aten.bmm.default(view_303, permute_184);  view_303 = permute_184 = None
        view_304: "f32[3, 12, 5, 64]" = torch.ops.aten.reshape.default(bmm_28, [3, 12, 5, 64]);  bmm_28 = None
        view_305: "f32[3, 12, 5, 5]" = torch.ops.aten.reshape.default(bmm_29, [3, 12, 5, 5]);  bmm_29 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:351, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_6: "f32[3, 12, 5, 5]" = torch.ops.prims.convert_element_type.default(gt_31, torch.float32);  gt_31 = None
        mul_217: "f32[3, 12, 5, 5]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 1.1111111111111112);  convert_element_type_6 = None
        mul_218: "f32[3, 12, 5, 5]" = torch.ops.aten.mul.Tensor(view_305, mul_217);  view_305 = mul_217 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:347, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_219: "f32[3, 12, 5, 5]" = torch.ops.aten.mul.Tensor(mul_218, alias_15);  mul_218 = None
        sum_40: "f32[3, 12, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_219, [-1], True)
        mul_220: "f32[3, 12, 5, 5]" = torch.ops.aten.mul.Tensor(alias_15, sum_40);  alias_15 = sum_40 = None
        sub_52: "f32[3, 12, 5, 5]" = torch.ops.aten.sub.Tensor(mul_219, mul_220);  mul_219 = mul_220 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:341, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_29: "f32[3, 12, 5, 5]" = torch.ops.aten.div.Tensor(sub_52, 8.0);  sub_52 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:323, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        view_306: "f32[36, 5, 5]" = torch.ops.aten.reshape.default(div_29, [36, 5, 5]);  div_29 = None
        bmm_30: "f32[36, 64, 5]" = torch.ops.aten.bmm.default(permute_185, view_306);  permute_185 = None
        bmm_31: "f32[36, 5, 64]" = torch.ops.aten.bmm.default(view_306, permute_186);  view_306 = permute_186 = None
        view_307: "f32[3, 12, 64, 5]" = torch.ops.aten.reshape.default(bmm_30, [3, 12, 64, 5]);  bmm_30 = None
        view_308: "f32[3, 12, 5, 64]" = torch.ops.aten.reshape.default(bmm_31, [3, 12, 5, 64]);  bmm_31 = None
        permute_187: "f32[3, 12, 5, 64]" = torch.ops.aten.permute.default(view_307, [0, 1, 3, 2]);  view_307 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_188: "f32[3, 5, 12, 64]" = torch.ops.aten.permute.default(view_308, [0, 2, 1, 3]);  view_308 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        clone_59: "f32[3, 5, 12, 64]" = torch.ops.aten.clone.default(permute_188, memory_format = torch.contiguous_format);  permute_188 = None
        view_309: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(clone_59, [3, 5, 768]);  clone_59 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_189: "f32[3, 5, 12, 64]" = torch.ops.aten.permute.default(view_304, [0, 2, 1, 3]);  view_304 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        clone_60: "f32[3, 5, 12, 64]" = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format);  permute_189 = None
        view_310: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(clone_60, [3, 5, 768]);  clone_60 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_311: "f32[15, 768]" = torch.ops.aten.reshape.default(view_310, [15, 768]);  view_310 = None
        mm_20: "f32[15, 768]" = torch.ops.aten.mm.default(view_311, permute_190);  permute_190 = None
        permute_191: "f32[768, 15]" = torch.ops.aten.permute.default(view_311, [1, 0])
        mm_21: "f32[768, 768]" = torch.ops.aten.mm.default(permute_191, view_220);  permute_191 = None
        permute_192: "f32[768, 768]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
        sum_41: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_311, [0], True);  view_311 = None
        view_312: "f32[768]" = torch.ops.aten.reshape.default(sum_41, [768]);  sum_41 = None
        permute_193: "f32[768, 768]" = torch.ops.aten.permute.default(permute_192, [1, 0]);  permute_192 = None
        view_313: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_20, [3, 5, 768]);  mm_20 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        add_110: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(mul_213, view_313);  mul_213 = view_313 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_194: "f32[3, 5, 12, 64]" = torch.ops.aten.permute.default(permute_187, [0, 2, 1, 3]);  permute_187 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_314: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(permute_194, [3, 5, 768]);  permute_194 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        clone_61: "f32[3, 5, 768]" = torch.ops.aten.clone.default(view_314, memory_format = torch.contiguous_format);  view_314 = None
        view_315: "f32[15, 768]" = torch.ops.aten.reshape.default(clone_61, [15, 768]);  clone_61 = None
        mm_22: "f32[15, 768]" = torch.ops.aten.mm.default(view_315, permute_195);  permute_195 = None
        permute_196: "f32[768, 15]" = torch.ops.aten.permute.default(view_315, [1, 0])
        mm_23: "f32[768, 768]" = torch.ops.aten.mm.default(permute_196, view_220);  permute_196 = None
        permute_197: "f32[768, 768]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
        sum_42: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_315, [0], True);  view_315 = None
        view_316: "f32[768]" = torch.ops.aten.reshape.default(sum_42, [768]);  sum_42 = None
        permute_198: "f32[768, 768]" = torch.ops.aten.permute.default(permute_197, [1, 0]);  permute_197 = None
        view_317: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_22, [3, 5, 768]);  mm_22 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        add_111: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(add_110, view_317);  add_110 = view_317 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        view_318: "f32[15, 768]" = torch.ops.aten.reshape.default(view_309, [15, 768]);  view_309 = None
        mm_24: "f32[15, 768]" = torch.ops.aten.mm.default(view_318, permute_199);  permute_199 = None
        permute_200: "f32[768, 15]" = torch.ops.aten.permute.default(view_318, [1, 0])
        mm_25: "f32[768, 768]" = torch.ops.aten.mm.default(permute_200, view_220);  permute_200 = view_220 = None
        permute_201: "f32[768, 768]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
        sum_43: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_318, [0], True);  view_318 = None
        view_319: "f32[768]" = torch.ops.aten.reshape.default(sum_43, [768]);  sum_43 = None
        permute_202: "f32[768, 768]" = torch.ops.aten.permute.default(permute_201, [1, 0]);  permute_201 = None
        view_320: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_24, [3, 5, 768]);  mm_24 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        add_112: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(add_111, view_320);  add_111 = view_320 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_222: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_112, primals_164);  primals_164 = None
        mul_223: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_222, 768)
        sum_44: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_222, [2], True)
        mul_224: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_222, mul_133);  mul_222 = None
        sum_45: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_224, [2], True);  mul_224 = None
        mul_225: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_133, sum_45);  sum_45 = None
        sub_54: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(mul_223, sum_44);  mul_223 = sum_44 = None
        sub_55: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(sub_54, mul_225);  sub_54 = mul_225 = None
        mul_226: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(div_30, sub_55);  div_30 = sub_55 = None
        mul_227: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_112, mul_133);  mul_133 = None
        sum_46: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_227, [0, 1]);  mul_227 = None
        sum_47: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_112, [0, 1]);  add_112 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:457, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_7: "f32[3, 5, 768]" = torch.ops.prims.convert_element_type.default(gt_30, torch.float32);  gt_30 = None
        mul_228: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_7, 1.1111111111111112);  convert_element_type_7 = None
        mul_229: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_226, mul_228);  mul_228 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:456, code: hidden_states = self.dense(hidden_states)
        view_321: "f32[15, 768]" = torch.ops.aten.reshape.default(mul_229, [15, 768]);  mul_229 = None
        mm_26: "f32[15, 3072]" = torch.ops.aten.mm.default(view_321, permute_203);  permute_203 = None
        permute_204: "f32[768, 15]" = torch.ops.aten.permute.default(view_321, [1, 0])
        mm_27: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_204, view_218);  permute_204 = view_218 = None
        permute_205: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
        sum_48: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_321, [0], True);  view_321 = None
        view_322: "f32[768]" = torch.ops.aten.reshape.default(sum_48, [768]);  sum_48 = None
        permute_206: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_205, [1, 0]);  permute_205 = None
        view_323: "f32[3, 5, 3072]" = torch.ops.aten.reshape.default(mm_26, [3, 5, 3072]);  mm_26 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_231: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(add_80, 0.5);  add_80 = None
        mul_232: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(view_217, view_217)
        mul_233: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(mul_232, -0.5);  mul_232 = None
        exp_14: "f32[3, 5, 3072]" = torch.ops.aten.exp.default(mul_233);  mul_233 = None
        mul_234: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(exp_14, 0.3989422804014327);  exp_14 = None
        mul_235: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(view_217, mul_234);  view_217 = mul_234 = None
        add_114: "f32[3, 5, 3072]" = torch.ops.aten.add.Tensor(mul_231, mul_235);  mul_231 = mul_235 = None
        mul_236: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(view_323, add_114);  view_323 = add_114 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        view_324: "f32[15, 3072]" = torch.ops.aten.reshape.default(mul_236, [15, 3072]);  mul_236 = None
        mm_28: "f32[15, 768]" = torch.ops.aten.mm.default(view_324, permute_207);  permute_207 = None
        permute_208: "f32[3072, 15]" = torch.ops.aten.permute.default(view_324, [1, 0])
        mm_29: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_208, view_216);  permute_208 = view_216 = None
        permute_209: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
        sum_49: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_324, [0], True);  view_324 = None
        view_325: "f32[3072]" = torch.ops.aten.reshape.default(sum_49, [3072]);  sum_49 = None
        permute_210: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_209, [1, 0]);  permute_209 = None
        view_326: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_28, [3, 5, 768]);  mm_28 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        add_115: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(mul_226, view_326);  mul_226 = view_326 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_238: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_115, primals_158);  primals_158 = None
        mul_239: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_238, 768)
        sum_50: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_238, [2], True)
        mul_240: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_238, mul_126);  mul_238 = None
        sum_51: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_240, [2], True);  mul_240 = None
        mul_241: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_126, sum_51);  sum_51 = None
        sub_57: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(mul_239, sum_50);  mul_239 = sum_50 = None
        sub_58: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(sub_57, mul_241);  sub_57 = mul_241 = None
        mul_242: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(div_31, sub_58);  div_31 = sub_58 = None
        mul_243: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_115, mul_126);  mul_126 = None
        sum_52: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_243, [0, 1]);  mul_243 = None
        sum_53: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_115, [0, 1]);  add_115 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:379, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_8: "f32[3, 5, 768]" = torch.ops.prims.convert_element_type.default(gt_29, torch.float32);  gt_29 = None
        mul_244: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 1.1111111111111112);  convert_element_type_8 = None
        mul_245: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_242, mul_244);  mul_244 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:378, code: hidden_states = self.dense(hidden_states)
        view_327: "f32[15, 768]" = torch.ops.aten.reshape.default(mul_245, [15, 768]);  mul_245 = None
        mm_30: "f32[15, 768]" = torch.ops.aten.mm.default(view_327, permute_211);  permute_211 = None
        permute_212: "f32[768, 15]" = torch.ops.aten.permute.default(view_327, [1, 0])
        mm_31: "f32[768, 768]" = torch.ops.aten.mm.default(permute_212, view_214);  permute_212 = view_214 = None
        permute_213: "f32[768, 768]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
        sum_54: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_327, [0], True);  view_327 = None
        view_328: "f32[768]" = torch.ops.aten.reshape.default(sum_54, [768]);  sum_54 = None
        permute_214: "f32[768, 768]" = torch.ops.aten.permute.default(permute_213, [1, 0]);  permute_213 = None
        view_329: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_30, [3, 5, 768]);  mm_30 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:361, code: context_layer = context_layer.view(new_context_layer_shape)
        view_330: "f32[3, 5, 12, 64]" = torch.ops.aten.reshape.default(view_329, [3, 5, 12, 64]);  view_329 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_215: "f32[3, 12, 5, 64]" = torch.ops.aten.permute.default(view_330, [0, 2, 1, 3]);  view_330 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:357, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_64: "f32[3, 12, 5, 64]" = torch.ops.aten.clone.default(permute_215, memory_format = torch.contiguous_format);  permute_215 = None
        view_331: "f32[36, 5, 64]" = torch.ops.aten.reshape.default(clone_64, [36, 5, 64]);  clone_64 = None
        bmm_32: "f32[36, 5, 64]" = torch.ops.aten.bmm.default(permute_216, view_331);  permute_216 = None
        bmm_33: "f32[36, 5, 5]" = torch.ops.aten.bmm.default(view_331, permute_217);  view_331 = permute_217 = None
        view_332: "f32[3, 12, 5, 64]" = torch.ops.aten.reshape.default(bmm_32, [3, 12, 5, 64]);  bmm_32 = None
        view_333: "f32[3, 12, 5, 5]" = torch.ops.aten.reshape.default(bmm_33, [3, 12, 5, 5]);  bmm_33 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:351, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_9: "f32[3, 12, 5, 5]" = torch.ops.prims.convert_element_type.default(gt_28, torch.float32);  gt_28 = None
        mul_246: "f32[3, 12, 5, 5]" = torch.ops.aten.mul.Tensor(convert_element_type_9, 1.1111111111111112);  convert_element_type_9 = None
        mul_247: "f32[3, 12, 5, 5]" = torch.ops.aten.mul.Tensor(view_333, mul_246);  view_333 = mul_246 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:347, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_248: "f32[3, 12, 5, 5]" = torch.ops.aten.mul.Tensor(mul_247, alias_16);  mul_247 = None
        sum_55: "f32[3, 12, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_248, [-1], True)
        mul_249: "f32[3, 12, 5, 5]" = torch.ops.aten.mul.Tensor(alias_16, sum_55);  alias_16 = sum_55 = None
        sub_59: "f32[3, 12, 5, 5]" = torch.ops.aten.sub.Tensor(mul_248, mul_249);  mul_248 = mul_249 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:341, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_32: "f32[3, 12, 5, 5]" = torch.ops.aten.div.Tensor(sub_59, 8.0);  sub_59 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:323, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        view_334: "f32[36, 5, 5]" = torch.ops.aten.reshape.default(div_32, [36, 5, 5]);  div_32 = None
        bmm_34: "f32[36, 64, 5]" = torch.ops.aten.bmm.default(permute_218, view_334);  permute_218 = None
        bmm_35: "f32[36, 5, 64]" = torch.ops.aten.bmm.default(view_334, permute_219);  view_334 = permute_219 = None
        view_335: "f32[3, 12, 64, 5]" = torch.ops.aten.reshape.default(bmm_34, [3, 12, 64, 5]);  bmm_34 = None
        view_336: "f32[3, 12, 5, 64]" = torch.ops.aten.reshape.default(bmm_35, [3, 12, 5, 64]);  bmm_35 = None
        permute_220: "f32[3, 12, 5, 64]" = torch.ops.aten.permute.default(view_335, [0, 1, 3, 2]);  view_335 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_221: "f32[3, 5, 12, 64]" = torch.ops.aten.permute.default(view_336, [0, 2, 1, 3]);  view_336 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        clone_66: "f32[3, 5, 12, 64]" = torch.ops.aten.clone.default(permute_221, memory_format = torch.contiguous_format);  permute_221 = None
        view_337: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(clone_66, [3, 5, 768]);  clone_66 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_222: "f32[3, 5, 12, 64]" = torch.ops.aten.permute.default(view_332, [0, 2, 1, 3]);  view_332 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        clone_67: "f32[3, 5, 12, 64]" = torch.ops.aten.clone.default(permute_222, memory_format = torch.contiguous_format);  permute_222 = None
        view_338: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(clone_67, [3, 5, 768]);  clone_67 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_339: "f32[15, 768]" = torch.ops.aten.reshape.default(view_338, [15, 768]);  view_338 = None
        mm_32: "f32[15, 768]" = torch.ops.aten.mm.default(view_339, permute_223);  permute_223 = None
        permute_224: "f32[768, 15]" = torch.ops.aten.permute.default(view_339, [1, 0])
        mm_33: "f32[768, 768]" = torch.ops.aten.mm.default(permute_224, view_198);  permute_224 = None
        permute_225: "f32[768, 768]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
        sum_56: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_339, [0], True);  view_339 = None
        view_340: "f32[768]" = torch.ops.aten.reshape.default(sum_56, [768]);  sum_56 = None
        permute_226: "f32[768, 768]" = torch.ops.aten.permute.default(permute_225, [1, 0]);  permute_225 = None
        view_341: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_32, [3, 5, 768]);  mm_32 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        add_116: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(mul_242, view_341);  mul_242 = view_341 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_227: "f32[3, 5, 12, 64]" = torch.ops.aten.permute.default(permute_220, [0, 2, 1, 3]);  permute_220 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_342: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(permute_227, [3, 5, 768]);  permute_227 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        clone_68: "f32[3, 5, 768]" = torch.ops.aten.clone.default(view_342, memory_format = torch.contiguous_format);  view_342 = None
        view_343: "f32[15, 768]" = torch.ops.aten.reshape.default(clone_68, [15, 768]);  clone_68 = None
        mm_34: "f32[15, 768]" = torch.ops.aten.mm.default(view_343, permute_228);  permute_228 = None
        permute_229: "f32[768, 15]" = torch.ops.aten.permute.default(view_343, [1, 0])
        mm_35: "f32[768, 768]" = torch.ops.aten.mm.default(permute_229, view_198);  permute_229 = None
        permute_230: "f32[768, 768]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
        sum_57: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_343, [0], True);  view_343 = None
        view_344: "f32[768]" = torch.ops.aten.reshape.default(sum_57, [768]);  sum_57 = None
        permute_231: "f32[768, 768]" = torch.ops.aten.permute.default(permute_230, [1, 0]);  permute_230 = None
        view_345: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_34, [3, 5, 768]);  mm_34 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        add_117: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(add_116, view_345);  add_116 = view_345 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        view_346: "f32[15, 768]" = torch.ops.aten.reshape.default(view_337, [15, 768]);  view_337 = None
        mm_36: "f32[15, 768]" = torch.ops.aten.mm.default(view_346, permute_232);  permute_232 = None
        permute_233: "f32[768, 15]" = torch.ops.aten.permute.default(view_346, [1, 0])
        mm_37: "f32[768, 768]" = torch.ops.aten.mm.default(permute_233, view_198);  permute_233 = view_198 = None
        permute_234: "f32[768, 768]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
        sum_58: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_346, [0], True);  view_346 = None
        view_347: "f32[768]" = torch.ops.aten.reshape.default(sum_58, [768]);  sum_58 = None
        permute_235: "f32[768, 768]" = torch.ops.aten.permute.default(permute_234, [1, 0]);  permute_234 = None
        view_348: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_36, [3, 5, 768]);  mm_36 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        add_118: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(add_117, view_348);  add_117 = view_348 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_251: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_118, primals_148);  primals_148 = None
        mul_252: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_251, 768)
        sum_59: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_251, [2], True)
        mul_253: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_251, mul_120);  mul_251 = None
        sum_60: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_253, [2], True);  mul_253 = None
        mul_254: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_120, sum_60);  sum_60 = None
        sub_61: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(mul_252, sum_59);  mul_252 = sum_59 = None
        sub_62: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(sub_61, mul_254);  sub_61 = mul_254 = None
        mul_255: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(div_33, sub_62);  div_33 = sub_62 = None
        mul_256: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_118, mul_120);  mul_120 = None
        sum_61: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_256, [0, 1]);  mul_256 = None
        sum_62: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_118, [0, 1]);  add_118 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:457, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_10: "f32[3, 5, 768]" = torch.ops.prims.convert_element_type.default(gt_27, torch.float32);  gt_27 = None
        mul_257: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 1.1111111111111112);  convert_element_type_10 = None
        mul_258: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_255, mul_257);  mul_257 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:456, code: hidden_states = self.dense(hidden_states)
        view_349: "f32[15, 768]" = torch.ops.aten.reshape.default(mul_258, [15, 768]);  mul_258 = None
        mm_38: "f32[15, 3072]" = torch.ops.aten.mm.default(view_349, permute_236);  permute_236 = None
        permute_237: "f32[768, 15]" = torch.ops.aten.permute.default(view_349, [1, 0])
        mm_39: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_237, view_196);  permute_237 = view_196 = None
        permute_238: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
        sum_63: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_349, [0], True);  view_349 = None
        view_350: "f32[768]" = torch.ops.aten.reshape.default(sum_63, [768]);  sum_63 = None
        permute_239: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_238, [1, 0]);  permute_238 = None
        view_351: "f32[3, 5, 3072]" = torch.ops.aten.reshape.default(mm_38, [3, 5, 3072]);  mm_38 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_260: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(add_72, 0.5);  add_72 = None
        mul_261: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(view_195, view_195)
        mul_262: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(mul_261, -0.5);  mul_261 = None
        exp_15: "f32[3, 5, 3072]" = torch.ops.aten.exp.default(mul_262);  mul_262 = None
        mul_263: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(exp_15, 0.3989422804014327);  exp_15 = None
        mul_264: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(view_195, mul_263);  view_195 = mul_263 = None
        add_120: "f32[3, 5, 3072]" = torch.ops.aten.add.Tensor(mul_260, mul_264);  mul_260 = mul_264 = None
        mul_265: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(view_351, add_120);  view_351 = add_120 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        view_352: "f32[15, 3072]" = torch.ops.aten.reshape.default(mul_265, [15, 3072]);  mul_265 = None
        mm_40: "f32[15, 768]" = torch.ops.aten.mm.default(view_352, permute_240);  permute_240 = None
        permute_241: "f32[3072, 15]" = torch.ops.aten.permute.default(view_352, [1, 0])
        mm_41: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_241, view_194);  permute_241 = view_194 = None
        permute_242: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
        sum_64: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_352, [0], True);  view_352 = None
        view_353: "f32[3072]" = torch.ops.aten.reshape.default(sum_64, [3072]);  sum_64 = None
        permute_243: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_242, [1, 0]);  permute_242 = None
        view_354: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_40, [3, 5, 768]);  mm_40 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        add_121: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(mul_255, view_354);  mul_255 = view_354 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_267: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_121, primals_142);  primals_142 = None
        mul_268: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_267, 768)
        sum_65: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_267, [2], True)
        mul_269: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_267, mul_113);  mul_267 = None
        sum_66: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_269, [2], True);  mul_269 = None
        mul_270: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_113, sum_66);  sum_66 = None
        sub_64: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(mul_268, sum_65);  mul_268 = sum_65 = None
        sub_65: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(sub_64, mul_270);  sub_64 = mul_270 = None
        mul_271: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(div_34, sub_65);  div_34 = sub_65 = None
        mul_272: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_121, mul_113);  mul_113 = None
        sum_67: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_272, [0, 1]);  mul_272 = None
        sum_68: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_121, [0, 1]);  add_121 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:379, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_11: "f32[3, 5, 768]" = torch.ops.prims.convert_element_type.default(gt_26, torch.float32);  gt_26 = None
        mul_273: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 1.1111111111111112);  convert_element_type_11 = None
        mul_274: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_271, mul_273);  mul_273 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:378, code: hidden_states = self.dense(hidden_states)
        view_355: "f32[15, 768]" = torch.ops.aten.reshape.default(mul_274, [15, 768]);  mul_274 = None
        mm_42: "f32[15, 768]" = torch.ops.aten.mm.default(view_355, permute_244);  permute_244 = None
        permute_245: "f32[768, 15]" = torch.ops.aten.permute.default(view_355, [1, 0])
        mm_43: "f32[768, 768]" = torch.ops.aten.mm.default(permute_245, view_192);  permute_245 = view_192 = None
        permute_246: "f32[768, 768]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
        sum_69: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_355, [0], True);  view_355 = None
        view_356: "f32[768]" = torch.ops.aten.reshape.default(sum_69, [768]);  sum_69 = None
        permute_247: "f32[768, 768]" = torch.ops.aten.permute.default(permute_246, [1, 0]);  permute_246 = None
        view_357: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_42, [3, 5, 768]);  mm_42 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:361, code: context_layer = context_layer.view(new_context_layer_shape)
        view_358: "f32[3, 5, 12, 64]" = torch.ops.aten.reshape.default(view_357, [3, 5, 12, 64]);  view_357 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_248: "f32[3, 12, 5, 64]" = torch.ops.aten.permute.default(view_358, [0, 2, 1, 3]);  view_358 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:357, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_71: "f32[3, 12, 5, 64]" = torch.ops.aten.clone.default(permute_248, memory_format = torch.contiguous_format);  permute_248 = None
        view_359: "f32[36, 5, 64]" = torch.ops.aten.reshape.default(clone_71, [36, 5, 64]);  clone_71 = None
        bmm_36: "f32[36, 5, 64]" = torch.ops.aten.bmm.default(permute_249, view_359);  permute_249 = None
        bmm_37: "f32[36, 5, 5]" = torch.ops.aten.bmm.default(view_359, permute_250);  view_359 = permute_250 = None
        view_360: "f32[3, 12, 5, 64]" = torch.ops.aten.reshape.default(bmm_36, [3, 12, 5, 64]);  bmm_36 = None
        view_361: "f32[3, 12, 5, 5]" = torch.ops.aten.reshape.default(bmm_37, [3, 12, 5, 5]);  bmm_37 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:351, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_12: "f32[3, 12, 5, 5]" = torch.ops.prims.convert_element_type.default(gt_25, torch.float32);  gt_25 = None
        mul_275: "f32[3, 12, 5, 5]" = torch.ops.aten.mul.Tensor(convert_element_type_12, 1.1111111111111112);  convert_element_type_12 = None
        mul_276: "f32[3, 12, 5, 5]" = torch.ops.aten.mul.Tensor(view_361, mul_275);  view_361 = mul_275 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:347, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_277: "f32[3, 12, 5, 5]" = torch.ops.aten.mul.Tensor(mul_276, alias_17);  mul_276 = None
        sum_70: "f32[3, 12, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_277, [-1], True)
        mul_278: "f32[3, 12, 5, 5]" = torch.ops.aten.mul.Tensor(alias_17, sum_70);  alias_17 = sum_70 = None
        sub_66: "f32[3, 12, 5, 5]" = torch.ops.aten.sub.Tensor(mul_277, mul_278);  mul_277 = mul_278 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:341, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_35: "f32[3, 12, 5, 5]" = torch.ops.aten.div.Tensor(sub_66, 8.0);  sub_66 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:323, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        view_362: "f32[36, 5, 5]" = torch.ops.aten.reshape.default(div_35, [36, 5, 5]);  div_35 = None
        bmm_38: "f32[36, 64, 5]" = torch.ops.aten.bmm.default(permute_251, view_362);  permute_251 = None
        bmm_39: "f32[36, 5, 64]" = torch.ops.aten.bmm.default(view_362, permute_252);  view_362 = permute_252 = None
        view_363: "f32[3, 12, 64, 5]" = torch.ops.aten.reshape.default(bmm_38, [3, 12, 64, 5]);  bmm_38 = None
        view_364: "f32[3, 12, 5, 64]" = torch.ops.aten.reshape.default(bmm_39, [3, 12, 5, 64]);  bmm_39 = None
        permute_253: "f32[3, 12, 5, 64]" = torch.ops.aten.permute.default(view_363, [0, 1, 3, 2]);  view_363 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_254: "f32[3, 5, 12, 64]" = torch.ops.aten.permute.default(view_364, [0, 2, 1, 3]);  view_364 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        clone_73: "f32[3, 5, 12, 64]" = torch.ops.aten.clone.default(permute_254, memory_format = torch.contiguous_format);  permute_254 = None
        view_365: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(clone_73, [3, 5, 768]);  clone_73 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_255: "f32[3, 5, 12, 64]" = torch.ops.aten.permute.default(view_360, [0, 2, 1, 3]);  view_360 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        clone_74: "f32[3, 5, 12, 64]" = torch.ops.aten.clone.default(permute_255, memory_format = torch.contiguous_format);  permute_255 = None
        view_366: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(clone_74, [3, 5, 768]);  clone_74 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_367: "f32[15, 768]" = torch.ops.aten.reshape.default(view_366, [15, 768]);  view_366 = None
        mm_44: "f32[15, 768]" = torch.ops.aten.mm.default(view_367, permute_256);  permute_256 = None
        permute_257: "f32[768, 15]" = torch.ops.aten.permute.default(view_367, [1, 0])
        mm_45: "f32[768, 768]" = torch.ops.aten.mm.default(permute_257, view_176);  permute_257 = None
        permute_258: "f32[768, 768]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
        sum_71: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_367, [0], True);  view_367 = None
        view_368: "f32[768]" = torch.ops.aten.reshape.default(sum_71, [768]);  sum_71 = None
        permute_259: "f32[768, 768]" = torch.ops.aten.permute.default(permute_258, [1, 0]);  permute_258 = None
        view_369: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_44, [3, 5, 768]);  mm_44 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        add_122: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(mul_271, view_369);  mul_271 = view_369 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_260: "f32[3, 5, 12, 64]" = torch.ops.aten.permute.default(permute_253, [0, 2, 1, 3]);  permute_253 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_370: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(permute_260, [3, 5, 768]);  permute_260 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        clone_75: "f32[3, 5, 768]" = torch.ops.aten.clone.default(view_370, memory_format = torch.contiguous_format);  view_370 = None
        view_371: "f32[15, 768]" = torch.ops.aten.reshape.default(clone_75, [15, 768]);  clone_75 = None
        mm_46: "f32[15, 768]" = torch.ops.aten.mm.default(view_371, permute_261);  permute_261 = None
        permute_262: "f32[768, 15]" = torch.ops.aten.permute.default(view_371, [1, 0])
        mm_47: "f32[768, 768]" = torch.ops.aten.mm.default(permute_262, view_176);  permute_262 = None
        permute_263: "f32[768, 768]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
        sum_72: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_371, [0], True);  view_371 = None
        view_372: "f32[768]" = torch.ops.aten.reshape.default(sum_72, [768]);  sum_72 = None
        permute_264: "f32[768, 768]" = torch.ops.aten.permute.default(permute_263, [1, 0]);  permute_263 = None
        view_373: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_46, [3, 5, 768]);  mm_46 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        add_123: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(add_122, view_373);  add_122 = view_373 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        view_374: "f32[15, 768]" = torch.ops.aten.reshape.default(view_365, [15, 768]);  view_365 = None
        mm_48: "f32[15, 768]" = torch.ops.aten.mm.default(view_374, permute_265);  permute_265 = None
        permute_266: "f32[768, 15]" = torch.ops.aten.permute.default(view_374, [1, 0])
        mm_49: "f32[768, 768]" = torch.ops.aten.mm.default(permute_266, view_176);  permute_266 = view_176 = None
        permute_267: "f32[768, 768]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
        sum_73: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_374, [0], True);  view_374 = None
        view_375: "f32[768]" = torch.ops.aten.reshape.default(sum_73, [768]);  sum_73 = None
        permute_268: "f32[768, 768]" = torch.ops.aten.permute.default(permute_267, [1, 0]);  permute_267 = None
        view_376: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_48, [3, 5, 768]);  mm_48 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        add_124: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(add_123, view_376);  add_123 = view_376 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_280: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_124, primals_132);  primals_132 = None
        mul_281: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_280, 768)
        sum_74: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_280, [2], True)
        mul_282: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_280, mul_107);  mul_280 = None
        sum_75: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_282, [2], True);  mul_282 = None
        mul_283: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_107, sum_75);  sum_75 = None
        sub_68: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(mul_281, sum_74);  mul_281 = sum_74 = None
        sub_69: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(sub_68, mul_283);  sub_68 = mul_283 = None
        mul_284: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(div_36, sub_69);  div_36 = sub_69 = None
        mul_285: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_124, mul_107);  mul_107 = None
        sum_76: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_285, [0, 1]);  mul_285 = None
        sum_77: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_124, [0, 1]);  add_124 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:457, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_13: "f32[3, 5, 768]" = torch.ops.prims.convert_element_type.default(gt_24, torch.float32);  gt_24 = None
        mul_286: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
        mul_287: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_284, mul_286);  mul_286 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:456, code: hidden_states = self.dense(hidden_states)
        view_377: "f32[15, 768]" = torch.ops.aten.reshape.default(mul_287, [15, 768]);  mul_287 = None
        mm_50: "f32[15, 3072]" = torch.ops.aten.mm.default(view_377, permute_269);  permute_269 = None
        permute_270: "f32[768, 15]" = torch.ops.aten.permute.default(view_377, [1, 0])
        mm_51: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_270, view_174);  permute_270 = view_174 = None
        permute_271: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
        sum_78: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_377, [0], True);  view_377 = None
        view_378: "f32[768]" = torch.ops.aten.reshape.default(sum_78, [768]);  sum_78 = None
        permute_272: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_271, [1, 0]);  permute_271 = None
        view_379: "f32[3, 5, 3072]" = torch.ops.aten.reshape.default(mm_50, [3, 5, 3072]);  mm_50 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_289: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(add_64, 0.5);  add_64 = None
        mul_290: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(view_173, view_173)
        mul_291: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(mul_290, -0.5);  mul_290 = None
        exp_16: "f32[3, 5, 3072]" = torch.ops.aten.exp.default(mul_291);  mul_291 = None
        mul_292: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
        mul_293: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(view_173, mul_292);  view_173 = mul_292 = None
        add_126: "f32[3, 5, 3072]" = torch.ops.aten.add.Tensor(mul_289, mul_293);  mul_289 = mul_293 = None
        mul_294: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(view_379, add_126);  view_379 = add_126 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        view_380: "f32[15, 3072]" = torch.ops.aten.reshape.default(mul_294, [15, 3072]);  mul_294 = None
        mm_52: "f32[15, 768]" = torch.ops.aten.mm.default(view_380, permute_273);  permute_273 = None
        permute_274: "f32[3072, 15]" = torch.ops.aten.permute.default(view_380, [1, 0])
        mm_53: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_274, view_172);  permute_274 = view_172 = None
        permute_275: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
        sum_79: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_380, [0], True);  view_380 = None
        view_381: "f32[3072]" = torch.ops.aten.reshape.default(sum_79, [3072]);  sum_79 = None
        permute_276: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_275, [1, 0]);  permute_275 = None
        view_382: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_52, [3, 5, 768]);  mm_52 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        add_127: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(mul_284, view_382);  mul_284 = view_382 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_296: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_127, primals_126);  primals_126 = None
        mul_297: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_296, 768)
        sum_80: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_296, [2], True)
        mul_298: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_296, mul_100);  mul_296 = None
        sum_81: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_298, [2], True);  mul_298 = None
        mul_299: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_100, sum_81);  sum_81 = None
        sub_71: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(mul_297, sum_80);  mul_297 = sum_80 = None
        sub_72: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(sub_71, mul_299);  sub_71 = mul_299 = None
        mul_300: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(div_37, sub_72);  div_37 = sub_72 = None
        mul_301: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_127, mul_100);  mul_100 = None
        sum_82: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_301, [0, 1]);  mul_301 = None
        sum_83: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_127, [0, 1]);  add_127 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:379, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_14: "f32[3, 5, 768]" = torch.ops.prims.convert_element_type.default(gt_23, torch.float32);  gt_23 = None
        mul_302: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 1.1111111111111112);  convert_element_type_14 = None
        mul_303: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_300, mul_302);  mul_302 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:378, code: hidden_states = self.dense(hidden_states)
        view_383: "f32[15, 768]" = torch.ops.aten.reshape.default(mul_303, [15, 768]);  mul_303 = None
        mm_54: "f32[15, 768]" = torch.ops.aten.mm.default(view_383, permute_277);  permute_277 = None
        permute_278: "f32[768, 15]" = torch.ops.aten.permute.default(view_383, [1, 0])
        mm_55: "f32[768, 768]" = torch.ops.aten.mm.default(permute_278, view_170);  permute_278 = view_170 = None
        permute_279: "f32[768, 768]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
        sum_84: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_383, [0], True);  view_383 = None
        view_384: "f32[768]" = torch.ops.aten.reshape.default(sum_84, [768]);  sum_84 = None
        permute_280: "f32[768, 768]" = torch.ops.aten.permute.default(permute_279, [1, 0]);  permute_279 = None
        view_385: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_54, [3, 5, 768]);  mm_54 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:361, code: context_layer = context_layer.view(new_context_layer_shape)
        view_386: "f32[3, 5, 12, 64]" = torch.ops.aten.reshape.default(view_385, [3, 5, 12, 64]);  view_385 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_281: "f32[3, 12, 5, 64]" = torch.ops.aten.permute.default(view_386, [0, 2, 1, 3]);  view_386 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:357, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_78: "f32[3, 12, 5, 64]" = torch.ops.aten.clone.default(permute_281, memory_format = torch.contiguous_format);  permute_281 = None
        view_387: "f32[36, 5, 64]" = torch.ops.aten.reshape.default(clone_78, [36, 5, 64]);  clone_78 = None
        bmm_40: "f32[36, 5, 64]" = torch.ops.aten.bmm.default(permute_282, view_387);  permute_282 = None
        bmm_41: "f32[36, 5, 5]" = torch.ops.aten.bmm.default(view_387, permute_283);  view_387 = permute_283 = None
        view_388: "f32[3, 12, 5, 64]" = torch.ops.aten.reshape.default(bmm_40, [3, 12, 5, 64]);  bmm_40 = None
        view_389: "f32[3, 12, 5, 5]" = torch.ops.aten.reshape.default(bmm_41, [3, 12, 5, 5]);  bmm_41 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:351, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_15: "f32[3, 12, 5, 5]" = torch.ops.prims.convert_element_type.default(gt_22, torch.float32);  gt_22 = None
        mul_304: "f32[3, 12, 5, 5]" = torch.ops.aten.mul.Tensor(convert_element_type_15, 1.1111111111111112);  convert_element_type_15 = None
        mul_305: "f32[3, 12, 5, 5]" = torch.ops.aten.mul.Tensor(view_389, mul_304);  view_389 = mul_304 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:347, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_306: "f32[3, 12, 5, 5]" = torch.ops.aten.mul.Tensor(mul_305, alias_18);  mul_305 = None
        sum_85: "f32[3, 12, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_306, [-1], True)
        mul_307: "f32[3, 12, 5, 5]" = torch.ops.aten.mul.Tensor(alias_18, sum_85);  alias_18 = sum_85 = None
        sub_73: "f32[3, 12, 5, 5]" = torch.ops.aten.sub.Tensor(mul_306, mul_307);  mul_306 = mul_307 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:341, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_38: "f32[3, 12, 5, 5]" = torch.ops.aten.div.Tensor(sub_73, 8.0);  sub_73 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:323, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        view_390: "f32[36, 5, 5]" = torch.ops.aten.reshape.default(div_38, [36, 5, 5]);  div_38 = None
        bmm_42: "f32[36, 64, 5]" = torch.ops.aten.bmm.default(permute_284, view_390);  permute_284 = None
        bmm_43: "f32[36, 5, 64]" = torch.ops.aten.bmm.default(view_390, permute_285);  view_390 = permute_285 = None
        view_391: "f32[3, 12, 64, 5]" = torch.ops.aten.reshape.default(bmm_42, [3, 12, 64, 5]);  bmm_42 = None
        view_392: "f32[3, 12, 5, 64]" = torch.ops.aten.reshape.default(bmm_43, [3, 12, 5, 64]);  bmm_43 = None
        permute_286: "f32[3, 12, 5, 64]" = torch.ops.aten.permute.default(view_391, [0, 1, 3, 2]);  view_391 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_287: "f32[3, 5, 12, 64]" = torch.ops.aten.permute.default(view_392, [0, 2, 1, 3]);  view_392 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        clone_80: "f32[3, 5, 12, 64]" = torch.ops.aten.clone.default(permute_287, memory_format = torch.contiguous_format);  permute_287 = None
        view_393: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(clone_80, [3, 5, 768]);  clone_80 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_288: "f32[3, 5, 12, 64]" = torch.ops.aten.permute.default(view_388, [0, 2, 1, 3]);  view_388 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        clone_81: "f32[3, 5, 12, 64]" = torch.ops.aten.clone.default(permute_288, memory_format = torch.contiguous_format);  permute_288 = None
        view_394: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(clone_81, [3, 5, 768]);  clone_81 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_395: "f32[15, 768]" = torch.ops.aten.reshape.default(view_394, [15, 768]);  view_394 = None
        mm_56: "f32[15, 768]" = torch.ops.aten.mm.default(view_395, permute_289);  permute_289 = None
        permute_290: "f32[768, 15]" = torch.ops.aten.permute.default(view_395, [1, 0])
        mm_57: "f32[768, 768]" = torch.ops.aten.mm.default(permute_290, view_154);  permute_290 = None
        permute_291: "f32[768, 768]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
        sum_86: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_395, [0], True);  view_395 = None
        view_396: "f32[768]" = torch.ops.aten.reshape.default(sum_86, [768]);  sum_86 = None
        permute_292: "f32[768, 768]" = torch.ops.aten.permute.default(permute_291, [1, 0]);  permute_291 = None
        view_397: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_56, [3, 5, 768]);  mm_56 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        add_128: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(mul_300, view_397);  mul_300 = view_397 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_293: "f32[3, 5, 12, 64]" = torch.ops.aten.permute.default(permute_286, [0, 2, 1, 3]);  permute_286 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_398: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(permute_293, [3, 5, 768]);  permute_293 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        clone_82: "f32[3, 5, 768]" = torch.ops.aten.clone.default(view_398, memory_format = torch.contiguous_format);  view_398 = None
        view_399: "f32[15, 768]" = torch.ops.aten.reshape.default(clone_82, [15, 768]);  clone_82 = None
        mm_58: "f32[15, 768]" = torch.ops.aten.mm.default(view_399, permute_294);  permute_294 = None
        permute_295: "f32[768, 15]" = torch.ops.aten.permute.default(view_399, [1, 0])
        mm_59: "f32[768, 768]" = torch.ops.aten.mm.default(permute_295, view_154);  permute_295 = None
        permute_296: "f32[768, 768]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
        sum_87: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_399, [0], True);  view_399 = None
        view_400: "f32[768]" = torch.ops.aten.reshape.default(sum_87, [768]);  sum_87 = None
        permute_297: "f32[768, 768]" = torch.ops.aten.permute.default(permute_296, [1, 0]);  permute_296 = None
        view_401: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_58, [3, 5, 768]);  mm_58 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        add_129: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(add_128, view_401);  add_128 = view_401 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        view_402: "f32[15, 768]" = torch.ops.aten.reshape.default(view_393, [15, 768]);  view_393 = None
        mm_60: "f32[15, 768]" = torch.ops.aten.mm.default(view_402, permute_298);  permute_298 = None
        permute_299: "f32[768, 15]" = torch.ops.aten.permute.default(view_402, [1, 0])
        mm_61: "f32[768, 768]" = torch.ops.aten.mm.default(permute_299, view_154);  permute_299 = view_154 = None
        permute_300: "f32[768, 768]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
        sum_88: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_402, [0], True);  view_402 = None
        view_403: "f32[768]" = torch.ops.aten.reshape.default(sum_88, [768]);  sum_88 = None
        permute_301: "f32[768, 768]" = torch.ops.aten.permute.default(permute_300, [1, 0]);  permute_300 = None
        view_404: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_60, [3, 5, 768]);  mm_60 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        add_130: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(add_129, view_404);  add_129 = view_404 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_309: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_130, primals_116);  primals_116 = None
        mul_310: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_309, 768)
        sum_89: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_309, [2], True)
        mul_311: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_309, mul_94);  mul_309 = None
        sum_90: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_311, [2], True);  mul_311 = None
        mul_312: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_94, sum_90);  sum_90 = None
        sub_75: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(mul_310, sum_89);  mul_310 = sum_89 = None
        sub_76: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(sub_75, mul_312);  sub_75 = mul_312 = None
        mul_313: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(div_39, sub_76);  div_39 = sub_76 = None
        mul_314: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_130, mul_94);  mul_94 = None
        sum_91: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_314, [0, 1]);  mul_314 = None
        sum_92: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_130, [0, 1]);  add_130 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:457, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_16: "f32[3, 5, 768]" = torch.ops.prims.convert_element_type.default(gt_21, torch.float32);  gt_21 = None
        mul_315: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_16, 1.1111111111111112);  convert_element_type_16 = None
        mul_316: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_313, mul_315);  mul_315 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:456, code: hidden_states = self.dense(hidden_states)
        view_405: "f32[15, 768]" = torch.ops.aten.reshape.default(mul_316, [15, 768]);  mul_316 = None
        mm_62: "f32[15, 3072]" = torch.ops.aten.mm.default(view_405, permute_302);  permute_302 = None
        permute_303: "f32[768, 15]" = torch.ops.aten.permute.default(view_405, [1, 0])
        mm_63: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_303, view_152);  permute_303 = view_152 = None
        permute_304: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
        sum_93: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_405, [0], True);  view_405 = None
        view_406: "f32[768]" = torch.ops.aten.reshape.default(sum_93, [768]);  sum_93 = None
        permute_305: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_304, [1, 0]);  permute_304 = None
        view_407: "f32[3, 5, 3072]" = torch.ops.aten.reshape.default(mm_62, [3, 5, 3072]);  mm_62 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_318: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(add_56, 0.5);  add_56 = None
        mul_319: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(view_151, view_151)
        mul_320: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(mul_319, -0.5);  mul_319 = None
        exp_17: "f32[3, 5, 3072]" = torch.ops.aten.exp.default(mul_320);  mul_320 = None
        mul_321: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(exp_17, 0.3989422804014327);  exp_17 = None
        mul_322: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(view_151, mul_321);  view_151 = mul_321 = None
        add_132: "f32[3, 5, 3072]" = torch.ops.aten.add.Tensor(mul_318, mul_322);  mul_318 = mul_322 = None
        mul_323: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(view_407, add_132);  view_407 = add_132 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        view_408: "f32[15, 3072]" = torch.ops.aten.reshape.default(mul_323, [15, 3072]);  mul_323 = None
        mm_64: "f32[15, 768]" = torch.ops.aten.mm.default(view_408, permute_306);  permute_306 = None
        permute_307: "f32[3072, 15]" = torch.ops.aten.permute.default(view_408, [1, 0])
        mm_65: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_307, view_150);  permute_307 = view_150 = None
        permute_308: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
        sum_94: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_408, [0], True);  view_408 = None
        view_409: "f32[3072]" = torch.ops.aten.reshape.default(sum_94, [3072]);  sum_94 = None
        permute_309: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_308, [1, 0]);  permute_308 = None
        view_410: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_64, [3, 5, 768]);  mm_64 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        add_133: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(mul_313, view_410);  mul_313 = view_410 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_325: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_133, primals_110);  primals_110 = None
        mul_326: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_325, 768)
        sum_95: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_325, [2], True)
        mul_327: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_325, mul_87);  mul_325 = None
        sum_96: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_327, [2], True);  mul_327 = None
        mul_328: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_87, sum_96);  sum_96 = None
        sub_78: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(mul_326, sum_95);  mul_326 = sum_95 = None
        sub_79: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(sub_78, mul_328);  sub_78 = mul_328 = None
        mul_329: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(div_40, sub_79);  div_40 = sub_79 = None
        mul_330: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_133, mul_87);  mul_87 = None
        sum_97: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_330, [0, 1]);  mul_330 = None
        sum_98: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_133, [0, 1]);  add_133 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:379, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_17: "f32[3, 5, 768]" = torch.ops.prims.convert_element_type.default(gt_20, torch.float32);  gt_20 = None
        mul_331: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_17, 1.1111111111111112);  convert_element_type_17 = None
        mul_332: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_329, mul_331);  mul_331 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:378, code: hidden_states = self.dense(hidden_states)
        view_411: "f32[15, 768]" = torch.ops.aten.reshape.default(mul_332, [15, 768]);  mul_332 = None
        mm_66: "f32[15, 768]" = torch.ops.aten.mm.default(view_411, permute_310);  permute_310 = None
        permute_311: "f32[768, 15]" = torch.ops.aten.permute.default(view_411, [1, 0])
        mm_67: "f32[768, 768]" = torch.ops.aten.mm.default(permute_311, view_148);  permute_311 = view_148 = None
        permute_312: "f32[768, 768]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
        sum_99: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_411, [0], True);  view_411 = None
        view_412: "f32[768]" = torch.ops.aten.reshape.default(sum_99, [768]);  sum_99 = None
        permute_313: "f32[768, 768]" = torch.ops.aten.permute.default(permute_312, [1, 0]);  permute_312 = None
        view_413: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_66, [3, 5, 768]);  mm_66 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:361, code: context_layer = context_layer.view(new_context_layer_shape)
        view_414: "f32[3, 5, 12, 64]" = torch.ops.aten.reshape.default(view_413, [3, 5, 12, 64]);  view_413 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_314: "f32[3, 12, 5, 64]" = torch.ops.aten.permute.default(view_414, [0, 2, 1, 3]);  view_414 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:357, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_85: "f32[3, 12, 5, 64]" = torch.ops.aten.clone.default(permute_314, memory_format = torch.contiguous_format);  permute_314 = None
        view_415: "f32[36, 5, 64]" = torch.ops.aten.reshape.default(clone_85, [36, 5, 64]);  clone_85 = None
        bmm_44: "f32[36, 5, 64]" = torch.ops.aten.bmm.default(permute_315, view_415);  permute_315 = None
        bmm_45: "f32[36, 5, 5]" = torch.ops.aten.bmm.default(view_415, permute_316);  view_415 = permute_316 = None
        view_416: "f32[3, 12, 5, 64]" = torch.ops.aten.reshape.default(bmm_44, [3, 12, 5, 64]);  bmm_44 = None
        view_417: "f32[3, 12, 5, 5]" = torch.ops.aten.reshape.default(bmm_45, [3, 12, 5, 5]);  bmm_45 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:351, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_18: "f32[3, 12, 5, 5]" = torch.ops.prims.convert_element_type.default(gt_19, torch.float32);  gt_19 = None
        mul_333: "f32[3, 12, 5, 5]" = torch.ops.aten.mul.Tensor(convert_element_type_18, 1.1111111111111112);  convert_element_type_18 = None
        mul_334: "f32[3, 12, 5, 5]" = torch.ops.aten.mul.Tensor(view_417, mul_333);  view_417 = mul_333 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:347, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_335: "f32[3, 12, 5, 5]" = torch.ops.aten.mul.Tensor(mul_334, alias_19);  mul_334 = None
        sum_100: "f32[3, 12, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_335, [-1], True)
        mul_336: "f32[3, 12, 5, 5]" = torch.ops.aten.mul.Tensor(alias_19, sum_100);  alias_19 = sum_100 = None
        sub_80: "f32[3, 12, 5, 5]" = torch.ops.aten.sub.Tensor(mul_335, mul_336);  mul_335 = mul_336 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:341, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_41: "f32[3, 12, 5, 5]" = torch.ops.aten.div.Tensor(sub_80, 8.0);  sub_80 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:323, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        view_418: "f32[36, 5, 5]" = torch.ops.aten.reshape.default(div_41, [36, 5, 5]);  div_41 = None
        bmm_46: "f32[36, 64, 5]" = torch.ops.aten.bmm.default(permute_317, view_418);  permute_317 = None
        bmm_47: "f32[36, 5, 64]" = torch.ops.aten.bmm.default(view_418, permute_318);  view_418 = permute_318 = None
        view_419: "f32[3, 12, 64, 5]" = torch.ops.aten.reshape.default(bmm_46, [3, 12, 64, 5]);  bmm_46 = None
        view_420: "f32[3, 12, 5, 64]" = torch.ops.aten.reshape.default(bmm_47, [3, 12, 5, 64]);  bmm_47 = None
        permute_319: "f32[3, 12, 5, 64]" = torch.ops.aten.permute.default(view_419, [0, 1, 3, 2]);  view_419 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_320: "f32[3, 5, 12, 64]" = torch.ops.aten.permute.default(view_420, [0, 2, 1, 3]);  view_420 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        clone_87: "f32[3, 5, 12, 64]" = torch.ops.aten.clone.default(permute_320, memory_format = torch.contiguous_format);  permute_320 = None
        view_421: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(clone_87, [3, 5, 768]);  clone_87 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_321: "f32[3, 5, 12, 64]" = torch.ops.aten.permute.default(view_416, [0, 2, 1, 3]);  view_416 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        clone_88: "f32[3, 5, 12, 64]" = torch.ops.aten.clone.default(permute_321, memory_format = torch.contiguous_format);  permute_321 = None
        view_422: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(clone_88, [3, 5, 768]);  clone_88 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_423: "f32[15, 768]" = torch.ops.aten.reshape.default(view_422, [15, 768]);  view_422 = None
        mm_68: "f32[15, 768]" = torch.ops.aten.mm.default(view_423, permute_322);  permute_322 = None
        permute_323: "f32[768, 15]" = torch.ops.aten.permute.default(view_423, [1, 0])
        mm_69: "f32[768, 768]" = torch.ops.aten.mm.default(permute_323, view_132);  permute_323 = None
        permute_324: "f32[768, 768]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
        sum_101: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_423, [0], True);  view_423 = None
        view_424: "f32[768]" = torch.ops.aten.reshape.default(sum_101, [768]);  sum_101 = None
        permute_325: "f32[768, 768]" = torch.ops.aten.permute.default(permute_324, [1, 0]);  permute_324 = None
        view_425: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_68, [3, 5, 768]);  mm_68 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        add_134: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(mul_329, view_425);  mul_329 = view_425 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_326: "f32[3, 5, 12, 64]" = torch.ops.aten.permute.default(permute_319, [0, 2, 1, 3]);  permute_319 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_426: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(permute_326, [3, 5, 768]);  permute_326 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        clone_89: "f32[3, 5, 768]" = torch.ops.aten.clone.default(view_426, memory_format = torch.contiguous_format);  view_426 = None
        view_427: "f32[15, 768]" = torch.ops.aten.reshape.default(clone_89, [15, 768]);  clone_89 = None
        mm_70: "f32[15, 768]" = torch.ops.aten.mm.default(view_427, permute_327);  permute_327 = None
        permute_328: "f32[768, 15]" = torch.ops.aten.permute.default(view_427, [1, 0])
        mm_71: "f32[768, 768]" = torch.ops.aten.mm.default(permute_328, view_132);  permute_328 = None
        permute_329: "f32[768, 768]" = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
        sum_102: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_427, [0], True);  view_427 = None
        view_428: "f32[768]" = torch.ops.aten.reshape.default(sum_102, [768]);  sum_102 = None
        permute_330: "f32[768, 768]" = torch.ops.aten.permute.default(permute_329, [1, 0]);  permute_329 = None
        view_429: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_70, [3, 5, 768]);  mm_70 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        add_135: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(add_134, view_429);  add_134 = view_429 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        view_430: "f32[15, 768]" = torch.ops.aten.reshape.default(view_421, [15, 768]);  view_421 = None
        mm_72: "f32[15, 768]" = torch.ops.aten.mm.default(view_430, permute_331);  permute_331 = None
        permute_332: "f32[768, 15]" = torch.ops.aten.permute.default(view_430, [1, 0])
        mm_73: "f32[768, 768]" = torch.ops.aten.mm.default(permute_332, view_132);  permute_332 = view_132 = None
        permute_333: "f32[768, 768]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
        sum_103: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_430, [0], True);  view_430 = None
        view_431: "f32[768]" = torch.ops.aten.reshape.default(sum_103, [768]);  sum_103 = None
        permute_334: "f32[768, 768]" = torch.ops.aten.permute.default(permute_333, [1, 0]);  permute_333 = None
        view_432: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_72, [3, 5, 768]);  mm_72 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        add_136: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(add_135, view_432);  add_135 = view_432 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_338: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_136, primals_100);  primals_100 = None
        mul_339: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_338, 768)
        sum_104: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_338, [2], True)
        mul_340: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_338, mul_81);  mul_338 = None
        sum_105: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_340, [2], True);  mul_340 = None
        mul_341: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_81, sum_105);  sum_105 = None
        sub_82: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(mul_339, sum_104);  mul_339 = sum_104 = None
        sub_83: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(sub_82, mul_341);  sub_82 = mul_341 = None
        mul_342: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(div_42, sub_83);  div_42 = sub_83 = None
        mul_343: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_136, mul_81);  mul_81 = None
        sum_106: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_343, [0, 1]);  mul_343 = None
        sum_107: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_136, [0, 1]);  add_136 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:457, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_19: "f32[3, 5, 768]" = torch.ops.prims.convert_element_type.default(gt_18, torch.float32);  gt_18 = None
        mul_344: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_19, 1.1111111111111112);  convert_element_type_19 = None
        mul_345: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_342, mul_344);  mul_344 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:456, code: hidden_states = self.dense(hidden_states)
        view_433: "f32[15, 768]" = torch.ops.aten.reshape.default(mul_345, [15, 768]);  mul_345 = None
        mm_74: "f32[15, 3072]" = torch.ops.aten.mm.default(view_433, permute_335);  permute_335 = None
        permute_336: "f32[768, 15]" = torch.ops.aten.permute.default(view_433, [1, 0])
        mm_75: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_336, view_130);  permute_336 = view_130 = None
        permute_337: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
        sum_108: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_433, [0], True);  view_433 = None
        view_434: "f32[768]" = torch.ops.aten.reshape.default(sum_108, [768]);  sum_108 = None
        permute_338: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_337, [1, 0]);  permute_337 = None
        view_435: "f32[3, 5, 3072]" = torch.ops.aten.reshape.default(mm_74, [3, 5, 3072]);  mm_74 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_347: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(add_48, 0.5);  add_48 = None
        mul_348: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(view_129, view_129)
        mul_349: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(mul_348, -0.5);  mul_348 = None
        exp_18: "f32[3, 5, 3072]" = torch.ops.aten.exp.default(mul_349);  mul_349 = None
        mul_350: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
        mul_351: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(view_129, mul_350);  view_129 = mul_350 = None
        add_138: "f32[3, 5, 3072]" = torch.ops.aten.add.Tensor(mul_347, mul_351);  mul_347 = mul_351 = None
        mul_352: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(view_435, add_138);  view_435 = add_138 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        view_436: "f32[15, 3072]" = torch.ops.aten.reshape.default(mul_352, [15, 3072]);  mul_352 = None
        mm_76: "f32[15, 768]" = torch.ops.aten.mm.default(view_436, permute_339);  permute_339 = None
        permute_340: "f32[3072, 15]" = torch.ops.aten.permute.default(view_436, [1, 0])
        mm_77: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_340, view_128);  permute_340 = view_128 = None
        permute_341: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
        sum_109: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_436, [0], True);  view_436 = None
        view_437: "f32[3072]" = torch.ops.aten.reshape.default(sum_109, [3072]);  sum_109 = None
        permute_342: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_341, [1, 0]);  permute_341 = None
        view_438: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_76, [3, 5, 768]);  mm_76 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        add_139: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(mul_342, view_438);  mul_342 = view_438 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_354: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_139, primals_94);  primals_94 = None
        mul_355: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_354, 768)
        sum_110: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_354, [2], True)
        mul_356: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_354, mul_74);  mul_354 = None
        sum_111: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_356, [2], True);  mul_356 = None
        mul_357: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_74, sum_111);  sum_111 = None
        sub_85: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(mul_355, sum_110);  mul_355 = sum_110 = None
        sub_86: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(sub_85, mul_357);  sub_85 = mul_357 = None
        mul_358: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(div_43, sub_86);  div_43 = sub_86 = None
        mul_359: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_139, mul_74);  mul_74 = None
        sum_112: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_359, [0, 1]);  mul_359 = None
        sum_113: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_139, [0, 1]);  add_139 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:379, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_20: "f32[3, 5, 768]" = torch.ops.prims.convert_element_type.default(gt_17, torch.float32);  gt_17 = None
        mul_360: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_20, 1.1111111111111112);  convert_element_type_20 = None
        mul_361: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_358, mul_360);  mul_360 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:378, code: hidden_states = self.dense(hidden_states)
        view_439: "f32[15, 768]" = torch.ops.aten.reshape.default(mul_361, [15, 768]);  mul_361 = None
        mm_78: "f32[15, 768]" = torch.ops.aten.mm.default(view_439, permute_343);  permute_343 = None
        permute_344: "f32[768, 15]" = torch.ops.aten.permute.default(view_439, [1, 0])
        mm_79: "f32[768, 768]" = torch.ops.aten.mm.default(permute_344, view_126);  permute_344 = view_126 = None
        permute_345: "f32[768, 768]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
        sum_114: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_439, [0], True);  view_439 = None
        view_440: "f32[768]" = torch.ops.aten.reshape.default(sum_114, [768]);  sum_114 = None
        permute_346: "f32[768, 768]" = torch.ops.aten.permute.default(permute_345, [1, 0]);  permute_345 = None
        view_441: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_78, [3, 5, 768]);  mm_78 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:361, code: context_layer = context_layer.view(new_context_layer_shape)
        view_442: "f32[3, 5, 12, 64]" = torch.ops.aten.reshape.default(view_441, [3, 5, 12, 64]);  view_441 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_347: "f32[3, 12, 5, 64]" = torch.ops.aten.permute.default(view_442, [0, 2, 1, 3]);  view_442 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:357, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_92: "f32[3, 12, 5, 64]" = torch.ops.aten.clone.default(permute_347, memory_format = torch.contiguous_format);  permute_347 = None
        view_443: "f32[36, 5, 64]" = torch.ops.aten.reshape.default(clone_92, [36, 5, 64]);  clone_92 = None
        bmm_48: "f32[36, 5, 64]" = torch.ops.aten.bmm.default(permute_348, view_443);  permute_348 = None
        bmm_49: "f32[36, 5, 5]" = torch.ops.aten.bmm.default(view_443, permute_349);  view_443 = permute_349 = None
        view_444: "f32[3, 12, 5, 64]" = torch.ops.aten.reshape.default(bmm_48, [3, 12, 5, 64]);  bmm_48 = None
        view_445: "f32[3, 12, 5, 5]" = torch.ops.aten.reshape.default(bmm_49, [3, 12, 5, 5]);  bmm_49 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:351, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_21: "f32[3, 12, 5, 5]" = torch.ops.prims.convert_element_type.default(gt_16, torch.float32);  gt_16 = None
        mul_362: "f32[3, 12, 5, 5]" = torch.ops.aten.mul.Tensor(convert_element_type_21, 1.1111111111111112);  convert_element_type_21 = None
        mul_363: "f32[3, 12, 5, 5]" = torch.ops.aten.mul.Tensor(view_445, mul_362);  view_445 = mul_362 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:347, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_364: "f32[3, 12, 5, 5]" = torch.ops.aten.mul.Tensor(mul_363, alias_20);  mul_363 = None
        sum_115: "f32[3, 12, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_364, [-1], True)
        mul_365: "f32[3, 12, 5, 5]" = torch.ops.aten.mul.Tensor(alias_20, sum_115);  alias_20 = sum_115 = None
        sub_87: "f32[3, 12, 5, 5]" = torch.ops.aten.sub.Tensor(mul_364, mul_365);  mul_364 = mul_365 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:341, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_44: "f32[3, 12, 5, 5]" = torch.ops.aten.div.Tensor(sub_87, 8.0);  sub_87 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:323, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        view_446: "f32[36, 5, 5]" = torch.ops.aten.reshape.default(div_44, [36, 5, 5]);  div_44 = None
        bmm_50: "f32[36, 64, 5]" = torch.ops.aten.bmm.default(permute_350, view_446);  permute_350 = None
        bmm_51: "f32[36, 5, 64]" = torch.ops.aten.bmm.default(view_446, permute_351);  view_446 = permute_351 = None
        view_447: "f32[3, 12, 64, 5]" = torch.ops.aten.reshape.default(bmm_50, [3, 12, 64, 5]);  bmm_50 = None
        view_448: "f32[3, 12, 5, 64]" = torch.ops.aten.reshape.default(bmm_51, [3, 12, 5, 64]);  bmm_51 = None
        permute_352: "f32[3, 12, 5, 64]" = torch.ops.aten.permute.default(view_447, [0, 1, 3, 2]);  view_447 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_353: "f32[3, 5, 12, 64]" = torch.ops.aten.permute.default(view_448, [0, 2, 1, 3]);  view_448 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        clone_94: "f32[3, 5, 12, 64]" = torch.ops.aten.clone.default(permute_353, memory_format = torch.contiguous_format);  permute_353 = None
        view_449: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(clone_94, [3, 5, 768]);  clone_94 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_354: "f32[3, 5, 12, 64]" = torch.ops.aten.permute.default(view_444, [0, 2, 1, 3]);  view_444 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        clone_95: "f32[3, 5, 12, 64]" = torch.ops.aten.clone.default(permute_354, memory_format = torch.contiguous_format);  permute_354 = None
        view_450: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(clone_95, [3, 5, 768]);  clone_95 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_451: "f32[15, 768]" = torch.ops.aten.reshape.default(view_450, [15, 768]);  view_450 = None
        mm_80: "f32[15, 768]" = torch.ops.aten.mm.default(view_451, permute_355);  permute_355 = None
        permute_356: "f32[768, 15]" = torch.ops.aten.permute.default(view_451, [1, 0])
        mm_81: "f32[768, 768]" = torch.ops.aten.mm.default(permute_356, view_110);  permute_356 = None
        permute_357: "f32[768, 768]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
        sum_116: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_451, [0], True);  view_451 = None
        view_452: "f32[768]" = torch.ops.aten.reshape.default(sum_116, [768]);  sum_116 = None
        permute_358: "f32[768, 768]" = torch.ops.aten.permute.default(permute_357, [1, 0]);  permute_357 = None
        view_453: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_80, [3, 5, 768]);  mm_80 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        add_140: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(mul_358, view_453);  mul_358 = view_453 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_359: "f32[3, 5, 12, 64]" = torch.ops.aten.permute.default(permute_352, [0, 2, 1, 3]);  permute_352 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_454: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(permute_359, [3, 5, 768]);  permute_359 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        clone_96: "f32[3, 5, 768]" = torch.ops.aten.clone.default(view_454, memory_format = torch.contiguous_format);  view_454 = None
        view_455: "f32[15, 768]" = torch.ops.aten.reshape.default(clone_96, [15, 768]);  clone_96 = None
        mm_82: "f32[15, 768]" = torch.ops.aten.mm.default(view_455, permute_360);  permute_360 = None
        permute_361: "f32[768, 15]" = torch.ops.aten.permute.default(view_455, [1, 0])
        mm_83: "f32[768, 768]" = torch.ops.aten.mm.default(permute_361, view_110);  permute_361 = None
        permute_362: "f32[768, 768]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
        sum_117: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_455, [0], True);  view_455 = None
        view_456: "f32[768]" = torch.ops.aten.reshape.default(sum_117, [768]);  sum_117 = None
        permute_363: "f32[768, 768]" = torch.ops.aten.permute.default(permute_362, [1, 0]);  permute_362 = None
        view_457: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_82, [3, 5, 768]);  mm_82 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        add_141: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(add_140, view_457);  add_140 = view_457 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        view_458: "f32[15, 768]" = torch.ops.aten.reshape.default(view_449, [15, 768]);  view_449 = None
        mm_84: "f32[15, 768]" = torch.ops.aten.mm.default(view_458, permute_364);  permute_364 = None
        permute_365: "f32[768, 15]" = torch.ops.aten.permute.default(view_458, [1, 0])
        mm_85: "f32[768, 768]" = torch.ops.aten.mm.default(permute_365, view_110);  permute_365 = view_110 = None
        permute_366: "f32[768, 768]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
        sum_118: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_458, [0], True);  view_458 = None
        view_459: "f32[768]" = torch.ops.aten.reshape.default(sum_118, [768]);  sum_118 = None
        permute_367: "f32[768, 768]" = torch.ops.aten.permute.default(permute_366, [1, 0]);  permute_366 = None
        view_460: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_84, [3, 5, 768]);  mm_84 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        add_142: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(add_141, view_460);  add_141 = view_460 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_367: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_142, primals_84);  primals_84 = None
        mul_368: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_367, 768)
        sum_119: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_367, [2], True)
        mul_369: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_367, mul_68);  mul_367 = None
        sum_120: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_369, [2], True);  mul_369 = None
        mul_370: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_68, sum_120);  sum_120 = None
        sub_89: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(mul_368, sum_119);  mul_368 = sum_119 = None
        sub_90: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(sub_89, mul_370);  sub_89 = mul_370 = None
        mul_371: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(div_45, sub_90);  div_45 = sub_90 = None
        mul_372: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_142, mul_68);  mul_68 = None
        sum_121: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_372, [0, 1]);  mul_372 = None
        sum_122: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_142, [0, 1]);  add_142 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:457, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_22: "f32[3, 5, 768]" = torch.ops.prims.convert_element_type.default(gt_15, torch.float32);  gt_15 = None
        mul_373: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 1.1111111111111112);  convert_element_type_22 = None
        mul_374: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_371, mul_373);  mul_373 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:456, code: hidden_states = self.dense(hidden_states)
        view_461: "f32[15, 768]" = torch.ops.aten.reshape.default(mul_374, [15, 768]);  mul_374 = None
        mm_86: "f32[15, 3072]" = torch.ops.aten.mm.default(view_461, permute_368);  permute_368 = None
        permute_369: "f32[768, 15]" = torch.ops.aten.permute.default(view_461, [1, 0])
        mm_87: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_369, view_108);  permute_369 = view_108 = None
        permute_370: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
        sum_123: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_461, [0], True);  view_461 = None
        view_462: "f32[768]" = torch.ops.aten.reshape.default(sum_123, [768]);  sum_123 = None
        permute_371: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_370, [1, 0]);  permute_370 = None
        view_463: "f32[3, 5, 3072]" = torch.ops.aten.reshape.default(mm_86, [3, 5, 3072]);  mm_86 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_376: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(add_40, 0.5);  add_40 = None
        mul_377: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(view_107, view_107)
        mul_378: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(mul_377, -0.5);  mul_377 = None
        exp_19: "f32[3, 5, 3072]" = torch.ops.aten.exp.default(mul_378);  mul_378 = None
        mul_379: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(exp_19, 0.3989422804014327);  exp_19 = None
        mul_380: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(view_107, mul_379);  view_107 = mul_379 = None
        add_144: "f32[3, 5, 3072]" = torch.ops.aten.add.Tensor(mul_376, mul_380);  mul_376 = mul_380 = None
        mul_381: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(view_463, add_144);  view_463 = add_144 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        view_464: "f32[15, 3072]" = torch.ops.aten.reshape.default(mul_381, [15, 3072]);  mul_381 = None
        mm_88: "f32[15, 768]" = torch.ops.aten.mm.default(view_464, permute_372);  permute_372 = None
        permute_373: "f32[3072, 15]" = torch.ops.aten.permute.default(view_464, [1, 0])
        mm_89: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_373, view_106);  permute_373 = view_106 = None
        permute_374: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
        sum_124: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_464, [0], True);  view_464 = None
        view_465: "f32[3072]" = torch.ops.aten.reshape.default(sum_124, [3072]);  sum_124 = None
        permute_375: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_374, [1, 0]);  permute_374 = None
        view_466: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_88, [3, 5, 768]);  mm_88 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        add_145: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(mul_371, view_466);  mul_371 = view_466 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_383: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_145, primals_78);  primals_78 = None
        mul_384: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_383, 768)
        sum_125: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_383, [2], True)
        mul_385: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_383, mul_61);  mul_383 = None
        sum_126: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_385, [2], True);  mul_385 = None
        mul_386: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_61, sum_126);  sum_126 = None
        sub_92: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(mul_384, sum_125);  mul_384 = sum_125 = None
        sub_93: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(sub_92, mul_386);  sub_92 = mul_386 = None
        mul_387: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(div_46, sub_93);  div_46 = sub_93 = None
        mul_388: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_145, mul_61);  mul_61 = None
        sum_127: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_388, [0, 1]);  mul_388 = None
        sum_128: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_145, [0, 1]);  add_145 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:379, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_23: "f32[3, 5, 768]" = torch.ops.prims.convert_element_type.default(gt_14, torch.float32);  gt_14 = None
        mul_389: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_23, 1.1111111111111112);  convert_element_type_23 = None
        mul_390: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_387, mul_389);  mul_389 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:378, code: hidden_states = self.dense(hidden_states)
        view_467: "f32[15, 768]" = torch.ops.aten.reshape.default(mul_390, [15, 768]);  mul_390 = None
        mm_90: "f32[15, 768]" = torch.ops.aten.mm.default(view_467, permute_376);  permute_376 = None
        permute_377: "f32[768, 15]" = torch.ops.aten.permute.default(view_467, [1, 0])
        mm_91: "f32[768, 768]" = torch.ops.aten.mm.default(permute_377, view_104);  permute_377 = view_104 = None
        permute_378: "f32[768, 768]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
        sum_129: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_467, [0], True);  view_467 = None
        view_468: "f32[768]" = torch.ops.aten.reshape.default(sum_129, [768]);  sum_129 = None
        permute_379: "f32[768, 768]" = torch.ops.aten.permute.default(permute_378, [1, 0]);  permute_378 = None
        view_469: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_90, [3, 5, 768]);  mm_90 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:361, code: context_layer = context_layer.view(new_context_layer_shape)
        view_470: "f32[3, 5, 12, 64]" = torch.ops.aten.reshape.default(view_469, [3, 5, 12, 64]);  view_469 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_380: "f32[3, 12, 5, 64]" = torch.ops.aten.permute.default(view_470, [0, 2, 1, 3]);  view_470 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:357, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_99: "f32[3, 12, 5, 64]" = torch.ops.aten.clone.default(permute_380, memory_format = torch.contiguous_format);  permute_380 = None
        view_471: "f32[36, 5, 64]" = torch.ops.aten.reshape.default(clone_99, [36, 5, 64]);  clone_99 = None
        bmm_52: "f32[36, 5, 64]" = torch.ops.aten.bmm.default(permute_381, view_471);  permute_381 = None
        bmm_53: "f32[36, 5, 5]" = torch.ops.aten.bmm.default(view_471, permute_382);  view_471 = permute_382 = None
        view_472: "f32[3, 12, 5, 64]" = torch.ops.aten.reshape.default(bmm_52, [3, 12, 5, 64]);  bmm_52 = None
        view_473: "f32[3, 12, 5, 5]" = torch.ops.aten.reshape.default(bmm_53, [3, 12, 5, 5]);  bmm_53 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:351, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_24: "f32[3, 12, 5, 5]" = torch.ops.prims.convert_element_type.default(gt_13, torch.float32);  gt_13 = None
        mul_391: "f32[3, 12, 5, 5]" = torch.ops.aten.mul.Tensor(convert_element_type_24, 1.1111111111111112);  convert_element_type_24 = None
        mul_392: "f32[3, 12, 5, 5]" = torch.ops.aten.mul.Tensor(view_473, mul_391);  view_473 = mul_391 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:347, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_393: "f32[3, 12, 5, 5]" = torch.ops.aten.mul.Tensor(mul_392, alias_21);  mul_392 = None
        sum_130: "f32[3, 12, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_393, [-1], True)
        mul_394: "f32[3, 12, 5, 5]" = torch.ops.aten.mul.Tensor(alias_21, sum_130);  alias_21 = sum_130 = None
        sub_94: "f32[3, 12, 5, 5]" = torch.ops.aten.sub.Tensor(mul_393, mul_394);  mul_393 = mul_394 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:341, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_47: "f32[3, 12, 5, 5]" = torch.ops.aten.div.Tensor(sub_94, 8.0);  sub_94 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:323, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        view_474: "f32[36, 5, 5]" = torch.ops.aten.reshape.default(div_47, [36, 5, 5]);  div_47 = None
        bmm_54: "f32[36, 64, 5]" = torch.ops.aten.bmm.default(permute_383, view_474);  permute_383 = None
        bmm_55: "f32[36, 5, 64]" = torch.ops.aten.bmm.default(view_474, permute_384);  view_474 = permute_384 = None
        view_475: "f32[3, 12, 64, 5]" = torch.ops.aten.reshape.default(bmm_54, [3, 12, 64, 5]);  bmm_54 = None
        view_476: "f32[3, 12, 5, 64]" = torch.ops.aten.reshape.default(bmm_55, [3, 12, 5, 64]);  bmm_55 = None
        permute_385: "f32[3, 12, 5, 64]" = torch.ops.aten.permute.default(view_475, [0, 1, 3, 2]);  view_475 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_386: "f32[3, 5, 12, 64]" = torch.ops.aten.permute.default(view_476, [0, 2, 1, 3]);  view_476 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        clone_101: "f32[3, 5, 12, 64]" = torch.ops.aten.clone.default(permute_386, memory_format = torch.contiguous_format);  permute_386 = None
        view_477: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(clone_101, [3, 5, 768]);  clone_101 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_387: "f32[3, 5, 12, 64]" = torch.ops.aten.permute.default(view_472, [0, 2, 1, 3]);  view_472 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        clone_102: "f32[3, 5, 12, 64]" = torch.ops.aten.clone.default(permute_387, memory_format = torch.contiguous_format);  permute_387 = None
        view_478: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(clone_102, [3, 5, 768]);  clone_102 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_479: "f32[15, 768]" = torch.ops.aten.reshape.default(view_478, [15, 768]);  view_478 = None
        mm_92: "f32[15, 768]" = torch.ops.aten.mm.default(view_479, permute_388);  permute_388 = None
        permute_389: "f32[768, 15]" = torch.ops.aten.permute.default(view_479, [1, 0])
        mm_93: "f32[768, 768]" = torch.ops.aten.mm.default(permute_389, view_88);  permute_389 = None
        permute_390: "f32[768, 768]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
        sum_131: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_479, [0], True);  view_479 = None
        view_480: "f32[768]" = torch.ops.aten.reshape.default(sum_131, [768]);  sum_131 = None
        permute_391: "f32[768, 768]" = torch.ops.aten.permute.default(permute_390, [1, 0]);  permute_390 = None
        view_481: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_92, [3, 5, 768]);  mm_92 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        add_146: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(mul_387, view_481);  mul_387 = view_481 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_392: "f32[3, 5, 12, 64]" = torch.ops.aten.permute.default(permute_385, [0, 2, 1, 3]);  permute_385 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_482: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(permute_392, [3, 5, 768]);  permute_392 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        clone_103: "f32[3, 5, 768]" = torch.ops.aten.clone.default(view_482, memory_format = torch.contiguous_format);  view_482 = None
        view_483: "f32[15, 768]" = torch.ops.aten.reshape.default(clone_103, [15, 768]);  clone_103 = None
        mm_94: "f32[15, 768]" = torch.ops.aten.mm.default(view_483, permute_393);  permute_393 = None
        permute_394: "f32[768, 15]" = torch.ops.aten.permute.default(view_483, [1, 0])
        mm_95: "f32[768, 768]" = torch.ops.aten.mm.default(permute_394, view_88);  permute_394 = None
        permute_395: "f32[768, 768]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
        sum_132: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_483, [0], True);  view_483 = None
        view_484: "f32[768]" = torch.ops.aten.reshape.default(sum_132, [768]);  sum_132 = None
        permute_396: "f32[768, 768]" = torch.ops.aten.permute.default(permute_395, [1, 0]);  permute_395 = None
        view_485: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_94, [3, 5, 768]);  mm_94 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        add_147: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(add_146, view_485);  add_146 = view_485 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        view_486: "f32[15, 768]" = torch.ops.aten.reshape.default(view_477, [15, 768]);  view_477 = None
        mm_96: "f32[15, 768]" = torch.ops.aten.mm.default(view_486, permute_397);  permute_397 = None
        permute_398: "f32[768, 15]" = torch.ops.aten.permute.default(view_486, [1, 0])
        mm_97: "f32[768, 768]" = torch.ops.aten.mm.default(permute_398, view_88);  permute_398 = view_88 = None
        permute_399: "f32[768, 768]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
        sum_133: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_486, [0], True);  view_486 = None
        view_487: "f32[768]" = torch.ops.aten.reshape.default(sum_133, [768]);  sum_133 = None
        permute_400: "f32[768, 768]" = torch.ops.aten.permute.default(permute_399, [1, 0]);  permute_399 = None
        view_488: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_96, [3, 5, 768]);  mm_96 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        add_148: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(add_147, view_488);  add_147 = view_488 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_396: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_148, primals_68);  primals_68 = None
        mul_397: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_396, 768)
        sum_134: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_396, [2], True)
        mul_398: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_396, mul_55);  mul_396 = None
        sum_135: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_398, [2], True);  mul_398 = None
        mul_399: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_55, sum_135);  sum_135 = None
        sub_96: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(mul_397, sum_134);  mul_397 = sum_134 = None
        sub_97: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(sub_96, mul_399);  sub_96 = mul_399 = None
        mul_400: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(div_48, sub_97);  div_48 = sub_97 = None
        mul_401: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_148, mul_55);  mul_55 = None
        sum_136: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_401, [0, 1]);  mul_401 = None
        sum_137: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_148, [0, 1]);  add_148 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:457, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_25: "f32[3, 5, 768]" = torch.ops.prims.convert_element_type.default(gt_12, torch.float32);  gt_12 = None
        mul_402: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_25, 1.1111111111111112);  convert_element_type_25 = None
        mul_403: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_400, mul_402);  mul_402 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:456, code: hidden_states = self.dense(hidden_states)
        view_489: "f32[15, 768]" = torch.ops.aten.reshape.default(mul_403, [15, 768]);  mul_403 = None
        mm_98: "f32[15, 3072]" = torch.ops.aten.mm.default(view_489, permute_401);  permute_401 = None
        permute_402: "f32[768, 15]" = torch.ops.aten.permute.default(view_489, [1, 0])
        mm_99: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_402, view_86);  permute_402 = view_86 = None
        permute_403: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
        sum_138: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_489, [0], True);  view_489 = None
        view_490: "f32[768]" = torch.ops.aten.reshape.default(sum_138, [768]);  sum_138 = None
        permute_404: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_403, [1, 0]);  permute_403 = None
        view_491: "f32[3, 5, 3072]" = torch.ops.aten.reshape.default(mm_98, [3, 5, 3072]);  mm_98 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_405: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(add_32, 0.5);  add_32 = None
        mul_406: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(view_85, view_85)
        mul_407: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(mul_406, -0.5);  mul_406 = None
        exp_20: "f32[3, 5, 3072]" = torch.ops.aten.exp.default(mul_407);  mul_407 = None
        mul_408: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
        mul_409: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(view_85, mul_408);  view_85 = mul_408 = None
        add_150: "f32[3, 5, 3072]" = torch.ops.aten.add.Tensor(mul_405, mul_409);  mul_405 = mul_409 = None
        mul_410: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(view_491, add_150);  view_491 = add_150 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        view_492: "f32[15, 3072]" = torch.ops.aten.reshape.default(mul_410, [15, 3072]);  mul_410 = None
        mm_100: "f32[15, 768]" = torch.ops.aten.mm.default(view_492, permute_405);  permute_405 = None
        permute_406: "f32[3072, 15]" = torch.ops.aten.permute.default(view_492, [1, 0])
        mm_101: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_406, view_84);  permute_406 = view_84 = None
        permute_407: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_101, [1, 0]);  mm_101 = None
        sum_139: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_492, [0], True);  view_492 = None
        view_493: "f32[3072]" = torch.ops.aten.reshape.default(sum_139, [3072]);  sum_139 = None
        permute_408: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_407, [1, 0]);  permute_407 = None
        view_494: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_100, [3, 5, 768]);  mm_100 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        add_151: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(mul_400, view_494);  mul_400 = view_494 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_412: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_151, primals_62);  primals_62 = None
        mul_413: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_412, 768)
        sum_140: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_412, [2], True)
        mul_414: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_412, mul_48);  mul_412 = None
        sum_141: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_414, [2], True);  mul_414 = None
        mul_415: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_48, sum_141);  sum_141 = None
        sub_99: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(mul_413, sum_140);  mul_413 = sum_140 = None
        sub_100: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(sub_99, mul_415);  sub_99 = mul_415 = None
        mul_416: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(div_49, sub_100);  div_49 = sub_100 = None
        mul_417: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_151, mul_48);  mul_48 = None
        sum_142: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_417, [0, 1]);  mul_417 = None
        sum_143: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_151, [0, 1]);  add_151 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:379, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_26: "f32[3, 5, 768]" = torch.ops.prims.convert_element_type.default(gt_11, torch.float32);  gt_11 = None
        mul_418: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_26, 1.1111111111111112);  convert_element_type_26 = None
        mul_419: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_416, mul_418);  mul_418 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:378, code: hidden_states = self.dense(hidden_states)
        view_495: "f32[15, 768]" = torch.ops.aten.reshape.default(mul_419, [15, 768]);  mul_419 = None
        mm_102: "f32[15, 768]" = torch.ops.aten.mm.default(view_495, permute_409);  permute_409 = None
        permute_410: "f32[768, 15]" = torch.ops.aten.permute.default(view_495, [1, 0])
        mm_103: "f32[768, 768]" = torch.ops.aten.mm.default(permute_410, view_82);  permute_410 = view_82 = None
        permute_411: "f32[768, 768]" = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
        sum_144: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_495, [0], True);  view_495 = None
        view_496: "f32[768]" = torch.ops.aten.reshape.default(sum_144, [768]);  sum_144 = None
        permute_412: "f32[768, 768]" = torch.ops.aten.permute.default(permute_411, [1, 0]);  permute_411 = None
        view_497: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_102, [3, 5, 768]);  mm_102 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:361, code: context_layer = context_layer.view(new_context_layer_shape)
        view_498: "f32[3, 5, 12, 64]" = torch.ops.aten.reshape.default(view_497, [3, 5, 12, 64]);  view_497 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_413: "f32[3, 12, 5, 64]" = torch.ops.aten.permute.default(view_498, [0, 2, 1, 3]);  view_498 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:357, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_106: "f32[3, 12, 5, 64]" = torch.ops.aten.clone.default(permute_413, memory_format = torch.contiguous_format);  permute_413 = None
        view_499: "f32[36, 5, 64]" = torch.ops.aten.reshape.default(clone_106, [36, 5, 64]);  clone_106 = None
        bmm_56: "f32[36, 5, 64]" = torch.ops.aten.bmm.default(permute_414, view_499);  permute_414 = None
        bmm_57: "f32[36, 5, 5]" = torch.ops.aten.bmm.default(view_499, permute_415);  view_499 = permute_415 = None
        view_500: "f32[3, 12, 5, 64]" = torch.ops.aten.reshape.default(bmm_56, [3, 12, 5, 64]);  bmm_56 = None
        view_501: "f32[3, 12, 5, 5]" = torch.ops.aten.reshape.default(bmm_57, [3, 12, 5, 5]);  bmm_57 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:351, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_27: "f32[3, 12, 5, 5]" = torch.ops.prims.convert_element_type.default(gt_10, torch.float32);  gt_10 = None
        mul_420: "f32[3, 12, 5, 5]" = torch.ops.aten.mul.Tensor(convert_element_type_27, 1.1111111111111112);  convert_element_type_27 = None
        mul_421: "f32[3, 12, 5, 5]" = torch.ops.aten.mul.Tensor(view_501, mul_420);  view_501 = mul_420 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:347, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_422: "f32[3, 12, 5, 5]" = torch.ops.aten.mul.Tensor(mul_421, alias_22);  mul_421 = None
        sum_145: "f32[3, 12, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_422, [-1], True)
        mul_423: "f32[3, 12, 5, 5]" = torch.ops.aten.mul.Tensor(alias_22, sum_145);  alias_22 = sum_145 = None
        sub_101: "f32[3, 12, 5, 5]" = torch.ops.aten.sub.Tensor(mul_422, mul_423);  mul_422 = mul_423 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:341, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_50: "f32[3, 12, 5, 5]" = torch.ops.aten.div.Tensor(sub_101, 8.0);  sub_101 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:323, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        view_502: "f32[36, 5, 5]" = torch.ops.aten.reshape.default(div_50, [36, 5, 5]);  div_50 = None
        bmm_58: "f32[36, 64, 5]" = torch.ops.aten.bmm.default(permute_416, view_502);  permute_416 = None
        bmm_59: "f32[36, 5, 64]" = torch.ops.aten.bmm.default(view_502, permute_417);  view_502 = permute_417 = None
        view_503: "f32[3, 12, 64, 5]" = torch.ops.aten.reshape.default(bmm_58, [3, 12, 64, 5]);  bmm_58 = None
        view_504: "f32[3, 12, 5, 64]" = torch.ops.aten.reshape.default(bmm_59, [3, 12, 5, 64]);  bmm_59 = None
        permute_418: "f32[3, 12, 5, 64]" = torch.ops.aten.permute.default(view_503, [0, 1, 3, 2]);  view_503 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_419: "f32[3, 5, 12, 64]" = torch.ops.aten.permute.default(view_504, [0, 2, 1, 3]);  view_504 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        clone_108: "f32[3, 5, 12, 64]" = torch.ops.aten.clone.default(permute_419, memory_format = torch.contiguous_format);  permute_419 = None
        view_505: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(clone_108, [3, 5, 768]);  clone_108 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_420: "f32[3, 5, 12, 64]" = torch.ops.aten.permute.default(view_500, [0, 2, 1, 3]);  view_500 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        clone_109: "f32[3, 5, 12, 64]" = torch.ops.aten.clone.default(permute_420, memory_format = torch.contiguous_format);  permute_420 = None
        view_506: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(clone_109, [3, 5, 768]);  clone_109 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_507: "f32[15, 768]" = torch.ops.aten.reshape.default(view_506, [15, 768]);  view_506 = None
        mm_104: "f32[15, 768]" = torch.ops.aten.mm.default(view_507, permute_421);  permute_421 = None
        permute_422: "f32[768, 15]" = torch.ops.aten.permute.default(view_507, [1, 0])
        mm_105: "f32[768, 768]" = torch.ops.aten.mm.default(permute_422, view_66);  permute_422 = None
        permute_423: "f32[768, 768]" = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
        sum_146: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_507, [0], True);  view_507 = None
        view_508: "f32[768]" = torch.ops.aten.reshape.default(sum_146, [768]);  sum_146 = None
        permute_424: "f32[768, 768]" = torch.ops.aten.permute.default(permute_423, [1, 0]);  permute_423 = None
        view_509: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_104, [3, 5, 768]);  mm_104 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        add_152: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(mul_416, view_509);  mul_416 = view_509 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_425: "f32[3, 5, 12, 64]" = torch.ops.aten.permute.default(permute_418, [0, 2, 1, 3]);  permute_418 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_510: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(permute_425, [3, 5, 768]);  permute_425 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        clone_110: "f32[3, 5, 768]" = torch.ops.aten.clone.default(view_510, memory_format = torch.contiguous_format);  view_510 = None
        view_511: "f32[15, 768]" = torch.ops.aten.reshape.default(clone_110, [15, 768]);  clone_110 = None
        mm_106: "f32[15, 768]" = torch.ops.aten.mm.default(view_511, permute_426);  permute_426 = None
        permute_427: "f32[768, 15]" = torch.ops.aten.permute.default(view_511, [1, 0])
        mm_107: "f32[768, 768]" = torch.ops.aten.mm.default(permute_427, view_66);  permute_427 = None
        permute_428: "f32[768, 768]" = torch.ops.aten.permute.default(mm_107, [1, 0]);  mm_107 = None
        sum_147: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_511, [0], True);  view_511 = None
        view_512: "f32[768]" = torch.ops.aten.reshape.default(sum_147, [768]);  sum_147 = None
        permute_429: "f32[768, 768]" = torch.ops.aten.permute.default(permute_428, [1, 0]);  permute_428 = None
        view_513: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_106, [3, 5, 768]);  mm_106 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        add_153: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(add_152, view_513);  add_152 = view_513 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        view_514: "f32[15, 768]" = torch.ops.aten.reshape.default(view_505, [15, 768]);  view_505 = None
        mm_108: "f32[15, 768]" = torch.ops.aten.mm.default(view_514, permute_430);  permute_430 = None
        permute_431: "f32[768, 15]" = torch.ops.aten.permute.default(view_514, [1, 0])
        mm_109: "f32[768, 768]" = torch.ops.aten.mm.default(permute_431, view_66);  permute_431 = view_66 = None
        permute_432: "f32[768, 768]" = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
        sum_148: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_514, [0], True);  view_514 = None
        view_515: "f32[768]" = torch.ops.aten.reshape.default(sum_148, [768]);  sum_148 = None
        permute_433: "f32[768, 768]" = torch.ops.aten.permute.default(permute_432, [1, 0]);  permute_432 = None
        view_516: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_108, [3, 5, 768]);  mm_108 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        add_154: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(add_153, view_516);  add_153 = view_516 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_425: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_154, primals_52);  primals_52 = None
        mul_426: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_425, 768)
        sum_149: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_425, [2], True)
        mul_427: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_425, mul_42);  mul_425 = None
        sum_150: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_427, [2], True);  mul_427 = None
        mul_428: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_42, sum_150);  sum_150 = None
        sub_103: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(mul_426, sum_149);  mul_426 = sum_149 = None
        sub_104: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(sub_103, mul_428);  sub_103 = mul_428 = None
        mul_429: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(div_51, sub_104);  div_51 = sub_104 = None
        mul_430: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_154, mul_42);  mul_42 = None
        sum_151: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_430, [0, 1]);  mul_430 = None
        sum_152: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_154, [0, 1]);  add_154 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:457, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_28: "f32[3, 5, 768]" = torch.ops.prims.convert_element_type.default(gt_9, torch.float32);  gt_9 = None
        mul_431: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_28, 1.1111111111111112);  convert_element_type_28 = None
        mul_432: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_429, mul_431);  mul_431 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:456, code: hidden_states = self.dense(hidden_states)
        view_517: "f32[15, 768]" = torch.ops.aten.reshape.default(mul_432, [15, 768]);  mul_432 = None
        mm_110: "f32[15, 3072]" = torch.ops.aten.mm.default(view_517, permute_434);  permute_434 = None
        permute_435: "f32[768, 15]" = torch.ops.aten.permute.default(view_517, [1, 0])
        mm_111: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_435, view_64);  permute_435 = view_64 = None
        permute_436: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_111, [1, 0]);  mm_111 = None
        sum_153: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_517, [0], True);  view_517 = None
        view_518: "f32[768]" = torch.ops.aten.reshape.default(sum_153, [768]);  sum_153 = None
        permute_437: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_436, [1, 0]);  permute_436 = None
        view_519: "f32[3, 5, 3072]" = torch.ops.aten.reshape.default(mm_110, [3, 5, 3072]);  mm_110 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_434: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(add_24, 0.5);  add_24 = None
        mul_435: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(view_63, view_63)
        mul_436: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(mul_435, -0.5);  mul_435 = None
        exp_21: "f32[3, 5, 3072]" = torch.ops.aten.exp.default(mul_436);  mul_436 = None
        mul_437: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
        mul_438: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(view_63, mul_437);  view_63 = mul_437 = None
        add_156: "f32[3, 5, 3072]" = torch.ops.aten.add.Tensor(mul_434, mul_438);  mul_434 = mul_438 = None
        mul_439: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(view_519, add_156);  view_519 = add_156 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        view_520: "f32[15, 3072]" = torch.ops.aten.reshape.default(mul_439, [15, 3072]);  mul_439 = None
        mm_112: "f32[15, 768]" = torch.ops.aten.mm.default(view_520, permute_438);  permute_438 = None
        permute_439: "f32[3072, 15]" = torch.ops.aten.permute.default(view_520, [1, 0])
        mm_113: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_439, view_62);  permute_439 = view_62 = None
        permute_440: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_113, [1, 0]);  mm_113 = None
        sum_154: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_520, [0], True);  view_520 = None
        view_521: "f32[3072]" = torch.ops.aten.reshape.default(sum_154, [3072]);  sum_154 = None
        permute_441: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_440, [1, 0]);  permute_440 = None
        view_522: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_112, [3, 5, 768]);  mm_112 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        add_157: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(mul_429, view_522);  mul_429 = view_522 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_441: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_157, primals_46);  primals_46 = None
        mul_442: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_441, 768)
        sum_155: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_441, [2], True)
        mul_443: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_441, mul_35);  mul_441 = None
        sum_156: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_443, [2], True);  mul_443 = None
        mul_444: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_35, sum_156);  sum_156 = None
        sub_106: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(mul_442, sum_155);  mul_442 = sum_155 = None
        sub_107: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(sub_106, mul_444);  sub_106 = mul_444 = None
        mul_445: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(div_52, sub_107);  div_52 = sub_107 = None
        mul_446: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_157, mul_35);  mul_35 = None
        sum_157: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_446, [0, 1]);  mul_446 = None
        sum_158: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_157, [0, 1]);  add_157 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:379, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_29: "f32[3, 5, 768]" = torch.ops.prims.convert_element_type.default(gt_8, torch.float32);  gt_8 = None
        mul_447: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_29, 1.1111111111111112);  convert_element_type_29 = None
        mul_448: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_445, mul_447);  mul_447 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:378, code: hidden_states = self.dense(hidden_states)
        view_523: "f32[15, 768]" = torch.ops.aten.reshape.default(mul_448, [15, 768]);  mul_448 = None
        mm_114: "f32[15, 768]" = torch.ops.aten.mm.default(view_523, permute_442);  permute_442 = None
        permute_443: "f32[768, 15]" = torch.ops.aten.permute.default(view_523, [1, 0])
        mm_115: "f32[768, 768]" = torch.ops.aten.mm.default(permute_443, view_60);  permute_443 = view_60 = None
        permute_444: "f32[768, 768]" = torch.ops.aten.permute.default(mm_115, [1, 0]);  mm_115 = None
        sum_159: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_523, [0], True);  view_523 = None
        view_524: "f32[768]" = torch.ops.aten.reshape.default(sum_159, [768]);  sum_159 = None
        permute_445: "f32[768, 768]" = torch.ops.aten.permute.default(permute_444, [1, 0]);  permute_444 = None
        view_525: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_114, [3, 5, 768]);  mm_114 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:361, code: context_layer = context_layer.view(new_context_layer_shape)
        view_526: "f32[3, 5, 12, 64]" = torch.ops.aten.reshape.default(view_525, [3, 5, 12, 64]);  view_525 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_446: "f32[3, 12, 5, 64]" = torch.ops.aten.permute.default(view_526, [0, 2, 1, 3]);  view_526 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:357, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_113: "f32[3, 12, 5, 64]" = torch.ops.aten.clone.default(permute_446, memory_format = torch.contiguous_format);  permute_446 = None
        view_527: "f32[36, 5, 64]" = torch.ops.aten.reshape.default(clone_113, [36, 5, 64]);  clone_113 = None
        bmm_60: "f32[36, 5, 64]" = torch.ops.aten.bmm.default(permute_447, view_527);  permute_447 = None
        bmm_61: "f32[36, 5, 5]" = torch.ops.aten.bmm.default(view_527, permute_448);  view_527 = permute_448 = None
        view_528: "f32[3, 12, 5, 64]" = torch.ops.aten.reshape.default(bmm_60, [3, 12, 5, 64]);  bmm_60 = None
        view_529: "f32[3, 12, 5, 5]" = torch.ops.aten.reshape.default(bmm_61, [3, 12, 5, 5]);  bmm_61 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:351, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_30: "f32[3, 12, 5, 5]" = torch.ops.prims.convert_element_type.default(gt_7, torch.float32);  gt_7 = None
        mul_449: "f32[3, 12, 5, 5]" = torch.ops.aten.mul.Tensor(convert_element_type_30, 1.1111111111111112);  convert_element_type_30 = None
        mul_450: "f32[3, 12, 5, 5]" = torch.ops.aten.mul.Tensor(view_529, mul_449);  view_529 = mul_449 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:347, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_451: "f32[3, 12, 5, 5]" = torch.ops.aten.mul.Tensor(mul_450, alias_23);  mul_450 = None
        sum_160: "f32[3, 12, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_451, [-1], True)
        mul_452: "f32[3, 12, 5, 5]" = torch.ops.aten.mul.Tensor(alias_23, sum_160);  alias_23 = sum_160 = None
        sub_108: "f32[3, 12, 5, 5]" = torch.ops.aten.sub.Tensor(mul_451, mul_452);  mul_451 = mul_452 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:341, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_53: "f32[3, 12, 5, 5]" = torch.ops.aten.div.Tensor(sub_108, 8.0);  sub_108 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:323, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        view_530: "f32[36, 5, 5]" = torch.ops.aten.reshape.default(div_53, [36, 5, 5]);  div_53 = None
        bmm_62: "f32[36, 64, 5]" = torch.ops.aten.bmm.default(permute_449, view_530);  permute_449 = None
        bmm_63: "f32[36, 5, 64]" = torch.ops.aten.bmm.default(view_530, permute_450);  view_530 = permute_450 = None
        view_531: "f32[3, 12, 64, 5]" = torch.ops.aten.reshape.default(bmm_62, [3, 12, 64, 5]);  bmm_62 = None
        view_532: "f32[3, 12, 5, 64]" = torch.ops.aten.reshape.default(bmm_63, [3, 12, 5, 64]);  bmm_63 = None
        permute_451: "f32[3, 12, 5, 64]" = torch.ops.aten.permute.default(view_531, [0, 1, 3, 2]);  view_531 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_452: "f32[3, 5, 12, 64]" = torch.ops.aten.permute.default(view_532, [0, 2, 1, 3]);  view_532 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        clone_115: "f32[3, 5, 12, 64]" = torch.ops.aten.clone.default(permute_452, memory_format = torch.contiguous_format);  permute_452 = None
        view_533: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(clone_115, [3, 5, 768]);  clone_115 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_453: "f32[3, 5, 12, 64]" = torch.ops.aten.permute.default(view_528, [0, 2, 1, 3]);  view_528 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        clone_116: "f32[3, 5, 12, 64]" = torch.ops.aten.clone.default(permute_453, memory_format = torch.contiguous_format);  permute_453 = None
        view_534: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(clone_116, [3, 5, 768]);  clone_116 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_535: "f32[15, 768]" = torch.ops.aten.reshape.default(view_534, [15, 768]);  view_534 = None
        mm_116: "f32[15, 768]" = torch.ops.aten.mm.default(view_535, permute_454);  permute_454 = None
        permute_455: "f32[768, 15]" = torch.ops.aten.permute.default(view_535, [1, 0])
        mm_117: "f32[768, 768]" = torch.ops.aten.mm.default(permute_455, view_44);  permute_455 = None
        permute_456: "f32[768, 768]" = torch.ops.aten.permute.default(mm_117, [1, 0]);  mm_117 = None
        sum_161: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_535, [0], True);  view_535 = None
        view_536: "f32[768]" = torch.ops.aten.reshape.default(sum_161, [768]);  sum_161 = None
        permute_457: "f32[768, 768]" = torch.ops.aten.permute.default(permute_456, [1, 0]);  permute_456 = None
        view_537: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_116, [3, 5, 768]);  mm_116 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        add_158: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(mul_445, view_537);  mul_445 = view_537 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_458: "f32[3, 5, 12, 64]" = torch.ops.aten.permute.default(permute_451, [0, 2, 1, 3]);  permute_451 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_538: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(permute_458, [3, 5, 768]);  permute_458 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        clone_117: "f32[3, 5, 768]" = torch.ops.aten.clone.default(view_538, memory_format = torch.contiguous_format);  view_538 = None
        view_539: "f32[15, 768]" = torch.ops.aten.reshape.default(clone_117, [15, 768]);  clone_117 = None
        mm_118: "f32[15, 768]" = torch.ops.aten.mm.default(view_539, permute_459);  permute_459 = None
        permute_460: "f32[768, 15]" = torch.ops.aten.permute.default(view_539, [1, 0])
        mm_119: "f32[768, 768]" = torch.ops.aten.mm.default(permute_460, view_44);  permute_460 = None
        permute_461: "f32[768, 768]" = torch.ops.aten.permute.default(mm_119, [1, 0]);  mm_119 = None
        sum_162: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_539, [0], True);  view_539 = None
        view_540: "f32[768]" = torch.ops.aten.reshape.default(sum_162, [768]);  sum_162 = None
        permute_462: "f32[768, 768]" = torch.ops.aten.permute.default(permute_461, [1, 0]);  permute_461 = None
        view_541: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_118, [3, 5, 768]);  mm_118 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        add_159: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(add_158, view_541);  add_158 = view_541 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        view_542: "f32[15, 768]" = torch.ops.aten.reshape.default(view_533, [15, 768]);  view_533 = None
        mm_120: "f32[15, 768]" = torch.ops.aten.mm.default(view_542, permute_463);  permute_463 = None
        permute_464: "f32[768, 15]" = torch.ops.aten.permute.default(view_542, [1, 0])
        mm_121: "f32[768, 768]" = torch.ops.aten.mm.default(permute_464, view_44);  permute_464 = view_44 = None
        permute_465: "f32[768, 768]" = torch.ops.aten.permute.default(mm_121, [1, 0]);  mm_121 = None
        sum_163: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_542, [0], True);  view_542 = None
        view_543: "f32[768]" = torch.ops.aten.reshape.default(sum_163, [768]);  sum_163 = None
        permute_466: "f32[768, 768]" = torch.ops.aten.permute.default(permute_465, [1, 0]);  permute_465 = None
        view_544: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_120, [3, 5, 768]);  mm_120 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        add_160: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(add_159, view_544);  add_159 = view_544 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_454: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_160, primals_36);  primals_36 = None
        mul_455: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_454, 768)
        sum_164: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_454, [2], True)
        mul_456: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_454, mul_29);  mul_454 = None
        sum_165: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_456, [2], True);  mul_456 = None
        mul_457: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_29, sum_165);  sum_165 = None
        sub_110: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(mul_455, sum_164);  mul_455 = sum_164 = None
        sub_111: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(sub_110, mul_457);  sub_110 = mul_457 = None
        mul_458: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(div_54, sub_111);  div_54 = sub_111 = None
        mul_459: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_160, mul_29);  mul_29 = None
        sum_166: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_459, [0, 1]);  mul_459 = None
        sum_167: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_160, [0, 1]);  add_160 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:457, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_31: "f32[3, 5, 768]" = torch.ops.prims.convert_element_type.default(gt_6, torch.float32);  gt_6 = None
        mul_460: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_31, 1.1111111111111112);  convert_element_type_31 = None
        mul_461: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_458, mul_460);  mul_460 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:456, code: hidden_states = self.dense(hidden_states)
        view_545: "f32[15, 768]" = torch.ops.aten.reshape.default(mul_461, [15, 768]);  mul_461 = None
        mm_122: "f32[15, 3072]" = torch.ops.aten.mm.default(view_545, permute_467);  permute_467 = None
        permute_468: "f32[768, 15]" = torch.ops.aten.permute.default(view_545, [1, 0])
        mm_123: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_468, view_42);  permute_468 = view_42 = None
        permute_469: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_123, [1, 0]);  mm_123 = None
        sum_168: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_545, [0], True);  view_545 = None
        view_546: "f32[768]" = torch.ops.aten.reshape.default(sum_168, [768]);  sum_168 = None
        permute_470: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_469, [1, 0]);  permute_469 = None
        view_547: "f32[3, 5, 3072]" = torch.ops.aten.reshape.default(mm_122, [3, 5, 3072]);  mm_122 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_463: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(add_16, 0.5);  add_16 = None
        mul_464: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(view_41, view_41)
        mul_465: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(mul_464, -0.5);  mul_464 = None
        exp_22: "f32[3, 5, 3072]" = torch.ops.aten.exp.default(mul_465);  mul_465 = None
        mul_466: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
        mul_467: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(view_41, mul_466);  view_41 = mul_466 = None
        add_162: "f32[3, 5, 3072]" = torch.ops.aten.add.Tensor(mul_463, mul_467);  mul_463 = mul_467 = None
        mul_468: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(view_547, add_162);  view_547 = add_162 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        view_548: "f32[15, 3072]" = torch.ops.aten.reshape.default(mul_468, [15, 3072]);  mul_468 = None
        mm_124: "f32[15, 768]" = torch.ops.aten.mm.default(view_548, permute_471);  permute_471 = None
        permute_472: "f32[3072, 15]" = torch.ops.aten.permute.default(view_548, [1, 0])
        mm_125: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_472, view_40);  permute_472 = view_40 = None
        permute_473: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_125, [1, 0]);  mm_125 = None
        sum_169: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_548, [0], True);  view_548 = None
        view_549: "f32[3072]" = torch.ops.aten.reshape.default(sum_169, [3072]);  sum_169 = None
        permute_474: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_473, [1, 0]);  permute_473 = None
        view_550: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_124, [3, 5, 768]);  mm_124 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        add_163: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(mul_458, view_550);  mul_458 = view_550 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_470: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_163, primals_30);  primals_30 = None
        mul_471: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_470, 768)
        sum_170: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_470, [2], True)
        mul_472: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_470, mul_22);  mul_470 = None
        sum_171: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_472, [2], True);  mul_472 = None
        mul_473: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_22, sum_171);  sum_171 = None
        sub_113: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(mul_471, sum_170);  mul_471 = sum_170 = None
        sub_114: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(sub_113, mul_473);  sub_113 = mul_473 = None
        mul_474: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(div_55, sub_114);  div_55 = sub_114 = None
        mul_475: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_163, mul_22);  mul_22 = None
        sum_172: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_475, [0, 1]);  mul_475 = None
        sum_173: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_163, [0, 1]);  add_163 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:379, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_32: "f32[3, 5, 768]" = torch.ops.prims.convert_element_type.default(gt_5, torch.float32);  gt_5 = None
        mul_476: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_32, 1.1111111111111112);  convert_element_type_32 = None
        mul_477: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_474, mul_476);  mul_476 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:378, code: hidden_states = self.dense(hidden_states)
        view_551: "f32[15, 768]" = torch.ops.aten.reshape.default(mul_477, [15, 768]);  mul_477 = None
        mm_126: "f32[15, 768]" = torch.ops.aten.mm.default(view_551, permute_475);  permute_475 = None
        permute_476: "f32[768, 15]" = torch.ops.aten.permute.default(view_551, [1, 0])
        mm_127: "f32[768, 768]" = torch.ops.aten.mm.default(permute_476, view_38);  permute_476 = view_38 = None
        permute_477: "f32[768, 768]" = torch.ops.aten.permute.default(mm_127, [1, 0]);  mm_127 = None
        sum_174: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_551, [0], True);  view_551 = None
        view_552: "f32[768]" = torch.ops.aten.reshape.default(sum_174, [768]);  sum_174 = None
        permute_478: "f32[768, 768]" = torch.ops.aten.permute.default(permute_477, [1, 0]);  permute_477 = None
        view_553: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_126, [3, 5, 768]);  mm_126 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:361, code: context_layer = context_layer.view(new_context_layer_shape)
        view_554: "f32[3, 5, 12, 64]" = torch.ops.aten.reshape.default(view_553, [3, 5, 12, 64]);  view_553 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_479: "f32[3, 12, 5, 64]" = torch.ops.aten.permute.default(view_554, [0, 2, 1, 3]);  view_554 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:357, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_120: "f32[3, 12, 5, 64]" = torch.ops.aten.clone.default(permute_479, memory_format = torch.contiguous_format);  permute_479 = None
        view_555: "f32[36, 5, 64]" = torch.ops.aten.reshape.default(clone_120, [36, 5, 64]);  clone_120 = None
        bmm_64: "f32[36, 5, 64]" = torch.ops.aten.bmm.default(permute_480, view_555);  permute_480 = None
        bmm_65: "f32[36, 5, 5]" = torch.ops.aten.bmm.default(view_555, permute_481);  view_555 = permute_481 = None
        view_556: "f32[3, 12, 5, 64]" = torch.ops.aten.reshape.default(bmm_64, [3, 12, 5, 64]);  bmm_64 = None
        view_557: "f32[3, 12, 5, 5]" = torch.ops.aten.reshape.default(bmm_65, [3, 12, 5, 5]);  bmm_65 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:351, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_33: "f32[3, 12, 5, 5]" = torch.ops.prims.convert_element_type.default(gt_4, torch.float32);  gt_4 = None
        mul_478: "f32[3, 12, 5, 5]" = torch.ops.aten.mul.Tensor(convert_element_type_33, 1.1111111111111112);  convert_element_type_33 = None
        mul_479: "f32[3, 12, 5, 5]" = torch.ops.aten.mul.Tensor(view_557, mul_478);  view_557 = mul_478 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:347, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_480: "f32[3, 12, 5, 5]" = torch.ops.aten.mul.Tensor(mul_479, alias_24);  mul_479 = None
        sum_175: "f32[3, 12, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_480, [-1], True)
        mul_481: "f32[3, 12, 5, 5]" = torch.ops.aten.mul.Tensor(alias_24, sum_175);  alias_24 = sum_175 = None
        sub_115: "f32[3, 12, 5, 5]" = torch.ops.aten.sub.Tensor(mul_480, mul_481);  mul_480 = mul_481 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:341, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_56: "f32[3, 12, 5, 5]" = torch.ops.aten.div.Tensor(sub_115, 8.0);  sub_115 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:323, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        view_558: "f32[36, 5, 5]" = torch.ops.aten.reshape.default(div_56, [36, 5, 5]);  div_56 = None
        bmm_66: "f32[36, 64, 5]" = torch.ops.aten.bmm.default(permute_482, view_558);  permute_482 = None
        bmm_67: "f32[36, 5, 64]" = torch.ops.aten.bmm.default(view_558, permute_483);  view_558 = permute_483 = None
        view_559: "f32[3, 12, 64, 5]" = torch.ops.aten.reshape.default(bmm_66, [3, 12, 64, 5]);  bmm_66 = None
        view_560: "f32[3, 12, 5, 64]" = torch.ops.aten.reshape.default(bmm_67, [3, 12, 5, 64]);  bmm_67 = None
        permute_484: "f32[3, 12, 5, 64]" = torch.ops.aten.permute.default(view_559, [0, 1, 3, 2]);  view_559 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_485: "f32[3, 5, 12, 64]" = torch.ops.aten.permute.default(view_560, [0, 2, 1, 3]);  view_560 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        clone_122: "f32[3, 5, 12, 64]" = torch.ops.aten.clone.default(permute_485, memory_format = torch.contiguous_format);  permute_485 = None
        view_561: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(clone_122, [3, 5, 768]);  clone_122 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_486: "f32[3, 5, 12, 64]" = torch.ops.aten.permute.default(view_556, [0, 2, 1, 3]);  view_556 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        clone_123: "f32[3, 5, 12, 64]" = torch.ops.aten.clone.default(permute_486, memory_format = torch.contiguous_format);  permute_486 = None
        view_562: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(clone_123, [3, 5, 768]);  clone_123 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_563: "f32[15, 768]" = torch.ops.aten.reshape.default(view_562, [15, 768]);  view_562 = None
        mm_128: "f32[15, 768]" = torch.ops.aten.mm.default(view_563, permute_487);  permute_487 = None
        permute_488: "f32[768, 15]" = torch.ops.aten.permute.default(view_563, [1, 0])
        mm_129: "f32[768, 768]" = torch.ops.aten.mm.default(permute_488, view_22);  permute_488 = None
        permute_489: "f32[768, 768]" = torch.ops.aten.permute.default(mm_129, [1, 0]);  mm_129 = None
        sum_176: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_563, [0], True);  view_563 = None
        view_564: "f32[768]" = torch.ops.aten.reshape.default(sum_176, [768]);  sum_176 = None
        permute_490: "f32[768, 768]" = torch.ops.aten.permute.default(permute_489, [1, 0]);  permute_489 = None
        view_565: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_128, [3, 5, 768]);  mm_128 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        add_164: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(mul_474, view_565);  mul_474 = view_565 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_491: "f32[3, 5, 12, 64]" = torch.ops.aten.permute.default(permute_484, [0, 2, 1, 3]);  permute_484 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_566: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(permute_491, [3, 5, 768]);  permute_491 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        clone_124: "f32[3, 5, 768]" = torch.ops.aten.clone.default(view_566, memory_format = torch.contiguous_format);  view_566 = None
        view_567: "f32[15, 768]" = torch.ops.aten.reshape.default(clone_124, [15, 768]);  clone_124 = None
        mm_130: "f32[15, 768]" = torch.ops.aten.mm.default(view_567, permute_492);  permute_492 = None
        permute_493: "f32[768, 15]" = torch.ops.aten.permute.default(view_567, [1, 0])
        mm_131: "f32[768, 768]" = torch.ops.aten.mm.default(permute_493, view_22);  permute_493 = None
        permute_494: "f32[768, 768]" = torch.ops.aten.permute.default(mm_131, [1, 0]);  mm_131 = None
        sum_177: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_567, [0], True);  view_567 = None
        view_568: "f32[768]" = torch.ops.aten.reshape.default(sum_177, [768]);  sum_177 = None
        permute_495: "f32[768, 768]" = torch.ops.aten.permute.default(permute_494, [1, 0]);  permute_494 = None
        view_569: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_130, [3, 5, 768]);  mm_130 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        add_165: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(add_164, view_569);  add_164 = view_569 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        view_570: "f32[15, 768]" = torch.ops.aten.reshape.default(view_561, [15, 768]);  view_561 = None
        mm_132: "f32[15, 768]" = torch.ops.aten.mm.default(view_570, permute_496);  permute_496 = None
        permute_497: "f32[768, 15]" = torch.ops.aten.permute.default(view_570, [1, 0])
        mm_133: "f32[768, 768]" = torch.ops.aten.mm.default(permute_497, view_22);  permute_497 = view_22 = None
        permute_498: "f32[768, 768]" = torch.ops.aten.permute.default(mm_133, [1, 0]);  mm_133 = None
        sum_178: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_570, [0], True);  view_570 = None
        view_571: "f32[768]" = torch.ops.aten.reshape.default(sum_178, [768]);  sum_178 = None
        permute_499: "f32[768, 768]" = torch.ops.aten.permute.default(permute_498, [1, 0]);  permute_498 = None
        view_572: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_132, [3, 5, 768]);  mm_132 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        add_166: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(add_165, view_572);  add_165 = view_572 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_483: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_166, primals_20);  primals_20 = None
        mul_484: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_483, 768)
        sum_179: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_483, [2], True)
        mul_485: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_483, mul_16);  mul_483 = None
        sum_180: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_485, [2], True);  mul_485 = None
        mul_486: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_16, sum_180);  sum_180 = None
        sub_117: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(mul_484, sum_179);  mul_484 = sum_179 = None
        sub_118: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(sub_117, mul_486);  sub_117 = mul_486 = None
        mul_487: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(div_57, sub_118);  div_57 = sub_118 = None
        mul_488: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_166, mul_16);  mul_16 = None
        sum_181: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_488, [0, 1]);  mul_488 = None
        sum_182: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_166, [0, 1]);  add_166 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:457, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_34: "f32[3, 5, 768]" = torch.ops.prims.convert_element_type.default(gt_3, torch.float32);  gt_3 = None
        mul_489: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_34, 1.1111111111111112);  convert_element_type_34 = None
        mul_490: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_487, mul_489);  mul_489 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:456, code: hidden_states = self.dense(hidden_states)
        view_573: "f32[15, 768]" = torch.ops.aten.reshape.default(mul_490, [15, 768]);  mul_490 = None
        mm_134: "f32[15, 3072]" = torch.ops.aten.mm.default(view_573, permute_500);  permute_500 = None
        permute_501: "f32[768, 15]" = torch.ops.aten.permute.default(view_573, [1, 0])
        mm_135: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_501, view_20);  permute_501 = view_20 = None
        permute_502: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_135, [1, 0]);  mm_135 = None
        sum_183: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_573, [0], True);  view_573 = None
        view_574: "f32[768]" = torch.ops.aten.reshape.default(sum_183, [768]);  sum_183 = None
        permute_503: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_502, [1, 0]);  permute_502 = None
        view_575: "f32[3, 5, 3072]" = torch.ops.aten.reshape.default(mm_134, [3, 5, 3072]);  mm_134 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_492: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(add_8, 0.5);  add_8 = None
        mul_493: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(view_19, view_19)
        mul_494: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(mul_493, -0.5);  mul_493 = None
        exp_23: "f32[3, 5, 3072]" = torch.ops.aten.exp.default(mul_494);  mul_494 = None
        mul_495: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
        mul_496: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(view_19, mul_495);  view_19 = mul_495 = None
        add_168: "f32[3, 5, 3072]" = torch.ops.aten.add.Tensor(mul_492, mul_496);  mul_492 = mul_496 = None
        mul_497: "f32[3, 5, 3072]" = torch.ops.aten.mul.Tensor(view_575, add_168);  view_575 = add_168 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        view_576: "f32[15, 3072]" = torch.ops.aten.reshape.default(mul_497, [15, 3072]);  mul_497 = None
        mm_136: "f32[15, 768]" = torch.ops.aten.mm.default(view_576, permute_504);  permute_504 = None
        permute_505: "f32[3072, 15]" = torch.ops.aten.permute.default(view_576, [1, 0])
        mm_137: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_505, view_18);  permute_505 = view_18 = None
        permute_506: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_137, [1, 0]);  mm_137 = None
        sum_184: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_576, [0], True);  view_576 = None
        view_577: "f32[3072]" = torch.ops.aten.reshape.default(sum_184, [3072]);  sum_184 = None
        permute_507: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_506, [1, 0]);  permute_506 = None
        view_578: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_136, [3, 5, 768]);  mm_136 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        add_169: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(mul_487, view_578);  mul_487 = view_578 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_499: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_169, primals_14);  primals_14 = None
        mul_500: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_499, 768)
        sum_185: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_499, [2], True)
        mul_501: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_499, mul_9);  mul_499 = None
        sum_186: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_501, [2], True);  mul_501 = None
        mul_502: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_9, sum_186);  sum_186 = None
        sub_120: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(mul_500, sum_185);  mul_500 = sum_185 = None
        sub_121: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(sub_120, mul_502);  sub_120 = mul_502 = None
        mul_503: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(div_58, sub_121);  div_58 = sub_121 = None
        mul_504: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_169, mul_9);  mul_9 = None
        sum_187: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_504, [0, 1]);  mul_504 = None
        sum_188: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_169, [0, 1]);  add_169 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:379, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_35: "f32[3, 5, 768]" = torch.ops.prims.convert_element_type.default(gt_2, torch.float32);  gt_2 = None
        mul_505: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_35, 1.1111111111111112);  convert_element_type_35 = None
        mul_506: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_503, mul_505);  mul_505 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:378, code: hidden_states = self.dense(hidden_states)
        view_579: "f32[15, 768]" = torch.ops.aten.reshape.default(mul_506, [15, 768]);  mul_506 = None
        mm_138: "f32[15, 768]" = torch.ops.aten.mm.default(view_579, permute_508);  permute_508 = None
        permute_509: "f32[768, 15]" = torch.ops.aten.permute.default(view_579, [1, 0])
        mm_139: "f32[768, 768]" = torch.ops.aten.mm.default(permute_509, view_16);  permute_509 = view_16 = None
        permute_510: "f32[768, 768]" = torch.ops.aten.permute.default(mm_139, [1, 0]);  mm_139 = None
        sum_189: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_579, [0], True);  view_579 = None
        view_580: "f32[768]" = torch.ops.aten.reshape.default(sum_189, [768]);  sum_189 = None
        permute_511: "f32[768, 768]" = torch.ops.aten.permute.default(permute_510, [1, 0]);  permute_510 = None
        view_581: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_138, [3, 5, 768]);  mm_138 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:361, code: context_layer = context_layer.view(new_context_layer_shape)
        view_582: "f32[3, 5, 12, 64]" = torch.ops.aten.reshape.default(view_581, [3, 5, 12, 64]);  view_581 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_512: "f32[3, 12, 5, 64]" = torch.ops.aten.permute.default(view_582, [0, 2, 1, 3]);  view_582 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:357, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_127: "f32[3, 12, 5, 64]" = torch.ops.aten.clone.default(permute_512, memory_format = torch.contiguous_format);  permute_512 = None
        view_583: "f32[36, 5, 64]" = torch.ops.aten.reshape.default(clone_127, [36, 5, 64]);  clone_127 = None
        bmm_68: "f32[36, 5, 64]" = torch.ops.aten.bmm.default(permute_513, view_583);  permute_513 = None
        bmm_69: "f32[36, 5, 5]" = torch.ops.aten.bmm.default(view_583, permute_514);  view_583 = permute_514 = None
        view_584: "f32[3, 12, 5, 64]" = torch.ops.aten.reshape.default(bmm_68, [3, 12, 5, 64]);  bmm_68 = None
        view_585: "f32[3, 12, 5, 5]" = torch.ops.aten.reshape.default(bmm_69, [3, 12, 5, 5]);  bmm_69 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:351, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_36: "f32[3, 12, 5, 5]" = torch.ops.prims.convert_element_type.default(gt_1, torch.float32);  gt_1 = None
        mul_507: "f32[3, 12, 5, 5]" = torch.ops.aten.mul.Tensor(convert_element_type_36, 1.1111111111111112);  convert_element_type_36 = None
        mul_508: "f32[3, 12, 5, 5]" = torch.ops.aten.mul.Tensor(view_585, mul_507);  view_585 = mul_507 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:347, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_509: "f32[3, 12, 5, 5]" = torch.ops.aten.mul.Tensor(mul_508, alias_25);  mul_508 = None
        sum_190: "f32[3, 12, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_509, [-1], True)
        mul_510: "f32[3, 12, 5, 5]" = torch.ops.aten.mul.Tensor(alias_25, sum_190);  alias_25 = sum_190 = None
        sub_122: "f32[3, 12, 5, 5]" = torch.ops.aten.sub.Tensor(mul_509, mul_510);  mul_509 = mul_510 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:341, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_59: "f32[3, 12, 5, 5]" = torch.ops.aten.div.Tensor(sub_122, 8.0);  sub_122 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:323, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        view_586: "f32[36, 5, 5]" = torch.ops.aten.reshape.default(div_59, [36, 5, 5]);  div_59 = None
        bmm_70: "f32[36, 64, 5]" = torch.ops.aten.bmm.default(permute_515, view_586);  permute_515 = None
        bmm_71: "f32[36, 5, 64]" = torch.ops.aten.bmm.default(view_586, permute_516);  view_586 = permute_516 = None
        view_587: "f32[3, 12, 64, 5]" = torch.ops.aten.reshape.default(bmm_70, [3, 12, 64, 5]);  bmm_70 = None
        view_588: "f32[3, 12, 5, 64]" = torch.ops.aten.reshape.default(bmm_71, [3, 12, 5, 64]);  bmm_71 = None
        permute_517: "f32[3, 12, 5, 64]" = torch.ops.aten.permute.default(view_587, [0, 1, 3, 2]);  view_587 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_518: "f32[3, 5, 12, 64]" = torch.ops.aten.permute.default(view_588, [0, 2, 1, 3]);  view_588 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        clone_129: "f32[3, 5, 12, 64]" = torch.ops.aten.clone.default(permute_518, memory_format = torch.contiguous_format);  permute_518 = None
        view_589: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(clone_129, [3, 5, 768]);  clone_129 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_519: "f32[3, 5, 12, 64]" = torch.ops.aten.permute.default(view_584, [0, 2, 1, 3]);  view_584 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        clone_130: "f32[3, 5, 12, 64]" = torch.ops.aten.clone.default(permute_519, memory_format = torch.contiguous_format);  permute_519 = None
        view_590: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(clone_130, [3, 5, 768]);  clone_130 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_591: "f32[15, 768]" = torch.ops.aten.reshape.default(view_590, [15, 768]);  view_590 = None
        mm_140: "f32[15, 768]" = torch.ops.aten.mm.default(view_591, permute_520);  permute_520 = None
        permute_521: "f32[768, 15]" = torch.ops.aten.permute.default(view_591, [1, 0])
        mm_141: "f32[768, 768]" = torch.ops.aten.mm.default(permute_521, view);  permute_521 = None
        permute_522: "f32[768, 768]" = torch.ops.aten.permute.default(mm_141, [1, 0]);  mm_141 = None
        sum_191: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_591, [0], True);  view_591 = None
        view_592: "f32[768]" = torch.ops.aten.reshape.default(sum_191, [768]);  sum_191 = None
        permute_523: "f32[768, 768]" = torch.ops.aten.permute.default(permute_522, [1, 0]);  permute_522 = None
        view_593: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_140, [3, 5, 768]);  mm_140 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        add_170: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(mul_503, view_593);  mul_503 = view_593 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_524: "f32[3, 5, 12, 64]" = torch.ops.aten.permute.default(permute_517, [0, 2, 1, 3]);  permute_517 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_594: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(permute_524, [3, 5, 768]);  permute_524 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        clone_131: "f32[3, 5, 768]" = torch.ops.aten.clone.default(view_594, memory_format = torch.contiguous_format);  view_594 = None
        view_595: "f32[15, 768]" = torch.ops.aten.reshape.default(clone_131, [15, 768]);  clone_131 = None
        mm_142: "f32[15, 768]" = torch.ops.aten.mm.default(view_595, permute_525);  permute_525 = None
        permute_526: "f32[768, 15]" = torch.ops.aten.permute.default(view_595, [1, 0])
        mm_143: "f32[768, 768]" = torch.ops.aten.mm.default(permute_526, view);  permute_526 = None
        permute_527: "f32[768, 768]" = torch.ops.aten.permute.default(mm_143, [1, 0]);  mm_143 = None
        sum_192: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_595, [0], True);  view_595 = None
        view_596: "f32[768]" = torch.ops.aten.reshape.default(sum_192, [768]);  sum_192 = None
        permute_528: "f32[768, 768]" = torch.ops.aten.permute.default(permute_527, [1, 0]);  permute_527 = None
        view_597: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_142, [3, 5, 768]);  mm_142 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        add_171: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(add_170, view_597);  add_170 = view_597 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        view_598: "f32[15, 768]" = torch.ops.aten.reshape.default(view_589, [15, 768]);  view_589 = None
        mm_144: "f32[15, 768]" = torch.ops.aten.mm.default(view_598, permute_529);  permute_529 = None
        permute_530: "f32[768, 15]" = torch.ops.aten.permute.default(view_598, [1, 0])
        mm_145: "f32[768, 768]" = torch.ops.aten.mm.default(permute_530, view);  permute_530 = view = None
        permute_531: "f32[768, 768]" = torch.ops.aten.permute.default(mm_145, [1, 0]);  mm_145 = None
        sum_193: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_598, [0], True);  view_598 = None
        view_599: "f32[768]" = torch.ops.aten.reshape.default(sum_193, [768]);  sum_193 = None
        permute_532: "f32[768, 768]" = torch.ops.aten.permute.default(permute_531, [1, 0]);  permute_531 = None
        view_600: "f32[3, 5, 768]" = torch.ops.aten.reshape.default(mm_144, [3, 5, 768]);  mm_144 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        add_172: "f32[3, 5, 768]" = torch.ops.aten.add.Tensor(add_171, view_600);  add_171 = view_600 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:239, code: embeddings = self.dropout(embeddings)
        convert_element_type_37: "f32[3, 5, 768]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_511: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_37, 1.1111111111111112);  convert_element_type_37 = None
        mul_512: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(add_172, mul_511);  add_172 = mul_511 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:238, code: embeddings = self.LayerNorm(embeddings)
        mul_514: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_512, primals_4);  primals_4 = None
        mul_515: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_514, 768)
        sum_194: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_514, [2], True)
        mul_516: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_514, mul_1);  mul_514 = None
        sum_195: "f32[3, 5, 1]" = torch.ops.aten.sum.dim_IntList(mul_516, [2], True);  mul_516 = None
        mul_517: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_1, sum_195);  sum_195 = None
        sub_124: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(mul_515, sum_194);  mul_515 = sum_194 = None
        sub_125: "f32[3, 5, 768]" = torch.ops.aten.sub.Tensor(sub_124, mul_517);  sub_124 = mul_517 = None
        mul_518: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(div_60, sub_125);  div_60 = sub_125 = None
        mul_519: "f32[3, 5, 768]" = torch.ops.aten.mul.Tensor(mul_512, mul_1);  mul_1 = None
        sum_196: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_519, [0, 1]);  mul_519 = None
        sum_197: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_512, [0, 1]);  mul_512 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:237, code: embeddings += position_embeddings
        sum_198: "f32[1, 5, 768]" = torch.ops.aten.sum.dim_IntList(mul_518, [0], True)
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:236, code: position_embeddings = self.position_embeddings(position_ids)
        eq: "b8[1, 5]" = torch.ops.aten.eq.Scalar(slice_6, -1)
        unsqueeze_2: "b8[1, 5, 1]" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
        full_default_2: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[1, 5, 768]" = torch.ops.aten.where.self(unsqueeze_2, full_default_2, sum_198);  unsqueeze_2 = sum_198 = None
        full_default_3: "f32[512, 768]" = torch.ops.aten.full.default([512, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _unsafe_index_put: "f32[512, 768]" = torch.ops.prims._unsafe_index_put_.default(full_default_3, [slice_6], where, True);  full_default_3 = slice_6 = where = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:232, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        eq_1: "b8[3, 5]" = torch.ops.aten.eq.Scalar(expand, -1)
        unsqueeze_3: "b8[3, 5, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
        where_1: "f32[3, 5, 768]" = torch.ops.aten.where.self(unsqueeze_3, full_default_2, mul_518);  unsqueeze_3 = None
        full_default_5: "f32[2, 768]" = torch.ops.aten.full.default([2, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _unsafe_index_put_1: "f32[2, 768]" = torch.ops.prims._unsafe_index_put_.default(full_default_5, [expand], where_1, True);  full_default_5 = expand = where_1 = None
        
        # File: /home/zibo/anaconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:231, code: inputs_embeds = self.word_embeddings(input_ids)
        eq_2: "b8[3, 5]" = torch.ops.aten.eq.Scalar(primals_202, 0)
        unsqueeze_4: "b8[3, 5, 1]" = torch.ops.aten.unsqueeze.default(eq_2, -1);  eq_2 = None
        where_2: "f32[3, 5, 768]" = torch.ops.aten.where.self(unsqueeze_4, full_default_2, mul_518);  unsqueeze_4 = full_default_2 = mul_518 = None
        full_default_7: "f32[30522, 768]" = torch.ops.aten.full.default([30522, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _unsafe_index_put_2: "f32[30522, 768]" = torch.ops.prims._unsafe_index_put_.default(full_default_7, [primals_202], where_2, True);  full_default_7 = primals_202 = where_2 = None
        return [_unsafe_index_put_2, _unsafe_index_put_1, _unsafe_index_put, sum_196, sum_197, permute_532, view_599, permute_528, view_596, permute_523, view_592, permute_511, view_580, sum_187, sum_188, permute_507, view_577, permute_503, view_574, sum_181, sum_182, permute_499, view_571, permute_495, view_568, permute_490, view_564, permute_478, view_552, sum_172, sum_173, permute_474, view_549, permute_470, view_546, sum_166, sum_167, permute_466, view_543, permute_462, view_540, permute_457, view_536, permute_445, view_524, sum_157, sum_158, permute_441, view_521, permute_437, view_518, sum_151, sum_152, permute_433, view_515, permute_429, view_512, permute_424, view_508, permute_412, view_496, sum_142, sum_143, permute_408, view_493, permute_404, view_490, sum_136, sum_137, permute_400, view_487, permute_396, view_484, permute_391, view_480, permute_379, view_468, sum_127, sum_128, permute_375, view_465, permute_371, view_462, sum_121, sum_122, permute_367, view_459, permute_363, view_456, permute_358, view_452, permute_346, view_440, sum_112, sum_113, permute_342, view_437, permute_338, view_434, sum_106, sum_107, permute_334, view_431, permute_330, view_428, permute_325, view_424, permute_313, view_412, sum_97, sum_98, permute_309, view_409, permute_305, view_406, sum_91, sum_92, permute_301, view_403, permute_297, view_400, permute_292, view_396, permute_280, view_384, sum_82, sum_83, permute_276, view_381, permute_272, view_378, sum_76, sum_77, permute_268, view_375, permute_264, view_372, permute_259, view_368, permute_247, view_356, sum_67, sum_68, permute_243, view_353, permute_239, view_350, sum_61, sum_62, permute_235, view_347, permute_231, view_344, permute_226, view_340, permute_214, view_328, sum_52, sum_53, permute_210, view_325, permute_206, view_322, sum_46, sum_47, permute_202, view_319, permute_198, view_316, permute_193, view_312, permute_181, view_300, sum_37, sum_38, permute_177, view_297, permute_173, view_294, sum_31, sum_32, permute_169, view_291, permute_165, view_288, permute_160, view_284, permute_148, view_272, sum_22, sum_23, permute_144, view_269, permute_140, view_266, sum_16, sum_17, permute_136, view_264, None, None, None, None]
        