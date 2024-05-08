class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[16, 3, 3, 3]", primals_2: "f32[16]", primals_3: "f32[16]", primals_4: "f32[16, 1, 3, 3]", primals_5: "f32[16]", primals_6: "f32[16]", primals_7: "f32[8, 16, 1, 1]", primals_8: "f32[8]", primals_9: "f32[16, 8, 1, 1]", primals_10: "f32[16]", primals_11: "f32[16, 16, 1, 1]", primals_12: "f32[16]", primals_13: "f32[16]", primals_14: "f32[72, 16, 1, 1]", primals_15: "f32[72]", primals_16: "f32[72]", primals_17: "f32[72, 1, 3, 3]", primals_18: "f32[72]", primals_19: "f32[72]", primals_20: "f32[24, 72, 1, 1]", primals_21: "f32[24]", primals_22: "f32[24]", primals_23: "f32[88, 24, 1, 1]", primals_24: "f32[88]", primals_25: "f32[88]", primals_26: "f32[88, 1, 3, 3]", primals_27: "f32[88]", primals_28: "f32[88]", primals_29: "f32[24, 88, 1, 1]", primals_30: "f32[24]", primals_31: "f32[24]", primals_32: "f32[96, 24, 1, 1]", primals_33: "f32[96]", primals_34: "f32[96]", primals_35: "f32[96, 1, 5, 5]", primals_36: "f32[96]", primals_37: "f32[96]", primals_38: "f32[24, 96, 1, 1]", primals_39: "f32[24]", primals_40: "f32[96, 24, 1, 1]", primals_41: "f32[96]", primals_42: "f32[40, 96, 1, 1]", primals_43: "f32[40]", primals_44: "f32[40]", primals_45: "f32[240, 40, 1, 1]", primals_46: "f32[240]", primals_47: "f32[240]", primals_48: "f32[240, 1, 5, 5]", primals_49: "f32[240]", primals_50: "f32[240]", primals_51: "f32[64, 240, 1, 1]", primals_52: "f32[64]", primals_53: "f32[240, 64, 1, 1]", primals_54: "f32[240]", primals_55: "f32[40, 240, 1, 1]", primals_56: "f32[40]", primals_57: "f32[40]", primals_58: "f32[240, 40, 1, 1]", primals_59: "f32[240]", primals_60: "f32[240]", primals_61: "f32[240, 1, 5, 5]", primals_62: "f32[240]", primals_63: "f32[240]", primals_64: "f32[64, 240, 1, 1]", primals_65: "f32[64]", primals_66: "f32[240, 64, 1, 1]", primals_67: "f32[240]", primals_68: "f32[40, 240, 1, 1]", primals_69: "f32[40]", primals_70: "f32[40]", primals_71: "f32[120, 40, 1, 1]", primals_72: "f32[120]", primals_73: "f32[120]", primals_74: "f32[120, 1, 5, 5]", primals_75: "f32[120]", primals_76: "f32[120]", primals_77: "f32[32, 120, 1, 1]", primals_78: "f32[32]", primals_79: "f32[120, 32, 1, 1]", primals_80: "f32[120]", primals_81: "f32[48, 120, 1, 1]", primals_82: "f32[48]", primals_83: "f32[48]", primals_84: "f32[144, 48, 1, 1]", primals_85: "f32[144]", primals_86: "f32[144]", primals_87: "f32[144, 1, 5, 5]", primals_88: "f32[144]", primals_89: "f32[144]", primals_90: "f32[40, 144, 1, 1]", primals_91: "f32[40]", primals_92: "f32[144, 40, 1, 1]", primals_93: "f32[144]", primals_94: "f32[48, 144, 1, 1]", primals_95: "f32[48]", primals_96: "f32[48]", primals_97: "f32[288, 48, 1, 1]", primals_98: "f32[288]", primals_99: "f32[288]", primals_100: "f32[288, 1, 5, 5]", primals_101: "f32[288]", primals_102: "f32[288]", primals_103: "f32[72, 288, 1, 1]", primals_104: "f32[72]", primals_105: "f32[288, 72, 1, 1]", primals_106: "f32[288]", primals_107: "f32[96, 288, 1, 1]", primals_108: "f32[96]", primals_109: "f32[96]", primals_110: "f32[576, 96, 1, 1]", primals_111: "f32[576]", primals_112: "f32[576]", primals_113: "f32[576, 1, 5, 5]", primals_114: "f32[576]", primals_115: "f32[576]", primals_116: "f32[144, 576, 1, 1]", primals_117: "f32[144]", primals_118: "f32[576, 144, 1, 1]", primals_119: "f32[576]", primals_120: "f32[96, 576, 1, 1]", primals_121: "f32[96]", primals_122: "f32[96]", primals_123: "f32[576, 96, 1, 1]", primals_124: "f32[576]", primals_125: "f32[576]", primals_126: "f32[576, 1, 5, 5]", primals_127: "f32[576]", primals_128: "f32[576]", primals_129: "f32[144, 576, 1, 1]", primals_130: "f32[144]", primals_131: "f32[576, 144, 1, 1]", primals_132: "f32[576]", primals_133: "f32[96, 576, 1, 1]", primals_134: "f32[96]", primals_135: "f32[96]", primals_136: "f32[576, 96, 1, 1]", primals_137: "f32[576]", primals_138: "f32[576]", primals_139: "f32[1024, 576]", primals_140: "f32[1024]", primals_141: "f32[1000, 1024]", primals_142: "f32[1000]", primals_143: "f32[16]", primals_144: "f32[16]", primals_145: "i64[]", primals_146: "f32[16]", primals_147: "f32[16]", primals_148: "i64[]", primals_149: "f32[16]", primals_150: "f32[16]", primals_151: "i64[]", primals_152: "f32[72]", primals_153: "f32[72]", primals_154: "i64[]", primals_155: "f32[72]", primals_156: "f32[72]", primals_157: "i64[]", primals_158: "f32[24]", primals_159: "f32[24]", primals_160: "i64[]", primals_161: "f32[88]", primals_162: "f32[88]", primals_163: "i64[]", primals_164: "f32[88]", primals_165: "f32[88]", primals_166: "i64[]", primals_167: "f32[24]", primals_168: "f32[24]", primals_169: "i64[]", primals_170: "f32[96]", primals_171: "f32[96]", primals_172: "i64[]", primals_173: "f32[96]", primals_174: "f32[96]", primals_175: "i64[]", primals_176: "f32[40]", primals_177: "f32[40]", primals_178: "i64[]", primals_179: "f32[240]", primals_180: "f32[240]", primals_181: "i64[]", primals_182: "f32[240]", primals_183: "f32[240]", primals_184: "i64[]", primals_185: "f32[40]", primals_186: "f32[40]", primals_187: "i64[]", primals_188: "f32[240]", primals_189: "f32[240]", primals_190: "i64[]", primals_191: "f32[240]", primals_192: "f32[240]", primals_193: "i64[]", primals_194: "f32[40]", primals_195: "f32[40]", primals_196: "i64[]", primals_197: "f32[120]", primals_198: "f32[120]", primals_199: "i64[]", primals_200: "f32[120]", primals_201: "f32[120]", primals_202: "i64[]", primals_203: "f32[48]", primals_204: "f32[48]", primals_205: "i64[]", primals_206: "f32[144]", primals_207: "f32[144]", primals_208: "i64[]", primals_209: "f32[144]", primals_210: "f32[144]", primals_211: "i64[]", primals_212: "f32[48]", primals_213: "f32[48]", primals_214: "i64[]", primals_215: "f32[288]", primals_216: "f32[288]", primals_217: "i64[]", primals_218: "f32[288]", primals_219: "f32[288]", primals_220: "i64[]", primals_221: "f32[96]", primals_222: "f32[96]", primals_223: "i64[]", primals_224: "f32[576]", primals_225: "f32[576]", primals_226: "i64[]", primals_227: "f32[576]", primals_228: "f32[576]", primals_229: "i64[]", primals_230: "f32[96]", primals_231: "f32[96]", primals_232: "i64[]", primals_233: "f32[576]", primals_234: "f32[576]", primals_235: "i64[]", primals_236: "f32[576]", primals_237: "f32[576]", primals_238: "i64[]", primals_239: "f32[96]", primals_240: "f32[96]", primals_241: "i64[]", primals_242: "f32[576]", primals_243: "f32[576]", primals_244: "i64[]", primals_245: "f32[64, 3, 224, 224]"):
        # No stacktrace found for following nodes
        clone: "f32[16]" = torch.ops.aten.clone.default(primals_143);  primals_143 = None
        clone_1: "f32[16]" = torch.ops.aten.clone.default(primals_144);  primals_144 = None
        clone_2: "i64[]" = torch.ops.aten.clone.default(primals_145);  primals_145 = None
        clone_3: "f32[16]" = torch.ops.aten.clone.default(primals_146);  primals_146 = None
        clone_4: "f32[16]" = torch.ops.aten.clone.default(primals_147);  primals_147 = None
        clone_5: "i64[]" = torch.ops.aten.clone.default(primals_148);  primals_148 = None
        clone_6: "f32[16]" = torch.ops.aten.clone.default(primals_149);  primals_149 = None
        clone_7: "f32[16]" = torch.ops.aten.clone.default(primals_150);  primals_150 = None
        clone_8: "i64[]" = torch.ops.aten.clone.default(primals_151);  primals_151 = None
        clone_9: "f32[72]" = torch.ops.aten.clone.default(primals_152);  primals_152 = None
        clone_10: "f32[72]" = torch.ops.aten.clone.default(primals_153);  primals_153 = None
        clone_11: "i64[]" = torch.ops.aten.clone.default(primals_154);  primals_154 = None
        clone_12: "f32[72]" = torch.ops.aten.clone.default(primals_155);  primals_155 = None
        clone_13: "f32[72]" = torch.ops.aten.clone.default(primals_156);  primals_156 = None
        clone_14: "i64[]" = torch.ops.aten.clone.default(primals_157);  primals_157 = None
        clone_15: "f32[24]" = torch.ops.aten.clone.default(primals_158);  primals_158 = None
        clone_16: "f32[24]" = torch.ops.aten.clone.default(primals_159);  primals_159 = None
        clone_17: "i64[]" = torch.ops.aten.clone.default(primals_160);  primals_160 = None
        clone_18: "f32[88]" = torch.ops.aten.clone.default(primals_161);  primals_161 = None
        clone_19: "f32[88]" = torch.ops.aten.clone.default(primals_162);  primals_162 = None
        clone_20: "i64[]" = torch.ops.aten.clone.default(primals_163);  primals_163 = None
        clone_21: "f32[88]" = torch.ops.aten.clone.default(primals_164);  primals_164 = None
        clone_22: "f32[88]" = torch.ops.aten.clone.default(primals_165);  primals_165 = None
        clone_23: "i64[]" = torch.ops.aten.clone.default(primals_166);  primals_166 = None
        clone_24: "f32[24]" = torch.ops.aten.clone.default(primals_167);  primals_167 = None
        clone_25: "f32[24]" = torch.ops.aten.clone.default(primals_168);  primals_168 = None
        clone_26: "i64[]" = torch.ops.aten.clone.default(primals_169);  primals_169 = None
        clone_27: "f32[96]" = torch.ops.aten.clone.default(primals_170);  primals_170 = None
        clone_28: "f32[96]" = torch.ops.aten.clone.default(primals_171);  primals_171 = None
        clone_29: "i64[]" = torch.ops.aten.clone.default(primals_172);  primals_172 = None
        clone_30: "f32[96]" = torch.ops.aten.clone.default(primals_173);  primals_173 = None
        clone_31: "f32[96]" = torch.ops.aten.clone.default(primals_174);  primals_174 = None
        clone_32: "i64[]" = torch.ops.aten.clone.default(primals_175);  primals_175 = None
        clone_33: "f32[40]" = torch.ops.aten.clone.default(primals_176);  primals_176 = None
        clone_34: "f32[40]" = torch.ops.aten.clone.default(primals_177);  primals_177 = None
        clone_35: "i64[]" = torch.ops.aten.clone.default(primals_178);  primals_178 = None
        clone_36: "f32[240]" = torch.ops.aten.clone.default(primals_179);  primals_179 = None
        clone_37: "f32[240]" = torch.ops.aten.clone.default(primals_180);  primals_180 = None
        clone_38: "i64[]" = torch.ops.aten.clone.default(primals_181);  primals_181 = None
        clone_39: "f32[240]" = torch.ops.aten.clone.default(primals_182);  primals_182 = None
        clone_40: "f32[240]" = torch.ops.aten.clone.default(primals_183);  primals_183 = None
        clone_41: "i64[]" = torch.ops.aten.clone.default(primals_184);  primals_184 = None
        clone_42: "f32[40]" = torch.ops.aten.clone.default(primals_185);  primals_185 = None
        clone_43: "f32[40]" = torch.ops.aten.clone.default(primals_186);  primals_186 = None
        clone_44: "i64[]" = torch.ops.aten.clone.default(primals_187);  primals_187 = None
        clone_45: "f32[240]" = torch.ops.aten.clone.default(primals_188);  primals_188 = None
        clone_46: "f32[240]" = torch.ops.aten.clone.default(primals_189);  primals_189 = None
        clone_47: "i64[]" = torch.ops.aten.clone.default(primals_190);  primals_190 = None
        clone_48: "f32[240]" = torch.ops.aten.clone.default(primals_191);  primals_191 = None
        clone_49: "f32[240]" = torch.ops.aten.clone.default(primals_192);  primals_192 = None
        clone_50: "i64[]" = torch.ops.aten.clone.default(primals_193);  primals_193 = None
        clone_51: "f32[40]" = torch.ops.aten.clone.default(primals_194);  primals_194 = None
        clone_52: "f32[40]" = torch.ops.aten.clone.default(primals_195);  primals_195 = None
        clone_53: "i64[]" = torch.ops.aten.clone.default(primals_196);  primals_196 = None
        clone_54: "f32[120]" = torch.ops.aten.clone.default(primals_197);  primals_197 = None
        clone_55: "f32[120]" = torch.ops.aten.clone.default(primals_198);  primals_198 = None
        clone_56: "i64[]" = torch.ops.aten.clone.default(primals_199);  primals_199 = None
        clone_57: "f32[120]" = torch.ops.aten.clone.default(primals_200);  primals_200 = None
        clone_58: "f32[120]" = torch.ops.aten.clone.default(primals_201);  primals_201 = None
        clone_59: "i64[]" = torch.ops.aten.clone.default(primals_202);  primals_202 = None
        clone_60: "f32[48]" = torch.ops.aten.clone.default(primals_203);  primals_203 = None
        clone_61: "f32[48]" = torch.ops.aten.clone.default(primals_204);  primals_204 = None
        clone_62: "i64[]" = torch.ops.aten.clone.default(primals_205);  primals_205 = None
        clone_63: "f32[144]" = torch.ops.aten.clone.default(primals_206);  primals_206 = None
        clone_64: "f32[144]" = torch.ops.aten.clone.default(primals_207);  primals_207 = None
        clone_65: "i64[]" = torch.ops.aten.clone.default(primals_208);  primals_208 = None
        clone_66: "f32[144]" = torch.ops.aten.clone.default(primals_209);  primals_209 = None
        clone_67: "f32[144]" = torch.ops.aten.clone.default(primals_210);  primals_210 = None
        clone_68: "i64[]" = torch.ops.aten.clone.default(primals_211);  primals_211 = None
        clone_69: "f32[48]" = torch.ops.aten.clone.default(primals_212);  primals_212 = None
        clone_70: "f32[48]" = torch.ops.aten.clone.default(primals_213);  primals_213 = None
        clone_71: "i64[]" = torch.ops.aten.clone.default(primals_214);  primals_214 = None
        clone_72: "f32[288]" = torch.ops.aten.clone.default(primals_215);  primals_215 = None
        clone_73: "f32[288]" = torch.ops.aten.clone.default(primals_216);  primals_216 = None
        clone_74: "i64[]" = torch.ops.aten.clone.default(primals_217);  primals_217 = None
        clone_75: "f32[288]" = torch.ops.aten.clone.default(primals_218);  primals_218 = None
        clone_76: "f32[288]" = torch.ops.aten.clone.default(primals_219);  primals_219 = None
        clone_77: "i64[]" = torch.ops.aten.clone.default(primals_220);  primals_220 = None
        clone_78: "f32[96]" = torch.ops.aten.clone.default(primals_221);  primals_221 = None
        clone_79: "f32[96]" = torch.ops.aten.clone.default(primals_222);  primals_222 = None
        clone_80: "i64[]" = torch.ops.aten.clone.default(primals_223);  primals_223 = None
        clone_81: "f32[576]" = torch.ops.aten.clone.default(primals_224);  primals_224 = None
        clone_82: "f32[576]" = torch.ops.aten.clone.default(primals_225);  primals_225 = None
        clone_83: "i64[]" = torch.ops.aten.clone.default(primals_226);  primals_226 = None
        clone_84: "f32[576]" = torch.ops.aten.clone.default(primals_227);  primals_227 = None
        clone_85: "f32[576]" = torch.ops.aten.clone.default(primals_228);  primals_228 = None
        clone_86: "i64[]" = torch.ops.aten.clone.default(primals_229);  primals_229 = None
        clone_87: "f32[96]" = torch.ops.aten.clone.default(primals_230);  primals_230 = None
        clone_88: "f32[96]" = torch.ops.aten.clone.default(primals_231);  primals_231 = None
        clone_89: "i64[]" = torch.ops.aten.clone.default(primals_232);  primals_232 = None
        clone_90: "f32[576]" = torch.ops.aten.clone.default(primals_233);  primals_233 = None
        clone_91: "f32[576]" = torch.ops.aten.clone.default(primals_234);  primals_234 = None
        clone_92: "i64[]" = torch.ops.aten.clone.default(primals_235);  primals_235 = None
        clone_93: "f32[576]" = torch.ops.aten.clone.default(primals_236);  primals_236 = None
        clone_94: "f32[576]" = torch.ops.aten.clone.default(primals_237);  primals_237 = None
        clone_95: "i64[]" = torch.ops.aten.clone.default(primals_238);  primals_238 = None
        clone_96: "f32[96]" = torch.ops.aten.clone.default(primals_239);  primals_239 = None
        clone_97: "f32[96]" = torch.ops.aten.clone.default(primals_240);  primals_240 = None
        clone_98: "i64[]" = torch.ops.aten.clone.default(primals_241);  primals_241 = None
        clone_99: "f32[576]" = torch.ops.aten.clone.default(primals_242);  primals_242 = None
        clone_100: "f32[576]" = torch.ops.aten.clone.default(primals_243);  primals_243 = None
        clone_101: "i64[]" = torch.ops.aten.clone.default(primals_244);  primals_244 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:210, code: x = self.features(x)
        convolution: "f32[64, 16, 112, 112]" = torch.ops.aten.convolution.default(primals_245, primals_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
        add: "i64[]" = torch.ops.aten.add.Tensor(clone_2, 1);  clone_2 = None
        var_mean = torch.ops.aten.var_mean.correction(convolution, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 16, 1, 1]" = var_mean[0]
        getitem_1: "f32[1, 16, 1, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 0.001)
        rsqrt: "f32[1, 16, 1, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[64, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
        mul: "f32[64, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        squeeze: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        squeeze_1: "f32[16]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
        mul_1: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze, 0.01)
        mul_2: "f32[16]" = torch.ops.aten.mul.Tensor(clone, 0.99);  clone = None
        add_2: "f32[16]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        squeeze_2: "f32[16]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_3: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.0000012456169853);  squeeze_2 = None
        mul_4: "f32[16]" = torch.ops.aten.mul.Tensor(mul_3, 0.01);  mul_3 = None
        mul_5: "f32[16]" = torch.ops.aten.mul.Tensor(clone_1, 0.99);  clone_1 = None
        add_3: "f32[16]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        unsqueeze: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_2, -1)
        unsqueeze_1: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        unsqueeze_2: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_3, -1);  primals_3 = None
        unsqueeze_3: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        mul_6: "f32[64, 16, 112, 112]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        add_4: "f32[64, 16, 112, 112]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
        clone_102: "f32[64, 16, 112, 112]" = torch.ops.aten.clone.default(add_4)
        add_5: "f32[64, 16, 112, 112]" = torch.ops.aten.add.Tensor(add_4, 3)
        clamp_min: "f32[64, 16, 112, 112]" = torch.ops.aten.clamp_min.default(add_5, 0);  add_5 = None
        clamp_max: "f32[64, 16, 112, 112]" = torch.ops.aten.clamp_max.default(clamp_min, 6);  clamp_min = None
        mul_7: "f32[64, 16, 112, 112]" = torch.ops.aten.mul.Tensor(add_4, clamp_max);  add_4 = clamp_max = None
        div: "f32[64, 16, 112, 112]" = torch.ops.aten.div.Tensor(mul_7, 6);  mul_7 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
        convolution_1: "f32[64, 16, 56, 56]" = torch.ops.aten.convolution.default(div, primals_4, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 16)
        add_6: "i64[]" = torch.ops.aten.add.Tensor(clone_5, 1);  clone_5 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
        getitem_2: "f32[1, 16, 1, 1]" = var_mean_1[0]
        getitem_3: "f32[1, 16, 1, 1]" = var_mean_1[1];  var_mean_1 = None
        add_7: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 0.001)
        rsqrt_1: "f32[1, 16, 1, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        sub_1: "f32[64, 16, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
        mul_8: "f32[64, 16, 56, 56]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        squeeze_3: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
        squeeze_4: "f32[16]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
        mul_9: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.01)
        mul_10: "f32[16]" = torch.ops.aten.mul.Tensor(clone_3, 0.99);  clone_3 = None
        add_8: "f32[16]" = torch.ops.aten.add.Tensor(mul_9, mul_10);  mul_9 = mul_10 = None
        squeeze_5: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
        mul_11: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.0000049824865598);  squeeze_5 = None
        mul_12: "f32[16]" = torch.ops.aten.mul.Tensor(mul_11, 0.01);  mul_11 = None
        mul_13: "f32[16]" = torch.ops.aten.mul.Tensor(clone_4, 0.99);  clone_4 = None
        add_9: "f32[16]" = torch.ops.aten.add.Tensor(mul_12, mul_13);  mul_12 = mul_13 = None
        unsqueeze_4: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1)
        unsqueeze_5: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        unsqueeze_6: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1);  primals_6 = None
        unsqueeze_7: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        mul_14: "f32[64, 16, 56, 56]" = torch.ops.aten.mul.Tensor(mul_8, unsqueeze_5);  mul_8 = unsqueeze_5 = None
        add_10: "f32[64, 16, 56, 56]" = torch.ops.aten.add.Tensor(mul_14, unsqueeze_7);  mul_14 = unsqueeze_7 = None
        relu: "f32[64, 16, 56, 56]" = torch.ops.aten.relu.default(add_10);  add_10 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
        mean: "f32[64, 16, 1, 1]" = torch.ops.aten.mean.dim(relu, [-1, -2], True)
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
        convolution_2: "f32[64, 8, 1, 1]" = torch.ops.aten.convolution.default(mean, primals_7, primals_8, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_8 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
        relu_1: "f32[64, 8, 1, 1]" = torch.ops.aten.relu.default(convolution_2);  convolution_2 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
        convolution_3: "f32[64, 16, 1, 1]" = torch.ops.aten.convolution.default(relu_1, primals_9, primals_10, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_10 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
        add_11: "f32[64, 16, 1, 1]" = torch.ops.aten.add.Tensor(convolution_3, 3)
        clamp_min_1: "f32[64, 16, 1, 1]" = torch.ops.aten.clamp_min.default(add_11, 0);  add_11 = None
        clamp_max_1: "f32[64, 16, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_1, 6);  clamp_min_1 = None
        div_1: "f32[64, 16, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_1, 6);  clamp_max_1 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:260, code: return scale * input
        mul_15: "f32[64, 16, 56, 56]" = torch.ops.aten.mul.Tensor(div_1, relu)
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
        convolution_4: "f32[64, 16, 56, 56]" = torch.ops.aten.convolution.default(mul_15, primals_11, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_12: "i64[]" = torch.ops.aten.add.Tensor(clone_8, 1);  clone_8 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(convolution_4, [0, 2, 3], correction = 0, keepdim = True)
        getitem_4: "f32[1, 16, 1, 1]" = var_mean_2[0]
        getitem_5: "f32[1, 16, 1, 1]" = var_mean_2[1];  var_mean_2 = None
        add_13: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 0.001)
        rsqrt_2: "f32[1, 16, 1, 1]" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        sub_2: "f32[64, 16, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, getitem_5)
        mul_16: "f32[64, 16, 56, 56]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        squeeze_6: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
        squeeze_7: "f32[16]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
        mul_17: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.01)
        mul_18: "f32[16]" = torch.ops.aten.mul.Tensor(clone_6, 0.99);  clone_6 = None
        add_14: "f32[16]" = torch.ops.aten.add.Tensor(mul_17, mul_18);  mul_17 = mul_18 = None
        squeeze_8: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
        mul_19: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.0000049824865598);  squeeze_8 = None
        mul_20: "f32[16]" = torch.ops.aten.mul.Tensor(mul_19, 0.01);  mul_19 = None
        mul_21: "f32[16]" = torch.ops.aten.mul.Tensor(clone_7, 0.99);  clone_7 = None
        add_15: "f32[16]" = torch.ops.aten.add.Tensor(mul_20, mul_21);  mul_20 = mul_21 = None
        unsqueeze_8: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1)
        unsqueeze_9: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        unsqueeze_10: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_13, -1);  primals_13 = None
        unsqueeze_11: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        mul_22: "f32[64, 16, 56, 56]" = torch.ops.aten.mul.Tensor(mul_16, unsqueeze_9);  mul_16 = unsqueeze_9 = None
        add_16: "f32[64, 16, 56, 56]" = torch.ops.aten.add.Tensor(mul_22, unsqueeze_11);  mul_22 = unsqueeze_11 = None
        convolution_5: "f32[64, 72, 56, 56]" = torch.ops.aten.convolution.default(add_16, primals_14, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_17: "i64[]" = torch.ops.aten.add.Tensor(clone_11, 1);  clone_11 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(convolution_5, [0, 2, 3], correction = 0, keepdim = True)
        getitem_6: "f32[1, 72, 1, 1]" = var_mean_3[0]
        getitem_7: "f32[1, 72, 1, 1]" = var_mean_3[1];  var_mean_3 = None
        add_18: "f32[1, 72, 1, 1]" = torch.ops.aten.add.Tensor(getitem_6, 0.001)
        rsqrt_3: "f32[1, 72, 1, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_3: "f32[64, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, getitem_7)
        mul_23: "f32[64, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        squeeze_9: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
        squeeze_10: "f32[72]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
        mul_24: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_9, 0.01)
        mul_25: "f32[72]" = torch.ops.aten.mul.Tensor(clone_9, 0.99);  clone_9 = None
        add_19: "f32[72]" = torch.ops.aten.add.Tensor(mul_24, mul_25);  mul_24 = mul_25 = None
        squeeze_11: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
        mul_26: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.0000049824865598);  squeeze_11 = None
        mul_27: "f32[72]" = torch.ops.aten.mul.Tensor(mul_26, 0.01);  mul_26 = None
        mul_28: "f32[72]" = torch.ops.aten.mul.Tensor(clone_10, 0.99);  clone_10 = None
        add_20: "f32[72]" = torch.ops.aten.add.Tensor(mul_27, mul_28);  mul_27 = mul_28 = None
        unsqueeze_12: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1)
        unsqueeze_13: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        unsqueeze_14: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_16, -1);  primals_16 = None
        unsqueeze_15: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        mul_29: "f32[64, 72, 56, 56]" = torch.ops.aten.mul.Tensor(mul_23, unsqueeze_13);  mul_23 = unsqueeze_13 = None
        add_21: "f32[64, 72, 56, 56]" = torch.ops.aten.add.Tensor(mul_29, unsqueeze_15);  mul_29 = unsqueeze_15 = None
        relu_2: "f32[64, 72, 56, 56]" = torch.ops.aten.relu.default(add_21);  add_21 = None
        convolution_6: "f32[64, 72, 28, 28]" = torch.ops.aten.convolution.default(relu_2, primals_17, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 72)
        add_22: "i64[]" = torch.ops.aten.add.Tensor(clone_14, 1);  clone_14 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(convolution_6, [0, 2, 3], correction = 0, keepdim = True)
        getitem_8: "f32[1, 72, 1, 1]" = var_mean_4[0]
        getitem_9: "f32[1, 72, 1, 1]" = var_mean_4[1];  var_mean_4 = None
        add_23: "f32[1, 72, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 0.001)
        rsqrt_4: "f32[1, 72, 1, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        sub_4: "f32[64, 72, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_6, getitem_9)
        mul_30: "f32[64, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        squeeze_12: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
        squeeze_13: "f32[72]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
        mul_31: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_12, 0.01)
        mul_32: "f32[72]" = torch.ops.aten.mul.Tensor(clone_12, 0.99);  clone_12 = None
        add_24: "f32[72]" = torch.ops.aten.add.Tensor(mul_31, mul_32);  mul_31 = mul_32 = None
        squeeze_14: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
        mul_33: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_14, 1.0000199302441455);  squeeze_14 = None
        mul_34: "f32[72]" = torch.ops.aten.mul.Tensor(mul_33, 0.01);  mul_33 = None
        mul_35: "f32[72]" = torch.ops.aten.mul.Tensor(clone_13, 0.99);  clone_13 = None
        add_25: "f32[72]" = torch.ops.aten.add.Tensor(mul_34, mul_35);  mul_34 = mul_35 = None
        unsqueeze_16: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1)
        unsqueeze_17: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        unsqueeze_18: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1);  primals_19 = None
        unsqueeze_19: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        mul_36: "f32[64, 72, 28, 28]" = torch.ops.aten.mul.Tensor(mul_30, unsqueeze_17);  mul_30 = unsqueeze_17 = None
        add_26: "f32[64, 72, 28, 28]" = torch.ops.aten.add.Tensor(mul_36, unsqueeze_19);  mul_36 = unsqueeze_19 = None
        relu_3: "f32[64, 72, 28, 28]" = torch.ops.aten.relu.default(add_26);  add_26 = None
        convolution_7: "f32[64, 24, 28, 28]" = torch.ops.aten.convolution.default(relu_3, primals_20, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_27: "i64[]" = torch.ops.aten.add.Tensor(clone_17, 1);  clone_17 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(convolution_7, [0, 2, 3], correction = 0, keepdim = True)
        getitem_10: "f32[1, 24, 1, 1]" = var_mean_5[0]
        getitem_11: "f32[1, 24, 1, 1]" = var_mean_5[1];  var_mean_5 = None
        add_28: "f32[1, 24, 1, 1]" = torch.ops.aten.add.Tensor(getitem_10, 0.001)
        rsqrt_5: "f32[1, 24, 1, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        sub_5: "f32[64, 24, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_7, getitem_11)
        mul_37: "f32[64, 24, 28, 28]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        squeeze_15: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
        squeeze_16: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
        mul_38: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_15, 0.01)
        mul_39: "f32[24]" = torch.ops.aten.mul.Tensor(clone_15, 0.99);  clone_15 = None
        add_29: "f32[24]" = torch.ops.aten.add.Tensor(mul_38, mul_39);  mul_38 = mul_39 = None
        squeeze_17: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
        mul_40: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_17, 1.0000199302441455);  squeeze_17 = None
        mul_41: "f32[24]" = torch.ops.aten.mul.Tensor(mul_40, 0.01);  mul_40 = None
        mul_42: "f32[24]" = torch.ops.aten.mul.Tensor(clone_16, 0.99);  clone_16 = None
        add_30: "f32[24]" = torch.ops.aten.add.Tensor(mul_41, mul_42);  mul_41 = mul_42 = None
        unsqueeze_20: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_21, -1)
        unsqueeze_21: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        unsqueeze_22: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_22, -1);  primals_22 = None
        unsqueeze_23: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        mul_43: "f32[64, 24, 28, 28]" = torch.ops.aten.mul.Tensor(mul_37, unsqueeze_21);  mul_37 = unsqueeze_21 = None
        add_31: "f32[64, 24, 28, 28]" = torch.ops.aten.add.Tensor(mul_43, unsqueeze_23);  mul_43 = unsqueeze_23 = None
        convolution_8: "f32[64, 88, 28, 28]" = torch.ops.aten.convolution.default(add_31, primals_23, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_32: "i64[]" = torch.ops.aten.add.Tensor(clone_20, 1);  clone_20 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(convolution_8, [0, 2, 3], correction = 0, keepdim = True)
        getitem_12: "f32[1, 88, 1, 1]" = var_mean_6[0]
        getitem_13: "f32[1, 88, 1, 1]" = var_mean_6[1];  var_mean_6 = None
        add_33: "f32[1, 88, 1, 1]" = torch.ops.aten.add.Tensor(getitem_12, 0.001)
        rsqrt_6: "f32[1, 88, 1, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
        sub_6: "f32[64, 88, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_8, getitem_13)
        mul_44: "f32[64, 88, 28, 28]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
        squeeze_18: "f32[88]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
        squeeze_19: "f32[88]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
        mul_45: "f32[88]" = torch.ops.aten.mul.Tensor(squeeze_18, 0.01)
        mul_46: "f32[88]" = torch.ops.aten.mul.Tensor(clone_18, 0.99);  clone_18 = None
        add_34: "f32[88]" = torch.ops.aten.add.Tensor(mul_45, mul_46);  mul_45 = mul_46 = None
        squeeze_20: "f32[88]" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
        mul_47: "f32[88]" = torch.ops.aten.mul.Tensor(squeeze_20, 1.0000199302441455);  squeeze_20 = None
        mul_48: "f32[88]" = torch.ops.aten.mul.Tensor(mul_47, 0.01);  mul_47 = None
        mul_49: "f32[88]" = torch.ops.aten.mul.Tensor(clone_19, 0.99);  clone_19 = None
        add_35: "f32[88]" = torch.ops.aten.add.Tensor(mul_48, mul_49);  mul_48 = mul_49 = None
        unsqueeze_24: "f32[88, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1)
        unsqueeze_25: "f32[88, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
        unsqueeze_26: "f32[88, 1]" = torch.ops.aten.unsqueeze.default(primals_25, -1);  primals_25 = None
        unsqueeze_27: "f32[88, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
        mul_50: "f32[64, 88, 28, 28]" = torch.ops.aten.mul.Tensor(mul_44, unsqueeze_25);  mul_44 = unsqueeze_25 = None
        add_36: "f32[64, 88, 28, 28]" = torch.ops.aten.add.Tensor(mul_50, unsqueeze_27);  mul_50 = unsqueeze_27 = None
        relu_4: "f32[64, 88, 28, 28]" = torch.ops.aten.relu.default(add_36);  add_36 = None
        convolution_9: "f32[64, 88, 28, 28]" = torch.ops.aten.convolution.default(relu_4, primals_26, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 88)
        add_37: "i64[]" = torch.ops.aten.add.Tensor(clone_23, 1);  clone_23 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(convolution_9, [0, 2, 3], correction = 0, keepdim = True)
        getitem_14: "f32[1, 88, 1, 1]" = var_mean_7[0]
        getitem_15: "f32[1, 88, 1, 1]" = var_mean_7[1];  var_mean_7 = None
        add_38: "f32[1, 88, 1, 1]" = torch.ops.aten.add.Tensor(getitem_14, 0.001)
        rsqrt_7: "f32[1, 88, 1, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        sub_7: "f32[64, 88, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_9, getitem_15)
        mul_51: "f32[64, 88, 28, 28]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
        squeeze_21: "f32[88]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
        squeeze_22: "f32[88]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
        mul_52: "f32[88]" = torch.ops.aten.mul.Tensor(squeeze_21, 0.01)
        mul_53: "f32[88]" = torch.ops.aten.mul.Tensor(clone_21, 0.99);  clone_21 = None
        add_39: "f32[88]" = torch.ops.aten.add.Tensor(mul_52, mul_53);  mul_52 = mul_53 = None
        squeeze_23: "f32[88]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
        mul_54: "f32[88]" = torch.ops.aten.mul.Tensor(squeeze_23, 1.0000199302441455);  squeeze_23 = None
        mul_55: "f32[88]" = torch.ops.aten.mul.Tensor(mul_54, 0.01);  mul_54 = None
        mul_56: "f32[88]" = torch.ops.aten.mul.Tensor(clone_22, 0.99);  clone_22 = None
        add_40: "f32[88]" = torch.ops.aten.add.Tensor(mul_55, mul_56);  mul_55 = mul_56 = None
        unsqueeze_28: "f32[88, 1]" = torch.ops.aten.unsqueeze.default(primals_27, -1)
        unsqueeze_29: "f32[88, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        unsqueeze_30: "f32[88, 1]" = torch.ops.aten.unsqueeze.default(primals_28, -1);  primals_28 = None
        unsqueeze_31: "f32[88, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
        mul_57: "f32[64, 88, 28, 28]" = torch.ops.aten.mul.Tensor(mul_51, unsqueeze_29);  mul_51 = unsqueeze_29 = None
        add_41: "f32[64, 88, 28, 28]" = torch.ops.aten.add.Tensor(mul_57, unsqueeze_31);  mul_57 = unsqueeze_31 = None
        relu_5: "f32[64, 88, 28, 28]" = torch.ops.aten.relu.default(add_41);  add_41 = None
        convolution_10: "f32[64, 24, 28, 28]" = torch.ops.aten.convolution.default(relu_5, primals_29, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_42: "i64[]" = torch.ops.aten.add.Tensor(clone_26, 1);  clone_26 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(convolution_10, [0, 2, 3], correction = 0, keepdim = True)
        getitem_16: "f32[1, 24, 1, 1]" = var_mean_8[0]
        getitem_17: "f32[1, 24, 1, 1]" = var_mean_8[1];  var_mean_8 = None
        add_43: "f32[1, 24, 1, 1]" = torch.ops.aten.add.Tensor(getitem_16, 0.001)
        rsqrt_8: "f32[1, 24, 1, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        sub_8: "f32[64, 24, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_10, getitem_17)
        mul_58: "f32[64, 24, 28, 28]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
        squeeze_24: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
        squeeze_25: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
        mul_59: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_24, 0.01)
        mul_60: "f32[24]" = torch.ops.aten.mul.Tensor(clone_24, 0.99);  clone_24 = None
        add_44: "f32[24]" = torch.ops.aten.add.Tensor(mul_59, mul_60);  mul_59 = mul_60 = None
        squeeze_26: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
        mul_61: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_26, 1.0000199302441455);  squeeze_26 = None
        mul_62: "f32[24]" = torch.ops.aten.mul.Tensor(mul_61, 0.01);  mul_61 = None
        mul_63: "f32[24]" = torch.ops.aten.mul.Tensor(clone_25, 0.99);  clone_25 = None
        add_45: "f32[24]" = torch.ops.aten.add.Tensor(mul_62, mul_63);  mul_62 = mul_63 = None
        unsqueeze_32: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1)
        unsqueeze_33: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        unsqueeze_34: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_31, -1);  primals_31 = None
        unsqueeze_35: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
        mul_64: "f32[64, 24, 28, 28]" = torch.ops.aten.mul.Tensor(mul_58, unsqueeze_33);  mul_58 = unsqueeze_33 = None
        add_46: "f32[64, 24, 28, 28]" = torch.ops.aten.add.Tensor(mul_64, unsqueeze_35);  mul_64 = unsqueeze_35 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:113, code: result += input
        add_47: "f32[64, 24, 28, 28]" = torch.ops.aten.add.Tensor(add_46, add_31);  add_46 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
        convolution_11: "f32[64, 96, 28, 28]" = torch.ops.aten.convolution.default(add_47, primals_32, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_48: "i64[]" = torch.ops.aten.add.Tensor(clone_29, 1);  clone_29 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(convolution_11, [0, 2, 3], correction = 0, keepdim = True)
        getitem_18: "f32[1, 96, 1, 1]" = var_mean_9[0]
        getitem_19: "f32[1, 96, 1, 1]" = var_mean_9[1];  var_mean_9 = None
        add_49: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem_18, 0.001)
        rsqrt_9: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        sub_9: "f32[64, 96, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_11, getitem_19)
        mul_65: "f32[64, 96, 28, 28]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
        squeeze_27: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
        squeeze_28: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
        mul_66: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_27, 0.01)
        mul_67: "f32[96]" = torch.ops.aten.mul.Tensor(clone_27, 0.99);  clone_27 = None
        add_50: "f32[96]" = torch.ops.aten.add.Tensor(mul_66, mul_67);  mul_66 = mul_67 = None
        squeeze_29: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
        mul_68: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_29, 1.0000199302441455);  squeeze_29 = None
        mul_69: "f32[96]" = torch.ops.aten.mul.Tensor(mul_68, 0.01);  mul_68 = None
        mul_70: "f32[96]" = torch.ops.aten.mul.Tensor(clone_28, 0.99);  clone_28 = None
        add_51: "f32[96]" = torch.ops.aten.add.Tensor(mul_69, mul_70);  mul_69 = mul_70 = None
        unsqueeze_36: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_33, -1)
        unsqueeze_37: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
        unsqueeze_38: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_34, -1);  primals_34 = None
        unsqueeze_39: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
        mul_71: "f32[64, 96, 28, 28]" = torch.ops.aten.mul.Tensor(mul_65, unsqueeze_37);  mul_65 = unsqueeze_37 = None
        add_52: "f32[64, 96, 28, 28]" = torch.ops.aten.add.Tensor(mul_71, unsqueeze_39);  mul_71 = unsqueeze_39 = None
        clone_103: "f32[64, 96, 28, 28]" = torch.ops.aten.clone.default(add_52)
        add_53: "f32[64, 96, 28, 28]" = torch.ops.aten.add.Tensor(add_52, 3)
        clamp_min_2: "f32[64, 96, 28, 28]" = torch.ops.aten.clamp_min.default(add_53, 0);  add_53 = None
        clamp_max_2: "f32[64, 96, 28, 28]" = torch.ops.aten.clamp_max.default(clamp_min_2, 6);  clamp_min_2 = None
        mul_72: "f32[64, 96, 28, 28]" = torch.ops.aten.mul.Tensor(add_52, clamp_max_2);  add_52 = clamp_max_2 = None
        div_2: "f32[64, 96, 28, 28]" = torch.ops.aten.div.Tensor(mul_72, 6);  mul_72 = None
        convolution_12: "f32[64, 96, 14, 14]" = torch.ops.aten.convolution.default(div_2, primals_35, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 96)
        add_54: "i64[]" = torch.ops.aten.add.Tensor(clone_32, 1);  clone_32 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(convolution_12, [0, 2, 3], correction = 0, keepdim = True)
        getitem_20: "f32[1, 96, 1, 1]" = var_mean_10[0]
        getitem_21: "f32[1, 96, 1, 1]" = var_mean_10[1];  var_mean_10 = None
        add_55: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 0.001)
        rsqrt_10: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
        sub_10: "f32[64, 96, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_12, getitem_21)
        mul_73: "f32[64, 96, 14, 14]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
        squeeze_30: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
        squeeze_31: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
        mul_74: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_30, 0.01)
        mul_75: "f32[96]" = torch.ops.aten.mul.Tensor(clone_30, 0.99);  clone_30 = None
        add_56: "f32[96]" = torch.ops.aten.add.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None
        squeeze_32: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
        mul_76: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_32, 1.0000797257434426);  squeeze_32 = None
        mul_77: "f32[96]" = torch.ops.aten.mul.Tensor(mul_76, 0.01);  mul_76 = None
        mul_78: "f32[96]" = torch.ops.aten.mul.Tensor(clone_31, 0.99);  clone_31 = None
        add_57: "f32[96]" = torch.ops.aten.add.Tensor(mul_77, mul_78);  mul_77 = mul_78 = None
        unsqueeze_40: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1)
        unsqueeze_41: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        unsqueeze_42: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_37, -1);  primals_37 = None
        unsqueeze_43: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        mul_79: "f32[64, 96, 14, 14]" = torch.ops.aten.mul.Tensor(mul_73, unsqueeze_41);  mul_73 = unsqueeze_41 = None
        add_58: "f32[64, 96, 14, 14]" = torch.ops.aten.add.Tensor(mul_79, unsqueeze_43);  mul_79 = unsqueeze_43 = None
        clone_104: "f32[64, 96, 14, 14]" = torch.ops.aten.clone.default(add_58)
        add_59: "f32[64, 96, 14, 14]" = torch.ops.aten.add.Tensor(add_58, 3)
        clamp_min_3: "f32[64, 96, 14, 14]" = torch.ops.aten.clamp_min.default(add_59, 0);  add_59 = None
        clamp_max_3: "f32[64, 96, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_3, 6);  clamp_min_3 = None
        mul_80: "f32[64, 96, 14, 14]" = torch.ops.aten.mul.Tensor(add_58, clamp_max_3);  add_58 = clamp_max_3 = None
        div_3: "f32[64, 96, 14, 14]" = torch.ops.aten.div.Tensor(mul_80, 6);  mul_80 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
        mean_1: "f32[64, 96, 1, 1]" = torch.ops.aten.mean.dim(div_3, [-1, -2], True)
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
        convolution_13: "f32[64, 24, 1, 1]" = torch.ops.aten.convolution.default(mean_1, primals_38, primals_39, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_39 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
        relu_6: "f32[64, 24, 1, 1]" = torch.ops.aten.relu.default(convolution_13);  convolution_13 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
        convolution_14: "f32[64, 96, 1, 1]" = torch.ops.aten.convolution.default(relu_6, primals_40, primals_41, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_41 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
        add_60: "f32[64, 96, 1, 1]" = torch.ops.aten.add.Tensor(convolution_14, 3)
        clamp_min_4: "f32[64, 96, 1, 1]" = torch.ops.aten.clamp_min.default(add_60, 0);  add_60 = None
        clamp_max_4: "f32[64, 96, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_4, 6);  clamp_min_4 = None
        div_4: "f32[64, 96, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_4, 6);  clamp_max_4 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:260, code: return scale * input
        mul_81: "f32[64, 96, 14, 14]" = torch.ops.aten.mul.Tensor(div_4, div_3)
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
        convolution_15: "f32[64, 40, 14, 14]" = torch.ops.aten.convolution.default(mul_81, primals_42, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_61: "i64[]" = torch.ops.aten.add.Tensor(clone_35, 1);  clone_35 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(convolution_15, [0, 2, 3], correction = 0, keepdim = True)
        getitem_22: "f32[1, 40, 1, 1]" = var_mean_11[0]
        getitem_23: "f32[1, 40, 1, 1]" = var_mean_11[1];  var_mean_11 = None
        add_62: "f32[1, 40, 1, 1]" = torch.ops.aten.add.Tensor(getitem_22, 0.001)
        rsqrt_11: "f32[1, 40, 1, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        sub_11: "f32[64, 40, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_15, getitem_23)
        mul_82: "f32[64, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
        squeeze_33: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
        squeeze_34: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
        mul_83: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_33, 0.01)
        mul_84: "f32[40]" = torch.ops.aten.mul.Tensor(clone_33, 0.99);  clone_33 = None
        add_63: "f32[40]" = torch.ops.aten.add.Tensor(mul_83, mul_84);  mul_83 = mul_84 = None
        squeeze_35: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
        mul_85: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_35, 1.0000797257434426);  squeeze_35 = None
        mul_86: "f32[40]" = torch.ops.aten.mul.Tensor(mul_85, 0.01);  mul_85 = None
        mul_87: "f32[40]" = torch.ops.aten.mul.Tensor(clone_34, 0.99);  clone_34 = None
        add_64: "f32[40]" = torch.ops.aten.add.Tensor(mul_86, mul_87);  mul_86 = mul_87 = None
        unsqueeze_44: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_43, -1)
        unsqueeze_45: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
        unsqueeze_46: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_44, -1);  primals_44 = None
        unsqueeze_47: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
        mul_88: "f32[64, 40, 14, 14]" = torch.ops.aten.mul.Tensor(mul_82, unsqueeze_45);  mul_82 = unsqueeze_45 = None
        add_65: "f32[64, 40, 14, 14]" = torch.ops.aten.add.Tensor(mul_88, unsqueeze_47);  mul_88 = unsqueeze_47 = None
        convolution_16: "f32[64, 240, 14, 14]" = torch.ops.aten.convolution.default(add_65, primals_45, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_66: "i64[]" = torch.ops.aten.add.Tensor(clone_38, 1);  clone_38 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(convolution_16, [0, 2, 3], correction = 0, keepdim = True)
        getitem_24: "f32[1, 240, 1, 1]" = var_mean_12[0]
        getitem_25: "f32[1, 240, 1, 1]" = var_mean_12[1];  var_mean_12 = None
        add_67: "f32[1, 240, 1, 1]" = torch.ops.aten.add.Tensor(getitem_24, 0.001)
        rsqrt_12: "f32[1, 240, 1, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
        sub_12: "f32[64, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_16, getitem_25)
        mul_89: "f32[64, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
        squeeze_36: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
        squeeze_37: "f32[240]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
        mul_90: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_36, 0.01)
        mul_91: "f32[240]" = torch.ops.aten.mul.Tensor(clone_36, 0.99);  clone_36 = None
        add_68: "f32[240]" = torch.ops.aten.add.Tensor(mul_90, mul_91);  mul_90 = mul_91 = None
        squeeze_38: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
        mul_92: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_38, 1.0000797257434426);  squeeze_38 = None
        mul_93: "f32[240]" = torch.ops.aten.mul.Tensor(mul_92, 0.01);  mul_92 = None
        mul_94: "f32[240]" = torch.ops.aten.mul.Tensor(clone_37, 0.99);  clone_37 = None
        add_69: "f32[240]" = torch.ops.aten.add.Tensor(mul_93, mul_94);  mul_93 = mul_94 = None
        unsqueeze_48: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_46, -1)
        unsqueeze_49: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
        unsqueeze_50: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_47, -1);  primals_47 = None
        unsqueeze_51: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
        mul_95: "f32[64, 240, 14, 14]" = torch.ops.aten.mul.Tensor(mul_89, unsqueeze_49);  mul_89 = unsqueeze_49 = None
        add_70: "f32[64, 240, 14, 14]" = torch.ops.aten.add.Tensor(mul_95, unsqueeze_51);  mul_95 = unsqueeze_51 = None
        clone_105: "f32[64, 240, 14, 14]" = torch.ops.aten.clone.default(add_70)
        add_71: "f32[64, 240, 14, 14]" = torch.ops.aten.add.Tensor(add_70, 3)
        clamp_min_5: "f32[64, 240, 14, 14]" = torch.ops.aten.clamp_min.default(add_71, 0);  add_71 = None
        clamp_max_5: "f32[64, 240, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_5, 6);  clamp_min_5 = None
        mul_96: "f32[64, 240, 14, 14]" = torch.ops.aten.mul.Tensor(add_70, clamp_max_5);  add_70 = clamp_max_5 = None
        div_5: "f32[64, 240, 14, 14]" = torch.ops.aten.div.Tensor(mul_96, 6);  mul_96 = None
        convolution_17: "f32[64, 240, 14, 14]" = torch.ops.aten.convolution.default(div_5, primals_48, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 240)
        add_72: "i64[]" = torch.ops.aten.add.Tensor(clone_41, 1);  clone_41 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(convolution_17, [0, 2, 3], correction = 0, keepdim = True)
        getitem_26: "f32[1, 240, 1, 1]" = var_mean_13[0]
        getitem_27: "f32[1, 240, 1, 1]" = var_mean_13[1];  var_mean_13 = None
        add_73: "f32[1, 240, 1, 1]" = torch.ops.aten.add.Tensor(getitem_26, 0.001)
        rsqrt_13: "f32[1, 240, 1, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
        sub_13: "f32[64, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_17, getitem_27)
        mul_97: "f32[64, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
        squeeze_39: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
        squeeze_40: "f32[240]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
        mul_98: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_39, 0.01)
        mul_99: "f32[240]" = torch.ops.aten.mul.Tensor(clone_39, 0.99);  clone_39 = None
        add_74: "f32[240]" = torch.ops.aten.add.Tensor(mul_98, mul_99);  mul_98 = mul_99 = None
        squeeze_41: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
        mul_100: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0000797257434426);  squeeze_41 = None
        mul_101: "f32[240]" = torch.ops.aten.mul.Tensor(mul_100, 0.01);  mul_100 = None
        mul_102: "f32[240]" = torch.ops.aten.mul.Tensor(clone_40, 0.99);  clone_40 = None
        add_75: "f32[240]" = torch.ops.aten.add.Tensor(mul_101, mul_102);  mul_101 = mul_102 = None
        unsqueeze_52: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_49, -1)
        unsqueeze_53: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        unsqueeze_54: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_50, -1);  primals_50 = None
        unsqueeze_55: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        mul_103: "f32[64, 240, 14, 14]" = torch.ops.aten.mul.Tensor(mul_97, unsqueeze_53);  mul_97 = unsqueeze_53 = None
        add_76: "f32[64, 240, 14, 14]" = torch.ops.aten.add.Tensor(mul_103, unsqueeze_55);  mul_103 = unsqueeze_55 = None
        clone_106: "f32[64, 240, 14, 14]" = torch.ops.aten.clone.default(add_76)
        add_77: "f32[64, 240, 14, 14]" = torch.ops.aten.add.Tensor(add_76, 3)
        clamp_min_6: "f32[64, 240, 14, 14]" = torch.ops.aten.clamp_min.default(add_77, 0);  add_77 = None
        clamp_max_6: "f32[64, 240, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_6, 6);  clamp_min_6 = None
        mul_104: "f32[64, 240, 14, 14]" = torch.ops.aten.mul.Tensor(add_76, clamp_max_6);  add_76 = clamp_max_6 = None
        div_6: "f32[64, 240, 14, 14]" = torch.ops.aten.div.Tensor(mul_104, 6);  mul_104 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
        mean_2: "f32[64, 240, 1, 1]" = torch.ops.aten.mean.dim(div_6, [-1, -2], True)
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
        convolution_18: "f32[64, 64, 1, 1]" = torch.ops.aten.convolution.default(mean_2, primals_51, primals_52, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_52 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
        relu_7: "f32[64, 64, 1, 1]" = torch.ops.aten.relu.default(convolution_18);  convolution_18 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
        convolution_19: "f32[64, 240, 1, 1]" = torch.ops.aten.convolution.default(relu_7, primals_53, primals_54, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_54 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
        add_78: "f32[64, 240, 1, 1]" = torch.ops.aten.add.Tensor(convolution_19, 3)
        clamp_min_7: "f32[64, 240, 1, 1]" = torch.ops.aten.clamp_min.default(add_78, 0);  add_78 = None
        clamp_max_7: "f32[64, 240, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_7, 6);  clamp_min_7 = None
        div_7: "f32[64, 240, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_7, 6);  clamp_max_7 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:260, code: return scale * input
        mul_105: "f32[64, 240, 14, 14]" = torch.ops.aten.mul.Tensor(div_7, div_6)
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
        convolution_20: "f32[64, 40, 14, 14]" = torch.ops.aten.convolution.default(mul_105, primals_55, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_79: "i64[]" = torch.ops.aten.add.Tensor(clone_44, 1);  clone_44 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(convolution_20, [0, 2, 3], correction = 0, keepdim = True)
        getitem_28: "f32[1, 40, 1, 1]" = var_mean_14[0]
        getitem_29: "f32[1, 40, 1, 1]" = var_mean_14[1];  var_mean_14 = None
        add_80: "f32[1, 40, 1, 1]" = torch.ops.aten.add.Tensor(getitem_28, 0.001)
        rsqrt_14: "f32[1, 40, 1, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        sub_14: "f32[64, 40, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_20, getitem_29)
        mul_106: "f32[64, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
        squeeze_42: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
        squeeze_43: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
        mul_107: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_42, 0.01)
        mul_108: "f32[40]" = torch.ops.aten.mul.Tensor(clone_42, 0.99);  clone_42 = None
        add_81: "f32[40]" = torch.ops.aten.add.Tensor(mul_107, mul_108);  mul_107 = mul_108 = None
        squeeze_44: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
        mul_109: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0000797257434426);  squeeze_44 = None
        mul_110: "f32[40]" = torch.ops.aten.mul.Tensor(mul_109, 0.01);  mul_109 = None
        mul_111: "f32[40]" = torch.ops.aten.mul.Tensor(clone_43, 0.99);  clone_43 = None
        add_82: "f32[40]" = torch.ops.aten.add.Tensor(mul_110, mul_111);  mul_110 = mul_111 = None
        unsqueeze_56: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1)
        unsqueeze_57: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        unsqueeze_58: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1);  primals_57 = None
        unsqueeze_59: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
        mul_112: "f32[64, 40, 14, 14]" = torch.ops.aten.mul.Tensor(mul_106, unsqueeze_57);  mul_106 = unsqueeze_57 = None
        add_83: "f32[64, 40, 14, 14]" = torch.ops.aten.add.Tensor(mul_112, unsqueeze_59);  mul_112 = unsqueeze_59 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:113, code: result += input
        add_84: "f32[64, 40, 14, 14]" = torch.ops.aten.add.Tensor(add_83, add_65);  add_83 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
        convolution_21: "f32[64, 240, 14, 14]" = torch.ops.aten.convolution.default(add_84, primals_58, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_85: "i64[]" = torch.ops.aten.add.Tensor(clone_47, 1);  clone_47 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(convolution_21, [0, 2, 3], correction = 0, keepdim = True)
        getitem_30: "f32[1, 240, 1, 1]" = var_mean_15[0]
        getitem_31: "f32[1, 240, 1, 1]" = var_mean_15[1];  var_mean_15 = None
        add_86: "f32[1, 240, 1, 1]" = torch.ops.aten.add.Tensor(getitem_30, 0.001)
        rsqrt_15: "f32[1, 240, 1, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        sub_15: "f32[64, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_21, getitem_31)
        mul_113: "f32[64, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
        squeeze_45: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
        squeeze_46: "f32[240]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
        mul_114: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_45, 0.01)
        mul_115: "f32[240]" = torch.ops.aten.mul.Tensor(clone_45, 0.99);  clone_45 = None
        add_87: "f32[240]" = torch.ops.aten.add.Tensor(mul_114, mul_115);  mul_114 = mul_115 = None
        squeeze_47: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
        mul_116: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0000797257434426);  squeeze_47 = None
        mul_117: "f32[240]" = torch.ops.aten.mul.Tensor(mul_116, 0.01);  mul_116 = None
        mul_118: "f32[240]" = torch.ops.aten.mul.Tensor(clone_46, 0.99);  clone_46 = None
        add_88: "f32[240]" = torch.ops.aten.add.Tensor(mul_117, mul_118);  mul_117 = mul_118 = None
        unsqueeze_60: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_59, -1)
        unsqueeze_61: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
        unsqueeze_62: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_60, -1);  primals_60 = None
        unsqueeze_63: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
        mul_119: "f32[64, 240, 14, 14]" = torch.ops.aten.mul.Tensor(mul_113, unsqueeze_61);  mul_113 = unsqueeze_61 = None
        add_89: "f32[64, 240, 14, 14]" = torch.ops.aten.add.Tensor(mul_119, unsqueeze_63);  mul_119 = unsqueeze_63 = None
        clone_107: "f32[64, 240, 14, 14]" = torch.ops.aten.clone.default(add_89)
        add_90: "f32[64, 240, 14, 14]" = torch.ops.aten.add.Tensor(add_89, 3)
        clamp_min_8: "f32[64, 240, 14, 14]" = torch.ops.aten.clamp_min.default(add_90, 0);  add_90 = None
        clamp_max_8: "f32[64, 240, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_8, 6);  clamp_min_8 = None
        mul_120: "f32[64, 240, 14, 14]" = torch.ops.aten.mul.Tensor(add_89, clamp_max_8);  add_89 = clamp_max_8 = None
        div_8: "f32[64, 240, 14, 14]" = torch.ops.aten.div.Tensor(mul_120, 6);  mul_120 = None
        convolution_22: "f32[64, 240, 14, 14]" = torch.ops.aten.convolution.default(div_8, primals_61, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 240)
        add_91: "i64[]" = torch.ops.aten.add.Tensor(clone_50, 1);  clone_50 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(convolution_22, [0, 2, 3], correction = 0, keepdim = True)
        getitem_32: "f32[1, 240, 1, 1]" = var_mean_16[0]
        getitem_33: "f32[1, 240, 1, 1]" = var_mean_16[1];  var_mean_16 = None
        add_92: "f32[1, 240, 1, 1]" = torch.ops.aten.add.Tensor(getitem_32, 0.001)
        rsqrt_16: "f32[1, 240, 1, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        sub_16: "f32[64, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_22, getitem_33)
        mul_121: "f32[64, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
        squeeze_48: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
        squeeze_49: "f32[240]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
        mul_122: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_48, 0.01)
        mul_123: "f32[240]" = torch.ops.aten.mul.Tensor(clone_48, 0.99);  clone_48 = None
        add_93: "f32[240]" = torch.ops.aten.add.Tensor(mul_122, mul_123);  mul_122 = mul_123 = None
        squeeze_50: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
        mul_124: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0000797257434426);  squeeze_50 = None
        mul_125: "f32[240]" = torch.ops.aten.mul.Tensor(mul_124, 0.01);  mul_124 = None
        mul_126: "f32[240]" = torch.ops.aten.mul.Tensor(clone_49, 0.99);  clone_49 = None
        add_94: "f32[240]" = torch.ops.aten.add.Tensor(mul_125, mul_126);  mul_125 = mul_126 = None
        unsqueeze_64: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_62, -1)
        unsqueeze_65: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        unsqueeze_66: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_63, -1);  primals_63 = None
        unsqueeze_67: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        mul_127: "f32[64, 240, 14, 14]" = torch.ops.aten.mul.Tensor(mul_121, unsqueeze_65);  mul_121 = unsqueeze_65 = None
        add_95: "f32[64, 240, 14, 14]" = torch.ops.aten.add.Tensor(mul_127, unsqueeze_67);  mul_127 = unsqueeze_67 = None
        clone_108: "f32[64, 240, 14, 14]" = torch.ops.aten.clone.default(add_95)
        add_96: "f32[64, 240, 14, 14]" = torch.ops.aten.add.Tensor(add_95, 3)
        clamp_min_9: "f32[64, 240, 14, 14]" = torch.ops.aten.clamp_min.default(add_96, 0);  add_96 = None
        clamp_max_9: "f32[64, 240, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_9, 6);  clamp_min_9 = None
        mul_128: "f32[64, 240, 14, 14]" = torch.ops.aten.mul.Tensor(add_95, clamp_max_9);  add_95 = clamp_max_9 = None
        div_9: "f32[64, 240, 14, 14]" = torch.ops.aten.div.Tensor(mul_128, 6);  mul_128 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
        mean_3: "f32[64, 240, 1, 1]" = torch.ops.aten.mean.dim(div_9, [-1, -2], True)
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
        convolution_23: "f32[64, 64, 1, 1]" = torch.ops.aten.convolution.default(mean_3, primals_64, primals_65, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_65 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
        relu_8: "f32[64, 64, 1, 1]" = torch.ops.aten.relu.default(convolution_23);  convolution_23 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
        convolution_24: "f32[64, 240, 1, 1]" = torch.ops.aten.convolution.default(relu_8, primals_66, primals_67, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_67 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
        add_97: "f32[64, 240, 1, 1]" = torch.ops.aten.add.Tensor(convolution_24, 3)
        clamp_min_10: "f32[64, 240, 1, 1]" = torch.ops.aten.clamp_min.default(add_97, 0);  add_97 = None
        clamp_max_10: "f32[64, 240, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_10, 6);  clamp_min_10 = None
        div_10: "f32[64, 240, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_10, 6);  clamp_max_10 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:260, code: return scale * input
        mul_129: "f32[64, 240, 14, 14]" = torch.ops.aten.mul.Tensor(div_10, div_9)
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
        convolution_25: "f32[64, 40, 14, 14]" = torch.ops.aten.convolution.default(mul_129, primals_68, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_98: "i64[]" = torch.ops.aten.add.Tensor(clone_53, 1);  clone_53 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(convolution_25, [0, 2, 3], correction = 0, keepdim = True)
        getitem_34: "f32[1, 40, 1, 1]" = var_mean_17[0]
        getitem_35: "f32[1, 40, 1, 1]" = var_mean_17[1];  var_mean_17 = None
        add_99: "f32[1, 40, 1, 1]" = torch.ops.aten.add.Tensor(getitem_34, 0.001)
        rsqrt_17: "f32[1, 40, 1, 1]" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
        sub_17: "f32[64, 40, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, getitem_35)
        mul_130: "f32[64, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
        squeeze_51: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
        squeeze_52: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
        mul_131: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_51, 0.01)
        mul_132: "f32[40]" = torch.ops.aten.mul.Tensor(clone_51, 0.99);  clone_51 = None
        add_100: "f32[40]" = torch.ops.aten.add.Tensor(mul_131, mul_132);  mul_131 = mul_132 = None
        squeeze_53: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
        mul_133: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0000797257434426);  squeeze_53 = None
        mul_134: "f32[40]" = torch.ops.aten.mul.Tensor(mul_133, 0.01);  mul_133 = None
        mul_135: "f32[40]" = torch.ops.aten.mul.Tensor(clone_52, 0.99);  clone_52 = None
        add_101: "f32[40]" = torch.ops.aten.add.Tensor(mul_134, mul_135);  mul_134 = mul_135 = None
        unsqueeze_68: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_69, -1)
        unsqueeze_69: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
        unsqueeze_70: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_70, -1);  primals_70 = None
        unsqueeze_71: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
        mul_136: "f32[64, 40, 14, 14]" = torch.ops.aten.mul.Tensor(mul_130, unsqueeze_69);  mul_130 = unsqueeze_69 = None
        add_102: "f32[64, 40, 14, 14]" = torch.ops.aten.add.Tensor(mul_136, unsqueeze_71);  mul_136 = unsqueeze_71 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:113, code: result += input
        add_103: "f32[64, 40, 14, 14]" = torch.ops.aten.add.Tensor(add_102, add_84);  add_102 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
        convolution_26: "f32[64, 120, 14, 14]" = torch.ops.aten.convolution.default(add_103, primals_71, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_104: "i64[]" = torch.ops.aten.add.Tensor(clone_56, 1);  clone_56 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(convolution_26, [0, 2, 3], correction = 0, keepdim = True)
        getitem_36: "f32[1, 120, 1, 1]" = var_mean_18[0]
        getitem_37: "f32[1, 120, 1, 1]" = var_mean_18[1];  var_mean_18 = None
        add_105: "f32[1, 120, 1, 1]" = torch.ops.aten.add.Tensor(getitem_36, 0.001)
        rsqrt_18: "f32[1, 120, 1, 1]" = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
        sub_18: "f32[64, 120, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, getitem_37)
        mul_137: "f32[64, 120, 14, 14]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
        squeeze_54: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
        squeeze_55: "f32[120]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
        mul_138: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_54, 0.01)
        mul_139: "f32[120]" = torch.ops.aten.mul.Tensor(clone_54, 0.99);  clone_54 = None
        add_106: "f32[120]" = torch.ops.aten.add.Tensor(mul_138, mul_139);  mul_138 = mul_139 = None
        squeeze_56: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
        mul_140: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0000797257434426);  squeeze_56 = None
        mul_141: "f32[120]" = torch.ops.aten.mul.Tensor(mul_140, 0.01);  mul_140 = None
        mul_142: "f32[120]" = torch.ops.aten.mul.Tensor(clone_55, 0.99);  clone_55 = None
        add_107: "f32[120]" = torch.ops.aten.add.Tensor(mul_141, mul_142);  mul_141 = mul_142 = None
        unsqueeze_72: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_72, -1)
        unsqueeze_73: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
        unsqueeze_74: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_73, -1);  primals_73 = None
        unsqueeze_75: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
        mul_143: "f32[64, 120, 14, 14]" = torch.ops.aten.mul.Tensor(mul_137, unsqueeze_73);  mul_137 = unsqueeze_73 = None
        add_108: "f32[64, 120, 14, 14]" = torch.ops.aten.add.Tensor(mul_143, unsqueeze_75);  mul_143 = unsqueeze_75 = None
        clone_109: "f32[64, 120, 14, 14]" = torch.ops.aten.clone.default(add_108)
        add_109: "f32[64, 120, 14, 14]" = torch.ops.aten.add.Tensor(add_108, 3)
        clamp_min_11: "f32[64, 120, 14, 14]" = torch.ops.aten.clamp_min.default(add_109, 0);  add_109 = None
        clamp_max_11: "f32[64, 120, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_11, 6);  clamp_min_11 = None
        mul_144: "f32[64, 120, 14, 14]" = torch.ops.aten.mul.Tensor(add_108, clamp_max_11);  add_108 = clamp_max_11 = None
        div_11: "f32[64, 120, 14, 14]" = torch.ops.aten.div.Tensor(mul_144, 6);  mul_144 = None
        convolution_27: "f32[64, 120, 14, 14]" = torch.ops.aten.convolution.default(div_11, primals_74, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120)
        add_110: "i64[]" = torch.ops.aten.add.Tensor(clone_59, 1);  clone_59 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(convolution_27, [0, 2, 3], correction = 0, keepdim = True)
        getitem_38: "f32[1, 120, 1, 1]" = var_mean_19[0]
        getitem_39: "f32[1, 120, 1, 1]" = var_mean_19[1];  var_mean_19 = None
        add_111: "f32[1, 120, 1, 1]" = torch.ops.aten.add.Tensor(getitem_38, 0.001)
        rsqrt_19: "f32[1, 120, 1, 1]" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
        sub_19: "f32[64, 120, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_27, getitem_39)
        mul_145: "f32[64, 120, 14, 14]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
        squeeze_57: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
        squeeze_58: "f32[120]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
        mul_146: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_57, 0.01)
        mul_147: "f32[120]" = torch.ops.aten.mul.Tensor(clone_57, 0.99);  clone_57 = None
        add_112: "f32[120]" = torch.ops.aten.add.Tensor(mul_146, mul_147);  mul_146 = mul_147 = None
        squeeze_59: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
        mul_148: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0000797257434426);  squeeze_59 = None
        mul_149: "f32[120]" = torch.ops.aten.mul.Tensor(mul_148, 0.01);  mul_148 = None
        mul_150: "f32[120]" = torch.ops.aten.mul.Tensor(clone_58, 0.99);  clone_58 = None
        add_113: "f32[120]" = torch.ops.aten.add.Tensor(mul_149, mul_150);  mul_149 = mul_150 = None
        unsqueeze_76: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_75, -1)
        unsqueeze_77: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        unsqueeze_78: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_76, -1);  primals_76 = None
        unsqueeze_79: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        mul_151: "f32[64, 120, 14, 14]" = torch.ops.aten.mul.Tensor(mul_145, unsqueeze_77);  mul_145 = unsqueeze_77 = None
        add_114: "f32[64, 120, 14, 14]" = torch.ops.aten.add.Tensor(mul_151, unsqueeze_79);  mul_151 = unsqueeze_79 = None
        clone_110: "f32[64, 120, 14, 14]" = torch.ops.aten.clone.default(add_114)
        add_115: "f32[64, 120, 14, 14]" = torch.ops.aten.add.Tensor(add_114, 3)
        clamp_min_12: "f32[64, 120, 14, 14]" = torch.ops.aten.clamp_min.default(add_115, 0);  add_115 = None
        clamp_max_12: "f32[64, 120, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_12, 6);  clamp_min_12 = None
        mul_152: "f32[64, 120, 14, 14]" = torch.ops.aten.mul.Tensor(add_114, clamp_max_12);  add_114 = clamp_max_12 = None
        div_12: "f32[64, 120, 14, 14]" = torch.ops.aten.div.Tensor(mul_152, 6);  mul_152 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
        mean_4: "f32[64, 120, 1, 1]" = torch.ops.aten.mean.dim(div_12, [-1, -2], True)
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
        convolution_28: "f32[64, 32, 1, 1]" = torch.ops.aten.convolution.default(mean_4, primals_77, primals_78, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_78 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
        relu_9: "f32[64, 32, 1, 1]" = torch.ops.aten.relu.default(convolution_28);  convolution_28 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
        convolution_29: "f32[64, 120, 1, 1]" = torch.ops.aten.convolution.default(relu_9, primals_79, primals_80, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_80 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
        add_116: "f32[64, 120, 1, 1]" = torch.ops.aten.add.Tensor(convolution_29, 3)
        clamp_min_13: "f32[64, 120, 1, 1]" = torch.ops.aten.clamp_min.default(add_116, 0);  add_116 = None
        clamp_max_13: "f32[64, 120, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_13, 6);  clamp_min_13 = None
        div_13: "f32[64, 120, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_13, 6);  clamp_max_13 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:260, code: return scale * input
        mul_153: "f32[64, 120, 14, 14]" = torch.ops.aten.mul.Tensor(div_13, div_12)
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
        convolution_30: "f32[64, 48, 14, 14]" = torch.ops.aten.convolution.default(mul_153, primals_81, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_117: "i64[]" = torch.ops.aten.add.Tensor(clone_62, 1);  clone_62 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(convolution_30, [0, 2, 3], correction = 0, keepdim = True)
        getitem_40: "f32[1, 48, 1, 1]" = var_mean_20[0]
        getitem_41: "f32[1, 48, 1, 1]" = var_mean_20[1];  var_mean_20 = None
        add_118: "f32[1, 48, 1, 1]" = torch.ops.aten.add.Tensor(getitem_40, 0.001)
        rsqrt_20: "f32[1, 48, 1, 1]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
        sub_20: "f32[64, 48, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, getitem_41)
        mul_154: "f32[64, 48, 14, 14]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
        squeeze_60: "f32[48]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
        squeeze_61: "f32[48]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
        mul_155: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_60, 0.01)
        mul_156: "f32[48]" = torch.ops.aten.mul.Tensor(clone_60, 0.99);  clone_60 = None
        add_119: "f32[48]" = torch.ops.aten.add.Tensor(mul_155, mul_156);  mul_155 = mul_156 = None
        squeeze_62: "f32[48]" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
        mul_157: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_62, 1.0000797257434426);  squeeze_62 = None
        mul_158: "f32[48]" = torch.ops.aten.mul.Tensor(mul_157, 0.01);  mul_157 = None
        mul_159: "f32[48]" = torch.ops.aten.mul.Tensor(clone_61, 0.99);  clone_61 = None
        add_120: "f32[48]" = torch.ops.aten.add.Tensor(mul_158, mul_159);  mul_158 = mul_159 = None
        unsqueeze_80: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(primals_82, -1)
        unsqueeze_81: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
        unsqueeze_82: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(primals_83, -1);  primals_83 = None
        unsqueeze_83: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
        mul_160: "f32[64, 48, 14, 14]" = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_81);  mul_154 = unsqueeze_81 = None
        add_121: "f32[64, 48, 14, 14]" = torch.ops.aten.add.Tensor(mul_160, unsqueeze_83);  mul_160 = unsqueeze_83 = None
        convolution_31: "f32[64, 144, 14, 14]" = torch.ops.aten.convolution.default(add_121, primals_84, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_122: "i64[]" = torch.ops.aten.add.Tensor(clone_65, 1);  clone_65 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(convolution_31, [0, 2, 3], correction = 0, keepdim = True)
        getitem_42: "f32[1, 144, 1, 1]" = var_mean_21[0]
        getitem_43: "f32[1, 144, 1, 1]" = var_mean_21[1];  var_mean_21 = None
        add_123: "f32[1, 144, 1, 1]" = torch.ops.aten.add.Tensor(getitem_42, 0.001)
        rsqrt_21: "f32[1, 144, 1, 1]" = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
        sub_21: "f32[64, 144, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, getitem_43)
        mul_161: "f32[64, 144, 14, 14]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
        squeeze_63: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
        squeeze_64: "f32[144]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
        mul_162: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_63, 0.01)
        mul_163: "f32[144]" = torch.ops.aten.mul.Tensor(clone_63, 0.99);  clone_63 = None
        add_124: "f32[144]" = torch.ops.aten.add.Tensor(mul_162, mul_163);  mul_162 = mul_163 = None
        squeeze_65: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
        mul_164: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_65, 1.0000797257434426);  squeeze_65 = None
        mul_165: "f32[144]" = torch.ops.aten.mul.Tensor(mul_164, 0.01);  mul_164 = None
        mul_166: "f32[144]" = torch.ops.aten.mul.Tensor(clone_64, 0.99);  clone_64 = None
        add_125: "f32[144]" = torch.ops.aten.add.Tensor(mul_165, mul_166);  mul_165 = mul_166 = None
        unsqueeze_84: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_85, -1)
        unsqueeze_85: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
        unsqueeze_86: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_86, -1);  primals_86 = None
        unsqueeze_87: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
        mul_167: "f32[64, 144, 14, 14]" = torch.ops.aten.mul.Tensor(mul_161, unsqueeze_85);  mul_161 = unsqueeze_85 = None
        add_126: "f32[64, 144, 14, 14]" = torch.ops.aten.add.Tensor(mul_167, unsqueeze_87);  mul_167 = unsqueeze_87 = None
        clone_111: "f32[64, 144, 14, 14]" = torch.ops.aten.clone.default(add_126)
        add_127: "f32[64, 144, 14, 14]" = torch.ops.aten.add.Tensor(add_126, 3)
        clamp_min_14: "f32[64, 144, 14, 14]" = torch.ops.aten.clamp_min.default(add_127, 0);  add_127 = None
        clamp_max_14: "f32[64, 144, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_14, 6);  clamp_min_14 = None
        mul_168: "f32[64, 144, 14, 14]" = torch.ops.aten.mul.Tensor(add_126, clamp_max_14);  add_126 = clamp_max_14 = None
        div_14: "f32[64, 144, 14, 14]" = torch.ops.aten.div.Tensor(mul_168, 6);  mul_168 = None
        convolution_32: "f32[64, 144, 14, 14]" = torch.ops.aten.convolution.default(div_14, primals_87, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 144)
        add_128: "i64[]" = torch.ops.aten.add.Tensor(clone_68, 1);  clone_68 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(convolution_32, [0, 2, 3], correction = 0, keepdim = True)
        getitem_44: "f32[1, 144, 1, 1]" = var_mean_22[0]
        getitem_45: "f32[1, 144, 1, 1]" = var_mean_22[1];  var_mean_22 = None
        add_129: "f32[1, 144, 1, 1]" = torch.ops.aten.add.Tensor(getitem_44, 0.001)
        rsqrt_22: "f32[1, 144, 1, 1]" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
        sub_22: "f32[64, 144, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_32, getitem_45)
        mul_169: "f32[64, 144, 14, 14]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
        squeeze_66: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
        squeeze_67: "f32[144]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
        mul_170: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_66, 0.01)
        mul_171: "f32[144]" = torch.ops.aten.mul.Tensor(clone_66, 0.99);  clone_66 = None
        add_130: "f32[144]" = torch.ops.aten.add.Tensor(mul_170, mul_171);  mul_170 = mul_171 = None
        squeeze_68: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
        mul_172: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_68, 1.0000797257434426);  squeeze_68 = None
        mul_173: "f32[144]" = torch.ops.aten.mul.Tensor(mul_172, 0.01);  mul_172 = None
        mul_174: "f32[144]" = torch.ops.aten.mul.Tensor(clone_67, 0.99);  clone_67 = None
        add_131: "f32[144]" = torch.ops.aten.add.Tensor(mul_173, mul_174);  mul_173 = mul_174 = None
        unsqueeze_88: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_88, -1)
        unsqueeze_89: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
        unsqueeze_90: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_89, -1);  primals_89 = None
        unsqueeze_91: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
        mul_175: "f32[64, 144, 14, 14]" = torch.ops.aten.mul.Tensor(mul_169, unsqueeze_89);  mul_169 = unsqueeze_89 = None
        add_132: "f32[64, 144, 14, 14]" = torch.ops.aten.add.Tensor(mul_175, unsqueeze_91);  mul_175 = unsqueeze_91 = None
        clone_112: "f32[64, 144, 14, 14]" = torch.ops.aten.clone.default(add_132)
        add_133: "f32[64, 144, 14, 14]" = torch.ops.aten.add.Tensor(add_132, 3)
        clamp_min_15: "f32[64, 144, 14, 14]" = torch.ops.aten.clamp_min.default(add_133, 0);  add_133 = None
        clamp_max_15: "f32[64, 144, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_15, 6);  clamp_min_15 = None
        mul_176: "f32[64, 144, 14, 14]" = torch.ops.aten.mul.Tensor(add_132, clamp_max_15);  add_132 = clamp_max_15 = None
        div_15: "f32[64, 144, 14, 14]" = torch.ops.aten.div.Tensor(mul_176, 6);  mul_176 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
        mean_5: "f32[64, 144, 1, 1]" = torch.ops.aten.mean.dim(div_15, [-1, -2], True)
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
        convolution_33: "f32[64, 40, 1, 1]" = torch.ops.aten.convolution.default(mean_5, primals_90, primals_91, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_91 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
        relu_10: "f32[64, 40, 1, 1]" = torch.ops.aten.relu.default(convolution_33);  convolution_33 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
        convolution_34: "f32[64, 144, 1, 1]" = torch.ops.aten.convolution.default(relu_10, primals_92, primals_93, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_93 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
        add_134: "f32[64, 144, 1, 1]" = torch.ops.aten.add.Tensor(convolution_34, 3)
        clamp_min_16: "f32[64, 144, 1, 1]" = torch.ops.aten.clamp_min.default(add_134, 0);  add_134 = None
        clamp_max_16: "f32[64, 144, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_16, 6);  clamp_min_16 = None
        div_16: "f32[64, 144, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_16, 6);  clamp_max_16 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:260, code: return scale * input
        mul_177: "f32[64, 144, 14, 14]" = torch.ops.aten.mul.Tensor(div_16, div_15)
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
        convolution_35: "f32[64, 48, 14, 14]" = torch.ops.aten.convolution.default(mul_177, primals_94, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_135: "i64[]" = torch.ops.aten.add.Tensor(clone_71, 1);  clone_71 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(convolution_35, [0, 2, 3], correction = 0, keepdim = True)
        getitem_46: "f32[1, 48, 1, 1]" = var_mean_23[0]
        getitem_47: "f32[1, 48, 1, 1]" = var_mean_23[1];  var_mean_23 = None
        add_136: "f32[1, 48, 1, 1]" = torch.ops.aten.add.Tensor(getitem_46, 0.001)
        rsqrt_23: "f32[1, 48, 1, 1]" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
        sub_23: "f32[64, 48, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, getitem_47)
        mul_178: "f32[64, 48, 14, 14]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
        squeeze_69: "f32[48]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
        squeeze_70: "f32[48]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
        mul_179: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_69, 0.01)
        mul_180: "f32[48]" = torch.ops.aten.mul.Tensor(clone_69, 0.99);  clone_69 = None
        add_137: "f32[48]" = torch.ops.aten.add.Tensor(mul_179, mul_180);  mul_179 = mul_180 = None
        squeeze_71: "f32[48]" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
        mul_181: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_71, 1.0000797257434426);  squeeze_71 = None
        mul_182: "f32[48]" = torch.ops.aten.mul.Tensor(mul_181, 0.01);  mul_181 = None
        mul_183: "f32[48]" = torch.ops.aten.mul.Tensor(clone_70, 0.99);  clone_70 = None
        add_138: "f32[48]" = torch.ops.aten.add.Tensor(mul_182, mul_183);  mul_182 = mul_183 = None
        unsqueeze_92: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(primals_95, -1)
        unsqueeze_93: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
        unsqueeze_94: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(primals_96, -1);  primals_96 = None
        unsqueeze_95: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
        mul_184: "f32[64, 48, 14, 14]" = torch.ops.aten.mul.Tensor(mul_178, unsqueeze_93);  mul_178 = unsqueeze_93 = None
        add_139: "f32[64, 48, 14, 14]" = torch.ops.aten.add.Tensor(mul_184, unsqueeze_95);  mul_184 = unsqueeze_95 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:113, code: result += input
        add_140: "f32[64, 48, 14, 14]" = torch.ops.aten.add.Tensor(add_139, add_121);  add_139 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
        convolution_36: "f32[64, 288, 14, 14]" = torch.ops.aten.convolution.default(add_140, primals_97, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_141: "i64[]" = torch.ops.aten.add.Tensor(clone_74, 1);  clone_74 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(convolution_36, [0, 2, 3], correction = 0, keepdim = True)
        getitem_48: "f32[1, 288, 1, 1]" = var_mean_24[0]
        getitem_49: "f32[1, 288, 1, 1]" = var_mean_24[1];  var_mean_24 = None
        add_142: "f32[1, 288, 1, 1]" = torch.ops.aten.add.Tensor(getitem_48, 0.001)
        rsqrt_24: "f32[1, 288, 1, 1]" = torch.ops.aten.rsqrt.default(add_142);  add_142 = None
        sub_24: "f32[64, 288, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, getitem_49)
        mul_185: "f32[64, 288, 14, 14]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
        squeeze_72: "f32[288]" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
        squeeze_73: "f32[288]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
        mul_186: "f32[288]" = torch.ops.aten.mul.Tensor(squeeze_72, 0.01)
        mul_187: "f32[288]" = torch.ops.aten.mul.Tensor(clone_72, 0.99);  clone_72 = None
        add_143: "f32[288]" = torch.ops.aten.add.Tensor(mul_186, mul_187);  mul_186 = mul_187 = None
        squeeze_74: "f32[288]" = torch.ops.aten.squeeze.dims(getitem_48, [0, 2, 3]);  getitem_48 = None
        mul_188: "f32[288]" = torch.ops.aten.mul.Tensor(squeeze_74, 1.0000797257434426);  squeeze_74 = None
        mul_189: "f32[288]" = torch.ops.aten.mul.Tensor(mul_188, 0.01);  mul_188 = None
        mul_190: "f32[288]" = torch.ops.aten.mul.Tensor(clone_73, 0.99);  clone_73 = None
        add_144: "f32[288]" = torch.ops.aten.add.Tensor(mul_189, mul_190);  mul_189 = mul_190 = None
        unsqueeze_96: "f32[288, 1]" = torch.ops.aten.unsqueeze.default(primals_98, -1)
        unsqueeze_97: "f32[288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
        unsqueeze_98: "f32[288, 1]" = torch.ops.aten.unsqueeze.default(primals_99, -1);  primals_99 = None
        unsqueeze_99: "f32[288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
        mul_191: "f32[64, 288, 14, 14]" = torch.ops.aten.mul.Tensor(mul_185, unsqueeze_97);  mul_185 = unsqueeze_97 = None
        add_145: "f32[64, 288, 14, 14]" = torch.ops.aten.add.Tensor(mul_191, unsqueeze_99);  mul_191 = unsqueeze_99 = None
        clone_113: "f32[64, 288, 14, 14]" = torch.ops.aten.clone.default(add_145)
        add_146: "f32[64, 288, 14, 14]" = torch.ops.aten.add.Tensor(add_145, 3)
        clamp_min_17: "f32[64, 288, 14, 14]" = torch.ops.aten.clamp_min.default(add_146, 0);  add_146 = None
        clamp_max_17: "f32[64, 288, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_17, 6);  clamp_min_17 = None
        mul_192: "f32[64, 288, 14, 14]" = torch.ops.aten.mul.Tensor(add_145, clamp_max_17);  add_145 = clamp_max_17 = None
        div_17: "f32[64, 288, 14, 14]" = torch.ops.aten.div.Tensor(mul_192, 6);  mul_192 = None
        convolution_37: "f32[64, 288, 7, 7]" = torch.ops.aten.convolution.default(div_17, primals_100, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 288)
        add_147: "i64[]" = torch.ops.aten.add.Tensor(clone_77, 1);  clone_77 = None
        var_mean_25 = torch.ops.aten.var_mean.correction(convolution_37, [0, 2, 3], correction = 0, keepdim = True)
        getitem_50: "f32[1, 288, 1, 1]" = var_mean_25[0]
        getitem_51: "f32[1, 288, 1, 1]" = var_mean_25[1];  var_mean_25 = None
        add_148: "f32[1, 288, 1, 1]" = torch.ops.aten.add.Tensor(getitem_50, 0.001)
        rsqrt_25: "f32[1, 288, 1, 1]" = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
        sub_25: "f32[64, 288, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_37, getitem_51)
        mul_193: "f32[64, 288, 7, 7]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
        squeeze_75: "f32[288]" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
        squeeze_76: "f32[288]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
        mul_194: "f32[288]" = torch.ops.aten.mul.Tensor(squeeze_75, 0.01)
        mul_195: "f32[288]" = torch.ops.aten.mul.Tensor(clone_75, 0.99);  clone_75 = None
        add_149: "f32[288]" = torch.ops.aten.add.Tensor(mul_194, mul_195);  mul_194 = mul_195 = None
        squeeze_77: "f32[288]" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
        mul_196: "f32[288]" = torch.ops.aten.mul.Tensor(squeeze_77, 1.0003189792663476);  squeeze_77 = None
        mul_197: "f32[288]" = torch.ops.aten.mul.Tensor(mul_196, 0.01);  mul_196 = None
        mul_198: "f32[288]" = torch.ops.aten.mul.Tensor(clone_76, 0.99);  clone_76 = None
        add_150: "f32[288]" = torch.ops.aten.add.Tensor(mul_197, mul_198);  mul_197 = mul_198 = None
        unsqueeze_100: "f32[288, 1]" = torch.ops.aten.unsqueeze.default(primals_101, -1)
        unsqueeze_101: "f32[288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
        unsqueeze_102: "f32[288, 1]" = torch.ops.aten.unsqueeze.default(primals_102, -1);  primals_102 = None
        unsqueeze_103: "f32[288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
        mul_199: "f32[64, 288, 7, 7]" = torch.ops.aten.mul.Tensor(mul_193, unsqueeze_101);  mul_193 = unsqueeze_101 = None
        add_151: "f32[64, 288, 7, 7]" = torch.ops.aten.add.Tensor(mul_199, unsqueeze_103);  mul_199 = unsqueeze_103 = None
        clone_114: "f32[64, 288, 7, 7]" = torch.ops.aten.clone.default(add_151)
        add_152: "f32[64, 288, 7, 7]" = torch.ops.aten.add.Tensor(add_151, 3)
        clamp_min_18: "f32[64, 288, 7, 7]" = torch.ops.aten.clamp_min.default(add_152, 0);  add_152 = None
        clamp_max_18: "f32[64, 288, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_18, 6);  clamp_min_18 = None
        mul_200: "f32[64, 288, 7, 7]" = torch.ops.aten.mul.Tensor(add_151, clamp_max_18);  add_151 = clamp_max_18 = None
        div_18: "f32[64, 288, 7, 7]" = torch.ops.aten.div.Tensor(mul_200, 6);  mul_200 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
        mean_6: "f32[64, 288, 1, 1]" = torch.ops.aten.mean.dim(div_18, [-1, -2], True)
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
        convolution_38: "f32[64, 72, 1, 1]" = torch.ops.aten.convolution.default(mean_6, primals_103, primals_104, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_104 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
        relu_11: "f32[64, 72, 1, 1]" = torch.ops.aten.relu.default(convolution_38);  convolution_38 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
        convolution_39: "f32[64, 288, 1, 1]" = torch.ops.aten.convolution.default(relu_11, primals_105, primals_106, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_106 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
        add_153: "f32[64, 288, 1, 1]" = torch.ops.aten.add.Tensor(convolution_39, 3)
        clamp_min_19: "f32[64, 288, 1, 1]" = torch.ops.aten.clamp_min.default(add_153, 0);  add_153 = None
        clamp_max_19: "f32[64, 288, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_19, 6);  clamp_min_19 = None
        div_19: "f32[64, 288, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_19, 6);  clamp_max_19 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:260, code: return scale * input
        mul_201: "f32[64, 288, 7, 7]" = torch.ops.aten.mul.Tensor(div_19, div_18)
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
        convolution_40: "f32[64, 96, 7, 7]" = torch.ops.aten.convolution.default(mul_201, primals_107, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_154: "i64[]" = torch.ops.aten.add.Tensor(clone_80, 1);  clone_80 = None
        var_mean_26 = torch.ops.aten.var_mean.correction(convolution_40, [0, 2, 3], correction = 0, keepdim = True)
        getitem_52: "f32[1, 96, 1, 1]" = var_mean_26[0]
        getitem_53: "f32[1, 96, 1, 1]" = var_mean_26[1];  var_mean_26 = None
        add_155: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem_52, 0.001)
        rsqrt_26: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
        sub_26: "f32[64, 96, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_40, getitem_53)
        mul_202: "f32[64, 96, 7, 7]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
        squeeze_78: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
        squeeze_79: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
        mul_203: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_78, 0.01)
        mul_204: "f32[96]" = torch.ops.aten.mul.Tensor(clone_78, 0.99);  clone_78 = None
        add_156: "f32[96]" = torch.ops.aten.add.Tensor(mul_203, mul_204);  mul_203 = mul_204 = None
        squeeze_80: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
        mul_205: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_80, 1.0003189792663476);  squeeze_80 = None
        mul_206: "f32[96]" = torch.ops.aten.mul.Tensor(mul_205, 0.01);  mul_205 = None
        mul_207: "f32[96]" = torch.ops.aten.mul.Tensor(clone_79, 0.99);  clone_79 = None
        add_157: "f32[96]" = torch.ops.aten.add.Tensor(mul_206, mul_207);  mul_206 = mul_207 = None
        unsqueeze_104: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_108, -1)
        unsqueeze_105: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
        unsqueeze_106: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_109, -1);  primals_109 = None
        unsqueeze_107: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
        mul_208: "f32[64, 96, 7, 7]" = torch.ops.aten.mul.Tensor(mul_202, unsqueeze_105);  mul_202 = unsqueeze_105 = None
        add_158: "f32[64, 96, 7, 7]" = torch.ops.aten.add.Tensor(mul_208, unsqueeze_107);  mul_208 = unsqueeze_107 = None
        convolution_41: "f32[64, 576, 7, 7]" = torch.ops.aten.convolution.default(add_158, primals_110, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_159: "i64[]" = torch.ops.aten.add.Tensor(clone_83, 1);  clone_83 = None
        var_mean_27 = torch.ops.aten.var_mean.correction(convolution_41, [0, 2, 3], correction = 0, keepdim = True)
        getitem_54: "f32[1, 576, 1, 1]" = var_mean_27[0]
        getitem_55: "f32[1, 576, 1, 1]" = var_mean_27[1];  var_mean_27 = None
        add_160: "f32[1, 576, 1, 1]" = torch.ops.aten.add.Tensor(getitem_54, 0.001)
        rsqrt_27: "f32[1, 576, 1, 1]" = torch.ops.aten.rsqrt.default(add_160);  add_160 = None
        sub_27: "f32[64, 576, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_41, getitem_55)
        mul_209: "f32[64, 576, 7, 7]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
        squeeze_81: "f32[576]" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
        squeeze_82: "f32[576]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
        mul_210: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_81, 0.01)
        mul_211: "f32[576]" = torch.ops.aten.mul.Tensor(clone_81, 0.99);  clone_81 = None
        add_161: "f32[576]" = torch.ops.aten.add.Tensor(mul_210, mul_211);  mul_210 = mul_211 = None
        squeeze_83: "f32[576]" = torch.ops.aten.squeeze.dims(getitem_54, [0, 2, 3]);  getitem_54 = None
        mul_212: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_83, 1.0003189792663476);  squeeze_83 = None
        mul_213: "f32[576]" = torch.ops.aten.mul.Tensor(mul_212, 0.01);  mul_212 = None
        mul_214: "f32[576]" = torch.ops.aten.mul.Tensor(clone_82, 0.99);  clone_82 = None
        add_162: "f32[576]" = torch.ops.aten.add.Tensor(mul_213, mul_214);  mul_213 = mul_214 = None
        unsqueeze_108: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(primals_111, -1)
        unsqueeze_109: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
        unsqueeze_110: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(primals_112, -1);  primals_112 = None
        unsqueeze_111: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
        mul_215: "f32[64, 576, 7, 7]" = torch.ops.aten.mul.Tensor(mul_209, unsqueeze_109);  mul_209 = unsqueeze_109 = None
        add_163: "f32[64, 576, 7, 7]" = torch.ops.aten.add.Tensor(mul_215, unsqueeze_111);  mul_215 = unsqueeze_111 = None
        clone_115: "f32[64, 576, 7, 7]" = torch.ops.aten.clone.default(add_163)
        add_164: "f32[64, 576, 7, 7]" = torch.ops.aten.add.Tensor(add_163, 3)
        clamp_min_20: "f32[64, 576, 7, 7]" = torch.ops.aten.clamp_min.default(add_164, 0);  add_164 = None
        clamp_max_20: "f32[64, 576, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_20, 6);  clamp_min_20 = None
        mul_216: "f32[64, 576, 7, 7]" = torch.ops.aten.mul.Tensor(add_163, clamp_max_20);  add_163 = clamp_max_20 = None
        div_20: "f32[64, 576, 7, 7]" = torch.ops.aten.div.Tensor(mul_216, 6);  mul_216 = None
        convolution_42: "f32[64, 576, 7, 7]" = torch.ops.aten.convolution.default(div_20, primals_113, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 576)
        add_165: "i64[]" = torch.ops.aten.add.Tensor(clone_86, 1);  clone_86 = None
        var_mean_28 = torch.ops.aten.var_mean.correction(convolution_42, [0, 2, 3], correction = 0, keepdim = True)
        getitem_56: "f32[1, 576, 1, 1]" = var_mean_28[0]
        getitem_57: "f32[1, 576, 1, 1]" = var_mean_28[1];  var_mean_28 = None
        add_166: "f32[1, 576, 1, 1]" = torch.ops.aten.add.Tensor(getitem_56, 0.001)
        rsqrt_28: "f32[1, 576, 1, 1]" = torch.ops.aten.rsqrt.default(add_166);  add_166 = None
        sub_28: "f32[64, 576, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_42, getitem_57)
        mul_217: "f32[64, 576, 7, 7]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
        squeeze_84: "f32[576]" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
        squeeze_85: "f32[576]" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
        mul_218: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_84, 0.01)
        mul_219: "f32[576]" = torch.ops.aten.mul.Tensor(clone_84, 0.99);  clone_84 = None
        add_167: "f32[576]" = torch.ops.aten.add.Tensor(mul_218, mul_219);  mul_218 = mul_219 = None
        squeeze_86: "f32[576]" = torch.ops.aten.squeeze.dims(getitem_56, [0, 2, 3]);  getitem_56 = None
        mul_220: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_86, 1.0003189792663476);  squeeze_86 = None
        mul_221: "f32[576]" = torch.ops.aten.mul.Tensor(mul_220, 0.01);  mul_220 = None
        mul_222: "f32[576]" = torch.ops.aten.mul.Tensor(clone_85, 0.99);  clone_85 = None
        add_168: "f32[576]" = torch.ops.aten.add.Tensor(mul_221, mul_222);  mul_221 = mul_222 = None
        unsqueeze_112: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(primals_114, -1)
        unsqueeze_113: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
        unsqueeze_114: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(primals_115, -1);  primals_115 = None
        unsqueeze_115: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
        mul_223: "f32[64, 576, 7, 7]" = torch.ops.aten.mul.Tensor(mul_217, unsqueeze_113);  mul_217 = unsqueeze_113 = None
        add_169: "f32[64, 576, 7, 7]" = torch.ops.aten.add.Tensor(mul_223, unsqueeze_115);  mul_223 = unsqueeze_115 = None
        clone_116: "f32[64, 576, 7, 7]" = torch.ops.aten.clone.default(add_169)
        add_170: "f32[64, 576, 7, 7]" = torch.ops.aten.add.Tensor(add_169, 3)
        clamp_min_21: "f32[64, 576, 7, 7]" = torch.ops.aten.clamp_min.default(add_170, 0);  add_170 = None
        clamp_max_21: "f32[64, 576, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_21, 6);  clamp_min_21 = None
        mul_224: "f32[64, 576, 7, 7]" = torch.ops.aten.mul.Tensor(add_169, clamp_max_21);  add_169 = clamp_max_21 = None
        div_21: "f32[64, 576, 7, 7]" = torch.ops.aten.div.Tensor(mul_224, 6);  mul_224 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
        mean_7: "f32[64, 576, 1, 1]" = torch.ops.aten.mean.dim(div_21, [-1, -2], True)
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
        convolution_43: "f32[64, 144, 1, 1]" = torch.ops.aten.convolution.default(mean_7, primals_116, primals_117, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_117 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
        relu_12: "f32[64, 144, 1, 1]" = torch.ops.aten.relu.default(convolution_43);  convolution_43 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
        convolution_44: "f32[64, 576, 1, 1]" = torch.ops.aten.convolution.default(relu_12, primals_118, primals_119, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_119 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
        add_171: "f32[64, 576, 1, 1]" = torch.ops.aten.add.Tensor(convolution_44, 3)
        clamp_min_22: "f32[64, 576, 1, 1]" = torch.ops.aten.clamp_min.default(add_171, 0);  add_171 = None
        clamp_max_22: "f32[64, 576, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_22, 6);  clamp_min_22 = None
        div_22: "f32[64, 576, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_22, 6);  clamp_max_22 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:260, code: return scale * input
        mul_225: "f32[64, 576, 7, 7]" = torch.ops.aten.mul.Tensor(div_22, div_21)
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
        convolution_45: "f32[64, 96, 7, 7]" = torch.ops.aten.convolution.default(mul_225, primals_120, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_172: "i64[]" = torch.ops.aten.add.Tensor(clone_89, 1);  clone_89 = None
        var_mean_29 = torch.ops.aten.var_mean.correction(convolution_45, [0, 2, 3], correction = 0, keepdim = True)
        getitem_58: "f32[1, 96, 1, 1]" = var_mean_29[0]
        getitem_59: "f32[1, 96, 1, 1]" = var_mean_29[1];  var_mean_29 = None
        add_173: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem_58, 0.001)
        rsqrt_29: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_173);  add_173 = None
        sub_29: "f32[64, 96, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_45, getitem_59)
        mul_226: "f32[64, 96, 7, 7]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
        squeeze_87: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
        squeeze_88: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
        mul_227: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_87, 0.01)
        mul_228: "f32[96]" = torch.ops.aten.mul.Tensor(clone_87, 0.99);  clone_87 = None
        add_174: "f32[96]" = torch.ops.aten.add.Tensor(mul_227, mul_228);  mul_227 = mul_228 = None
        squeeze_89: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
        mul_229: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_89, 1.0003189792663476);  squeeze_89 = None
        mul_230: "f32[96]" = torch.ops.aten.mul.Tensor(mul_229, 0.01);  mul_229 = None
        mul_231: "f32[96]" = torch.ops.aten.mul.Tensor(clone_88, 0.99);  clone_88 = None
        add_175: "f32[96]" = torch.ops.aten.add.Tensor(mul_230, mul_231);  mul_230 = mul_231 = None
        unsqueeze_116: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_121, -1)
        unsqueeze_117: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
        unsqueeze_118: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_122, -1);  primals_122 = None
        unsqueeze_119: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
        mul_232: "f32[64, 96, 7, 7]" = torch.ops.aten.mul.Tensor(mul_226, unsqueeze_117);  mul_226 = unsqueeze_117 = None
        add_176: "f32[64, 96, 7, 7]" = torch.ops.aten.add.Tensor(mul_232, unsqueeze_119);  mul_232 = unsqueeze_119 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:113, code: result += input
        add_177: "f32[64, 96, 7, 7]" = torch.ops.aten.add.Tensor(add_176, add_158);  add_176 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
        convolution_46: "f32[64, 576, 7, 7]" = torch.ops.aten.convolution.default(add_177, primals_123, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_178: "i64[]" = torch.ops.aten.add.Tensor(clone_92, 1);  clone_92 = None
        var_mean_30 = torch.ops.aten.var_mean.correction(convolution_46, [0, 2, 3], correction = 0, keepdim = True)
        getitem_60: "f32[1, 576, 1, 1]" = var_mean_30[0]
        getitem_61: "f32[1, 576, 1, 1]" = var_mean_30[1];  var_mean_30 = None
        add_179: "f32[1, 576, 1, 1]" = torch.ops.aten.add.Tensor(getitem_60, 0.001)
        rsqrt_30: "f32[1, 576, 1, 1]" = torch.ops.aten.rsqrt.default(add_179);  add_179 = None
        sub_30: "f32[64, 576, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_46, getitem_61)
        mul_233: "f32[64, 576, 7, 7]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
        squeeze_90: "f32[576]" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
        squeeze_91: "f32[576]" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
        mul_234: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_90, 0.01)
        mul_235: "f32[576]" = torch.ops.aten.mul.Tensor(clone_90, 0.99);  clone_90 = None
        add_180: "f32[576]" = torch.ops.aten.add.Tensor(mul_234, mul_235);  mul_234 = mul_235 = None
        squeeze_92: "f32[576]" = torch.ops.aten.squeeze.dims(getitem_60, [0, 2, 3]);  getitem_60 = None
        mul_236: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_92, 1.0003189792663476);  squeeze_92 = None
        mul_237: "f32[576]" = torch.ops.aten.mul.Tensor(mul_236, 0.01);  mul_236 = None
        mul_238: "f32[576]" = torch.ops.aten.mul.Tensor(clone_91, 0.99);  clone_91 = None
        add_181: "f32[576]" = torch.ops.aten.add.Tensor(mul_237, mul_238);  mul_237 = mul_238 = None
        unsqueeze_120: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(primals_124, -1)
        unsqueeze_121: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
        unsqueeze_122: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(primals_125, -1);  primals_125 = None
        unsqueeze_123: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
        mul_239: "f32[64, 576, 7, 7]" = torch.ops.aten.mul.Tensor(mul_233, unsqueeze_121);  mul_233 = unsqueeze_121 = None
        add_182: "f32[64, 576, 7, 7]" = torch.ops.aten.add.Tensor(mul_239, unsqueeze_123);  mul_239 = unsqueeze_123 = None
        clone_117: "f32[64, 576, 7, 7]" = torch.ops.aten.clone.default(add_182)
        add_183: "f32[64, 576, 7, 7]" = torch.ops.aten.add.Tensor(add_182, 3)
        clamp_min_23: "f32[64, 576, 7, 7]" = torch.ops.aten.clamp_min.default(add_183, 0);  add_183 = None
        clamp_max_23: "f32[64, 576, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_23, 6);  clamp_min_23 = None
        mul_240: "f32[64, 576, 7, 7]" = torch.ops.aten.mul.Tensor(add_182, clamp_max_23);  add_182 = clamp_max_23 = None
        div_23: "f32[64, 576, 7, 7]" = torch.ops.aten.div.Tensor(mul_240, 6);  mul_240 = None
        convolution_47: "f32[64, 576, 7, 7]" = torch.ops.aten.convolution.default(div_23, primals_126, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 576)
        add_184: "i64[]" = torch.ops.aten.add.Tensor(clone_95, 1);  clone_95 = None
        var_mean_31 = torch.ops.aten.var_mean.correction(convolution_47, [0, 2, 3], correction = 0, keepdim = True)
        getitem_62: "f32[1, 576, 1, 1]" = var_mean_31[0]
        getitem_63: "f32[1, 576, 1, 1]" = var_mean_31[1];  var_mean_31 = None
        add_185: "f32[1, 576, 1, 1]" = torch.ops.aten.add.Tensor(getitem_62, 0.001)
        rsqrt_31: "f32[1, 576, 1, 1]" = torch.ops.aten.rsqrt.default(add_185);  add_185 = None
        sub_31: "f32[64, 576, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_47, getitem_63)
        mul_241: "f32[64, 576, 7, 7]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
        squeeze_93: "f32[576]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
        squeeze_94: "f32[576]" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
        mul_242: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_93, 0.01)
        mul_243: "f32[576]" = torch.ops.aten.mul.Tensor(clone_93, 0.99);  clone_93 = None
        add_186: "f32[576]" = torch.ops.aten.add.Tensor(mul_242, mul_243);  mul_242 = mul_243 = None
        squeeze_95: "f32[576]" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
        mul_244: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_95, 1.0003189792663476);  squeeze_95 = None
        mul_245: "f32[576]" = torch.ops.aten.mul.Tensor(mul_244, 0.01);  mul_244 = None
        mul_246: "f32[576]" = torch.ops.aten.mul.Tensor(clone_94, 0.99);  clone_94 = None
        add_187: "f32[576]" = torch.ops.aten.add.Tensor(mul_245, mul_246);  mul_245 = mul_246 = None
        unsqueeze_124: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(primals_127, -1)
        unsqueeze_125: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
        unsqueeze_126: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(primals_128, -1);  primals_128 = None
        unsqueeze_127: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
        mul_247: "f32[64, 576, 7, 7]" = torch.ops.aten.mul.Tensor(mul_241, unsqueeze_125);  mul_241 = unsqueeze_125 = None
        add_188: "f32[64, 576, 7, 7]" = torch.ops.aten.add.Tensor(mul_247, unsqueeze_127);  mul_247 = unsqueeze_127 = None
        clone_118: "f32[64, 576, 7, 7]" = torch.ops.aten.clone.default(add_188)
        add_189: "f32[64, 576, 7, 7]" = torch.ops.aten.add.Tensor(add_188, 3)
        clamp_min_24: "f32[64, 576, 7, 7]" = torch.ops.aten.clamp_min.default(add_189, 0);  add_189 = None
        clamp_max_24: "f32[64, 576, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_24, 6);  clamp_min_24 = None
        mul_248: "f32[64, 576, 7, 7]" = torch.ops.aten.mul.Tensor(add_188, clamp_max_24);  add_188 = clamp_max_24 = None
        div_24: "f32[64, 576, 7, 7]" = torch.ops.aten.div.Tensor(mul_248, 6);  mul_248 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
        mean_8: "f32[64, 576, 1, 1]" = torch.ops.aten.mean.dim(div_24, [-1, -2], True)
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
        convolution_48: "f32[64, 144, 1, 1]" = torch.ops.aten.convolution.default(mean_8, primals_129, primals_130, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_130 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
        relu_13: "f32[64, 144, 1, 1]" = torch.ops.aten.relu.default(convolution_48);  convolution_48 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
        convolution_49: "f32[64, 576, 1, 1]" = torch.ops.aten.convolution.default(relu_13, primals_131, primals_132, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_132 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
        add_190: "f32[64, 576, 1, 1]" = torch.ops.aten.add.Tensor(convolution_49, 3)
        clamp_min_25: "f32[64, 576, 1, 1]" = torch.ops.aten.clamp_min.default(add_190, 0);  add_190 = None
        clamp_max_25: "f32[64, 576, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_25, 6);  clamp_min_25 = None
        div_25: "f32[64, 576, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_25, 6);  clamp_max_25 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:260, code: return scale * input
        mul_249: "f32[64, 576, 7, 7]" = torch.ops.aten.mul.Tensor(div_25, div_24)
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
        convolution_50: "f32[64, 96, 7, 7]" = torch.ops.aten.convolution.default(mul_249, primals_133, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_191: "i64[]" = torch.ops.aten.add.Tensor(clone_98, 1);  clone_98 = None
        var_mean_32 = torch.ops.aten.var_mean.correction(convolution_50, [0, 2, 3], correction = 0, keepdim = True)
        getitem_64: "f32[1, 96, 1, 1]" = var_mean_32[0]
        getitem_65: "f32[1, 96, 1, 1]" = var_mean_32[1];  var_mean_32 = None
        add_192: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem_64, 0.001)
        rsqrt_32: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
        sub_32: "f32[64, 96, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_50, getitem_65)
        mul_250: "f32[64, 96, 7, 7]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
        squeeze_96: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
        squeeze_97: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
        mul_251: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_96, 0.01)
        mul_252: "f32[96]" = torch.ops.aten.mul.Tensor(clone_96, 0.99);  clone_96 = None
        add_193: "f32[96]" = torch.ops.aten.add.Tensor(mul_251, mul_252);  mul_251 = mul_252 = None
        squeeze_98: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
        mul_253: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_98, 1.0003189792663476);  squeeze_98 = None
        mul_254: "f32[96]" = torch.ops.aten.mul.Tensor(mul_253, 0.01);  mul_253 = None
        mul_255: "f32[96]" = torch.ops.aten.mul.Tensor(clone_97, 0.99);  clone_97 = None
        add_194: "f32[96]" = torch.ops.aten.add.Tensor(mul_254, mul_255);  mul_254 = mul_255 = None
        unsqueeze_128: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_134, -1)
        unsqueeze_129: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
        unsqueeze_130: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_135, -1);  primals_135 = None
        unsqueeze_131: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
        mul_256: "f32[64, 96, 7, 7]" = torch.ops.aten.mul.Tensor(mul_250, unsqueeze_129);  mul_250 = unsqueeze_129 = None
        add_195: "f32[64, 96, 7, 7]" = torch.ops.aten.add.Tensor(mul_256, unsqueeze_131);  mul_256 = unsqueeze_131 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:113, code: result += input
        add_196: "f32[64, 96, 7, 7]" = torch.ops.aten.add.Tensor(add_195, add_177);  add_195 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:210, code: x = self.features(x)
        convolution_51: "f32[64, 576, 7, 7]" = torch.ops.aten.convolution.default(add_196, primals_136, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_197: "i64[]" = torch.ops.aten.add.Tensor(clone_101, 1);  clone_101 = None
        var_mean_33 = torch.ops.aten.var_mean.correction(convolution_51, [0, 2, 3], correction = 0, keepdim = True)
        getitem_66: "f32[1, 576, 1, 1]" = var_mean_33[0]
        getitem_67: "f32[1, 576, 1, 1]" = var_mean_33[1];  var_mean_33 = None
        add_198: "f32[1, 576, 1, 1]" = torch.ops.aten.add.Tensor(getitem_66, 0.001)
        rsqrt_33: "f32[1, 576, 1, 1]" = torch.ops.aten.rsqrt.default(add_198);  add_198 = None
        sub_33: "f32[64, 576, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_51, getitem_67)
        mul_257: "f32[64, 576, 7, 7]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
        squeeze_99: "f32[576]" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
        squeeze_100: "f32[576]" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
        mul_258: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_99, 0.01)
        mul_259: "f32[576]" = torch.ops.aten.mul.Tensor(clone_99, 0.99);  clone_99 = None
        add_199: "f32[576]" = torch.ops.aten.add.Tensor(mul_258, mul_259);  mul_258 = mul_259 = None
        squeeze_101: "f32[576]" = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
        mul_260: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_101, 1.0003189792663476);  squeeze_101 = None
        mul_261: "f32[576]" = torch.ops.aten.mul.Tensor(mul_260, 0.01);  mul_260 = None
        mul_262: "f32[576]" = torch.ops.aten.mul.Tensor(clone_100, 0.99);  clone_100 = None
        add_200: "f32[576]" = torch.ops.aten.add.Tensor(mul_261, mul_262);  mul_261 = mul_262 = None
        unsqueeze_132: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(primals_137, -1)
        unsqueeze_133: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
        unsqueeze_134: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(primals_138, -1);  primals_138 = None
        unsqueeze_135: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
        mul_263: "f32[64, 576, 7, 7]" = torch.ops.aten.mul.Tensor(mul_257, unsqueeze_133);  mul_257 = unsqueeze_133 = None
        add_201: "f32[64, 576, 7, 7]" = torch.ops.aten.add.Tensor(mul_263, unsqueeze_135);  mul_263 = unsqueeze_135 = None
        clone_119: "f32[64, 576, 7, 7]" = torch.ops.aten.clone.default(add_201)
        add_202: "f32[64, 576, 7, 7]" = torch.ops.aten.add.Tensor(add_201, 3)
        clamp_min_26: "f32[64, 576, 7, 7]" = torch.ops.aten.clamp_min.default(add_202, 0);  add_202 = None
        clamp_max_26: "f32[64, 576, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_26, 6);  clamp_min_26 = None
        mul_264: "f32[64, 576, 7, 7]" = torch.ops.aten.mul.Tensor(add_201, clamp_max_26);  add_201 = clamp_max_26 = None
        div_26: "f32[64, 576, 7, 7]" = torch.ops.aten.div.Tensor(mul_264, 6);  mul_264 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:212, code: x = self.avgpool(x)
        mean_9: "f32[64, 576, 1, 1]" = torch.ops.aten.mean.dim(div_26, [-1, -2], True);  div_26 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:213, code: x = torch.flatten(x, 1)
        view: "f32[64, 576]" = torch.ops.aten.view.default(mean_9, [64, 576]);  mean_9 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:215, code: x = self.classifier(x)
        permute: "f32[576, 1024]" = torch.ops.aten.permute.default(primals_139, [1, 0]);  primals_139 = None
        addmm: "f32[64, 1024]" = torch.ops.aten.addmm.default(primals_140, view, permute);  primals_140 = None
        add_203: "f32[64, 1024]" = torch.ops.aten.add.Tensor(addmm, 3)
        clamp_min_27: "f32[64, 1024]" = torch.ops.aten.clamp_min.default(add_203, 0);  add_203 = None
        clamp_max_27: "f32[64, 1024]" = torch.ops.aten.clamp_max.default(clamp_min_27, 6);  clamp_min_27 = None
        mul_265: "f32[64, 1024]" = torch.ops.aten.mul.Tensor(addmm, clamp_max_27);  clamp_max_27 = None
        div_27: "f32[64, 1024]" = torch.ops.aten.div.Tensor(mul_265, 6);  mul_265 = None
        philox_seed_like: "i32[]" = torch.ops.prims.philox_seed_like.default(div_27)
        philox_rand_like: "f32[64, 1024]" = torch.ops.prims.philox_rand_like.default(div_27, philox_seed_like, 0)
        gt: "b8[64, 1024]" = torch.ops.aten.gt.Scalar(philox_rand_like, 0.2);  philox_rand_like = None
        convert_element_type: "f32[64, 1024]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_266: "f32[64, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type, div_27);  convert_element_type = div_27 = None
        mul_267: "f32[64, 1024]" = torch.ops.aten.mul.Tensor(mul_266, 1.25);  mul_266 = None
        permute_1: "f32[1024, 1000]" = torch.ops.aten.permute.default(primals_141, [1, 0]);  primals_141 = None
        addmm_1: "f32[64, 1000]" = torch.ops.aten.addmm.default(primals_142, mul_267, permute_1);  primals_142 = None
        permute_2: "f32[1000, 1024]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        permute_6: "f32[1024, 576]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:210, code: x = self.features(x)
        unsqueeze_136: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(squeeze_99, 0);  squeeze_99 = None
        unsqueeze_137: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, 2);  unsqueeze_136 = None
        unsqueeze_138: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_137, 3);  unsqueeze_137 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
        unsqueeze_148: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_96, 0);  squeeze_96 = None
        unsqueeze_149: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, 2);  unsqueeze_148 = None
        unsqueeze_150: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_149, 3);  unsqueeze_149 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
        gt_2: "b8[64, 576, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_49, -3.0)
        lt_2: "b8[64, 576, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_49, 3.0);  convolution_49 = None
        bitwise_and: "b8[64, 576, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_2, lt_2);  gt_2 = lt_2 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
        unsqueeze_160: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
        unsqueeze_161: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, 2);  unsqueeze_160 = None
        unsqueeze_162: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_161, 3);  unsqueeze_161 = None
        unsqueeze_172: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
        unsqueeze_173: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, 2);  unsqueeze_172 = None
        unsqueeze_174: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_173, 3);  unsqueeze_173 = None
        unsqueeze_184: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
        unsqueeze_185: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, 2);  unsqueeze_184 = None
        unsqueeze_186: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_185, 3);  unsqueeze_185 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
        gt_3: "b8[64, 576, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_44, -3.0)
        lt_5: "b8[64, 576, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_44, 3.0);  convolution_44 = None
        bitwise_and_1: "b8[64, 576, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_3, lt_5);  gt_3 = lt_5 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
        unsqueeze_196: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
        unsqueeze_197: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, 2);  unsqueeze_196 = None
        unsqueeze_198: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_197, 3);  unsqueeze_197 = None
        unsqueeze_208: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
        unsqueeze_209: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, 2);  unsqueeze_208 = None
        unsqueeze_210: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_209, 3);  unsqueeze_209 = None
        unsqueeze_220: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
        unsqueeze_221: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, 2);  unsqueeze_220 = None
        unsqueeze_222: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_221, 3);  unsqueeze_221 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
        gt_4: "b8[64, 288, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_39, -3.0)
        lt_8: "b8[64, 288, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_39, 3.0);  convolution_39 = None
        bitwise_and_2: "b8[64, 288, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_4, lt_8);  gt_4 = lt_8 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
        unsqueeze_232: "f32[1, 288]" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
        unsqueeze_233: "f32[1, 288, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, 2);  unsqueeze_232 = None
        unsqueeze_234: "f32[1, 288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_233, 3);  unsqueeze_233 = None
        unsqueeze_244: "f32[1, 288]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
        unsqueeze_245: "f32[1, 288, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, 2);  unsqueeze_244 = None
        unsqueeze_246: "f32[1, 288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_245, 3);  unsqueeze_245 = None
        unsqueeze_256: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
        unsqueeze_257: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, 2);  unsqueeze_256 = None
        unsqueeze_258: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 3);  unsqueeze_257 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
        gt_5: "b8[64, 144, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_34, -3.0)
        lt_11: "b8[64, 144, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_34, 3.0);  convolution_34 = None
        bitwise_and_3: "b8[64, 144, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_5, lt_11);  gt_5 = lt_11 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
        unsqueeze_268: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
        unsqueeze_269: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, 2);  unsqueeze_268 = None
        unsqueeze_270: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 3);  unsqueeze_269 = None
        unsqueeze_280: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
        unsqueeze_281: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, 2);  unsqueeze_280 = None
        unsqueeze_282: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 3);  unsqueeze_281 = None
        unsqueeze_292: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
        unsqueeze_293: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, 2);  unsqueeze_292 = None
        unsqueeze_294: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 3);  unsqueeze_293 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
        gt_6: "b8[64, 120, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_29, -3.0)
        lt_14: "b8[64, 120, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_29, 3.0);  convolution_29 = None
        bitwise_and_4: "b8[64, 120, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_6, lt_14);  gt_6 = lt_14 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
        unsqueeze_304: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
        unsqueeze_305: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, 2);  unsqueeze_304 = None
        unsqueeze_306: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 3);  unsqueeze_305 = None
        unsqueeze_316: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
        unsqueeze_317: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, 2);  unsqueeze_316 = None
        unsqueeze_318: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 3);  unsqueeze_317 = None
        unsqueeze_328: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
        unsqueeze_329: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, 2);  unsqueeze_328 = None
        unsqueeze_330: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 3);  unsqueeze_329 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
        gt_7: "b8[64, 240, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_24, -3.0)
        lt_17: "b8[64, 240, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_24, 3.0);  convolution_24 = None
        bitwise_and_5: "b8[64, 240, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_7, lt_17);  gt_7 = lt_17 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
        unsqueeze_340: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
        unsqueeze_341: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, 2);  unsqueeze_340 = None
        unsqueeze_342: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 3);  unsqueeze_341 = None
        unsqueeze_352: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
        unsqueeze_353: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, 2);  unsqueeze_352 = None
        unsqueeze_354: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 3);  unsqueeze_353 = None
        unsqueeze_364: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
        unsqueeze_365: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, 2);  unsqueeze_364 = None
        unsqueeze_366: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 3);  unsqueeze_365 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
        gt_8: "b8[64, 240, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_19, -3.0)
        lt_20: "b8[64, 240, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_19, 3.0);  convolution_19 = None
        bitwise_and_6: "b8[64, 240, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_8, lt_20);  gt_8 = lt_20 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
        unsqueeze_376: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
        unsqueeze_377: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, 2);  unsqueeze_376 = None
        unsqueeze_378: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 3);  unsqueeze_377 = None
        unsqueeze_388: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
        unsqueeze_389: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, 2);  unsqueeze_388 = None
        unsqueeze_390: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 3);  unsqueeze_389 = None
        unsqueeze_400: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
        unsqueeze_401: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, 2);  unsqueeze_400 = None
        unsqueeze_402: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 3);  unsqueeze_401 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
        gt_9: "b8[64, 96, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_14, -3.0)
        lt_23: "b8[64, 96, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_14, 3.0);  convolution_14 = None
        bitwise_and_7: "b8[64, 96, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_9, lt_23);  gt_9 = lt_23 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
        unsqueeze_412: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
        unsqueeze_413: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, 2);  unsqueeze_412 = None
        unsqueeze_414: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 3);  unsqueeze_413 = None
        unsqueeze_424: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
        unsqueeze_425: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, 2);  unsqueeze_424 = None
        unsqueeze_426: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 3);  unsqueeze_425 = None
        unsqueeze_436: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
        unsqueeze_437: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, 2);  unsqueeze_436 = None
        unsqueeze_438: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 3);  unsqueeze_437 = None
        unsqueeze_448: "f32[1, 88]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
        unsqueeze_449: "f32[1, 88, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, 2);  unsqueeze_448 = None
        unsqueeze_450: "f32[1, 88, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 3);  unsqueeze_449 = None
        unsqueeze_460: "f32[1, 88]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
        unsqueeze_461: "f32[1, 88, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, 2);  unsqueeze_460 = None
        unsqueeze_462: "f32[1, 88, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 3);  unsqueeze_461 = None
        unsqueeze_472: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
        unsqueeze_473: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, 2);  unsqueeze_472 = None
        unsqueeze_474: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 3);  unsqueeze_473 = None
        unsqueeze_484: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
        unsqueeze_485: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, 2);  unsqueeze_484 = None
        unsqueeze_486: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 3);  unsqueeze_485 = None
        unsqueeze_496: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
        unsqueeze_497: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, 2);  unsqueeze_496 = None
        unsqueeze_498: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 3);  unsqueeze_497 = None
        unsqueeze_508: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
        unsqueeze_509: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, 2);  unsqueeze_508 = None
        unsqueeze_510: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 3);  unsqueeze_509 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
        gt_10: "b8[64, 16, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_3, -3.0)
        lt_26: "b8[64, 16, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_3, 3.0);  convolution_3 = None
        bitwise_and_8: "b8[64, 16, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_10, lt_26);  gt_10 = lt_26 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
        unsqueeze_520: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
        unsqueeze_521: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, 2);  unsqueeze_520 = None
        unsqueeze_522: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_521, 3);  unsqueeze_521 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:210, code: x = self.features(x)
        unsqueeze_532: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_533: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_532, 2);  unsqueeze_532 = None
        unsqueeze_534: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_533, 3);  unsqueeze_533 = None
        return [add_2, add_3, add, add_8, add_9, add_6, add_14, add_15, add_12, add_19, add_20, add_17, add_24, add_25, add_22, add_29, add_30, add_27, add_34, add_35, add_32, add_39, add_40, add_37, add_44, add_45, add_42, add_50, add_51, add_48, add_56, add_57, add_54, add_63, add_64, add_61, add_68, add_69, add_66, add_74, add_75, add_72, add_81, add_82, add_79, add_87, add_88, add_85, add_93, add_94, add_91, add_100, add_101, add_98, add_106, add_107, add_104, add_112, add_113, add_110, add_119, add_120, add_117, add_124, add_125, add_122, add_130, add_131, add_128, add_137, add_138, add_135, add_143, add_144, add_141, add_149, add_150, add_147, add_156, add_157, add_154, add_161, add_162, add_159, add_167, add_168, add_165, add_174, add_175, add_172, add_180, add_181, add_178, add_186, add_187, add_184, add_193, add_194, add_191, add_199, add_200, add_197, addmm_1, primals_1, primals_2, primals_4, primals_5, primals_7, primals_9, primals_11, primals_12, primals_14, primals_15, primals_17, primals_18, primals_20, primals_21, primals_23, primals_24, primals_26, primals_27, primals_29, primals_30, primals_32, primals_33, primals_35, primals_36, primals_38, primals_40, primals_42, primals_43, primals_45, primals_46, primals_48, primals_49, primals_51, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_66, primals_68, primals_69, primals_71, primals_72, primals_74, primals_75, primals_77, primals_79, primals_81, primals_82, primals_84, primals_85, primals_87, primals_88, primals_90, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_105, primals_107, primals_108, primals_110, primals_111, primals_113, primals_114, primals_116, primals_118, primals_120, primals_121, primals_123, primals_124, primals_126, primals_127, primals_129, primals_131, primals_133, primals_134, primals_136, primals_137, primals_245, convolution, squeeze_1, clone_102, div, convolution_1, squeeze_4, relu, mean, relu_1, div_1, mul_15, convolution_4, squeeze_7, add_16, convolution_5, squeeze_10, relu_2, convolution_6, squeeze_13, relu_3, convolution_7, squeeze_16, add_31, convolution_8, squeeze_19, relu_4, convolution_9, squeeze_22, relu_5, convolution_10, squeeze_25, add_47, convolution_11, squeeze_28, clone_103, div_2, convolution_12, squeeze_31, clone_104, div_3, mean_1, relu_6, div_4, mul_81, convolution_15, squeeze_34, add_65, convolution_16, squeeze_37, clone_105, div_5, convolution_17, squeeze_40, clone_106, div_6, mean_2, relu_7, div_7, mul_105, convolution_20, squeeze_43, add_84, convolution_21, squeeze_46, clone_107, div_8, convolution_22, squeeze_49, clone_108, div_9, mean_3, relu_8, div_10, mul_129, convolution_25, squeeze_52, add_103, convolution_26, squeeze_55, clone_109, div_11, convolution_27, squeeze_58, clone_110, div_12, mean_4, relu_9, div_13, mul_153, convolution_30, squeeze_61, add_121, convolution_31, squeeze_64, clone_111, div_14, convolution_32, squeeze_67, clone_112, div_15, mean_5, relu_10, div_16, mul_177, convolution_35, squeeze_70, add_140, convolution_36, squeeze_73, clone_113, div_17, convolution_37, squeeze_76, clone_114, div_18, mean_6, relu_11, div_19, mul_201, convolution_40, squeeze_79, add_158, convolution_41, squeeze_82, clone_115, div_20, convolution_42, squeeze_85, clone_116, div_21, mean_7, relu_12, div_22, mul_225, convolution_45, squeeze_88, add_177, convolution_46, squeeze_91, clone_117, div_23, convolution_47, squeeze_94, clone_118, div_24, mean_8, relu_13, div_25, mul_249, convolution_50, squeeze_97, add_196, convolution_51, squeeze_100, clone_119, view, addmm, philox_seed_like, mul_267, permute_2, permute_6, unsqueeze_138, unsqueeze_150, bitwise_and, unsqueeze_162, unsqueeze_174, unsqueeze_186, bitwise_and_1, unsqueeze_198, unsqueeze_210, unsqueeze_222, bitwise_and_2, unsqueeze_234, unsqueeze_246, unsqueeze_258, bitwise_and_3, unsqueeze_270, unsqueeze_282, unsqueeze_294, bitwise_and_4, unsqueeze_306, unsqueeze_318, unsqueeze_330, bitwise_and_5, unsqueeze_342, unsqueeze_354, unsqueeze_366, bitwise_and_6, unsqueeze_378, unsqueeze_390, unsqueeze_402, bitwise_and_7, unsqueeze_414, unsqueeze_426, unsqueeze_438, unsqueeze_450, unsqueeze_462, unsqueeze_474, unsqueeze_486, unsqueeze_498, unsqueeze_510, bitwise_and_8, unsqueeze_522, unsqueeze_534]
        
